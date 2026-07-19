"""
MoE-generalized Subtractive Filter Block
=========================================

The original SFP paper (Jiang et al., AAAI-26, "Less Is More: Rethinking
Parameter-Efficient Fine-tuning from a Subtractive Perspective") replaces a
whole transformer block f_l with a SINGLE lightweight linear filter F,
initialized via the pseudo-inverse solution to

    min_W || X_{l-1} W - X_l ||_F^2                              (paper Eq. 3-4)

so the filter starts out reproducing the original block's behaviour, and is
then fine-tuned (with the rest of the backbone frozen) to strip out
redundant / interfering knowledge.

This module generalizes that single filter into a small Mixture-of-Experts
of such linear filters. Each expert is a D x D linear map -- structurally
identical to the paper's filter -- and a *noisy top-k* router (Shazeer et
al., 2017, "Outrageously Large Neural Networks") decides, per token, which
k experts process it. A differentiable load-balancing auxiliary loss
(importance + load, both penalized via their squared coefficient of
variation) is computed *before* the sparse dispatch is applied, exactly as
in the original sparsely-gated MoE formulation, so that gradient signal
that discourages expert collapse flows alongside the routing decision
itself rather than being bolted on afterward.

We use noisy top-k gating (rather than a plain top-k + separate balancing
scheme) because the noise term is what makes the router's expert choice
differentiable/explorable in the first place -- without it the balancing
loss has nothing informative to act on early in training, since a
deterministic top-k has near-zero gradient signal for reassigning tokens
away from a currently-dominant expert. The auxiliary load-balancing loss is
then layered on top of that noisy routing (not instead of it), which is the
same importance+load combination introduced by Shazeer et al.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def _cv_squared(x: torch.Tensor) -> torch.Tensor:
    """Squared coefficient of variation: variance / mean^2. This is the
    standard scale-invariant "how uneven is this distribution" penalty used
    for MoE load balancing. Returns 0 when there's nothing to balance
    (a single value, e.g. num_experts == 1)."""
    eps = 1e-10
    if x.numel() <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return x.float().var() / (x.float().mean() ** 2 + eps)


class NoisyTopKGate(nn.Module):
    """Noisy top-k router (Shazeer et al., 2017).

    forward(x) -> (gates, aux_loss)
      gates:    (n_tokens, num_experts), exactly `k` nonzero entries per row,
                summing to 1 (a softmax restricted to the chosen top-k).
      aux_loss: scalar load-balancing penalty (0 in eval mode, since it's
                only meant to shape training dynamics).
    """

    def __init__(self, dim: int, num_experts: int, k: int = 2, noise_eps: float = 1e-2):
        super().__init__()
        assert 1 <= k <= num_experts, "k must be in [1, num_experts]"
        self.num_experts = num_experts
        self.k = k
        self.noise_eps = noise_eps

        self.w_gate = nn.Linear(dim, num_experts, bias=False)
        self.w_noise = nn.Linear(dim, num_experts, bias=False)
        # Zero init -> uniform routing logits at the very start of training,
        # so no expert has an a-priori advantage before the filter block has
        # seen any gradient (standard practice for MoE gates).
        nn.init.zeros_(self.w_gate.weight)
        nn.init.zeros_(self.w_noise.weight)

    def _prob_in_top_k(self, clean_logits, noisy_logits, noise_stddev, top_logits):
        """Smooth (differentiable) estimate of "how likely is it that this
        expert would have been selected", used as the 'load' signal instead
        of the non-differentiable hard count. See Shazeer et al. (2017),
        Appendix A, for the derivation of this estimator."""
        n_tokens = clean_logits.size(0)
        m = top_logits.size(1)  # k + 1 (or num_experts if smaller) columns
        top_values_flat = top_logits.flatten()

        threshold_positions_if_in = torch.arange(n_tokens, device=clean_logits.device) * m + self.k
        threshold_if_in = top_values_flat[threshold_positions_if_in].unsqueeze(1)
        is_in = noisy_logits > threshold_if_in

        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = top_values_flat[threshold_positions_if_out].unsqueeze(1)

        normal = Normal(0.0, 1.0)
        prob_if_in = normal.cdf((clean_logits - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_logits - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def forward(self, x: torch.Tensor):
        clean_logits = self.w_gate(x)
        k_for_topk = min(self.k + 1, self.num_experts)

        if self.training:
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = F.softplus(raw_noise_stddev) + self.noise_eps
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits_for_topk = noisy_logits
        else:
            noisy_logits, noise_stddev = clean_logits, None
            logits_for_topk = clean_logits

        top_logits, top_indices = logits_for_topk.topk(k_for_topk, dim=-1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        gates = torch.zeros_like(clean_logits).scatter(-1, top_k_indices, top_k_gates)

        if self.training and self.num_experts > 1:
            importance = gates.sum(0)
            if self.k < self.num_experts:
                load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
            else:
                load = (gates > 0).float().sum(0)
            aux_loss = _cv_squared(importance) + _cv_squared(load)
        else:
            aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        return gates, aux_loss


class MoEFilterBlock(nn.Module):
    """Drop-in replacement for a transformer block, playing the role of the
    filter block F in the SFP paper's component-substitution scheme:

        f_sub(X) = f_L o ... o f_{l+1} o F o f_{l-1} o ... o f_1(X)

    Instead of a single D x D linear filter, F here is `num_experts` linear
    filters gated by a noisy top-k router, with `k` of them active per
    token. Setting num_experts = k = 1 recovers the original paper's filter
    block exactly.
    """

    def __init__(self, dim: int, num_experts: int = 4, k: int = 2,
                 noise_eps: float = 1e-2, expert_init_noise_std: float = 1e-3,
                 aux_loss_weight: float = 1e-2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.aux_loss_weight = aux_loss_weight
        self.expert_init_noise_std = expert_init_noise_std

        self.experts = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_experts)])
        self.gate = NoisyTopKGate(dim, num_experts, k=self.k, noise_eps=noise_eps)

        self.register_buffer("last_aux_loss", torch.zeros(()), persistent=False)

    @torch.no_grad()
    def init_from_pseudo_inverse(self, x_in: torch.Tensor, x_out: torch.Tensor):
        """Initialize every expert with the paper's pseudo-inverse solution
        to min_W || x_in @ W - x_out ||_F^2 (Eq. 3-4), so the filter block as
        a whole starts out reproducing the replaced block's behaviour
        regardless of which expert(s) the (initially near-uniform) router
        picks. A small amount of noise is added to every expert but the
        first so that they are not exact duplicates and the router has
        something to eventually differentiate between.

        x_in, x_out: (M, D) calibration features gathered from the ORIGINAL
        (pre-substitution) block on a small calibration batch.
        """
        w = torch.linalg.pinv(x_in) @ x_out  # (D, D): x_in @ w ~= x_out
        for i, expert in enumerate(self.experts):
            expert.weight.copy_(w.t())  # nn.Linear.weight is (out, in) = w^T
            if self.num_experts > 1 and i > 0:
                expert.weight.add_(torch.randn_like(expert.weight) * self.expert_init_noise_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        flat_x = x.reshape(-1, d)
        gates, aux_loss = self.gate(flat_x)

        out = flat_x.new_zeros(flat_x.shape)
        for e_idx, expert in enumerate(self.experts):
            col = gates[:, e_idx]
            mask = col > 0
            if mask.any():
                out[mask] += col[mask].unsqueeze(-1) * expert(flat_x[mask])

        self.last_aux_loss = self.aux_loss_weight * aux_loss
        return out.reshape(b, n, d)
