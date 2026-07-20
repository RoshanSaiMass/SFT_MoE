"""
Moving-average SNIP candidate search (paper Eq. 8-10).

This replaces the earlier single-shot SNIP shortcut with the paper's actual
search procedure, which is what produced the numbers in Table 1 (the paper
states alpha=0.3 for Eq. 10 in its Implementation Details, i.e. Table 1's
70.5%/74.04%/etc. come from THIS search, not a one-shot ranking).

How it works, matching the paper's description:
  - Every possible transformer block index is a "candidate" replacement
    position, each with its own MoE filter block (candidate-specific
    weights), while LayerNorm affine parameters and the classification head
    are SHARED across all candidates and trained continuously regardless of
    which candidate is currently active -- a one-shot, weight-sharing search
    (in the same family as zero-cost/weight-sharing NAS methods).
  - At each search round, every surviving candidate is substituted in,
    trained for a few steps (shared optimizer, shared LN/head state), then
    scored with one extra backward pass:
        S_p(theta) = |dL/dtheta * theta|              (Eq. 8)
        S = sum_i S_p(theta)_i over all trainable params under this
            candidate's configuration                  (Eq. 9)
  - That score updates a per-candidate exponential moving average:
        q_n^t = alpha * q_n^{t-1} + (1 - alpha) * S_n   (Eq. 10)
  - The lowest-q candidates are pruned each round (successive halving);
    training continues on the survivors. This is a standard "keep the
    candidates zero-cost metrics predict will perform best" NAS proxy, so
    HIGHER q is better and survives -- the opposite convention from a raw
    "how redundant is this pretrained layer" saliency ranking.
  - For dual-layer substitution the paper reuses the single-layer q scores
    directly ("a computationally zero-cost approach"): we just take the
    top-2 (or top-N) indices by final q instead of running a second search.

This is compute-heavier than the single-shot shortcut in snip_selection.py
(every round, every surviving candidate needs its own forward+backward
through the whole frozen backbone), which is expected -- it's the real
search, not an approximation of it.
"""
import math
import torch
import torch.nn as nn

from moe_filter_block import MoEFilterBlock


def _calibration_batches(loader, n_images):
    it = iter(loader)
    collected = 0
    while collected < n_images:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        collected += batch[0].size(0)
        yield batch


@torch.no_grad()
def _collect_all_block_io(vit_model, layer_indices, calib_batches, device, n_images):
    """Same idea as train_sfp_moe.collect_block_io, but for every candidate
    layer index at once (needed since the search considers all of them)."""
    io_store = {idx: ([], []) for idx in layer_indices}
    hooks = []

    for idx in layer_indices:
        blk = vit_model.blocks[idx]

        def pre_hook(module, inp, _idx=idx):
            io_store[_idx][0].append(inp[0].detach().cpu())

        def fwd_hook(module, inp, out, _idx=idx):
            io_store[_idx][1].append(out.detach().cpu())

        hooks.append(blk.register_forward_pre_hook(pre_hook))
        hooks.append(blk.register_forward_hook(fwd_hook))

    vit_model.eval()
    collected = 0
    for x, _ in calib_batches:
        vit_model(x.to(device))
        collected += x.size(0)
        if collected >= n_images:
            break

    for h in hooks:
        h.remove()

    result = {}
    for idx in layer_indices:
        x_in = torch.cat(io_store[idx][0], dim=0)[:n_images]
        x_out = torch.cat(io_store[idx][1], dim=0)[:n_images]
        b, n, d = x_in.shape
        result[idx] = (x_in.reshape(-1, d), x_out.reshape(-1, d))
    return result


def run_moving_average_snip_search(
    vit_model: nn.Module,
    train_loader,
    loss_fn,
    device,
    moe_kwargs: dict,
    num_replace_layers: int = 1,
    calib_images: int = 64,
    alpha: float = 0.3,
    steps_per_candidate: int = 5,
    prune_fraction: float = 0.5,
    max_rounds: int = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    tune_layernorm: bool = True,
    verbose: bool = True,
):
    """Run the paper's Eq. 8-10 candidate search over every possible
    single-layer replacement position, then (for dual-layer substitution)
    reuse those same final scores zero-cost to pick the top-N positions.

    Returns:
        chosen_indices   : list[int], the final replacement layer(s)
        chosen_blocks    : {idx: MoEFilterBlock}, already substituted into
                            vit_model.blocks at `chosen_indices`
        final_q_scores   : {idx: float}, the final moving-average score of
                            every candidate that was ever considered
    """
    num_layers = len(vit_model.blocks)
    original_blocks = list(vit_model.blocks)  # frozen originals, kept by reference

    # 1. build one candidate MoE filter block per possible layer index
    candidates = {idx: MoEFilterBlock(vit_model.embed_dim, **moe_kwargs).to(device)
                  for idx in range(num_layers)}

    # 2. calibration IO + pseudo-inverse init for every candidate up front
    calib_iter = _calibration_batches(train_loader, calib_images)
    io_pairs = _collect_all_block_io(vit_model, list(range(num_layers)), calib_iter, device, calib_images)
    for idx, fb in candidates.items():
        x_in, x_out = io_pairs[idx]
        fb.init_from_pseudo_inverse(x_in.to(device), x_out.to(device))

    # 3. freeze the backbone; only LayerNorm (shared) + head (shared) +
    #    every candidate's own filter block are trainable
    for p in vit_model.parameters():
        p.requires_grad_(False)

    shared_params = []
    if tune_layernorm:
        for m in vit_model.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad_(True)
                    shared_params.append(p)
    head = vit_model.get_classifier() if hasattr(vit_model, "get_classifier") else vit_model.head
    for p in head.parameters():
        p.requires_grad_(True)
        shared_params.append(p)

    all_candidate_params = [p for fb in candidates.values() for p in fb.parameters()]
    optimizer = torch.optim.AdamW(shared_params + all_candidate_params, lr=lr, weight_decay=weight_decay)

    train_iter = iter(train_loader)

    def next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    q = {idx: 0.0 for idx in candidates}
    surviving = set(candidates.keys())
    round_num = 0

    while len(surviving) > 1 and (max_rounds is None or round_num < max_rounds):
        round_num += 1
        vit_model.train()
        for idx in list(surviving):
            fb = candidates[idx]
            vit_model.blocks[idx] = fb  # swap this candidate in

            # a few real training steps on this candidate (shared LN/head
            # keep being updated too -- one-shot weight sharing)
            for _ in range(steps_per_candidate):
                x, y = next_batch()
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = vit_model(x)
                loss = loss_fn(out, y) + fb.last_aux_loss
                loss.backward()
                optimizer.step()

            # one extra backward pass purely to score this candidate
            # (Eq. 8-9), gradients from this pass are NOT applied
            calib_x, calib_y = next_batch()
            calib_x, calib_y = calib_x.to(device), calib_y.to(device)
            optimizer.zero_grad()
            out = vit_model(calib_x)
            loss = loss_fn(out, calib_y) + fb.last_aux_loss
            loss.backward()

            s_n = 0.0
            for p in shared_params + list(fb.parameters()):
                if p.grad is not None:
                    s_n += (p.grad * p).abs().sum().item()
            optimizer.zero_grad()

            q[idx] = alpha * q[idx] + (1 - alpha) * s_n
            vit_model.blocks[idx] = original_blocks[idx]  # swap back out

        n_keep = max(1, int(math.ceil(len(surviving) * (1 - prune_fraction))))
        ranked = sorted(surviving, key=lambda i: q[i], reverse=True)  # higher q = better, survives
        pruned = set(ranked[n_keep:])
        surviving = set(ranked[:n_keep])

        if verbose:
            print(f"[SFP search] round {round_num}: scored {len(ranked)} candidates, "
                  f"pruned {sorted(pruned)}, surviving {sorted(surviving)}  "
                  f"(top q={q[ranked[0]]:.4f})")

    all_ranked_by_q = sorted(q.keys(), key=lambda i: q[i], reverse=True)
    chosen_indices = all_ranked_by_q[:num_replace_layers]

    if verbose:
        print(f"[SFP search] final choice: layer(s) {chosen_indices} "
              f"(final q scores: {[(i, round(q[i], 4)) for i in chosen_indices]})")

    chosen_blocks = {idx: candidates[idx] for idx in chosen_indices}
    for idx, fb in chosen_blocks.items():
        vit_model.blocks[idx] = fb

    return chosen_indices, chosen_blocks, q
