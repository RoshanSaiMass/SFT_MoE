"""
SNIP-based layer selection (paper Eq. 8-9): pick which transformer block(s)
are the best candidates for subtractive replacement.

The paper computes, for a single-shot backward pass, the saliency
S_p(theta) = |dL/dtheta * theta| of every parameter, sums it over each
candidate block, and uses that as a zero-cost proxy for how much a block's
*current, pretrained* computation matters to the downstream task. Blocks
with LOW aggregate saliency are the ones whose original behaviour barely
affects the task loss under the pretrained weights -- i.e. the best
candidates for "this holds interfering/redundant knowledge, replace it with
a cheap filter". We reproduce that scoring here (a single backward pass is
enough) rather than the full moving-average architecture-search loop from
the paper, since one calibration batch is sufficient to rank all 12 ViT-B
blocks.
"""
import copy
import torch
import torch.nn as nn


def snip_layer_scores(vit_model: nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor,
                       loss_fn, device) -> dict:
    """Returns {block_index: saliency_score} from one backward pass on a
    single calibration batch. Operates on a deep copy so it never disturbs
    the real model's gradients/state."""
    model = copy.deepcopy(vit_model).to(device)
    model.train()
    model.zero_grad()

    x, y = batch_x.to(device), batch_y.to(device)
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()

    scores = {}
    for i, blk in enumerate(model.blocks):
        s = 0.0
        for p in blk.parameters():
            if p.grad is not None:
                s += (p.grad * p).abs().sum().item()
        scores[i] = s

    del model
    return scores


def rank_layers_for_replacement(scores: dict, num_layers_to_replace: int = 1) -> list:
    """Lowest-saliency blocks first -- these contribute the least to the
    task loss under the pretrained model and are the paper's preferred
    substitution targets."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    return [idx for idx, _ in ranked[:num_layers_to_replace]]
