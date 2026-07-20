"""
Component substitution + freezing utilities for SFP on a timm ViT.

timm's VisionTransformer keeps its transformer blocks in `model.blocks`, a
plain nn.Sequential of modules whose forward signature is `x -> x`
(same shape in, same shape out). That's exactly the signature our
MoEFilterBlock implements, so "replace layer l with a filter block"
(paper Eq. 2) is literally `model.blocks[l] = filter_block` -- no need to
reimplement `forward_features`.
"""
import torch
import torch.nn as nn

from moe_filter_block import MoEFilterBlock


def substitute_layers(vit_model, layer_indices, moe_kwargs) -> dict:
    """Replace `vit_model.blocks[idx]` with a fresh MoEFilterBlock for every
    idx in layer_indices. Returns {idx: MoEFilterBlock} for later
    pseudo-inverse initialization and inspection."""
    dim = vit_model.embed_dim
    filter_blocks = {}
    for idx in layer_indices:
        fb = MoEFilterBlock(dim, **moe_kwargs)
        vit_model.blocks[idx] = fb
        filter_blocks[idx] = fb
    return filter_blocks


def collect_aux_loss(vit_model) -> torch.Tensor:
    """Sum the load-balancing auxiliary loss across every MoEFilterBlock
    currently in the model (populated during the most recent forward pass).
    Add this to the task loss before calling .backward()."""
    total = None
    for m in vit_model.blocks:
        if isinstance(m, MoEFilterBlock):
            total = m.last_aux_loss if total is None else total + m.last_aux_loss
    if total is None:
        return torch.zeros(())
    return total


def freeze_all_except(vit_model, filter_blocks: dict, tune_layernorm: bool = True,
                       tune_head: bool = True) -> None:
    """Freeze the whole backbone, then selectively unfreeze:
      - every MoEFilterBlock's parameters (the whole point of SFP: this is
        the ONLY thing that meaningfully changes the model's function),
      - all LayerNorm affine parameters (paper: "selectively unfreezing the
        Layer Normalization parameters... facilitates dynamic recalibration
        of feature statistics... adds only 0.038M parameters"),
      - the classification head (needed since it's task-specific and was
        never pretrained for this task).
    """
    for p in vit_model.parameters():
        p.requires_grad_(False)

    for fb in filter_blocks.values():
        for p in fb.parameters():
            p.requires_grad_(True)

    if tune_layernorm:
        for m in vit_model.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad_(True)

    if tune_head:
        head = vit_model.get_classifier() if hasattr(vit_model, "get_classifier") else vit_model.head
        for p in head.parameters():
            p.requires_grad_(True)


def count_trainable_params(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
