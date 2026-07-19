# SFP + MoE Filter Block

An implementation of the **Subtractive Fine-tuning Paradigm (SFP)** from
*"Less Is More: Rethinking Parameter-Efficient Fine-Tuning from a
Subtractive Perspective"* (Jiang et al., AAAI-26), generalized so the
filter block is a small **Mixture-of-Experts** of linear filters, routed
with **noisy top-k gating** and an **auxiliary load-balancing loss**
(Shazeer et al., 2017), instead of a single linear filter.

## Why noisy top-k + aux load balancing (not aux-loss-alone)

The prompt asked for auxiliary load balancing applied before top-k noisy
routing, and to fall back to "whichever is better" if both couldn't be
combined. In practice they aren't alternatives, they compose:

- **Noisy top-k gating** is what makes the router's expert assignment
  differentiable/explorable in the first place: a plain deterministic
  top-k has ~zero useful gradient for moving tokens away from a currently
  dominant expert, since gates for non-selected experts are exactly zero
  and don't vary smoothly as logits cross the threshold. Adding
  input-dependent Gaussian noise to the logits before the top-k turns
  "would this expert be selected" into a smooth probability.
- **The auxiliary load-balancing loss** (`importance` = sum of gate values
  per expert, `load` = smooth top-k-inclusion probability per expert, both
  penalized via squared coefficient of variation) is computed *from that
  same noisy routing distribution*, every forward pass, before the sparse
  dispatch to experts happens. It's added to the task loss during
  training.

This is exactly the design in Shazeer et al.'s "Outrageously Large Neural
Networks" and is implemented that way in `moe_filter_block.py`.

## Files

| File | Purpose |
|---|---|
| `moe_filter_block.py` | `NoisyTopKGate` + `MoEFilterBlock`: the MoE-generalized filter block, with pseudo-inverse initialization (paper Eq. 3-4). |
| `model_surgery.py` | Replace `vit.blocks[idx]` with an `MoEFilterBlock` (paper Eq. 2), freeze everything except filter block(s) + LayerNorm + head. |
| `snip_selection.py` | SNIP saliency (paper Eq. 8-9) to rank transformer blocks and pick the least task-salient one(s) to replace. |
| `data.py` | VTAB-1k task loader (standard `train800/val200/test` folder layout) + a CIFAR-100 1000-shot smoke test that needs no external download setup. |
| `train_sfp_moe.py` | Ties it all together: SNIP selection → calibration → pseudo-inverse init → substitution → selective fine-tuning. |

## Pipeline (mirrors the paper's 3 steps)

1. **SNIP layer selection** — one backward pass on a calibration batch
   ranks all 12 ViT-B/16 transformer blocks by `|dL/dtheta * theta|`
   summed per block; the lowest-saliency block(s) are the ones whose
   *pretrained* computation matters least to this task, and are chosen for
   replacement (`--num-replace-layers 1` or `2`, matching the paper's
   single-/dual-layer substitution schemes).
2. **Calibration + pseudo-inverse init** — forward hooks record the
   *original* block's input/output activations on `--calib-images` (64 by
   default, matching the paper) calibration images; every expert in the
   MoE filter block is initialized from `W = pinv(X_in) @ X_out`, so the
   filter starts out approximating the original block's function. Experts
   after the first get a small amount of noise added so they aren't exact
   duplicates.
3. **Component substitution + selective fine-tuning** — `vit.blocks[idx]`
   is replaced with the `MoEFilterBlock`; everything else in the backbone
   is frozen except LayerNorm affine parameters (paper's trick, ~0.038M
   params) and the classification head. Only the filter block(s), the
   LayerNorms, and the head are trained, with `task_loss + aux_loss` as
   the objective.

## Usage

```bash
pip install -r requirements.txt

# quick smoke test on CIFAR-100 (1000-shot), no VTAB download needed
python train_sfp_moe.py --smoketest --epochs 10

# real VTAB-1k task (point --data-root at your local VTAB-1k dump)
python train_sfp_moe.py --data-root /path/to/vtab-1k --task cifar100 \
    --num-replace-layers 1 --num-experts 4 --top-k 2 --epochs 100
```

Key MoE knobs (all CLI flags):

- `--num-experts` (default 4): experts per filter block. Set to `1` (with
  `--top-k 1`) to recover the original paper's single-filter SFP exactly.
- `--top-k` (default 2): active experts per token.
- `--noise-eps`: floor on the noisy-gate stddev (Shazeer default `1e-2`).
- `--aux-loss-weight`: weight on the importance+load balancing loss.
- `--expert-init-noise-std`: symmetry-breaking noise added on top of the
  shared pseudo-inverse initialization.
- `--num-replace-layers {1,2}`: single- vs. dual-layer substitution.

## Notes / simplifications vs. the full paper

- The paper's architectural search additionally samples/prunes candidate
  replacement configurations across training using a moving-average SNIP
  score (Eq. 10). This code uses a single-shot SNIP ranking (one backward
  pass) to pick the replacement layer(s) up front, which the paper itself
  notes is already a strong, near-zero-cost predictor (Fig. 5).
- `build_vtab_task` assumes a `train800/val200/test` folder-per-class
  layout; adapt it if your local VTAB-1k copy is structured differently
  (e.g. raw tfds records) — nothing downstream depends on how the
  `DataLoader`s are built.
- Expert dispatch in `MoEFilterBlock.forward` is a straightforward
  masked-loop implementation (clear and correct, not throughput-optimized
  for very large batch x sequence-length x num_experts products).
