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
| `moe_filter_block.py` | `NoisyTopKGate` + `MoEFilterBlock`: the MoE-generalized filter block, with pseudo-inverse (or truncated-SVD, for low-rank experts) initialization (paper Eq. 3-4). |
| `model_surgery.py` | Replace `vit.blocks[idx]` with an `MoEFilterBlock` (paper Eq. 2), freeze everything except filter block(s) + LayerNorm + head. |
| `moving_average_snip_search.py` | **The paper's real layer-selection algorithm** (Eq. 8-10): a one-shot, weight-sharing candidate search with moving-average SNIP scoring and successive-halving pruning. This is what produced Table 1's numbers. |
| `snip_selection.py` | A cheaper single-shot approximation (one backward pass, no candidate training) for fast iteration -- NOT what the paper used for its reported results. |
| `data.py` | VTAB-1k task loader (NOAH/SSF-style `images/` + `train800.txt`/`val200.txt`/`test.txt` annotation files, with an `ImageFolder` fallback) + a CIFAR-100 1000-shot smoke test that needs no external download setup. |
| `train_sfp_moe.py` | Ties it all together: layer selection (moving-average search by default, or single-shot via `--layer-selection snip_once`) → substitution → selective fine-tuning. |
| `run_all_vtab.py` | Loops `train_sfp_moe.py` over all 19 VTAB-1k tasks and aggregates into a Table-1-style CSV. |

## Pipeline (mirrors the paper's 3 steps)

1. **Layer selection.** Two options, controlled by `--layer-selection`:
   - **`moving_average` (default, paper-faithful):** implements Eq. 8-10
     exactly. Every possible transformer block index becomes a candidate
     with its own MoE filter block; LayerNorm affine params and the
     classification head are shared across all candidates and trained
     continuously (one-shot, weight-sharing search). Each round, every
     surviving candidate is substituted in, trained for
     `--search-steps-per-candidate` steps, then scored with one extra
     backward pass (`S = sum |dL/dtheta * theta|`, Eq. 8-9), which updates
     a per-candidate exponential moving average (`q^t = alpha*q^{t-1} +
     (1-alpha)*S`, Eq. 10, `--search-alpha` default `0.3` matches the
     paper). The lowest-`q` candidates are pruned (`--search-prune-fraction`,
     default half) each round until one survives. For `--num-replace-layers 2`,
     the paper reuses single-layer scores directly ("a computationally
     zero-cost approach") -- we do the same: just take the top-2 final `q`
     scores instead of running a second search.
   - **`snip_once` (fast approximation):** a single backward pass ranks
     blocks by how little the *original pretrained* computation matters to
     the task loss, and picks the lowest-scoring one(s) directly -- no
     candidate training or pruning. Useful for quick sanity checks, but
     this is **not** the algorithm that produced the paper's Table 1
     numbers, and may pick a different layer than the real search would.
2. **Calibration + pseudo-inverse init** — forward hooks record the
   *original* block's input/output activations on `--calib-images` (64 by
   default, matching the paper) calibration images; every expert in the
   MoE filter block is initialized from `W = pinv(X_in) @ X_out` (or its
   best rank-r approximation via truncated SVD, if `--rank` is set), so the
   filter starts out approximating the original block's function. (For
   `moving_average`, this happens once per candidate, inside the search.)
3. **Component substitution + selective fine-tuning** — the winning
   layer(s)' `vit.blocks[idx]` now holds the `MoEFilterBlock`; everything
   else in the backbone is frozen except LayerNorm affine parameters
   (paper's trick, ~0.038M params) and the classification head. Only the
   filter block(s), the LayerNorms, and the head are trained for
   `--epochs`, with `task_loss + aux_loss` as the objective.

## Usage

```bash
pip install -r requirements.txt

# quick smoke test on CIFAR-100 (1000-shot), no VTAB download needed
python train_sfp_moe.py --smoketest --epochs 10

# real VTAB-1k task, paper-faithful moving-average layer search (default)
python train_sfp_moe.py --data-root /path/to/vtab-1k --task cifar100 \
    --num-replace-layers 1 --num-experts 4 --top-k 2 --epochs 100

# same, but with parameter-efficient low-rank shared-basis experts
python train_sfp_moe.py --data-root /path/to/vtab-1k --task cifar100 \
    --num-experts 4 --top-k 2 --rank 32 --share-basis --epochs 100

# fast single-shot layer selection instead (not paper-faithful, for quick iteration)
python train_sfp_moe.py --data-root /path/to/vtab-1k --task cifar100 \
    --layer-selection snip_once --epochs 100
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
- `--rank`: factorize each expert into a rank-r `down`/`up` pair
  (`2*dim*r` params) instead of a dense `dim*dim` filter. Set
  `rank <= dim / (2*num_experts)` to keep the whole MoE filter block at or
  below the paper's single dense filter's parameter count regardless of
  how many experts you use.
- `--share-basis`: (with `--rank` set) share one `down` projection across
  all experts, only `up` differs per expert -- the cheapest option.

Layer-selection search knobs:

- `--layer-selection {moving_average, snip_once}`
- `--search-alpha` (default `0.3`, matches the paper's Eq. 10)
- `--search-steps-per-candidate` (default `5`)
- `--search-prune-fraction` (default `0.5`)
- `--search-max-rounds` (default: run until 1 candidate survives)

## Notes / simplifications vs. the full paper

- The moving-average search's exact pruning schedule (how many
  steps-per-candidate, what fraction to prune each round, how the search
  epochs count against the total training budget) isn't fully specified in
  the paper beyond "equal resource allocation" and `alpha=0.3`; the
  defaults above are a reasonable successive-halving schedule, not a
  verbatim reproduction of an undisclosed schedule. Tune
  `--search-steps-per-candidate` / `--search-prune-fraction` /
  `--search-max-rounds` if you want to trade search cost against how
  thoroughly candidates are evaluated before being pruned.
- `build_vtab_task` targets the NOAH/SSF-style `images/` + annotation-file
  layout; use `build_vtab_task_imagefolder` instead if your local VTAB-1k
  copy uses per-class subfolders.
- Expert dispatch in `MoEFilterBlock.forward` is a straightforward
  masked-loop implementation (clear and correct, not throughput-optimized
  for very large batch x sequence-length x num_experts products).
