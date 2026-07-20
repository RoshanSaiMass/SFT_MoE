"""
Subtractive Fine-tuning (SFP) with an MoE filter block, on a pretrained ViT.

Pipeline (mirrors the paper's 3 steps, Method section, with the filter
block generalized to a noisy-top-k MoE):

  1. Layer selection  -- by default, the paper's REAL search (Eq. 8-10):
                          a one-shot, weight-sharing candidate search where
                          every possible layer index is trained briefly and
                          scored via a moving-average SNIP metric, with the
                          lowest-scoring candidates pruned each round until
                          the best one (or few, for dual-layer) survive.
                          This is what produced Table 1's numbers in the
                          paper. A cheaper single-shot approximation
                          (one backward pass, no candidate training) is
                          available via --layer-selection snip_once if you
                          want a fast sanity-check run instead.
  2. Calibration + init -- forward hooks capture the ORIGINAL block's
                          input/output on ~64 calibration images; every
                          expert in the MoE filter block is initialized
                          from the pseudo-inverse solution (Eq. 3-4), so the
                          filter starts out reproducing the original block.
                          (Done inside the search for `moving_average`;
                          done as an explicit step below for `snip_once`.)
  3. Component substitution -- vit.blocks[idx] <- MoEFilterBlock(idx);
                          everything except the filter block(s), LayerNorm
                          affine params, and the head is frozen; fine-tune
                          only those for the remaining epochs.

Usage (smoke test, no VTAB download needed):
    python train_sfp_moe.py --smoketest --epochs 10

Usage (real VTAB-1k task, paper-faithful moving-average search):
    python train_sfp_moe.py --data-root /path/to/vtab-1k --task cifar100 --epochs 100

Usage (fast single-shot layer selection instead, for quick iteration):
    python train_sfp_moe.py --data-root /path/to/vtab-1k --task cifar100 \
        --layer-selection snip_once --epochs 100
"""
import argparse

import torch
import torch.nn as nn
import timm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model_surgery import substitute_layers, collect_aux_loss, freeze_all_except, count_trainable_params
from snip_selection import snip_layer_scores, rank_layers_for_replacement
from moving_average_snip_search import run_moving_average_snip_search
from data import build_vtab_task, build_cifar100_smoketest


def parse_args():
    p = argparse.ArgumentParser(description="Subtractive Fine-tuning (SFP) with an MoE filter block")
    p.add_argument("--backbone", default="vit_base_patch16_224.augreg_in21k",
                    help="timm model name; default matches the paper's ViT-B/16 pretrained on ImageNet-21K")
    p.add_argument("--data-root", default=None, help="root dir holding VTAB-1k tasks")
    p.add_argument("--task", default=None, help="VTAB-1k task name (subfolder of --data-root)")
    p.add_argument("--smoketest", action="store_true",
                    help="use CIFAR-100 (1000-shot) instead of real VTAB-1k data")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100,
                    help="fine-tuning epochs AFTER layer selection finishes")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # --- layer selection ---
    p.add_argument("--layer-selection", choices=["moving_average", "snip_once"], default="moving_average",
                    help="'moving_average' = paper's real Eq. 8-10 search (what produced Table 1); "
                         "'snip_once' = cheap one-backward-pass approximation for quick iteration")
    p.add_argument("--num-replace-layers", type=int, default=1, choices=[1, 2],
                    help="single-layer or dual-layer substitution (paper Fig. 2 / Table 3)")
    p.add_argument("--calib-images", type=int, default=64,
                    help="number of images used for the pseudo-inverse calibration (paper uses 64)")

    # --- moving-average search hyperparameters (paper Eq. 10: alpha=0.3) ---
    p.add_argument("--search-alpha", type=float, default=0.3,
                    help="EMA momentum for the moving-average SNIP score (paper's alpha)")
    p.add_argument("--search-steps-per-candidate", type=int, default=5,
                    help="training steps per surviving candidate, per search round")
    p.add_argument("--search-prune-fraction", type=float, default=0.5,
                    help="fraction of surviving candidates pruned each round")
    p.add_argument("--search-max-rounds", type=int, default=None,
                    help="cap on search rounds (default: run until 1 candidate survives)")

    # --- MoE filter block ---
    p.add_argument("--num-experts", type=int, default=4, help="experts per filter block")
    p.add_argument("--top-k", type=int, default=2, help="active experts per token (noisy top-k)")
    p.add_argument("--noise-eps", type=float, default=1e-2, help="noisy-gate stddev floor")
    p.add_argument("--expert-init-noise-std", type=float, default=1e-3,
                    help="symmetry-breaking noise added to experts 1..N-1 after pseudo-inverse init")
    p.add_argument("--aux-loss-weight", type=float, default=1e-2,
                    help="weight on the importance+load load-balancing auxiliary loss")
    p.add_argument("--rank", type=int, default=None,
                    help="if set, experts are rank-r factorized (2*dim*r params) instead of "
                         "dense (dim*dim params). Set rank <= dim/(2*num_experts) to keep the "
                         "whole MoE filter block at or below the paper's single-filter budget.")
    p.add_argument("--share-basis", action="store_true",
                    help="if --rank is set, share one `down` projection across all experts "
                         "(only `up` differs per expert) for maximum parameter savings")

    p.add_argument("--no-layernorm-tuning", action="store_true",
                    help="disable the paper's LayerNorm-unfreezing trick")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _calibration_batches(loader, n_images):
    """Yield batches from `loader`, cycling it if needed, until at least
    n_images images have been produced."""
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
def collect_block_io(model, layer_indices, calib_batches, device, n_images):
    """Hook the ORIGINAL (pre-substitution) blocks at `layer_indices` and
    record their input/output activations over calibration images, for the
    pseudo-inverse initialization (paper Eq. 3-4). Only used by the
    snip_once fast path -- the moving_average search does this internally
    for every candidate."""
    io_store = {idx: ([], []) for idx in layer_indices}
    hooks = []

    for idx in layer_indices:
        blk = model.blocks[idx]

        def pre_hook(module, inp, _idx=idx):
            io_store[_idx][0].append(inp[0].detach().cpu())

        def fwd_hook(module, inp, out, _idx=idx):
            io_store[_idx][1].append(out.detach().cpu())

        hooks.append(blk.register_forward_pre_hook(pre_hook))
        hooks.append(blk.register_forward_hook(fwd_hook))

    model.eval()
    collected = 0
    for x, _ in calib_batches:
        model(x.to(device))
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


def train_one_epoch(model, loader, optim, loss_fn, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        out = model(x)
        task_loss = loss_fn(out, y)
        aux_loss = collect_aux_loss(model).to(device)
        loss = task_loss + aux_loss
        loss.backward()
        optim.step()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.smoketest or args.data_root is None:
        train_loader, val_loader, test_loader, num_classes = build_cifar100_smoketest(
            img_size=args.img_size, batch_size=args.batch_size)
    else:
        train_loader, val_loader, test_loader, num_classes = build_vtab_task(
            args.data_root, args.task, img_size=args.img_size, batch_size=args.batch_size)

    # 0. pretrained backbone (ViT-B/16, ImageNet-21K, as in the paper)
    model = timm.create_model(args.backbone, pretrained=True, num_classes=num_classes)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    moe_kwargs = dict(
        num_experts=args.num_experts,
        k=args.top_k,
        noise_eps=args.noise_eps,
        expert_init_noise_std=args.expert_init_noise_std,
        aux_loss_weight=args.aux_loss_weight,
        rank=args.rank,
        share_basis=args.share_basis,
    )

    if args.layer_selection == "moving_average":
        # 1+2+3 combined: the search itself trains every candidate, scores
        # them with the moving-average SNIP metric, prunes down to the
        # winner(s), and leaves the winner(s) already substituted in.
        layer_indices, filter_blocks, q_scores = run_moving_average_snip_search(
            model, train_loader, loss_fn, device, moe_kwargs,
            num_replace_layers=args.num_replace_layers,
            calib_images=args.calib_images,
            alpha=args.search_alpha,
            steps_per_candidate=args.search_steps_per_candidate,
            prune_fraction=args.search_prune_fraction,
            max_rounds=args.search_max_rounds,
            lr=args.lr, weight_decay=args.weight_decay,
            tune_layernorm=not args.no_layernorm_tuning,
        )
    else:
        # cheap single-shot approximation: one backward pass ranks blocks by
        # how little the ORIGINAL pretrained computation matters to the
        # task loss; lowest-saliency block(s) are picked directly, no
        # candidate training/pruning.
        calib_x, calib_y = next(iter(train_loader))
        scores = snip_layer_scores(model, calib_x, calib_y, loss_fn, device)
        layer_indices = rank_layers_for_replacement(scores, args.num_replace_layers)
        print(f"[SFP] one-shot SNIP scores (lower = more replaceable): "
              f"{dict(sorted(scores.items(), key=lambda kv: kv[1]))}")
        print(f"[SFP] Replacing transformer block(s) {layer_indices} out of {len(scores)} total blocks")

        calib_batches = _calibration_batches(train_loader, args.calib_images)
        io_pairs = collect_block_io(model, layer_indices, calib_batches, device, args.calib_images)

        filter_blocks = substitute_layers(model, layer_indices, moe_kwargs)
        for idx, fb in filter_blocks.items():
            x_in, x_out = io_pairs[idx]
            fb.to(device)
            fb.init_from_pseudo_inverse(x_in.to(device), x_out.to(device))

    dim = model.embed_dim
    paper_single_filter_params = dim * dim
    moe_filter_params = sum(fb.filter_param_count() for fb in filter_blocks.values())
    print(f"[SFP] filter block params: {moe_filter_params:,} for {len(filter_blocks)} block(s) "
          f"vs. paper's dense single-filter equivalent of {paper_single_filter_params:,} per block "
          f"({100*moe_filter_params/(paper_single_filter_params*len(filter_blocks)):.1f}% of that budget)")

    # freeze everything except filter block(s) + LayerNorm + head, for the
    # main fine-tuning phase (the moving_average path already trained the
    # survivor briefly during search; this continues from there with a
    # fresh optimizer/scheduler for the full requested --epochs)
    freeze_all_except(model, filter_blocks, tune_layernorm=not args.no_layernorm_tuning, tune_head=True)
    model.to(device)

    n_trainable, n_total = count_trainable_params(model)
    print(f"[SFP] trainable params: {n_trainable/1e6:.3f}M / {n_total/1e6:.2f}M "
          f"({100*n_trainable/n_total:.2f}%)")

    optim = AdamW([p for p in model.parameters() if p.requires_grad],
                   lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=args.epochs)

    best_val = 0.0
    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, optim, loss_fn, device)
        sched.step()
        val_acc = evaluate(model, val_loader, device)
        best_val = max(best_val, val_acc)
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f"[SFP] epoch {epoch+1}/{args.epochs}  val_acc={val_acc:.2f}%  best={best_val:.2f}%")

    test_acc = evaluate(model, test_loader, device)
    print(f"[SFP] FINAL test_acc={test_acc:.2f}%  trainable_params={n_trainable/1e6:.3f}M "
          f"({100*n_trainable/n_total:.2f}% of backbone)")


if __name__ == "__main__":
    main()
