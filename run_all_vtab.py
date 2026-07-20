"""
Run SFP+MoE across all 19 VTAB-1k tasks and aggregate into a Table-1-style
summary CSV.

Usage:
    python run_all_vtab.py --data-root /path/to/vtab-1k --epochs 100

This just shells out to train_sfp_moe.py once per task (simplest way to keep
each run's state fully isolated) and parses the printed "FINAL test_acc=" line.
Swap --tasks if your dump uses different folder names for a subset of tasks.
"""
import argparse
import csv
import re
import subprocess
import sys

DEFAULT_VTAB_TASKS = [
    "cifar100", "caltech101", "dtd", "oxford_flowers102", "oxford_iiit_pet",
    "svhn", "sun397",
    "patch_camelyon", "eurosat", "resisc45", "diabetic_retinopathy",
    "clevr_count", "clevr_dist", "dmlab", "kitti",
    "dsprites_loc", "dsprites_ori", "smallnorb_azi", "smallnorb_ele",
]

FINAL_ACC_RE = re.compile(r"FINAL test_acc=([\d.]+)%")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--tasks", nargs="+", default=DEFAULT_VTAB_TASKS)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--num-replace-layers", type=int, default=1, choices=[1, 2])
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--out-csv", default="vtab_results.csv")
    p.add_argument("--extra-args", default="", help="extra flags passed through verbatim, e.g. '--lr 5e-4'")
    return p.parse_args()


def main():
    args = parse_args()
    results = []
    for task in args.tasks:
        cmd = [
            sys.executable, "train_sfp_moe.py",
            "--data-root", args.data_root,
            "--task", task,
            "--epochs", str(args.epochs),
            "--num-replace-layers", str(args.num_replace_layers),
            "--num-experts", str(args.num_experts),
            "--top-k", str(args.top_k),
        ] + args.extra_args.split()

        print(f"\n===== {task} =====\n{' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout[-2000:])
        if proc.returncode != 0:
            print(proc.stderr[-2000:], file=sys.stderr)
            results.append((task, None))
            continue

        m = FINAL_ACC_RE.search(proc.stdout)
        acc = float(m.group(1)) if m else None
        results.append((task, acc))

    valid = [acc for _, acc in results if acc is not None]
    mean_acc = sum(valid) / len(valid) if valid else float("nan")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "test_acc"])
        for task, acc in results:
            writer.writerow([task, acc])
        writer.writerow(["MEAN", mean_acc])

    print(f"\nWrote results to {args.out_csv}")
    print(f"MEAN accuracy across {len(valid)}/{len(args.tasks)} tasks: {mean_acc:.2f}%")


if __name__ == "__main__":
    main()
