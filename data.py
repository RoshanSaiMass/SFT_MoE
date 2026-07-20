"""
Data loading.

`build_vtab_task` targets the VTAB-1k layout used by the NOAH / SSF /
AdaptFormer setup scripts (the community-standard pre-extracted VTAB-1k
copy that this paper's experimental protocol, via Xin et al. 2024b's
V-PETL bench, is built on top of). That layout is NOT one-folder-per-class
(`ImageFolder`); it's one flat `images/` directory per task plus plain-text
annotation files:

    <root>/<task_name>/images/*.jpg          (or nested subfolders, doesn't matter)
    <root>/<task_name>/train800.txt          (each line: "<rel_path> <label>")
    <root>/<task_name>/val200.txt
    <root>/<task_name>/test.txt

`VTABAnnotatedDataset` reads exactly that. If your local copy instead uses
per-class subfolders, use `build_vtab_task_imagefolder` instead (same
return signature, drop-in replacement in train_sfp_moe.py).

`build_cifar100_smoketest` needs no external data download setup beyond
torchvision's built-in CIFAR-100 fetcher, and mimics VTAB-1k's 1000-shot
(800 train / 200 val) regime, so you can exercise the whole pipeline
end-to-end before pointing it at real VTAB-1k data.
"""
import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image


def build_transforms(img_size: int = 224):
    normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    train_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    eval_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


class VTABAnnotatedDataset(Dataset):
    """One flat `images/` directory (or any nested layout, doesn't matter --
    paths in the annotation file are relative to `task_root`) plus a
    `<split>.txt` file where each line is `<relative_path> <label>`."""

    def __init__(self, task_root: str, ann_filename: str, transform=None):
        self.task_root = task_root
        self.transform = transform
        self.samples = []
        ann_path = os.path.join(task_root, ann_filename)
        with open(ann_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, label = line.rsplit(" ", 1)
                self.samples.append((rel_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img_path = os.path.join(self.task_root, rel_path)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def num_classes(self):
        return max(label for _, label in self.samples) + 1


def build_vtab_task(root: str, task_name: str, img_size: int = 224,
                     batch_size: int = 64, num_workers: int = 4,
                     train_ann: str = "train800.txt", val_ann: str = "val200.txt",
                     test_ann: str = "test.txt"):
    """Default: NOAH/SSF-style annotation-file layout. If your VTAB-1k
    dump uses different annotation filenames, pass `train_ann=...` etc."""
    train_tf, eval_tf = build_transforms(img_size)
    task_root = os.path.join(root, task_name)

    train_set = VTABAnnotatedDataset(task_root, train_ann, transform=train_tf)
    val_set = VTABAnnotatedDataset(task_root, val_ann, transform=eval_tf)
    test_set = VTABAnnotatedDataset(task_root, test_ann, transform=eval_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = max(train_set.num_classes, test_set.num_classes)
    return train_loader, val_loader, test_loader, num_classes


def build_vtab_task_imagefolder(root: str, task_name: str, img_size: int = 224,
                                 batch_size: int = 64, num_workers: int = 4):
    """Fallback loader for VTAB-1k copies already reorganized into
    per-class subfolders (`train800/<class>/*.jpg`, etc). Same return
    signature as build_vtab_task -- use this in train_sfp_moe.py instead if
    your data is laid out this way."""
    train_tf, eval_tf = build_transforms(img_size)
    task_root = os.path.join(root, task_name)

    train_set = torchvision.datasets.ImageFolder(os.path.join(task_root, "train800"), transform=train_tf)
    val_set = torchvision.datasets.ImageFolder(os.path.join(task_root, "val200"), transform=eval_tf)
    test_set = torchvision.datasets.ImageFolder(os.path.join(task_root, "test"), transform=eval_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(train_set.classes)
    return train_loader, val_loader, test_loader, num_classes


def build_cifar100_smoketest(root: str = "./data", img_size: int = 224,
                              batch_size: int = 64, num_workers: int = 2):
    train_tf, eval_tf = build_transforms(img_size)
    train_full = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=eval_tf)

    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(train_full), generator=g)[:1000]
    train_idx, val_idx = idx[:800].tolist(), idx[800:].tolist()
    train_set = Subset(train_full, train_idx)
    val_set = Subset(train_full, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, 100
