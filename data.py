"""
Data loading.

`build_vtab_task` assumes the common VTAB-1k export layout used by most
public dump scripts (e.g. the one behind Xin et al. 2024b's V-PETL bench,
which this paper's experimental setting is explicitly based on):

    <root>/<task_name>/train800/<class>/*.jpg
    <root>/<task_name>/val200/<class>/*.jpg
    <root>/<task_name>/test/<class>/*.jpg

If your local VTAB-1k copy uses a different layout (e.g. raw tfds records),
swap the body of `build_vtab_task` -- everything downstream (model surgery,
training loop) only depends on getting back three DataLoaders + num_classes.

`build_cifar100_smoketest` needs no external data download setup beyond
torchvision's built-in CIFAR-100 fetcher, and mimics VTAB-1k's 1000-shot
(800 train / 200 val) regime, so you can exercise the whole pipeline
end-to-end before pointing it at real VTAB-1k data.
"""
import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset


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


def build_vtab_task(root: str, task_name: str, img_size: int = 224,
                     batch_size: int = 64, num_workers: int = 4):
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
