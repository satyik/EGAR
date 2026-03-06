"""
Training script for SCALE-C3 Ultimate on CIFAR-100.

Features:
  - CutMix augmentation (p=0.5)
  - Dual hierarchical loss: L_fine + 0.5 * L_super
  - Label smoothing (ε=0.1) on fine head
  - Cosine LR schedule with linear warmup
  - Gradient logging for analysis (optional)
"""

import argparse
import os
import time
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
# from torch.cuda.amp import GradScaler, autocast


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models_cifar import create_scale_c3_super

CIFAR100_SUPERCLASS = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]
_SUPER_LABEL_MAP = torch.tensor(CIFAR100_SUPERCLASS, dtype=torch.long)

def fine_to_super(fine_labels: torch.Tensor) -> torch.Tensor:
    return _SUPER_LABEL_MAP.to(fine_labels.device)[fine_labels]


# ---------------------------------------------------------------------------
# CutMix
# ---------------------------------------------------------------------------

def rand_bbox(size, lam):
    """Return (x1, y1, x2, y2) of the cut box given mixing ratio lam."""
    W, H = size[3], size[2]
    cut_rat = (1.0 - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def cutmix_data(x, y, alpha=1.0, prob=0.5):
    """
    Apply CutMix with probability `prob`.
    Returns mixed images, (target_a, target_b, lambda).
    lambda = area of kept region / total area (target_a proportion).
    """
    if random.random() > prob:
        return x, y, y, 1.0   # no mix

    lam = random.betavariate(alpha, alpha)
    B = x.size(0)
    idx = torch.randperm(B, device=x.device)

    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x_mixed = x.clone()
    x_mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]

    # Adjust lambda to true pixel ratio
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (x.size(2) * x.size(3))
    return x_mixed, y, y[idx], lam


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def hierarchical_loss(logits_fine, logits_super, target_a, target_b, lam,
                      criterion_fine, criterion_super):
    """
    L = lam * CE_fine(a) + (1-lam) * CE_fine(b)
      + 0.5 * [lam * CE_super(a) + (1-lam) * CE_super(b)]
    """
    super_a = fine_to_super(target_a)
    super_b = fine_to_super(target_b)

    L_fine = lam * criterion_fine(logits_fine, target_a) + \
             (1 - lam) * criterion_fine(logits_fine, target_b)
    L_super = lam * criterion_super(logits_super, super_a) + \
              (1 - lam) * criterion_super(logits_super, super_b)

    return L_fine + 0.5 * L_super


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataloaders(data_root: str, batch_size: int, num_workers: int = 4):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR100(data_root, train=True,
                                              download=True, transform=train_tf)
    val_ds   = torchvision.datasets.CIFAR100(data_root, train=False,
                                              download=True, transform=val_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int,
                    steps_per_epoch: int):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item())

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Gradient magnitude logging
# ---------------------------------------------------------------------------

class GradLogger:
    """
    Attaches hooks to the recursive block of a given stage to log
    per-iteration gradient norms. Call .reset() each batch,
    .summarise() to get the dict.
    """

    def __init__(self, model, stage_idx: int = 0):
        self.stage_idx = stage_idx
        self.grads = []
        stage = model.stages[stage_idx]
        # hook on the depthwise convs (first dilation)
        stage.conv3x3_dw.weight.register_hook(self._hook)

    def _hook(self, grad):
        self.grads.append(grad.norm().item())

    def reset(self):
        self.grads.clear()

    def summarise(self):
        return {f"grad_stage{self.stage_idx}_iter{i}": v
                for i, v in enumerate(self.grads)}


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        lf, _ = model(imgs)
        pred = lf.argmax(1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def train_one_epoch(model, loader, optimizer, scheduler, scaler,
                    criterion_fine, criterion_super, device, log_grads=False):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        imgs_mix, ta, tb, lam = cutmix_data(imgs, labels, prob=0.5)

        optimizer.zero_grad()
        with torch.amp.autocast(device):
            lf, ls = model(imgs_mix)
            loss   = hierarchical_loss(lf, ls, ta, tb, lam,

                                       criterion_fine, criterion_super)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        
        # Only step the learning rate scheduler if optimizer step wasn't skipped (due to nan/inf gradients)
        if scale_before <= scaler.get_scale():
            scheduler.step()

        running_loss += loss.item() * imgs.size(0)
        pred = lf.argmax(1)
        # accuracy measured against the dominant label only
        dominant = ta if lam >= 0.5 else tb
        correct += (pred == dominant).sum().item()
        total   += labels.size(0)

    return running_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',        default='./data')
    p.add_argument('--epochs',      type=int, default=200)
    p.add_argument('--batch-size',  type=int, default=128)
    p.add_argument('--lr',          type=float, default=0.1)
    p.add_argument('--wd',          type=float, default=5e-4)
    p.add_argument('--warmup',      type=int, default=5)
    p.add_argument('--workers',     type=int, default=4)
    p.add_argument('--save-dir',    default='./checkpoints')
    p.add_argument('--log-grads',   action='store_true',
                   help='Log per-iteration gradient norms for stage 0')
    p.add_argument('--resume',      default='')
    return p.parse_args()


def main():
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    train_loader, val_loader = build_dataloaders(
        args.data, args.batch_size, args.workers)

    # Model
    model = create_scale_c3_super(num_macro_classes=None).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Loss
    criterion_fine  = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_super = nn.CrossEntropyLoss()

    # Optimiser
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.wd, nesterov=True)
    scheduler = build_scheduler(optimizer, args.warmup, args.epochs,
                                 len(train_loader))
    scaler = torch.amp.GradScaler(device)

    start_epoch = 0
    best_acc    = 0.0
    history     = []

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_acc    = ckpt.get('best_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}")

    grad_logger = GradLogger(model, stage_idx=0) if args.log_grads else None

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion_fine, criterion_super, device, args.log_grads)
        val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        row = dict(epoch=epoch, train_loss=round(train_loss, 4),
                   train_acc=round(train_acc, 2), val_acc=round(val_acc, 2),
                   lr=round(optimizer.param_groups[0]['lr'], 6))
        if grad_logger:
            row.update(grad_logger.summarise())
            grad_logger.reset()
        history.append(row)

        print(f"[{epoch:03d}/{args.epochs}] "
              f"loss={train_loss:.4f}  train={train_acc:.1f}%  "
              f"val={val_acc:.1f}%  ({elapsed:.0f}s)")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch, 'best_acc': best_acc},
                       os.path.join(args.save_dir, 'best.pt'))

        # periodic checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch, 'best_acc': best_acc},
                       os.path.join(args.save_dir, f'epoch_{epoch:03d}.pt'))

    # save history
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
