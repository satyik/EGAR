"""
SCALE-C3 "Ultimate" — Dual-Head CIFAR-100 Training Script
Architecture strictly follows the specification:
  - 6-stage asymmetric pyramid: Cs=[64,256,224,256,384,320], Ts=[8,10,10,10,10,10]
  - Parameter-free channel tiling (no 1x1 projection)
  - Recursive block: X_{t+1} = X_t + Gate(X_t) * F(X_t)  with momentum term
  - Dual head: Fine(320->100) + Super(320->20)
  - Loss: L_fine + 0.5 * L_super
  - SAM optimizer (AdamW base)
  - CutMix (prob=0.5) + RandAugment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# CIFAR-100 superclass mapping (fine -> superclass, 0-indexed)
# ---------------------------------------------------------------------------
CIFAR100_TO_SUPER = torch.tensor([
    4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
    3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
    6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
    0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
    5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
   16,  4, 17,  4,  2,  0, 17,  4, 18, 17,
   10,  3,  2, 12, 12, 16, 12,  1,  9, 19,
    2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
   16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
   18,  1,  2, 15,  6,  0, 17,  8, 14, 13,
])

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RecursiveBlock(nn.Module):
    """
    Single shared weight-set iterated T times.
    F(x): groups=2 channel-mix 1x1 -> shuffle -> depthwise 3x3
    Gate: entropy gate derived from dilation-shifted differences.
    Momentum term: alpha * (x - x_prev)
    """
    def __init__(self, C: int, T: int):
        super().__init__()
        self.T = T

        # 1x1 grouped channel mixing (groups=2)
        self.ch_mix = nn.Conv2d(C, C, kernel_size=1, groups=2, bias=False)

        # Depthwise 3x3 (groups=C)
        self.dw3x3 = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False)

        # Per-step learnable IS-Norm affine parameters
        self.gammas = nn.Parameter(torch.ones(T, 1, C, 1, 1))
        self.betas  = nn.Parameter(torch.zeros(T, 1, C, 1, 1))

        # Momentum scalar
        self.alpha = nn.Parameter(torch.tensor(0.25))

    @staticmethod
    def _entropy_gate(x):
        """Pixel-level gate using dilation-1,2,4 shift differences."""
        gate = 0.0
        for d in (1, 2, 4):
            gate = gate + (torch.abs(x - x.roll(d, 2)) + torch.abs(x - x.roll(d, 3)))
        return torch.sigmoid(gate / 3.0)

    def forward(self, x):
        x_prev = x
        C = x.shape[1]

        for t in range(self.T):
            gate = self._entropy_gate(x)

            # Channel mix -> channel shuffle
            h = self.ch_mix(x)
            B, _, H, W = h.shape
            h = h.view(B, 2, C // 2, H, W).transpose(1, 2).reshape(B, C, H, W)

            # Depthwise spatial conv
            h = self.dw3x3(h)

            # IS-Norm with per-step affine
            h = F.instance_norm(h) * self.gammas[t] + self.betas[t]
            h = F.gelu(h)

            x_new = x + gate * h + self.alpha * (x - x_prev)
            x_prev = x
            x = x_new

        return x


def _tile_channels(x: torch.Tensor, target_C: int) -> torch.Tensor:
    """Parameter-free channel expansion via tiling."""
    C = x.shape[1]
    if target_C == C:
        return x
    if target_C > C:
        repeats = (target_C + C - 1) // C
        return x.repeat(1, repeats, 1, 1)[:, :target_C]
    return x[:, :target_C]


class ScaleC3Ultimate(nn.Module):
    """
    6-stage asymmetric pyramid with dual classification head.

    Stage | T  | C   | Downsample
    ------|----|-----|----------
      0   |  8 |  64 | No  (stem)
      1   | 10 | 256 | No
      2   | 10 | 256 | Yes (16x16)
      3   | 10 | 224 | Yes (8x8)
      4   | 10 | 256 | No
      5   | 10 | 384 | Yes (4x4)
      6   | 10 | 320 | No

    Wait — the spec has 6 stages (indices 0-5) with Stem=stage-0.
    Cs   = [64, 256, 224, 256, 384, 320]
    Ts   = [ 8,  10,  10,  10,  10,  10]
    down = [F,   T,   T,   F,   T,   F]

    Dual head:
      fine  head : Linear(320, num_classes)       — default 100
      super head : Linear(320, num_super_classes)  — default 20
    """
    Cs   = [64, 256, 224, 256, 384, 320]
    Ts   = [ 8,  10,  10,  10,  10,  10]
    DOWN = [False, True, True, False, True, False]

    def __init__(self, num_classes: int = 100, num_super_classes: int = 20):
        super().__init__()
        self.stages = nn.ModuleList(
            RecursiveBlock(C, T) for C, T in zip(self.Cs, self.Ts)
        )
        embed = self.Cs[-1]
        self.head_fine  = nn.Linear(embed, num_classes,       bias=False)
        self.head_super = nn.Linear(embed, num_super_classes, bias=False)

    def forward(self, x: torch.Tensor):
        # Parameter-free stem: tile 3-channel input to 64 channels
        x = _tile_channels(x, self.Cs[0])

        for i, stage in enumerate(self.stages):
            if i > 0:
                if self.DOWN[i]:
                    x = F.avg_pool2d(x, 2, 2)
                x = _tile_channels(x, self.Cs[i])
            x = stage(x)

        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head_fine(x), self.head_super(x)


# ---------------------------------------------------------------------------
# SAM Optimizer
# ---------------------------------------------------------------------------

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        g_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (g_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale.to(p))
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack([
                p.grad.norm(2).to(device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            2,
        )

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# CutMix
# ---------------------------------------------------------------------------

def _rand_bbox(W, H, lam):
    cut_w = int(W * np.sqrt(1.0 - lam))
    cut_h = int(H * np.sqrt(1.0 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix(inputs, targets, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    B, _, H, W = inputs.shape
    idx = torch.randperm(B, device=inputs.device)
    x1, y1, x2, y2 = _rand_bbox(W, H, lam)
    mixed = inputs.clone()
    mixed[:, :, y1:y2, x1:x2] = inputs[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, targets, targets[idx], lam


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_loaders(batch_size=128):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    trainset = torchvision.datasets.CIFAR100('./data', train=True,  download=True, transform=train_tf)
    testset  = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(testset,  batch_size=256,        shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _sam_pass(model, inputs, labels_fine, labels_super, ce_fine, ce_super, supermap, device):
    """Compute combined loss for one SAM forward pass."""
    out_fine, out_super = model(inputs)
    l_fine  = ce_fine(out_fine, labels_fine)
    l_super = ce_super(out_super, labels_super)
    return l_fine + 0.5 * l_super


def _sam_pass_cutmix(model, inputs, fa, fb, lam, ce_fine, ce_super, supermap, device):
    sa = supermap[fa]
    sb = supermap[fb]
    out_fine, out_super = model(inputs)
    l_fine  = lam * ce_fine(out_fine, fa)  + (1 - lam) * ce_fine(out_fine, fb)
    l_super = lam * ce_super(out_super, sa) + (1 - lam) * ce_super(out_super, sb)
    return l_fine + 0.5 * l_super


def train(epochs=150, batch_size=128, lr=1e-3, cutmix_prob=0.5, weight_decay=5e-4, rho=0.05):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = ScaleC3Ultimate(num_classes=100, num_super_classes=20).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")

    supermap = CIFAR100_TO_SUPER.to(device)

    ce_fine  = nn.CrossEntropyLoss(label_smoothing=0.1)
    ce_super = nn.CrossEntropyLoss()

    optimizer = SAM(model.parameters(), optim.AdamW, rho=rho, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=epochs, eta_min=1e-5)

    train_loader, test_loader = build_loaders(batch_size)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (imgs, labels) in enumerate(train_loader):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            labels_super = supermap[labels]

            optimizer.zero_grad()

            if np.random.rand() < cutmix_prob:
                mixed, fa, fb, lam = cutmix(imgs, labels)

                # SAM step 1
                loss = _sam_pass_cutmix(model, mixed, fa, fb, lam, ce_fine, ce_super, supermap, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.first_step(zero_grad=True)

                # SAM step 2
                loss = _sam_pass_cutmix(model, mixed, fa, fb, lam, ce_fine, ce_super, supermap, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.second_step(zero_grad=True)
            else:
                # SAM step 1
                loss = _sam_pass(model, imgs, labels, labels_super, ce_fine, ce_super, supermap, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.first_step(zero_grad=True)

                # SAM step 2
                loss = _sam_pass(model, imgs, labels, labels_super, ce_fine, ce_super, supermap, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.second_step(zero_grad=True)

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"[{epoch+1}/{epochs}, step {i+1}] loss={running_loss/100:.3f}")
                running_loss = 0.0

        scheduler.step()

        # Evaluation
        model.eval()
        top1 = top5 = super_correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                ls     = supermap[labels]

                out_fine, out_super = model(imgs)

                pred1 = out_fine.argmax(1)
                top1 += (pred1 == labels).sum().item()
                super_correct += (out_super.argmax(1) == ls).sum().item()
                total += labels.size(0)

                _, pred5_idx = out_fine.topk(5, dim=1)
                top5 += pred5_idx.eq(labels.view(-1, 1)).any(1).sum().item()

        acc1   = 100 * top1  / total
        acc5   = 100 * top5  / total
        acc_s  = 100 * super_correct / total
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{epochs} lr={lr_now:.5f} | Super: {acc_s:.2f}% | Top-1: {acc1:.2f}% | Top-5: {acc5:.2f}%")

        if acc1 > best_acc:
            best_acc = acc1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, "best_scale_c3_ultimate_dual.pth")
            print(f"  => Saved new best ({best_acc:.2f}%)")

    print(f"\nTraining complete. Best Top-1: {best_acc:.2f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=150)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--cutmix_prob",  type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--rho",          type=float, default=0.05)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        cutmix_prob=args.cutmix_prob,
        weight_decay=args.weight_decay,
        rho=args.rho,
    )
