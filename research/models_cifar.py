"""
SCALE-C3 Ultimate — recursive depthwise-dilated CNN for CIFAR-100.
~299,360 parameters. No 1×1 expansion convolutions; channel growth via tiling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CIFAR-100 superclass mapping (20 coarse labels)
# ---------------------------------------------------------------------------
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
# tensor for fast GPU lookup
_SUPER_LABEL_MAP = torch.tensor(CIFAR100_SUPERCLASS, dtype=torch.long)


def fine_to_super(fine_labels: torch.Tensor) -> torch.Tensor:
    return _SUPER_LABEL_MAP.to(fine_labels.device)[fine_labels]


# ---------------------------------------------------------------------------
# Recursive block ingredients
# ---------------------------------------------------------------------------

class _ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        g = self.groups
        x = x.view(B, g, C // g, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return x.view(B, C, H, W)


class _RecursiveBlock(nn.Module):
    """
    One set of shared weights, applied T times.

    F(x) = DilatedEnsemble(PointwiseCompress(ChannelMix+Shuffle(x)))
    X_{t+1} = X_t + sigmoid(gate(X_t)) * F(X_t)
    """

    def __init__(self, channels: int, dilations=(1, 2, 4)):
        super().__init__()
        C = channels

        # --- channel mixing (groups=2) + shuffle --------------------------
        self.ch_mix   = nn.Conv2d(C, C, 1, groups=2, bias=False)
        self.ch_bn    = nn.BatchNorm2d(C)
        self.shuffle  = _ChannelShuffle(2)

        # --- pointwise compression ----------------------------------------
        self.pw       = nn.Conv2d(C, C, 1, bias=False)
        self.pw_bn    = nn.BatchNorm2d(C)

        # --- dilated depthwise ensemble -----------------------------------
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(C, C, 3, padding=d, dilation=d, groups=C, bias=False)
            for d in dilations
        ])
        self.dw_bn    = nn.BatchNorm2d(C)

        # --- gating (scalar gate per channel, applied spatially) ----------
        self.gate     = nn.Conv2d(C, C, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        # channel mix + shuffle
        h = F.relu(self.ch_bn(self.ch_mix(x)))
        h = self.shuffle(h)
        # pointwise compress
        h = F.relu(self.pw_bn(self.pw(h)))
        # dilated ensemble (sum)
        h = sum(dw(h) for dw in self.dw_convs)
        h = F.relu(self.dw_bn(h))
        # gated residual
        g = torch.sigmoid(self.gate(x))
        return x + g * h

    def forward(self, x: torch.Tensor, T: int,
                capture_iters: bool = False):
        """
        Run T recursive iterations.
        If capture_iters=True, return list of intermediate X_t tensors
        (detached, on CPU, as float32) for analysis — avoids OOM.
        """
        if not capture_iters:
            for _ in range(T):
                x = self.forward_once(x)
            return x, None

        snapshots = []
        for _ in range(T):
            x = self.forward_once(x)
            snapshots.append(x.detach().cpu().float())
        return x, snapshots


# ---------------------------------------------------------------------------
# Full architecture
# ---------------------------------------------------------------------------

class _ScaleC3SuperArchitecture(nn.Module):
    """
    SCALE-C3 Ultimate  (~299 k parameters)

    Stage layout  (T=iterations, C=channels, downsample=AvgPool2d):
      Stem  : 3→64, 3×3 conv
      S1    : T=8,  C=64,  32×32
      S2    : T=10, C=256, 16×16  (tile ×4, then AvgPool)
      S3    : T=10, C=224,  8×8   (AvgPool first, then tile)
      S4    : T=10, C=256,  8×8
      S5    : T=10, C=384,  4×4   (AvgPool)
      S6    : T=10, C=320,  4×4
    """

    STAGE_CFG = [
        # (T,  C_in,  C_out,  pool_before, tile_factor)
        (8,   64,   64,   False, 1),
        (10,  64,  256,   True,  4),   # pool then tile
        (10, 256,  224,   True,  1),   # pool then slice
        (10, 224,  256,   False, 1),
        (10, 256,  384,   True,  1),
        (10, 384,  320,   False, 1),
    ]

    def __init__(self, num_classes: int = 100, num_superclasses: int = 20):
        super().__init__()

        # Stem: 3 → 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Recursive blocks — one per stage, C = max(C_in, C_out)
        self.blocks = nn.ModuleList()
        self.stage_channels = []
        for T, C_in, C_out, pool_before, tile in self.STAGE_CFG:
            C = C_out  # block operates at output channels
            self.blocks.append(_RecursiveBlock(C))
            self.stage_channels.append(C)

        # Heads
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head_fine  = nn.Linear(320, num_classes)
        self.head_super = nn.Linear(320, num_superclasses)

    # -----------------------------------------------------------------------
    def _prepare_input(self, x, C_in, C_out, pool_before, tile):
        """Adapt spatial size and channel count before block."""
        if pool_before:
            x = F.avg_pool2d(x, 2, 2)
        if tile > 1:
            # repeat channels — zero params
            x = x.repeat(1, tile, 1, 1)
        elif C_out < C_in:
            # slice (never needed in default config but kept for flexibility)
            x = x[:, :C_out]
        elif C_out > C_in:
            # pad with zeros
            pad = torch.zeros(x.size(0), C_out - C_in, x.size(2), x.size(3),
                              device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        return x

    def forward(self, x: torch.Tensor,
                capture_stage: int = -1,
                capture_iters: bool = False):
        """
        Args:
            capture_stage  : if >=0, capture iteration snapshots for that stage
            capture_iters  : whether to actually collect snapshots

        Returns:
            logits_fine, logits_super[, snapshots_list]
        """
        x = self.stem(x)

        all_snapshots = None
        for i, (block, (T, C_in, C_out, pool, tile)) in enumerate(
                zip(self.blocks, self.STAGE_CFG)):
            x = self._prepare_input(x, C_in, C_out, pool, tile)
            do_cap = capture_iters and (i == capture_stage)
            x, snaps = block(x, T, capture_iters=do_cap)
            if do_cap:
                all_snapshots = snaps

        emb = self.pool(x).flatten(1)            # [B, 320]
        lf  = self.head_fine(emb)
        ls  = self.head_super(emb)

        if capture_iters and capture_stage >= 0:
            return lf, ls, all_snapshots
        return lf, ls


# ---------------------------------------------------------------------------
# Public constructor
# ---------------------------------------------------------------------------

def scale_c3_ultimate(num_classes: int = 100,
                      num_superclasses: int = 20) -> _ScaleC3SuperArchitecture:
    return _ScaleC3SuperArchitecture(num_classes, num_superclasses)


# ---------------------------------------------------------------------------
# Quick parameter count
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = scale_c3_ultimate()
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    x = torch.randn(2, 3, 32, 32)
    lf, ls = model(x)
    print(f"Fine logits: {lf.shape}, Super logits: {ls.shape}")
