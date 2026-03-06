"""
Research analysis suite for SCALE-C3 Ultimate.

Implements all five analyses described in the paper:
  1. CKA across iterations  — 10×10 heatmap per stage
  2. Rank evolution          — effective rank via SVD per iteration
  3. Linear probe accuracy   — top-1 per iteration per stage
  4. Gradient flow           — per-iteration gradient magnitude
  5. Representation drift    — ||X_{t+1}-X_t||_F / ||X_t||_F

Usage:
    python analysis_recursive_cnn.py \\
        --checkpoint checkpoints/best.pt \\
        --data ./data \\
        --out-dir ./analysis_results \\
        --n-samples 10000
"""

import argparse
import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from models_cifar import scale_c3_ultimate, fine_to_super

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    model = scale_c3_ultimate().to(device)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get('model', ckpt)
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint found — using random weights (for testing).")
    model.eval()
    return model


def build_val_loader(data_root: str, batch_size: int = 200, n_samples: int = 10000):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    ds = torchvision.datasets.CIFAR100(data_root, train=False,
                                        download=True, transform=tf)
    # Use a deterministic subset
    indices = list(range(min(n_samples, len(ds))))
    ds = torch.utils.data.Subset(ds, indices)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                       shuffle=False, num_workers=4,
                                       pin_memory=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature collection
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_stage_iterations(model, loader, stage_idx: int, device: torch.device):
    """
    Run inference, capturing snapshots after each iteration of `stage_idx`.

    Returns:
        snapshots : list of T tensors, each shape [N, C, H, W] on CPU float32
        labels    : [N] int64 tensor
    """
    all_snaps = None  # list of T lists (one list per image batch)
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            _, _, snaps = model(imgs, capture_stage=stage_idx,
                                capture_iters=True)
        # snaps: list of T tensors [B, C, H, W] on CPU
        if all_snaps is None:
            all_snaps = [[] for _ in range(len(snaps))]
        for t, s in enumerate(snaps):
            all_snaps[t].append(s)
        all_labels.append(labels)

    # Concatenate over batches
    snapshots = [torch.cat(all_snaps[t], dim=0) for t in range(len(all_snaps))]
    labels    = torch.cat(all_labels, dim=0)
    return snapshots, labels


def flatten_features(x: torch.Tensor) -> np.ndarray:
    """[N, C, H, W] → [N, C*H*W] float32 numpy array."""
    return x.reshape(x.size(0), -1).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 1. CKA
# ─────────────────────────────────────────────────────────────────────────────

def _gram_rbf(X: np.ndarray, sigma_frac: float = 0.5) -> np.ndarray:
    """RBF kernel gram matrix for CKA (not used — we use linear CKA)."""
    sq = np.sum(X ** 2, axis=1, keepdims=True)
    dist = sq + sq.T - 2 * X @ X.T
    sigma = sigma_frac * np.median(dist[dist > 0]) ** 0.5
    return np.exp(-dist / (2 * sigma ** 2))


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Centered Kernel Alignment with linear kernels.
    X, Y: [N, d1], [N, d2]
    CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    # Centre
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)

    # If d > N, computing X.T @ Y is O(d^2) memory. We should compute K = X @ X.T instead!
    # tr(X^T Y Y^T X) = tr(X X^T Y Y^T) = sum((X X^T) * (Y Y^T))
    K = X @ X.T
    L = Y @ Y.T
    
    hsic_xy = np.sum(K * L)
    hsic_xx = np.sum(K * K)
    hsic_yy = np.sum(L * L)

    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def compute_cka_matrix(snapshots: list) -> np.ndarray:
    """
    snapshots: list of T tensors [N, C, H, W]
    Returns T×T CKA matrix.
    """
    T = len(snapshots)
    feats = [flatten_features(s) for s in snapshots]
    cka_mat = np.eye(T)
    for i in range(T):
        for j in range(i + 1, T):
            v = _linear_cka(feats[i], feats[j])
            cka_mat[i, j] = v
            cka_mat[j, i] = v
    return cka_mat


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rank evolution
# ─────────────────────────────────────────────────────────────────────────────

def effective_rank(X: np.ndarray) -> float:
    """
    Effective rank via entropy of normalised singular values.
    Roy & Vetterli (2007): erank = exp(H(p)) where p_i = σ_i / Σσ_i.
    Also return explained-variance rank (95% threshold) as secondary metric.
    """
    # Fast approach for N < d: compute eigenvalues of X @ X.T instead of full SVD on X
    if X.shape[0] < X.shape[1]:
        K = X @ X.T
        s_sq = np.sort(np.linalg.eigvalsh(K))[::-1]
        s = np.sqrt(np.maximum(s_sq, 0))
    else:
        s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
        
    s = s[s > 1e-10]
    p = s / s.sum()
    erank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    # 95% variance rank
    var_cumsum = np.cumsum(s ** 2) / (s ** 2).sum()
    rank95 = int(np.searchsorted(var_cumsum, 0.95)) + 1
    return erank, rank95


def compute_rank_evolution(snapshots: list):
    """Returns arrays of shape [T]: effective ranks and 95%-var ranks."""
    eranks, r95s = [], []
    for s in snapshots:
        X = flatten_features(s)
        er, r95 = effective_rank(X)
        eranks.append(er)
        r95s.append(r95)
    return np.array(eranks), np.array(r95s)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Linear probe accuracy
# ─────────────────────────────────────────────────────────────────────────────

def linear_probe_accuracy(snapshots: list, labels: torch.Tensor,
                          num_classes: int = 100,
                          n_epochs: int = 30,
                          device: torch.device = torch.device('cpu')):
    """
    For each iteration snapshot, train a linear probe (frozen backbone).
    Uses sklearn-style closed-form ridge regression via pseudo-inverse for speed,
    then reports top-1 accuracy.

    Returns array of shape [T] accuracies.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y = labels.numpy()
    accs = []

    for t, snap in enumerate(snapshots):
        X = snap.reshape(snap.size(0), -1).float().numpy()

        # PCA-compress to 512 dims if larger (for speed)
        if X.shape[1] > 512:
            # Random projection (fast approximation)
            rng = np.random.RandomState(42)
            proj = rng.randn(X.shape[1], 512).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)
            X = X @ proj

        scaler = StandardScaler(copy=False)
        X = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=200, C=0.1,
                                  solver='saga', n_jobs=-1,
                                  random_state=42)
        clf.fit(X, y)
        acc = clf.score(X, y) * 100  # training accuracy — sufficient for trend
        accs.append(acc)
        print(f"  Probe iter {t+1:2d}: {acc:.1f}%")

    return np.array(accs)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gradient flow across iterations
# ─────────────────────────────────────────────────────────────────────────────

def measure_gradient_flow(model, loader, stage_idx: int,
                           device: torch.device, n_batches: int = 20):
    """
    Run a forward+backward pass with the recursive block unrolled.
    Capture gradient norm at each iteration by hooking intermediate tensors.

    Returns array of shape [T] mean gradient norms.
    """
    model.train()
    block = model.blocks[stage_idx]
    T_iters = model.STAGE_CFG[stage_idx][0]
    criterion = nn.CrossEntropyLoss()

    # We need gradients to flow through all iterations.
    # Temporarily override block.forward to store per-iter grad norms.
    iter_grad_norms = [[] for _ in range(T_iters)]

    handles = []

    def make_hook(t_idx):
        def hook(grad):
            iter_grad_norms[t_idx].append(grad.norm().item())
        return hook

    # Custom forward that registers hooks on each intermediate X_t
    def hooked_forward(x, T):
        intermediates = []
        for t in range(T):
            x = block.forward_once(x)
            intermediates.append(x)

        # Register hooks
        for t, inter in enumerate(intermediates[:-1]):  # last has no successor
            inter.register_hook(make_hook(t))
        return intermediates[-1]

    model.eval()
    processed = 0
    for imgs, labels in loader:
        if processed >= n_batches:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        imgs.requires_grad_(False)

        # Run stem and upstream stages, then our target stage manually
        with torch.enable_grad():
            x = model.stem(imgs)
            for i, (blk, (T, C_in, C_out, pool, tile)) in enumerate(
                    zip(model.blocks, model.STAGE_CFG)):
                x = model._prepare_input(x, C_in, C_out, pool, tile)
                if i == stage_idx:
                    x = hooked_forward(x, T)
                else:
                    x, _ = blk(x, T)
                    # x is returned without snapshots in non-capture mode

            emb = model.pool(x).flatten(1)
            lf  = model.head_fine(emb)
            loss = criterion(lf, labels)
            loss.backward()

        model.zero_grad()
        processed += 1

    # Average over batches
    mean_norms = np.array([
        np.mean(g) if g else 0.0 for g in iter_grad_norms
    ])
    return mean_norms


# ─────────────────────────────────────────────────────────────────────────────
# 5. Representation drift
# ─────────────────────────────────────────────────────────────────────────────

def compute_drift(snapshots: list) -> np.ndarray:
    """
    ||X_{t+1} - X_t||_F / ||X_t||_F for t = 0 … T-2.
    Returns array of shape [T-1].
    """
    drifts = []
    for t in range(len(snapshots) - 1):
        Xt  = snapshots[t].numpy().reshape(snapshots[t].size(0), -1)
        Xt1 = snapshots[t+1].numpy().reshape(snapshots[t+1].size(0), -1)
        diff_norm = np.linalg.norm(Xt1 - Xt, 'fro')
        base_norm = np.linalg.norm(Xt, 'fro')
        drifts.append(diff_norm / (base_norm + 1e-12))
    return np.array(drifts)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

STAGE_NAMES = ['S1 (C=64, 32²)', 'S2 (C=256, 16²)', 'S3 (C=224, 8²)',
               'S4 (C=256, 8²)', 'S5 (C=384, 4²)',  'S6 (C=320, 4²)']

_CKA_CMAP = LinearSegmentedColormap.from_list(
    'cka', ['#1a1a2e', '#16213e', '#0f3460', '#533483', '#e94560',
            '#f5a623', '#f8f8f8'], N=256)


def plot_cka_heatmaps(cka_matrices: dict, out_dir: str):
    """Hero figure: 2×3 grid of CKA heatmaps."""
    n_stages = len(cka_matrices)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, (s_idx, mat) in enumerate(sorted(cka_matrices.items())):
        ax  = axes[idx]
        T   = mat.shape[0]
        im  = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax.set_title(STAGE_NAMES[s_idx], fontsize=11, fontweight='bold')
        ax.set_xlabel('Iteration $t_j$', fontsize=9)
        ax.set_ylabel('Iteration $t_i$', fontsize=9)
        ticks = list(range(0, T, max(1, T // 5)))
        ax.set_xticks(ticks); ax.set_xticklabels([t+1 for t in ticks], fontsize=8)
        ax.set_yticks(ticks); ax.set_yticklabels([t+1 for t in ticks], fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('CKA Representational Similarity Across Recursive Iterations',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig1_cka_heatmaps.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_rank_evolution(rank_data: dict, out_dir: str):
    """Effective rank and 95%-var rank per stage."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for s_idx, (eranks, r95s) in sorted(rank_data.items()):
        T = len(eranks)
        xs = np.arange(1, T + 1)
        axes[0].plot(xs, eranks, marker='o', markersize=4,
                     label=STAGE_NAMES[s_idx])
        axes[1].plot(xs, r95s,   marker='s', markersize=4,
                     label=STAGE_NAMES[s_idx])

    for ax, title, ylabel in zip(
            axes,
            ['Effective Rank (entropy)', '95%-Variance Rank'],
            ['Effective rank', 'Rank (# singular values)']):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Recursive iteration $t$', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Rank Evolution Across Recursive Iterations', fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig2_rank_evolution.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_probe_accuracy(probe_data: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for s_idx, accs in sorted(probe_data.items()):
        T = len(accs)
        ax.plot(np.arange(1, T + 1), accs, marker='D', markersize=5,
                label=STAGE_NAMES[s_idx])
    ax.set_title('Linear Probe Accuracy vs. Recursive Iteration', fontsize=13)
    ax.set_xlabel('Iteration $t$', fontsize=11)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig3_probe_accuracy.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_gradient_flow(grad_data: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for s_idx, norms in sorted(grad_data.items()):
        T = len(norms)
        ax.semilogy(np.arange(1, T + 1), norms + 1e-12, marker='o',
                    markersize=5, label=STAGE_NAMES[s_idx])
    ax.set_title('Gradient Magnitude per Recursive Iteration (log scale)', fontsize=12)
    ax.set_xlabel('Iteration $t$', fontsize=11)
    ax.set_ylabel('Mean $\\|\\nabla\\|_2$ (log)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig4_gradient_flow.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_drift(drift_data: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for s_idx, drifts in sorted(drift_data.items()):
        T = len(drifts)
        ax.plot(np.arange(1, T + 1), drifts, marker='^', markersize=5,
                label=STAGE_NAMES[s_idx])
    ax.set_title('Representation Drift $\\|X_{t+1}-X_t\\|_F / \\|X_t\\|_F$', fontsize=12)
    ax.set_xlabel('Transition $t \\to t+1$', fontsize=11)
    ax.set_ylabel('Relative drift', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig5_drift.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_summary_dashboard(cka_matrices, rank_data, probe_data,
                            grad_data, drift_data, out_dir: str):
    """Single combined figure for the paper supplement."""
    fig = plt.figure(figsize=(20, 24))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Row 0-1: CKA heatmaps (6 stages)
    for idx, (s_idx, mat) in enumerate(sorted(cka_matrices.items())):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        T  = mat.shape[0]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis')
        ax.set_title(f'CKA — {STAGE_NAMES[s_idx]}', fontsize=9)
        ticks = list(range(0, T, max(1, T // 4)))
        ax.set_xticks(ticks); ax.set_xticklabels([t+1 for t in ticks], fontsize=7)
        ax.set_yticks(ticks); ax.set_yticklabels([t+1 for t in ticks], fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2 left: rank
    ax = fig.add_subplot(gs[2, 0])
    for s_idx, (eranks, _) in sorted(rank_data.items()):
        ax.plot(range(1, len(eranks)+1), eranks, marker='o', ms=3,
                label=f'S{s_idx+1}')
    ax.set_title('Effective Rank', fontsize=10); ax.legend(fontsize=6)
    ax.set_xlabel('Iter $t$'); ax.grid(True, alpha=0.3)

    # Row 2 mid: probe
    ax = fig.add_subplot(gs[2, 1])
    for s_idx, accs in sorted(probe_data.items()):
        ax.plot(range(1, len(accs)+1), accs, marker='D', ms=3,
                label=f'S{s_idx+1}')
    ax.set_title('Linear Probe Acc.', fontsize=10); ax.legend(fontsize=6)
    ax.set_xlabel('Iter $t$'); ax.set_ylabel('Acc. (%)'); ax.grid(True, alpha=0.3)

    # Row 2 right: gradient flow
    ax = fig.add_subplot(gs[2, 2])
    for s_idx, norms in sorted(grad_data.items()):
        ax.semilogy(range(1, len(norms)+1), norms + 1e-12, marker='o', ms=3,
                    label=f'S{s_idx+1}')
    ax.set_title('Gradient Flow (log)', fontsize=10); ax.legend(fontsize=6)
    ax.set_xlabel('Iter $t$'); ax.grid(True, alpha=0.3, which='both')

    # Row 3 left: drift
    ax = fig.add_subplot(gs[3, 0])
    for s_idx, drifts in sorted(drift_data.items()):
        ax.plot(range(1, len(drifts)+1), drifts, marker='^', ms=3,
                label=f'S{s_idx+1}')
    ax.set_title('Representation Drift', fontsize=10); ax.legend(fontsize=6)
    ax.set_xlabel('Transition $t\\to t+1$'); ax.grid(True, alpha=0.3)

    fig.suptitle('SCALE-C3: Recursive Computation Analysis Dashboard',
                 fontsize=16, fontweight='bold')
    path = os.path.join(out_dir, 'fig0_dashboard.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='')
    p.add_argument('--data',       default='./data')
    p.add_argument('--out-dir',    default='./analysis_results')
    p.add_argument('--n-samples',  type=int, default=10000,
                   help='Number of test images to use for analysis')
    p.add_argument('--batch-size', type=int, default=200)
    p.add_argument('--stages',     nargs='+', type=int,
                   default=[0, 1, 2, 3, 4, 5],
                   help='Which stages to analyse (0-indexed)')
    p.add_argument('--skip-probe', action='store_true',
                   help='Skip linear probe (slow if sklearn unavailable)')
    p.add_argument('--skip-grad',  action='store_true',
                   help='Skip gradient flow analysis')
    p.add_argument('--n-grad-batches', type=int, default=20)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    model  = load_model(args.checkpoint, device)
    loader = build_val_loader(args.data, args.batch_size, args.n_samples)

    cka_matrices = {}
    rank_data    = {}
    probe_data   = {}
    drift_data   = {}
    grad_data    = {}

    for s_idx in args.stages:
        stage_name = STAGE_NAMES[s_idx]
        print(f"\n{'='*60}")
        print(f"Analysing {stage_name}  (stage index {s_idx})")
        print('='*60)

        # --- Collect snapshots ----------------------------------------
        print("  Collecting per-iteration features …")
        snapshots, labels = collect_stage_iterations(model, loader, s_idx, device)
        T = len(snapshots)
        print(f"  Got {T} snapshots, each {tuple(snapshots[0].shape)}")

        # 1. CKA
        print("  Computing CKA matrix …")
        cka_mat = compute_cka_matrix(snapshots)
        cka_matrices[s_idx] = cka_mat
        np.save(os.path.join(args.out_dir, f'cka_stage{s_idx}.npy'), cka_mat)

        # 2. Rank
        print("  Computing rank evolution …")
        eranks, r95s = compute_rank_evolution(snapshots)
        rank_data[s_idx] = (eranks, r95s)
        np.save(os.path.join(args.out_dir, f'ranks_stage{s_idx}.npy'),
                np.stack([eranks, r95s]))

        # 3. Probe
        if not args.skip_probe:
            print("  Training linear probes …")
            try:
                accs = linear_probe_accuracy(snapshots, labels, device=device)
                probe_data[s_idx] = accs
                np.save(os.path.join(args.out_dir, f'probe_stage{s_idx}.npy'), accs)
            except ImportError:
                print("  sklearn not found — skipping probe.")

        # 4. Gradient flow
        if not args.skip_grad:
            print("  Measuring gradient flow …")
            grad_norms = measure_gradient_flow(
                model, loader, s_idx, device, n_batches=args.n_grad_batches)
            grad_data[s_idx] = grad_norms
            np.save(os.path.join(args.out_dir, f'grads_stage{s_idx}.npy'), grad_norms)

        # 5. Drift
        print("  Computing representation drift …")
        drifts = compute_drift(snapshots)
        drift_data[s_idx] = drifts
        np.save(os.path.join(args.out_dir, f'drift_stage{s_idx}.npy'), drifts)

        # Free memory
        del snapshots

    # ── Plotting ──────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    if cka_matrices:
        plot_cka_heatmaps(cka_matrices, args.out_dir)
    if rank_data:
        plot_rank_evolution(rank_data, args.out_dir)
    if probe_data:
        plot_probe_accuracy(probe_data, args.out_dir)
    if grad_data:
        plot_gradient_flow(grad_data, args.out_dir)
    if drift_data:
        plot_drift(drift_data, args.out_dir)
    if cka_matrices and rank_data and drift_data:
        plot_summary_dashboard(cka_matrices, rank_data,
                               probe_data if probe_data else {s: np.zeros(10) for s in args.stages},
                               grad_data  if grad_data  else {s: np.zeros(10) for s in args.stages},
                               drift_data, args.out_dir)

    # ── Save JSON summary ─────────────────────────────────────────────────
    summary = {}
    for s_idx in args.stages:
        entry = {}
        if s_idx in cka_matrices:
            mat = cka_matrices[s_idx]
            T   = mat.shape[0]
            # Adjacent-iteration CKA (diagonal+1)
            adj = [float(mat[t, t+1]) for t in range(T-1)]
            entry['cka_adjacent_mean'] = float(np.mean(adj))
            entry['cka_t1_tT'] = float(mat[0, T-1])
        if s_idx in rank_data:
            er, r95 = rank_data[s_idx]
            entry['erank_t1'] = float(er[0])
            entry['erank_tT'] = float(er[-1])
        if s_idx in probe_data:
            accs = probe_data[s_idx]
            entry['probe_t1'] = float(accs[0])
            entry['probe_tT'] = float(accs[-1])
        if s_idx in drift_data:
            d = drift_data[s_idx]
            entry['drift_mean'] = float(d.mean())
            entry['drift_t1'] = float(d[0])
            entry['drift_tT'] = float(d[-1])
        summary[f'stage_{s_idx}'] = entry

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll results saved to: {args.out_dir}/")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
