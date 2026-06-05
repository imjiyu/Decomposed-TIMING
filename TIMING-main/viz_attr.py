"""
Visualize Trend vs Residual attribution heatmaps from saved .npy files.
fold 0~4 전체를 concat해서 데이터셋별로 heatmap 2개(abs/signed) 생성.

Usage:
    python viz_attribution.py --data all
    python viz_attribution.py --data PAM --n_samples 10 --n_channels 8
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pytorch_lightning import seed_everything

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.PAM import PAM
from datasets.boiler import Boiler
from datasets.epilepsy import Epilepsy
from datasets.wafer import Wafer
from datasets.freezer import Freezer

# ── 데이터셋 설정 ─────────────────────────────────────────────────────────────
CFG = {
    "PAM":      (PAM,      17, 8, 600, False),
    "boiler":   (Boiler,   20, 2,  36, False),
    "epilepsy": (Epilepsy,  1, 2, 178, False),
    "wafer":    (Wafer,     1, 2, 152, True),
    "freezer":  (Freezer,   1, 2, 301, True),
}

# (trend_abs, residual_abs, trend_signed, residual_signed)
NPY_PATTERNS = {
    "PAM": (
        "PAM_state_timing_td_trend_kalman_seg10_min10_max600_result_{fold}_42.npy",
        "PAM_state_timing_td_residual_kalman_seg10_min10_max600_result_{fold}_42.npy",
        "PAM_state_timing_td_trend_signed_kalman_seg10_min10_max600_result_{fold}_42.npy",
        "PAM_state_timing_td_residual_signed_kalman_seg10_min10_max600_result_{fold}_42.npy",
    ),
    "boiler": (
        "boiler_state_timing_td_trend_kalman_seg50_min1_max36_result_{fold}_42.npy",
        "boiler_state_timing_td_residual_kalman_seg50_min1_max36_result_{fold}_42.npy",
        "boiler_state_timing_td_trend_signed_kalman_seg50_min1_max36_result_{fold}_42.npy",
        "boiler_state_timing_td_residual_signed_kalman_seg50_min1_max36_result_{fold}_42.npy",
    ),
    "epilepsy": (
        "epilepsy_state_timing_td_trend_kalman_seg10_min10_max10_result_{fold}_42.npy",
        "epilepsy_state_timing_td_residual_kalman_seg10_min10_max10_result_{fold}_42.npy",
        "epilepsy_state_timing_td_trend_signed_kalman_seg10_min10_max10_result_{fold}_42.npy",
        "epilepsy_state_timing_td_residual_signed_kalman_seg10_min10_max10_result_{fold}_42.npy",
    ),
    "wafer": (
        "wafer_state_timing_td_trend_kalman_seg5_min10_max152_result_{fold}_42.npy",
        "wafer_state_timing_td_residual_kalman_seg5_min10_max152_result_{fold}_42.npy",
        "wafer_state_timing_td_trend_signed_kalman_seg5_min10_max152_result_{fold}_42.npy",
        "wafer_state_timing_td_residual_signed_kalman_seg5_min10_max152_result_{fold}_42.npy",
    ),
    "freezer": (
        "freezer_state_timing_td_trend_kalman_seg5_min10_max100_result_{fold}_42.npy",
        "freezer_state_timing_td_residual_kalman_seg5_min10_max100_result_{fold}_42.npy",
        "freezer_state_timing_td_trend_signed_kalman_seg5_min10_max100_result_{fold}_42.npy",
        "freezer_state_timing_td_residual_signed_kalman_seg5_min10_max100_result_{fold}_42.npy",
    ),
}


def build_datamodule(data, fold, seed):
    DM, _, _, _, needs_folds = CFG[data]
    if needs_folds:
        return DM(n_folds=5, fold=fold, seed=seed)
    return DM(fold=fold, seed=seed)


def plot_heatmap(x, attr_trend, attr_residual,
                 n_samples=5, n_channels=5,
                 save_path=None, tag="", signed=False):
    """
    x             : (N, T, C)
    attr_trend    : (N, T, C)  — 파일에서 그대로 로드한 값
    attr_residual : (N, T, C)
    signed=False  : abs 파일 → Reds colormap
    signed=True   : signed 파일 → RdBu_r colormap
    """
    N, T, C = x.shape
    n_samples  = min(n_samples, N)
    n_channels = min(n_channels, C)
    suffix = "signed" if signed else "abs"

    fig_w = max(14, T // 20)
    fig_h = n_samples * n_channels * 1.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(f"{tag}  —  Trend vs Residual Attribution ({suffix})",
                 fontsize=13, y=1.01)

    outer = gridspec.GridSpec(n_samples, 1, figure=fig, hspace=0.6)

    for s in range(n_samples):
        inner = gridspec.GridSpecFromSubplotSpec(
            n_channels, 3, subplot_spec=outer[s],
            wspace=0.05, hspace=0.15,
            width_ratios=[6, 6, 0.3],
        )
        for c in range(n_channels):
            ax_t  = fig.add_subplot(inner[c, 0])
            ax_r  = fig.add_subplot(inner[c, 1])
            ax_cb = fig.add_subplot(inner[c, 2])

            t_row = attr_trend[s, :, c]
            r_row = attr_residual[s, :, c]

            if signed:
                vmax = max(np.abs(t_row).max(), np.abs(r_row).max(), 1e-9)
                vmin, cmap = -vmax, "RdBu_r"
            else:
                vmin = 0.0
                vmax = max(t_row.max(), r_row.max(), 1e-9)
                cmap = "Greens"

            im = ax_t.imshow(t_row[np.newaxis, :], aspect="auto",
                             vmin=vmin, vmax=vmax, cmap=cmap)
            ax_r.imshow(r_row[np.newaxis, :], aspect="auto",
                        vmin=vmin, vmax=vmax, cmap=cmap)
            plt.colorbar(im, cax=ax_cb)

            for ax in (ax_t, ax_r):
                ax.set_yticks([])
                ax.set_xticks([])
            ax_t.set_ylabel(f"ch{c}", fontsize=7, rotation=0, labelpad=18)
            ax_r.set_ylabel(f"ch{c}", fontsize=7, rotation=0, labelpad=18)
            if c == 0:
                ax_t.set_title(f"Sample {s}  —  Trend",    fontsize=8)
                ax_r.set_title(f"Sample {s}  —  Residual", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  saved → {save_path}")
    plt.close()


def run_dataset(data, args):
    """fold 0~4 concat → abs / signed heatmap 각 1장 생성."""
    seed_everything(args.seed, workers=True)

    abs_trend_list, abs_resid_list = [], []
    sgn_trend_list, sgn_resid_list = [], []
    x_list = []

    pat_at, pat_ar, pat_st, pat_sr = NPY_PATTERNS[data]

    for fold in range(5):
        paths = {
            "abs_trend":    os.path.join(args.npy_dir, pat_at.format(fold=fold)),
            "abs_resid":    os.path.join(args.npy_dir, pat_ar.format(fold=fold)),
            "signed_trend": os.path.join(args.npy_dir, pat_st.format(fold=fold)),
            "signed_resid": os.path.join(args.npy_dir, pat_sr.format(fold=fold)),
        }
        missing = [k for k, v in paths.items() if not os.path.exists(v)]
        if missing:
            print(f"  [skip fold {fold}] missing: {missing}")
            continue

        abs_trend_list.append(np.load(paths["abs_trend"]))
        abs_resid_list.append(np.load(paths["abs_resid"]))
        sgn_trend_list.append(np.load(paths["signed_trend"]))
        sgn_resid_list.append(np.load(paths["signed_resid"]))

        dm = build_datamodule(data, fold, args.seed)
        _ = dm.preprocess(split="train")
        x_list.append(dm.preprocess(split="test")["x"].numpy())

    if not x_list:
        print(f"  [skip {data}] no files found")
        return

    x_test        = np.concatenate(x_list,        axis=0)
    abs_trend     = np.concatenate(abs_trend_list, axis=0)
    abs_resid     = np.concatenate(abs_resid_list, axis=0)
    signed_trend  = np.concatenate(sgn_trend_list, axis=0)
    signed_resid  = np.concatenate(sgn_resid_list, axis=0)

    print(f"  {data}: N={x_test.shape[0]}, T={x_test.shape[1]}, C={x_test.shape[2]}")

    viz_dir = os.path.join(args.viz_dir, data)
    os.makedirs(viz_dir, exist_ok=True)

    # abs heatmap
    plot_heatmap(x_test, abs_trend, abs_resid,
                 n_samples=args.n_samples, n_channels=args.n_channels,
                 save_path=os.path.join(viz_dir, "heatmap_abs.png"),
                 tag=f"{data} (all folds)", signed=False)

    # signed heatmap
    plot_heatmap(x_test, signed_trend, signed_resid,
                 n_samples=args.n_samples, n_channels=args.n_channels,
                 save_path=os.path.join(viz_dir, "heatmap_signed.png"),
                 tag=f"{data} (all folds)", signed=True)

    print(f"[done] {data} → {viz_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="all",
                   choices=list(CFG) + ["all"])
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--npy_dir",    default="results_new")
    p.add_argument("--viz_dir",    default="viz_out")
    p.add_argument("--n_samples",  type=int, default=5)
    p.add_argument("--n_channels", type=int, default=20)
    args = p.parse_args()

    datasets = list(CFG) if args.data == "all" else [args.data]
    for data in datasets:
        print(f"\n── {data} ──")
        run_dataset(data, args)


if __name__ == "__main__":
    main()