"""
전체 샘플 평균 attribution heatmap 생성 (데이터셋별).
fold 0~4 concat → 샘플 축 평균 → 채널별 1 row heatmap.

Usage:
    python viz_mean.py --data all
    python viz_mean.py --data PAM
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

CFG = {
    "PAM":      (PAM,      17, 8, 600, False),
    "boiler":   (Boiler,   20, 2,  36, False),
    "epilepsy": (Epilepsy,  1, 2, 178, False),
    "wafer":    (Wafer,     1, 2, 152, True),
    "freezer":  (Freezer,   1, 2, 301, True),
}
### boiler 잠시 rsd 로 바꿈! 
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


def plot_mean_heatmap(mean_trend, mean_resid, n_channels,
                      save_path, tag, signed):
    """
    mean_trend / mean_resid : (T, C) — 샘플 평균
    채널별로 row 1개씩, Trend / Residual 나란히 표시
    """
    T, C = mean_trend.shape
    n_channels = min(n_channels, C)
    suffix = "signed" if signed else "abs"
    cmap   = "RdBu_r" if signed else "Greens"

    fig_w = max(14, T // 20)
    fig_h = n_channels * 0.8 + 1.5

    fig, axes = plt.subplots(
        n_channels, 3,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [6, 6, 0.3], "wspace": 0.05, "hspace": 0.3},
    )
    if n_channels == 1:
        axes = [axes]   # 채널이 1개일 때 shape 통일

    fig.suptitle(f"{tag} - Mean Attribution ({suffix})", fontsize=13, y=0.98)

    if signed:
        vmax_global = max(
            np.abs(mean_trend[:, :n_channels]).max(),
            np.abs(mean_resid[:, :n_channels]).max(),
            1e-9,
        )
        vmin_global = -vmax_global
    else:
        vmin_global = 0.0
        vmax_global = max(
            mean_trend[:, :n_channels].max(),
            mean_resid[:, :n_channels].max(),
            1e-9,
        )

    for c in range(n_channels):
        ax_t, ax_r, ax_cb = axes[c]

        t_row = mean_trend[:, c]
        r_row = mean_resid[:, c]

        im = ax_t.imshow(t_row[np.newaxis, :], aspect="auto",
                        vmin=vmin_global, vmax=vmax_global, cmap=cmap)

        ax_r.imshow(r_row[np.newaxis, :], aspect="auto",
                    vmin=vmin_global, vmax=vmax_global, cmap=cmap)
        plt.colorbar(im, cax=ax_cb)

        for ax in (ax_t, ax_r):
            ax.set_yticks([])
            ax.set_xticks([])
        ax_t.set_ylabel(f"ch{c}", fontsize=8, rotation=0, labelpad=18)

        if c == 0:
            ax_t.set_title("Trend",    fontsize=9)
            ax_r.set_title("Residual", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"  saved → {save_path}")
    plt.close()


def run_dataset(data, args):
    seed_everything(args.seed, workers=True)

    pat_at, pat_ar, pat_st, pat_sr = NPY_PATTERNS[data]
    abs_trend_list, abs_resid_list = [], []
    sgn_trend_list, sgn_resid_list = [], []

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

    if not abs_trend_list:
        print(f"  [skip {data}] no files found")
        return

    # fold concat → 샘플 축 평균 → (T, C)
    mean_abs_trend    = np.concatenate(abs_trend_list, axis=0).mean(axis=0)
    mean_abs_resid    = np.concatenate(abs_resid_list, axis=0).mean(axis=0)
    mean_signed_trend = np.concatenate(sgn_trend_list, axis=0).mean(axis=0)
    mean_signed_resid = np.concatenate(sgn_resid_list, axis=0).mean(axis=0)

    viz_dir = os.path.join(args.viz_dir, data)
    os.makedirs(viz_dir, exist_ok=True)

    plot_mean_heatmap(mean_abs_trend, mean_abs_resid,
                      n_channels=args.n_channels,
                      save_path=os.path.join(viz_dir, "heatmap_mean_abs.png"),
                      tag=f"{data} (all folds)",
                      signed=False)

    plot_mean_heatmap(mean_signed_trend, mean_signed_resid,
                      n_channels=args.n_channels,
                      save_path=os.path.join(viz_dir, "heatmap_mean_signed.png"),
                      tag=f"{data} (all folds)",
                      signed=True)

    print(f"[done] {data} → {viz_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="all",
                   choices=list(CFG) + ["all"])
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--npy_dir",    default="results_new")
    p.add_argument("--viz_dir",    default="viz_out")
    p.add_argument("--n_channels", type=int, default=20, ### 최대 보일러가 20이라서! 
                   help="표시할 채널 수 (전체 보려면 크게 설정)")
    args = p.parse_args()

    datasets = list(CFG) if args.data == "all" else [args.data]
    for data in datasets:
        print(f"\n── {data} ──")
        run_dataset(data, args)


if __name__ == "__main__":
    main()