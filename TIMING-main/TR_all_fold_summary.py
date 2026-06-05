import pandas as pd
import numpy as np
from pathlib import Path

#in_path  = Path("results_new/full_eval.csv")
#out_path = Path("results_table/result_CPD_summary.csv")
in_path  = Path("results_residual_first/full_eval.csv")
out_path = Path("results_table_rsd/result_CPD_summary.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(in_path)

def extract_tr(m):
    if "trend"    in m: return "Trend"
    if "residual" in m: return "Residual"
    return "Unknown"

df["TR"] = df["method"].apply(extract_tr)

cpd = (
    df[df["metric"] == "CPD"][["data", "fold", "TR", "cum_diff"]]
    .pivot_table(index=["data", "fold"], columns="TR", values="cum_diff")
    .rename(columns={"Trend": "CPD_T", "Residual": "CPD_R"})
    .reset_index()
)
cpd["fold"]       = pd.to_numeric(cpd["fold"], errors="raise").astype(int)
cpd["ΔCPD"]       = cpd["CPD_T"] - cpd["CPD_R"]
cpd["TrendRatio"] = cpd["CPD_T"] / (cpd["CPD_T"] + cpd["CPD_R"])
cpd["ResidualRatio"] = cpd["CPD_R"] / (cpd["CPD_T"] + cpd["CPD_R"])

dataset_order = df["data"].unique().tolist()
cpd["data"] = pd.Categorical(cpd["data"], categories=dataset_order, ordered=True)
cpd = cpd.sort_values(["data", "fold"]).reset_index(drop=True)

n_folds = cpd.groupby("data")["fold"].count()   # 데이터셋별 fold 수

# ── Dominance 판정 기준 (필요시 조정...!) ────────────────────────────────
# TrendRatio 기반 & Fold count 기반(fold_trend_count / total 도 함께 고려)
def dominance_label(ratio_mean: float, trend_count: int, total: int) -> str:
    frac = trend_count / total

    # 1. 강한 Trend 우세
    if ratio_mean >= 0.60 and frac >= 0.8:
        return "trend-dominant"
    elif ratio_mean >= 0.55 and frac >= 0.6:
        return "trend-favored"

    # 2. 강한 Residual 우세
    elif ratio_mean <= 0.40 and frac <= 0.2:
        return "residual-dominant"
    elif ratio_mean <= 0.45 and frac <= 0.4:
        return "residual-favored"

    # 3. 0.5 근처지만 방향성이 약간 있는 경우 <- boiler 경향 잡기
    elif 0.45 <= ratio_mean <= 0.55:
        if frac > 0.5:
            return "mixed / weak trend"
        elif frac < 0.5:
            return "mixed / weak residual"
        else:
            return "mixed"

    else:
        return "mixed"

# ── 데이터셋별 요약 ────────────────────────────────────────────────────
rows = []
for ds in dataset_order:
    g = cpd[cpd["data"] == ds]
    total = len(g)

    m_t  = g["CPD_T"].mean();       s_t  = g["CPD_T"].std(ddof=1)
    m_r  = g["CPD_R"].mean();       s_r  = g["CPD_R"].std(ddof=1)
    m_d  = g["ΔCPD"].mean();        s_d  = g["ΔCPD"].std(ddof=1)
    m_tr = g["TrendRatio"].mean();  s_tr = g["TrendRatio"].std(ddof=1)
    m_rr = g["ResidualRatio"].mean();  s_rr = g["ResidualRatio"].std(ddof=1)

    trend_wins = int((g["CPD_T"] > g["CPD_R"]).sum())
    resid_wins = total - trend_wins

    if trend_wins >= resid_wins:
        fold_dir = f"{trend_wins}/{total} Trend"
    else:
        fold_dir = f"{resid_wins}/{total} Residual"

    dom = dominance_label(m_tr, trend_wins, total)

    rows.append({
        "Dataset":       ds,
        "CPD_T":         f"{m_t:.3f}±{s_t:.3f}",
        "CPD_R":         f"{m_r:.3f}±{s_r:.3f}",
        "ΔCPD":          f"{m_d:+.3f}±{s_d:.3f}",
        "TrendRatio":    f"{m_tr:.3f}±{s_tr:.3f}",
        "ResidualRatio": f"{m_rr:.3f}±{s_rr:.3f}",
        "FoldDirection": fold_dir,
        "Dominance":     dom,
    })

result = pd.DataFrame(rows)

result.to_csv(out_path, index=False)
print(f"Saved → {out_path}")
print(result.to_string(index=False))