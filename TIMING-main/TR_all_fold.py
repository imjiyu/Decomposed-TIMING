import pandas as pd
import numpy as np
from pathlib import Path

in_path  = Path("results_new/full_eval.csv")
out_path = Path("results_table/result_CPD_fold.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(in_path)

def extract_tr(m):
    if "trend"    in m: return "Trend"
    if "residual" in m: return "Residual"
    return "Unknown"

df["TR"] = df["method"].apply(extract_tr)

# CPD 행만, cum_diff 컬럼만 추출 → pivot으로 Trend/Residual 나란히
cpd = (
    df[df["metric"] == "CPD"][["data", "fold", "TR", "cum_diff"]]
    .pivot_table(index=["data", "fold"], columns="TR", values="cum_diff")
    .rename(columns={"Trend": "CPD_T", "Residual": "CPD_R"})
    .reset_index()
)
cpd["fold"] = pd.to_numeric(cpd["fold"], errors="raise").astype(int)
cpd["ΔCPD"]       = cpd["CPD_T"] - cpd["CPD_R"]
cpd["TrendRatio"] = cpd["CPD_T"] / (cpd["CPD_T"] + cpd["CPD_R"])
cpd["ResidualRatio"] = cpd["CPD_R"] / (cpd["CPD_T"] + cpd["CPD_R"])

# ── fold별 상세 블록 ──────────────────────────────────────────────────
dataset_order = df["data"].unique().tolist()
cpd["data"] = pd.Categorical(cpd["data"], categories=dataset_order, ordered=True)
cpd = cpd.sort_values(["data", "fold"]).reset_index(drop=True)

# ── 데이터셋별 ΔCPD mean / std 요약 행 ───────────────────────────────
def summary_rows(grp):
    name = grp.name
    rows = []
    for stat, fn in [
        ("mean", lambda x: np.mean(x)),
        ("std",  lambda x: np.std(x, ddof=1)),
        ]:
        rows.append({
            "data":        name,
            "fold":        stat,          # fold 컬럼에 'mean'/'std' 표시하기
            "CPD_T":       fn(grp["CPD_T"]),
            "CPD_R":       fn(grp["CPD_R"]),
            "ΔCPD":        fn(grp["ΔCPD"]),
            "TrendRatio":  fn(grp["TrendRatio"]),
            "ResidualRatio": fn(grp["ResidualRatio"]),
        })
    return pd.DataFrame(rows)

summary = (
    cpd.groupby("data", observed=True)
       .apply(summary_rows, include_groups=False)
       .reset_index(drop=True)
)
summary["data"] = pd.Categorical(summary["data"], categories=dataset_order, ordered=True)

# ── fold 상세 + 요약 블록을 데이터셋 단위로 합치기 ───────────────────
blocks = []
for ds in dataset_order:
    detail = cpd[cpd["data"] == ds].sort_values("fold")
    summ    = summary[summary["data"] == ds]
    spacer  = pd.DataFrame([{c: "" for c in cpd.columns}])   ### 빈 구분행 추가
    blocks.extend([detail, summ, spacer])

result = pd.concat(blocks, ignore_index=True)

# ── 저장 (raw) ────────────────────────────────────────────────────────
result.to_csv(out_path, index=False, float_format="%.6g")
print(f"Saved → {out_path}")
print(result.to_string(index=False))

# 소수점 3자리 반올림 ver.
num_cols = ["CPD_T", "CPD_R", "ΔCPD", "TrendRatio", "ResidualRatio"]
rounded  = result.copy()
for c in num_cols:
    rounded[c] = pd.to_numeric(rounded[c], errors="coerce").round(3)

round_out = out_path.with_name("result_CPD_fold_r3.csv")
rounded.to_csv(round_out, index=False)
print(f"Saved rounded → {round_out}")