import pandas as pd
import numpy as np
from pathlib import Path

in_path = Path("results_new/full_eval.csv")
raw_out_path = Path("results_table/result_TR_raw.csv")
round_out_path = Path("results_table/result_TR.csv")

raw_out_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(in_path)

def extract_tr(method_str):
    if "trend" in method_str:
        return "Trend"
    elif "residual" in method_str:
        return "Residual"
    return "Unknown"

df["TR"] = df["method"].apply(extract_tr)

cpd = df[df["metric"] == "CPD"].copy()
cpp = df[df["metric"] == "CPP"].copy()

# CPD/CPP 전용 컬럼! (data, TR) 기준 집계
cpd_tr_cols = ["cum_diff", "AUCC", "cum_50_diff"]
cpp_tr_cols = ["cum_diff", "AUCC", "cum_50_diff"]

def agg_tr(data_df, cols, prefix):
    grp = data_df.groupby(["data", "TR"])[cols]
    mean_df = grp.mean().rename(columns={c: f"{prefix}_{c}" for c in cols})
    std_df  = grp.std(ddof=1).rename(columns={c: f"{prefix}_{c}" for c in cols})
    mean_df["stat"] = "mean"
    std_df["stat"]  = "std"
    return pd.concat([mean_df, std_df]).reset_index()

cpd_tr_agg = agg_tr(cpd, cpd_tr_cols, "CPD")
cpp_tr_agg = agg_tr(cpp, cpp_tr_cols, "CPP")

# accuracy 등 5개 컬럼: (data, TR) 기준으로 각각 집계
shared_cols = ["accuracy", "comprehensiveness", "cross_entropy", "log_odds", "sufficiency"]

grp_shared = cpd.groupby(["data", "TR"])[shared_cols]
shared_mean = grp_shared.mean()
shared_std  = grp_shared.std(ddof=1)
shared_mean["stat"] = "mean"
shared_std["stat"]  = "std"
shared_agg = pd.concat([shared_mean, shared_std]).reset_index()

# 합치기
merged_tr = cpd_tr_agg.merge(cpp_tr_agg, on=["data", "TR", "stat"], how="outer")
combined  = merged_tr.merge(shared_agg,  on=["data", "TR", "stat"], how="left")

# 정렬: 데이터셋 원본 순서, TR: Trend→Residual, stat: mean→std
dataset_order = df["data"].unique().tolist()
combined["data"] = pd.Categorical(combined["data"], categories=dataset_order, ordered=True)
combined["TR"]   = pd.Categorical(combined["TR"],   categories=["Trend", "Residual"], ordered=True)
combined["stat"] = pd.Categorical(combined["stat"], categories=["mean", "std"], ordered=True)

# 순서: 같은 데이터셋 내에서 Trend mean, Residual mean, Trend std, Residual std
combined = combined.sort_values(["data", "stat", "TR"]).reset_index(drop=True)

# 출력 컬럼 순서
out_cols = (
    ["data", "TR", "stat"]
    + [f"CPD_{c}" for c in cpd_tr_cols]
    + shared_cols
    + [f"CPP_{c}" for c in cpp_tr_cols]
)
combined = combined[out_cols]

# raw 저장 따로 
combined.to_csv(raw_out_path, index=False)
print(f"Saved raw → {raw_out_path}")

# 소수점 3자리 반올림 ver. 
numeric_cols = [c for c in combined.columns if c not in ("data", "TR", "stat")]
rounded = combined.copy()
rounded[numeric_cols] = rounded[numeric_cols].round(3)
rounded.to_csv(round_out_path, index=False)
print(f"Saved rounded → {round_out_path}")

print("\n=== Preview (rounded) ===")
print(rounded.to_string(index=False))