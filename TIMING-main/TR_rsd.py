import pandas as pd
from pathlib import Path

in_path = Path("results_residual_first/full_eval.csv")

fold_raw_out_path = Path("results_table_rsd/result_TR_fold_raw.csv")
raw_out_path = Path("results_table_rsd/result_TR_raw.csv")
round_out_path = Path("results_table_rsd/result_TR.csv")

raw_out_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(in_path)

# 혹시 기존 결과가 섞여 있어도 CPD만 사용
df = df[df["metric"] == "CPD"].copy()

# append 중복 방지: 같은 data/fold/seed/method/metric이면 마지막 것만 사용
df = df.drop_duplicates(
    subset=["data", "fold", "seed", "method", "metric"],
    keep="last"
)

def extract_tr(method_str):
    if "trend" in method_str:
        return "Trend"
    elif "residual" in method_str:
        return "Residual"
    return "Unknown"

df["TR"] = df["method"].apply(extract_tr)

# 비교할 컬럼
metric_cols = [
    "cum_diff",          # CPD
    "AUCC",
    "cum_50_diff",      # 50% CPD
    "accuracy",
    "comprehensiveness",
    "cross_entropy",
    "log_odds",
    "sufficiency",
]

# 보기 좋은 이름으로 변경
rename_cols = {
    "cum_diff": "CPD",
    "cum_50_diff": "CPD_50",
    "accuracy": "Accuracy",
    "comprehensiveness": "Comprehensiveness",
    "cross_entropy": "CrossEntropy",
    "log_odds": "LogOdds",
    "sufficiency": "Sufficiency",
}

# 숫자 변환
for c in metric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------
# 1) fold별 raw 결과 저장
# -----------------------------
fold_raw = df[
    ["data", "fold", "seed", "TR", "method"] + metric_cols
].copy()

fold_raw = fold_raw.rename(columns=rename_cols)

# 정렬
dataset_order = df["data"].drop_duplicates().tolist()
fold_raw["data"] = pd.Categorical(
    fold_raw["data"],
    categories=dataset_order,
    ordered=True
)
fold_raw["TR"] = pd.Categorical(
    fold_raw["TR"],
    categories=["Trend", "Residual"],
    ordered=True
)

fold_raw = fold_raw.sort_values(
    ["data", "fold", "TR"]
).reset_index(drop=True)

fold_raw.to_csv(fold_raw_out_path, index=False)
print(f"Saved fold raw → {fold_raw_out_path}")


# -----------------------------
# 2) data, TR 기준 mean/std 집계
# -----------------------------
grp = df.groupby(["data", "TR"])[metric_cols]

mean_df = grp.mean().rename(columns=rename_cols)
std_df = grp.std(ddof=1).rename(columns=rename_cols)

mean_df["stat"] = "mean"
std_df["stat"] = "std"

combined = pd.concat([mean_df, std_df]).reset_index()

# 정렬: data별로 Trend mean, Residual mean, Trend std, Residual std
combined["data"] = pd.Categorical(
    combined["data"],
    categories=dataset_order,
    ordered=True
)
combined["TR"] = pd.Categorical(
    combined["TR"],
    categories=["Trend", "Residual"],
    ordered=True
)
combined["stat"] = pd.Categorical(
    combined["stat"],
    categories=["mean", "std"],
    ordered=True
)

combined = combined.sort_values(
    ["data", "stat", "TR"]
).reset_index(drop=True)

out_cols = [
    "data",
    "TR",
    "stat",
    "CPD",
    "CPD_50",
    "AUCC",
    "Accuracy",
    "Comprehensiveness",
    "CrossEntropy",
    "LogOdds",
    "Sufficiency",
]

combined = combined[out_cols]

# raw 저장
combined.to_csv(raw_out_path, index=False)
print(f"Saved raw summary → {raw_out_path}")

# 소수점 3자리 반올림 저장
numeric_cols = [c for c in combined.columns if c not in ("data", "TR", "stat")]
rounded = combined.copy()
rounded[numeric_cols] = rounded[numeric_cols].round(3)

rounded.to_csv(round_out_path, index=False)
print(f"Saved rounded summary → {round_out_path}")

print("\n=== Fold raw preview ===")
print(fold_raw.to_string(index=False))

print("\n=== Summary preview rounded ===")
print(rounded.to_string(index=False))