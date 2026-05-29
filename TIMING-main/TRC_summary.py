from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# 1. 경로 설정
# =========================

INPUT_CSV = Path("results_TRC/results_td_eval_all_with_header.csv")

OUT_DIR = Path("results_TRC/method_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2. CSV 읽기
# =========================

df = pd.read_csv(INPUT_CSV)

# =========================
# 3. 사용할 metric 컬럼
# =========================
# lambda_3 이후의 모든 평가 metric

metric_cols = [
    "cum_50_diff",
    "cum_diff",
    "AUCC",
    "accuracy",
    "comprehensiveness",
    "cross_entropy",
    "log_odds",
    "sufficiency",
]

df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce")

# =========================
# 4. dataset 컬럼 추가
# =========================
# 데이터셋 순서: B, E, F, P, W
# 각 데이터셋마다 fold 0~4 × method 3개 = 15행

dataset_order = ["boiler", "epilepsy", "freezer", "PAM", "wafer"]

n_dataset = len(dataset_order)
n_method = 3
n_fold = 5
n_per_dataset = n_fold * n_method  # 15

assert len(df) == n_dataset * n_per_dataset, (
    f"행 수가 예상과 다릅니다. "
    f"현재 {len(df)}행, 예상 {n_dataset * n_per_dataset}행입니다."
)

df["dataset"] = np.repeat(dataset_order, n_per_dataset)

# =========================
# 5. method 이름 정리
# =========================

def parse_method(method_name: str) -> str:
    method_name = str(method_name).lower()

    if "trend" in method_name:
        return "Trend"
    elif "residual" in method_name:
        return "Residual"
    elif "combined" in method_name:
        return "Combined(|T+R|)"
    else:
        return "Unknown"

df["method_group"] = df["method"].apply(parse_method)

method_order = ["Trend", "Residual", "Combined(|T+R|)"]

# 순서 고정용 categorical
df["dataset"] = pd.Categorical(
    df["dataset"],
    categories=dataset_order,
    ordered=True
)

df["method_group"] = pd.Categorical(
    df["method_group"],
    categories=method_order,
    ordered=True
)

# =========================
# 6. 계산용 컬럼만 남기기
# =========================

df_clean = df[["dataset", "fold", "method_group"] + metric_cols].copy()

# =========================
# 7. 확인용 count 체크
# =========================
# 각 dataset-method마다 fold 5개가 있어야 정상

count_check = (
    df_clean
    .groupby(["dataset", "method_group"], observed=False, sort=False)
    .size()
    .reset_index(name="count")
)

print("\n=== count check ===")
print(count_check)

# =========================
# 8. dataset-method별 평균/표준편차
# =========================
# 여기서 mean/std는 fold 0~4, 즉 5개 값 기준

summary_numeric = (
    df_clean
    .groupby(["dataset", "method_group"], observed=False, sort=False)[metric_cols]
    .agg(["mean", "std"])
)

# =========================
# 9. 보기 좋은 mean ± std 표
# =========================

pretty = pd.DataFrame(index=summary_numeric.index)

for col in metric_cols:
    mean = summary_numeric[(col, "mean")]
    std = summary_numeric[(col, "std")]
    pretty[col] = mean.map(lambda x: f"{x:.4f}") + " ± " + std.map(lambda x: f"{x:.4f}")

pretty = pretty.reset_index()

# =========================
# 10. 숫자형 long table
# =========================

summary_long = []

for dataset in dataset_order:
    for method in method_order:
        row = {
            "dataset": dataset,
            "method": method,
        }

        for metric in metric_cols:
            row[f"{metric}_mean"] = summary_numeric.loc[(dataset, method), (metric, "mean")]
            row[f"{metric}_std"] = summary_numeric.loc[(dataset, method), (metric, "std")]

        summary_long.append(row)

summary_long = pd.DataFrame(summary_long)
summary_long = summary_long.round(4)

# =========================
# 11. 저장
# =========================

df_clean.to_csv(
    OUT_DIR / "raw_clean.csv",
    index=False,
    encoding="utf-8-sig"
)

count_check.to_csv(
    OUT_DIR / "count_check.csv",
    index=False,
    encoding="utf-8-sig"
)

summary_numeric.round(4).to_csv(
    OUT_DIR / "dataset_method_summary_numeric.csv",
    encoding="utf-8-sig"
)

summary_long.to_csv(
    OUT_DIR / "dataset_method_summary_numeric_long.csv",
    index=False,
    encoding="utf-8-sig"
)

pretty.to_csv(
    OUT_DIR / "dataset_method_summary_pretty.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== final pretty summary ===")
print(pretty)

print("\n저장 완료:")
print(OUT_DIR / "raw_clean.csv")
print(OUT_DIR / "count_check.csv")
print(OUT_DIR / "dataset_method_summary_numeric.csv")
print(OUT_DIR / "dataset_method_summary_numeric_long.csv")
print(OUT_DIR / "dataset_method_summary_pretty.csv")