from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# 1. 경로 설정
# =========================

INPUT_CSV = Path("results_TRC_random/results_random_all_with_header.csv")

OUT_DIR = Path("results_TRC_random/random_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2. CSV 읽기
# =========================

df = pd.read_csv(INPUT_CSV)

# =========================
# 3. 사용할 metric 컬럼
# =========================

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
# 현재 CSV에 dataset 컬럼이 없으므로 행 순서 기반으로 부여
# 순서: B, E, F, P, W

dataset_order = ["boiler", "epilepsy", "freezer", "PAM", "wafer"]

n_dataset = len(dataset_order)
n_per_dataset = len(df) // n_dataset

assert len(df) % n_dataset == 0, "전체 행 수가 데이터셋 개수로 나누어떨어지지 않습니다."
assert n_per_dataset == 25, f"데이터셋당 25행이어야 하는데 현재 {n_per_dataset}행입니다."

df["dataset"] = np.repeat(dataset_order, n_per_dataset)

# =========================
# 5. 계산용 컬럼만 남기기
# =========================

df_clean = df[["dataset", "fold"] + metric_cols].copy()

# =========================
# 6. dataset-fold별 random 5회 평균/표준편차
# =========================
# fold_summary:
# 각 dataset의 각 fold 안에서 random seed 5회에 대한 mean/std

fold_summary = df_clean.groupby(["dataset", "fold"])[metric_cols].agg(["mean", "std"])

# fold_mean:
# 최종 비교에 사용할 fold-level random 결과
# dataset당 5행이 됨

fold_mean = df_clean.groupby(["dataset", "fold"])[metric_cols].mean().reset_index()

# =========================
# 7. 최종 dataset별 평균/표준편차
# =========================
# 여기서 std는 25개 row 전체 std가 아니라
# fold별 random 평균 5개에 대한 std

final_summary = fold_mean.groupby("dataset")[metric_cols].agg(["mean", "std"])

# =========================
# 8. 보기 좋은 mean ± std 표
# =========================

pretty = pd.DataFrame(index=final_summary.index)

for col in metric_cols:
    mean = final_summary[(col, "mean")]
    std = final_summary[(col, "std")]
    pretty[col] = mean.map(lambda x: f"{x:.4f}") + " ± " + std.map(lambda x: f"{x:.4f}")

pretty = pretty.reset_index()

# =========================
# 9. 저장
# =========================

df_clean.to_csv(
    OUT_DIR / "raw_clean.csv",
    index=False,
    encoding="utf-8-sig"
)

fold_mean.to_csv(
    OUT_DIR / "fold_random_mean_only.csv",
    index=False,
    encoding="utf-8-sig"
)

fold_summary.to_csv(
    OUT_DIR / "fold_random_mean_std.csv",
    encoding="utf-8-sig"
)

final_summary.to_csv(
    OUT_DIR / "dataset_summary_numeric.csv",
    encoding="utf-8-sig"
)

pretty.to_csv(
    OUT_DIR / "dataset_summary_pretty.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== fold-level random mean ===")
print(fold_mean)

print("\n=== final dataset summary ===")
print(pretty)

print("\n저장 완료:")
print(OUT_DIR / "raw_clean.csv")
print(OUT_DIR / "fold_random_mean_only.csv")
print(OUT_DIR / "fold_random_mean_std.csv")
print(OUT_DIR / "dataset_summary_numeric.csv")
print(OUT_DIR / "dataset_summary_pretty.csv")