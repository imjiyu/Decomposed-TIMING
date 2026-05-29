import pandas as pd
from pathlib import Path

method_path = Path("results_TRC/method_summary/dataset_method_summary_pretty.csv")
random_path = Path("results_TRC_random/random_summary/dataset_summary_pretty.csv")

out_path = Path("results_TRC/all_final_comparison_summary.csv")

method_df = pd.read_csv(method_path)
random_df = pd.read_csv(random_path)

# 컬럼 이름 맞추기
# method_df는 method_group 컬럼이 있을 가능성이 큼
if "method_group" in method_df.columns:
    method_df = method_df.rename(columns={"method_group": "method"})

# random 표에 method 컬럼 추가
random_df.insert(1, "method", "Random")

# 최종 결합
final_df = pd.concat([method_df, random_df], ignore_index=True)

# 정렬 순서 고정
dataset_order = ["boiler", "epilepsy", "freezer", "PAM", "wafer"]
method_order = ["Trend", "Residual", "Combined(|T+R|)", "Random"]

final_df["dataset"] = pd.Categorical(
    final_df["dataset"],
    categories=dataset_order,
    ordered=True
)

final_df["method"] = pd.Categorical(
    final_df["method"],
    categories=method_order,
    ordered=True
)

final_df = final_df.sort_values(["dataset", "method"])

final_df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(final_df)
print(f"saved: {out_path}")
