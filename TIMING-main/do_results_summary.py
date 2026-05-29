import pandas as pd
import glob

# CSV 컬럼 헤더
cols = [
    'seed', 'fold', 'eval_baseline', 'topk', 'explainer', 'l1', 'l2', 'l3',
    'cum_50_diff', 'cum_diff', 'AUCC', 'acc', 'comp', 'ce', 'lodds', 'suff'
]

# metric들
metrics = [
    'cum_50_diff', 'cum_diff', 'AUCC', 'acc',
    'comp', 'ce', 'lodds', 'suff'
]

# 데이터셋 이름만 바꾸기!
what = "epilepsy" #  wafer / freezer / boiler / epilepsy / PAM
file_groups = {
    'timing': f'state_{what}_*_results_ht.csv',
    'td_permask_spline': f'zip_B_results/B_{what}_*_results_td_permask.csv',
    'td_permask_kalman': f'zip_K_results/K_{what}_*_results_td_kalman.csv',
    'baseline': f'state_{what}_*_results_baseline.csv',
}

dfs = []

for source, pattern in file_groups.items():
    matched_files = glob.glob(pattern)

    print(f"[{source}] {len(matched_files)} files")
    for f in matched_files:
        print("  ", f)

        df = pd.read_csv(f, header=None, names=cols)
        df["source"] = source
        df["file"] = f
        dfs.append(df)

if len(dfs) == 0:
    raise RuntimeError("No CSV files found. Check file paths or glob patterns.")

df_all = pd.concat(dfs, ignore_index=True)

print("\nrows by source:")
print(df_all.groupby("source").size())

# mean 집계
mean_summary = df_all.groupby(
    ['source', 'explainer', 'eval_baseline', 'topk']
)[metrics].mean().reset_index()

# std 집계
std_summary = df_all.groupby(
    ['source', 'explainer', 'eval_baseline', 'topk']
)[metrics].std().reset_index()

# 컬럼명 보기 좋게 변경
mean_summary = mean_summary.rename(
    columns={m: f"{m}_mean" for m in metrics}
)

std_summary = std_summary.rename(
    columns={m: f"{m}_std" for m in metrics}
)

# 출력
for baseline in ["Average", "Zeros"]:
    print(f"\n\n================ eval_baseline = {baseline} / MEAN ================\n")

    mean_out = (
        mean_summary[mean_summary["eval_baseline"] == baseline]
        .drop(columns=["eval_baseline"])
        .sort_values(["source", "explainer", "topk"])
        .reset_index(drop=True)
    )

    print(mean_out.to_string(index=False))

    print(f"\n\n================ eval_baseline = {baseline} / STD ================\n")

    std_out = (
        std_summary[std_summary["eval_baseline"] == baseline]
        .drop(columns=["eval_baseline"])
        .sort_values(["source", "explainer", "topk"])
        .reset_index(drop=True)
    )

    print(std_out.to_string(index=False))