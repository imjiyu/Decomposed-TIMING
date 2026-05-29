"""
diagnostic.py — 그림 그리기 전 진단
"""
import numpy as np

# 로드
data = "wafer"; fold = 0; seed = 42 ### 우선은 빠른애부터! epilepsy
results_dir = "./results_our/"
load = lambda k: np.load(f"{results_dir}/{data}_state_{k}_result_{fold}_{seed}.npy")

timing = load("timing_sample50_seg10_min10_max10")  # [N, T, D]
trend  = load("timing_td_trend_seg10_min10_max10")
resid  = load("timing_td_residual_seg10_min10_max10")

print(f"Shape: {timing.shape}")
N = timing.shape[0]

# === 1. 샘플별 강도 ===
trend_str = np.abs(trend).sum(axis=(1,2))   # [N]
resid_str = np.abs(resid).sum(axis=(1,2))
ratio = trend_str / (resid_str + 1e-8)

print(f"\n[샘플별 trend/residual 강도]")
print(f"  trend_strength: mean={trend_str.mean():.4f}, std={trend_str.std():.4f}")
print(f"  resid_strength: mean={resid_str.mean():.4f}, std={resid_str.std():.4f}")
print(f"  ratio (trend/resid): mean={ratio.mean():.3f}, std={ratio.std():.3f}")
print(f"  ratio min={ratio.min():.3f}, max={ratio.max():.3f}")
# → ratio의 std/mean이 크면 (>0.3) 샘플마다 비율이 다양 = 시나리오 1 ✅
# → std/mean이 작으면 (<0.1) 모든 샘플이 비슷한 비율 = 분해 의미 없음 ❌

# === 2. 시점별 correlation (분리 정도) ===
# 각 샘플 내에서 trend와 residual의 attribution 패턴이 얼마나 다른지
correlations = []
for i in range(N):
    t = trend[i].flatten()
    r = resid[i].flatten()
    if t.std() > 0 and r.std() > 0:
        correlations.append(np.corrcoef(t, r)[0, 1])
corr_mean = np.mean(correlations)

print(f"\n[시점별 패턴 분리]")
print(f"  trend vs residual correlation per sample: mean={corr_mean:.3f}")
# → 1에 가까우면 같은 시점을 highlight (분리 X, 시나리오 3 실패)
# → 0이나 음수면 다른 시점을 highlight (시나리오 3 ✅)
# → 0.3~0.6 정도면 부분적으로 다름 (OK)

# === 3. 원본 TIMING과의 관계 ===
# trend + residual ≈ timing 인지
timing_str = np.abs(timing).sum(axis=(1,2))
sum_str = trend_str + resid_str

print(f"\n[합 vs 원본]")
print(f"  |timing| mean: {timing_str.mean():.4f}")
print(f"  |trend|+|resid| mean: {sum_str.mean():.4f}")
print(f"  ratio: {(sum_str/timing_str).mean():.3f} (1에 가까울수록 분해가 깔끔)")

# === 4. (label 있으면) class별 분석 ===
# y_test 로드 가능하면:
try:
    # 너의 dataset loader 경로에 맞춰서 수정
    # import sys; sys.path.append('...')
    # from ... import load_epilepsy_test
    # y_test = load_epilepsy_test(fold=0)
    y_test = None  # 일단 None
    if y_test is not None:
        print(f"\n[Class별 패턴]")
        for c in np.unique(y_test):
            mask = (y_test == c)
            print(f"  Class {c}: trend={trend_str[mask].mean():.4f}, "
                  f"resid={resid_str[mask].mean():.4f}, "
                  f"ratio={ratio[mask].mean():.3f}")
        # → class별로 ratio가 다르면 시나리오 2 ✅
except:
    print("\n[Class 분석 스킵 — y_test 로드 필요]")