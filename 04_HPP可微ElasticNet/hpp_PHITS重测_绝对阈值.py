#!/usr/bin/env python3
"""
HPP PHITS 8×12 重测 — 绝对阈值评估
=====================================
目标：在7个有效列中，HPP恢复了几个？
对比 V14 强列 100% 恢复率标准。
使用绝对阈值（不依赖预测最大值的相对比例）。
"""

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 1. 加载 PHITS 数据
# ============================================================
DATA_PATH = '/root/.openclaw/workspace/elasticnet/phits_data/phits_10x8_data/PHITS_results_10x8/A_matrix_8x12_GBq.npy'
A_full = np.load(DATA_PATH)

valid_cols = np.where(~np.all(A_full == 0, axis=0))[0]
valid_rows = np.where(~np.all(A_full == 0, axis=1))[0]
A_sub = A_full[np.ix_(valid_rows, valid_cols)]  # 7×7

# 列归一化
col_norms = np.linalg.norm(A_sub, axis=0)
col_norms_safe = np.where(col_norms == 0, 1.0, col_norms)
A_norm = A_sub / col_norms_safe

N = A_norm.shape[1]  # 7
print(f"PHITS 8×12 → 有效子矩阵 {A_sub.shape}, rank={np.linalg.matrix_rank(A_norm)}, cond={np.linalg.cond(A_norm):.1e}")
print(f"列范数: {np.array2string(col_norms, precision=2)}")
print(f"强列(范数>1): {np.where(col_norms > 1.0)[0]} → {col_norms[col_norms > 1.0]}")
print(f"弱列(范数<1): {np.where(col_norms < 1.0)[0]} → {col_norms[col_norms < 1.0]}")
print()

# ============================================================
# 2. HPP Elastic Net (from v3, simplified)
# ============================================================
class HPPElasticNet:
    """HPP EN with curriculum training"""
    def __init__(self, lambda1=0.01, lambda2=0.01, lr=0.01,
                 max_iter=3000, clip_norm=1.0):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lr0 = lr
        self.max_iter = max_iter
        self.clip_norm = clip_norm

    def fit(self, A, b, x_init=None):
        m, n = A.shape
        if x_init is not None:
            abs_x = np.abs(x_init) + 1e-4
            u = np.sign(x_init) * np.sqrt(abs_x)
            v = np.sqrt(abs_x)
        else:
            u = np.random.randn(n) * 0.01
            v = np.random.randn(n) * 0.01

        best_loss = np.inf
        best_beta = u * v
        warmup_end = int(self.max_iter * 0.3)
        ramp_end = int(self.max_iter * 0.6)

        for it in range(self.max_iter):
            lr = self.lr0 * 0.5 * (1 + np.cos(np.pi * it / self.max_iter))
            if it < warmup_end:
                l1_cur = 0.0
            elif it < ramp_end:
                progress = (it - warmup_end) / (ramp_end - warmup_end)
                l1_cur = self.lambda1 * progress
            else:
                l1_cur = self.lambda1

            beta = u * v
            residual = A @ beta - b
            AtR = A.T @ residual
            grad_u = AtR * v
            grad_v = AtR * u

            if l1_cur > 0:
                abs_uv = np.sqrt(beta**2 + 1e-6)
                grad_u += l1_cur * u * v**2 / (abs_uv + 1e-8)
                grad_v += l1_cur * v * u**2 / (abs_uv + 1e-8)

            grad_u += 2 * self.lambda2 * (v**2) * u
            grad_v += 2 * self.lambda2 * (u**2) * v

            g_norm = np.sqrt(np.sum(grad_u**2) + np.sum(grad_v**2))
            if g_norm > self.clip_norm:
                s = self.clip_norm / g_norm
                grad_u *= s
                grad_v *= s

            u -= lr * grad_u
            v -= lr * grad_v

            beta = u * v
            loss = np.sum(residual**2) + l1_cur * np.sum(np.abs(beta)) + self.lambda2 * np.sum(beta**2)

            if loss < best_loss:
                best_loss = loss
                best_beta = beta.copy()

        self.coef_ = best_beta
        return self

def run_hpp(A, b, l1, l2, n_starts=5, max_iter=3000):
    """多起点 HPP"""
    best_beta, best_loss = None, np.inf
    n = A.shape[1]
    ridge_init = Ridge(alpha=1.0).fit(A, b).coef_
    for s in range(n_starts):
        init = ridge_init if s == 0 else ridge_init + np.random.randn(n) * 0.01 * (np.std(ridge_init) + 1e-6)
        model = HPPElasticNet(lambda1=l1, lambda2=l2, lr=0.01, max_iter=max_iter)
        model.fit(A, b, x_init=init)
        loss = np.sum((A @ model.coef_ - b)**2)
        if loss < best_loss:
            best_loss = loss
            best_beta = model.coef_.copy()
    return best_beta

# ============================================================
# 3. V14-style EN (GridSearchCV, positive=True)
# ============================================================
def run_v14_en(A, b):
    """V14 风格 ElasticNet：GridSearchCV + positive=True"""
    param_grid = {
        'alpha': np.logspace(-6, 2, 40),
        'l1_ratio': np.linspace(1e-6, 1-1e-6, 40)
    }
    model = ElasticNet(max_iter=50000, fit_intercept=True,
                       positive=True, random_state=42, tol=1e-6)
    grid = GridSearchCV(model, param_grid, cv=3,
                        scoring='neg_mean_squared_error', n_jobs=-1, refit=True)
    grid.fit(A, b)
    return grid.best_estimator_.coef_

# ============================================================
# 4. 绝对阈值评估
# ============================================================
def eval_absolute(x_true, x_pred, abs_threshold=0.5):
    """
    绝对阈值评估：
    - 强列：x_true > 0 的位置
    - 恢复：强列位置上 |x_pred| > abs_threshold
    - 误报：x_true == 0 但 |x_pred| > abs_threshold
    """
    strong_mask = x_true > 0  # 真实非零位置
    n_strong = strong_mask.sum()
    if n_strong == 0:
        return {'recovered': 0, 'total': 0, 'fp': 0, 'fn': 0}

    # 强列恢复
    recovered = np.sum((np.abs(x_pred[strong_mask]) > abs_threshold))
    # 误报（零列位置预测 > 阈值）
    zero_mask = x_true == 0
    fp = np.sum(np.abs(x_pred[zero_mask]) > abs_threshold) if zero_mask.sum() > 0 else 0
    # 漏报
    fn = n_strong - recovered

    return {
        'recovered': int(recovered),
        'total': int(n_strong),
        'fp': int(fp),
        'fn': int(fn),
        'recovery_rate': recovered / n_strong,
    }

# ============================================================
# 5. 主实验
# ============================================================
print("=" * 70)
print("HPP PHITS 8×12 重测 — 绝对阈值评估")
print("=" * 70)

# 测试配置
NOISE_LEVELS = [0.01, 0.05, 0.10]  # 噪声水平（相对b的范数）
ABS_THRESHOLDS = [0.1, 0.5, 1.0]   # 绝对阈值 (GBq)
N_TRIALS = 10
SPARSE_K = 3  # 3个非零源（out of 7）

# 不同场景：非零源放在不同位置
# 场景1：放在强列位置（col 0,3,4，范数 > 1）
# 场景2：混合（强列+弱列）
# 场景3：放在弱列位置（col 1,2,5,6，范数 < 1）

strong_cols = np.where(col_norms > 1.0)[0]  # [0, 3, 4]
weak_cols = np.where(col_norms < 1.0)[0]    # [1, 2, 5, 6]
print(f"强列索引: {strong_cols} (范数: {col_norms[strong_cols]})")
print(f"弱列索引: {weak_cols} (范数: {col_norms[weak_cols]})")
print()

# ============================================================
# 场景 A：3个非零源全部放在强列 → "最优场景"
# ============================================================
print("=" * 70)
print("场景A：3个非零源全部放在强列（col 0,3,4）")
print("=" * 70)

for noise_lvl in NOISE_LEVELS:
    print(f"\n--- 噪声 {noise_lvl*100:.0f}% ---")
    hpp_rec = []
    v14_rec = []
    en_rec = []
    ridge_rec = []

    for trial in range(N_TRIALS):
        rng = np.random.RandomState(42 + trial)
        x_true = np.zeros(N)
        # 3个强列位置
        x_true[strong_cols] = rng.uniform(1.0, 10.0, size=len(strong_cols))

        # 生成观测
        b_clean = A_norm @ x_true
        b_noisy = b_clean + noise_lvl * np.linalg.norm(b_clean) * rng.randn(A_norm.shape[0])

        # --- HPP ---
        beta_hpp = run_hpp(A_norm, b_noisy, l1=0.01, l2=0.01, n_starts=3)
        x_hpp = beta_hpp / col_norms_safe  # 反归一化

        # --- V14-style EN ---
        x_v14 = run_v14_en(A_norm, b_noisy) / col_norms_safe

        # --- 标准 EN (not positive, for comparison) ---
        en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=50000).fit(A_norm, b_noisy)
        x_en = en.coef_ / col_norms_safe

        # --- Ridge ---
        ridge = Ridge(alpha=1.0).fit(A_norm, b_noisy)
        x_ridge = ridge.coef_ / col_norms_safe

        # 评估（用阈值 0.5 GBq）
        abs_thr = 0.5
        hpp_r = eval_absolute(x_true, x_hpp, abs_thr)
        v14_r = eval_absolute(x_true, x_v14, abs_thr)
        en_r = eval_absolute(x_true, x_en, abs_thr)
        ridge_r = eval_absolute(x_true, x_ridge, abs_thr)

        hpp_rec.append(hpp_r['recovered'])
        v14_rec.append(v14_r['recovered'])
        en_rec.append(en_r['recovered'])
        ridge_rec.append(ridge_r['recovered'])

        if trial < 3:  # 打印前3个trial的详情
            print(f"  Trial {trial}: true={np.array2string(x_true, precision=2)}")
            print(f"    HPP={np.array2string(x_hpp, precision=3)} → rec={hpp_r['recovered']}/{hpp_r['total']}, FP={hpp_r['fp']}")
            print(f"    V14={np.array2string(x_v14, precision=3)} → rec={v14_r['recovered']}/{v14_r['total']}, FP={v14_r['fp']}")
            print(f"    EN ={np.array2string(x_en, precision=3)} → rec={en_r['recovered']}/{en_r['total']}")
            print(f"    Rid={np.array2string(x_ridge, precision=3)} → rec={ridge_r['recovered']}/{ridge_r['total']}")

    print(f"\n  汇总 (noise={noise_lvl*100:.0f}%, abs_thr={abs_thr}):")
    print(f"  HPP:   {np.sum(hpp_rec)}/{N_TRIALS * len(strong_cols)} = {np.mean(hpp_rec)/len(strong_cols):.0%} 平均恢复率")
    print(f"  V14EN: {np.sum(v14_rec)}/{N_TRIALS * len(strong_cols)} = {np.mean(v14_rec)/len(strong_cols):.0%}")
    print(f"  EN:    {np.sum(en_rec)}/{N_TRIALS * len(strong_cols)} = {np.mean(en_rec)/len(strong_cols):.0%}")
    print(f"  Ridge: {np.sum(ridge_rec)}/{N_TRIALS * len(strong_cols)} = {np.mean(ridge_rec)/len(strong_cols):.0%}")

# ============================================================
# 场景 B：随机3个非零（强列+弱列混合）→ "真实场景"
# ============================================================
print("\n" + "=" * 70)
print("场景B：随机3个非零源（强列+弱列混合）→ 更真实")
print("=" * 70)

for noise_lvl in NOISE_LEVELS:
    print(f"\n--- 噪声 {noise_lvl*100:.0f}% ---")
    hpp_rec = []
    v14_rec = []
    hpp_strong_rec = []
    hpp_weak_rec = []
    v14_strong_rec = []
    v14_weak_rec = []

    for trial in range(N_TRIALS):
        rng = np.random.RandomState(100 + trial)
        x_true = np.zeros(N)
        active = rng.choice(N, size=SPARSE_K, replace=False)
        x_true[active] = rng.uniform(1.0, 10.0, size=SPARSE_K)

        b_clean = A_norm @ x_true
        b_noisy = b_clean + noise_lvl * np.linalg.norm(b_clean) * rng.randn(A_norm.shape[0])

        beta_hpp = run_hpp(A_norm, b_noisy, l1=0.01, l2=0.01, n_starts=3)
        x_hpp = beta_hpp / col_norms_safe

        x_v14 = run_v14_en(A_norm, b_noisy) / col_norms_safe

        abs_thr = 0.5
        hpp_r = eval_absolute(x_true, x_hpp, abs_thr)
        v14_r = eval_absolute(x_true, x_v14, abs_thr)
        hpp_rec.append(hpp_r['recovered'])
        v14_rec.append(v14_r['recovered'])

        # 强列/弱列分别统计
        strong_active = [i for i in active if i in strong_cols]
        weak_active = [i for i in active if i in weak_cols]
        if strong_active:
            hpp_sr = sum(1 for i in strong_active if abs(x_hpp[i]) > abs_thr)
            v14_sr = sum(1 for i in strong_active if abs(x_v14[i]) > abs_thr)
            hpp_strong_rec.append(hpp_sr)
            v14_strong_rec.append(v14_sr)
        if weak_active:
            hpp_wr = sum(1 for i in weak_active if abs(x_hpp[i]) > abs_thr)
            v14_wr = sum(1 for i in weak_active if abs(x_v14[i]) > abs_thr)
            hpp_weak_rec.append(hpp_wr)
            v14_weak_rec.append(v14_wr)

        if trial < 3:
            print(f"  Trial {trial}: active={active}, true={np.array2string(x_true, precision=2)}")
            print(f"    HPP={np.array2string(x_hpp, precision=3)} → rec={hpp_r['recovered']}/{hpp_r['total']}, FP={hpp_r['fp']}")
            print(f"    V14={np.array2string(x_v14, precision=3)} → rec={v14_r['recovered']}/{v14_r['total']}, FP={v14_r['fp']}")

    total_possible = N_TRIALS * SPARSE_K
    print(f"\n  汇总 (noise={noise_lvl*100:.0f}%, abs_thr={abs_thr}):")
    print(f"  HPP:   总恢复 {np.sum(hpp_rec)}/{total_possible} = {np.mean(hpp_rec)/SPARSE_K:.0%}")
    print(f"  V14EN: 总恢复 {np.sum(v14_rec)}/{total_possible} = {np.mean(v14_rec)/SPARSE_K:.0%}")
    if hpp_strong_rec:
        print(f"  HPP 强列恢复: {np.sum(hpp_strong_rec)}/{len(hpp_strong_rec)} 列被命中")
    if v14_strong_rec:
        print(f"  V14 强列恢复: {np.sum(v14_strong_rec)}/{len(v14_strong_rec)} 列被命中")
    if hpp_weak_rec:
        print(f"  HPP 弱列恢复: {np.sum(hpp_weak_rec)}/{len(hpp_weak_rec)} 列被命中")
    if v14_weak_rec:
        print(f"  V14 弱列恢复: {np.sum(v14_weak_rec)}/{len(v14_weak_rec)} 列被命中")

# ============================================================
# 场景 C：无噪声 → 理论上限
# ============================================================
print("\n" + "=" * 70)
print("场景C：无噪声 → 理论上限测试")
print("=" * 70)

rng = np.random.RandomState(7)
x_true = np.zeros(N)
active = np.array([0, 3, 4])  # 3个强列
x_true[active] = [5.0, 3.0, 8.0]

b_clean = A_norm @ x_true

# HPP
beta_hpp = run_hpp(A_norm, b_clean, l1=0.01, l2=0.01, n_starts=5, max_iter=5000)
x_hpp = beta_hpp / col_norms_safe

# V14 EN
x_v14 = run_v14_en(A_norm, b_clean) / col_norms_safe

# Ridge
ridge = Ridge(alpha=0.01).fit(A_norm, b_clean)
x_ridge = ridge.coef_ / col_norms_safe

# Pinv (pseudo-inverse)
x_pinv = np.linalg.lstsq(A_norm, b_clean, rcond=None)[0] / col_norms_safe

print(f"真值:   {np.array2string(x_true, precision=2)}")
print(f"HPP:    {np.array2string(x_hpp, precision=4)}")
print(f"V14EN:  {np.array2string(x_v14, precision=4)}")
print(f"Ridge:  {np.array2string(x_ridge, precision=4)}")
print(f"Pinv:   {np.array2string(x_pinv, precision=4)}")

abs_thr = 0.5
for name, x_pred in [('HPP', x_hpp), ('V14EN', x_v14), ('Ridge', x_ridge), ('Pinv', x_pinv)]:
    r = eval_absolute(x_true, x_pred, abs_thr)
    print(f"  {name:6s}: 恢复 {r['recovered']}/{r['total']}, FP={r['fp']}, "
          f"rel_err={np.mean(np.abs(x_true[x_true>0]-x_pred[x_true>0])/(x_true[x_true>0]+1e-10)):.3f}")

# ============================================================
# 6. 核心结论：不同绝对阈值下的恢复率
# ============================================================
print("\n" + "=" * 70)
print("核心数字：7个有效列中恢复了几个？")
print("=" * 70)

print("\n场景A（强列3个源，1%噪声），不同绝对阈值：")
rng = np.random.RandomState(42)
x_true_a = np.zeros(N)
x_true_a[strong_cols] = [5.0, 3.0, 8.0]
b_a = A_norm @ x_true_a + 0.01 * np.linalg.norm(A_norm @ x_true_a) * rng.randn(A_norm.shape[0])

beta_hpp = run_hpp(A_norm, b_a, 0.01, 0.01, n_starts=5, max_iter=5000)
x_hpp = beta_hpp / col_norms_safe
x_v14 = run_v14_en(A_norm, b_a) / col_norms_safe

for thr in [0.01, 0.1, 0.5, 1.0]:
    hpp_r = eval_absolute(x_true_a, x_hpp, thr)
    v14_r = eval_absolute(x_true_a, x_v14, thr)
    print(f"  阈值={thr:4.2f}: HPP {hpp_r['recovered']}/{hpp_r['total']}, V14 {v14_r['recovered']}/{v14_r['total']}")

print(f"\n7个有效列详情 (1%噪声, abs_thr=0.5):")
print(f"  {'Col':>3} {'True':>8} {'HPP':>8} {'V14':>8} {'ColNorm':>10} {'HPP OK':>7} {'V14 OK':>7}")
for i in range(N):
    hpp_ok = "✓" if abs(x_hpp[i]) > 0.5 and x_true_a[i] > 0 else ("FP" if abs(x_hpp[i]) > 0.5 and x_true_a[i] == 0 else "✗")
    v14_ok = "✓" if abs(x_v14[i]) > 0.5 and x_true_a[i] > 0 else ("FP" if abs(x_v14[i]) > 0.5 and x_true_a[i] == 0 else "✗")
    print(f"  {i:>3} {x_true_a[i]:>8.2f} {x_hpp[i]:>8.3f} {x_v14[i]:>8.3f} {col_norms[i]:>10.2e} {hpp_ok:>7} {v14_ok:>7}")

print("\n" + "=" * 70)
print("=== 重测完成 ===")
print("=" * 70)
