#!/usr/bin/env python3
"""
PHITS 真实数据验证：固定α vs 动态α + OMP/Lasso/Ridge 对比
只用 A_src_8x10.npy (8det × 10src, rank=6)
"""
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.linalg import pinv
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 1. 加载 PHITS 数据
# ============================================================
A = np.load('/root/.openclaw/workspace/elasticnet/phits_data/phits_10x8_data/PHITS_results_10x8/A_src_8x10.npy')
m, n = A.shape
print(f"=== PHITS 数据 ===")
print(f"A: {m}×{n}, rank={np.linalg.matrix_rank(A)}, cond={np.linalg.cond(A):.1f}")

# 处理 cond=inf：Tikhonov 正则化截断
epsilon = 1e-10 * np.max(A)
A_clean = np.where(np.abs(A) < epsilon, epsilon, A)
print(f"清洗后 cond={np.linalg.cond(A_clean):.1f}")

# 伪逆（用于基准）
A_pinv = pinv(A_clean)

# ============================================================
# 2. 生成稀疏源测试数据
# ============================================================
noise_levels = [0.01, 0.05, 0.10]
sparsity_levels = [0.2]  # sparsity_ratio=0.8 means 20% nonzero

results = []

def make_sparse_source(n, sparsity, max_val=1e4):
    x = np.zeros(n)
    k = max(1, int(n * (1 - sparsity)))
    idx = np.random.choice(n, k, replace=False)
    x[idx] = np.random.uniform(1e2, max_val, size=k)
    return x

def dynamic_alpha(A, b, x_init, k):
    """动态α：基于残差自适应调整 L1/L2 权重"""
    res = A @ x_init - b
    res_norm = np.linalg.norm(res)
    b_norm = np.linalg.norm(b)
    # 残差大 → 更多 L2（平滑）；残差小 → 更多 L1（稀疏）
    ratio = min(res_norm / (b_norm + 1e-10), 1.0)
    l1_ratio = 0.9 - 0.5 * ratio  # 0.4~0.9
    alpha = 1e-4 * (1 + ratio * 10)
    return alpha, l1_ratio

def metrics(x_true, x_pred):
    mse = np.mean((x_true - x_pred)**2)
    r2 = 1 - np.sum((x_true - x_pred)**2) / (np.sum((x_true - np.mean(x_true))**2) + 1e-10)
    support_true = (np.abs(x_true) > 1e-6).astype(int)
    support_pred = (np.abs(x_pred) > 1e-3 * np.max(np.abs(x_pred) + 1e-10)).astype(int)
    if support_true.sum() + support_pred.sum() == 0:
        iou = 1.0
    else:
        iou = np.sum(support_true & support_pred) / (np.sum(support_true | support_pred) + 1e-10)
    return mse, r2, iou

# ============================================================
# 3. 主实验循环
# ============================================================
n_trials = 20

for noise_lvl in noise_levels:
    for sparsity in sparsity_levels:
        trial_results = {k: [] for k in ['OMP', 'Lasso', 'Ridge', 'EN_fixed', 'EN_dynamic', 'Pinv']}
        
        for trial in range(n_trials):
            x_true = make_sparse_source(n, sparsity)
            # 列归一化
            col_norms = np.linalg.norm(A_clean, axis=0)
            col_norms[col_norms == 0] = 1
            A_norm = A_clean / col_norms
            
            b = A_clean @ x_true
            b_noisy = b + noise_lvl * np.linalg.norm(b) * np.random.randn(m)
            
            # --- OMP ---
            try:
                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=max(1, int(n*(1-sparsity))))
                omp.fit(A_norm, b_noisy)
                x_omp = omp.coef_ / col_norms
                trial_results['OMP'].append(metrics(x_true, x_omp))
            except:
                trial_results['OMP'].append((1e10, -999, 0))
            
            # --- Lasso (fixed α) ---
            x_lasso = Lasso(alpha=1e-3, max_iter=50000).fit(A_norm, b_noisy).coef_ / col_norms
            trial_results['Lasso'].append(metrics(x_true, x_lasso))
            
            # --- Ridge (fixed α) ---
            x_ridge = Ridge(alpha=1e-2).fit(A_norm, b_noisy).coef_ / col_norms
            trial_results['Ridge'].append(metrics(x_true, x_ridge))
            
            # --- Elastic Net fixed α ---
            en_fixed = ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=50000).fit(A_norm, b_noisy)
            x_en_fixed = en_fixed.coef_ / col_norms
            trial_results['EN_fixed'].append(metrics(x_true, x_en_fixed))
            
            # --- Elastic Net dynamic α (2-stage) ---
            # Stage 1: Ridge warm start
            x_warm = Ridge(alpha=1e-2).fit(A_norm, b_noisy).coef_ / col_norms
            x_warm_norm = x_warm * col_norms  # back to normalized space for warm start
            # Stage 2: dynamic EN
            for k_iter in range(3):
                alpha_dyn, l1_dyn = dynamic_alpha(A_norm, b_noisy, x_warm_norm, k_iter)
                en_dyn = ElasticNet(alpha=alpha_dyn, l1_ratio=l1_dyn, max_iter=50000)
                en_dyn.fit(A_norm, b_noisy)
                x_warm_norm = en_dyn.coef_
                x_warm = en_dyn.coef_
            trial_results['EN_dynamic'].append(metrics(x_true, x_warm_norm / col_norms))
            
            # --- Pseudoinverse baseline ---
            x_pinv = pinv(A_norm) @ b_noisy / col_norms
            trial_results['Pinv'].append(metrics(x_true, x_pinv))
        
        # Aggregate
        for method, vals in trial_results.items():
            mse_arr = np.array([v[0] for v in vals])
            r2_arr = np.array([v[1] for v in vals])
            iou_arr = np.array([v[2] for v in vals])
            results.append({
                'method': method,
                'noise': noise_lvl,
                'sparsity': sparsity,
                'MSE_mean': mse_arr.mean(),
                'MSE_std': mse_arr.std(),
                'R2_mean': r2_arr.mean(),
                'R2_std': r2_arr.std(),
                'IoU_mean': iou_arr.mean(),
                'IoU_std': iou_arr.std(),
            })

# ============================================================
# 4. 输出结果
# ============================================================
print("\n" + "="*80)
print(f"{'Method':<12} {'Noise':>6} {'Sparse':>7} {'MSE':>12} {'R²':>8} {'IoU':>7}")
print("="*80)

# Sort by R2 descending
results.sort(key=lambda r: r['R2_mean'], reverse=True)
for r in results:
    print(f"{r['method']:<12} {r['noise']:>6.2f} {r['sparsity']:>7.1f} "
          f"{r['MSE_mean']:>10.1f}±{r['MSE_std']:<5.1f} "
          f"{r['R2_mean']:>6.3f}±{r['R2_std']:<4.3f} "
          f"{r['IoU_mean']:>5.3f}±{r['IoU_std']:<4.3f}")

# ============================================================
# 5. 关键对比：固定α vs 动态α EN
# ============================================================
print("\n" + "="*80)
print("=== 关键对比：EN_fixed vs EN_dynamic ===")
for noise_lvl in noise_levels:
    for sparsity in sparsity_levels:
        fixed = [r for r in results if r['method']=='EN_fixed' and r['noise']==noise_lvl and r['sparsity']==sparsity][0]
        dynamic = [r for r in results if r['method']=='EN_dynamic' and r['noise']==noise_lvl and r['sparsity']==sparsity][0]
        winner = "DYNAMIC ✓" if dynamic['R2_mean'] > fixed['R2_mean'] else "FIXED ✓"
        delta_r2 = dynamic['R2_mean'] - fixed['R2_mean']
        print(f"  noise={noise_lvl:.2f} sparse={sparsity:.1f}: "
              f"fixed_R²={fixed['R2_mean']:.3f} vs dyn_R²={dynamic['R2_mean']:.3f} "
              f"(Δ={delta_r2:+.3f}) → {winner}")

# ============================================================
# 6. 各方法总体排名
# ============================================================
print("\n" + "="*80)
print("=== 总体排名（按 R² 均值）===")
from collections import defaultdict
method_r2 = defaultdict(list)
for r in results:
    method_r2[r['method']].append(r['R2_mean'])
ranking = sorted([(m, np.mean(v)) for m, v in method_r2.items()], key=lambda x: -x[1])
for i, (m, v) in enumerate(ranking, 1):
    print(f"  #{i}: {m:<14} R²={v:.4f}")
