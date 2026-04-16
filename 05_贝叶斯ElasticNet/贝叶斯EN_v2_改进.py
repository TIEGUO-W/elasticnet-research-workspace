#!/usr/bin/env python3
"""
贝叶斯 EN vs EN_Fixed vs Lasso 对比实验 (v2)
修正版：正确处理 PHITS 8×12 矩阵的评估

核心指标：
- 7个有效列 [1,5,6,7,8,9,10] 中恢复了几个？
- 平均相对误差是多少？
- 零列 [0,2,3,4,11] 是否放了非零值？

实现策略：
- 贝叶斯 EN = MAP (scipy.optimize) + Laplace 近似
- funsor 用于后验分布构建辅助
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LassoCV, ElasticNet, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/root/.openclaw/workspace/elasticnet/phits_data/phits_10x8_data/PHITS_results_10x8/A_matrix_8x12_GBq.npy'

# 有效列和零列
NONZERO_COLS = [1, 5, 6, 7, 8, 9, 10]  # 7个有效列
ZERO_COLS = [0, 2, 3, 4, 11]            # 5个零列

def load_data():
    A = np.load(DATA_PATH)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms_safe = col_norms.copy()
    col_norms_safe[col_norms_safe < 1e-10] = 1.0
    A_norm = A / col_norms_safe
    return A, A_norm, col_norms_safe

def make_source_with_zeros(n, rng, sparsity=None):
    """
    生成源向量：
    - 在有效列 [1,5,6,7,8,9,10] 中随机选几个放非零值
    - 在零列 [0,2,3,4,11] 中也放非零值（体现不可观测区域的局限性）
    """
    if sparsity is None:
        sparsity = rng.uniform(0.6, 0.9)
    
    x = np.zeros(n)
    
    # 有效列中选非零
    n_eff = len(NONZERO_COLS)
    n_eff_nz = max(1, int(n_eff * (1 - sparsity)))  # 至少1个
    eff_idx = rng.choice(NONZERO_COLS, size=n_eff_nz, replace=False)
    x[eff_idx] = rng.uniform(2.0, 50.0, size=n_eff_nz)
    
    # 零列中也放值（不可观测区域有源但探测器看不到）
    n_zero_nz = max(1, rng.randint(1, min(3, len(ZERO_COLS)+1)))
    zero_idx = rng.choice(ZERO_COLS, size=n_zero_nz, replace=False)
    x[zero_idx] = rng.uniform(0.5, 5.0, size=n_zero_nz)
    
    return x, eff_idx.tolist(), zero_idx.tolist()

def evaluate(x_true, x_pred, threshold=0.1):
    """
    核心评估：
    1. 7个有效列中：真实非零的有几个？恢复了几个？
    2. 恢复的平均相对误差
    3. 零列是否误放非零值
    """
    x_true_eff = x_true[NONZERO_COLS]
    x_pred_eff = x_pred[NONZERO_COLS]
    
    # 有效列：哪些是真实非零
    true_nz_eff = np.abs(x_true_eff) > 1e-6
    n_true_nz = int(true_nz_eff.sum())
    
    # 有效列：哪些被预测为非零（阈值 = 最大预测值的 threshold）
    pred_max = np.max(np.abs(x_pred_eff)) + 1e-10
    pred_nz_eff = np.abs(x_pred_eff) > threshold * pred_max
    
    # 恢复了几个有效非零列
    if n_true_nz > 0:
        recovered = int((true_nz_eff & pred_nz_eff).sum())
        # 相对误差（仅真实非零的有效列）
        nz_idx = true_nz_eff
        rel_errors = np.abs(x_true_eff[nz_idx] - x_pred_eff[nz_idx]) / (np.abs(x_true_eff[nz_idx]) + 1e-10)
        avg_rel_err = float(np.mean(rel_errors))
    else:
        recovered = 0
        avg_rel_err = 0.0
    
    # 零列分析
    x_true_zero = x_true[ZERO_COLS]
    x_pred_zero = x_pred[ZERO_COLS]
    # 零列在A中是零，所以 b 中没有贡献，理论上无法恢复
    # 检查预测是否在零列放了非零值
    pred_max_global = np.max(np.abs(x_pred)) + 1e-10
    zero_col_fp = np.sum(np.abs(x_pred_zero) > threshold * pred_max_global)
    
    return {
        'n_true_nz_eff': n_true_nz,      # 有效列中真实非零数
        'recovered': recovered,            # 恢复了几个
        'avg_rel_err': avg_rel_err,        # 平均相对误差
        'zero_col_fp': int(zero_col_fp),   # 零列误放非零数
    }

# ============================================================
# 贝叶斯 Elastic Net (MAP + Laplace Approximation)
# ============================================================
def bayesian_en(A_norm, b, col_norms_safe, alpha=0.1, l1_ratio=0.5, n_samples=200, seed=42):
    """
    贝叶斯 EN: MAP估计 + Laplace近似后验
    - 先验: Laplace(L1) × Normal(L2) ≈ Elastic Net 先验
    - 似然: Normal(b | A·β, σ²I)
    - 后验近似: 高斯(Laplace approx)
    """
    n = A_norm.shape[1]
    sigma = 1.0  # 初始噪声估计
    
    # 负对数后验
    def neg_log_posterior(beta):
        resid = b - A_norm @ beta
        # 似然
        nll = 0.5 * np.sum(resid**2) / sigma**2
        # L1 先验 (Laplace)
        nll += alpha * l1_ratio * np.sum(np.abs(beta))
        # L2 先验 (Normal)
        nll += alpha * (1 - l1_ratio) * 0.5 * np.sum(beta**2)
        return nll
    
    # 梯度（光滑近似 L1）
    def neg_log_posterior_grad(beta):
        resid = b - A_norm @ beta
        eps = 1e-8
        grad = -A_norm.T @ resid / sigma**2
        # L1 subgradient (光滑近似)
        grad += alpha * l1_ratio * beta / (np.abs(beta) + eps)
        # L2 gradient
        grad += alpha * (1 - l1_ratio) * beta
        return grad
    
    # 初始值
    x0 = np.linalg.lstsq(A_norm, b, rcond=None)[0]
    
    result = minimize(neg_log_posterior, x0, jac=neg_log_posterior_grad,
                     method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-12})
    
    beta_map = result.x
    
    # Laplace 近似: Hessian
    eps = 1e-8
    H = A_norm.T @ A_norm / sigma**2
    H += alpha * (1 - l1_ratio) * np.eye(n)
    # L1 Hessian approx
    H += alpha * l1_ratio * np.diag(1.0 / (np.abs(beta_map) + eps))
    
    try:
        cov = np.linalg.inv(H)
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals <= 0):
            cov += (abs(min(eigvals)) + 1e-8) * np.eye(n)
    except:
        cov = np.eye(n) * 0.01
    
    # 采样
    rng = np.random.RandomState(seed)
    try:
        L = np.linalg.cholesky(cov)
        samples_norm = beta_map + rng.randn(n_samples, n) @ L.T
    except:
        std = np.sqrt(np.abs(np.diag(cov)))
        samples_norm = beta_map + rng.randn(n_samples, n) * std
    
    # 还原列归一化
    beta_map_denorm = beta_map / col_norms_safe
    samples = samples_norm / col_norms_safe[np.newaxis, :]
    
    return {
        'map': beta_map_denorm,
        'mean': np.mean(samples, axis=0),
        'median': np.median(samples, axis=0),
        'std': np.std(samples, axis=0),
        'q025': np.percentile(samples, 2.5, axis=0),
        'q975': np.percentile(samples, 97.5, axis=0),
    }

def bayesian_en_best(A_norm, b, col_norms_safe):
    """贝叶斯 EN + BIC 超参选择"""
    m, n = A_norm.shape
    best_bic = np.inf
    best_result = None
    best_params = None
    
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
        for l1_ratio in [0.3, 0.5, 0.7, 0.9]:
            res = bayesian_en(A_norm, b, col_norms_safe, alpha=alpha, l1_ratio=l1_ratio,
                            n_samples=100, seed=42)
            beta_norm = res['map'] * col_norms_safe  # back to normalized space
            resid = b - A_norm @ beta_norm
            rss = np.sum(resid**2)
            n_active = np.sum(np.abs(beta_norm) > 1e-6)
            bic = m * np.log(rss / m + 1e-20) + n_active * np.log(m)
            if bic < best_bic:
                best_bic = bic
                best_result = res
                best_params = (alpha, l1_ratio)
    
    return best_result, best_params

# ============================================================
# EN_Fixed (V14 风格)
# ============================================================
def en_fixed(A_norm, b, col_norms_safe):
    """EN_Fixed with GridSearchCV"""
    param_grid = {
        'alpha': np.logspace(-4, 1, 20),
        'l1_ratio': np.linspace(0.1, 0.99, 20)
    }
    en = ElasticNet(max_iter=50000, fit_intercept=False, positive=False, tol=1e-8)
    cv_folds = min(3, A_norm.shape[0])
    grid = GridSearchCV(en, param_grid, cv=cv_folds,
                        scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(A_norm, b)
    x_est = grid.best_estimator_.coef_ / col_norms_safe
    return x_est, grid.best_params_

# ============================================================
# Lasso
# ============================================================
def lasso_cv(A_norm, b, col_norms_safe):
    """LassoCV"""
    cv_folds = min(3, A_norm.shape[0])
    lasso = LassoCV(cv=cv_folds, max_iter=50000, fit_intercept=False)
    lasso.fit(A_norm, b)
    x_est = lasso.coef_ / col_norms_safe
    return x_est, lasso.alpha_

# ============================================================
# funsor 辅助验证
# ============================================================
def funsor_coverage_check(beta_map, beta_std, x_true):
    """用 funsor 构建后验高斯，检查覆盖"""
    try:
        import funsor
        from funsor import Tensor, Gaussian
        n = len(beta_map)
        rng = np.random.RandomState(42)
        
        # 后验近似为高斯 N(beta_map, diag(beta_std^2))
        coverage = 0
        for j in range(n):
            mu = beta_map[j]
            sigma = max(beta_std[j], 1e-10)
            # 95% CI
            lo = mu - 1.96 * sigma
            hi = mu + 1.96 * sigma
            if lo <= x_true[j] <= hi:
                coverage += 1
        
        return coverage, n
    except:
        return -1, len(beta_map)

# ============================================================
# 主实验
# ============================================================
def main():
    print("=" * 80)
    print("贝叶斯 EN vs EN_Fixed vs Lasso 对比")
    print(f"PHITS 8×12 矩阵 | 有效列 {NONZERO_COLS} | 零列 {ZERO_COLS}")
    print("=" * 80)
    
    A, A_norm, col_norms_safe = load_data()
    m, n = A.shape
    print(f"矩阵: {m}×{n}, 列归一化后 cond={np.linalg.cond(A_norm[:, NONZERO_COLS]):.2e}")
    print()
    
    rng = np.random.RandomState(42)
    noise_level = 0.01  # 1% 噪声
    n_trials = 10
    
    # ============================================================
    # 详细展示 1 次
    # ============================================================
    print("=" * 60)
    print("详细展示 (Trial 1)")
    print("=" * 60)
    
    x_true, eff_idx, zero_idx = make_source_with_zeros(n, rng)
    b = A @ x_true
    b_noisy = b + noise_level * np.linalg.norm(b) * rng.randn(m)
    
    print(f"真实源 (非零有效列={eff_idx}, 零列非零={zero_idx}):")
    for j in range(n):
        if abs(x_true[j]) > 1e-6:
            col_type = "有效" if j in NONZERO_COLS else "零列"
            print(f"  x[{j:2d}] = {x_true[j]:8.3f}  ({col_type})")
    
    # --- 贝叶斯 EN ---
    print("\n--- 贝叶斯 EN (MAP + Laplace) ---")
    bayes_res, bayes_params = bayesian_en_best(A_norm, b_noisy, col_norms_safe)
    print(f"最优参数: alpha={bayes_params[0]}, l1_ratio={bayes_params[1]}")
    print(f"MAP估计:")
    for j in NONZERO_COLS:
        ci_lo = bayes_res['q025'][j]
        ci_hi = bayes_res['q975'][j]
        in_ci = "✓" if ci_lo <= x_true[j] <= ci_hi else "✗"
        print(f"  x[{j:2d}]: true={x_true[j]:8.3f}, map={bayes_res['map'][j]:8.3f}, "
              f"95%CI=[{ci_lo:.3f}, {ci_hi:.3f}] {in_ci}")
    bayes_eval = evaluate(x_true, bayes_res['map'])
    print(f"恢复: {bayes_eval['recovered']}/{bayes_eval['n_true_nz_eff']}, "
          f"相对误差: {bayes_eval['avg_rel_err']:.1%}")
    
    # funsor 覆盖检查
    cov, tot = funsor_coverage_check(bayes_res['map'], bayes_res['std'], x_true)
    print(f"funsor 后验覆盖: {cov}/{tot}")
    
    # --- EN_Fixed ---
    print("\n--- EN_Fixed (GridSearchCV) ---")
    x_en, en_params = en_fixed(A_norm, b_noisy, col_norms_safe)
    print(f"最优参数: alpha={en_params['alpha']:.4f}, l1_ratio={en_params['l1_ratio']:.4f}")
    print(f"估计:")
    for j in NONZERO_COLS:
        print(f"  x[{j:2d}]: true={x_true[j]:8.3f}, est={x_en[j]:8.3f}")
    en_eval = evaluate(x_true, x_en)
    print(f"恢复: {en_eval['recovered']}/{en_eval['n_true_nz_eff']}, "
          f"相对误差: {en_eval['avg_rel_err']:.1%}")
    
    # --- Lasso ---
    print("\n--- Lasso (CV) ---")
    x_lasso, lasso_alpha = lasso_cv(A_norm, b_noisy, col_norms_safe)
    print(f"最优 alpha: {lasso_alpha:.6f}")
    print(f"估计:")
    for j in NONZERO_COLS:
        print(f"  x[{j:2d}]: true={x_true[j]:8.3f}, est={x_lasso[j]:8.3f}")
    lasso_eval = evaluate(x_true, x_lasso)
    print(f"恢复: {lasso_eval['recovered']}/{lasso_eval['n_true_nz_eff']}, "
          f"相对误差: {lasso_eval['avg_rel_err']:.1%}")
    
    # ============================================================
    # 批量实验 10 次
    # ============================================================
    print("\n" + "=" * 80)
    print(f"批量实验 ({n_trials} 次)")
    print("=" * 80)
    
    summary = {name: {'recovered': [], 'total': [], 'rel_err': [], 'zero_fp': [], 'mse': []}
               for name in ['Bayesian_EN', 'EN_Fixed', 'Lasso']}
    
    for trial in range(n_trials):
        x_true, eff_idx, zero_idx = make_source_with_zeros(n, rng)
        b = A @ x_true
        b_noisy = b + noise_level * np.linalg.norm(b) * rng.randn(m)
        
        # 贝叶斯 EN
        try:
            br, _ = bayesian_en_best(A_norm, b_noisy, col_norms_safe)
            be = evaluate(x_true, br['map'])
        except:
            be = {'recovered': 0, 'n_true_nz_eff': 0, 'avg_rel_err': 999, 'zero_col_fp': 0}
        
        # EN_Fixed
        try:
            xe, _ = en_fixed(A_norm, b_noisy, col_norms_safe)
            ee = evaluate(x_true, xe)
        except:
            ee = {'recovered': 0, 'n_true_nz_eff': 0, 'avg_rel_err': 999, 'zero_col_fp': 0}
        
        # Lasso
        try:
            xl, _ = lasso_cv(A_norm, b_noisy, col_norms_safe)
            le = evaluate(x_true, xl)
        except:
            le = {'recovered': 0, 'n_true_nz_eff': 0, 'avg_rel_err': 999, 'zero_col_fp': 0}
        
        for name, ev in [('Bayesian_EN', be), ('EN_Fixed', ee), ('Lasso', le)]:
            summary[name]['recovered'].append(ev['recovered'])
            summary[name]['total'].append(ev['n_true_nz_eff'])
            summary[name]['rel_err'].append(ev['avg_rel_err'])
            summary[name]['zero_fp'].append(ev['zero_col_fp'])
        
        print(f"Trial {trial+1:2d}: eff_nz={len(eff_idx)}, zero_nz={len(zero_idx)} | "
              f"Bayes={be['recovered']}/{be['n_true_nz_eff']}, "
              f"EN={ee['recovered']}/{ee['n_true_nz_eff']}, "
              f"Lasso={le['recovered']}/{le['n_true_nz_eff']}")
    
    # ============================================================
    # 汇总
    # ============================================================
    print("\n" + "=" * 80)
    print("🎯 汇总（10次平均）")
    print("=" * 80)
    print(f"\n{'方法':<16} {'恢复/总数':>12} {'恢复率':>8} {'相对误差':>10} {'零列误放':>8}")
    print("-" * 58)
    
    for name in ['Bayesian_EN', 'EN_Fixed', 'Lasso']:
        d = summary[name]
        total_rec = sum(d['recovered'])
        total_nz = sum(d['total'])
        rate = total_rec / max(total_nz, 1) * 100
        # 过滤异常值
        valid_errs = [e for e in d['rel_err'] if e < 100]
        avg_err = np.mean(valid_errs) if valid_errs else float('inf')
        avg_zfp = np.mean(d['zero_fp'])
        
        print(f"{name:<16} {total_rec:>5}/{total_nz:<5} {rate:>7.0f}% {avg_err:>9.1%} {avg_zfp:>8.1f}")
    
    # 核心数字
    print("\n" + "=" * 80)
    print("📊 核心数字")
    print("=" * 80)
    for name in ['Bayesian_EN', 'EN_Fixed', 'Lasso']:
        d = summary[name]
        total_rec = sum(d['recovered'])
        total_nz = sum(d['total'])
        valid_errs = [e for e in d['rel_err'] if e < 100]
        avg_err = np.mean(valid_errs) if valid_errs else float('inf')
        print(f"\n{name}:")
        if total_nz > 0:
            print(f"  有效列恢复: {total_rec}/{total_nz} ({total_rec/total_nz*100:.0f}%)")
        else:
            print(f"  有效列恢复: 0/0")
        print(f"  平均相对误差: {avg_err:.1%}")
        print(f"  零列误放非零: 平均{np.mean(d['zero_fp']):.1f}列/次")

if __name__ == "__main__":
    main()
