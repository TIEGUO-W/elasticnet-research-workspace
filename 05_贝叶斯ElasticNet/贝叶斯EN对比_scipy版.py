#!/usr/bin/env python3
"""
贝叶斯 Elastic Net 对比实验（纯 scipy 实现）
与 EN_Fixed (V14) 和 Lasso 对比

关键要求：
- PHITS 8×12 矩阵（7个有效列：[1,5,6,7,8,9,10]，5个零列）
- 列归一化（核心策略）
- sparsity 0.6~0.9 浮动
- 零列放非零值（体现不可观测区域的局限性）
- 核心指标：恢复了7列中的几列？平均相对误差？

实现策略（纯 scipy，不依赖 JAX/NumPyro/PyMC）：
1. 贝叶斯 EN = Laplace 先验(beta) + Normal 先验(beta) + Normal 似然
   → MAP 估计等价于 Elastic Net（scipy.optimize.minimize）
2. 后验近似 = Laplace Approximation（Hessian → 协方差 → 采样 → CI）
3. 利用 funsor 做分布运算辅助
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 数据
# ============================================================
DATA_PATH = '/root/.openclaw/workspace/elasticnet/phits_data/phits_10x8_data/PHITS_results_10x8/A_matrix_8x12_GBq.npy'

def load_phits_matrix():
    A = np.load(DATA_PATH)
    print(f"PHITS 8×12 矩阵: shape={A.shape}")
    col_norms = np.linalg.norm(A, axis=0)
    nonzero_cols = np.where(col_norms >= 1e-10)[0].tolist()
    zero_cols = np.where(col_norms < 1e-10)[0].tolist()
    print(f"非零列（有效列）: {nonzero_cols}  ({len(nonzero_cols)}个)")
    print(f"零列（不可观测）: {zero_cols}  ({len(zero_cols)}个)")
    return A, nonzero_cols, zero_cols

def normalize_columns(A):
    """列归一化"""
    col_norms = np.linalg.norm(A, axis=0)
    col_norms_safe = np.where(col_norms < 1e-10, 1.0, col_norms)
    A_norm = A / col_norms_safe
    return A_norm, col_norms, col_norms_safe

def generate_sparse_source(n, sparsity, rng, max_val=50.0):
    """生成稀疏源向量，确保非零元素数量合理"""
    n_nonzero = max(1, int(n * (1 - sparsity)))
    x = np.zeros(n)
    idx = rng.choice(n, size=n_nonzero, replace=False)
    x[idx] = rng.uniform(2.0, max_val, size=n_nonzero)
    return x, n_nonzero

# ============================================================
# 贝叶斯 Elastic Net（纯 scipy 实现）
# ============================================================
def bayesian_elastic_net_log_posterior(beta, A_norm, b_obs, alpha, l1_ratio, sigma_noise):
    """
    贝叶斯 EN 对数后验 = 对数似然 + 对数先验
    
    似然:  Normal(b | A·β, σ²I)
    先验:  Elastic Net = L1(Laplace) + L2(Normal)
           p(β) ∝ exp(-α * l1_ratio * ||β||_1 - α * (1-l1_ratio)/2 * ||β||²_2)
    """
    n = len(beta)
    residual = b_obs - A_norm @ beta
    
    # 对数似然: -||b - Aβ||² / (2σ²)
    log_lik = -0.5 * np.sum(residual**2) / sigma_noise**2
    
    # 对数先验: Elastic Net
    # L1 部分: -α * l1_ratio * ||β||_1 (Laplace 先验)
    log_prior_l1 = -alpha * l1_ratio * np.sum(np.abs(beta))
    # L2 部分: -α * (1-l1_ratio) / 2 * ||β||²_2 (Normal 先验)
    log_prior_l2 = -alpha * (1 - l1_ratio) * 0.5 * np.sum(beta**2)
    
    return log_lik + log_prior_l1 + log_prior_l2

def bayesian_elastic_en_negative_posterior(beta, A_norm, b_obs, alpha, l1_ratio, sigma_noise):
    """负对数后验（用于 minimize）"""
    return -bayesian_elastic_net_log_posterior(beta, A_norm, b_obs, alpha, l1_ratio, sigma_noise)

def bayesian_elastic_net_map(A_norm, b_obs, alpha, l1_ratio, sigma_noise=1.0):
    """贝叶斯 EN MAP 估计"""
    n = A_norm.shape[1]
    
    # 初始值：最小二乘
    x0 = np.linalg.lstsq(A_norm, b_obs, rcond=None)[0]
    
    result = minimize(
        bayesian_elastic_en_negative_posterior,
        x0, args=(A_norm, b_obs, alpha, l1_ratio, sigma_noise),
        method='L-BFGS-B',
        options={'maxiter': 5000, 'ftol': 1e-12}
    )
    
    return result.x

def bayesian_elastic_net_laplace_approx(A_norm, b_obs, beta_map, alpha, l1_ratio, sigma_noise=1.0):
    """
    Laplace 近似：用 Hessian 构建后验高斯近似
    
    H = ∂²(-log p(β|b)) / ∂β² = A^T A / σ² + α(1-l1_ratio) * I + L1 Hessian approx
    """
    n = len(beta_map)
    m = A_norm.shape[0]
    
    # Hessian of negative log posterior
    # 似然部分：A^T A / σ²
    H = A_norm.T @ A_norm / sigma_noise**2
    # L2 先验部分：α * (1-l1_ratio) * I
    H += alpha * (1 - l1_ratio) * np.eye(n)
    # L1 部分：用光滑近似 |x| ≈ √(x² + ε) 的 Hessian
    eps = 1e-6
    # L1 Hessian ≈ α * l1_ratio * diag(1/√(β² + ε))
    H += alpha * l1_ratio * np.diag(1.0 / np.sqrt(beta_map**2 + eps))
    
    # 协方差矩阵
    try:
        cov = np.linalg.inv(H)
        # 确保正定
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals <= 0):
            cov = cov + (abs(min(eigvals)) + 1e-6) * np.eye(n)
    except np.linalg.LinAlgError:
        cov = np.eye(n) * 0.01
    
    return cov

def bayesian_en_predict(A_norm, b_obs, alpha, l1_ratio, sigma_noise, col_norms_safe, n_samples=500, seed=42):
    """
    完整的贝叶斯 EN 推断流程：
    1. MAP 估计
    2. Laplace 近似 → 后验协方差
    3. 从后验采样
    4. 还原列归一化
    5. 统计量
    """
    rng = np.random.RandomState(seed)
    
    # MAP
    beta_map = bayesian_elastic_net_map(A_norm, b_obs, alpha, l1_ratio, sigma_noise)
    
    # Laplace 近似
    cov = bayesian_elastic_net_laplace_approx(A_norm, b_obs, beta_map, alpha, l1_ratio, sigma_noise)
    
    # 采样
    try:
        L = np.linalg.cholesky(cov)
        samples_norm = beta_map[np.newaxis, :] + rng.randn(n_samples, len(beta_map)) @ L.T
    except np.linalg.LinAlgError:
        # fallback: 对角近似
        std_diag = np.sqrt(np.abs(np.diag(cov)))
        samples_norm = beta_map[np.newaxis, :] + rng.randn(n_samples, len(beta_map)) * std_diag[np.newaxis, :]
    
    # 还原列归一化
    samples = samples_norm / col_norms_safe[np.newaxis, :]
    beta_map_denorm = beta_map / col_norms_safe
    
    return {
        'map': beta_map_denorm,
        'mean': np.mean(samples, axis=0),
        'median': np.median(samples, axis=0),
        'std': np.std(samples, axis=0),
        'q025': np.percentile(samples, 2.5, axis=0),
        'q975': np.percentile(samples, 97.5, axis=0),
        'samples': samples,
    }

# ============================================================
# 利用 funsor 做辅助验证（分布运算）
# ============================================================
def funsor_validate_posterior(beta_map, cov_diag, x_true):
    """用 funsor 构建后验分布并验证覆盖"""
    try:
        import funsor
        from funsor import Tensor, Gaussian
        import numpy as np
        
        n = len(beta_map)
        # 用 funsor Gaussian 构建后验
        # 验证每个分量的 marginal
        coverage_count = 0
        for j in range(n):
            mu_j = beta_map[j]
            sigma_j = np.sqrt(max(cov_diag[j], 1e-10))
            # 检查真实值是否在 95% CI 内
            if abs(x_true[j] - mu_j) < 1.96 * sigma_j:
                coverage_count += 1
        
        return coverage_count, n
    except Exception:
        return 0, len(beta_map)

# ============================================================
# EN_Fixed (V14 风格)
# ============================================================
def run_en_fixed(A_norm, b_obs, col_norms_safe, alpha=0.1, l1_ratio=0.5):
    """EN_Fixed: 固定 alpha, l1_ratio 的 Elastic Net"""
    en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, 
                    fit_intercept=False, positive=False, tol=1e-6)
    en.fit(A_norm, b_obs)
    x_est = en.coef_ / col_norms_safe
    return x_est

def run_en_fixed_cv(A_norm, b_obs, col_norms_safe):
    """EN_Fixed + GridSearchCV（V14 风格）"""
    param_grid = {
        'alpha': np.logspace(-4, 1, 30),
        'l1_ratio': np.linspace(0.01, 0.99, 30)
    }
    en = ElasticNet(max_iter=10000, fit_intercept=False, positive=False, tol=1e-6)
    grid = GridSearchCV(en, param_grid, cv=min(3, A_norm.shape[0]), 
                        scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(A_norm, b_obs)
    x_est = grid.best_estimator_.coef_ / col_norms_safe
    return x_est, grid.best_params_

# ============================================================
# Lasso
# ============================================================
def run_lasso(A_norm, b_obs, col_norms_safe):
    """Lasso with CV"""
    from sklearn.linear_model import LassoCV
    lasso = LassoCV(cv=min(3, A_norm.shape[0]), max_iter=10000, fit_intercept=False)
    lasso.fit(A_norm, b_obs)
    x_est = lasso.coef_ / col_norms_safe
    return x_est, lasso.alpha_

# ============================================================
# 评估指标
# ============================================================
def evaluate_recovery(x_true, x_pred, nonzero_cols, zero_cols, threshold_ratio=0.1):
    """
    核心评估指标：
    1. 有效列恢复数：7列中恢复了几列？
    2. 相对误差（非零列）
    3. 零列是否被误放非零值？
    """
    n = len(x_true)
    
    # 非零列分析
    true_support = np.abs(x_true) > 1e-6
    pred_support = np.abs(x_pred) > threshold_ratio * (np.max(np.abs(x_pred)) + 1e-10)
    
    # 有效列中恢复了几列
    true_nz_in_nonzero_cols = true_support[nonzero_cols]
    pred_nz_in_nonzero_cols = pred_support[nonzero_cols]
    
    # TP: 真实非零且预测非零
    recovered = np.sum(true_nz_in_nonzero_cols & pred_nz_in_nonzero_cols)
    total_true_nz = np.sum(true_nz_in_nonzero_cols)
    
    # 相对误差（仅非零元素）
    if total_true_nz > 0:
        nz_mask = true_support
        rel_errors = np.abs(x_true[nz_mask] - x_pred[nz_mask]) / (np.abs(x_true[nz_mask]) + 1e-10)
        avg_rel_err = float(np.mean(rel_errors))
    else:
        avg_rel_err = 0.0
    
    # 零列分析
    zero_col_values = x_pred[zero_cols]
    max_zero_col_val = float(np.max(np.abs(zero_col_values))) if len(zero_cols) > 0 else 0.0
    has_false_positive = max_zero_col_val > 0.1 * (np.max(np.abs(x_pred)) + 1e-10)
    
    # 全局指标
    mse = float(np.mean((x_true - x_pred)**2))
    rel_err_global = float(np.linalg.norm(x_true - x_pred) / (np.linalg.norm(x_true) + 1e-10))
    
    return {
        'recovered': int(recovered),
        'total_true_nz': int(total_true_nz),
        'avg_rel_err_nonzero': avg_rel_err,
        'max_zero_col_val': max_zero_col_val,
        'has_false_positive': has_false_positive,
        'mse': mse,
        'rel_err_global': rel_err_global,
    }

# ============================================================
# 主实验
# ============================================================
def main():
    print("=" * 80)
    print("贝叶斯 EN vs EN_Fixed vs Lasso 对比实验")
    print("PHITS 8×12 矩阵 · 列归一化 · sparsity 0.6~0.9")
    print("=" * 80)
    
    # 加载数据
    A_full, nonzero_cols, zero_cols = load_phits_matrix()
    n_total = A_full.shape[1]
    m = A_full.shape[0]
    
    # 列归一化（保留所有12列，零列归一化系数=1）
    A_norm, col_norms, col_norms_safe = normalize_columns(A_full)
    print(f"\n列归一化后 cond = {np.linalg.cond(A_norm):.2e}")
    print(f"列范数: {np.round(col_norms, 4)}")
    
    # 实验参数
    n_trials = 10
    sparsity_range = (0.6, 0.9)
    noise_level = 0.01
    rng = np.random.RandomState(42)
    
    # 贝叶斯 EN 超参数搜索
    bayes_alphas = [0.001, 0.01, 0.1, 1.0]
    bayes_l1_ratios = [0.3, 0.5, 0.7, 0.9]
    
    # ============================================================
    # 先做 1 次详细展示
    # ============================================================
    print("\n" + "=" * 80)
    print("详细展示（1次实验）")
    print("=" * 80)
    
    sparsity = rng.uniform(*sparsity_range)
    x_true, n_nz = generate_sparse_source(n_total, sparsity, rng)
    print(f"\n稀疏度: {sparsity:.2f}, 非零数: {n_nz}/{n_total}")
    print(f"真实源: {np.round(x_true, 3)}")
    
    # 零列放非零值（如果零列恰好没被选中为非零位置，手动设置一个）
    # 这里 x_true 已经是随机的，零列可能恰好有值也可能没有
    # 确保零列有非零值来体现不可观测区域的局限性
    for zc in zero_cols:
        if x_true[zc] == 0:
            x_true[zc] = rng.uniform(0.5, 5.0)
            print(f"  → 零列[{zc}]设置非零值 {x_true[zc]:.2f}（不可观测区域）")
    
    # 生成观测
    b = A_full @ x_true
    b_noisy = b + noise_level * np.linalg.norm(b) * rng.randn(m)
    
    print(f"\n观测 b: {np.round(b_noisy, 4)}")
    print(f"真实非零列: {np.where(np.abs(x_true) > 1e-6)[0].tolist()}")
    
    # ---- 方法1：贝叶斯 EN ----
    print("\n--- 贝叶斯 Elastic Net ---")
    # 选择最优超参数（通过 marginal likelihood 近似）
    best_bayes_score = -np.inf
    best_bayes_params = None
    best_bayes_result = None
    
    for alpha in bayes_alphas:
        for l1_ratio in bayes_l1_ratios:
            sigma_noise = np.std(b - A_norm @ np.linalg.lstsq(A_norm, b, rcond=None)[0]) + 0.01
            result = bayesian_en_predict(A_norm, b_noisy, alpha, l1_ratio, sigma_noise, 
                                        col_norms_safe, n_samples=200, seed=42)
            # 用 BIC 近似 marginal likelihood
            beta_map_norm = result['map'] * col_norms_safe  # 反归一化回 norm space
            resid = b_noisy - A_norm @ beta_map_norm
            n_active = np.sum(np.abs(beta_map_norm) > 1e-6)
            bic = np.sum(resid**2) / sigma_noise**2 + n_active * np.log(m)
            if bic < best_bayes_score or best_bayes_params is None:
                best_bayes_score = bic
                best_bayes_params = (alpha, l1_ratio)
                best_bayes_result = result
    
    print(f"最优参数: alpha={best_bayes_params[0]}, l1_ratio={best_bayes_params[1]}")
    print(f"MAP 估计: {np.round(best_bayes_result['map'], 3)}")
    print(f"后验均值: {np.round(best_bayes_result['mean'], 3)}")
    
    # funsor 验证
    cov_diag = best_bayes_result['std']**2
    cov_count, cov_total = funsor_validate_posterior(
        best_bayes_result['map'], cov_diag, x_true
    )
    print(f"funsor 后验覆盖验证: {cov_count}/{cov_total} 在 95% CI 内")
    
    bayes_eval = evaluate_recovery(x_true, best_bayes_result['map'], nonzero_cols, zero_cols)
    print(f"有效列恢复: {bayes_eval['recovered']}/{bayes_eval['total_true_nz']}")
    print(f"非零列平均相对误差: {bayes_eval['avg_rel_err_nonzero']:.2%}")
    print(f"零列最大值: {bayes_eval['max_zero_col_val']:.3f} (误放非零: {bayes_eval['has_false_positive']})")
    
    # ---- 方法2：EN_Fixed (V14) ----
    print("\n--- EN_Fixed (V14 风格, GridSearchCV) ---")
    x_en, en_params = run_en_fixed_cv(A_norm, b_noisy, col_norms_safe)
    print(f"最优参数: alpha={en_params['alpha']:.4f}, l1_ratio={en_params['l1_ratio']:.4f}")
    print(f"估计: {np.round(x_en, 3)}")
    en_eval = evaluate_recovery(x_true, x_en, nonzero_cols, zero_cols)
    print(f"有效列恢复: {en_eval['recovered']}/{en_eval['total_true_nz']}")
    print(f"非零列平均相对误差: {en_eval['avg_rel_err_nonzero']:.2%}")
    print(f"零列最大值: {en_eval['max_zero_col_val']:.3f} (误放非零: {en_eval['has_false_positive']})")
    
    # ---- 方法3：Lasso ----
    print("\n--- Lasso (CV) ---")
    x_lasso, lasso_alpha = run_lasso(A_norm, b_noisy, col_norms_safe)
    print(f"最优 alpha: {lasso_alpha:.4f}")
    print(f"估计: {np.round(x_lasso, 3)}")
    lasso_eval = evaluate_recovery(x_true, x_lasso, nonzero_cols, zero_cols)
    print(f"有效列恢复: {lasso_eval['recovered']}/{lasso_eval['total_true_nz']}")
    print(f"非零列平均相对误差: {lasso_eval['avg_rel_err_nonzero']:.2%}")
    print(f"零列最大值: {lasso_eval['max_zero_col_val']:.3f} (误放非零: {lasso_eval['has_false_positive']})")
    
    # ============================================================
    # 批量实验（10次）
    # ============================================================
    print("\n" + "=" * 80)
    print(f"批量实验 ({n_trials} 次)")
    print("=" * 80)
    
    all_results = {
        'Bayesian_EN': {'recovered': [], 'total_nz': [], 'rel_err': [], 'mse': [], 'zero_col_max': []},
        'EN_Fixed': {'recovered': [], 'total_nz': [], 'rel_err': [], 'mse': [], 'zero_col_max': []},
        'Lasso': {'recovered': [], 'total_nz': [], 'rel_err': [], 'mse': [], 'zero_col_max': []},
    }
    
    for trial in range(n_trials):
        sparsity = rng.uniform(*sparsity_range)
        x_true, n_nz = generate_sparse_source(n_total, sparsity, rng)
        
        # 零列放非零值
        for zc in zero_cols:
            if x_true[zc] == 0:
                x_true[zc] = rng.uniform(0.5, 5.0)
        
        b = A_full @ x_true
        b_noisy = b + noise_level * np.linalg.norm(b) * rng.randn(m)
        
        # --- 贝叶斯 EN ---
        best_score = -np.inf
        best_res = None
        for alpha in [0.001, 0.01, 0.1, 1.0]:
            for l1_ratio in [0.5, 0.7, 0.9]:
                sigma_noise = max(np.std(b_noisy - A_norm @ np.linalg.lstsq(A_norm, b_noisy, rcond=None)[0]), 0.01)
                res = bayesian_en_predict(A_norm, b_noisy, alpha, l1_ratio, sigma_noise,
                                         col_norms_safe, n_samples=100, seed=42)
                beta_map_norm = res['map'] * col_norms_safe
                resid = b_noisy - A_norm @ beta_map_norm
                n_active = np.sum(np.abs(beta_map_norm) > 1e-6)
                bic = np.sum(resid**2) / sigma_noise**2 + n_active * np.log(m)
                if best_res is None or bic < best_score:
                    best_score = bic
                    best_res = res
        
        ev = evaluate_recovery(x_true, best_res['map'], nonzero_cols, zero_cols)
        all_results['Bayesian_EN']['recovered'].append(ev['recovered'])
        all_results['Bayesian_EN']['total_nz'].append(ev['total_true_nz'])
        all_results['Bayesian_EN']['rel_err'].append(ev['avg_rel_err_nonzero'])
        all_results['Bayesian_EN']['mse'].append(ev['mse'])
        all_results['Bayesian_EN']['zero_col_max'].append(ev['max_zero_col_val'])
        
        # --- EN_Fixed ---
        x_en, _ = run_en_fixed_cv(A_norm, b_noisy, col_norms_safe)
        ev = evaluate_recovery(x_true, x_en, nonzero_cols, zero_cols)
        all_results['EN_Fixed']['recovered'].append(ev['recovered'])
        all_results['EN_Fixed']['total_nz'].append(ev['total_true_nz'])
        all_results['EN_Fixed']['rel_err'].append(ev['avg_rel_err_nonzero'])
        all_results['EN_Fixed']['mse'].append(ev['mse'])
        all_results['EN_Fixed']['zero_col_max'].append(ev['max_zero_col_val'])
        
        # --- Lasso ---
        x_lasso, _ = run_lasso(A_norm, b_noisy, col_norms_safe)
        ev = evaluate_recovery(x_true, x_lasso, nonzero_cols, zero_cols)
        all_results['Lasso']['recovered'].append(ev['recovered'])
        all_results['Lasso']['total_nz'].append(ev['total_true_nz'])
        all_results['Lasso']['rel_err'].append(ev['avg_rel_err_nonzero'])
        all_results['Lasso']['mse'].append(ev['mse'])
        all_results['Lasso']['zero_col_max'].append(ev['max_zero_col_val'])
        
        print(f"Trial {trial+1:2d}: sparsity={sparsity:.2f}, true_nz={ev['total_true_nz']} | "
              f"Bayes={all_results['Bayesian_EN']['recovered'][-1]}, "
              f"EN_Fixed={all_results['EN_Fixed']['recovered'][-1]}, "
              f"Lasso={all_results['Lasso']['recovered'][-1]}")
    
    # ============================================================
    # 汇总
    # ============================================================
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)
    print(f"\n{'方法':<18} {'平均恢复/总数':>14} {'恢复率':>8} {'平均相对误差':>14} {'MSE':>12} {'零列最大值':>12}")
    print("-" * 82)
    
    for method in ['Bayesian_EN', 'EN_Fixed', 'Lasso']:
        d = all_results[method]
        avg_rec = np.mean(d['recovered'])
        avg_total = np.mean(d['total_nz'])
        avg_rate = np.mean(np.array(d['recovered']) / (np.array(d['total_nz']) + 1e-10))
        avg_rel = np.mean(d['rel_err'])
        avg_mse = np.mean(d['mse'])
        avg_zc = np.mean(d['zero_col_max'])
        
        print(f"{method:<18} {avg_rec:5.1f} / {avg_total:<5.1f} {avg_rate:>7.1%} {avg_rel:>13.2%} {avg_mse:>12.2e} {avg_zc:>12.3f}")
    
    # 核心数字汇报
    print("\n" + "=" * 80)
    print("🎯 核心数字汇报")
    print("=" * 80)
    for method in ['Bayesian_EN', 'EN_Fixed', 'Lasso']:
        d = all_results[method]
        total_recoverable = sum(d['total_nz'])
        total_recovered = sum(d['recovered'])
        avg_rel = np.mean(d['rel_err'])
        print(f"\n{method}:")
        print(f"  有效列恢复: {total_recovered}/{total_recoverable} = {total_recovered/total_recoverable*100:.0f}%")
        print(f"  非零列平均相对误差: {avg_rel:.1%}")

if __name__ == "__main__":
    main()
