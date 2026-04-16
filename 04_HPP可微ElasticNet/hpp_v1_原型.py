#!/usr/bin/env python3
"""
HPP (Hadamard Overparametrization) Elastic Net Prototype
=======================================================
核心思想: β = u ⊙ v，通过梯度下降同时优化 u, v
- L1 正则化通过 |u⊙v| ≈ Σ(u²ᵢ+v²ᵢ)/2 自然近似
- L2 正则化 ||u⊙v||² 是光滑的
- 完全可微，端到端训练

PHITS 8×12 数据验证:
- 有效列 [1,5,6,7,8,9,10]（7列），零列 [0,2,3,4,11]
- 列归一化为核心策略

对比: HPP EN vs EN_Fixed vs 贝叶斯 EN
"""

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 1. 加载 PHITS 数据
# ============================================================
DATA_PATH = '/root/.openclaw/workspace/elasticnet/phits_data/phits_10x8_data/PHITS_results_10x8/A_matrix_8x12_GBq.npy'
A_full = np.load(DATA_PATH)
m, n = A_full.shape
print(f"=== PHITS A_matrix_8x12_GBq ===")
print(f"Shape: {m}×{n}")
print(f"Zero cols: {list(np.where(np.all(A_full==0, axis=0))[0])}")
print(f"Zero rows: {list(np.where(np.all(A_full==0, axis=1))[0])}")

# 提取有效子矩阵
valid_cols = np.where(~np.all(A_full==0, axis=0))[0]
valid_rows = np.where(~np.all(A_full==0, axis=1))[0]
A_sub = A_full[np.ix_(valid_rows, valid_cols)]
print(f"有效子矩阵: {A_sub.shape}, rank={np.linalg.matrix_rank(A_sub)}")
print(f"Valid columns: {list(valid_cols)} (7 columns: 1,5,6,7,8,9,10)")
print()

# ============================================================
# 2. HPP Elastic Net 核心实现
# ============================================================

class HPPElasticNet:
    """Hadamard Overparametrized Elastic Net
    
    β = u ⊙ v
    
    Loss = ||A(u⊙v) - b||² + λ₁||u⊙v||₁ + λ₂||u⊙v||₂²
    
    L1 近似: |u⊙v| ≈ (u²+v²)/2 ( Hoffman et al. )
    实际中用更紧的: |uv| ≈ sqrt(u²v²) = |u||v|, 梯度可用
    这里直接用 u²+v² 的 l1 近似
    
    梯度:
    ∂L/∂u = diag(v) @ A.T @ (A(u⊙v) - b) + λ₁*u + 2*λ₂*(u⊙v⊙v)
    ∂L/∂v = diag(u) @ A.T @ (A(u⊙v) - b) + λ₁*v + 2*λ₂*(u⊙u⊙v)
    """
    
    def __init__(self, lambda1=0.01, lambda2=0.01, lr=0.001, 
                 max_iter=2000, tol=1e-6, verbose=False):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
    
    def fit(self, A, b, x_init=None):
        m, n = A.shape
        
        # 列归一化
        col_norms = np.linalg.norm(A, axis=0)
        col_norms_safe = np.where(col_norms == 0, 1.0, col_norms)
        A_norm = A / col_norms_safe
        
        # 初始化 u, v
        if x_init is not None:
            # 用初始解拆分
            abs_x = np.abs(x_init) + 1e-8
            sign_x = np.sign(x_init)
            u = sign_x * np.sqrt(abs_x)
            v = np.sqrt(abs_x)
        else:
            u = np.random.randn(n) * 0.1
            v = np.random.randn(n) * 0.1
        
        # Adam 优化器参数
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mu_u = np.zeros(n)
        mu_v = np.zeros(n)
        nu_u = np.zeros(n)
        nu_v = np.zeros(n)
        
        prev_loss = np.inf
        best_loss = np.inf
        best_beta = u * v
        
        for it in range(self.max_iter):
            beta = u * v
            residual = A_norm @ beta - b
            
            # 数据拟合梯度
            grad_data_u = (A_norm.T @ residual) * v
            grad_data_v = (A_norm.T @ residual) * u
            
            # L1 近似梯度: |uv| ≈ (u²+v²)/2 → ∂/∂u = u
            grad_l1_u = self.lambda1 * u
            grad_l1_v = self.lambda1 * v
            
            # L2 惩罚梯度: ||uv||² → ∂/∂u = 2*v²*u
            grad_l2_u = 2 * self.lambda2 * (v**2) * u
            grad_l2_v = 2 * self.lambda2 * (u**2) * v
            
            grad_u = grad_data_u + grad_l1_u + grad_l2_u
            grad_v = grad_data_v + grad_l1_v + grad_l2_v
            
            # Adam update
            mu_u = beta1 * mu_u + (1 - beta1) * grad_u
            mu_v = beta1 * mu_v + (1 - beta1) * grad_v
            nu_u = beta2 * nu_u + (1 - beta2) * grad_u**2
            nu_v = beta2 * nu_v + (1 - beta2) * grad_v**2
            
            mu_u_hat = mu_u / (1 - beta1**(it+1))
            mu_v_hat = mu_v / (1 - beta1**(it+1))
            nu_u_hat = nu_u / (1 - beta2**(it+1))
            nu_v_hat = nu_v / (1 - beta2**(it+1))
            
            u = u - self.lr * mu_u_hat / (np.sqrt(nu_u_hat) + eps)
            v = v - self.lr * mu_v_hat / (np.sqrt(nu_v_hat) + eps)
            
            # 计算loss
            beta = u * v
            data_loss = np.sum(residual**2)
            l1_loss = self.lambda1 * np.sum(np.abs(beta))
            l2_loss = self.lambda2 * np.sum(beta**2)
            loss = data_loss + l1_loss + l2_loss
            
            if loss < best_loss:
                best_loss = loss
                best_beta = beta.copy()
            
            if it > 0 and abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"  Converged at iter {it}, loss={loss:.6f}")
                break
            prev_loss = loss
            
            if self.verbose and it % 500 == 0:
                print(f"  iter {it}: loss={loss:.4f}, |β|_0={np.sum(np.abs(beta) > 0.01 * np.max(np.abs(beta)))}")
        
        # 反归一化
        self.coef_ = best_beta / col_norms_safe
        self.u_ = u
        self.v_ = v
        return self
    
    def predict(self, A):
        return A @ self.coef_


class HPPElasticNetMultiStart(HPPElasticNet):
    """HPP EN with multiple random starts + Ridge warm-start"""
    
    def fit(self, A, b, n_starts=5):
        m, n = A.shape
        
        # 列归一化
        col_norms = np.linalg.norm(A, axis=0)
        col_norms_safe = np.where(col_norms == 0, 1.0, col_norms)
        A_norm = A / col_norms_safe
        
        best_coef = None
        best_loss = np.inf
        
        # Start 1: Ridge warm-start
        ridge = Ridge(alpha=1.0)
        ridge.fit(A_norm, b)
        x_init = ridge.coef_
        
        for start_idx in range(n_starts):
            if start_idx == 0:
                init = x_init
            else:
                init = None  # random init
            
            super().fit(A, b, x_init=init)
            
            # Evaluate
            beta_norm = self.coef_ * col_norms_safe
            residual = A_norm @ beta_norm - b
            loss = np.sum(residual**2)
            
            if loss < best_loss:
                best_loss = loss
                best_coef = self.coef_.copy()
        
        self.coef_ = best_coef
        return self


# ============================================================
# 3. 辅助函数
# ============================================================

def make_sparse_source(n, n_nonzero, max_val=50.0, rng=None):
    if rng is None:
        rng = np.random
    x = np.zeros(n)
    idx = rng.choice(n, n_nonzero, replace=False)
    x[idx] = rng.uniform(1.0, max_val, size=n_nonzero)
    return x

def recovery_metrics(x_true, x_pred, threshold_ratio=0.05):
    """核心指标：恢复了几列？平均相对误差？"""
    mask_true = np.abs(x_true) > 1e-10
    n_active = mask_true.sum()
    
    if n_active == 0:
        return {'n_active': 0, 'n_recovered': 0, 'recovery_rate': 0,
                'mean_rel_err': 0, 'mse': 0, 'r2': -999}
    
    # 检测恢复：真值非零列中，预测也非零
    pred_max = np.max(np.abs(x_pred)) + 1e-10
    threshold = threshold_ratio * pred_max
    mask_pred = np.abs(x_pred) > threshold
    
    # 恢复计数
    n_recovered = np.sum(mask_true & mask_pred)
    
    # 非零列的相对误差
    rel_errs = np.abs(x_true[mask_true] - x_pred[mask_true]) / (np.abs(x_true[mask_true]) + 1e-10)
    mean_rel_err = np.mean(rel_errs)
    
    # MSE and R²
    mse = np.mean((x_true - x_pred)**2)
    ss_res = np.sum((x_true - x_pred)**2)
    ss_tot = np.sum((x_true - np.mean(x_true))**2)
    r2 = 1 - ss_res / (ss_tot + 1e-20)
    
    return {
        'n_active': n_active,
        'n_recovered': n_recovered,
        'recovery_rate': n_recovered / n_active,
        'mean_rel_err': mean_rel_err,
        'mse': mse,
        'r2': r2,
        'rel_errs_per_col': rel_errs
    }

# ============================================================
# 4. HPP EN 超参数扫描 (小规模)
# ============================================================

def hpp_grid_search(A, b, x_true):
    """在有效子矩阵上做小规模超参搜索"""
    lambda1_grid = [0.001, 0.01, 0.05, 0.1]
    lambda2_grid = [0.001, 0.01, 0.05, 0.1]
    lr_grid = [0.001, 0.005, 0.01]
    
    best_coef = None
    best_score = -np.inf
    best_params = None
    
    for l1 in lambda1_grid:
        for l2 in lambda2_grid:
            for lr in lr_grid:
                try:
                    model = HPPElasticNet(lambda1=l1, lambda2=l2, lr=lr, 
                                          max_iter=1000, tol=1e-6)
                    model.fit(A, b)
                    score = -np.mean((x_true - model.coef_)**2)
                    if score > best_score:
                        best_score = score
                        best_coef = model.coef_.copy()
                        best_params = (l1, l2, lr)
                except:
                    pass
    
    return best_coef, best_params

# ============================================================
# 5. 主实验：PHITS 有效子矩阵验证
# ============================================================

print("="*80)
print("=== HPP Elastic Net 原型验证 ===")
print("="*80)

m_sub, n_sub = A_sub.shape
print(f"\n有效子矩阵: {m_sub}×{n_sub}")
print(f"有效列索引: {list(valid_cols)} → 7列中恢复几列？\n")

# 小规模先验证 (5 trials)
noise_levels = [0.01, 0.05, 0.10]
N_TRIALS = 10  # 先小量

results = {method: {noise: [] for noise in noise_levels} 
           for method in ['HPP_EN', 'EN_Fixed', 'Ridge']}

for noise_lvl in noise_levels:
    print(f"\n{'='*60}")
    print(f"  noise = {noise_lvl:.0%}")
    print(f"{'='*60}")
    
    for trial in range(N_TRIALS):
        rng = np.random.RandomState(42 + trial)
        n_active = rng.randint(2, n_sub + 1)
        x_true = make_sparse_source(n_sub, n_active, max_val=50.0, rng=rng)
        b = A_sub @ x_true
        b_noisy = b + noise_lvl * np.linalg.norm(b) * rng.randn(m_sub)
        
        # ---- HPP EN ----
        try:
            # Ridge warm-start
            col_norms = np.linalg.norm(A_sub, axis=0)
            col_norms[col_norms == 0] = 1.0
            A_norm = A_sub / col_norms
            ridge_init = Ridge(alpha=1.0).fit(A_norm, b_noisy).coef_
            
            model_hpp = HPPElasticNet(lambda1=0.01, lambda2=0.01, lr=0.005,
                                       max_iter=2000, tol=1e-6)
            model_hpp.fit(A_sub, b_noisy, x_init=ridge_init)
            met = recovery_metrics(x_true, model_hpp.coef_)
            results['HPP_EN'][noise_lvl].append(met)
        except Exception as e:
            results['HPP_EN'][noise_lvl].append(
                {'n_active': n_active, 'n_recovered': 0, 'recovery_rate': 0,
                 'mean_rel_err': 999, 'mse': 1e30, 'r2': -999})
        
        # ---- EN Fixed ----
        try:
            col_norms = np.linalg.norm(A_sub, axis=0)
            col_norms[col_norms == 0] = 1.0
            A_norm = A_sub / col_norms
            en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000)
            en.fit(A_norm, b_noisy)
            coef_en = en.coef_ / col_norms
            met = recovery_metrics(x_true, coef_en)
            results['EN_Fixed'][noise_lvl].append(met)
        except:
            results['EN_Fixed'][noise_lvl].append(
                {'n_active': n_active, 'n_recovered': 0, 'recovery_rate': 0,
                 'mean_rel_err': 999, 'mse': 1e30, 'r2': -999})
        
        # ---- Ridge ----
        try:
            col_norms = np.linalg.norm(A_sub, axis=0)
            col_norms[col_norms == 0] = 1.0
            A_norm = A_sub / col_norms
            ridge = Ridge(alpha=1.0)
            ridge.fit(A_norm, b_noisy)
            coef_ridge = ridge.coef_ / col_norms
            met = recovery_metrics(x_true, coef_ridge)
            results['Ridge'][noise_lvl].append(met)
        except:
            results['Ridge'][noise_lvl].append(
                {'n_active': n_active, 'n_recovered': 0, 'recovery_rate': 0,
                 'mean_rel_err': 999, 'mse': 1e30, 'r2': -999})

# ============================================================
# 6. 结果汇总
# ============================================================

print("\n" + "="*80)
print("=== 结果汇总 ===")
print("="*80)

for noise_lvl in noise_levels:
    print(f"\n--- noise = {noise_lvl:.0%} ---")
    print(f"{'Method':<12} {'AvgRecov':>8} {'AvgRelErr':>10} {'AvgR2':>8} {'AvgMSE':>12}")
    
    for method in ['HPP_EN', 'EN_Fixed', 'Ridge']:
        vals = results[method][noise_lvl]
        if not vals:
            continue
        
        avg_recov = np.mean([v['n_recovered'] for v in vals])
        avg_total = np.mean([v['n_active'] for v in vals])
        avg_rel = np.mean([v['mean_rel_err'] for v in vals if v['r2'] > -100])
        avg_r2 = np.mean([v['r2'] for v in vals if v['r2'] > -100])
        avg_mse = np.mean([v['mse'] for v in vals if v['r2'] > -100])
        
        recov_str = f"{avg_recov:.1f}/{avg_total:.1f}"
        print(f"{method:<12} {recov_str:>8} {avg_rel:>10.3f} {avg_r2:>8.3f} {avg_mse:>12.2f}")

# ============================================================
# 7. 逐Trial详情（noise=1%）
# ============================================================
print("\n" + "="*80)
print("=== 逐Trial详情 (noise=1%) ===")
print("="*80)

noise_lvl = 0.01
for method in ['HPP_EN', 'EN_Fixed', 'Ridge']:
    vals = results[method][noise_lvl]
    print(f"\n[{method}]")
    for i, v in enumerate(vals):
        recov = f"{v['n_recovered']}/{v['n_active']}"
        print(f"  Trial {i}: 恢复 {recov} 列, rel_err={v['mean_rel_err']:.3f}, R²={v['r2']:.3f}")

# ============================================================
# 8. 超参数敏感度（HPP专用）
# ============================================================
print("\n" + "="*80)
print("=== HPP 超参数敏感度 (1 trial, noise=1%) ===")
print("="*80)

rng = np.random.RandomState(99)
n_active = 3
x_true_sens = make_sparse_source(n_sub, n_active, max_val=50.0, rng=rng)
b_sens = A_sub @ x_true_sens
b_sens_noisy = b_sens + 0.01 * np.linalg.norm(b_sens) * rng.randn(m_sub)

for l1 in [0.001, 0.01, 0.1]:
    for l2 in [0.001, 0.01, 0.1]:
        for lr in [0.001, 0.005, 0.01]:
            model = HPPElasticNet(lambda1=l1, lambda2=l2, lr=lr,
                                   max_iter=1500, tol=1e-6)
            col_norms = np.linalg.norm(A_sub, axis=0)
            col_norms[col_norms == 0] = 1.0
            A_norm = A_sub / col_norms
            ridge_init = Ridge(alpha=1.0).fit(A_norm, b_sens_noisy).coef_
            model.fit(A_sub, b_sens_noisy, x_init=ridge_init)
            met = recovery_metrics(x_true_sens, model.coef_)
            recov = f"{met['n_recovered']}/{met['n_active']}"
            if met['r2'] > -10:  # 只打印合理结果
                print(f"  λ1={l1:.3f} λ2={l2:.3f} lr={lr:.3f} → 恢复{recov}, "
                      f"rel_err={met['mean_rel_err']:.3f}, R²={met['r2']:.3f}")

# ============================================================
# 9. 单Trial详细展示
# ============================================================
print("\n" + "="*80)
print("=== 单Trial详细展示 ===")
print("="*80)

rng = np.random.RandomState(7)
x_demo = make_sparse_source(n_sub, 3, max_val=30.0, rng=rng)
b_demo = A_sub @ x_demo
b_demo_noisy = b_demo + 0.01 * np.linalg.norm(b_demo) * rng.randn(m_sub)

# 列归一化
col_norms = np.linalg.norm(A_sub, axis=0)
col_norms[col_norms == 0] = 1.0
A_norm = A_sub / col_norms

# HPP
ridge_init = Ridge(alpha=1.0).fit(A_norm, b_demo_noisy).coef_
model_hpp = HPPElasticNet(lambda1=0.01, lambda2=0.01, lr=0.005,
                           max_iter=2000, verbose=True)
model_hpp.fit(A_sub, b_demo_noisy, x_init=ridge_init)

# EN Fixed
en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000)
en.fit(A_norm, b_demo_noisy)
coef_en = en.coef_ / col_norms

print(f"\n真值:      {np.array2string(x_demo, precision=2)}")
print(f"HPP EN:    {np.array2string(model_hpp.coef_, precision=2)}")
print(f"EN Fixed:  {np.array2string(coef_en, precision=2)}")

met_hpp = recovery_metrics(x_demo, model_hpp.coef_)
met_en = recovery_metrics(x_demo, coef_en)

print(f"\nHPP EN:  恢复 {met_hpp['n_recovered']}/{met_hpp['n_active']} 列, "
      f"rel_err={met_hpp['mean_rel_err']:.3f}, R²={met_hpp['r2']:.3f}")
print(f"EN Fixed: 恢复 {met_en['n_recovered']}/{met_en['n_active']} 列, "
      f"rel_err={met_en['mean_rel_err']:.3f}, R²={met_en['r2']:.3f}")

# 逐列对比
print(f"\n逐列相对误差:")
for j in range(n_sub):
    if abs(x_demo[j]) > 1e-10:
        rel_hpp = abs(model_hpp.coef_[j] - x_demo[j]) / (abs(x_demo[j]) + 1e-10)
        rel_en = abs(coef_en[j] - x_demo[j]) / (abs(x_demo[j]) + 1e-10)
        detect_hpp = "✓" if abs(model_hpp.coef_[j]) > 0.05 * max(abs(model_hpp.coef_.max()), 1e-10) else "✗"
        detect_en = "✓" if abs(coef_en[j]) > 0.05 * max(abs(coef_en.max()), 1e-10) else "✗"
        print(f"  col {j}: 真值={x_demo[j]:6.2f} | HPP={model_hpp.coef_[j]:7.2f}({detect_hpp}, err={rel_hpp:.2f}) | "
              f"EN={coef_en[j]:7.2f}({detect_en}, err={rel_en:.2f})")

print("\n" + "="*80)
print("=== 完成 ===")
print("="*80)
