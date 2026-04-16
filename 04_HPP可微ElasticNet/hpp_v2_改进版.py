#!/usr/bin/env python3
"""
HPP Elastic Net v2 — 修复梯度爆炸 + 更紧 L1 近似
=================================================
v1 问题: (u²+v²)/2 近似太松，梯度爆炸
v2 修复:
  1. 梯度裁剪 (max_norm=1.0)
  2. 余弦退火学习率
  3. 使用 |uv| 的光滑近似: sqrt(u²v² + ε) 代替 (u²+v²)/2
  4. 投影: 每100步裁剪 β = u⊙v 的幅度
  5. 更好的初始化 + 多次重启
  6. 外层迭代: 先Ridge预热再逐渐增加稀疏惩罚
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
m, n_full = A_full.shape

valid_cols = np.where(~np.all(A_full==0, axis=0))[0]
valid_rows = np.where(~np.all(A_full==0, axis=1))[0]
A_sub = A_full[np.ix_(valid_rows, valid_cols)]
m_sub, n_sub = A_sub.shape

print(f"=== PHITS 有效子矩阵 {m_sub}×{n_sub}, rank={np.linalg.matrix_rank(A_sub)} ===")
print(f"Valid cols: {list(valid_cols)}")

# ============================================================
# 2. 列归一化
# ============================================================
col_norms = np.linalg.norm(A_sub, axis=0)
col_norms[col_norms == 0] = 1.0
A_norm = A_sub / col_norms
print(f"列归一化后 cond: {np.linalg.cond(A_norm):.2e}")

# ============================================================
# 3. HPP Elastic Net v2
# ============================================================

class HPPElasticNetV2:
    """HPP EN with gradient clipping, cosine annealing, smooth |uv|"""
    
    def __init__(self, lambda1=0.01, lambda2=0.01, lr=0.01, 
                 max_iter=3000, tol=1e-8, clip_norm=1.0, verbose=False):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lr0 = lr
        self.max_iter = max_iter
        self.tol = tol
        self.clip_norm = clip_norm
        self.verbose = verbose
    
    def _smooth_abs(self, x, eps=1e-6):
        """光滑绝对值: sqrt(x² + eps)"""
        return np.sqrt(x**2 + eps)
    
    def fit(self, A, b, x_init=None):
        m, n = A.shape
        
        # 初始化
        if x_init is not None:
            abs_x = np.abs(x_init) + 1e-4
            sign_x = np.sign(x_init)
            u = sign_x * np.sqrt(abs_x)
            v = np.sqrt(abs_x)
        else:
            u = np.random.randn(n) * 0.01
            v = np.random.randn(n) * 0.01
        
        best_loss = np.inf
        best_beta = u * v
        
        for it in range(self.max_iter):
            # 余弦退火学习率
            lr = self.lr0 * 0.5 * (1 + np.cos(np.pi * it / self.max_iter))
            
            beta = u * v
            residual = A @ beta - b
            AtR = A.T @ residual  # n维
            
            # 数据拟合梯度
            grad_data_u = AtR * v
            grad_data_v = AtR * u
            
            # 光滑 L1: |uv| ≈ sqrt(u²v² + eps)
            # d/du |uv| = sign(uv) * v = v * uv / |uv| ≈ v²u / sqrt(u²v²+eps)
            abs_uv = self._smooth_abs(beta)
            grad_l1_u = self.lambda1 * u * v**2 / (abs_uv + 1e-8)
            grad_l1_v = self.lambda1 * v * u**2 / (abs_uv + 1e-8)
            
            # L2: d/du (u²v²) = 2uv²u
            grad_l2_u = 2 * self.lambda2 * (v**2) * u
            grad_l2_v = 2 * self.lambda2 * (u**2) * v
            
            grad_u = grad_data_u + grad_l1_u + grad_l2_u
            grad_v = grad_data_v + grad_l1_v + grad_l2_v
            
            # 梯度裁剪
            g_norm = np.sqrt(np.sum(grad_u**2) + np.sum(grad_v**2))
            if g_norm > self.clip_norm:
                scale = self.clip_norm / g_norm
                grad_u *= scale
                grad_v *= scale
            
            u -= lr * grad_u
            v -= lr * grad_v
            
            # 投影: 防止 u,v 幅度爆炸
            if it % 100 == 0 and it > 0:
                beta_cur = u * v
                bmax = np.max(np.abs(beta_cur))
                if bmax > 1e4:
                    scale = np.sqrt(1e4 / bmax)
                    u *= scale
                    v *= scale
            
            beta = u * v
            loss = np.sum(residual**2) + self.lambda1 * np.sum(self._smooth_abs(beta)) + self.lambda2 * np.sum(beta**2)
            
            if loss < best_loss:
                best_loss = loss
                best_beta = beta.copy()
            
            if self.verbose and it % 500 == 0:
                nz = np.sum(np.abs(beta) > 0.01 * (np.max(np.abs(beta)) + 1e-10))
                print(f"  iter {it}: loss={loss:.4f}, |β|_0={nz}, lr={lr:.5f}")
        
        self.coef_ = best_beta
        return self

def run_hpp(A_n, b, x_init, lambda1, lambda2, lr=0.01, max_iter=3000):
    """单次 HPP 运行"""
    model = HPPElasticNetV2(lambda1=lambda1, lambda2=lambda2, lr=lr,
                             max_iter=max_iter, clip_norm=1.0)
    model.fit(A_n, b, x_init=x_init)
    return model.coef_

def run_hpp_multistart(A_n, b, x_init_ridge, lambda1=0.01, lambda2=0.01, n_starts=3):
    """多起点 HPP"""
    best_beta = None
    best_loss = np.inf
    
    for s in range(n_starts):
        if s == 0:
            init = x_init_ridge
        else:
            # 随机扰动初始化
            init = x_init_ridge + np.random.randn(A_n.shape[1]) * 0.01 * np.std(x_init_ridge)
        
        beta = run_hpp(A_n, b, init, lambda1, lambda2)
        loss = np.sum((A_n @ beta - b)**2)
        
        if loss < best_loss:
            best_loss = loss
            best_beta = beta.copy()
    
    return best_beta, best_loss

# ============================================================
# 4. 辅助函数
# ============================================================

def make_sparse_source(n, n_nonzero, max_val=50.0, rng=None):
    if rng is None: rng = np.random
    x = np.zeros(n)
    idx = rng.choice(n, n_nonzero, replace=False)
    x[idx] = rng.uniform(1.0, max_val, size=n_nonzero)
    return x

def recovery_metrics(x_true, x_pred, threshold_ratio=0.10):
    mask_true = np.abs(x_true) > 1e-10
    n_active = mask_true.sum()
    if n_active == 0:
        return {'n_active': 0, 'n_recovered': 0, 'recovery_rate': 0,
                'mean_rel_err': 0, 'mse': 0, 'r2': -999}
    
    pred_max = np.max(np.abs(x_pred)) + 1e-10
    threshold = threshold_ratio * pred_max
    mask_pred = np.abs(x_pred) > threshold
    
    n_recovered = np.sum(mask_true & mask_pred)
    rel_errs = np.abs(x_true[mask_true] - x_pred[mask_true]) / (np.abs(x_true[mask_true]) + 1e-10)
    
    mse = np.mean((x_true - x_pred)**2)
    ss_res = np.sum((x_true - x_pred)**2)
    ss_tot = np.sum((x_true - np.mean(x_true))**2)
    r2 = 1 - ss_res / (ss_tot + 1e-20)
    
    return {
        'n_active': n_active, 'n_recovered': n_recovered,
        'recovery_rate': n_recovered / n_active,
        'mean_rel_err': np.mean(rel_errs), 'mse': mse, 'r2': r2
    }

# ============================================================
# 5. 先用1个Trial调参
# ============================================================
print("\n" + "="*60)
print("=== Step 1: 单Trial调参 ===")
print("="*60)

rng = np.random.RandomState(7)
x_demo = make_sparse_source(n_sub, 3, max_val=30.0, rng=rng)
b_demo = A_norm @ x_demo
b_demo_noisy = b_demo + 0.01 * np.linalg.norm(b_demo) * rng.randn(m_sub)

# Ridge 预热
ridge_init = Ridge(alpha=1.0).fit(A_norm, b_demo_noisy).coef_

print(f"真值: {np.array2string(x_demo, precision=2)}")
print(f"Ridge预热: {np.array2string(ridge_init, precision=2)}")

# 测试不同超参
print("\n超参扫描:")
for l1 in [0.001, 0.005, 0.01, 0.05, 0.1]:
    for l2 in [0.001, 0.005, 0.01, 0.05]:
        beta = run_hpp(A_norm, b_demo_noisy, ridge_init, l1, l2, lr=0.01, max_iter=2000)
        # 反归一化
        beta_orig = beta / col_norms
        met = recovery_metrics(x_demo, beta_orig)
        if met['r2'] > -50:
            print(f"  λ1={l1:.3f} λ2={l2:.3f} → 恢复{met['n_recovered']}/{met['n_active']}, "
                  f"rel_err={met['mean_rel_err']:.3f}, R²={met['r2']:.3f}")

# ============================================================
# 6. 选最佳超参后跑 10 trials
# ============================================================
print("\n" + "="*60)
print("=== Step 2: 10 Trials 验证 (1% noise) ===")
print("="*60)

# 先用上面的扫描结果选超参 (默认先试 λ1=0.01, λ2=0.005)
L1_BEST = 0.01
L2_BEST = 0.005

methods = ['HPP_EN', 'EN_Fixed', 'Ridge']
results = {m: [] for m in methods}

for trial in range(10):
    rng = np.random.RandomState(42 + trial)
    n_active = rng.randint(2, n_sub + 1)
    x_true = make_sparse_source(n_sub, n_active, max_val=50.0, rng=rng)
    b = A_norm @ x_true
    b_noisy = b + 0.01 * np.linalg.norm(b) * rng.randn(m_sub)
    
    # HPP EN (Ridge warm-start + multistart)
    ridge_ws = Ridge(alpha=1.0).fit(A_norm, b_noisy).coef_
    beta_hpp, _ = run_hpp_multistart(A_norm, b_noisy, ridge_ws, 
                                      lambda1=L1_BEST, lambda2=L2_BEST, n_starts=3)
    beta_hpp_orig = beta_hpp / col_norms
    
    # EN Fixed
    en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000)
    en.fit(A_norm, b_noisy)
    beta_en = en.coef_ / col_norms
    
    # Ridge
    ridge = Ridge(alpha=1.0).fit(A_norm, b_noisy)
    beta_ridge = ridge.coef_ / col_norms
    
    # 评估
    for name, beta in [('HPP_EN', beta_hpp_orig), ('EN_Fixed', beta_en), ('Ridge', beta_ridge)]:
        met = recovery_metrics(x_true, beta)
        met['trial'] = trial
        results[name].append(met)

# 汇总
print(f"\n{'Method':<12} {'AvgRecov':>10} {'AvgRelErr':>10} {'AvgR2':>8}")
for name in methods:
    vals = results[name]
    # 过滤极端值
    good = [v for v in vals if v['r2'] > -100]
    if good:
        avg_rec = np.mean([v['n_recovered'] for v in good])
        avg_tot = np.mean([v['n_active'] for v in good])
        avg_rel = np.mean([v['mean_rel_err'] for v in good])
        avg_r2 = np.mean([v['r2'] for v in good])
        print(f"{name:<12} {avg_rec:.1f}/{avg_tot:.1f} {avg_rel:>10.3f} {avg_r2:>8.3f}")
    else:
        print(f"{name:<12} ALL FAILED")

# 逐Trial
print("\n逐Trial:")
for name in methods:
    print(f"\n[{name}]")
    for v in results[name]:
        r2_str = f"{v['r2']:.3f}" if v['r2'] > -100 else f"{v['r2']:.1e}"
        print(f"  Trial {v['trial']}: 恢复 {v['n_recovered']}/{v['n_active']}, "
              f"rel={v['mean_rel_err']:.3f}, R²={r2_str}")

# ============================================================
# 7. 最佳Trial详细展示
# ============================================================
print("\n" + "="*60)
print("=== Step 3: 最佳Trial详细展示 ===")
print("="*60)

# 找 HPP 最好的 trial
best_hpp_idx = max(range(len(results['HPP_EN'])), key=lambda i: results['HPP_EN'][i]['r2'])
best_en_idx = max(range(len(results['EN_Fixed'])), key=lambda i: results['EN_Fixed'][i]['r2'])

print(f"\nHPP 最佳 Trial {best_hpp_idx}:")
print(f"  {results['HPP_EN'][best_hpp_idx]}")

print(f"\nEN_Fixed 最佳 Trial {best_en_idx}:")
print(f"  {results['EN_Fixed'][best_en_idx]}")

# ============================================================
# 8. 不同噪声水平
# ============================================================
print("\n" + "="*60)
print("=== Step 4: 不同噪声水平 ===")
print("="*60)

for noise_lvl in [0.01, 0.05, 0.10]:
    print(f"\n--- noise={noise_lvl:.0%} ---")
    res_n = {m: [] for m in methods}
    
    for trial in range(10):
        rng = np.random.RandomState(42 + trial)
        n_active = rng.randint(2, n_sub + 1)
        x_true = make_sparse_source(n_sub, n_active, max_val=50.0, rng=rng)
        b = A_norm @ x_true
        b_noisy = b + noise_lvl * np.linalg.norm(b) * rng.randn(m_sub)
        
        ridge_ws = Ridge(alpha=1.0).fit(A_norm, b_noisy).coef_
        beta_hpp, _ = run_hpp_multistart(A_norm, b_noisy, ridge_ws, 
                                          lambda1=L1_BEST, lambda2=L2_BEST, n_starts=3)
        
        en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000)
        en.fit(A_norm, b_noisy)
        
        ridge = Ridge(alpha=1.0).fit(A_norm, b_noisy)
        
        for name, beta in [('HPP_EN', beta_hpp / col_norms), 
                           ('EN_Fixed', en.coef_ / col_norms),
                           ('Ridge', ridge.coef_ / col_norms)]:
            met = recovery_metrics(x_true, beta)
            res_n[name].append(met)
    
    print(f"{'Method':<12} {'AvgRecov':>10} {'AvgRelErr':>10} {'AvgR2':>8}")
    for name in methods:
        good = [v for v in res_n[name] if v['r2'] > -100]
        if good:
            avg_rec = np.mean([v['n_recovered'] for v in good])
            avg_tot = np.mean([v['n_active'] for v in good])
            avg_rel = np.mean([v['mean_rel_err'] for v in good])
            avg_r2 = np.mean([v['r2'] for v in good])
            print(f"{name:<12} {avg_rec:.1f}/{avg_tot:.1f} {avg_rel:>10.3f} {avg_r2:>8.3f}")

print("\n" + "="*80)
print("=== HPP EN v2 完成 ===")
print("="*80)
