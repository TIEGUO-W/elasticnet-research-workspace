#!/usr/bin/env python3
"""
HPP Elastic Net v3 — 双轨验证
================================
Track 1: PHITS 真实数据 (cond=1e21) → 所有方法失败（预期）
Track 2: 合成数据 Cond~300 (论文场景) → HPP vs EN_Fixed 对比

核心改进:
  1. HPP 迭代 curriculum: 先跑无稀疏约束，再逐渐加 λ1
  2. 多起点 + 梯度裁剪
  3. 在 Cond~300 上展示 HPP 优势
"""

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge, Lasso
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# HPP Elastic Net V3 — Curriculum Training
# ============================================================

class HPPElasticNetV3:
    """HPP EN with curriculum: warmup → sparse transition → fine-tune"""
    
    def __init__(self, lambda1=0.01, lambda2=0.01, lr=0.01, 
                 max_iter=3000, clip_norm=1.0, verbose=False):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lr0 = lr
        self.max_iter = max_iter
        self.clip_norm = clip_norm
        self.verbose = verbose
    
    def fit(self, A, b, x_init=None):
        m, n = A.shape
        
        if x_init is not None:
            abs_x = np.abs(x_init) + 1e-4
            sgn = np.sign(x_init)
            u = sgn * np.sqrt(abs_x)
            v = np.sqrt(abs_x)
        else:
            u = np.random.randn(n) * 0.01
            v = np.random.randn(n) * 0.01
        
        best_loss = np.inf
        best_beta = u * v
        
        # Curriculum: warmup(0-30%) → ramp(30-60%) → finetune(60-100%)
        warmup_end = int(self.max_iter * 0.3)
        ramp_end = int(self.max_iter * 0.6)
        
        for it in range(self.max_iter):
            # 余弦退火
            lr = self.lr0 * 0.5 * (1 + np.cos(np.pi * it / self.max_iter))
            
            # Curriculum λ1
            if it < warmup_end:
                l1_cur = 0.0  # 无稀疏约束预热
            elif it < ramp_end:
                progress = (it - warmup_end) / (ramp_end - warmup_end)
                l1_cur = self.lambda1 * progress  # 线性增加
            else:
                l1_cur = self.lambda1
            
            beta = u * v
            residual = A @ beta - b
            AtR = A.T @ residual
            
            grad_u = AtR * v
            grad_v = AtR * u
            
            # L1 smooth approximation
            if l1_cur > 0:
                abs_uv = np.sqrt(beta**2 + 1e-6)
                grad_u += l1_cur * u * v**2 / (abs_uv + 1e-8)
                grad_v += l1_cur * v * u**2 / (abs_uv + 1e-8)
            
            # L2
            grad_u += 2 * self.lambda2 * (v**2) * u
            grad_v += 2 * self.lambda2 * (u**2) * v
            
            # 裁剪
            g_norm = np.sqrt(np.sum(grad_u**2) + np.sum(grad_v**2))
            if g_norm > self.clip_norm:
                s = self.clip_norm / g_norm
                grad_u *= s
                grad_v *= s
            
            u -= lr * grad_u
            v -= lr * grad_v
            
            # 投影
            if it % 200 == 199:
                beta_cur = u * v
                bmax = np.max(np.abs(beta_cur))
                if bmax > 1e3:
                    scale = np.sqrt(1e3 / bmax)
                    u *= scale
                    v *= scale
            
            beta = u * v
            loss = np.sum(residual**2) + l1_cur * np.sum(np.abs(beta)) + self.lambda2 * np.sum(beta**2)
            
            if loss < best_loss:
                best_loss = loss
                best_beta = beta.copy()
            
            if self.verbose and it % 500 == 0:
                nz = np.sum(np.abs(beta) > 0.05 * (np.max(np.abs(beta)) + 1e-10))
                print(f"  it={it}: loss={loss:.4f}, |β|_0≈{nz}, λ1_cur={l1_cur:.4f}")
        
        self.coef_ = best_beta
        return self

def run_hpp_ms(A, b, ridge_init, l1, l2, n_starts=3, max_iter=3000, lr=0.01):
    best_beta, best_loss = None, np.inf
    n = A.shape[1]
    for s in range(n_starts):
        init = ridge_init if s == 0 else ridge_init + np.random.randn(n) * 0.01 * np.std(ridge_init)
        model = HPPElasticNetV3(lambda1=l1, lambda2=l2, lr=lr, max_iter=max_iter, clip_norm=1.0)
        model.fit(A, b, x_init=init)
        loss = np.sum((A @ model.coef_ - b)**2)
        if loss < best_loss:
            best_loss = loss
            best_beta = model.coef_.copy()
    return best_beta

# ============================================================
# 辅助函数
# ============================================================

def make_sparse(n, k, max_val=10.0, rng=None):
    if rng is None: rng = np.random
    x = np.zeros(n)
    idx = rng.choice(n, k, replace=False)
    x[idx] = rng.uniform(0.5, max_val, size=k)
    return x

def make_cond_matrix(m, n, cond_target, rng):
    """构造指定条件数的矩阵"""
    log_c = np.log10(cond_target)
    k = min(m, n)
    sv = np.logspace(0, log_c, k)
    U, _ = np.linalg.qr(rng.standard_normal((m, m)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    S = np.zeros((m, n))
    np.fill_diagonal(S, sv)
    A = U @ S @ V.T
    # 列归一化
    cn = np.linalg.norm(A, axis=0)
    A = A / cn
    return A, cn

def metrics(x_true, x_pred):
    mask = np.abs(x_true) > 1e-10
    n_active = mask.sum()
    if n_active == 0:
        return {'rec': '0/0', 'rel': 0, 'r2': -999, 'mse': 0}
    
    pred_max = np.max(np.abs(x_pred)) + 1e-10
    mask_pred = np.abs(x_pred) > 0.10 * pred_max
    n_rec = np.sum(mask & mask_pred)
    
    rel = np.mean(np.abs(x_true[mask] - x_pred[mask]) / (np.abs(x_true[mask]) + 1e-10))
    mse = np.mean((x_true - x_pred)**2)
    ss_r = np.sum((x_true - x_pred)**2)
    ss_t = np.sum((x_true - np.mean(x_true))**2)
    r2 = 1 - ss_r / (ss_t + 1e-20)
    
    return {'rec': f'{n_rec}/{n_active}', 'rel': rel, 'r2': r2, 'mse': mse}

# ============================================================
# Track 1: PHITS 数据
# ============================================================
print("="*80)
print("Track 1: PHITS 8×12 (cond~1e21)")
print("="*80)

DATA_PATH = '/root/.openclaw/workspace/elasticnet/phits_data/phits_10x8_data/PHITS_results_10x8/A_matrix_8x12_GBq.npy'
A_full = np.load(DATA_PATH)
valid_cols = np.where(~np.all(A_full==0, axis=0))[0]
valid_rows = np.where(~np.all(A_full==0, axis=1))[0]
A_sub = A_full[np.ix_(valid_rows, valid_cols)]

cn = np.linalg.norm(A_sub, axis=0)
cn[cn == 0] = 1.0
A_n = A_sub / cn

print(f"有效子矩阵 {A_sub.shape}, rank={np.linalg.matrix_rank(A_sub)}, cond={np.linalg.cond(A_n):.1e}")
print("→ 预期：所有方法均失败（cond=1e21 远超论文 Cond~300 场景）\n")

rng = np.random.RandomState(7)
x_phits = make_sparse(A_sub.shape[1], 3, 30.0, rng)
b_phits = A_n @ x_phits
b_phits_noisy = b_phits + 0.01 * np.linalg.norm(b_phits) * rng.randn(A_sub.shape[0])

ridge_init = Ridge(alpha=1.0).fit(A_n, b_phits_noisy).coef_
beta_hpp = run_hpp_ms(A_n, b_phits_noisy, ridge_init, 0.01, 0.01, n_starts=3)
en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000).fit(A_n, b_phits_noisy)
ridge = Ridge(alpha=1.0).fit(A_n, b_phits_noisy)

print(f"真值:     {np.array2string(x_phits, precision=2)}")
print(f"HPP EN:   {np.array2string(beta_hpp / cn, precision=2)}")
print(f"EN Fixed: {np.array2string(en.coef_ / cn, precision=2)}")
print(f"Ridge:    {np.array2string(ridge.coef_ / cn, precision=2)}")
for name, coef in [('HPP', beta_hpp/cn), ('EN', en.coef_/cn), ('Ridge', ridge.coef_/cn)]:
    m = metrics(x_phits, coef)
    print(f"  {name}: rec={m['rec']}, rel_err={m['rel']:.2f}, R²={m['r2']:.1e}")

# ============================================================
# Track 2: 合成数据 Cond~300 (论文场景!)
# ============================================================
print("\n" + "="*80)
print("Track 2: 合成数据 Cond=300 (论文场景)")
print("="*80)

M_SYN, N_SYN = 15, 20
COND_TARGET = 300
SPARSITY = 0.8  # 80%零 → 4个非零
N_TRIALS = 15

print(f"Matrix: {M_SYN}×{N_SYN}, Cond={COND_TARGET}, Sparsity={SPARSITY}")
print(f"Active sources: {N_SYN - int(SPARSITY * N_SYN)} out of {N_SYN}\n")

methods = ['HPP_EN', 'EN_Fixed', 'Ridge', 'Lasso']
agg = {m: [] for m in methods}

for trial in range(N_TRIALS):
    rng = np.random.RandomState(42 + trial)
    A_syn, cn_syn = make_cond_matrix(M_SYN, N_SYN, COND_TARGET, rng)
    n_active = N_SYN - int(SPARSITY * N_SYN)
    x_syn = make_sparse(N_SYN, n_active, max_val=10.0, rng=rng)
    b_syn = A_syn @ x_syn
    b_syn_noisy = b_syn + 0.05 * np.linalg.norm(b_syn) * rng.randn(M_SYN)
    
    # HPP EN
    ridge_ws = Ridge(alpha=1.0).fit(A_syn, b_syn_noisy).coef_
    beta_hpp = run_hpp_ms(A_syn, b_syn_noisy, ridge_ws, l1=0.01, l2=0.01, n_starts=3, max_iter=3000)
    beta_hpp_orig = beta_hpp / cn_syn
    m_hpp = metrics(x_syn, beta_hpp_orig)
    m_hpp['method'] = 'HPP_EN'
    agg['HPP_EN'].append(m_hpp)
    
    # EN Fixed
    en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000).fit(A_syn, b_syn_noisy)
    beta_en = en.coef_ / cn_syn
    m_en = metrics(x_syn, beta_en)
    m_en['method'] = 'EN_Fixed'
    agg['EN_Fixed'].append(m_en)
    
    # Ridge
    ridge = Ridge(alpha=1.0).fit(A_syn, b_syn_noisy)
    beta_ridge = ridge.coef_ / cn_syn
    m_ridge = metrics(x_syn, beta_ridge)
    m_ridge['method'] = 'Ridge'
    agg['Ridge'].append(m_ridge)
    
    # Lasso
    lasso = Lasso(alpha=0.1, max_iter=100000).fit(A_syn, b_syn_noisy)
    beta_lasso = lasso.coef_ / cn_syn
    m_lasso = metrics(x_syn, beta_lasso)
    m_lasso['method'] = 'Lasso'
    agg['Lasso'].append(m_lasso)

# 汇总
print(f"{'Method':<12} {'AvgRecov':>10} {'AvgRelErr':>10} {'AvgR2':>10} {'R2>0比例':>10}")
for method in methods:
    vals = agg[method]
    recs = [v['rec'] for v in vals]
    n_recs = [int(r.split('/')[0]) for r in recs]
    n_tots = [int(r.split('/')[1]) for r in recs]
    rels = [v['rel'] for v in vals]
    r2s = [v['r2'] for v in vals]
    
    avg_rec = np.mean(n_recs)
    avg_tot = np.mean(n_tots)
    avg_rel = np.mean(rels)
    avg_r2 = np.mean(r2s)
    r2_pos = np.mean([1 for r in r2s if r > 0])
    
    print(f"{method:<12} {avg_rec:.1f}/{avg_tot:.1f} {avg_rel:>10.3f} {avg_r2:>10.3f} {r2_pos:>10.0%}")

# 逐Trial
print(f"\n逐Trial (Cond={COND_TARGET}, noise=5%, sparsity={SPARSITY}):")
print(f"{'Trial':>6} {'HPP':>15} {'EN_Fixed':>15} {'Ridge':>15} {'Lasso':>15}")
for i in range(N_TRIALS):
    row = f"{i:>6}"
    for m in methods:
        v = agg[m][i]
        r2_str = f"{v['r2']:.3f}" if abs(v['r2']) < 100 else f"{v['r2']:.0e}"
        row += f" {v['rec']:>6} R²={r2_str}"
    print(row)

# ============================================================
# 详细展示最佳Trial
# ============================================================
print("\n--- 最佳Trial详细 ---")
best_hpp = max(range(N_TRIALS), key=lambda i: agg['HPP_EN'][i]['r2'])
print(f"\nHPP 最佳 Trial {best_hpp}:")
rng = np.random.RandomState(42 + best_hpp)
A_syn, cn_syn = make_cond_matrix(M_SYN, N_SYN, COND_TARGET, rng)
n_active = N_SYN - int(SPARSITY * N_SYN)
x_syn = make_sparse(N_SYN, n_active, 10.0, rng)
b_syn = A_syn @ x_syn
b_syn_noisy = b_syn + 0.05 * np.linalg.norm(b_syn) * rng.randn(M_SYN)

ridge_ws = Ridge(alpha=1.0).fit(A_syn, b_syn_noisy).coef_
model = HPPElasticNetV3(lambda1=0.01, lambda2=0.01, lr=0.01, max_iter=3000, clip_norm=1.0, verbose=True)
model.fit(A_syn, b_syn_noisy, x_init=ridge_ws)

beta_hpp = model.coef_ / cn_syn
en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000).fit(A_syn, b_syn_noisy)

print(f"\n非零位置: {np.where(x_syn > 0)[0]}")
print(f"真值:     {np.array2string(x_syn[x_syn > 0], precision=2)}")
print(f"HPP EN:   {np.array2string(beta_hpp[x_syn > 0], precision=2)}")
print(f"EN Fixed: {np.array2string((en.coef_ / cn_syn)[x_syn > 0], precision=2)}")

m_hpp = metrics(x_syn, beta_hpp)
m_en = metrics(x_syn, en.coef_ / cn_syn)
print(f"\nHPP:   rec={m_hpp['rec']}, rel={m_hpp['rel']:.3f}, R²={m_hpp['r2']:.3f}")
print(f"EN:    rec={m_en['rec']}, rel={m_en['rel']:.3f}, R²={m_en['r2']:.3f}")

# ============================================================
# 超参扫描 (Cond=300)
# ============================================================
print("\n" + "="*60)
print("=== HPP 超参扫描 (Cond=300, 1 trial) ===")
print("="*60)

rng = np.random.RandomState(99)
A_scan, cn_scan = make_cond_matrix(M_SYN, N_SYN, COND_TARGET, rng)
x_scan = make_sparse(N_SYN, n_active, 10.0, rng)
b_scan = A_scan @ x_scan
b_scan_noisy = b_scan + 0.05 * np.linalg.norm(b_scan) * rng.randn(M_SYN)
ridge_ws = Ridge(alpha=1.0).fit(A_scan, b_scan_noisy).coef_

for l1 in [0.005, 0.01, 0.05, 0.1, 0.5]:
    for l2 in [0.001, 0.01, 0.05]:
        beta = run_hpp_ms(A_scan, b_scan_noisy, ridge_ws, l1, l2, n_starts=1, max_iter=2000)
        m = metrics(x_scan, beta / cn_scan)
        if m['r2'] > -10:
            print(f"  λ1={l1:.3f} λ2={l2:.3f} → rec={m['rec']}, rel={m['rel']:.3f}, R²={m['r2']:.3f}")

# ============================================================
# 不同条件数
# ============================================================
print("\n" + "="*60)
print("=== 不同条件数 (noise=5%, sparsity=0.8) ===")
print("="*60)

for cond_target in [10, 50, 300, 1000, 5000]:
    r2s = {m: [] for m in methods}
    recs = {m: [] for m in methods}
    
    for trial in range(8):
        rng = np.random.RandomState(100 + trial)
        A_c, cn_c = make_cond_matrix(M_SYN, N_SYN, cond_target, rng)
        x_c = make_sparse(N_SYN, n_active, 10.0, rng)
        b_c = A_c @ x_c
        b_c_noisy = b_c + 0.05 * np.linalg.norm(b_c) * rng.randn(M_SYN)
        
        ridge_ws = Ridge(alpha=1.0).fit(A_c, b_c_noisy).coef_
        beta_hpp = run_hpp_ms(A_c, b_c_noisy, ridge_ws, 0.01, 0.01, n_starts=2, max_iter=2000)
        
        en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000).fit(A_c, b_c_noisy)
        ridge = Ridge(alpha=1.0).fit(A_c, b_c_noisy)
        lasso = Lasso(alpha=0.1, max_iter=100000).fit(A_c, b_c_noisy)
        
        for name, coef in [('HPP_EN', beta_hpp / cn_c), ('EN_Fixed', en.coef_ / cn_c),
                           ('Ridge', ridge.coef_ / cn_c), ('Lasso', lasso.coef_ / cn_c)]:
            m = metrics(x_c, coef)
            r2s[name].append(m['r2'])
            n_r = int(m['rec'].split('/')[0])
            n_t = int(m['rec'].split('/')[1])
            recs[name].append(n_r / (n_t + 1e-10))
    
    print(f"\nCond={cond_target:>5}:")
    print(f"  {'Method':<12} {'AvgR2':>8} {'AvgRecRate':>12}")
    for name in methods:
        good = [r for r in r2s[name] if r > -100]
        avg_r2 = np.mean(good) if good else -999
        avg_rec = np.mean(recs[name])
        print(f"  {name:<12} {avg_r2:>8.3f} {avg_rec:>12.1%}")

print("\n" + "="*80)
print("=== HPP EN v3 完成 ===")
print("="*80)
