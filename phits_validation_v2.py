#!/usr/bin/env python3
"""
PHITS 真实数据验证 v2：A_matrix_8x12_GBq.npy
8det×12src, rank=6, 5个零列, 1个零行
策略：提取有效子矩阵 + 全矩阵对比
"""
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet, ElasticNetCV
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.linalg import pinv
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

A_full = np.load('/root/.openclaw/workspace/elasticnet/phits_data/phits_10x8_data/PHITS_results_10x8/A_matrix_8x12_GBq.npy')
m, n = A_full.shape
print(f"=== PHITS A_matrix_8x12_GBq ===")
print(f"Shape: {m}×{n}, Rank={np.linalg.matrix_rank(A_full)}")
print(f"Zero cols: {np.where(np.all(A_full==0, axis=0))[0]}")
print(f"Zero rows: {np.where(np.all(A_full==0, axis=1))[0]}")
print()

# 提取有效子矩阵
valid_cols = np.where(~np.all(A_full==0, axis=0))[0]
valid_rows = np.where(~np.all(A_full==0, axis=1))[0]
A_sub = A_full[np.ix_(valid_rows, valid_cols)]
print(f"有效子矩阵: {A_sub.shape}, rank={np.linalg.matrix_rank(A_sub)}")
print(f"Sub cond: {np.linalg.cond(A_sub):.1f}")
print(f"Sub matrix:\n{np.array2string(A_sub, precision=4, suppress_small=True)}")
print()

# ============================================================
# 实验：在有效子矩阵上对比（7det×7src, 更合理）
# 也做全矩阵对比（处理零列/行）
# ============================================================

def make_sparse_source(n, n_nonzero, max_val=100.0):
    x = np.zeros(n)
    idx = np.random.choice(n, n_nonzero, replace=False)
    x[idx] = np.random.uniform(1.0, max_val, size=n_nonzero)
    return x

def metrics(x_true, x_pred):
    mask = np.abs(x_true) > 1e-10
    mse = np.mean((x_true - x_pred)**2)
    ss_res = np.sum((x_true - x_pred)**2)
    ss_tot = np.sum((x_true - np.mean(x_true))**2)
    r2 = 1 - ss_res / (ss_tot + 1e-20)
    # Support recovery
    s_true = (np.abs(x_true) > 1e-6)
    s_pred = (np.abs(x_pred) > 0.05 * (np.max(np.abs(x_pred)) + 1e-10))
    if s_true.sum() + s_pred.sum() == 0:
        iou = 1.0
    else:
        iou = np.sum(s_true & s_pred) / (np.sum(s_true | s_pred) + 1e-10)
    rel_err = np.linalg.norm(x_true - x_pred) / (np.linalg.norm(x_true) + 1e-20)
    return {'mse': mse, 'r2': r2, 'iou': iou, 'rel_err': rel_err}

def run_methods(A, b, x_true, n_nonzero):
    res = {}
    m, n = A.shape
    
    # 列归一化
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1
    A_norm = A / col_norms
    
    # 1. OMP
    try:
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=min(n_nonzero, n))
        omp.fit(A_norm, b)
        res['OMP'] = metrics(x_true, omp.coef_ / col_norms)
    except Exception as e:
        res['OMP'] = {'mse': 1e30, 'r2': -999, 'iou': 0, 'rel_err': 999}
    
    # 2. Lasso (CV)
    try:
        lasso = Lasso(alpha=0.1, max_iter=100000)
        lasso.fit(A_norm, b)
        res['Lasso'] = metrics(x_true, lasso.coef_ / col_norms)
    except:
        res['Lasso'] = {'mse': 1e30, 'r2': -999, 'iou': 0, 'rel_err': 999}
    
    # 3. Ridge
    try:
        ridge = Ridge(alpha=1.0)
        ridge.fit(A_norm, b)
        res['Ridge'] = metrics(x_true, ridge.coef_ / col_norms)
    except:
        res['Ridge'] = {'mse': 1e30, 'r2': -999, 'iou': 0, 'rel_err': 999}
    
    # 4. EN fixed α=0.1, l1=0.5
    try:
        en_f = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000)
        en_f.fit(A_norm, b)
        res['EN_fixed'] = metrics(x_true, en_f.coef_ / col_norms)
    except:
        res['EN_fixed'] = {'mse': 1e30, 'r2': -999, 'iou': 0, 'rel_err': 999}
    
    # 5. EN CV (自动选α和l1_ratio)
    try:
        en_cv = ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9], 
                             alphas=np.logspace(-4, 2, 30),
                             cv=min(3, m), max_iter=100000)
        en_cv.fit(A_norm, b)
        res['EN_CV'] = metrics(x_true, en_cv.coef_ / col_norms)
    except:
        res['EN_CV'] = {'mse': 1e30, 'r2': -999, 'iou': 0, 'rel_err': 999}
    
    # 6. EN dynamic α (multi-stage)
    try:
        # Stage 1: Ridge warmup
        x_w = Ridge(alpha=1.0).fit(A_norm, b).coef_
        for k_iter in range(5):
            res_norm = np.linalg.norm(A_norm @ x_w - b)
            b_norm = np.linalg.norm(b) + 1e-10
            ratio = min(res_norm / b_norm, 1.0)
            l1_dyn = 0.9 - 0.5 * ratio
            alpha_dyn = 0.01 * (1 + ratio * 10)
            en_d = ElasticNet(alpha=alpha_dyn, l1_ratio=l1_dyn, max_iter=100000)
            en_d.fit(A_norm, b)
            x_w = en_d.coef_
        res['EN_dynamic'] = metrics(x_true, x_w / col_norms)
    except:
        res['EN_dynamic'] = {'mse': 1e30, 'r2': -999, 'iou': 0, 'rel_err': 999}
    
    # 7. Pinv
    try:
        x_pinv = pinv(A_norm) @ b / col_norms
        res['Pinv'] = metrics(x_true, x_pinv)
    except:
        res['Pinv'] = {'mse': 1e30, 'r2': -999, 'iou': 0, 'rel_err': 999}
    
    return res

# ============================================================
# Part 1: 有效子矩阵实验
# ============================================================
print("="*80)
print("Part 1: 有效子矩阵 ({}×{})".format(*A_sub.shape))
print("="*80)

m_sub, n_sub = A_sub.shape
noise_levels = [0.01, 0.05, 0.10]
n_trials = 30
methods = ['OMP', 'Lasso', 'Ridge', 'EN_fixed', 'EN_CV', 'EN_dynamic', 'Pinv']

agg = {noise: {m: [] for m in methods} for noise in noise_levels}

for noise_lvl in noise_levels:
    for trial in range(n_trials):
        n_active = np.random.randint(2, n_sub + 1)
        x_true = make_sparse_source(n_sub, n_active, max_val=50.0)
        b = A_sub @ x_true
        b_noisy = b + noise_lvl * np.linalg.norm(b) * np.random.randn(m_sub)
        
        r = run_methods(A_sub, b_noisy, x_true, n_active)
        for method in methods:
            agg[noise_lvl][method].append(r[method])

# 打印
for noise_lvl in noise_levels:
    print(f"\n--- noise={noise_lvl:.0%} ---")
    print(f"{'Method':<14} {'MSE':>12} {'R²':>8} {'IoU':>7} {'RelErr':>8}")
    rows = []
    for method in methods:
        vals = agg[noise_lvl][method]
        r2s = [v['r2'] for v in vals]
        # filter out catastrophics for mean
        r2_clean = [r for r in r2s if r > -100]
        if r2_clean:
            r2_mean = np.mean(r2_clean)
            mse_mean = np.mean([v['mse'] for v in vals if v['r2'] > -100])
            iou_mean = np.mean([v['iou'] for v in vals])
            rel_mean = np.mean([v['rel_err'] for v in vals if v['r2'] > -100])
        else:
            r2_mean = np.mean(r2s)
            mse_mean = np.mean([v['mse'] for v in vals])
            iou_mean = np.mean([v['iou'] for v in vals])
            rel_mean = np.mean([v['rel_err'] for v in vals])
        rows.append((method, mse_mean, r2_mean, iou_mean, rel_mean))
    rows.sort(key=lambda x: -x[2])
    for method, mse, r2, iou, rel in rows:
        print(f"{method:<14} {mse:>12.2f} {r2:>8.3f} {iou:>7.3f} {rel:>8.3f}")

# ============================================================
# Part 2: 全矩阵实验 (8×12, 含零列零行)
# ============================================================
print("\n" + "="*80)
print("Part 2: 全矩阵 ({}×{})".format(m, n))
print("="*80)

# 加小扰动消除零，让正则化方法能工作
A_reg = A_full.copy()
A_reg[A_reg == 0] = 1e-8
print(f"正则化后 cond: {np.linalg.cond(A_reg):.2e}")

agg2 = {noise: {m: [] for m in methods} for noise in noise_levels}

for noise_lvl in noise_levels:
    for trial in range(n_trials):
        n_active = np.random.randint(3, 8)
        x_true = make_sparse_source(n, n_active, max_val=50.0)
        b = A_full @ x_true  # 用原始矩阵生成观测
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-10:
            continue  # skip if all-zero measurement
        b_noisy = b + noise_lvl * b_norm * np.random.randn(m)
        
        # 对零列方法，用正则化矩阵
        r = run_methods(A_reg, b_noisy, x_true, n_active)
        for method in methods:
            agg2[noise_lvl][method].append(r[method])

for noise_lvl in noise_levels:
    print(f"\n--- noise={noise_lvl:.0%} ---")
    print(f"{'Method':<14} {'MSE':>12} {'R²':>8} {'IoU':>7} {'RelErr':>8}")
    rows = []
    for method in methods:
        vals = agg2[noise_lvl][method]
        if not vals:
            continue
        r2s = [v['r2'] for v in vals]
        r2_clean = [r for r in r2s if r > -100]
        if r2_clean:
            r2_mean = np.mean(r2_clean)
            mse_mean = np.mean([v['mse'] for v in vals if v['r2'] > -100])
            iou_mean = np.mean([v['iou'] for v in vals])
            rel_mean = np.mean([v['rel_err'] for v in vals if v['r2'] > -100])
        else:
            r2_mean = np.mean(r2s)
            mse_mean = np.mean([v['mse'] for v in vals])
            iou_mean = np.mean([v['iou'] for v in vals])
            rel_mean = np.mean([v['rel_err'] for v in vals])
        rows.append((method, mse_mean, r2_mean, iou_mean, rel_mean))
    rows.sort(key=lambda x: -x[2])
    for method, mse, r2, iou, rel in rows:
        print(f"{method:<14} {mse:>12.2f} {r2:>8.3f} {iou:>7.3f} {rel:>8.3f}")

# ============================================================
# 关键结论
# ============================================================
print("\n" + "="*80)
print("=== 关键对比: EN_fixed vs EN_dynamic ===")
print("="*80)
for part_name, agg_data in [("子矩阵", agg), ("全矩阵", agg2)]:
    print(f"\n[{part_name}]")
    for noise_lvl in noise_levels:
        f_r2 = np.mean([v['r2'] for v in agg_data[noise_lvl]['EN_fixed']])
        d_r2 = np.mean([v['r2'] for v in agg_data[noise_lvl]['EN_dynamic']])
        delta = d_r2 - f_r2
        tag = "DYNAMIC✓" if delta > 0.001 else ("FIXED✓" if delta < -0.001 else "TIE")
        print(f"  noise={noise_lvl:.0%}: fixed R²={f_r2:.4f}, dynamic R²={d_r2:.4f}, Δ={delta:+.4f} → {tag}")
