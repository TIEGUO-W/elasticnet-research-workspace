#!/usr/bin/env python3
"""
PHITS 8×12 矩阵稀疏恢复验证
核心问题：12列中7列非零，能恢复几列？

矩阵结构：
- 非零列索引: [1, 5, 6, 7, 8, 9, 10]
- 全零列索引: [0, 2, 3, 4, 11]
- 全零行: 行4
- 有效子矩阵: 去掉零行零列后 7×7

注意非零列量级差异巨大：
  列1:  22.64  ← 大
  列5:   0.000037 ← 极小
  列6:   0.000108 ← 极小
  列7:   9.26  ← 大
  列8:  38.48  ← 最大
  列9:   0.24  ← 中等
  列10:  0.006 ← 小
"""
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.linear_model import OrthogonalMatchingPursuit
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 1. 加载数据，确认结构
# ============================================================
A_full = np.load('/root/.openclaw/workspace/elasticnet/phits_data/phits_10x8_data/PHITS_results_10x8/A_matrix_8x12_GBq.npy')
m, n = A_full.shape
print(f"=== PHITS A_matrix_8x12_GBq ===")
print(f"Shape: {m}×{n}, Rank={np.linalg.matrix_rank(A_full)}")

nonzero_cols = np.where(np.linalg.norm(A_full, axis=0) > 1e-10)[0]
zero_cols = np.where(np.linalg.norm(A_full, axis=0) <= 1e-10)[0]
zero_rows = np.where(np.linalg.norm(A_full, axis=1) <= 1e-10)[0]
print(f"非零列 ({len(nonzero_cols)}): {nonzero_cols.tolist()}")
print(f"全零列 ({len(zero_cols)}): {zero_cols.tolist()}")
print(f"全零行: {zero_rows.tolist()}")
print()

# 非零列的量级
print("非零列范数:")
for j in nonzero_cols:
    print(f"  列{j}: {np.linalg.norm(A_full[:, j]):.6f}")
print()

# ============================================================
# 2. 提取有效子矩阵 (去掉零行零列)
# ============================================================
valid_rows = np.where(np.linalg.norm(A_full, axis=1) > 1e-10)[0]
A_sub = A_full[np.ix_(valid_rows, nonzero_cols)]
print(f"有效子矩阵: {A_sub.shape}, rank={np.linalg.matrix_rank(A_sub)}")
print(f"子矩阵 cond: {np.linalg.cond(A_sub):.2e}")
print(f"子矩阵内容:\n{np.array2string(A_sub, precision=6, suppress_small=True)}")
print()

# ============================================================
# 3. 关键测试：设一个稀疏源，看能恢复几列
# ============================================================
# 场景：在7个非零列中，选若干列有源
# 测试多种场景

def count_recovered(x_true, x_pred, threshold_ratio=0.1):
    """统计恢复了非零列中的几列"""
    nonzero_mask = np.abs(x_true) > 1e-8
    n_true_nonzero = np.sum(nonzero_mask)
    
    if np.max(np.abs(x_pred)) < 1e-15:
        return 0, n_true_nonzero, []
    
    # 预测中的显著列（相对于最大预测值）
    pred_max = np.max(np.abs(x_pred))
    pred_significant = np.abs(x_pred) > threshold_ratio * pred_max
    
    recovered = []
    missed = []
    false_positive = []
    
    for i in range(len(x_true)):
        if nonzero_mask[i]:
            if pred_significant[i]:
                recovered.append(i)
            else:
                missed.append(i)
        else:
            if pred_significant[i]:
                false_positive.append(i)
    
    return len(recovered), n_true_nonzero, recovered, missed, false_positive

def run_single_test(A, x_true, noise_level=0.01, label=""):
    """对给定源向量跑所有方法，统计恢复列数"""
    b = A @ x_true
    b_noisy = b + noise_level * np.linalg.norm(b) * np.random.randn(len(b))
    
    m, n = A.shape
    
    # 列归一化
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1
    A_norm = A / col_norms
    
    results = {}
    
    # Elastic Net (fixed α=0.01, l1=0.5)
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=100000)
    en.fit(A_norm, b_noisy)
    x_en = en.coef_ / col_norms
    r = count_recovered(x_true, x_en)
    results['EN_0.01'] = r
    
    # Elastic Net (fixed α=0.1, l1=0.5)
    en2 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000)
    en2.fit(A_norm, b_noisy)
    x_en2 = en2.coef_ / col_norms
    r2 = count_recovered(x_true, x_en2)
    results['EN_0.1'] = r2
    
    # Elastic Net (α=0.001, l1=0.3)
    en3 = ElasticNet(alpha=0.001, l1_ratio=0.3, max_iter=100000)
    en3.fit(A_norm, b_noisy)
    x_en3 = en3.coef_ / col_norms
    r3 = count_recovered(x_true, x_en3)
    results['EN_0.001'] = r3
    
    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(A_norm, b_noisy)
    x_ridge = ridge.coef_ / col_norms
    r_r = count_recovered(x_true, x_ridge)
    results['Ridge'] = r_r
    
    # Lasso
    lasso = Lasso(alpha=0.01, max_iter=100000)
    lasso.fit(A_norm, b_noisy)
    x_lasso = lasso.coef_ / col_norms
    r_l = count_recovered(x_true, x_lasso)
    results['Lasso'] = r_l
    
    # OMP
    n_nz = np.sum(np.abs(x_true) > 1e-8)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=min(n_nz, n))
    omp.fit(A_norm, b_noisy)
    x_omp = omp.coef_ / col_norms
    r_o = count_recovered(x_true, x_omp)
    results['OMP'] = r_o
    
    # 打印
    print(f"\n{'='*70}")
    print(f"场景: {label}")
    print(f"真实源: {np.where(np.abs(x_true) > 1e-8)[0].tolist()}, 共{n_nz}个非零")
    print(f"噪声: {noise_level*100:.0f}%")
    print(f"{'方法':<14} {'恢复/总数':>10} {'恢复列':>20} {'遗漏列':>20} {'误报列':>15}")
    print("-"*70)
    for method, (n_rec, n_total, rec, miss, fp) in results.items():
        if isinstance(rec, list):
            print(f"{method:<14} {n_rec}/{n_total:>4} {str(rec):>20} {str(miss):>20} {str(fp):>15}")
    
    return results

# ============================================================
# 测试场景
# ============================================================

# 场景1：只激活大列 (列1, 7, 8)
print("\n" + "="*70)
print("场景1：只激活大列 [1, 7, 8]")
x1 = np.zeros(7)
x1[0] = 50.0   # 对应原列1
x1[3] = 30.0   # 对应原列7
x1[4] = 20.0   # 对应原列8
run_single_test(A_sub, x1, noise_level=0.01, label="大列 [1,7,8], 1%噪声")

# 场景2：激活大列+中等列 (列1, 7, 8, 9)
print("\n" + "="*70)
print("场景2：大+中列 [1, 7, 8, 9]")
x2 = np.zeros(7)
x2[0] = 50.0   # 列1
x2[3] = 30.0   # 列7
x2[4] = 20.0   # 列8
x2[5] = 10.0   # 列9
run_single_test(A_sub, x2, noise_level=0.01, label="大+中列 [1,7,8,9], 1%噪声")

# 场景3：全部7列都激活
print("\n" + "="*70)
print("场景3：全部7列激活")
x3 = np.ones(7) * 10.0
run_single_test(A_sub, x3, noise_level=0.01, label="全部7列, 1%噪声")

# 场景4：只激活小列 (列5, 6, 10) — 极端场景
print("\n" + "="*70)
print("场景4：只激活小列 [5, 6, 10]")
x4 = np.zeros(7)
x4[1] = 100.0  # 列5 (极小列，源很大才能探测到)
x4[2] = 100.0  # 列6
x4[6] = 100.0  # 列10
run_single_test(A_sub, x4, noise_level=0.01, label="小列 [5,6,10], 1%噪声")

# 场景5：混合大小列
print("\n" + "="*70)
print("场景5：混合大小列 [1, 6, 8, 10]")
x5 = np.zeros(7)
x5[0] = 30.0   # 列1 (大)
x5[2] = 50.0   # 列6 (小)
x5[4] = 20.0   # 列8 (最大)
x5[6] = 40.0   # 列10 (小)
run_single_test(A_sub, x5, noise_level=0.01, label="混合 [1,6,8,10], 1%噪声")

# ============================================================
# 统计总结
# ============================================================
print("\n" + "="*70)
print("=== 批量统计 (30次随机源, 1%噪声) ===")
print("="*70)

# 在有效子矩阵上，随机30次
n_trials = 30
recovery_counts = {
    'EN_0.001': [],
    'EN_0.01': [],
    'EN_0.1': [],
    'Ridge': [],
    'Lasso': [],
    'OMP': [],
}

for trial in range(n_trials):
    # 随机选2-5个非零位置（在7列中）
    n_active = np.random.randint(2, 6)
    active_idx = np.random.choice(7, n_active, replace=False)
    x_true = np.zeros(7)
    x_true[active_idx] = np.random.uniform(5, 50, size=n_active)
    
    b = A_sub @ x_true
    b_noisy = b + 0.01 * np.linalg.norm(b) * np.random.randn(7)
    
    col_norms = np.linalg.norm(A_sub, axis=0)
    col_norms[col_norms == 0] = 1
    A_norm = A_sub / col_norms
    
    for method_name, model in [
        ('EN_0.001', ElasticNet(alpha=0.001, l1_ratio=0.3, max_iter=100000)),
        ('EN_0.01', ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=100000)),
        ('EN_0.1', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000)),
        ('Ridge', Ridge(alpha=1.0)),
        ('Lasso', Lasso(alpha=0.01, max_iter=100000)),
        ('OMP', OrthogonalMatchingPursuit(n_nonzero_coefs=min(n_active, 7))),
    ]:
        model.fit(A_norm, b_noisy)
        x_pred = model.coef_ / col_norms
        n_rec, n_total, *_ = count_recovered(x_true, x_pred)
        recovery_counts[method_name].append((n_rec, n_total))

print(f"{'方法':<14} {'平均恢复列数':>14} {'总非零列':>10} {'恢复率':>8}")
print("-"*50)
for method in recovery_counts:
    recs = recovery_counts[method]
    avg_rec = np.mean([r[0] for r in recs])
    avg_total = np.mean([r[1] for r in recs])
    rate = avg_rec / avg_total * 100 if avg_total > 0 else 0
    print(f"{method:<14} {avg_rec:>8.2f} / {avg_total:.1f} {rate:>7.1f}%")

# ============================================================
# 核心问题回答：详细看 EN 恢复的具体是哪几列
# ============================================================
print("\n" + "="*70)
print("=== 核心问题：EN (α=0.001) 恢复的具体列分布 ===")
print("非零列映射: 子矩阵索引 → 原矩阵列号")
print("  子矩阵[0]=列1(大), [1]=列5(极小), [2]=列6(极小), [3]=列7(大)")
print("  [4]=列8(最大), [5]=列9(中), [6]=列10(小)")
print("="*70)

col_recovery_count = np.zeros(7)  # 每列被成功恢复的次数
col_total_count = np.zeros(7)     # 每列有源的次数

for trial in range(n_trials):
    n_active = np.random.randint(2, 6)
    active_idx = np.random.choice(7, n_active, replace=False)
    x_true = np.zeros(7)
    x_true[active_idx] = np.random.uniform(5, 50, size=n_active)
    
    for idx in active_idx:
        col_total_count[idx] += 1
    
    b = A_sub @ x_true
    b_noisy = b + 0.01 * np.linalg.norm(b) * np.random.randn(7)
    
    col_norms_sub = np.linalg.norm(A_sub, axis=0)
    col_norms_sub[col_norms_sub == 0] = 1
    A_norm = A_sub / col_norms_sub
    
    en = ElasticNet(alpha=0.001, l1_ratio=0.3, max_iter=100000)
    en.fit(A_norm, b_noisy)
    x_pred = en.coef_ / col_norms_sub
    
    pred_max = np.max(np.abs(x_pred))
    if pred_max > 1e-15:
        for idx in active_idx:
            if np.abs(x_pred[idx]) > 0.1 * pred_max:
                col_recovery_count[idx] += 1

# 映射回原矩阵列号
col_map = {0:'列1(大)', 1:'列5(极小)', 2:'列6(极小)', 3:'列7(大)', 
           4:'列8(最大)', 5:'列9(中)', 6:'列10(小)'}
print(f"\n{'子矩阵索引':>10} {'原矩阵列':>12} {'有源次数':>10} {'恢复次数':>10} {'恢复率':>8}")
print("-"*55)
for i in range(7):
    total = int(col_total_count[i])
    rec = int(col_recovery_count[i])
    rate = rec / total * 100 if total > 0 else 0
    print(f"{i:>10} {col_map[i]:>12} {total:>10} {rec:>10} {rate:>7.0f}%")
