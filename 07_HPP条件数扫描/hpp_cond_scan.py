#!/usr/bin/env python3
"""
HPP Elastic Net 条件数扫描实验
================================
在 10 个不同条件数下对比 HPP EN vs 标准 EN vs Ridge vs Lasso。
含 HPP 超参扫描，找每个条件数下最佳参数组合。

复用自 hpp_v3_双轨验证版：HPPElasticNetV3, make_sparse, make_cond_matrix, metrics, run_hpp_ms
"""

import os, sys, time, csv
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge, Lasso
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 中文字体配置
# ============================================================
def setup_chinese_font():
    """配置 matplotlib 中文字体"""
    candidates = [
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'Noto Sans CJK SC',
        'Noto Sans CJK',
        'Source Han Sans SC',
        'SimHei',
        'AR PL UMing CN',
    ]
    from matplotlib.font_manager import fontManager
    available = {f.name for f in fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"[字体] 使用: {name}")
            return
    # 尝试 fc-match
    import subprocess
    try:
        out = subprocess.check_output(
            ['fc-match', '-f', '%{family}'], encoding='utf-8', timeout=5
        )
        if out:
            plt.rcParams['font.sans-serif'] = [out.split(',')[0].strip(), 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"[字体] fc-match 回退: {out}")
            return
    except Exception:
        pass
    print("[字体] ⚠ 未找到中文字体，图表标签可能显示为方块")

setup_chinese_font()

# ============================================================
# HPP Elastic Net V3 — 直接复用
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
    """多起点 HPP 优化"""
    best_beta, best_loss = None, np.inf
    n = A.shape[1]
    for s in range(n_starts):
        init = ridge_init if s == 0 else ridge_init + np.random.randn(n) * 0.01 * (np.std(ridge_init) + 1e-6)
        model = HPPElasticNetV3(lambda1=l1, lambda2=l2, lr=lr, max_iter=max_iter, clip_norm=1.0)
        model.fit(A, b, x_init=init)
        loss = np.sum((A @ model.coef_ - b)**2)
        if loss < best_loss:
            best_loss = loss
            best_beta = model.coef_.copy()
    return best_beta

# ============================================================
# 辅助函数 — 直接复用
# ============================================================

def make_sparse(n, k, max_val=10.0, rng=None):
    """生成稀疏向量：k个非零，值在 [0.5, max_val]"""
    if rng is None:
        rng = np.random
    x = np.zeros(n)
    idx = rng.choice(n, k, replace=False)
    x[idx] = rng.uniform(0.5, max_val, size=k)
    return x

def make_cond_matrix(m, n, cond_target, rng):
    """构造指定条件数的矩阵，列归一化"""
    log_c = np.log10(cond_target)
    k = min(m, n)
    sv = np.logspace(0, log_c, k)
    U, _ = np.linalg.qr(rng.standard_normal((m, m)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    S = np.zeros((m, n))
    np.fill_diagonal(S, sv)
    A = U @ S @ V.T
    cn = np.linalg.norm(A, axis=0)
    cn[cn == 0] = 1.0
    A = A / cn
    return A, cn

def metrics(x_true, x_pred):
    """计算恢复率、相对误差、R²、MSE、NMSE"""
    mask = np.abs(x_true) > 1e-10
    n_active = mask.sum()
    if n_active == 0:
        return {'rec_n': 0, 'rec_d': 0, 'rel': 0, 'r2': -999, 'mse': 0, 'nmse': 0}
    
    pred_max = np.max(np.abs(x_pred)) + 1e-10
    mask_pred = np.abs(x_pred) > 0.10 * pred_max
    n_rec = np.sum(mask & mask_pred)
    
    rel = np.mean(np.abs(x_true[mask] - x_pred[mask]) / (np.abs(x_true[mask]) + 1e-10))
    mse = np.mean((x_true - x_pred)**2)
    # NMSE = ||x_pred - x_true||^2 / ||x_true||^2
    norm2_true = np.sum(x_true**2) + 1e-20
    nmse = np.sum((x_pred - x_true)**2) / norm2_true
    ss_r = np.sum((x_true - x_pred)**2)
    ss_t = np.sum((x_true - np.mean(x_true))**2)
    r2 = 1 - ss_r / (ss_t + 1e-20)
    
    return {'rec_n': n_rec, 'rec_d': n_active, 'rel': rel, 'r2': r2, 'mse': mse, 'nmse': nmse}

# ============================================================
# 实验参数
# ============================================================

M, N = 15, 20
COND_LIST = [10, 30, 100, 300, 500, 1000, 3000, 5000, 10000, 50000]
SPARSITY = 0.8          # 80% 零 → 4个非零
N_ACTIVE = N - int(SPARSITY * N)  # = 4
NOISE_LEVEL = 0.05      # 5% 高斯噪声
N_TRIALS = 15

# HPP 超参扫描范围
L1_CANDIDATES = [0.005, 0.01, 0.05, 0.1, 0.5]
L2_CANDIDATES = [0.001, 0.01, 0.05]

METHODS = ['HPP_EN', 'EN_Fixed', 'Ridge', 'Lasso']
COLORS = {'HPP_EN': '#e74c3c', 'EN_Fixed': '#3498db', 'Ridge': '#2ecc71', 'Lasso': '#f39c12'}
MARKERS = {'HPP_EN': 'o', 'EN_Fixed': 's', 'Ridge': '^', 'Lasso': 'D'}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 核心实验函数
# ============================================================

def run_single_trial(A, b_noisy, cn, x_true, l1, l2):
    """运行一次试验，返回各方法的指标"""
    results = {}
    
    # Ridge 初始化（HPP 用）
    ridge_ws = Ridge(alpha=1.0).fit(A, b_noisy).coef_
    
    # 1) HPP EN（用给定超参）
    beta_hpp = run_hpp_ms(A, b_noisy, ridge_ws, l1=l1, l2=l2, n_starts=3, max_iter=3000)
    results['HPP_EN'] = metrics(x_true, beta_hpp / cn)
    
    # 2) 标准 ElasticNet
    en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000).fit(A, b_noisy)
    results['EN_Fixed'] = metrics(x_true, en.coef_ / cn)
    
    # 3) Ridge
    ridge = Ridge(alpha=1.0).fit(A, b_noisy)
    results['Ridge'] = metrics(x_true, ridge.coef_ / cn)
    
    # 4) Lasso
    lasso = Lasso(alpha=0.1, max_iter=100000).fit(A, b_noisy)
    results['Lasso'] = metrics(x_true, lasso.coef_ / cn)
    
    return results


def hpp_param_scan(A, b_noisy, cn, x_true, rng):
    """HPP 超参扫描，返回最佳 (l1, l2) 及扫描矩阵"""
    ridge_ws = Ridge(alpha=1.0).fit(A, b_noisy).coef_
    
    best_r2 = -np.inf
    best_l1, best_l2 = L1_CANDIDATES[1], L2_CANDIDATES[1]  # 默认值
    scan_matrix = np.zeros((len(L1_CANDIDATES), len(L2_CANDIDATES)))
    
    for i, l1 in enumerate(L1_CANDIDATES):
        for j, l2 in enumerate(L2_CANDIDATES):
            beta = run_hpp_ms(A, b_noisy, ridge_ws, l1, l2, n_starts=2, max_iter=2000)
            m = metrics(x_true, beta / cn)
            scan_matrix[i, j] = m['r2']
            if m['r2'] > best_r2:
                best_r2 = m['r2']
                best_l1, best_l2 = l1, l2
    
    return best_l1, best_l2, best_r2, scan_matrix


def run_full_experiment(cond_list, n_trials):
    """运行完整实验"""
    print("=" * 80)
    print("HPP Elastic Net 条件数扫描实验")
    print(f"矩阵: {M}×{N}, 稀疏度: {SPARSITY} ({N_ACTIVE}个非零), 噪声: {NOISE_LEVEL*100:.0f}%")
    print(f"条件数: {cond_list}")
    print(f"每条件数 {n_trials} 次随机试验")
    print(f"HPP 超参扫描: λ1∈{L1_CANDIDATES}, λ2∈{L2_CANDIDATES}")
    print("=" * 80)
    
    # 存储结果：results[method][cond_idx] = [trial_metrics, ...]
    results = {m: {c: [] for c in cond_list} for m in METHODS}
    # HPP 超参扫描结果：best_params[cond] = (l1, l2), scan_maps[cond] = matrix
    best_params = {}
    scan_maps = {}
    # CSV 详细数据
    csv_rows = []
    
    total = len(cond_list) * n_trials
    count = 0
    t0 = time.time()
    
    for ci, cond in enumerate(cond_list):
        print(f"\n--- 条件数 κ = {cond} ({ci+1}/{len(cond_list)}) ---")
        
        # 每个条件数先做超参扫描（用第 0 个 seed）
        rng_scan = np.random.RandomState(999 + ci)
        A_scan, cn_scan = make_cond_matrix(M, N, cond, rng_scan)
        x_scan = make_sparse(N, N_ACTIVE, 10.0, rng_scan)
        b_scan = A_scan @ x_scan
        b_scan_noisy = b_scan + NOISE_LEVEL * np.linalg.norm(b_scan) * rng_scan.randn(M)
        
        print(f"  超参扫描中...", end='', flush=True)
        best_l1, best_l2, best_r2, scan_mat = hpp_param_scan(
            A_scan, b_scan_noisy, cn_scan, x_scan, rng_scan
        )
        scan_maps[cond] = scan_mat
        best_params[cond] = (best_l1, best_l2)
        print(f" 最佳 λ1={best_l1}, λ2={best_l2} (scan R²={best_r2:.3f})")
        
        # 正式试验
        for trial in range(n_trials):
            count += 1
            rng = np.random.RandomState(42 + ci * 1000 + trial)
            A, cn = make_cond_matrix(M, N, cond, rng)
            x_true = make_sparse(N, N_ACTIVE, 10.0, rng)
            b = A @ x_true
            b_noisy = b + NOISE_LEVEL * np.linalg.norm(b) * rng.randn(M)
            
            trial_res = run_single_trial(A, b_noisy, cn, x_true, best_l1, best_l2)
            
            for method in METHODS:
                m = trial_res[method]
                results[method][cond].append(m)
                # 写入 CSV 行
                csv_rows.append({
                    'cond': cond,
                    'trial': trial,
                    'method': method,
                    'rec_n': m['rec_n'],
                    'rec_d': m['rec_d'],
                    'recovery_rate': m['rec_n'] / (m['rec_d'] + 1e-10),
                    'rel_err': m['rel'],
                    'r2': m['r2'],
                    'mse': m['mse'],
                    'nmse': m['nmse'],
                    'hpp_l1': best_l1 if method == 'HPP_EN' else '',
                    'hpp_l2': best_l2 if method == 'HPP_EN' else '',
                })
            
            # 进度
            elapsed = time.time() - t0
            rate = count / elapsed if elapsed > 0 else 0
            eta = (total - count) / rate if rate > 0 else 0
            if trial == n_trials - 1 or count % 10 == 0:
                print(f"  Trial {trial+1}/{n_trials} 完成 | 总进度 {count}/{total} | "
                      f"ETA {eta:.0f}s")
    
    return results, best_params, scan_maps, csv_rows

# ============================================================
# 输出与可视化
# ============================================================

def print_summary_table(results, cond_list):
    """打印汇总表"""
    print("\n" + "=" * 90)
    print("汇总结果")
    print("=" * 90)
    
    header = f"{'κ':>8}"
    for m in METHODS:
        header += f" | {m:^18}"
    print(header)
    print("-" * 90)
    
    sub_header = f"{'':>8}"
    for _ in METHODS:
        sub_header += f" | {'NMSE':>7} {'RecRate':>9}"
    print(sub_header)
    print("-" * 90)
    
    for cond in cond_list:
        row = f"{cond:>8}"
        for method in METHODS:
            vals = results[method][cond]
            nmses = [v['nmse'] for v in vals if v['nmse'] < 1e6]
            recs = [v['rec_n'] / (v['rec_d'] + 1e-10) for v in vals]
            avg_nmse = np.mean(nmses) if nmses else -999
            avg_rec = np.mean(recs)
            row += f" | {avg_nmse:>7.3f} {avg_rec:>8.1%}"
        print(row)
    
    print("=" * 90)
    
    # 方法间对比总结
    print("\n--- 各方法跨条件数平均 NMSE ---")
    for method in METHODS:
        all_nmse = []
        for cond in cond_list:
            nmses = [v['nmse'] for v in results[method][cond] if v['nmse'] < 1e6]
            if nmses:
                all_nmse.append(np.mean(nmses))
        if all_nmse:
            print(f"  {method:<12} 平均 NMSE = {np.mean(all_nmse):.4f}")


def save_csv(csv_rows, filepath):
    """保存 CSV"""
    if not csv_rows:
        return
    fields = ['cond', 'trial', 'method', 'rec_n', 'rec_d', 'recovery_rate',
              'rel_err', 'r2', 'mse', 'nmse', 'hpp_l1', 'hpp_l2']
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[CSV] 已保存: {filepath}")


def plot_nmse_vs_cond(results, cond_list, save_path):
    """图1：各方法 NMSE 随条件数变化折线图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cond_labels = [str(c) for c in cond_list]
    x_pos = np.arange(len(cond_list))
    
    for method in METHODS:
        means = []
        stds = []
        for cond in cond_list:
            nmses = [v['nmse'] for v in results[method][cond] if v['nmse'] < 1e6]
            if nmses:
                means.append(np.mean(nmses))
                stds.append(np.std(nmses))
            else:
                means.append(0)
                stds.append(0)
        means = np.array(means)
        stds = np.array(stds)
        
        ax.errorbar(x_pos, means, yerr=stds, label=method,
                    color=COLORS[method], marker=MARKERS[method],
                    markersize=8, linewidth=2, capsize=4, capthick=1.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cond_labels, rotation=45, ha='right')
    ax.set_xlabel('条件数 κ', fontsize=13)
    ax.set_ylabel('NMSE', fontsize=13)
    ax.set_title('各方法 NMSE 随条件数变化（越小越好）', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[图1] NMSE vs 条件数: {save_path}")


def plot_recovery_vs_cond(results, cond_list, save_path):
    """图2：各方法恢复率随条件数变化折线图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cond_labels = [str(c) for c in cond_list]
    x_pos = np.arange(len(cond_list))
    
    for method in METHODS:
        means = []
        stds = []
        for cond in cond_list:
            recs = [v['rec_n'] / (v['rec_d'] + 1e-10) for v in results[method][cond]]
            means.append(np.mean(recs))
            stds.append(np.std(recs))
        means = np.array(means)
        stds = np.array(stds)
        
        ax.errorbar(x_pos, means, yerr=stds, label=method,
                    color=COLORS[method], marker=MARKERS[method],
                    markersize=8, linewidth=2, capsize=4, capthick=1.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cond_labels, rotation=45, ha='right')
    ax.set_xlabel('条件数 κ', fontsize=13)
    ax.set_ylabel('恢复率', fontsize=13)
    ax.set_title('各方法恢复率随条件数变化', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[图2] 恢复率 vs 条件数: {save_path}")


def plot_hpp_heatmap(scan_maps, cond_list, save_path):
    """图3：HPP 超参扫描热力图"""
    n_conds = len(cond_list)
    n_l1 = len(L1_CANDIDATES)
    n_l2 = len(L2_CANDIDATES)
    
    # 综合热力图：每个条件数的最佳 R²
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()
    
    for ci, cond in enumerate(cond_list):
        ax = axes[ci]
        mat = scan_maps[cond]
        
        # 截断显示
        mat_show = np.clip(mat, -2, 1)
        
        im = ax.imshow(mat_show, cmap='RdYlGn', aspect='auto', vmin=-2, vmax=1)
        ax.set_xticks(range(n_l2))
        ax.set_xticklabels([f'{x:.3f}' for x in L2_CANDIDATES], fontsize=7, rotation=45)
        ax.set_yticks(range(n_l1))
        ax.set_yticklabels([f'{x:.3f}' for x in L1_CANDIDATES], fontsize=7)
        ax.set_title(f'κ={cond}', fontsize=10, fontweight='bold')
        ax.set_xlabel('λ2', fontsize=8)
        ax.set_ylabel('λ1', fontsize=8)
        
        # 标注数值
        for i in range(n_l1):
            for j in range(n_l2):
                val = mat[i, j]
                if val > -100:
                    txt_color = 'white' if val < -0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=6, color=txt_color)
    
    fig.suptitle('HPP 超参扫描热力图 (R²)', fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[图3] HPP 超参热力图: {save_path}")


# ============================================================
# 主入口
# ============================================================

def main():
    # 检查命令行参数，支持快速验证模式
    quick_mode = '--quick' in sys.argv
    
    if quick_mode:
        cond_list = [10, 300]
        n_trials = 3
        print("⚡ 快速验证模式：2 个条件数 × 3 次试验")
    else:
        cond_list = COND_LIST
        n_trials = N_TRIALS
        print("🔬 完整实验模式：10 个条件数 × 15 次试验")
    
    # 运行实验
    results, best_params, scan_maps, csv_rows = run_full_experiment(cond_list, n_trials)
    
    # 汇总表
    print_summary_table(results, cond_list)
    
    # HPP 最佳参数
    print("\n--- 各条件数下 HPP 最佳超参 ---")
    for cond in cond_list:
        l1, l2 = best_params[cond]
        print(f"  κ={cond:>6}: λ1={l1}, λ2={l2}")
    
    # 保存 CSV
    csv_path = os.path.join(OUTPUT_DIR, 'hpp_cond_scan_results.csv')
    save_csv(csv_rows, csv_path)
    
    # 绘图
    plot_nmse_vs_cond(results, cond_list, os.path.join(OUTPUT_DIR, 'fig1_nmse_vs_cond.png'))
    plot_recovery_vs_cond(results, cond_list, os.path.join(OUTPUT_DIR, 'fig2_recovery_vs_cond.png'))
    plot_hpp_heatmap(scan_maps, cond_list, os.path.join(OUTPUT_DIR, 'fig3_hpp_heatmap.png'))
    
    print("\n✅ 实验完成！所有输出保存在:", OUTPUT_DIR)


if __name__ == '__main__':
    main()
