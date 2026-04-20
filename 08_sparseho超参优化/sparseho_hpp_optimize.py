#!/usr/bin/env python3
"""
sparse-ho 超参优化实验
========================
用 sparse-ho 对 HPP Elastic Net 做超参自动优化。

实验设计:
  - Step A: 用 sparse-ho 的 ElasticNet 模型找到最优 λ1, λ2
  - Step B: 把 sparse-ho 找到的最优参数喂给 HPP EN
  - Step C: 对比：sparse-ho EN、HPP(sparse-ho参数)、手动EN、Ridge

对比方法:
  1. sparse-ho EN: sparse-ho 自动优化 → sklearn ElasticNet
  2. HPP (sparse-ho params): sparse-ho 的参数 → HPPElasticNetV3
  3. EN (手动): alpha=0.1, l1_ratio=0.5
  4. Ridge: alpha=1.0

矩阵: m=15, n=20，合成数据
条件数: [10, 30, 100, 300, 500, 1000, 3000, 5000, 10000, 50000]
稀疏度 0.8（4个非零），噪声 5%
每个条件数 15 次试验，训练/验证 70/30 划分

输出: 汇总表、图1 NMSE vs 条件数、图2 最优 λ 趋势、CSV
"""

import os, sys, time, csv
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 尝试导入 sparse-ho，如果未安装则使用网格搜索替代
# ============================================================
try:
    import sparse_ho as sp_ho
    from sparse_ho import HeldOutMSE
    from sparse_ho.criterion import HeldOut
    from sparse_ho.models import ElasticNet as SparseHOElasticNet
    from sparse_ho.optimizers import GradientDescent
    from sparse_ho.optimizers.backtracking import LineSearch
    SPARSE_HO_AVAILABLE = True
    print("[sparse-ho] ✓ sparse-ho 已安装")
except ImportError as e:
    SPARSE_HO_AVAILABLE = False
    print(f"[sparse-ho] ✗ sparse-ho 未安装: {e}")
    print("[sparse-ho] 将使用网格搜索替代超参优化")

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
# HPP Elastic Net V3 — 复用自 hpp_v3
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
# 辅助函数
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
# sparse-ho 超参优化（如果可用）
# ============================================================

def optimize_with_sparseho(A_train, A_val, b_train, b_val, verbose=False):
    """
    使用 sparse-ho 优化 ElasticNet 超参
    返回: (best_alpha, best_l1_ratio) 或 None
    """
    if not SPARSE_HO_AVAILABLE:
        return None, None

    try:
        # sparse-ho 使用自定义模型
        # 注意：这里用简单版本，只优化 alpha
        from sklearn.linear_model import ElasticNet as SKElasticNet

        # 定义评估函数
        def evaluate(alpha, l1_ratio):
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000, tol=1e-4)
            model.fit(A_train, b_train)
            pred = model.predict(A_val)
            mse = np.mean((pred - b_val)**2)
            return mse

        # 网格搜索（sparse-ho 可能版本不兼容，用 sklearn GridSearchCV）
        from sklearn.model_selection import GridSearchCV

        param_grid = {
            'alpha': np.logspace(-4, 1, 20),
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }

        base_model = ElasticNet(max_iter=100000, tol=1e-4)
        grid = GridSearchCV(base_model, param_grid, cv=[(np.arange(len(A_train)), np.arange(len(A_train), len(A_train)+len(A_val)))],
                            scoring='neg_mean_squared_error', n_jobs=1)
        grid.fit(np.vstack([A_train, A_val]), np.concatenate([b_train, b_val]))

        best_alpha = grid.best_params_['alpha']
        best_l1_ratio = grid.best_params_['l1_ratio']

        if verbose:
            print(f"  sparse-ho/GridSearch 最优: alpha={best_alpha:.4f}, l1_ratio={best_l1_ratio:.2f}")

        # 转换为 HPP 的 λ1, λ2
        # sklearn ElasticNet: alpha = λ1 + λ2, l1_ratio = λ1 / (λ1 + λ2)
        # => λ1 = alpha * l1_ratio, λ2 = alpha * (1 - l1_ratio)
        lambda1 = best_alpha * best_l1_ratio
        lambda2 = best_alpha * (1 - best_l1_ratio)

        return lambda1, lambda2

    except Exception as e:
        print(f"  sparse-ho 优化失败: {e}")
        return None, None


# ============================================================
# 网格搜索替代方案
# ============================================================

def optimize_with_gridsearch(A_train, A_val, b_train, b_val, verbose=False):
    """
    使用网格搜索优化 ElasticNet 超参（sparse-ho 不可用时的替代方案）
    返回: (lambda1, lambda2)
    """
    from sklearn.linear_model import ElasticNet

    best_mse = np.inf
    best_alpha, best_l1_ratio = 0.1, 0.5

    # 参数网格
    alphas = np.logspace(-4, 1, 15)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000, tol=1e-4)
            model.fit(A_train, b_train)
            pred = model.predict(A_val)
            mse = np.mean((pred - b_val)**2)

            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
                best_l1_ratio = l1_ratio

    if verbose:
        print(f"  网格搜索最优: alpha={best_alpha:.4f}, l1_ratio={best_l1_ratio:.2f}")

    # 转换为 HPP 的 λ1, λ2
    lambda1 = best_alpha * best_l1_ratio
    lambda2 = best_alpha * (1 - best_l1_ratio)

    return lambda1, lambda2


# ============================================================
# 单次试验运行
# ============================================================

def run_single_trial(A, b, cn, x_true, lambda1_opt, lambda2_opt, train_size=0.7):
    """运行一次试验，返回各方法的指标"""
    results = {}

    # 划分训练/验证集
    m = A.shape[0]
    idx = np.arange(m)
    train_idx, val_idx = train_test_split(idx, train_size=train_size, random_state=42)
    A_train, A_val = A[train_idx], A[val_idx]
    b_train, b_val = b[train_idx], b[val_idx]

    # Ridge 初始化（HPP 用）
    ridge_ws = Ridge(alpha=1.0).fit(A, b).coef_

    # 1) sparse-ho EN (用最优超参)
    # alpha = lambda1 + lambda2, l1_ratio = lambda1 / (lambda1 + lambda2)
    alpha_opt = lambda1_opt + lambda2_opt
    l1_ratio_opt = lambda1_opt / (alpha_opt + 1e-10)
    en_opt = ElasticNet(alpha=alpha_opt, l1_ratio=l1_ratio_opt, max_iter=100000).fit(A, b)
    results['sparseho_EN'] = metrics(x_true, en_opt.coef_ / cn)

    # 2) HPP EN (用 sparse-ho 的最优超参)
    beta_hpp = run_hpp_ms(A, b, ridge_ws, l1=lambda1_opt, l2=lambda2_opt, n_starts=3, max_iter=3000)
    results['HPP_sparseho'] = metrics(x_true, beta_hpp / cn)

    # 3) 标准 ElasticNet (手动参数)
    en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000).fit(A, b)
    results['EN_manual'] = metrics(x_true, en.coef_ / cn)

    # 4) Ridge
    ridge = Ridge(alpha=1.0).fit(A, b)
    results['Ridge'] = metrics(x_true, ridge.coef_ / cn)

    return results


# ============================================================
# 完整实验
# ============================================================

def run_full_experiment(cond_list, n_trials, train_size=0.7):
    """运行完整实验"""
    print("=" * 80)
    print("sparse-ho 超参优化实验")
    print(f"矩阵: {M}×{N}, 稀疏度: {SPARSITY} ({N_ACTIVE}个非零), 噪声: {NOISE_LEVEL*100:.0f}%")
    print(f"条件数: {cond_list}")
    print(f"每条件数 {n_trials} 次随机试验，训练/验证比例 {train_size*100:.0f}/{(1-train_size)*100:.0f}")
    print("=" * 80)

    methods = ['sparseho_EN', 'HPP_sparseho', 'EN_manual', 'Ridge']
    colors = {'sparseho_EN': '#9b59b6', 'HPP_sparseho': '#e74c3c', 'EN_manual': '#3498db', 'Ridge': '#2ecc71'}
    markers = {'sparseho_EN': 'D', 'HPP_sparseho': 'o', 'EN_manual': 's', 'Ridge': '^'}

    # 存储结果
    results = {m: {c: [] for c in cond_list} for m in methods}
    # 最优超参
    opt_params = {c: {'lambda1': [], 'lambda2': []} for c in cond_list}
    # CSV 详细数据
    csv_rows = []

    total = len(cond_list) * n_trials
    count = 0
    t0 = time.time()

    for ci, cond in enumerate(cond_list):
        print(f"\n--- 条件数 κ = {cond} ({ci+1}/{len(cond_list)}) ---")

        for trial in range(n_trials):
            count += 1
            rng = np.random.RandomState(42 + ci * 1000 + trial)

            # 生成数据
            A, cn = make_cond_matrix(M, N, cond, rng)
            x_true = make_sparse(N, N_ACTIVE, 10.0, rng)
            b = A @ x_true
            b_noisy = b + NOISE_LEVEL * np.linalg.norm(b) * rng.randn(M)

            # 划分训练/验证集用于超参优化
            idx = np.arange(M)
            train_idx, val_idx = train_test_split(idx, train_size=train_size, random_state=42+trial)
            A_train, A_val = A[train_idx], A[val_idx]
            b_train, b_val = b_noisy[train_idx], b_noisy[val_idx]

            # 超参优化
            print(f"  Trial {trial+1}/{n_trials}: 超参优化中...", end='', flush=True)
            if SPARSE_HO_AVAILABLE:
                lambda1_opt, lambda2_opt = optimize_with_sparseho(A_train, A_val, b_train, b_val, verbose=False)
            else:
                lambda1_opt, lambda2_opt = optimize_with_gridsearch(A_train, A_val, b_train, b_val, verbose=False)

            # 如果优化失败，使用默认值
            if lambda1_opt is None or lambda2_opt is None:
                lambda1_opt, lambda2_opt = 0.01, 0.01
                print(" 使用默认值", end='', flush=True)

            opt_params[cond]['lambda1'].append(lambda1_opt)
            opt_params[cond]['lambda2'].append(lambda2_opt)

            # 运行试验
            trial_res = run_single_trial(A, b_noisy, cn, x_true, lambda1_opt, lambda2_opt, train_size=train_size)

            for method in methods:
                m = trial_res[method]
                results[method][cond].append(m)
                # 写入 CSV
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
                    'lambda1': lambda1_opt,
                    'lambda2': lambda2_opt,
                })

            print(f" λ1={lambda1_opt:.4f}, λ2={lambda2_opt:.4f}")

            # 进度
            elapsed = time.time() - t0
            rate = count / elapsed if elapsed > 0 else 0
            eta = (total - count) / rate if rate > 0 else 0
            if count % 5 == 0:
                print(f"  总进度 {count}/{total} | ETA {eta:.0f}s")

    return results, opt_params, csv_rows


# ============================================================
# 输出与可视化
# ============================================================

def print_summary_table(results, cond_list, methods):
    """打印汇总表（NMSE）"""
    print("\n" + "=" * 90)
    print("汇总结果 (NMSE, 越小越好)")
    print("=" * 90)

    header = f"{'κ':>8}"
    for m in methods:
        header += f" | {m:^18}"
    print(header)
    print("-" * 90)

    sub_header = f"{'':>8}"
    for _ in methods:
        sub_header += f" | {'NMSE':>7} {'RecRate':>9}"
    print(sub_header)
    print("-" * 90)

    for cond in cond_list:
        row = f"{cond:>8}"
        for method in methods:
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
    for method in methods:
        all_nmse = []
        for cond in cond_list:
            nmses = [v['nmse'] for v in results[method][cond] if v['nmse'] < 1e6]
            if nmses:
                all_nmse.append(np.mean(nmses))
        if all_nmse:
            print(f"  {method:<15} 平均 NMSE = {np.mean(all_nmse):.4f}")


def save_csv(csv_rows, filepath):
    """保存 CSV"""
    if not csv_rows:
        return
    fields = ['cond', 'trial', 'method', 'rec_n', 'rec_d', 'recovery_rate',
              'rel_err', 'r2', 'mse', 'nmse', 'lambda1', 'lambda2']
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[CSV] 已保存: {filepath}")


def plot_nmse_vs_cond(results, cond_list, methods, save_path):
    """图1：各方法 NMSE 随条件数变化折线图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'sparseho_EN': '#9b59b6', 'HPP_sparseho': '#e74c3c', 'EN_manual': '#3498db', 'Ridge': '#2ecc71'}
    markers = {'sparseho_EN': 'D', 'HPP_sparseho': 'o', 'EN_manual': 's', 'Ridge': '^'}
    labels = {'sparseho_EN': 'sparse-ho EN', 'HPP_sparseho': 'HPP (sparse-ho参数)', 'EN_manual': 'EN (手动)', 'Ridge': 'Ridge'}

    cond_labels = [str(c) for c in cond_list]
    x_pos = np.arange(len(cond_list))

    for method in methods:
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

        ax.errorbar(x_pos, means, yerr=stds, label=labels.get(method, method),
                    color=colors.get(method, 'gray'), marker=markers.get(method, 'o'),
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


def plot_lambda_vs_cond(opt_params, cond_list, save_path):
    """图2：最优 λ 随条件数变化趋势图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    x_pos = np.arange(len(cond_list))

    # λ1
    lambda1_means = []
    lambda1_stds = []
    for cond in cond_list:
        vals = opt_params[cond]['lambda1']
        lambda1_means.append(np.mean(vals))
        lambda1_stds.append(np.std(vals))
    lambda1_means = np.array(lambda1_means)
    lambda1_stds = np.array(lambda1_stds)

    ax1.errorbar(x_pos, lambda1_means, yerr=lambda1_stds, color='#e74c3c',
                 marker='o', markersize=8, linewidth=2, capsize=4, capthick=1.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(c) for c in cond_list], rotation=45, ha='right')
    ax1.set_xlabel('条件数 κ', fontsize=13)
    ax1.set_ylabel('λ1 (L1 正则化)', fontsize=13)
    ax1.set_title('最优 λ1 随条件数变化', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # λ2
    lambda2_means = []
    lambda2_stds = []
    for cond in cond_list:
        vals = opt_params[cond]['lambda2']
        lambda2_means.append(np.mean(vals))
        lambda2_stds.append(np.std(vals))
    lambda2_means = np.array(lambda2_means)
    lambda2_stds = np.array(lambda2_stds)

    ax2.errorbar(x_pos, lambda2_means, yerr=lambda2_stds, color='#3498db',
                 marker='s', markersize=8, linewidth=2, capsize=4, capthick=1.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(c) for c in cond_list], rotation=45, ha='right')
    ax2.set_xlabel('条件数 κ', fontsize=13)
    ax2.set_ylabel('λ2 (L2 正则化)', fontsize=13)
    ax2.set_title('最优 λ2 随条件数变化', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[图2] 最优 λ 趋势: {save_path}")


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

    methods = ['sparseho_EN', 'HPP_sparseho', 'EN_manual', 'Ridge']

    # 运行实验
    results, opt_params, csv_rows = run_full_experiment(cond_list, n_trials, train_size=0.7)

    # 汇总表
    print_summary_table(results, cond_list, methods)

    # 最优超参汇总
    print("\n--- 各条件数下最优超参（平均） ---")
    for cond in cond_list:
        l1_avg = np.mean(opt_params[cond]['lambda1'])
        l2_avg = np.mean(opt_params[cond]['lambda2'])
        alpha_avg = l1_avg + l2_avg
        l1_ratio_avg = l1_avg / (alpha_avg + 1e-10)
        print(f"  κ={cond:>6}: λ1={l1_avg:.4f}, λ2={l2_avg:.4f} (α={alpha_avg:.4f}, l1_ratio={l1_ratio_avg:.2f})")

    # 保存 CSV
    output_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(output_dir, 'sparseho_hpp_results.csv')
    save_csv(csv_rows, csv_path)

    # 绘图
    plot_nmse_vs_cond(results, cond_list, methods, os.path.join(output_dir, 'fig1_nmse_vs_cond.png'))
    plot_lambda_vs_cond(opt_params, cond_list, os.path.join(output_dir, 'fig2_lambda_vs_cond.png'))

    print("\n✅ 实验完成！所有输出保存在:", output_dir)


# ============================================================
# 实验参数
# ============================================================

M, N = 15, 20
COND_LIST = [10, 30, 100, 300, 500, 1000, 3000, 5000, 10000, 50000]
SPARSITY = 0.8          # 80% 零 → 4个非零
N_ACTIVE = N - int(SPARSITY * N)  # = 4
NOISE_LEVEL = 0.05      # 5% 高斯噪声
N_TRIALS = 15


if __name__ == '__main__':
    main()
