#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案B：混合源分布重建实验
- 验证 Elastic Net 的 L2 项在连续区域 + 孤立稀疏点混合场景下的优势
- 稀疏度 0.6~0.9 浮动
- 源分布类型：(1) 纯稀疏 (2) 连续区域+孤立点混合 (3) 纯连续区域
- 条件数扫描：κ ∈ {50, 100, 200, 300, 500, 1000}
- 对比方法：EN (GridSearchCV) / Lasso / Ridge / OMP
"""

import warnings, sys, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.linear_model import ElasticNet, Lasso, Ridge, OrthogonalMatchingPursuit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

warnings.filterwarnings('ignore')

# === Font setup ===
def setup_font():
    for name in ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']:
        try:
            fp = font_manager.findfont(name)
            if fp and 'unknown' not in fp.lower():
                plt.rcParams['font.sans-serif'] = [name]
                plt.rcParams['axes.unicode_minus'] = False
                return name
        except:
            pass
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'

FONT = setup_font()


# ============================================================
# Source Distribution Generators
# ============================================================

def generate_pure_sparse(n, n_nz, rng, max_val=3.0):
    """纯稀疏：随机位置，随机独立幅值"""
    x = np.zeros(n)
    if n_nz > 0:
        idx = rng.choice(n, size=n_nz, replace=False)
        x[idx] = rng.uniform(0.5, max_val, size=n_nz)
    return x


def generate_mixed_source(n, n_nz, rng, max_val=3.0):
    """混合分布：部分连续区域 + 部分孤立稀疏点

    - 60%~80% 的非零元素形成连续区域（相邻位置有相似幅值）
    - 剩余为孤立稀疏点
    """
    x = np.zeros(n)
    if n_nz == 0:
        return x

    # 决定连续区域 vs 孤立点的比例
    n_contiguous = max(1, int(n_nz * rng.uniform(0.6, 0.8)))
    n_isolated = n_nz - n_contiguous

    occupied = set()

    # 生成 1~3 个连续区域
    n_regions = min(rng.integers(1, 4), n_contiguous)
    remaining = n_contiguous
    region_sizes = []

    for i in range(n_regions):
        if i == n_regions - 1:
            sz = remaining
        else:
            sz = max(1, remaining // (n_regions - i) + rng.integers(-1, 2))
            sz = min(sz, remaining - (n_regions - i - 1))
            sz = max(1, sz)
        region_sizes.append(sz)
        remaining -= sz

    for sz in region_sizes:
        if sz <= 0:
            continue
        # 随机选起始位置（确保能放下连续区域）
        attempts = 0
        while attempts < 50:
            start = rng.integers(0, n - sz + 1)
            region_indices = list(range(start, start + sz))
            if not any(idx in occupied for idx in region_indices):
                break
            attempts += 1
        else:
            # fallback: 随机位置
            available = [i for i in range(n) if i not in occupied]
            if len(available) < sz:
                region_indices = available[:sz] if available else []
            else:
                region_indices = list(rng.choice(available, size=sz, replace=False))

        if not region_indices:
            continue

        # 连续区域：基准幅值 + 小幅波动
        base_val = rng.uniform(1.0, max_val)
        for idx in region_indices:
            x[idx] = base_val * rng.uniform(0.7, 1.3)  # ±30% 波动
            occupied.add(idx)

    # 生成孤立稀疏点
    available = [i for i in range(n) if i not in occupied]
    if n_isolated > 0 and available:
        n_iso = min(n_isolated, len(available))
        iso_idx = rng.choice(available, size=n_iso, replace=False)
        x[iso_idx] = rng.uniform(0.5, max_val, size=n_iso)

    return x


def generate_smooth_region(n, n_nz, rng, max_val=3.0):
    """纯连续区域：1~2 个连续区域，幅值平滑变化"""
    x = np.zeros(n)
    if n_nz == 0:
        return x

    n_regions = min(rng.integers(1, 3), max(1, n_nz // 2))
    remaining = n_nz
    occupied = set()

    for r in range(n_regions):
        sz = max(2, remaining // (n_regions - r))
        if r == n_regions - 1:
            sz = remaining
        sz = min(sz, n - len(occupied))
        if sz <= 0:
            continue

        available = [i for i in range(n) if i not in occupied]
        if len(available) < sz:
            available = list(range(n))

        # 找一个连续块
        attempts = 0
        while attempts < 50:
            start_idx = rng.integers(0, len(available))
            # 尝试向两边扩展
            lo, hi = available[start_idx], available[start_idx]
            used = {lo}
            while len(used) < sz:
                candidates = []
                if lo - 1 >= 0 and (lo - 1) not in occupied and (lo - 1) not in used:
                    candidates.append(lo - 1)
                if hi + 1 < n and (hi + 1) not in occupied and (hi + 1) not in used:
                    candidates.append(hi + 1)
                if not candidates:
                    break
                pick = rng.choice(candidates)
                used.add(pick)
                lo = min(lo, pick)
                hi = max(hi, pick)
            if len(used) >= sz:
                break
            attempts += 1
        else:
            used = set(rng.choice([i for i in range(n) if i not in occupied],
                                  size=min(sz, n - len(occupied)), replace=False))

        indices = sorted(list(used))[:sz]
        # 平滑幅值：线性渐变 + 小噪声
        base = rng.uniform(1.0, max_val)
        for i, idx in enumerate(indices):
            t = i / max(1, len(indices) - 1)
            x[idx] = base * (0.5 + t) * rng.uniform(0.85, 1.15)
            occupied.add(idx)
        remaining -= len(indices)

    return x


# ============================================================
# Problem generation
# ============================================================

def generate_problem(m, n, target_kappa, sparsity_ratio, source_type, rng, max_val=3.0):
    """Generate A, x_true, b with specified source distribution type."""
    # 1/r² forward model
    det_pos = rng.uniform(0, 10.0, size=(m, 3))
    src_pos = rng.uniform(0, 10.0, size=(n, 3))
    A = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            d = max(np.linalg.norm(det_pos[i] - src_pos[j]), 0.3)
            A[i, j] = 1.0 / (d ** 2)

    # Adjust condition number via SVD rescaling
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_new = np.linspace(s[0], s[0] / target_kappa, len(s))
    A = U @ np.diag(s_new) @ Vt

    # Generate source based on type
    n_nz = n - int(sparsity_ratio * n)
    n_nz = max(1, n_nz)

    if source_type == 'pure_sparse':
        x_true = generate_pure_sparse(n, n_nz, rng, max_val)
    elif source_type == 'mixed':
        x_true = generate_mixed_source(n, n_nz, rng, max_val)
    elif source_type == 'smooth':
        x_true = generate_smooth_region(n, n_nz, rng, max_val)
    else:
        x_true = generate_pure_sparse(n, n_nz, rng, max_val)

    b = A @ x_true
    return A, x_true, b


def normalize_cols(A):
    norms = np.linalg.norm(A, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    return A / norms, norms


# ============================================================
# Metrics
# ============================================================

def compute_metrics(x_true, x_est):
    mse = float(mean_squared_error(x_true, x_est))

    # Support recovery
    true_support = set(np.where(np.abs(x_true) > 1e-8)[0])
    est_support = set(np.where(np.abs(x_est) > 1e-8)[0])

    if len(true_support) == 0:
        support_recall = 1.0
    else:
        support_recall = len(true_support & est_support) / len(true_support)

    if len(est_support) == 0:
        support_precision = 1.0 if len(true_support) == 0 else 0.0
    else:
        support_precision = len(true_support & est_support) / len(est_support)

    # Relative error
    norm_true = np.linalg.norm(x_true)
    rel_err = float(np.linalg.norm(x_est - x_true) / norm_true) if norm_true > 0 else float(np.linalg.norm(x_est - x_true))

    # Smoothness metric: measure how well the method recovers smooth regions
    # (low variation between adjacent non-zero elements)
    true_nz = np.where(np.abs(x_true) > 1e-8)[0]
    if len(true_nz) >= 2:
        sorted_idx = np.sort(true_nz)
        true_diffs = np.abs(np.diff(x_true[sorted_idx]))
        est_diffs = np.abs(np.diff(x_est[sorted_idx]))
        # Smoothness preservation: how close are estimated diffs to true diffs
        smooth_err = float(np.mean(np.abs(est_diffs - true_diffs)))
    else:
        smooth_err = 0.0

    # Correlation
    norm_t = np.linalg.norm(x_true)
    norm_e = np.linalg.norm(x_est)
    if norm_t > 0 and norm_e > 0:
        corr = float(np.dot(x_true, x_est) / (norm_t * norm_e))
    else:
        corr = 0.0

    return {
        'mse': mse,
        'support_recall': support_recall,
        'support_precision': support_precision,
        'rel_err': rel_err,
        'smooth_err': smooth_err,
        'corr': corr,
    }


# ============================================================
# Methods
# ============================================================

def run_elasticnet(A, b, rng_seed):
    """ElasticNet with GridSearchCV"""
    param_grid = {
        'alpha': np.logspace(-6, 0, 15),
        'l1_ratio': np.linspace(0.1, 0.99, 10),
    }
    model = ElasticNet(max_iter=10000, fit_intercept=False, positive=True,
                       random_state=rng_seed, tol=1e-6)
    grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error',
                        n_jobs=-1, refit=True)
    try:
        grid.fit(A, b)
        return grid.best_estimator_.coef_
    except Exception:
        return np.zeros(A.shape[1])


def run_lasso(A, b, rng_seed):
    best_score, best_coef = -np.inf, np.zeros(A.shape[1])
    for a in np.logspace(-6, 0, 20):
        model = Lasso(alpha=a, max_iter=10000, fit_intercept=False,
                      positive=True, random_state=rng_seed, tol=1e-6)
        try:
            model.fit(A, b)
            s = -mean_squared_error(b, model.predict(A))
            if s > best_score:
                best_score, best_coef = s, model.coef_.copy()
        except Exception:
            pass
    return best_coef


def run_ridge(A, b, rng_seed):
    best_score, best_coef = -np.inf, np.zeros(A.shape[1])
    for a in np.logspace(-4, 4, 20):
        model = Ridge(alpha=a, max_iter=10000, fit_intercept=False,
                      random_state=rng_seed)
        try:
            model.fit(A, b)
            s = -mean_squared_error(b, model.predict(A))
            if s > best_score:
                best_score, best_coef = s, model.coef_.copy()
        except Exception:
            pass
    return best_coef


def run_omp(A, b, n):
    n_nonzero = max(1, int(0.4 * n))
    model = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero, fit_intercept=False)
    try:
        model.fit(A, b)
        return model.coef_
    except Exception:
        return np.zeros(n)


# ============================================================
# Main experiment
# ============================================================

def main():
    print("=" * 70)
    print("方案B：混合源分布重建实验")
    print("  - 稀疏度 0.6~0.9 浮动")
    print("  - 源类型：纯稀疏 / 混合(连续+孤立) / 纯连续区域")
    print("  - 对比：EN / Lasso / Ridge / OMP")
    print("=" * 70)

    KAPPAS = [50, 100, 200, 300, 500, 1000]
    SOURCE_TYPES = ['pure_sparse', 'mixed', 'smooth']
    SOURCE_LABELS = {'pure_sparse': '纯稀疏', 'mixed': '混合(连续+孤立)', 'smooth': '连续区域'}
    N_RUNS = 15  # 先跑 15 次，验证设计
    m, n = 8, 12
    MAX_VAL = 3.0
    NOISE_LEVELS = [0.0, 0.01, 0.05]  # 无噪声 / 1% / 5%

    METHODS = ['EN', 'Lasso', 'Ridge', 'OMP']
    METHOD_COLORS = {'EN': '#2E86AB', 'Lasso': '#F3722C', 'Ridge': '#A8DADC', 'OMP': '#44AF69'}
    METHOD_MARKERS = {'EN': 'o', 'Lasso': 'd', 'Ridge': 'v', 'OMP': '^'}
    METHOD_LABELS = {'EN': 'Elastic Net', 'Lasso': 'Lasso', 'Ridge': 'Ridge', 'OMP': 'OMP'}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results/planB")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    rng_master = np.random.default_rng(42)

    total = len(KAPPAS) * len(SOURCE_TYPES) * len(NOISE_LEVELS) * N_RUNS * len(METHODS)
    pbar = tqdm(total=total, desc="PlanB", file=sys.stdout)

    for kappa_target in KAPPAS:
        for src_type in SOURCE_TYPES:
            for noise_level in NOISE_LEVELS:
                for run_id in range(N_RUNS):
                    seed = rng_master.integers(0, 2**31)
                    run_rng = np.random.default_rng(seed)

                    # 浮动稀疏度 0.6~0.9
                    sparsity = run_rng.uniform(0.6, 0.9)

                    A, x_true, b = generate_problem(
                        m, n, kappa_target, sparsity, src_type, run_rng, MAX_VAL
                    )
                    A_norm, col_norms = normalize_cols(A)
                    kappa_actual = np.linalg.cond(A_norm)

                    # 加噪声
                    if noise_level > 0:
                        b_noise = run_rng.normal(0, noise_level * np.std(b), size=b.shape)
                        b_noisy = b + b_noise
                    else:
                        b_noisy = b

                    results = {}

                    # EN
                    coef = run_elasticnet(A_norm, b_noisy, seed)
                    results['EN'] = coef / col_norms
                    pbar.update(1)

                    # Lasso
                    coef = run_lasso(A_norm, b_noisy, seed)
                    results['Lasso'] = coef / col_norms
                    pbar.update(1)

                    # Ridge
                    coef = run_ridge(A_norm, b_noisy, seed)
                    results['Ridge'] = coef / col_norms
                    pbar.update(1)

                    # OMP
                    coef = run_omp(A_norm, b_noisy, n)
                    results['OMP'] = coef / col_norms
                    pbar.update(1)

                    for method in METHODS:
                        x_est = results[method]
                        met = compute_metrics(x_true, x_est)
                        all_records.append({
                            'kappa_target': kappa_target,
                            'kappa_actual': kappa_actual,
                            'src_type': src_type,
                            'sparsity': sparsity,
                            'noise_level': noise_level,
                            'run_id': run_id,
                            'method': method,
                            **met,
                        })

    pbar.close()

    # Save
    df = pd.DataFrame(all_records)
    csv_path = out_dir / f"planB_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 CSV: {csv_path}")

    # ============================================================
    # Summary
    # ============================================================
    summary = df.groupby(['kappa_target', 'src_type', 'noise_level', 'method']).agg(
        mse_mean=('mse', 'mean'),
        mse_std=('mse', 'std'),
        rel_err_mean=('rel_err', 'mean'),
        rel_err_std=('rel_err', 'std'),
        support_recall_mean=('support_recall', 'mean'),
        smooth_err_mean=('smooth_err', 'mean'),
        corr_mean=('corr', 'mean'),
    ).reset_index()

    summary_path = out_dir / f"planB_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"💾 Summary: {summary_path}")

    # Print key results
    print("\n" + "=" * 90)
    print("📊 方案B结果汇总（noise=0.01, κ=300 重点场景）")
    print("=" * 90)
    key = summary[(summary['kappa_target'] == 300) & (summary['noise_level'] == 0.01)]
    for src in SOURCE_TYPES:
        print(f"\n  ── {SOURCE_LABELS[src]} ──")
        sub = key[key['src_type'] == src].sort_values('rel_err_mean')
        for _, row in sub.iterrows():
            print(f"    {row['method']:8s} | MSE={row['mse_mean']:.2e}±{row['mse_std']:.2e} | "
                  f"RelErr={row['rel_err_mean']:.3f} | SmoothErr={row['smooth_err_mean']:.3f} | "
                  f"Corr={row['corr_mean']:.3f}")

    # ============================================================
    # EN vs Lasso advantage on different source types
    # ============================================================
    print("\n" + "=" * 90)
    print("📊 EN vs Lasso 对比（各源类型，κ=300，noise=1%）")
    print("=" * 90)
    for src in SOURCE_TYPES:
        en_row = key[(key['src_type'] == src) & (key['method'] == 'EN')]
        lasso_row = key[(key['src_type'] == src) & (key['method'] == 'Lasso')]
        if len(en_row) > 0 and len(lasso_row) > 0:
            en_mse = en_row['mse_mean'].values[0]
            la_mse = lasso_row['mse_mean'].values[0]
            en_se = en_row['smooth_err_mean'].values[0]
            la_se = lasso_row['smooth_err_mean'].values[0]
            print(f"  {SOURCE_LABELS[src]:15s} | EN MSE={en_mse:.2e} vs Lasso MSE={la_mse:.2e} "
                  f"(ratio={en_mse/la_mse:.2f}) | SmoothErr: EN={en_se:.3f} Lasso={la_se:.3f}")

    # ============================================================
    # Plotting
    # ============================================================
    fp = font_manager.FontProperties(family=FONT)

    # --- Plot 1: MSE by source type (bar chart) ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    for ax_idx, src in enumerate(SOURCE_TYPES):
        ax = axes1[ax_idx]
        sub = summary[(summary['src_type'] == src) &
                      (summary['noise_level'] == 0.01) &
                      (summary['kappa_target'] == 300)]
        if len(sub) == 0:
            continue
        methods_plot = sub['method'].tolist()
        mses = sub['mse_mean'].tolist()
        stds = sub['mse_std'].tolist()
        colors = [METHOD_COLORS[m] for m in methods_plot]

        bars = ax.bar(range(len(methods_plot)), mses, yerr=stds,
                      color=colors, capsize=5, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(methods_plot)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods_plot],
                           fontproperties=fp, fontsize=9)
        ax.set_ylabel('MSE', fontproperties=fp)
        ax.set_title(f'{SOURCE_LABELS[src]}\n(κ=300, noise=1%)',
                     fontproperties=fp, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')

    fig1.suptitle('方案B：不同源分布类型的重建误差对比',
                  fontsize=14, fontweight='bold', fontproperties=fp)
    plt.tight_layout()
    p1 = out_dir / f"planB_mse_by_srctype_{timestamp}.png"
    plt.savefig(p1, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"🖼️ 图1 (MSE by src type): {p1}")
    plt.close(fig1)

    # --- Plot 2: RelErr vs kappa, faceted by source type ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    for ax_idx, src in enumerate(SOURCE_TYPES):
        ax = axes2[ax_idx]
        for method in METHODS:
            sub = summary[(summary['src_type'] == src) &
                         (summary['noise_level'] == 0.01) &
                         (summary['method'] == method)]
            if len(sub) == 0:
                continue
            ax.errorbar(sub['kappa_target'], sub['rel_err_mean'], yerr=sub['rel_err_std'],
                       marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                       label=METHOD_LABELS[method], linewidth=1.5, markersize=6, capsize=3)
        ax.set_xlabel('Condition Number (κ)', fontproperties=fp)
        ax.set_ylabel('Relative Error', fontproperties=fp)
        ax.set_title(SOURCE_LABELS[src], fontproperties=fp, fontweight='bold')
        ax.legend(prop=fp, fontsize=8)
        ax.grid(True, alpha=0.3)

    fig2.suptitle('方案B：不同源分布下的 RelErr vs κ (noise=1%)',
                  fontsize=14, fontweight='bold', fontproperties=fp)
    plt.tight_layout()
    p2 = out_dir / f"planB_relerr_vs_kappa_{timestamp}.png"
    plt.savefig(p2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"🖼️ 图2 (RelErr vs kappa): {p2}")
    plt.close(fig2)

    # --- Plot 3: Smooth error comparison ---
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6), dpi=150)

    noise01 = summary[summary['noise_level'] == 0.01]
    x_pos = np.arange(len(SOURCE_TYPES))
    width = 0.18

    for i, method in enumerate(METHODS):
        vals = []
        for src in SOURCE_TYPES:
            sub = noise01[(noise01['src_type'] == src) &
                         (noise01['method'] == method) &
                         (noise01['kappa_target'] == 300)]
            vals.append(sub['smooth_err_mean'].values[0] if len(sub) > 0 else 0)
        ax3.bar(x_pos + i * width, vals, width, label=METHOD_LABELS[method],
                color=METHOD_COLORS[method], alpha=0.85, edgecolor='black', linewidth=0.5)

    ax3.set_xticks(x_pos + width * 1.5)
    ax3.set_xticklabels([SOURCE_LABELS[s] for s in SOURCE_TYPES], fontproperties=fp)
    ax3.set_ylabel('平滑误差 (Smooth Error)', fontproperties=fp)
    ax3.set_title('方案B：平滑区域恢复误差对比 (κ=300, noise=1%)',
                  fontproperties=fp, fontweight='bold')
    ax3.legend(prop=fp, fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    p3 = out_dir / f"planB_smooth_error_{timestamp}.png"
    plt.savefig(p3, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"🖼️ 图3 (Smooth error): {p3}")
    plt.close(fig3)

    # --- Plot 4: Example reconstruction (one sample per source type) ---
    rng_vis = np.random.default_rng(123)
    fig4, axes4 = plt.subplots(3, 5, figsize=(22, 10), dpi=150)

    for row_idx, src in enumerate(SOURCE_TYPES):
        sparsity = 0.7
        A_vis, x_true_vis, b_vis = generate_problem(
            m, n, 300, sparsity, src, rng_vis, MAX_VAL
        )
        A_vis_norm, col_norms_vis = normalize_cols(A_vis)
        b_vis_noisy = b_vis + rng_vis.normal(0, 0.01 * np.std(b_vis), size=b_vis.shape)

        # True
        axes4[row_idx, 0].bar(range(n), x_true_vis, color='#333333', alpha=0.8)
        axes4[row_idx, 0].set_title('真实值', fontproperties=fp, fontsize=10)
        if row_idx == 0:
            axes4[row_idx, 0].set_ylabel(f'{SOURCE_LABELS[src]}', fontproperties=fp,
                                          fontsize=10, fontweight='bold')

        for col_idx, method in enumerate(METHODS):
            if method == 'EN':
                coef = run_elasticnet(A_vis_norm, b_vis_noisy, 123)
            elif method == 'Lasso':
                coef = run_lasso(A_vis_norm, b_vis_noisy, 123)
            elif method == 'Ridge':
                coef = run_ridge(A_vis_norm, b_vis_noisy, 123)
            else:
                coef = run_omp(A_vis_norm, b_vis_noisy, n)
            x_est = coef / col_norms_vis

            ax = axes4[row_idx, col_idx + 1]
            ax.bar(range(n), x_est, color=METHOD_COLORS[method], alpha=0.8)
            re = np.linalg.norm(x_est - x_true_vis) / max(np.linalg.norm(x_true_vis), 1e-10)
            ax.set_title(f'{METHOD_LABELS[method]}\nRelErr={re:.3f}',
                        fontproperties=fp, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(SOURCE_LABELS[src], fontproperties=fp, fontsize=10)

    fig4.suptitle('方案B：重建示例对比 (κ=300, noise=1%, sparsity=0.7)',
                  fontsize=14, fontweight='bold', fontproperties=fp)
    plt.tight_layout()
    p4 = out_dir / f"planB_reconstruction_example_{timestamp}.png"
    plt.savefig(p4, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"🖼️ 图4 (Reconstruction example): {p4}")
    plt.close(fig4)

    # ============================================================
    # EN advantage analysis: L2 contribution
    # ============================================================
    print("\n" + "=" * 90)
    print("📊 EN vs Lasso 平滑恢复优势分析（全条件数，noise=1%）")
    print("=" * 90)
    for src in SOURCE_TYPES:
        print(f"\n  ── {SOURCE_LABELS[src]} ──")
        for kappa in KAPPAS:
            en = summary[(summary['src_type'] == src) & (summary['method'] == 'EN') &
                        (summary['noise_level'] == 0.01) & (summary['kappa_target'] == kappa)]
            la = summary[(summary['src_type'] == src) & (summary['method'] == 'Lasso') &
                        (summary['noise_level'] == 0.01) & (summary['kappa_target'] == kappa)]
            if len(en) > 0 and len(la) > 0:
                en_mse = en['mse_mean'].values[0]
                la_mse = la['mse_mean'].values[0]
                en_sm = en['smooth_err_mean'].values[0]
                la_sm = la['smooth_err_mean'].values[0]
                mse_ratio = en_mse / la_mse if la_mse > 0 else float('inf')
                sm_ratio = en_sm / la_sm if la_sm > 0 else float('inf')
                marker_mse = "✅ EN胜" if mse_ratio < 0.95 else ("❌ Lasso胜" if mse_ratio > 1.05 else "≈持平")
                marker_sm = "✅ EN胜" if sm_ratio < 0.95 else ("❌ Lasso胜" if sm_ratio > 1.05 else "≈持平")
                print(f"    κ={kappa:5d} | MSE ratio={mse_ratio:.3f} {marker_mse} | "
                      f"SmoothErr ratio={sm_ratio:.3f} {marker_sm}")

    print(f"\n✅ 方案B实验完成！所有结果保存在: {out_dir}")
    return csv_path, summary_path


if __name__ == "__main__":
    main()
