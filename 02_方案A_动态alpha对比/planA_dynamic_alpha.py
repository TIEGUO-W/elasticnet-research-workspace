#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案A：动态 α vs 固定 α 对比实验
- 基于 V14 框架，新增 dynamic_alpha (logistic 调度)
- 条件数扫描：κ ∈ {50, 100, 200, 300, 500, 1000}
- 每个条件数跑 50 次
- 对比：固定 α (GridSearchCV) vs 动态 α(k) (logistic) vs OMP vs Lasso vs Ridge
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
        except: pass
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'

FONT = setup_font()

# === Problem generation with target condition number ===
def generate_problem(m, n, target_kappa, sparsity_ratio, max_val, rng):
    """Generate A, x_true, b with approximate condition number target_kappa."""
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
    A = U.dot(np.diag(s_new)).dot(Vt)
    
    x_true = np.zeros(n)
    n_nz = n - int(sparsity_ratio * n)
    if n_nz > 0:
        idx = rng.choice(n, size=n_nz, replace=False)
        x_true[idx] = rng.uniform(0, max_val, size=n_nz)
    
    b = A.dot(x_true)
    return A, x_true, b

def normalize_cols(A):
    norms = np.linalg.norm(A, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    return A / norms, norms

# === Dynamic alpha (logistic schedule) ===
def dynamic_alpha_logistic(kappa, alpha_min=1e-6, alpha_max=1.0, k_half=300, steepness=0.01):
    """Logistic schedule: α(k) = α_min + (α_max - α_min) / (1 + exp(steepness*(kappa - k_half)))"""
    return alpha_min + (alpha_max - alpha_min) / (1.0 + np.exp(steepness * (kappa - k_half)))

# === Metrics ===
def compute_metrics(x_true, x_est):
    mse = float(mean_squared_error(x_true, x_est))
    # Support recovery
    true_support = set(np.where(np.abs(x_true) > 1e-8)[0])
    est_support = set(np.where(np.abs(x_est) > 1e-8)[0])
    if len(true_support) == 0:
        support_recall = 1.0
    else:
        support_recall = len(true_support & est_support) / len(true_support)
    # Precision
    if len(est_support) == 0:
        support_precision = 1.0 if len(true_support) == 0 else 0.0
    else:
        support_precision = len(true_support & est_support) / len(est_support)
    # Relative error
    norm_true = np.linalg.norm(x_true)
    rel_err = float(np.linalg.norm(x_est - x_true) / norm_true) if norm_true > 0 else float(np.linalg.norm(x_est - x_true))
    return {'mse': mse, 'support_recall': support_recall, 'support_precision': support_precision, 'rel_err': rel_err}

# === Methods ===
def run_elasticnet_fixed(A, b, rng_seed):
    """ElasticNet with GridSearchCV (fixed alpha search)"""
    kappa = np.linalg.cond(A)
    if kappa > 1e6:
        alpha_range = (-4, 1)
    elif kappa > 1e4:
        alpha_range = (-5, 0)
    else:
        alpha_range = (-6, -1)
    
    param_grid = {
        'alpha': np.logspace(alpha_range[0], alpha_range[1], 15),
        'l1_ratio': np.linspace(0.1, 0.99, 10)
    }
    model = ElasticNet(max_iter=10000, fit_intercept=False, positive=True, random_state=rng_seed, tol=1e-6)
    grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, refit=True)
    try:
        grid.fit(A, b)
        return grid.best_estimator_.coef_
    except:
        return np.zeros(A.shape[1])

def run_elasticnet_dynamic(A, b, rng_seed):
    """ElasticNet with dynamic alpha from logistic schedule"""
    kappa = np.linalg.cond(A)
    alpha_dyn = dynamic_alpha_logistic(kappa)
    
    # Search l1_ratio only, fix alpha from schedule
    best_score = -np.inf
    best_coef = np.zeros(A.shape[1])
    for l1 in np.linspace(0.1, 0.99, 10):
        model = ElasticNet(alpha=alpha_dyn, l1_ratio=l1, max_iter=10000, 
                          fit_intercept=False, positive=True, random_state=rng_seed, tol=1e-6)
        try:
            model.fit(A, b)
            score = -mean_squared_error(b, model.predict(A))
            if score > best_score:
                best_score = score
                best_coef = model.coef_.copy()
        except: pass
    return best_coef

def run_omp(A, b):
    n_nonzero = max(1, int(0.4 * A.shape[1]))
    model = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero, fit_intercept=False)
    try:
        model.fit(A, b)
        return model.coef_
    except:
        return np.zeros(A.shape[1])

def run_lasso(A, b, rng_seed):
    kappa = np.linalg.cond(A)
    alphas = np.logspace(-6, 0, 20)
    best_score = -np.inf
    best_coef = np.zeros(A.shape[1])
    for a in alphas:
        model = Lasso(alpha=a, max_iter=10000, fit_intercept=False, positive=True, random_state=rng_seed, tol=1e-6)
        try:
            model.fit(A, b)
            score = -mean_squared_error(b, model.predict(A))
            if score > best_score:
                best_score = score
                best_coef = model.coef_.copy()
        except: pass
    return best_coef

def run_ridge(A, b, rng_seed):
    alphas = np.logspace(-4, 4, 20)
    best_score = -np.inf
    best_coef = np.zeros(A.shape[1])
    for a in alphas:
        model = Ridge(alpha=a, max_iter=10000, fit_intercept=False, random_state=rng_seed)
        try:
            model.fit(A, b)
            score = -mean_squared_error(b, model.predict(A))
            if score > best_score:
                best_score = score
                best_coef = model.coef_.copy()
        except: pass
    return best_coef

# === Main experiment ===
def main():
    print("=" * 70)
    print("方案A：动态 α vs 固定 α 对比实验")
    print("=" * 70)
    
    KAPPAS = [50, 100, 200, 300, 500, 1000]
    N_RUNS = 20
    m, n = 8, 10
    sparsity_ratio = 0.8
    max_val = 3.0
    
    METHODS = ['EN_Fixed', 'EN_Dynamic', 'OMP', 'Lasso', 'Ridge']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results/planA")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    rng = np.random.default_rng(42)
    
    total = len(KAPPAS) * N_RUNS * len(METHODS)
    pbar = tqdm(total=total, desc="experiment", file=sys.stdout)
    
    print("Starting experiment...", flush=True)
    
    for kappa_target in KAPPAS:
        print("  kappa_target = {}".format(kappa_target), flush=True)
        for run_id in range(N_RUNS):
            seed = rng.integers(0, 2**31)
            run_rng = np.random.default_rng(seed)
            
            A, x_true, b = generate_problem(m, n, kappa_target, sparsity_ratio, max_val, run_rng)
            A_norm, col_norms = normalize_cols(A)
            kappa_actual = np.linalg.cond(A_norm)
            
            results = {}
            # EN Fixed
            coef = run_elasticnet_fixed(A_norm, b, seed)
            results['EN_Fixed'] = coef / col_norms
            pbar.update(1)
            
            # EN Dynamic
            coef = run_elasticnet_dynamic(A_norm, b, seed)
            results['EN_Dynamic'] = coef / col_norms
            pbar.update(1)
            
            # OMP
            coef = run_omp(A_norm, b)
            results['OMP'] = coef / col_norms
            pbar.update(1)
            
            # Lasso
            coef = run_lasso(A_norm, b, seed)
            results['Lasso'] = coef / col_norms
            pbar.update(1)
            
            # Ridge
            coef = run_ridge(A_norm, b, seed)
            results['Ridge'] = coef / col_norms
            pbar.update(1)
            
            for method in METHODS:
                x_est = results[method]
                met = compute_metrics(x_true, x_est)
                all_records.append({
                    'kappa_target': kappa_target,
                    'kappa_actual': kappa_actual,
                    'run_id': run_id,
                    'method': method,
                    **met
                })
    
    pbar.close()
    
    # Save CSV
    df = pd.DataFrame(all_records)
    csv_path = out_dir / f"planA_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 CSV 已保存: {csv_path}")
    
    # === Summary statistics ===
    summary = df.groupby(['kappa_target', 'method']).agg(
        mse_mean=('mse', 'mean'),
        mse_std=('mse', 'std'),
        support_recall_mean=('support_recall', 'mean'),
        support_precision_mean=('support_precision', 'mean'),
        rel_err_mean=('rel_err', 'mean'),
        rel_err_std=('rel_err', 'std')
    ).reset_index()
    
    summary_path = out_dir / f"planA_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"💾 汇总 CSV: {summary_path}")
    
    # === Print summary ===
    print("\n" + "=" * 70)
    print("📊 实验结果汇总")
    print("=" * 70)
    for kappa in KAPPAS:
        print(f"\n--- κ_target = {kappa} ---")
        sub = summary[summary['kappa_target'] == kappa]
        for _, row in sub.iterrows():
            print(f"  {row['method']:12s} | MSE={row['mse_mean']:.2e}±{row['mse_std']:.2e} | "
                  f"Recall={row['support_recall_mean']:.3f} | RelErr={row['rel_err_mean']:.3f}±{row['rel_err_std']:.3f}")
    
    # === Plotting ===
    fp = font_manager.FontProperties(family=FONT)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    
    colors = {'EN_Fixed': '#2E86AB', 'EN_Dynamic': '#E94F37', 'OMP': '#44AF69', 'Lasso': '#F3722C', 'Ridge': '#A8DADC'}
    markers = {'EN_Fixed': 'o', 'EN_Dynamic': 's', 'OMP': '^', 'Lasso': 'd', 'Ridge': 'v'}
    labels = {'EN_Fixed': 'EN (Fixed α)', 'EN_Dynamic': 'EN (Dynamic α)', 'OMP': 'OMP', 'Lasso': 'Lasso', 'Ridge': 'Ridge'}
    
    # Plot 1: MSE vs kappa
    ax = axes[0]
    for method in METHODS:
        sub = summary[summary['method'] == method]
        ax.errorbar(sub['kappa_target'], sub['mse_mean'], yerr=sub['mse_std'], 
                   marker=markers[method], color=colors[method], label=labels[method],
                   linewidth=1.5, markersize=6, capsize=3)
    ax.set_xlabel('Condition Number (κ)', fontproperties=fp)
    ax.set_ylabel('MSE (log scale)', fontproperties=fp)
    ax.set_yscale('log')
    ax.set_title('MSE vs Condition Number', fontproperties=fp, fontweight='bold')
    ax.legend(prop=fp, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Support Recovery vs kappa
    ax = axes[1]
    for method in METHODS:
        sub = summary[summary['method'] == method]
        ax.plot(sub['kappa_target'], sub['support_recall_mean'],
               marker=markers[method], color=colors[method], label=labels[method],
               linewidth=1.5, markersize=6)
    ax.set_xlabel('Condition Number (κ)', fontproperties=fp)
    ax.set_ylabel('Support Recall', fontproperties=fp)
    ax.set_title('Support Recovery vs Condition Number', fontproperties=fp, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(prop=fp, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Relative Error vs kappa
    ax = axes[2]
    for method in METHODS:
        sub = summary[summary['method'] == method]
        ax.errorbar(sub['kappa_target'], sub['rel_err_mean'], yerr=sub['rel_err_std'],
                   marker=markers[method], color=colors[method], label=labels[method],
                   linewidth=1.5, markersize=6, capsize=3)
    ax.set_xlabel('Condition Number (κ)', fontproperties=fp)
    ax.set_ylabel('Relative Error', fontproperties=fp)
    ax.set_title('Relative Error vs Condition Number', fontproperties=fp, fontweight='bold')
    ax.legend(prop=fp, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Plan A: Dynamic α vs Fixed α Comparison', fontsize=14, fontweight='bold', fontproperties=fp)
    plt.tight_layout()
    plot_path = out_dir / f"planA_comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"🖼️ 对比图已保存: {plot_path}")
    plt.close(fig)
    
    # === Heatmap: EN_Dynamic vs EN_Fixed relative improvement ===
    pivot_mse = summary.pivot_table(index='kappa_target', columns='method', values='mse_mean')
    pivot_rel = summary.pivot_table(index='kappa_target', columns='method', values='rel_err_mean')
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    for ax, pivot, title, cmap in zip(axes2, [pivot_mse, pivot_rel], 
                                        ['MSE (mean)', 'Relative Error (mean)'],
                                        ['YlOrRd', 'YlOrRd']):
        im = ax.imshow(pivot.values, aspect='auto', cmap=cmap)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([labels[m] for m in METHODS], rotation=30, ha='right', fontproperties=fp, fontsize=8)
        ax.set_yticks(range(len(KAPPAS)))
        ax.set_yticklabels([str(k) for k in KAPPAS])
        ax.set_ylabel('Condition Number (κ)', fontproperties=fp)
        ax.set_title(title, fontproperties=fp, fontweight='bold')
        # Add text annotations
        for i in range(len(KAPPAS)):
            for j in range(len(METHODS)):
                val = pivot.values[i, j]
                ax.text(j, i, f'{val:.2e}', ha='center', va='center', fontsize=6,
                       color='white' if val > np.median(pivot.values) else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    fig2.suptitle('Plan A: Method Performance Heatmap', fontsize=13, fontweight='bold', fontproperties=fp)
    plt.tight_layout()
    heatmap_path = out_dir / f"planA_heatmap_{timestamp}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"🖼️ 热力图已保存: {heatmap_path}")
    plt.close(fig2)
    
    print(f"\n✅ 方案A实验完成！结果保存在: {out_dir}")
    return csv_path, summary_path

if __name__ == "__main__":
    main()
