#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案B（最终版·轻量）：混合源分布重建实验

锅哥三条决定：
1. 仿真轨道：8×10 干净矩阵（可控 κ），不加零列
2. PHITS 零列放非零值：要放。看不可观测区域局限性的程度
3. 矩阵维度：仿真用 8×10，PHITS 用 8×12

轻量化：减少 GridSearch 规模，避免 OOM
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
from sklearn.model_selection import cross_val_score
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
PHITS_MATRIX_PATH = '/root/.openclaw/workspace/elasticnet/phits_data/phits_10x8_data/PHITS_results_10x8/A_matrix_8x12_GBq.npy'

METHOD_COLORS = {'EN': '#2E86AB', 'Lasso': '#F3722C', 'Ridge': '#A8DADC', 'OMP': '#44AF69'}
METHOD_MARKERS = {'EN': 'o', 'Lasso': 'd', 'Ridge': 'v', 'OMP': '^'}
METHOD_LABELS = {'EN': 'Elastic Net', 'Lasso': 'Lasso', 'Ridge': 'Ridge', 'OMP': 'OMP'}
METHODS = ['EN', 'Lasso', 'Ridge', 'OMP']
SOURCE_TYPES = ['pure_sparse', 'mixed', 'smooth']
SOURCE_LABELS = {'pure_sparse': '纯稀疏', 'mixed': '混合(连续+孤立)', 'smooth': '连续区域'}


# ============================================================
# Source Distribution Generators
# ============================================================

def generate_pure_sparse(n, n_nz, rng, max_val=3.0):
    x = np.zeros(n)
    if n_nz > 0:
        idx = rng.choice(n, size=n_nz, replace=False)
        x[idx] = rng.uniform(0.5, max_val, size=n_nz)
    return x

def generate_mixed_source(n, n_nz, rng, max_val=3.0):
    x = np.zeros(n)
    if n_nz == 0:
        return x
    n_contiguous = max(1, int(n_nz * rng.uniform(0.6, 0.8)))
    n_isolated = n_nz - n_contiguous
    occupied = set()
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
        attempts = 0
        while attempts < 50:
            start = rng.integers(0, n - sz + 1)
            region_indices = list(range(start, start + sz))
            if not any(idx in occupied for idx in region_indices):
                break
            attempts += 1
        else:
            available = [i for i in range(n) if i not in occupied]
            if len(available) < sz:
                region_indices = available[:sz] if available else []
            else:
                region_indices = list(rng.choice(available, size=sz, replace=False))
        if not region_indices:
            continue
        base_val = rng.uniform(1.0, max_val)
        for idx in region_indices:
            x[idx] = base_val * rng.uniform(0.7, 1.3)
            occupied.add(idx)

    available = [i for i in range(n) if i not in occupied]
    if n_isolated > 0 and available:
        n_iso = min(n_isolated, len(available))
        iso_idx = rng.choice(available, size=n_iso, replace=False)
        x[iso_idx] = rng.uniform(0.5, max_val, size=n_iso)
    return x

def generate_smooth_region(n, n_nz, rng, max_val=3.0):
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
        attempts = 0
        while attempts < 50:
            start_idx = rng.integers(0, len(available))
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
        base = rng.uniform(1.0, max_val)
        for i, idx in enumerate(indices):
            t = i / max(1, len(indices) - 1)
            x[idx] = base * (0.5 + t) * rng.uniform(0.85, 1.15)
            occupied.add(idx)
        remaining -= len(indices)
    return x


# ============================================================
# Matrix Generation
# ============================================================

def generate_clean_matrix(m, n, target_kappa, rng):
    det_pos = rng.uniform(0, 10.0, size=(m, 3))
    src_pos = rng.uniform(0, 10.0, size=(n, 3))
    A = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            d = max(np.linalg.norm(det_pos[i] - src_pos[j]), 0.3)
            A[i, j] = 1.0 / (d ** 2)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_new = np.linspace(s[0], s[0] / target_kappa, len(s))
    A = U @ np.diag(s_new) @ Vt
    return A

def normalize_cols(A):
    norms = np.linalg.norm(A, axis=0)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return A / norms, norms


# ============================================================
# Metrics
# ============================================================

def compute_metrics(x_true, x_est):
    mse = float(mean_squared_error(x_true, x_est))
    true_support = set(np.where(np.abs(x_true) > 1e-8)[0])
    est_support = set(np.where(np.abs(x_est) > 1e-8)[0])
    support_recall = len(true_support & est_support) / len(true_support) if len(true_support) > 0 else 1.0
    support_precision = len(true_support & est_support) / len(est_support) if len(est_support) > 0 else (1.0 if len(true_support) == 0 else 0.0)
    norm_true = np.linalg.norm(x_true)
    rel_err = float(np.linalg.norm(x_est - x_true) / norm_true) if norm_true > 0 else float(np.linalg.norm(x_est - x_true))
    true_nz = np.where(np.abs(x_true) > 1e-8)[0]
    if len(true_nz) >= 2:
        sorted_idx = np.sort(true_nz)
        true_diffs = np.abs(np.diff(x_true[sorted_idx]))
        est_diffs = np.abs(np.diff(x_est[sorted_idx]))
        smooth_err = float(np.mean(np.abs(est_diffs - true_diffs)))
    else:
        smooth_err = 0.0
    norm_t = np.linalg.norm(x_true)
    norm_e = np.linalg.norm(x_est)
    corr = float(np.dot(x_true, x_est) / (norm_t * norm_e)) if norm_t > 0 and norm_e > 0 else 0.0
    return {'mse': mse, 'support_recall': support_recall, 'support_precision': support_precision,
            'rel_err': rel_err, 'smooth_err': smooth_err, 'corr': corr}


# ============================================================
# Lightweight Methods (avoid GridSearchCV for memory)
# ============================================================

def run_elasticnet_light(A, b, seed):
    """Lightweight EN: try a few alphas, pick best by BIC"""
    best_score, best_coef = np.inf, np.zeros(A.shape[1])
    for alpha in np.logspace(-5, 0, 10):
        for l1r in [0.3, 0.5, 0.7, 0.9]:
            model = ElasticNet(alpha=alpha, l1_ratio=l1r, max_iter=5000,
                               fit_intercept=False, positive=True,
                               random_state=seed, tol=1e-5, selection='random')
            try:
                model.fit(A, b)
                pred = model.predict(A)
                mse_val = np.mean((b - pred) ** 2)
                # BIC-like: n*log(mse) + k*log(n)
                k = np.sum(np.abs(model.coef_) > 1e-10)
                n = len(b)
                bic = n * np.log(mse_val + 1e-20) + k * np.log(n)
                if bic < best_score:
                    best_score = bic
                    best_coef = model.coef_.copy()
            except:
                pass
    return best_coef

def run_lasso_light(A, b, seed):
    best_score, best_coef = np.inf, np.zeros(A.shape[1])
    for alpha in np.logspace(-5, 0, 15):
        model = Lasso(alpha=alpha, max_iter=5000, fit_intercept=False,
                      positive=True, random_state=seed, tol=1e-5, selection='random')
        try:
            model.fit(A, b)
            pred = model.predict(A)
            mse_val = np.mean((b - pred) ** 2)
            k = np.sum(np.abs(model.coef_) > 1e-10)
            n = len(b)
            bic = n * np.log(mse_val + 1e-20) + k * np.log(n)
            if bic < best_score:
                best_score = bic
                best_coef = model.coef_.copy()
        except:
            pass
    return best_coef

def run_ridge_light(A, b, seed):
    best_score, best_coef = np.inf, np.zeros(A.shape[1])
    for alpha in np.logspace(-3, 4, 12):
        model = Ridge(alpha=alpha, max_iter=5000, fit_intercept=False, random_state=seed)
        try:
            model.fit(A, b)
            pred = model.predict(A)
            mse_val = np.mean((b - pred) ** 2)
            k = np.sum(np.abs(model.coef_) > 1e-10)
            n = len(b)
            bic = n * np.log(mse_val + 1e-20) + k * np.log(n)
            if bic < best_score:
                best_score = bic
                best_coef = model.coef_.copy()
        except:
            pass
    return best_coef

def run_omp_light(A, b, n):
    n_nonzero = max(1, int(0.4 * n))
    model = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero, fit_intercept=False)
    try:
        model.fit(A, b)
        return model.coef_
    except:
        return np.zeros(n)


# ============================================================
# Track 1: Simulation (clean 8×10)
# ============================================================

def run_simulation_track():
    print("\n" + "=" * 70)
    print("🚂 轨道一：仿真 8×10 干净矩阵")
    print("=" * 70)

    KAPPAS = [50, 100, 200, 300, 500, 1000]
    N_RUNS = 10
    m, n = 8, 10
    NOISE_LEVELS = [0.0, 0.01, 0.05]

    all_records = []
    rng_master = np.random.default_rng(42)
    total = len(KAPPAS) * len(SOURCE_TYPES) * len(NOISE_LEVELS) * N_RUNS
    pbar = tqdm(total=total, desc="Track1", file=sys.stdout)

    for kappa_target in KAPPAS:
        for src_type in SOURCE_TYPES:
            for noise_level in NOISE_LEVELS:
                for run_id in range(N_RUNS):
                    seed = rng_master.integers(0, 2**31)
                    run_rng = np.random.default_rng(seed)
                    sparsity = run_rng.uniform(0.6, 0.9)

                    A = generate_clean_matrix(m, n, kappa_target, run_rng)
                    n_nz = max(1, n - int(sparsity * n))
                    if src_type == 'pure_sparse':
                        x_true = generate_pure_sparse(n, n_nz, run_rng)
                    elif src_type == 'mixed':
                        x_true = generate_mixed_source(n, n_nz, run_rng)
                    else:
                        x_true = generate_smooth_region(n, n_nz, run_rng)

                    b = A @ x_true
                    A_norm, col_norms = normalize_cols(A)
                    if noise_level > 0:
                        b_noisy = b + run_rng.normal(0, noise_level * np.std(b), size=b.shape)
                    else:
                        b_noisy = b

                    results = {}
                    results['EN'] = run_elasticnet_light(A_norm, b_noisy, seed) / col_norms
                    results['Lasso'] = run_lasso_light(A_norm, b_noisy, seed) / col_norms
                    results['Ridge'] = run_ridge_light(A_norm, b_noisy, seed) / col_norms
                    results['OMP'] = run_omp_light(A_norm, b_noisy, n) / col_norms

                    for method in METHODS:
                        met = compute_metrics(x_true, results[method])
                        all_records.append({
                            'kappa_target': kappa_target,
                            'kappa_actual': np.linalg.cond(A_norm),
                            'src_type': src_type, 'sparsity': sparsity,
                            'noise_level': noise_level, 'run_id': run_id,
                            'method': method, 'track': 'simulation',
                            'm': m, 'n': n, **met,
                        })
                    pbar.update(1)
    pbar.close()
    return pd.DataFrame(all_records)


# ============================================================
# Track 2: PHITS real matrix (8×12)
# ============================================================

def run_phits_track():
    print("\n" + "=" * 70)
    print("🚂 轨道二：PHITS 8×12（零列放非零值）")
    print("=" * 70)

    A_phits = np.load(PHITS_MATRIX_PATH)
    m, n = A_phits.shape
    zero_cols = np.where(np.all(np.abs(A_phits) < 1e-15, axis=0))[0].tolist()
    nonzero_cols = [j for j in range(n) if j not in zero_cols]
    print(f"矩阵: {m}×{n}, 零列: {zero_cols}, 非零列: {nonzero_cols}")

    N_RUNS = 10
    NOISE_LEVELS = [0.0, 0.01, 0.05]

    all_records = []
    rng_master = np.random.default_rng(42)
    total = len(SOURCE_TYPES) * len(NOISE_LEVELS) * N_RUNS
    pbar = tqdm(total=total, desc="Track2", file=sys.stdout)

    for src_type in SOURCE_TYPES:
        for noise_level in NOISE_LEVELS:
            for run_id in range(N_RUNS):
                seed = rng_master.integers(0, 2**31)
                run_rng = np.random.default_rng(seed)
                sparsity = run_rng.uniform(0.6, 0.9)
                n_nz = max(1, n - int(sparsity * n))

                if src_type == 'pure_sparse':
                    x_true = generate_pure_sparse(n, n_nz, run_rng)
                elif src_type == 'mixed':
                    x_true = generate_mixed_source(n, n_nz, run_rng)
                else:
                    x_true = generate_smooth_region(n, n_nz, run_rng)

                # 强制零列有非零值
                zero_cols_with_val = [j for j in zero_cols if abs(x_true[j]) > 1e-10]
                if len(zero_cols_with_val) == 0 and len(zero_cols) > 0:
                    n_force = min(run_rng.integers(1, 3), len(zero_cols))
                    force_cols = run_rng.choice(zero_cols, size=n_force, replace=False)
                    for jc in force_cols:
                        x_true[jc] = run_rng.uniform(0.5, 3.0)

                b = A_phits @ x_true
                A_norm, col_norms = normalize_cols(A_phits)
                if noise_level > 0:
                    b_noisy = b + run_rng.normal(0, noise_level * (np.std(b) + 1e-10), size=b.shape)
                else:
                    b_noisy = b

                results = {}
                results['EN'] = run_elasticnet_light(A_norm, b_noisy, seed) / col_norms
                results['Lasso'] = run_lasso_light(A_norm, b_noisy, seed) / col_norms
                results['Ridge'] = run_ridge_light(A_norm, b_noisy, seed) / col_norms
                results['OMP'] = run_omp_light(A_norm, b_noisy, n) / col_norms

                for method in METHODS:
                    x_est = results[method]
                    met = compute_metrics(x_true, x_est)
                    obs_true = x_true[nonzero_cols]
                    obs_est = x_est[nonzero_cols]
                    unobs_true = x_true[zero_cols]
                    unobs_est = x_est[zero_cols]

                    all_records.append({
                        'src_type': src_type, 'sparsity': sparsity,
                        'noise_level': noise_level, 'run_id': run_id,
                        'method': method, 'track': 'phits', 'm': m, 'n': n,
                        'n_zero_cols': len(zero_cols), **met,
                        'obs_mse': float(mean_squared_error(obs_true, obs_est)),
                        'unobs_mse': float(mean_squared_error(unobs_true, unobs_est)),
                        'obs_relerr': float(np.linalg.norm(obs_est - obs_true) / max(np.linalg.norm(obs_true), 1e-10)),
                        'unobs_relerr': float(np.linalg.norm(unobs_est - unobs_true) / max(np.linalg.norm(unobs_true), 1e-10)),
                        'unobs_true_norm': float(np.linalg.norm(unobs_true)),
                        'unobs_est_norm': float(np.linalg.norm(unobs_est)),
                    })
                pbar.update(1)
    pbar.close()
    return pd.DataFrame(all_records), zero_cols, nonzero_cols


# ============================================================
# Analysis & Plotting
# ============================================================

def analyze_and_plot(df_sim, df_phits, zero_cols, nonzero_cols, out_dir, timestamp):
    fp = font_manager.FontProperties(family=FONT)

    # --- Track 1 Summary ---
    print("\n" + "=" * 90)
    print("📊 轨道一（仿真 8×10）结果汇总")
    print("=" * 90)

    sim_sum = df_sim.groupby(['kappa_target', 'src_type', 'noise_level', 'method']).agg(
        mse_mean=('mse', 'mean'), mse_std=('mse', 'std'),
        rel_err_mean=('rel_err', 'mean'), rel_err_std=('rel_err', 'std'),
        smooth_err_mean=('smooth_err', 'mean'), corr_mean=('corr', 'mean'),
    ).reset_index()

    key = sim_sum[(sim_sum['kappa_target'] == 300) & (sim_sum['noise_level'] == 0.01)]
    for src in SOURCE_TYPES:
        print(f"\n  {SOURCE_LABELS[src]} (κ=300, noise=1%):")
        sub = key[key['src_type'] == src].sort_values('rel_err_mean')
        for _, row in sub.iterrows():
            print(f"    {row['method']:8s} | MSE={row['mse_mean']:.2e} | RelErr={row['rel_err_mean']:.3f} | "
                  f"SmoothErr={row['smooth_err_mean']:.3f} | Corr={row['corr_mean']:.3f}")

    # EN vs Lasso
    print("\n── EN vs Lasso (仿真, noise=1%) ──")
    for src in SOURCE_TYPES:
        print(f"  {SOURCE_LABELS[src]}:")
        for kappa in [50, 100, 200, 300, 500, 1000]:
            en = sim_sum[(sim_sum['src_type'] == src) & (sim_sum['method'] == 'EN') &
                        (sim_sum['noise_level'] == 0.01) & (sim_sum['kappa_target'] == kappa)]
            la = sim_sum[(sim_sum['src_type'] == src) & (sim_sum['method'] == 'Lasso') &
                        (sim_sum['noise_level'] == 0.01) & (sim_sum['kappa_target'] == kappa)]
            if len(en) > 0 and len(la) > 0:
                ratio = en['mse_mean'].values[0] / la['mse_mean'].values[0] if la['mse_mean'].values[0] > 0 else float('inf')
                marker = "✅EN" if ratio < 0.95 else ("❌LA" if ratio > 1.05 else "≈")
                print(f"    κ={kappa:5d} | ratio={ratio:.3f} {marker}")

    # Plot 1: MSE by source type (κ=300)
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    for ax_idx, src in enumerate(SOURCE_TYPES):
        ax = axes1[ax_idx]
        sub = sim_sum[(sim_sum['src_type'] == src) &
                      (sim_sum['noise_level'] == 0.01) &
                      (sim_sum['kappa_target'] == 300)]
        if len(sub) == 0: continue
        mtds = sub['method'].tolist()
        ax.bar(range(len(mtds)), sub['mse_mean'].tolist(), yerr=sub['mse_std'].tolist(),
               color=[METHOD_COLORS[m] for m in mtds], capsize=5, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(mtds)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in mtds], fontproperties=fp, fontsize=9)
        ax.set_ylabel('MSE', fontproperties=fp)
        ax.set_title(f'{SOURCE_LABELS[src]}\n(8×10, κ=300, noise=1%)', fontproperties=fp, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
    fig1.suptitle('轨道一：仿真 8×10 不同源分布重建误差', fontsize=14, fontweight='bold', fontproperties=fp)
    plt.tight_layout()
    p1 = out_dir / f"track1_mse_{timestamp}.png"
    plt.savefig(p1, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n🖼️ {p1}")
    plt.close(fig1)

    # Plot 2: RelErr vs κ
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    for ax_idx, src in enumerate(SOURCE_TYPES):
        ax = axes2[ax_idx]
        for method in METHODS:
            sub = sim_sum[(sim_sum['src_type'] == src) & (sim_sum['noise_level'] == 0.01) & (sim_sum['method'] == method)]
            if len(sub) == 0: continue
            ax.errorbar(sub['kappa_target'], sub['rel_err_mean'], yerr=sub['rel_err_std'],
                       marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                       label=METHOD_LABELS[method], linewidth=1.5, markersize=6, capsize=3)
        ax.set_xlabel('κ', fontproperties=fp)
        ax.set_ylabel('RelErr', fontproperties=fp)
        ax.set_title(SOURCE_LABELS[src], fontproperties=fp, fontweight='bold')
        ax.legend(prop=fp, fontsize=8)
        ax.grid(True, alpha=0.3)
    fig2.suptitle('轨道一：RelErr vs κ (noise=1%)', fontsize=14, fontweight='bold', fontproperties=fp)
    plt.tight_layout()
    p2 = out_dir / f"track1_relerr_vs_kappa_{timestamp}.png"
    plt.savefig(p2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"🖼️ {p2}")
    plt.close(fig2)

    # --- Track 2 Summary ---
    print("\n" + "=" * 90)
    print("📊 轨道二（PHITS 8×12，零列放非零值）")
    print(f"  零列索引: {zero_cols}  非零列索引: {nonzero_cols}")
    print("=" * 90)

    phits_sum = df_phits.groupby(['src_type', 'noise_level', 'method']).agg(
        mse_mean=('mse', 'mean'), mse_std=('mse', 'std'),
        rel_err_mean=('rel_err', 'mean'),
        obs_mse_mean=('obs_mse', 'mean'), unobs_mse_mean=('unobs_mse', 'mean'),
        obs_relerr_mean=('obs_relerr', 'mean'), unobs_relerr_mean=('unobs_relerr', 'mean'),
        corr_mean=('corr', 'mean'),
        unobs_true_norm_mean=('unobs_true_norm', 'mean'),
        unobs_est_norm_mean=('unobs_est_norm', 'mean'),
    ).reset_index()

    key2 = phits_sum[phits_sum['noise_level'] == 0.01]
    for src in SOURCE_TYPES:
        print(f"\n  {SOURCE_LABELS[src]} (noise=1%):")
        sub = key2[key2['src_type'] == src].sort_values('mse_mean')
        for _, row in sub.iterrows():
            print(f"    {row['method']:8s} | TotalMSE={row['mse_mean']:.2e} | "
                  f"ObsMSE={row['obs_mse_mean']:.2e} | UnobsMSE={row['unobs_mse_mean']:.2e} | "
                  f"ObsRelErr={row['obs_relerr_mean']:.3f} | UnobsRelErr={row['unobs_relerr_mean']:.3f}")

    # 不可观测区域局限性
    print("\n── 不可观测区域恢复率 (zero cols) ──")
    for noise in [0.0, 0.01, 0.05]:
        print(f"  noise={noise}:")
        for src in SOURCE_TYPES:
            for method in METHODS:
                row = phits_sum[(phits_sum['src_type'] == src) &
                               (phits_sum['noise_level'] == noise) & (phits_sum['method'] == method)]
                if len(row) > 0:
                    r = row.iloc[0]
                    true_n = r['unobs_true_norm_mean']
                    est_n = r['unobs_est_norm_mean']
                    pct = (est_n / true_n * 100) if true_n > 1e-10 else 0
                    print(f"    {SOURCE_LABELS[src]:12s}-{method:8s} | true_norm={true_n:.3f} est_norm={est_n:.3f} 恢复={pct:.0f}%")

    # Plot 3: Obs vs Unobs MSE
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    for ax_idx, src in enumerate(SOURCE_TYPES):
        ax = axes3[ax_idx]
        sub = key2[key2['src_type'] == src]
        if len(sub) == 0: continue
        x_pos = np.arange(len(METHODS))
        width = 0.35
        obs_mses = [sub[sub['method'] == m]['obs_mse_mean'].values[0] if len(sub[sub['method'] == m]) > 0 else 0 for m in METHODS]
        unobs_mses = [sub[sub['method'] == m]['unobs_mse_mean'].values[0] if len(sub[sub['method'] == m]) > 0 else 0 for m in METHODS]
        ax.bar(x_pos - width/2, obs_mses, width, label='可观测列', color='#2E86AB', alpha=0.8)
        ax.bar(x_pos + width/2, unobs_mses, width, label='不可观测列', color='#F3722C', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontproperties=fp, fontsize=9)
        ax.set_ylabel('MSE', fontproperties=fp)
        ax.set_title(f'{SOURCE_LABELS[src]}\n(PHITS 8×12, noise=1%)', fontproperties=fp, fontweight='bold')
        ax.legend(prop=fp, fontsize=8)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
    fig3.suptitle('轨道二：PHITS 可观测 vs 不可观测区域误差', fontsize=14, fontweight='bold', fontproperties=fp)
    plt.tight_layout()
    p3 = out_dir / f"track2_obs_vs_unobs_{timestamp}.png"
    plt.savefig(p3, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n🖼️ {p3}")
    plt.close(fig3)

    # Plot 4: Example reconstruction (PHITS)
    A_phits = np.load(PHITS_MATRIX_PATH)
    m_p, n_p = A_phits.shape
    A_norm_vis, col_norms_vis = normalize_cols(A_phits)

    fig4, axes4 = plt.subplots(3, 5, figsize=(22, 10), dpi=150)
    rng_vis = np.random.default_rng(777)

    for row_idx, src in enumerate(SOURCE_TYPES):
        run_rng = np.random.default_rng(rng_vis.integers(0, 2**31))
        sparsity = 0.7
        n_nz = max(1, n_p - int(sparsity * n_p))

        if src == 'pure_sparse':
            x_true_vis = generate_pure_sparse(n_p, n_nz, run_rng)
        elif src == 'mixed':
            x_true_vis = generate_mixed_source(n_p, n_nz, run_rng)
        else:
            x_true_vis = generate_smooth_region(n_p, n_nz, run_rng)

        for jc in zero_cols:
            if abs(x_true_vis[jc]) < 1e-10:
                x_true_vis[jc] = run_rng.uniform(0.5, 3.0)

        b_vis = A_phits @ x_true_vis
        b_vis_noisy = b_vis + run_rng.normal(0, 0.01 * (np.std(b_vis) + 1e-10), size=b_vis.shape)

        colors_true = ['#F3722C' if j in zero_cols else '#333333' for j in range(n_p)]
        axes4[row_idx, 0].bar(range(n_p), x_true_vis, color=colors_true, alpha=0.8)
        axes4[row_idx, 0].set_title('真实值(橙=不可观测)', fontproperties=fp, fontsize=9)

        for col_idx, method in enumerate(METHODS):
            if method == 'EN':
                coef = run_elasticnet_light(A_norm_vis, b_vis_noisy, 777)
            elif method == 'Lasso':
                coef = run_lasso_light(A_norm_vis, b_vis_noisy, 777)
            elif method == 'Ridge':
                coef = run_ridge_light(A_norm_vis, b_vis_noisy, 777)
            else:
                coef = run_omp_light(A_norm_vis, b_vis_noisy, n_p)
            x_est = coef / col_norms_vis
            ax = axes4[row_idx, col_idx + 1]
            colors_est = ['#F3722C' if j in zero_cols else METHOD_COLORS[method] for j in range(n_p)]
            ax.bar(range(n_p), x_est, color=colors_est, alpha=0.8)
            re = np.linalg.norm(x_est - x_true_vis) / max(np.linalg.norm(x_true_vis), 1e-10)
            unobs_err = np.linalg.norm(x_est[zero_cols] - x_true_vis[zero_cols])
            ax.set_title(f'{METHOD_LABELS[method]}\nRelErr={re:.3f} Unobs={unobs_err:.2f}', fontproperties=fp, fontsize=8)

        axes4[row_idx, 0].set_ylabel(SOURCE_LABELS[src], fontproperties=fp, fontsize=10, fontweight='bold')

    fig4.suptitle('轨道二：PHITS 8×12 重建示例(橙=不可观测列, noise=1%)',
                  fontsize=14, fontweight='bold', fontproperties=fp)
    plt.tight_layout()
    p4 = out_dir / f"track2_reconstruction_{timestamp}.png"
    plt.savefig(p4, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"🖼️ {p4}")
    plt.close(fig4)

    # --- Cross-track comparison ---
    print("\n" + "=" * 90)
    print("📊 跨轨道对比：仿真(κ=300) vs PHITS, noise=1%")
    print("=" * 90)
    sim_key = df_sim[(df_sim['noise_level'] == 0.01) & (df_sim['kappa_target'] == 300)]
    phits_key = df_phits[df_phits['noise_level'] == 0.01]

    print(f"\n{'':20s} | {'Sim MSE':>12s} | {'PHITS MSE':>12s} | {'Sim RelErr':>12s} | {'PHITS RelErr':>12s}")
    print("-" * 85)
    for src in SOURCE_TYPES:
        for method in METHODS:
            s = sim_key[(sim_key['src_type'] == src) & (sim_key['method'] == method)]
            p = phits_key[(phits_key['src_type'] == src) & (phits_key['method'] == method)]
            s_mse = s['mse'].mean() if len(s) > 0 else 0
            p_mse = p['mse'].mean() if len(p) > 0 else 0
            s_re = s['rel_err'].mean() if len(s) > 0 else 0
            p_re = p['rel_err'].mean() if len(p) > 0 else 0
            label = f"{SOURCE_LABELS[src]}-{method}"
            print(f"{label:20s} | {s_mse:12.2e} | {p_mse:12.2e} | {s_re:12.3f} | {p_re:12.3f}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("方案B 最终版（轻量）：仿真 8×10 + PHITS 8×12")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results/planB_final")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Track 1
    df_sim = run_simulation_track()
    csv1 = out_dir / f"track1_sim_{timestamp}.csv"
    df_sim.to_csv(csv1, index=False)
    print(f"💾 {csv1}")

    # Track 2
    df_phits, zero_cols, nonzero_cols = run_phits_track()
    csv2 = out_dir / f"track2_phits_{timestamp}.csv"
    df_phits.to_csv(csv2, index=False)
    print(f"💾 {csv2}")

    # Analysis & Plots
    analyze_and_plot(df_sim, df_phits, zero_cols, nonzero_cols, out_dir, timestamp)

    elapsed = time.time() - t0
    print(f"\n⏱️ 总用时: {elapsed:.1f}s")
    print(f"💾 所有结果保存在: {out_dir}")
    print("✅ 方案B 最终版实验完成！")


if __name__ == "__main__":
    main()
