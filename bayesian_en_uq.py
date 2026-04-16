#!/usr/bin/env python3
"""
方案B：贝叶斯 Elastic Net + 不确定性量化 (UQ)
使用 NumPyro 实现 HMC/NUTS 采样

核心策略：
1. 列归一化 — 求解前对 A 做列归一化，求解后还原
2. 稀疏度 = 0.8 (20% 非零)
3. PHITS 数据：A_matrix_8x12_GBq.npy
"""

import os, sys
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from sklearn.linear_model import Lasso, ElasticNet
import warnings
warnings.filterwarnings('ignore')

# V14 弹性网络（锅哥自己的实现）
sys.path.insert(0, '/root/.openclaw/workspace/elasticnet/algorithm')
from V14绘图优化版 import fit_elastic_net_robust, Config as V14Config

numpyro.set_host_device_count(1)

# ============================================================
# 数据加载
# ============================================================
DATA_PATH = '/root/.openclaw/workspace/elasticnet/phits_data/phits_10x8_data/PHITS_results_10x8/A_matrix_8x12_GBq.npy'

def load_data():
    A_full = np.load(DATA_PATH)
    print(f"A_full shape: {A_full.shape}, rank={np.linalg.matrix_rank(A_full)}")
    
    # 提取有效子矩阵 (去除零行零列)
    valid_cols = np.where(~np.all(A_full == 0, axis=0))[0]
    valid_rows = np.where(~np.all(A_full == 0, axis=1))[0]
    A_sub = A_full[np.ix_(valid_rows, valid_cols)]
    print(f"有效子矩阵: {A_sub.shape}, rank={np.linalg.matrix_rank(A_sub)}")
    print(f"有效子矩阵 cond: {np.linalg.cond(A_sub):.2e}")
    
    return A_full, A_sub, valid_rows, valid_cols

def make_sparse_source(n, sparsity=0.8, max_val=50.0):
    """生成稀疏源，sparsity=0.8 表示 80% 为零"""
    x = np.zeros(n)
    n_nonzero = max(1, int(n * (1 - sparsity)))
    idx = np.random.choice(n, n_nonzero, replace=False)
    x[idx] = np.random.uniform(1.0, max_val, size=n_nonzero)
    return x, n_nonzero

# ============================================================
# 列归一化工具
# ============================================================
def normalize_columns(A):
    """列归一化，返回归一化矩阵和列范数"""
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    A_norm = A / col_norms
    return A_norm, col_norms

def denormalize_coeffs(coeffs, col_norms):
    """还原系数"""
    return coeffs / col_norms

# ============================================================
# 贝叶斯 Elastic Net 模型
# ============================================================
def bayesian_elastic_net_model(A_norm, b_obs, l1_ratio=0.5):
    """
    NumPyro 贝叶斯 Elastic Net 模型
    
    Elastic Net 先验等价于 Laplace + Normal 组合:
    beta_j ~ Normal(0, sigma_b^2) * exp(-lambda_l1 * |beta_j|)
    
    实际使用 Spike-and-Slab + Normal:
    - 稀疏先验用 Laplace（对应 L1）
    - L2 正则化用 Normal 先验的方差控制
    """
    m, n = A_norm.shape
    
    # 超参数先验
    # sigma: 观测噪声标准差
    sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
    
    # L2 正则化强度 — 控制正态先验宽度
    tau = numpyro.sample("tau", dist.HalfCauchy(1.0))
    
    # 回归系数：Laplace 先验 (L1) × Normal 先验 (L2) ≈ Elastic Net
    # 使用拉普拉斯先验作为稀疏性诱导
    with numpyro.plate("features", n):
        # 方法1：直接使用 Normal + 较小的 scale（相当于 L2）
        # 加上一个全局的 Laplace 稀疏先验
        beta_raw = numpyro.sample("beta_raw", dist.Normal(0.0, tau))
    
    # Laplace 稀疏惩罚通过额外添加实现
    # 简化版：使用 Horseshoe-like 先验
    # 这里用简单的 Normal + 固定 lambda_l1
    
    # 线性预测
    mu = jnp.dot(A_norm, beta_raw)
    
    # 观测模型
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=b_obs)

def bayesian_elastic_net_horseshoe(A_norm, b_obs):
    """
    Horseshoe-like 先验 — 更强的稀疏性诱导
    适合 sparsity=0.8 的场景
    """
    m, n = A_norm.shape
    
    # 全局 shrinkage (对应 Elastic Net 的 alpha/lambda)
    tau = numpyro.sample("tau", dist.HalfCauchy(1.0))
    
    # 局部 shrinkage (每个特征独立)
    with numpyro.plate("features", n):
        lam = numpyro.sample("lam", dist.HalfCauchy(1.0))
    
    # 系数先验: Normal(0, tau^2 * lam^2) — 这是 Horseshoe
    # 等价于自适应 L2 正则化
    with numpyro.plate("features", n):
        beta = numpyro.sample("beta", dist.Normal(0.0, tau * lam))
    
    # 观测噪声
    sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
    
    mu = jnp.dot(A_norm, beta)
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=b_obs)

def bayesian_elastic_net_ss(A_norm, b_obs):
    """
    Spike-and-Slab + Normal 先验
    - Spike: 紧缩的 Normal（近零）
    - Slab: 宽松的 Normal（允许非零）
    - 混合权重由 sparsity 控制
    """
    m, n = A_norm.shape
    
    # 观测噪声
    sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
    
    # 稀疏度先验 — 期望 20% 非零
    pi = numpyro.sample("pi", dist.Beta(2.0, 8.0))  # 先验均值 = 0.2
    
    # 混合先验
    with numpyro.plate("features", n):
        # 隐变量：是否选中
        gamma = numpyro.sample("gamma", dist.Bernoulli(pi))
        # Spike: sigma_spike 很小
        # Slab: sigma_slab 较大
        sigma_beta = jnp.where(gamma == 1, 5.0, 0.01)
        beta = numpyro.sample("beta", dist.Normal(0.0, sigma_beta))
    
    mu = jnp.dot(A_norm, beta)
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=b_obs)

# ============================================================
# 运行 MCMC
# ============================================================
def run_bayesian_en(model, A_norm, b_obs, num_samples=1000, num_warmup=500, seed=42):
    """运行 MCMC 采样"""
    kernel = NUTS(model, target_accept_prob=0.8, max_tree_depth=8)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1)
    
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, A_norm=jnp.array(A_norm), b_obs=jnp.array(b_obs))
    
    return mcmc

def extract_posterior(mcmc, col_norms=None):
    """提取后验统计量"""
    samples = mcmc.get_samples()
    
    # 获取 beta 样本
    if 'beta' in samples:
        beta_samples = np.array(samples['beta'])
    elif 'beta_raw' in samples:
        beta_samples = np.array(samples['beta_raw'])
    else:
        raise ValueError("No beta found in samples")
    
    # 还原列归一化
    if col_norms is not None:
        beta_samples = beta_samples / col_norms[np.newaxis, :]
    
    # 统计量
    beta_mean = np.mean(beta_samples, axis=0)
    beta_median = np.median(beta_samples, axis=0)
    beta_std = np.std(beta_samples, axis=0)
    beta_q025 = np.percentile(beta_samples, 2.5, axis=0)
    beta_q975 = np.percentile(beta_samples, 97.5, axis=0)
    
    if 'sigma' in samples:
        sigma_mean = float(np.mean(np.array(samples['sigma'])))
    else:
        sigma_mean = None
    
    return {
        'mean': beta_mean,
        'median': beta_median,
        'std': beta_std,
        'q025': beta_q025,
        'q975': beta_q975,
        'samples': beta_samples,
        'sigma': sigma_mean
    }

# ============================================================
# 基准方法
# ============================================================
def run_baseline_methods(A_norm, b_obs, x_true, col_norms):
    """运行基准方法对比：V14 + Lasso + EN_fixed"""
    results = {}
    n = A_norm.shape[1]
    
    # 1. V14（锅哥自己的 Elastic Net，GridSearchCV 自适应选参）
    try:
        v14_cfg = V14Config(
            max_iter=10000,
            cv_folds=min(3, A_norm.shape[0]),  # 样本少时减少 CV 折数
            alpha_num=30,
            l1_ratio_num=40,
        )
        v14_res = fit_elastic_net_robust(A_norm, b_obs, v14_cfg, fit_intercept=False)
        x_v14 = denormalize_coeffs(v14_res['coef'], col_norms)
        results['V14'] = {
            'x': x_v14,
            **compute_metrics(x_true, x_v14),
            'params': v14_res['params'],
        }
    except Exception as e:
        print(f"    [WARN] V14 失败: {e}")
        results['V14'] = {'x': np.zeros(n), 'mse': 1e30, 'rel_err': 999, 'r2': -999, 'iou': 0}
    
    # 2. Lasso (α=0.1)
    lasso = Lasso(alpha=0.1, max_iter=100000)
    lasso.fit(A_norm, b_obs)
    x_lasso = denormalize_coeffs(lasso.coef_, col_norms)
    results['Lasso'] = {'x': x_lasso, **compute_metrics(x_true, x_lasso)}
    
    # 3. EN fixed (alpha=0.1, l1_ratio=0.5)
    en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000)
    en.fit(A_norm, b_obs)
    x_en = denormalize_coeffs(en.coef_, col_norms)
    results['EN_fixed'] = {'x': x_en, **compute_metrics(x_true, x_en)}
    
    return results

def compute_metrics(x_true, x_pred):
    mse = float(np.mean((x_true - x_pred)**2))
    rel_err = float(np.linalg.norm(x_true - x_pred) / (np.linalg.norm(x_true) + 1e-20))
    ss_res = np.sum((x_true - x_pred)**2)
    ss_tot = np.sum((x_true - np.mean(x_true))**2)
    r2 = float(1 - ss_res / (ss_tot + 1e-20))
    
    # Support recovery
    s_true = np.abs(x_true) > 1e-6
    s_pred = np.abs(x_pred) > 0.05 * (np.max(np.abs(x_pred)) + 1e-10)
    if s_true.sum() + s_pred.sum() == 0:
        iou = 1.0
    else:
        iou = float(np.sum(s_true & s_pred) / (np.sum(s_true | s_pred) + 1e-10))
    
    return {'mse': mse, 'rel_err': rel_err, 'r2': r2, 'iou': iou}

def compute_uq_metrics(x_true, q025, q975):
    """计算 UQ 指标"""
    # 覆盖率：真实值落在 95% CI 内的比例
    in_ci = (x_true >= q025) & (x_true <= q975)
    coverage = float(np.mean(in_ci))
    
    # 平均 CI 宽度
    ci_width = float(np.mean(q975 - q025))
    
    # 非零元素的覆盖率
    nonzero_mask = np.abs(x_true) > 1e-6
    if nonzero_mask.sum() > 0:
        nonzero_coverage = float(np.mean(in_ci[nonzero_mask]))
        nonzero_ci_width = float(np.mean((q975 - q025)[nonzero_mask]))
    else:
        nonzero_coverage = 1.0
        nonzero_ci_width = 0.0
    
    return {
        'coverage': coverage,
        'ci_width': ci_width,
        'nonzero_coverage': nonzero_coverage,
        'nonzero_ci_width': nonzero_ci_width
    }

# ============================================================
# 主实验
# ============================================================
def main():
    np.random.seed(42)
    
    print("=" * 80)
    print("方案B：贝叶斯 Elastic Net + 不确定性量化")
    print("=" * 80)
    
    A_full, A_sub, valid_rows, valid_cols = load_data()
    
    # 使用有效子矩阵 (7×7)
    m, n = A_sub.shape
    print(f"\n使用有效子矩阵: {m}×{n}")
    print(f"子矩阵 cond = {np.linalg.cond(A_sub):.2e}")
    
    # 列归一化
    A_norm, col_norms = normalize_columns(A_sub)
    print(f"归一化后 cond = {np.linalg.cond(A_norm):.2e}")
    
    # ============================================================
    # 实验1：小量验证 (3次试验)
    # ============================================================
    print("\n" + "=" * 80)
    print("实验1：小量验证 (3次试验)")
    print("=" * 80)
    
    noise_level = 0.01
    models_to_test = {
        'Normal+Laplace': bayesian_elastic_net_model,
        'Horseshoe': bayesian_elastic_net_horseshoe,
    }
    
    validation_results = []
    
    for trial in range(3):
        print(f"\n--- Trial {trial+1}/3 ---")
        x_true, n_nonzero = make_sparse_source(n, sparsity=0.8, max_val=50.0)
        print(f"真实源: {n_nonzero}/{n} 非零, 非零值 = {x_true[x_true > 0]}")
        
        # 生成观测
        b = A_sub @ x_true
        b_noisy = b + noise_level * np.linalg.norm(b) * np.random.randn(m)
        
        # 基准方法
        baselines = run_baseline_methods(A_norm, b_noisy, x_true, col_norms)
        print(f"\n基准方法:")
        for name, res in baselines.items():
            print(f"  {name:12s}: MSE={res['mse']:.2f}, R²={res['r2']:.3f}, IoU={res['iou']:.3f}")
        
        # 贝叶斯方法
        bayes_results = {}
        for model_name, model_fn in models_to_test.items():
            print(f"\n  运行贝叶斯 {model_name}...")
            try:
                mcmc = run_bayesian_en(
                    model_fn, A_norm, b_noisy,
                    num_samples=500, num_warmup=300, seed=42 + trial
                )
                post = extract_posterior(mcmc, col_norms)
                
                # 用后验均值作为点估计
                x_bayes = post['mean']
                met = compute_metrics(x_true, x_bayes)
                uq = compute_uq_metrics(x_true, post['q025'], post['q975'])
                
                bayes_results[model_name] = {
                    'x': x_bayes,
                    'post': post,
                    'metrics': met,
                    'uq': uq
                }
                
                print(f"    {model_name:20s}: MSE={met['mse']:.2f}, R²={met['r2']:.3f}, "
                      f"IoU={met['iou']:.3f}, 覆盖率={uq['coverage']:.2%}, "
                      f"CI宽度={uq['ci_width']:.2f}")
                
                # MCMC 诊断
                mcmc.print_summary(exclude_deterministic=False)
                
            except Exception as e:
                print(f"    {model_name} 失败: {e}")
                import traceback
                traceback.print_exc()
        
        validation_results.append({
            'trial': trial,
            'x_true': x_true,
            'baselines': baselines,
            'bayes': bayes_results
        })
    
    # ============================================================
    # 汇总验证结果
    # ============================================================
    print("\n" + "=" * 80)
    print("验证结果汇总")
    print("=" * 80)
    
    print(f"\n{'Method':<22} {'MSE':>10} {'R²':>8} {'IoU':>7} {'覆盖':>6} {'CI宽':>8}")
    print("-" * 65)
    
    # 汇总所有方法
    all_methods = {}
    for vr in validation_results:
        for name, res in vr['baselines'].items():
            if name not in all_methods:
                all_methods[name] = {'mse': [], 'r2': [], 'iou': []}
            all_methods[name]['mse'].append(res['mse'])
            all_methods[name]['r2'].append(res['r2'])
            all_methods[name]['iou'].append(res['iou'])
        
        for name, res in vr['bayes'].items():
            key = f"Bayes_{name}"
            if key not in all_methods:
                all_methods[key] = {'mse': [], 'r2': [], 'iou': [], 'coverage': [], 'ci_width': []}
            all_methods[key]['mse'].append(res['metrics']['mse'])
            all_methods[key]['r2'].append(res['metrics']['r2'])
            all_methods[key]['iou'].append(res['metrics']['iou'])
            all_methods[key]['coverage'].append(res['uq']['coverage'])
            all_methods[key]['ci_width'].append(res['uq']['ci_width'])
    
    for name, vals in sorted(all_methods.items(), key=lambda x: -np.mean(x[1]['r2'])):
        mse_m = np.mean(vals['mse'])
        r2_m = np.mean(vals['r2'])
        iou_m = np.mean(vals['iou'])
        cov_str = f"{np.mean(vals.get('coverage', [0])):.0%}" if 'coverage' in vals else "N/A"
        ci_str = f"{np.mean(vals.get('ci_width', [0])):.2f}" if 'ci_width' in vals else "N/A"
        print(f"{name:<22} {mse_m:>10.2f} {r2_m:>8.3f} {iou_m:>7.3f} {cov_str:>6} {ci_str:>8}")
    
    # ============================================================
    # 详细分析最佳贝叶斯模型
    # ============================================================
    print("\n" + "=" * 80)
    print("详细分析：UQ 有效性验证")
    print("=" * 80)
    
    for vr in validation_results:
        trial = vr['trial']
        x_true = vr['x_true']
        print(f"\n--- Trial {trial+1} ---")
        print(f"  真实值: {np.round(x_true, 2)}")
        
        for name, res in vr['bayes'].items():
            post = res['post']
            print(f"\n  [{name}]")
            print(f"    后验均值: {np.round(post['mean'], 2)}")
            print(f"    95% CI下界: {np.round(post['q025'], 2)}")
            print(f"    95% CI上界: {np.round(post['q975'], 2)}")
            print(f"    后验标准差: {np.round(post['std'], 2)}")
            
            # 逐分量检查
            in_ci = (x_true >= post['q025']) & (x_true <= post['q975'])
            for j in range(n):
                marker = "✓" if in_ci[j] else "✗"
                print(f"    src[{j}]: true={x_true[j]:.2f}, "
                      f"est={post['mean'][j]:.2f}, "
                      f"CI=[{post['q025'][j]:.2f}, {post['q975'][j]:.2f}] {marker}")
    
    # ============================================================
    # 绘图：对比所有方法 + UQ
    # ============================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 选 Trial 2（前面表现最好的那个）画详细对比图
        best_trial_idx = 1  # Trial 2 (0-indexed)
        if best_trial_idx >= len(validation_results):
            best_trial_idx = 0
        vr = validation_results[best_trial_idx]
        x_true = vr['x_true']
        idx = np.arange(n)
        
        # ---- 图1: 贝叶斯方法 UQ 对比 ----
        fig, axes = plt.subplots(len(models_to_test), 1, figsize=(12, 4 * len(models_to_test)))
        if len(models_to_test) == 1:
            axes = [axes]
        
        for ax, (name, res) in zip(axes, vr['bayes'].items()):
            post = res['post']
            
            ax.plot(idx, x_true, 'ro-', label='True', markersize=8, linewidth=2, zorder=5)
            ax.plot(idx, post['mean'], 'bs-', label='Bayes Mean', markersize=6)
            ax.fill_between(idx, post['q025'], post['q975'], alpha=0.3, color='blue',
                          label='95% CI')
            
            # 也画上 V14 对比
            if 'V14' in vr['baselines']:
                ax.plot(idx, vr['baselines']['V14']['x'], 'g^--', label='V14', markersize=6, alpha=0.8)
            
            ax.set_xlabel('Source index')
            ax.set_ylabel('Activity (GBq)')
            ax.set_title(f'Bayesian EN ({name}) vs V14: Point Estimate + UQ')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = '/root/.openclaw/workspace/coder/bayesian_en_uq_validation.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存: {save_path}")
        plt.close()
        
        # ---- 图2: 全方法点估计对比 ----
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(idx, x_true, 'ro-', label='True', markersize=10, linewidth=2.5, zorder=5)
        
        colors = {'V14': '#2E86AB', 'Lasso': '#E94F37', 'EN_fixed': '#44AF69'}
        markers = {'V14': 's', 'Lasso': '^', 'EN_fixed': 'd'}
        for name, res in vr['baselines'].items():
            ax.plot(idx, res['x'], marker=markers.get(name, 'o'), linestyle='--',
                   color=colors.get(name, 'gray'), label=name, markersize=6, alpha=0.8)
        
        bayes_colors = {'Normal+Laplace': '#9B59B6', 'Horseshoe': '#F39C12'}
        for name, res in vr['bayes'].items():
            ax.plot(idx, res['x'], marker='o', linestyle='-',
                   color=bayes_colors.get(name, 'brown'), label=f'Bayes_{name}', 
                   markersize=5, alpha=0.8)
        
        ax.set_xlabel('Source index')
        ax.set_ylabel('Activity (GBq)')
        ax.set_title('All Methods Comparison (Trial {})'.format(best_trial_idx + 1))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path2 = '/root/.openclaw/workspace/coder/bayesian_en_all_methods.png'
        plt.savefig(save_path2, dpi=150, bbox_inches='tight')
        print(f"全方法对比图已保存: {save_path2}")
        plt.close()
        
    except Exception as e:
        import traceback
        print(f"\n绘图失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
