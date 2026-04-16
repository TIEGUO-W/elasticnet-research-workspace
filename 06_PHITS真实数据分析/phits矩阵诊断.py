import numpy as np
from sklearn.linear_model import ElasticNet, Ridge

A_full = np.load('/root/.openclaw/workspace/elasticnet/phits_data/phits_10x8_data/PHITS_results_10x8/A_matrix_8x12_GBq.npy')

nonzero_cols = [1, 5, 6, 7, 8, 9, 10]
valid_rows = [0, 1, 2, 3, 5, 6, 7]
A_sub = A_full[np.ix_(valid_rows, nonzero_cols)]

print('=== 不做列归一化 vs 归一化 ===')

np.random.seed(42)
x_true = np.zeros(7)
x_true[0] = 50.0  # 列1
x_true[3] = 30.0  # 列7
x_true[4] = 20.0  # 列8

b = A_sub @ x_true
b_noisy = b + 0.01 * np.linalg.norm(b) * np.random.randn(7)

print(f'b (观测): {b}')
print(f'b 范数: {np.linalg.norm(b):.4f}')
print()

# 不归一化
en = ElasticNet(alpha=0.001, l1_ratio=0.3, max_iter=100000)
en.fit(A_sub, b_noisy)
x_pred = en.coef_
print('EN (不归一化, a=0.001):')
print(f'  预测: {np.array2string(x_pred, precision=4)}')
print(f'  真实: {np.array2string(x_true, precision=4)}')
residual = np.linalg.norm(A_sub @ x_pred - b_noisy)
print(f'  残差: {residual:.6f}')

# 归一化
col_norms = np.linalg.norm(A_sub, axis=0)
col_norms[col_norms == 0] = 1
A_norm = A_sub / col_norms
en2 = ElasticNet(alpha=0.001, l1_ratio=0.3, max_iter=100000)
en2.fit(A_norm, b_noisy)
x_pred2 = en2.coef_ / col_norms
print('\nEN (归一化, a=0.001):')
print(f'  预测: {np.array2string(x_pred2, precision=4)}')
print(f'  真实: {np.array2string(x_true, precision=4)}')
print(f'  列范数: {np.array2string(col_norms, precision=6)}')

# 重构误差
print(f'\n不归一化重构误差: {np.linalg.norm(x_true - x_pred):.4f}')
print(f'归一化重构误差: {np.linalg.norm(x_true - x_pred2):.4f}')

# 条件数
print(f'\n原始子矩阵 cond: {np.linalg.cond(A_sub):.2e}')
print(f'归一化后 cond: {np.linalg.cond(A_norm):.2e}')

# 每列贡献
print('\n=== 每列对观测的贡献 ===')
for i, col_idx in enumerate(nonzero_cols):
    col = A_sub[:, i]
    contrib = col * x_true[i]
    print(f'  列{col_idx}(子矩阵{i}): x={x_true[i]:.1f}, A列范数={np.linalg.norm(col):.6f}, 贡献范数={np.linalg.norm(contrib):.6f}')

# 核心诊断：列之间的相关性
print('\n=== 子矩阵列间相关性 (归一化后) ===')
corr = np.corrcoef(A_norm.T)
print(np.array2string(corr, precision=3, suppress_small=True))

# SVD 分析
print('\n=== SVD 分析 ===')
U, s, Vt = np.linalg.svd(A_sub)
print(f'奇异值: {np.array2string(s, precision=6)}')
print(f'条件数: {s[0]/s[-1]:.2e}')
print(f'有效秩(>1e-10*s1): {np.sum(s > 1e-10 * s[0])}')

# 用零空间分析：哪些列是线性相关的？
print('\n=== 线性相关性分析 ===')
print('7列子矩阵的秩=6，说明有1个线性相关关系')
# 零空间
_, s_val, Vt_val = np.linalg.svd(A_sub)
null_space = Vt_val[6:, :]  # 第7个右奇异向量
print(f'零空间向量: {np.array2string(null_space[0], precision=6)}')
print('含义: 这个向量对应的线性组合使 A·v=0')
# 看哪个分量最大
max_idx = np.argmax(np.abs(null_space[0]))
print(f'最大分量: 索引{max_idx} (对应原列{nonzero_cols[max_idx]}), 值={null_space[0][max_idx]:.6f}')
