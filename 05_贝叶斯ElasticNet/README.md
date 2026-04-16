# 05 — 贝叶斯 Elastic Net + 不确定性量化

## 要回答的问题

> 能否给 Elastic Net 的解加上置信区间（credible intervals）？从"找到最稀疏解"升级为"量化解的不确定性"？

## 两条实现路线

| 文件 | 路线 | 核心方法 |
|------|------|---------|
| `贝叶斯EN对比_scipy版.py` | Laplace 近似 | MAP≈EN + Hessian→协方差→采样→CI |
| `贝叶斯EN_UQ_NumPyro版.py` | 完整贝叶斯 | NumPyro + NUTS 采样器 + Laplace+Gaussian 混合先验 |
| `贝叶斯EN_v2_改进.py` | scipy 优化版 | 改进的先验设置和优化策略 |

## 图片说明

| 文件 | 说明 |
|------|------|
| `图_贝叶斯EN验证结果.png` | 贝叶斯 EN 的点估计 + 95% 置信区间可视化 |

## 核心发现

- MAP 估计与 sklearn EN 完全一致（理论预期）
- 在 PHITS 数据（cond=3.1e21）上，CI 覆盖度不可靠（条件数太大）
- **需要在合成数据（Cond~300）上验证后再推广**

## 创新价值

- 放射源重建 + 贝叶斯UQ：**文献空白**
- 论文方向储备，优先级高于 HPP
