# 04 — HPP 可微 Elastic Net

## 要回答的问题

> 用 Hadamard Overparametrization（β = u⊙v）让 Elastic Net 完全可微，能否比标准 sklearn EN 更好？能否嵌入端到端训练？

## 什么是 HPP

核心思想来自论文：*Smoothing the Edges* (Machine Learning, 2026-04-02)
- 传统 Elastic Net 的 L1 项不可微 → 需要近端梯度/坐标下降
- HPP 变换：β = u ⊙ v，正则化变成 ||u||² + ||v||² → 完全可微
- 可以用标准梯度下降（Adam/SGD）求解

## 文件说明

| 文件 | 说明 |
|------|------|
| `hpp_v1_原型.py` | 第一版实现：基本 HPP + 梯度下降 + 多起点 |
| `hpp_v2_改进版.py` | 第二版：加入梯度裁剪 + 列归一化 |
| `hpp_v3_双轨验证版.py` | 第三版：curriculum training（预热→稀疏→精调）+ 合成/PHITS双轨 |
| `hpp_PHITS重测_绝对阈值.py` | 用绝对阈值重新评估 PHITS 恢复率（回答"7个有效列恢复几个"） |

## 核心发现

**合成数据（Cond=300）**：HPP 与标准 EN 持平，无显著优势
**PHITS 数据**：与标准 EN 一样只能恢复 3/7 个有效列

## 结论

- HPP 可微化本身正确，但在小规模（7×7）问题上体现不出优势
- 真正的价值在 end-to-end 训练（EN重建 + NN后处理），需要更大规模数据
- **暂不纳入论文核心，作为未来方向储备**
