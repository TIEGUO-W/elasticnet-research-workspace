# 当前最重要上下文

> 最后更新：2026-04-16 11:24

---

## 🔴 最高优先级规则：Deep-Reading 不可跳过

**所有调研和阅读任务必须先启动 Deep-Reading！**

---

## ⏰ 任务：14:10 PHITS 真实数据验证（subagent）

**模式**：subagent（depth 1/1），完成后自动回报

---

## 📋 当前任务

**PHITS 真实数据验证方案A**

### 有效数据（唯一）
- `A_matrix_8x12_GBq.npy`：8det×12src, rank=6, 5个零列, 1个零行
- 有效子矩阵 7×7, cond≈1e20

### 验证结果摘要（2026-04-16 14:15）
- **所有方法 R² 均为负** — 矩阵极度欠条件（cond~1e20）
- **排名（全矩阵，1%噪声）**：EN_fixed ≈ Ridge > EN_dynamic > EN_CV > Lasso > OMP >> Pinv
- **固定α vs 动态α**：固定α 在所有噪声水平下均优于动态α
- **Pinv/OMP 灾难性失败**：条件数过大导致数值爆炸
- **核心发现**：cond~1e20 远超论文 Cond~300 场景，当前数据不适合直接验证

### 关键结论
1. PHITS 8×12 矩阵条件数过高（1e20 vs 论文300），需要更多探测器或更好的几何布局
2. 固定α EN 在极端条件下最稳健（与Ridge持平）
3. 动态α 在此场景下不如固定α — 迭代调整在极低信噪比时反而引入不稳定
4. 验证脚本：`/root/.openclaw/workspace/coder/phits_validation_v2.py`

---

## 🔬 项目哲学逻辑

### 核心问题
**欠条件矩阵稀疏恢复** — 放射源重建与辐射探测应用
- 矩阵条件数 Cond~300（高条件数导致求解不稳定）
- 传统方法（OMP、Lasso）在高条件数下性能急剧下降
- Elastic Net 的 L2 项能改善条件数，L1 项保持稀疏性

### 技术路线哲学
**"不与端到端深度学习正面竞争，而是走务实路线"**

FBSEM 范式证明：
> "U-Net post-denoising can perform as good as DL reconstruction"

**我们的路线**：
```
Elastic Net 重建（物理驱动，可解释）
    ↓
轻量 NN 后处理（数据驱动校正）
    ↓
≈ 端到端 DL 质量，但可解释性更强
```

### 为什么 Elastic Net 适合这个场景
1. **L2 项改善条件数** — 高条件数矩阵的"救星"
2. **分组效应** — 相关特征一起被选中
3. **比 OMP/Lasso 更鲁棒** — 低 SNR 环境下稳定
4. **学术空白** — PET 重建领域几乎没有竞争者

---

## 🔬 核心研究方向

**Elastic Net 用于欠条件矩阵稀疏恢复**

- 欠条件矩阵：Cond~300 场景
- 目标：放射源重建与辐射探测应用
- 仓库：git@github.com:TIEGUO-W/elasticnet-chapter5.git

**核心代码（V14）**：
- `/root/.openclaw/workspace/coder/V14绘图优化版.py`

**关键发现（待实现）**：
1. **动态 ElasticNet 正则化**：α(k) 动态调度（MSE 超越固定权重）
2. **贝叶斯 UQ**：Elastic Net → 贝叶斯 Elastic Net，输出 posterior + credible intervals
3. **FBSEM 范式**：Elastic Net 重建 + 轻量 NN 后处理
4. **HPP Smooth Elastic Net**：Hadamard Overparametrization 可微分化

---

## ⚙️ 技术栈

- **主要语言**：Python（NumPy/SciPy）
- **实验框架**：MATLAB（对比基准）
- **蒙卡模拟**：PHITS（超算 192.168.1.139）
- **版本控制**：GitHub

---

## ⚙️ 记忆规则

- 每次任务完成后更新本文件
- 主动召回，不要等用户提醒
