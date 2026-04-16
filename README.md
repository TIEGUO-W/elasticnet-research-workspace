# Elastic Net 稀疏恢复研究 — 代码项目组工作仓库

> 项目：Elastic Net 回归算法用于欠条件矩阵稀疏恢复（放射源重建与辐射探测应用）

## 仓库结构

```
├── AGENTS.md              # 项目管理规则
├── SOUL.md                # 角色定义
├── memory/                # 记忆与报告
│   ├── CURRENT.md         # 当前上下文
│   └── REPORT-*.md        # 每日研究报告
├── results/               # 实验输出（CSV/PNG）
│   ├── planA/             # 方案A：动态α vs 固定α
│   └── planB_final/       # 方案B：混合源分布重建
├── V14绘图优化版.py         # 原始基线代码
├── planA_dynamic_alpha.py  # 方案A：动态α对比实验
├── planB_final.py          # 方案B：双轨验证（合成+PHITS）
├── planB_mixed_source.py   # 方案B：混合源分布
├── phits_validation.py     # PHITS真实数据验证
├── phits_validation_v2.py  # PHITS验证v2
├── phits_diag.py           # PHITS矩阵诊断
├── phits_recovery_check.py # PHITS恢复率检查
├── hpp_elastic_net_prototype.py  # HPP Elastic Net v1
├── hpp_elastic_net_v2.py         # HPP v2 改进
├── hpp_elastic_net_v3.py         # HPP v3 + 合成数据双轨
├── hpp_phits_retest.py           # HPP PHITS重测
├── bayesian_en_comparison.py     # 贝叶斯EN对比（scipy）
├── bayesian_en_uq.py             # 贝叶斯EN + NumPyro UQ
└── bayesian_en_v2.py             # 贝叶斯EN v2
```

## 核心数据规格

- **PHITS 矩阵**：8det × 12src, 7个有效列, cond=3.1e21
- **有效列**：[1, 5, 6, 7, 8, 9, 10]（强列: 1,7,8 | 弱列: 5,6,9,10）
- **论文目标场景**：Cond~300 合成数据

## 核心发现（2026-04-16）

1. PHITS 7个有效列中恢复3个强列（100%），弱列不可恢复
2. 固定α EN 最稳健，动态α(logistic)被否决
3. 贝叶斯EN + HPP 方向技术可行，待合成数据验证

## 团队

- **锅哥**（WangKaiQi）— 项目负责人 / 研究员
- **coder** — 代码项目经理（AI）

---

*数据路径：PHITS 数据位于 `phits_data/phits_10x8_data/`（未含入仓库，请单独获取）*
