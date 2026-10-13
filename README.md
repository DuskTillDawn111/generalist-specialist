# Generalist–Specialist

研究单 GPU 上通用模型与专用模型的联合部署、请求路由和性能优化。给定六类任务流量、质量准入与显存预算，算法选择模型及副本数，分配各任务流量，再通过实际请求日志验证性能。

本仓库保留分阶段研究代码与冻结证据。当前没有一个入口能代表全部算法；实验编号用于追踪阶段，不能用编号大小判断算法是否已验证或已成为正式基线。

## 从哪里开始

| 需要了解什么 | 入口 |
| --- | --- |
| 整体架构、算法路线及数据流 | [架构说明](docs/architecture.md) |
| 每个阶段的源码、入口、依赖和已有 Gate | [实验索引](docs/experiment_index.md) |
| 克隆后缺哪些材料、如何使用既有环境 | [复现说明](docs/reproduction.md) |
| Git 跟踪与后续整理规则 | [仓库管理约定](docs/repository_management.md) |
| 可供工具读取的源码清单与 SHA256 | [仓库清单](docs/repository_inventory.json) |

## 算法入口

| 路线 | 主要实现 | 边界 |
| --- | --- | --- |
| 可行部署初始化 | [MILP seeder](deployment_core/milp_seeder.py) | 最小显存可行 seed，性能优化还需后续搜索 |
| 部署搜索与冻结基线 | [deployment_search](deployment_search/README.md)、[MethodV2](method_v2/integer_search.py)、[冻结 V3](module9bc/frozen_v3.py) | 保留各自候选生成、排序、接受条件与历史身份 |
| 拥塞代理 J 的路由优化 | [E2E router](deployment_core/routing/e2e_router.py) | 代理目标的数值证书不等于真机服务保证 |
| 综合效用 U 的候选生成 | [Module20](module20/README.md) | MILP 有限候选生成与 FIFO 预测；预测器不足的历史 Gate 保留 |
| Poisson 排队与路由相关服务预测 | [Module21](module21/README.md)、[Module25 predictor](module25/predictor.py) | 与冻结 MethodV2 分开；预测支持范围需要检查 |
| 在线分流 | [Module28 router](results/module28_dynamic_split/dynamic_router.py)、[VQA-aware router](results/module32b_vqa_aware_routing/vqa_aware_dynamic_router.py) | 独立运行时策略；工程验收和效用证据分别记录 |
| 较新 U 优化实验 | [Module35 optimizer](results/module35_utility_routing/u_aligned_optimizer.py)、[阶段摘要](results/module35_utility_routing/module35_summary.md) | K=1 的任务子集/Pareto 动态规划；K>1 在固定部署上做有界离散路由搜索 |

Module33–35 分别包含 Mixed Oracle 比较、候选池扩展和 U 对齐优化。Module35 的有限域 CPU 验证不能覆盖连续域或所有真实 GPU 条件，也不会自动升级历史算法与预测器的 Gate。

## 目录导航

| 目录 | 用途 |
| --- | --- |
| `benchmark/` | 数据准备、质量评测、单模型与共驻留采样 |
| `candidate_pool_*`、`quality_workload_audit/` | 模型池登记、质量准入、服务路径与画像审计 |
| `replica_scaling/`、`replica_capacity_calibration/` | 独立副本测量、经验容量与显存标定 |
| `deployment_core/` | 数据结构、模型目录、MILP/LP、路由与约束校验 |
| `deployment_search/`、`method_v1/`、`method_v2/` | 搜索、反馈、候选选择、运行时适配与冻结基线 |
| `performance_model/` | 性能模型基础接口、采样、泛化、校准与排名实验 |
| `joint_deploy/` | 模型 worker、进程通信、测量框架与负载生成 |
| `placement/` | 早期放置实现与历史原型，参见其 README |
| `module*/` | 各阶段实验实现，暂保留原路径 |
| `results/` | 筛选后的证据与摘要，也包含多轮实验源码和冻结源码副本 |
| `docs/`、`tools/` | 项目导航、实验索引与维护工具 |

`models/`、`datasets/`、`envs/`、缓存以及大规模原始测量材料保留在研究工作区，不进入 Git。**本仓库是源码与精选元数据仓库，不是完整原始数据归档。**

## 安全的查阅与维护入口

以下命令不加载模型、不启动 GPU 实验：

```bash
python tools/repository_inventory.py --check
python tools/repository_inventory.py --write
git status --short
```

`--write` 只更新实验索引与仓库清单；`--check` 检查它们是否与当前可跟踪源码及元数据一致。

训练/推理环境因模型而异。CPU 规划主要使用 NumPy/SciPy；worker 通过注册表调用既有专属 Python 环境。`pyproject.toml` 提供现有源码的包描述，不统一安装所有 GPU 依赖。执行实验前请先阅读[复现说明](docs/reproduction.md)。

## 第一轮整理范围

建立 Git 基线、跟踪规则和项目索引；新增文档与维护工具。历史算法、实验路径、冻结权重、准入条件、Gate 和原始结果保持原样。公共库抽取及历史归档属于后续独立改动，见[管理约定](docs/repository_management.md)。
