# 配置导航

`paper_fifo_cpu_fixture.json` 是 J3O/USHER 与主阶段 FIFO 接线的手算软件样例（不是 GPU 测量）；使用和真实画像接口见[落实说明](../docs/paper_mechanism_integration_20261102.md)。

| 配置 | 用途与依赖 |
| --- | --- |
| `system_reference_v2_20261103.json` | 当前真实四任务正式入口；固定参考 v2，独立参考测量待冻结，暂不产生排名 |
| `utility_objective_reference_v2_20261103.json` | 当前六项公式、权重、来源与参考文件契约 |
| `normalization_reference_v2_20261103.json` | 已声明开发集质量锚点；D_t 和完整 episode 基准能量为 null，状态 pending |
| `paper_fifo_reference_v2_cpu_fixture.json` | 完整新公式与优化器的合成 CPU 样例；不是物理证据 |
| `resource_spillover_study_reference_v2_20261103.json` | 当前资源研究 CPU 准备入口；配套独立 workload/horizon 参考契约 |
| `demo.json` | CPU mock 快速开始与 CI；输入为 `examples/demo_samples.jsonl` |
| `continuous_fixture.json` | 独立连续路由 Phase 1 配置；合成共驻/质量/资源原语，只产生 CPU 算法证据 |
| `utility_objective_20261101.json` | 历史 2026-11-01 四任务目标；六项等权，四任务 b_t 均为 1000ms，属于共同研究尺度 |
| `system_one_second_20261101.json` | 既有静态 calibrated_replay_v2 域使用历史 1 秒尺度的配置；依赖本地审计目录中提取的 catalog，不含新搜索结论 |
| `resource_spillover_study_20261101.json` | 历史资源控制准备入口，使用 1 秒尺度；原物理域与未知项不变 |
| `catalogs/`、`low18.json`、`holdout.json`、`asr32b.json`、`text_half_mid18.json` | 历史模型池与静态规划回归 |
| `static_replicas.json` | 历史实测副本域；不代表当前四任务已验证画像 |
| `workers.example.json`、`admission.example.json`、`prompts.json` | worker、准入与提示模板示例 |
| `study*.json` | 四任务候选、数据/模型版本和内存变体的研究协议 |
| `compact_protocol*.json` | 紧凑模型协议与修订版本 |
| `compact_execution*.json` | 原生 worker 执行、内存与输入行变体 |
| `compact_environments*.json` | 对应独立 Python 环境与锁的声明 |
| `environments/` | 可复现依赖输入、带哈希锁、约束及兼容补丁 |
| `resource_spillover_*_20261024.json` | 历史资源控制与六项等权 spillover 实验设计；当前参考值见 reference_v2_20261103 版 |
| `baichuan_no_synthesis_20261021.json` | Baichuan 资源与非合成输入核验声明 |

`demo.json` 和历史规划配置可用于基础 CPU 环境。研究配置中的绝对路径、
`artifacts/` 来源、模型与数据均需在研究环境准备并通过相应哈希检查；
提交配置不代表提交了这些资源，也不代表真机质量/性能已通过验证。

修订版和旧版文件均可能被已冻结凭据引用，不按文件名合并或覆盖。
新增环境锁应一并保留生成命令、约束和版本身份。

归一化公式的文献依据与自定义边界见[文献核查](../docs/utility_normalization_literature_20261101.md)。当前使用 [reference v2](../docs/utility_normalization_switch_20261103.md)；历史协议、冻结实测与合成 fixture 不作覆盖式改写。
