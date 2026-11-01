# 配置导航

| 配置 | 用途与依赖 |
| --- | --- |
| `demo.json` | CPU mock 快速开始与 CI；输入为 `examples/demo_samples.jsonl` |
| `catalogs/`、`low18.json`、`holdout.json`、`asr32b.json`、`text_half_mid18.json` | 历史模型池与静态规划回归 |
| `static_replicas.json` | 历史实测副本域；不代表当前四任务已验证画像 |
| `workers.example.json`、`admission.example.json`、`prompts.json` | worker、准入与提示模板示例 |
| `study*.json` | 四任务候选、数据/模型版本和内存变体的研究协议 |
| `compact_protocol*.json` | 紧凑模型协议与修订版本 |
| `compact_execution*.json` | 原生 worker 执行、内存与输入行变体 |
| `compact_environments*.json` | 对应独立 Python 环境与锁的声明 |
| `environments/` | 可复现依赖输入、带哈希锁、约束及兼容补丁 |
| `resource_spillover_*_20261024.json` | 资源控制与六项等权 spillover 实验设计 |
| `baichuan_no_synthesis_20261021.json` | Baichuan 资源与非合成输入核验声明 |

`demo.json` 和历史规划配置可用于基础 CPU 环境。研究配置中的绝对路径、
`artifacts/` 来源、模型与数据均需在研究环境准备并通过相应哈希检查；
提交配置不代表提交了这些资源，也不代表真机质量/性能已通过验证。

修订版和旧版文件均可能被已冻结凭据引用，不按文件名合并或覆盖。
新增环境锁应一并保留生成命令、约束和版本身份。
