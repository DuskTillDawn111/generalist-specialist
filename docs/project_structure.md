# 项目结构与 Git 维护

## 目录职责

| 路径 | 内容与维护方式 |
| --- | --- |
| `gsdeploy/` | 唯一安装包；公开 API、规划/评分与共享实现 |
| `gsdeploy/backends/` | mock、原生模型、batch 与共享权重适配 |
| `gsdeploy/runtime/` | worker、传输、准入、请求生命周期与遥测 |
| `gsdeploy/profiling/` | 输入准备、画像、环境及证据检查 |
| `gsdeploy/forecasting/` | 到达流、队列、生命周期及 kernel 预测组件 |
| `configs/` | demo、历史 catalog、当前研究协议和独立环境锁，见[配置导航](../configs/README.md) |
| `examples/` | 随仓库分发的最小示例输入 |
| `experiments/` | 使用公开接口的候选池比较与原生验收 |
| `scripts/` | 可复用的操作入口，见[脚本导航](../scripts/README.md) |
| `tools/` | 按研究日期分组的生产器、分析器、复核器；见[研究工具导航](../tools/README.md) |
| `tests/` | 单元/集成回归与研究证据复核；见[测试分层](../tests/README.md) |
| `docs/` | 规范、设计、结果与历史记录；见[文档导航](README.md) |
| `docs/snapshots/` | 保存的旧源码文本，不参与安装或执行 |

研究工具的导入、源码哈希与实验凭据可能绑定现有路径，因此保留带日期的模块名和目录。
公共能力在 `gsdeploy/` 中维护；新增课题在 `tools/<topic>_<YYYYMMDD>/` 下组织，
操作入口在 `scripts/` 中组织，并更新相应导航。当前仓库只安装 `gsdeploy*`。

## Git 跟踪边界

跟踪 Python/C++ 源码、测试 fixture、研究协议、配置、依赖锁、补丁和研究文档。
依赖锁与 `configs/environments/` 中的路径记录属于复现元数据，应随源码提交。
部分研究配置引用本机资源；实际可运行范围见[配置说明](../configs/README.md)。

以下内容留在本地，由 `.gitignore` 排除：

- `artifacts/`：原始观测、拟合产物、运行收据、图表和报告。
- `models/`、`datasets/`、`envs/`、`cache/`：权重、数据、环境和缓存。
- Python 构建缓存、Jupyter 自动保存、编辑器状态、日志、SQLite 和 profiler 输出。
- `.env*`、私钥和模型检查点。

忽略规则不会删除本地文件；实验收据中引用的源数据仍需单独保留。
文档中的 `artifacts/...` 是本地证据地址，全新 checkout 不包含这些材料。
`.gitattributes` 保留原始文件字节（`* -text`），避免换行转换破坏既有源码/输入哈希。

根目录旧 `optimizer.py` 是包内优化器的早期副本，使用相对导入且没有独立入口；
已移至[历史源码快照](snapshots/optimizer_20261023.py.txt)，保留原字节。
当前实现位于 `gsdeploy/optimizer.py`。

## 日常检查

```bash
git status --short
git diff --check
python scripts/run_cpu_tests.py
```

CI 使用同一 CPU 入口，并执行 README 中的 mock plan/run/evaluate 流程。
完整研究复核所需的来源、环境与适用范围由各协议单独声明。
