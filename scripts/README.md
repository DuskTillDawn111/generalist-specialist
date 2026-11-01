# 操作脚本导航

在仓库根目录运行；使用 `python scripts/<name>.py --help` 查看参数。
原生执行、训练和采集脚本需按对应协议准备环境与数据。

| 工作 | 入口 |
| --- | --- |
| CI / 基础 CPU 回归 | `run_cpu_tests.py`；`--list` 显示模块清单 |
| 更新完整文档索引 | `update_docs_index.py` |
| 独立模型环境检查与封存 | `audit_compact_environment.py`、`seal_compact_environments.py`、`audit_gpu_prerequisites.py` |
| 输入检查与冻结 | `check_compact_inputs.py`、`freeze_compact_inputs.py`、`bound_compact_pages.py` |
| 静态研究准备 | `prepare_static_study.py`、`prepare_static_experiments.py` |
| 校准、原生 worker 与队列 | `static_study_worker.py`、`run_static_gpu_queue.py`、`run_compact_gpu_queue.py` |
| 求解与可恢复搜索 | `solve_static_study.py`、`solve_static_session_study.py`、`run_search_session.py` |
| 冻结代理与求解器比较 | `fit_static_surrogate.py`、`fit_static_replay_v2.py`、`compare_calibrated_pool_solvers.py` |
| 独立确认 | `confirm_static_study.py`、`run_static_confirmation_queue.py` |
| 保留副本画像 | `profile_retained_replicas_20261021.py`、`report_retained_replicas_20261021.py` |
| 结果与图表 | `report_static_*.py`、`plot_static_*.py`、`plot_paper_figures_20261025.py` |
| ChatQA 校准与训练 | `prepare_chatqa_*.py`、`calibrate_trained_chatqa.py`、`train_chatqa_lora.py` |
| spillover 独立确认 | `spillover_confirmation.py` |

旧 `run_compact_cpu_tests.py` 是需要 torch/safetensors 及本地研究证据的历史全量复核入口，
不是 CI 入口。其他带日期脚本与对应研究文档配套保留，见[文档导航](../docs/README.md)。
