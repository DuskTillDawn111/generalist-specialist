# 测试入口与依赖

## 可移植 CPU 回归

安装 `python -m pip install -e .` 后，在仓库根目录执行：

```bash
python scripts/run_cpu_tests.py
python scripts/run_cpu_tests.py --list
```

入口显式选择规划、路由、评分、证据、mock 运行时、校准、资源控制、
紧凑模型适配和研究准备的回归模块，使用仓库 fixture 或临时目录，
只依赖基础 NumPy/SciPy 环境。CI 使用相同入口，并运行 CPU mock 示例。
增加可移植测试模块时更新 `scripts/run_cpu_tests.py` 的 `MODULES` 清单。

## 研究测试与独立复核

带日期/版本的研究工具有对应 `test_*` 模块；部分读取本地 `artifacts/`、
冻结源码、模型环境、CUDA 头文件或历史原始观测。应先按协议准备相应来源，
再显式选择模块，例如：

```bash
python -m unittest tests.test_twoFIFO_matching_numerics_v1 -v
```

保留全部研究测试、历史反例和 expected failures；它们不自动取得可移植 CI 资格。
`test_compact_memory`、`test_compact_native`、`test_compact_diagnostics` 需要
torch 或图像/音频依赖；`test_compact_memory_pinned` 还需要冻结的 MiniCPM 环境，
因此这些模块单独运行，不在基础 CI 清单中。
`semantic_peft_cpu_probe.py` 是依赖 torch/transformers 的单独 CPU 探针，
不属于 `test_*.py` 自动发现范围。

完整研究环境可运行 `python -m unittest discover -s tests -v`；
该命令会加载所有研究测试，在缺少本地证据的 checkout 中可能失败或跳过，
也可能运行较长时间。研究测试结果与真实 GPU 预测/质量验证结论分开记录。
