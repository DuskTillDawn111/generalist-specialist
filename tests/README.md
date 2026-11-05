# 测试入口与依赖

`tests.test_activation_bilevel` 的 22 项测试覆盖目标 BO/MADS 双层控制器、激活空间、认证 GP 观测、网格方向/未解决 poll、Joint 初值与预算保留；已登记 CPU 入口。相关回归及四项追加反例两轮覆盖 157 个不同测试，见[当前算法实现](../docs/activation_bilevel_implementation_20261104.md)。

`tests.test_c1_lite` 的 22 项反例覆盖共享池、渐进内层、全域界/UNKNOWN、预算与独立渐进 DE；`tests.test_c1_qualification` 的 5 项反例覆盖历史字段清点、准备包与成对区间的证据边界。已登记到 CPU 入口；相关回归共 135 项通过，见 [C1-Lite 实验报告](../docs/c1_lite_hybrid_experiments.md)。

`tests.test_paper_structured` 增加 22 项 CPU 反例，覆盖完整捕获转换、共享 FIFO 界、映射缓存跨质量/rho 边界、一次 Joint 后无部署 MILP、预算/UNKNOWN、资源配置身份及 busy 续推。与直接相关模块共 183 项的小范围回归及复现见[本轮交付](../docs/paper_structured_completion_20261103.md)。

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

`test_continuous_bounds` 校验逐部署安全界、枚举等价、消融、候选热启动、UNKNOWN 和预算合同。
`test_continuous_diagnostics` 校验独立计时探针不改变赢家、预测次数，并在异常时恢复原方法。
`test_continuous_validated` 校验已验证回放界、独立枚举与四项消融等价、显式多轨迹、
不可变输入/session 隔离与失效、稀疏路径 UNKNOWN、预算及坏界证书清除。
`test_joint_scalability` 校验真实网格规则、部署/任务列计数与可恢复计时包装；
`test_routing_benefit_diagnosis` 校验诊断账本与原FIFO逐项一致、多副本/零流量驻留和比例端点守恒；
`test_continuous_refinement` 校验固定驻留、未部署模型拒绝、精确缓存、warm start、
局部搜索、共享预算、初始解回退、确定性和独立部署邻域。
`test_multi_candidate` 的8项测试校验独立笛卡尔网格排名、整个部署排除、反馈隔离、
真实效用选候选与多样性、共享多任务Specialist可行域、精确缓存、共享预算、初始解保留及未认证状态。
实验数据、51次有效运行与复现命令见[多候选连续精修报告](../docs/multi_candidate_continuous_refinement.md)。
`test_adaptive_candidate` 的10项测试检查整个连续域松弛在独立有限网格、非网格比例与
robust场景中的有效性，覆盖零流量驻留、门禁/UNKNOWN、预算不续期、候选保留和重启账本。
新增方法只管理候选和预算，不更改原DE；见[自适应预算报告](../docs/adaptive_candidate_refinement.md)。
`test_optimization_difficulty` 的4项测试检查Master观察器恢复、原结果不变、下界分解、消融保留原预测及完整有限域计数；合成fixture不作为GPU证据。见[难度诊断报告](../docs/gsdeploy_optimization_difficulty_diagnosis.md)。
`test_continuous_routing` 包含闭 simplex/激活边界、独立固定路由穷举、MILP、DE、
多轨迹顺序、预算中止、UNKNOWN 和缓存隔离反例；由同一 CPU 入口执行。

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

六项固定参考 v2 回归：`python -m unittest tests.test_reference_normalization tests.test_reference_freeze -q`，覆盖六项公式、真实/缺失证据、排空能量、质量超界与 MILP/DFS 对穷举，以及参考冻结凭据校验。
