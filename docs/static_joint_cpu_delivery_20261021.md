# 静态联合部署：CPU 阶段交付（2026-10-21）

按用户要求，使用三个子对话推进输入冻结、质量评价和 DFS/效率三个阶段；主对话负责 MILP 接入、集成审查与运行准备。
本轮没有运行任何真实 GPU 作业。真实模型池的六项最优部署、混合实测排名与正式效率结论待校准后产生。

## 当前完成内容

1. 冻结七模型、四任务、16 条可执行路径、61 档完成画像及失败/未知档位。
   输入池为 160 个唯一输入，逐输入质量可追溯；完整范围见
   `artifacts/static-study-v6-20261021/stage1-report.zh.md`。
2. 接入 `six_term_quality_v1`，T/L/P/M/E/Q 默认各 1/6，移除新契约的绝对精度准入。
   QA 用 1-EM、图像用 1-accuracy、OCR 用 CER、ASR 用 WER。
   原生实测评分使用当次输出；质量和能耗未知时不生成完整 U。
3. MILP 与独立 DFS 接入相同声明副本/分流网格和完整评分，支持未知证据、有效界、预算与证书。
   强竞争方案通过排除胜出部署、排除完整方案、固定部署或禁止分流重新优化。
   固定部署和其他受限域的证书保留限制条件，不转写成自由全域或物理最优证明。
   本轮优化固定负载下的静态分流比例，能够比较分流带来的排队收益与质量代价；
   按实时队长逐请求动态调整的策略不属于本轮静态最优证书。
4. 增加主动分流与质量权衡案例、CPU 冷进程对照、真实冻结池的代理诊断。
5. 准备 8 条去重校准条件、独立确认选择逻辑、源文件/输入/worker/检查点绑定以及显式执行入口。
   预热按各部署 worker 的正负载可执行任务选最大三个输入；大小按媒体文件 bytes 或文本 UTF-8 bytes 排序，
   该定义不构成最大激活显存证明。正式窗口有独立输入顺序。

## 冻结条件和证据边界

主池为 `full_retained`，`generalist_only` / `specialist_only` 只作结构控制。
旧 A/B/C 池使用旧模型身份，未擅自映射为新的嵌套池。
四任务各为约 0.074836 req/s，正式 H=1800s，校准 H=600s。
开发种子 42，确认 43/44/45，路由先 K=4 后检查 K=8，首轮关闭额外容量见证。
所有准确参数、输入/参考哈希与资源域以 `inputs/study-contract.json` 和 `system-config.json` 为准。

逐任务闭环均值转入 FIFO 仍是代理：混合干扰、输入服务尾部、开环稳定性、混合预热峰值未实测。
没有整方案能耗，因此真实池诊断的 E/U 均为 null，不从部分效用中选“最优”。
当前运行 checkpoint 全文件哈希与历史输出的配置身份分别记录；不存在历史全权重哈希时，
不宣称历史离线输出与当前权重已完成全文件等价证明。
本轮已在 CPU 逐一读取并核验七模型当前文件；PP 仅覆盖保留的 det/rec 目录，
QA 的 LoRA 配置和权重另按固定 manifest 逐文件核验。
结果见 `artifacts/static-study-v6-20261021/cpu-checkpoint-bindings.json`，没有重新进行质量推理。

## 已观察的 CPU 结果

合成可证小域中，MILP、DFS 与独立 Cartesian oracle 的最优效用一致。
分流案例的 QA 通用精度为 0.80、专用为 0.95；等权时最优将 25% QA 分给通用，
U 约 -0.223655，最佳不分流约 -0.236821。
质量权重提高到 0.6/0.9 后，最优恢复专用承担全部 QA。这些数值只验证算法机制。

五次冷进程中位结果：small 的 MILP/DFS 完整评分次数为 44/241，medium 为 35/1174。
评分调用减少尚未带来总搜索时间优势；small DFS 约 1.47s 完成证明，而 MILP 用满 10s 未完成证明。
medium 多数运行仍限时，MILP 另有一次数值未解决，不能把相同 incumbent 当作完成证明。
性能分析表明重复 HiGHS 求解占约 94%，完整预测只占约 2%。
原始数据和全部限制见[DFS 与效率报告](static_joint_dfs_benchmark_report_20261021.md)。

## CPU 复现和 GPU 接续

准备和验证只读取 CPU 文件，不加载模型、不查询或执行 GPU：

```bash
python scripts/prepare_static_experiments.py prepare \
  --inputs artifacts/static-study-v6-20261021/inputs \
  --output artifacts/static-study-v6-20261021/experiments-NEW
python scripts/prepare_static_experiments.py validate \
  --packet artifacts/static-study-v6-20261021/experiments-NEW/experiment-packet.json
```

输出目录必须是新的不可变版本。源码或输入改变后重新准备，旧作业与旧证书不能继承。
正式 GPU 运行入口为同一脚本的 `run --packet ... --job ... --output NEW --execute`；
本轮没有调用该入口。GPU 校准后冻结新评价证据，再重新求解各池和强竞争方案；
确认数据不同时用于拟合。

精确证据接续入口：

```bash
python scripts/prepare_static_experiments.py freeze-evaluator \
  --packet EXPERIMENT_PACKET.json --runs CALIBRATION_RUN_1.json CALIBRATION_RUN_2.json \
  --output artifacts/static-study-v6-20261021/evaluator-NEW
python scripts/prepare_static_experiments.py prepare \
  --inputs artifacts/static-study-v6-20261021/inputs \
  --config artifacts/static-study-v6-20261021/evaluator-NEW/system-config.json \
  --output artifacts/static-study-v6-20261021/experiments-NEXT
python scripts/prepare_static_experiments.py queue \
  --packet EXPERIMENT_PACKET.json \
  --config artifacts/static-study-v6-20261021/evaluator-NEW/system-config.json \
  --plans UNRESOLVED_PLANS.json --output ADAPTIVE_PACKET.json
```

`freeze-evaluator` 只导入实际有效完成的校准条件，保留服务均值 FIFO 预测的范围，
不构建未经留出验证的干扰外推模型，不把 H600 的功率/延迟扩散到 H1800。
未测候选继续未知并进入去重补点队列。因此仅有 8 条精确校准记录时，
通常仍无法认证整个正式域的最优；要减少 GPU 次数并获得大域证书，
仍需后续校准数据支持的可验证泛化模型或覆盖未测候选的有效界。

每池至多选择自由胜出、最强替代部署、同部署最强替代路由、最佳不分流四个初始方案，
完全相同的运行条件去重，预留一个自适应名额，各用 43/44/45 三个配对种子。
少量实测能支持这些方案的比较；若未测候选没有有效界，不能升级为物理全域最优。

本轮验证清单与最终作业包状态见同目录的 CPU 验证记录及
`artifacts/static-study-v6-20261021/experiments-final-v2/`。

全套 CPU 回归 220 项，通过 218 项，2 项要求 MiniCPM 专用解释器，普通解释器下跳过。
随后使用现有 `compact-minicpm/bin/python` 单独运行这 2 项 CPU 用例，均通过，220 项均已有通过记录。
最后一次运行源码身份修复后另跑 20 项相关回归，全部通过；源码编译与 `git diff --check` 通过。
完整测试版本、跳过原因和源码哈希见 `artifacts/static-study-v6-20261021/final-cpu-validation.json`。
最终 CPU 诊断见 `artifacts/static_joint_cpu_v6_20261021/diagnostics-final/`，阶段状态见
`artifacts/static-study-v6-20261021/stage-progress.json`。
