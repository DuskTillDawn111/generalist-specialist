# 校准通过后的五模型同域效率比较

此计划和新脚本不修改冻结 execution-source-v4，不启动 GPU，也不对未验证预测器求解。已有两模型及合成实验结果均保留。

## 实际有限域规模

选择 `generalist_2 + text + image + ocr + asr` 五个已纳入校准的模型，各只允许缺席或一个副本。保留每个模型完整的原始副本、显存和性能画像行，以 `replica_options={model:[1]}` 限制搜索副本域；不截断表以改变冻结身份。

路由合同是 **分母 1 到 K 的所有网格的并集**，不是只使用分母 K。因此：

| K | 每任务两个可用路径的不同分流比例数 | 五者全常驻的路由身份 | 所有覆盖部署的完整身份 | 任务列 |
| --- | ---: | ---: | ---: | ---: |
| 4 | 7 | 2401 | 4097 | 36 |
| 8 | 23 | 279841 | 331777 | 100 |

K=4 的比例为 `0,1/4,1/3,1/2,2/3,3/4,1`。若仅数分母 K，会错误得到 625/6561；这会漏掉现有算法实际声明的路由域。

共有 17 个覆盖全部任务的部署：G2 存在时其他四个专用任取的 16 个，加上 G2 缺席而四专用全存在的一个。设每任务不同分流比例数为 F，完整身份总数为 `(F+1)^4+1`；含闲置常驻模型的不同部署身份仍独立计数。任务列是 `4*(F+2)`，与完整部署/路由身份数不同。

原画像中的五模型安全显存合计为 41403.750288 MiB，预算为 49140 MiB；假设冻结资源代理的全局显存倍率不超过约 1.186849，五者同时常驻的 additive 下界才不超预算。真实拟合倍率或已见部署峰值更大时，由三种算法原有合法约束筛选，不修改预算或画像来保留候选。脚本会重新报告通过有效显存下界的部署数和身份上界，不能把该上界解释为容量/排队可行数。

## 公平预算与观测

先运行 K=4 的一轮 pilot，三种算法均使用同一个 120 秒求解预算、同一冻结 trace、同一质量/服务/能耗/显存评分器与约束。方法为 proposed MILP、带结构界 B&B DFS、无效用剪枝穷举 DFS。Plain DFS 保留合法显存/容量筛选，效用界计算和效用剪枝为零。

每次使用独立 CPU 子进程，串行交错；第二轮倒序，然后交替。正式批次预先固定五轮，不把不同预算的最快结果拼成同一对照。计时分两种：child wall 包含搜索器构造和求解；parent wall 另外包含 Python 导入、配置加载/门禁、结果序列化与传输。记录每次 peak RSS、完整求解结果、评价/HiGHS/剪枝次数、目标上下界、gap、预算状态、证书和部署/路由身份。最优身份不一致但效用一致可能是并列最优，应分别报告。

原始 H1800/seed42/Poisson trace 有 567 个请求，完整评分含三种服务场景。忽略合法约束拒绝时，穷举 K=4 约需 6968997 次 FIFO 请求处理，K=8 约需 564352677 次；还要加质量回放和身份验证成本。因此不能仅凭列数推断耗时，K=8 也不能假定一定能在 120 秒内取得穷举证书。

K=4 三算法五轮最多约 30 分钟求解预算，另有构造和进程开销。若 pilot 中 Plain DFS 未完成，保留其预算状态/gap；如果另做 300 秒批次，必须独立输出并对三算法统一使用 300 秒，五轮最多约 75 分钟。不能因某个方法耗时结果不理想而挑选预算。K=8 仅作为可选扩展，证书不足时报告预算内结果。

结构规模本身很小：32 个二进制部署组合、36/100 个任务列、最多 567 请求的单次三场景评分。主要内存风险来自完整加载的冻结输入/质量/校准证据、搜索历史和未知候选记录；脚本串行运行，每次记录真实 `peak_rss_kib`，不预报未经测量的峰值。若 pilot 内存超出宿主可用量，应报告该限制，不缩小某一方法的域来制造公平比较。框架额外 state 上限为 1e8、MILP iteration 上限为 1e6，均显式记录；若这些或列数上限先触发，也保留状态，不能作为时间证书。

## 独立脚本与执行参数

新增 `scripts/compare_calibrated_pool_solvers.py` 独立于 v4。`--source-root` 明确选择用于预测和求解的冻结实现；子进程把该根加入导入路径，不默认使用当前工作区的数学源码。父进程对该根的全部 gsdeploy Python 文件以及 benchmark 脚本自身计算前后哈希，报告是否全程不变。v1/v2 必须通过冻结源码的正式 validated gate；未经验证时退出，尚未启动任何求解。

只审计已验证域：

```bash
python scripts/compare_calibrated_pool_solvers.py --source-root artifacts/static-study-gpu-20261022/execution-source-v4 --config /absolute/path/to/validated-system-config.json --route-grid 4 --audit-only
```

先一轮 pilot，独立保存：

```bash
taskset -c 200-203 python scripts/compare_calibrated_pool_solvers.py --source-root artifacts/static-study-gpu-20261022/execution-source-v4 --config /absolute/path/to/validated-system-config.json --route-grid 4 --time-limit 120 --repeats 1 --output /absolute/path/to/pool-k4-pilot
```

预定五轮正式批次，使用相同机器 CPU affinity：

```bash
taskset -c 200-203 python scripts/compare_calibrated_pool_solvers.py --source-root artifacts/static-study-gpu-20261022/execution-source-v4 --config /absolute/path/to/validated-system-config.json --route-grid 4 --time-limit 120 --repeats 5 --output /absolute/path/to/pool-k4-five-repeats
```

输出目录必须尚不存在。每个已完成进程立刻写完整 JSON，最后汇总目标、证书和身份一致性。只有所有轮次都取得相同效用证书、合同一致且源码/配置未变，`benchmark_complete` 才为 true；预算不足的完整原始结果仍保留。

当前只运行了合成 CLI 门禁/计数自检：显式加载 v4 的已验证合成 v1 配置通过审计（两模型双任务域 51 个身份）；相同配置状态改为 `FITTED_UNVALIDATED` 后拒绝；没有启动求解。实际五模型尚未进行效率测量，不能预先宣称任一方法更快。
