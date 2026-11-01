# 用户四阶段请求：实际证据盘点

快照时间：`2026-10-27T05:11:03Z`。本盘点直接读取现有原生 plan/raw、硬件 probe、kernel 导出、独立报告/门槛、DFS 结果与执行源码；对应 30 项来源 SHA 及逐项计数见 [机器可读盘点](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/request-evidence-inventory-v1.json)。未启动额外 GPU、拟合、调参或重试。正在等待的 T8/V4/F8 保持原有限方案。

目前第一阶段微观画像与 shared 独立误差报告已有真实产物；shared 的完整预测验收仍为 **FAIL**。互补资源分类、共同 batch、公共权重子矩阵及缓存复用尚无物理验证，不能将它们写成已经完成的执行优化。当前 T8/V4/F8 接续将补整请求准入及有限对照，不能补齐这些尚不存在的后端。

## 1. 固定四驻留画像：实际采到什么

以下三套 raw 的 READY 均包含 `text#0,text#1,generalist_2#0,ocr#0`，所有请求完成、采集错误和 telemetry error 为空。这里的单路是“只有一路活动、四路仍驻留”。计数由 raw.batches 的 job_id 回查 plan.jobs 得到，未将计划中未执行的 validation job 算作实际采集。

| 实际数据 | 批次 / 请求 | 已采活动组合 | 板级/owned 显存采样 | 原始 owned 采样峰 |
|---|---:|---|---:|---:|
| native-training-v3 | 144 / 354 | S0、S1、G、O 各6；SS、SG、SSG、SSGO 各30 | 8329；最大间隔60.756ms | 31666 MiB |
| semantic-training-v5 | 144 / 354 | 同上，新增真实语言 forward 语义 | 8337；最大间隔57.472ms | 31666 MiB |
| background-training-v7 | 264 / 690 | 上述基础组合；G_image/G_asr 各6；SG、SSG、SSGO 各增加 image/asr 各18 | 15412；最大间隔70.696ms | 31634 MiB |

S0/S1 为单专用，SS 为双专用，SG 为专用＋G2，SSG 为双专用＋G2，SSGO 再加 OCR。v7 的 G2 分别实际执行 text/image/asr，OCR 由独立 OCR worker 执行。所有 32078 个采样都列出全部四 alias，foreign_pids 为空；这些是拥有进程显存的**离散采样峰**，不是连续真实峰值或安全上界。功率为整块 GPU 板级功率，不是单模型/单 kernel 归因；三套功率范围分别65.595–383.600、66.180–381.109、65.437–377.214 W。

来源：[v3 raw](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/native-training-v3/raw.json)、[v5 raw](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/semantic-training-v5/raw.json)、[v7 raw](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/background-training-v7/raw.json) 及各同目录 plan。

输入和错位启动已有覆盖，但范围须按实测描述：训练实际用了每任务3个已知输入，不能把计划每任务第4个保留输入算入训练覆盖。text 公共 input_size 为163/3899/9797；真实专用 prefill 序列长414/1247/2296，G2 text 为389/1253/2291。v7 图像 input_size 为12520/95726/132813，ASR 为83900/129327/203859，OCR 为90343/359264/749407；非文本 input_size 是公开排序特征，不冒充 token 数。原始 offset 含0/50/100/150/200/400/600ms，对应同时开始及50/200ms 步长的正反顺序；请求实际 START 另有记录，不能把计划 offset 当成真实 GPU 发射间隔。

阶段记录覆盖 v3/v5 全部354请求和 v7 全部690请求。v5 实际有318个 prefill、3668个 decode forward start；v7 为618/7348。Python forward 的墙钟阶段可能含主机发射与异步尾部，不是纯 GPU 忙碌时间。独立 [v5 kernel 预检](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/semantic-trace-v5/exports/summary.json) 仅4请求、锚全部匹配，观测到 G2 prefill kernel 越过 Python 返回约51.594ms；部分跨 worker kernel 确有重叠。这支持阶段对齐审计，不能外推所有画像都具有完整 kernel 序列真值。

**准入等待尚未由这些 shared 微画像证明。** 三套原生请求均没有 admission 字段；worker queue wait、submit→START、真实 permit wait 是不同量。整请求准入的实际 acquire/grant/release 与 permit-wait 需要 pending 三模式 pilot/T8/V4 的独立原生日志证明，不能将 shared 排队字段重命名为准入等待。

## 2. 独立预测验证：已有报告且保留失败

[H600 selected-v16 独立报告](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/flow-validation-h600-selected-v16/report/report.json) 为 fixed seed627701、四 shared route，各1200请求，四组实际六项 Q/T/L/P/M/E 都完整。预测先于实际封存，完整未来到达参与模型。其 [原 gate](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/flow-validation-h600-selected-v16/gate.json) 为 FAIL：6项失败、0项 UNKNOWN。ΔU 符号3/3及 pair 排序6/6正确，不能据此忽略排队/P95失败。

| 路由 | 服务 MARE | 完整 E2E MARE | queue MAE / 请求 | 原稳定性 |
|---|---:|---:|---:|---|
| balanced | 6.001% | 11.192% | 0.475 | 通过；recoverable_congested |
| skewed | 7.889% | 10.600% | **9.218** | **不通过；unstable** |
| spill_low | 8.999% | 10.035% | 0.356 | 通过；recoverable_congested |
| spill_high | 17.084% | 13.953% | 0.535 | 通过；recoverable_congested |

失败项为 skewed queue、low image P95、high 的 text/image/asr/ocr P95。skewed 原稳定性因“持续正尾增长且窗口末有非平凡积压”判 unstable，即使之后 drain 完成也不改判。能力附录还保留 balanced 原 rho 超限、skewed 统计服务适用性失败；旧 measured-profile resolver 仍 UNKNOWN。服务均值误差较小与长期队列/P95准确不是同一结论。

当前可支持的判断是：新预测器已能在固定已知输入域产生完整预测并解释部分经验干扰、正确预测本块效用排序；仍未通过原完整预测门槛，也未建立新输入总体泛化或置信界。独立结果不能回流到同一目标继续拟合。admission2 的固定 V4 还须单独检验，不能直接继承 shared 结论。

## 3. USHER / Roomie 机制：完成与未完成

| 用户要求 / 可借鉴机制 | 已有真实或代码证据 | 当前可声明范围 |
|---|---|---|
| 计算型/显存型互补放置 | 实际共驻、错位及阶段/kernel 预检；硬件 probe 的 SM/DRAM/L2 counter 失败 | **物理分类 UNKNOWN**。无互补放置干预/对照；显存容量占用不是 DRAM 带宽瓶颈 |
| Roomie 的序列/阶段干扰建模 | 真实 prefill/decode、完整任务组合与经验条件服务模型；全过程事件模拟 | 已实现受其启发的经验阶段/上下文预测；不是已验证的 Roomie occupancy/SM 资源公式，也不是完整 kernel scheduler |
| 默认共享 vs 整请求准入 | host permit、每 alias FIFO、global max1/max2 controller 与严格 native audit 已实现 | 三模式 GPU pilot 与 T8/V4 尚待完成；不能提前宣称降低干扰或净效用提高 |
| MPS 计算上限 | 配置、整数百分比、session 环境准备 | native launch 明确阻止未经验证的平台 adapter。daemon、真实客户端加入及 cap 生效无收据；硬件支持 UNKNOWN |
| 互斥 SM / Green Contexts | 声明、粒度/重叠等契约校验 | native context adapter 未实现，硬件支持/实际分区未知；百分比不证明 SM 互斥 |
| 分 batch / 同模型共同 batch | 当前执行 context 为 batch_size1；内部模型 microbatch 有自身用途 | 跨请求共同 batch 执行器未实现。模拟器对 batch>1 或 weight_reuse 明确 UNKNOWN；配批等待/额外KV/质量还无实测 |
| 双专用共享权重/缓存 | 两专用 alias 对应同一 text worker/spec，但各为独立 PID/模型 owner | 最有希望的精确共享候选；无单 owner 共享 tensor、联合 GEMM、跨请求 KV/prefix cache 执行证据 |
| 异构最大公共权重子矩阵/cache复用 | 论文机制复核与适配设计 | 没有 tensor 精确等值/量化元数据/公共块提取/算子合并与 cache-hit 验证。相同形状或近似权重不证明可共享；预测器未授权填折扣 |

[硬件 probe](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/hardware-probe/receipt.json) 中 nsys CUDA/NVTX returncode0，而 [counter 输出](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/hardware-probe/tiny-counters.csv) 明确 `ERR_NVGPUCTRPERM`、ncu returncode1。**没有 SM/DRAM/L2 counter，不能宣称计算与访存互补已物理验证、cache 命中率已提升。** 普通 KV cache、内部 microbatch 或 `empty_cache()` 也不能替代所要求的共享权重复用。

可执行边界直接依据 [resources.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/resources.py) 的 worker_environment blocker、[simulator.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/simulator.py) 的 `JOINT_BATCH_OR_MERGED_WEIGHT_EXECUTOR_REQUIRED` 和 [v7 context](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/background-training-v7/context.json) 的 `batch_size=1,weight_reuse=false,counters=UNKNOWN`，不只引用进展摘要。

## 4. 最终有限轮、强基线和 DFS 还缺什么

快照时 pilot/T8/V4/F8 completed 均未出现。原 T8 PID85446/start602865199、V4条件等待 PID87518/start603003048、F8条件等待 PID87937/start603025364 仍在原任务链；盘点不改变它们。固定剩余角色是：T seeds628101/628102 ×4准入路由＝8次/9600请求；V seed628201 ×4＝4800；F seed628301 ×shared/admission2×4＝9600。失败停止、不重试、不替换 seed，不新增无限开发轮。

这条有限链尚待产出的证据为：三模式真实准入激活与完整等待；T8 mode/route 专属 Q、能量、owned peak/启动开销、原容量估计与整seed CV；V4 全过程服务/排队/P95、ΔU符号/排序及原稳定性；F8 八格实际原六项、每模式内部禁/允许分流及强基线对照。任一缺失不从 shared、另一route或旧窗口补值。

F8 已冻结的主对比是 **admission2 的预锁定 spill_high，相对本块 balanced/skewed 两条声明禁分流路线中实测较优者的差值**，并保留两模式完整八格。这是比只用 balanced 更强的有限域基线；它不是全部部署/副本/资源配置内经过优化的最佳禁分流方案。锁定策略是预定诊断策略，不主张 admission2 优化器已经选出最优。一个独立 block 只给描述差值，不能把9600请求当9600个独立重复给显著性或正收益资格。

[shared H600 同域 DFS](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/cpu-search/h600-selected-v16/summary.json) 已实际完成：plain/bounded 各对4候选×1200公共请求重新 rollout，模型最优及全部 ties 一致（仅 spill_high），0剪枝、0调用节省。它验证同一模型有限域内的搜索算术，未认证原 rho 门槛约束的最优、物理最优或搜索提速；未搜索 admission、其他部署/副本/资源份额/batch/cache/自适应未来策略。admission 对应的新同域 fresh DFS 尚未完成；若最终主张“算法在该扩展域找到最优”，必须补该域核验，不能借 shared 证书。

现有有限 T/V/F 设计可以诚实交付准入机制与有限负结果/描述对照，不能据它宣布全部 USHER/Roomie 执行创新已完成或最终分流已获益。互补分类、batch、公共子矩阵/cache 和 MPS 缺口如未在本轮有实际后端与证据，最终交付应继续标未实现/未知，而非再扩展一轮 GPU 研究。

复现盘点的 CPU 入口为 [request_evidence_inventory_v1.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/request_evidence_inventory_v1.py)。本 JSON 是上述时间的证据快照，不会因后续 T/V/F 完成自动更新；最终结果须另以完成产物补充，保留本快照及旧失败。
