# Module 11B 简短收尾

状态：`MODULE11B_CLOSED_BY_USER`。当前 efficient Specialist 池冻结为 `EFFICIENCY_OPTIMAL_SPECIALIST_REGIME`，作为 efficiency-favorable baseline 保留。

保留 60 次 periodic、91 次 closed-loop 独立 session，共 15 个新增实测 g(x) 点。所有 session 已正常完成并释放 GPU；未重跑或覆盖已完成数据。Closed-loop 仅用于 service capacity，periodic 保留 arrival/E2E/latency/stability 职责。历史 g/M、s0、Module 9 与 worker/runtime 未修改，MPS 未开启。

| Model | 本轮 closed-loop 已测 counts | 新增 g(x) | 尚未测的更高 counts |
|---|---|---|---|
| qwen | [] | 仅保留历史 points | 3及以上 |
| phi4mm | [1, 2] | g(2)=1.636364 | 3及以上 |
| minicpm | [1, 2] | g(2)=0.961539 | 3及以上 |
| distilbert | [1, 2, 3] | g(2)=1.649029, g(3)=2.016013 | 4及以上 |
| resnet50 | [2, 3, 4] | g(3)=2.797950, g(4)=3.534635 | 5及以上 |
| convnext_tiny | [1, 2, 3] | g(2)=1.850654, g(3)=2.794931 | 4及以上 |
| paddleocr | [1, 2, 3] | g(2)=1.958262, g(3)=2.844786 | 4及以上 |
| whisper-small | [2, 3] | g(3)=1.319869 | 4及以上 |
| whisper_turbo | [1, 2] | g(2)=1.151897 | 3及以上 |
| distil_whisper | [1, 2, 3] | g(2)=1.123712, g(3)=1.159632 | 4及以上 |
| roberta_sst2 | [1, 2] | g(2)=1.901119 | 3及以上 |

全部未测 higher counts 标记为 `NOT_REQUIRED_FOR_CURRENT_NEXT_STAGE`，不是“不可行”或“已饱和”。完整物理 replica frontier 与 breakpoint evidence 均未假装完成，也不宣称 `BREAKPOINT_EVIDENCE_SUFFICIENT`。Qwen 的既有 count 1/2 points 保留，本轮未新增其 closed-loop 测量。

RoBERTa 3-replica frontier 仅完成了对应 2-replica 基线的一次 session；3 replicas 尚未测，未加入 g(3)。这份已完成基线 raw 保留；未执行的 repetitions 同样标为 `NOT_REQUIRED_FOR_CURRENT_NEXT_STAGE`。详见 `replica_evidence_closeout.csv/json`。

本次未获得更新 W1G/W2G 的完整 CPU 认证：增量 replay 的 replica-count tie-break 出现数值 infeasible，原 formulation 与更严格数值配置均未完成。日志已归档，未将该问题解释为物理 capacity infeasible；原 Module 11A conditional 结果保持不变。replica evidence gate 保留未定状态，但不阻塞当前下一阶段。

完整旧 Specialist frontier 不再继续追踪，因为下一阶段将先用 CPU-only Specialist Efficiency Sensitivity Audit 定位 transition region，再选择真实、更大/更慢且 quality-admitted 的同任务 Specialist，对新模型重新实测 Q、s0、M、g(x)。Counterfactual sensitivity 只定位搜索区间，最终性能结论仍以真实模型为依据。该下一阶段本轮尚未启动。

fixed-cost / low-QPS campaign 继续暂停，GPU runs=0；14 resident-set manifest、42 次计划测量的协议、脚本、CPU checks 与准备输出全部归档保留，不作为最终低-QPS deployment 证据。正式 multi-hotspot GPU、Task-Rich、low-QPS campaign 均未启动。

现有数据 SHA manifest：`frozen_existing_evidence_manifest.json`；baseline 快照：`efficiency_optimal_baseline/`。
