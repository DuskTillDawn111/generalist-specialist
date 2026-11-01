# 实际 learned components 与小 DAG 集成检查

本次使用冻结的 initial-prior v14 training02、SAVE hazard v14 controller、next-owner v14 training02、handoff v14 training02。未初始化代价较高的原 work/host providers；DAG 工作量和 host delay 显式为 synthetic，四 resident 身份保留，最多两个 text owner 有请求。它核验真实 solver 调用和状态推进，不是 GPU 干扰准确率或最终收益实验。没有 validation labels、GPU、协议 seal、P/M/E 补零或 full_U。

实现为 `tools/predictor_validation_20261027/real_learned_small_dag_smoke_v14.py`；主 `learned_context_oracle_v14.py` 和 `learned_availability_forecast_v14.py` 只读。完整组件初始化 1.423s，总耗时 1.477s。

| synthetic case | 结果 | scheduler calls | model switches | 完成请求 |
|---|---|---:|---:|---:|
| text#0 短输入单请求 | initial unknown holder mass=.25，明确 UNKNOWN | 1 | 0 | — |
| 双 text 同时到达 | pooled initial unknown mass=.0357143，明确 UNKNOWN | 1 | 0 | — |
| text#0 首到达、带 host gap | initial unknown mass=.25，明确 UNKNOWN | 1 | 0 | — |
| 已有完整支持的 text#1 首 frontier，单请求 | 完成 | 3 | 0 | 1 |
| text#1 首到达、text#0 晚 50µs | 完成 | 27 | 7 | 2 |
| text#1 首到达、text#0 晚 50µs，100µs host gap | 完成 | 27 | 5 | 2 |

首三个 case 的第一不支持原因均为 `UNRESOLVED_INITIAL_HOLDER_MASS_PROHIBITS_POINT_FORECAST`，没有删除 unknown 样本或重新归一化 known holder。额外三个支持 case 用训练 source 中已有完整支持的 text#1 首 frontier，目的是进入实际 hazard/mark/handoff solver 验证机械接线，不能当成未选择的总体准确率。原 cpu01 报告保留了早期 host fixture 格式错误；cpu02 修正为原引擎支持的 `after_predecessors_ns` host release 表达。

成功 case 全部 absolute restore age、left-truncated observation、不积分 preentry hazard、service-ready/restore 顺序以及 rate>0 对应 held owner 的不变量通过。由保存的 scheduler trace 独立积分 `rate × Δt`，1+2+4 个 GPU reference-work node 的工作量全部精确守恒。双请求 case 首次 SAVE 在 1000215112ns，context age=1197510ns，text#1→text#0，预测 SAVE→RESTORE gap=44129ns；lower-bound readiness 假设给 service delay=0。host-gap case 首次切换 gap=44164ns。所有这些均是模型输出和 synthetic work 机械行为，不是观察到的真实 GPU 延迟。

报告 `artifacts/predictor-real-learned-small-dag-v14-20261028-cpu02/report.json`，SHA256 `67ae51eea021d1e2992c66673b37d9658fb37b2fc28a24f26320981ec648da56`；守恒摘要 `mechanical-verification.json`，SHA256 `92795b27a0777ead2b46d14c3662f6fc2e9a39a839ee9b7602aa36c9d79b4e24`。报告结束时所有模型和主接线源码收据保持不变。

训练 prior 的 text#0 长输入首 frontier 没有 unknown mass，包含 88.55ms attained-age scenario；因此上面的短输入 first-unsupported 不能外推为主线 long-input 全 cohort 必然失败。主线 4×57 原 cohort 真实预检必须自行保留首次不支持位置和完整结果。
