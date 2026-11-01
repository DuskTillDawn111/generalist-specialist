# 显式 IN_TRANSITION 初始联合 prior 与接线

原 28 条 initial observations 中，27 条有已配对 holder，1 条位于记录的 scheduler handoff gap。新 `context_initial_prior_v15.py` 对完整原 supervision 全源重推（25.64s），保留全部原 row、scenario 顺序和 capture/job 权重，生成 **27 HELD_CONTEXT + 1 IN_TRANSITION**。若原事件 unpaired、映射不全或缺双侧证据，则仍保留 UNKNOWN；不删除或重新归一化样本。

该条真实 gap 样本是 SG-text#0-short，query clock=80213333678ns。前 text#0 SAVE=80213321391ns，已过12287ns；下一 G2 RESTORE=80213365551ns，尚余31873ns；总gap44160ns。附近没有 unpaired，四个已记录 context 都无覆盖 query 的 paired envelope。下一 G2 envelope 内没有 kernel，故 service-ready upper witness 保持 None，不能填0或用后续另一次 restore 的 kernel 作为本次 upper。该标签是记录的 scheduler 状态，不能解释为物理 GPU 不执行。

接口为 `JointInitialContextPrior(model).predict(...)`，参数与 v14 相同，仅使用第一 pending frontier 和 public setup。返回 `state_probability_masses`、`in_transition_mass`、`unknown_state_mass`；transition 的当前 holder/age 不适用，独立列为 `holder_not_applicable_mass`。原 short text#0 的 .25 unknown-holder mass 全部保留为 .25 transition mass；双 frontier pooling 的 .0357143 同样保留。9 个 exact/coarse/setup 条件域逐项 scenario ID、顺序和概率与 v14 完全一致。

`realize_joint_initial_scenario(..., query_clock_ns=now, readiness_rule='bounds_only')` 默认只给界限，不产生 service point。transition 返回 previous holder、elapsed gap、相对 now 的 next RESTORE 及 next owner，来源于冻结联合 prior 的监督标签；prospective predict/realize 没有 SQL 或文件 I/O，不读取 now 之后的 source 事件。缺 upper 时 upper/witnessed point 模式拒绝；显式 `lower_bound_hypothesis` 才数值化 reference readiness 并保留假设。

新 `held_context_scheduler_v15.py` 与 `learned_context_oracle_v15.py` 接入该联合 prior，未修改冻结 v14。初始 gap 期间 held_owner、restore clock、age 均为 None；只排未来残余 RESTORE 和独立 SERVICE_READY 事件。目标 owner 可以不在 pending frontier。之前的 owner 只在以后作为 mark 历史；普通 source replay 始终禁止。主 adapter 可替换为这两个新模块的同名 oracle/scheduler，构造参数保持 v14 接口。

13 项 CPU 测试通过。独立 verifier 直接扫描原 GPU_CONTEXT_SWITCH_EVENTS 验证 gap，并在禁止 prospective SQL/文件读取下比较9域概率。真实组件+synthetic tiny DAG 重验中，原三个 UNKNOWN case 均完成；额外 quantile=.375 显式选中 gap 联合场景，31.873µs 后恢复不 pending 的 G2，再经一次 learned mark 切回 text#0 完成请求。bounds_only 对照按预期拒绝点预测。7个成功 case 的15个 GPU reference-work node 逐个积分守恒；总流程3.064s、组件初始化2.959s。它们只证明 CPU 接线机械行为，没有独立 GPU 干扰或收益结论。

冻结产物：

- model：`artifacts/predictor-context-initial-prior-v15-20261028-training01/model.json`，SHA `e72704347055fbe4d662ef3a940d063245041b1afbd6fbcc6cb05062b5d624c0`。
- 原始 source 独立审查：`artifacts/predictor-context-initial-prior-v15-20261028-independent01/report.json`，SHA `8fdfe9c3eb48978d086b97d3b6ce0f5196e01c55a32f113c215e1af0c625a0b4`。
- 真实组件 smoke：`artifacts/predictor-real-learned-small-dag-v15-20261028-cpu01/report.json`，SHA `39e43bc529a64a94fb9109d6fff19d9192fc0884afc431953f0c644722bb2e2c`。
- 守恒摘要：同目录 `mechanical-verification.json`，SHA `1365471eb16722209c2bba8f32d351da5d7d4a9959ae790af85f787c0e4f6cb6`。
- 汇总验证：training01 的 `verification.json`，SHA `a1a63599ec0b2e30520dc7415c25c191e43a41437c65836599326490d35d9482`。

未启动 GPU，未读取 validation labels，未改主 v14 adapter/oracle/scheduler 或正在运行的 preflight 源码。
