# 空闲物理 SAVE 计数模型候选 v53

此候选只补全空闲状态的物理 SAVE_END rate，不提供 next-owner mark、GPU SM 服务、active target、
队列精度或六项效用收益结论。旧 v42/v44/v50 和其封存来源没有修改；全部工作在 CPU 上进行。

## 数据与预先固定的估计

只读原 v41 四份 training-only risk chronology，复用严格原 raw/review/ledger/gap/exposure 核验。
选择完整四个 projected handler_active 字段均为 false 且 exposure_qualified 的风险段。
每段曝光为精确 `end_ns-start_ns`，事件仅为其右边界 `physical_SAVE_END_observed`，
按 held_owner 分层。未资格曝光单独保留，不进入 fit；不按其他 owner 或活跃状态平滑。

| held_owner | 段数 | 曝光秒 T | SAVE 数 N | posterior mean /秒 | Gamma 95% credible /秒 |
|---|---:|---:|---:|---:|---|
| generalist_2#0 | 18 | 1.807605297 | 0 | 0.276609059 | [0.000271649, 1.389652430] |
| text#0 | 29 | 4.329668994 | 0 | 0.115482269 | [0.000113412, 0.580169777] |
| text#1 | 44 | 8.560044474 | 0 | 0.058410911 | [0.000057364, 0.293449771] |
| ocr#0 | 0 | 0 | 0 | UNKNOWN | UNKNOWN |

总计91段、14.697318765秒、0 SAVE；原 mark stream 中对应 all-handler-idle mark 也为0。
这些计数与独立 `idle_physical_transition_review_cpu.py` 的报告一致。

似然为 accumulated idle exposure 上按持有者分层的 homogeneous Poisson，固定 Jeffreys improper prior
`p(lambda) ∝ lambda^(-1/2)`。有正曝光时 posterior 为 Gamma(shape=N+0.5, rate=T_seconds)。
运行点取 posterior mean `(N+0.5)/T_seconds`，转成原接口 hazard/ms 再除1000。
无曝光不能产生 proper posterior，必须 UNKNOWN。这里未选零率、未根据哪条候选能跑通来选 prior 或分位点。

报告同时保留 Gamma equal-tailed 95% credible interval，及零事件下频率学单侧95%上界
`-log(0.05)/T_seconds`。Gamma credible interval 的正下界依赖先验，不是观测到正率的证据，
更不是频率学置信下界。零事件对 rate 提供上约束，不能确认恒定率、独立增量或 attained-age 规律。

## 四份完整采集留出

每次完整留出一个 route capture，其他三份独立重算各 holder posterior，再预测留出曝光中的计数。
Gamma-Poisson posterior predictive 为 Negative Binomial，shape=N_train+0.5、
`p=T_train/(T_train+T_holdout)`；保存均值、95%计数区间、实际计数概率与P(N=0)。
四fold共有10个有训练与留出曝光的 owner cell，实际0均在其95% predictive 区间内，
P(N=0)约0.521–0.908。另2个G2 cell无留出曝光，4个OCR cell无训练曝光，均不记成成功预测。
这是重叠训练fold上的计数诊断，不能据此宣称 independent accuracy、常率已证实或 idle mark 已识别。

## 严格运行域与未验证假设

`IdlePhysicalSaveHazard(model_dict, fallback)` 只在 runtime 四 owner×三个 lifecycle 字段
`handler_active / issued_outstanding / gpu_ready` 全部严格 `is False` 且 pending_frontier 为空时调用新rate。
任何 true/None 或非空frontier继续返回原 fallback.query。OCR idle 始终 UNKNOWN。

v41 实际训练只观察四个 projected handler booleans，未记录上述12字段/空frontier；
因此 runtime 域是未验证的 handler-only→complete-idle 迁移假设。原 median-clock projection 也没有
独立校准误差界。常率、跨route exchangeability、观测曝光可预测性以及独立增量均未从零事件数据识别。
projection、bridge、constant-rate 和 zero-event/prior flags 始终保留；年龄/预测期限超曝光范围另记OOD。

adapter 只提供 rate。原 CumulativeLifecycleHazard 仍积分同一个 hazard budget；
不重抽、不倒退、不提前终止、不把physical context视为SM进度。原 mark 完全不改；
若正率在没有 active target 时达到 SAVE，原 `NO_VISIBLE_ACTIVE_REFERENCE_TARGET` 仍可拒绝预测。
v53不是“强行跑通”的保证。

## 产物与验证

训练命令：

```bash
python -m tools.predictor_validation_20261027.idle_physical_save_hazard_v53 \
  --training-manifest artifacts/predictor-whole57-worker-handler-projection-v41-20261028-training01/training-manifest.json \
  --output artifacts/predictor-idle-physical-save-v53-20261028-training01
```

产物为该目录的 `training-manifest.json`、`model.json` 和 `report.json`。
模型含原字节来源闭包、精确N/T、完整per-capture计数与posterior参数；报告含4fold及上述限制。
`python -m unittest tests.test_idle_physical_save_hazard_v53 -v`：6项CPU测试通过，
覆盖分段曝光/单次事件、无暴露不pool、Gamma单位、精确NB、全12字段和frontier分派、posterior篡改拒绝，
以及真实累计控制器按旧rate积分、budget不重抽、时钟不回退。尚无独立GPU预测验证。
