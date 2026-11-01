# TwoFIFO 独立预测验证：公开合同与接入缺口

新增 `twoFIFO_prospective_contract_v1.py` 只做 CPU prepare/audit。它不读取独立
验证输出、不拟合、不生成预测、不启动 worker。复用现有队列引擎、六项 scorer、
known-input gate 和多候选 ranker；不重写一套队列或效用公式。

## 已冻结的参照

`artifacts/predictor-twoFIFO-prospective-public-contract-v1-20261029-cpu02-prepared/`
保存原 `flow-validation-v13-ocr-audited/plan.json` 的公开字段，原字节收据见
`report.json`。完整 trace、输入内容身份、四候选 routes、rates、window/drain、
normalizers、阈值和 startup comparison declaration 都按原计划保留。

| 项目 | 原计划及当前 artifact |
|---|---|
| offered cohort | 181 requests：text 167、image 5、ASR 5、OCR 4 |
| offered rates | text 1.85；image/ASR/OCR 各 0.05 rps |
| window / drain | 90 s / 120 s |
| 候选 | shared_balanced / shared_skewed / shared_spill_low / shared_spill_high |
| 驻留 | text#0、text#1、generalist_2#0、ocr#0 |
| 输入 | 10 个原 pool IDs 与 exact content SHA；媒体文件另绑定实际 bytes SHA |
| 效用 | `configs/resource_spillover_objective_20261024.json` 的 Q/T/L/P/M/E 各 1/6 |
| 状态 | `CPU_PUBLIC_CONTRACT_FROZEN_HOLD_NOT_PREDICTION_SEAL` |

这里的“冻结”仅指历史公开合同的副本和原始 bytes，不表示本轮 fresh execution
或 prediction seal。原 request IDs 和 backend context 身份保留，以免通过改名
伪造新实验或输入支持。真正执行之前必须另封 fresh run/execution/request IDs，
保存旧到新 request mapping，保持每条 arrival/task/input/route 和评分尺度。

其他域不能混用：57-request availability-v11 只有 text/OCR 两种请求，虽然驻留
四 owner；`configs/resource_spillover_study_20261024.json` 是另一阶段的
600 s、30 s warmup、300 s drain、Poisson、text 1.85 加三背景各 0.01 的设计；
static-study-v6 是四任务等率 1800 s。181-row 是已有四任务预测 pilot 参照，
不能据此认证这些不同 horizon/rates/arrival process 的实验。tiny generated
text=1、OCR=1、image/ASR=.01 的 scoring fixture 也不能替代上述任何原计划。

## 实际支持矩阵

`planned-support.json` 从 source-rebound exact12 的 **公开 measured case 日历**
逐 whole block 统计 `(owner, task, exact pool_id)`。priming、驻留模型声明支持
某 task、另一个候选/输入的 template，都不能填补缺失。

本参照需要 16 个 owner/task/input keys；每个训练 block 中只有 4 个出现：
三条 text owner 的同一短 text 输入，以及 OCR owner 的同一短 OCR 输入。
剩余 **12 keys 缺失**，包括 image/ASR、两种其他 text 输入在三个 text-capable
owner 上的模板，以及另一个 OCR 输入。该矩阵只证明 planned coverage；即使
某 key 有计数，也不能声称 fitted root、完整 request vectors、transport 或
跨时间域 transfer 已通过。

当前 native joint law 用精确 owner/task/pool lookup。不能重命名 pool_id、换成
短输入、删除 background requests，或把 12 个缺失 keys 的 world 权重丢掉。
可考虑现有 provider 的显式组合，但每个组件都需独立的原来源、同域 context/
backend/generation/input 身份桥接；本 adapter 不认证这样的组合。

## 最小执行接线顺序

1. 先完成当前 source-rebound/recovery **训练**采集、拟合和 whole-block 诊断。
   它们已读过训练 heldout labels，不是本轮独立验证。
2. 对原完整 cohort 的每个所需 key 补齐/认证 provider support，认证原 backend
   到当前 backend 的身份桥接、原 full horizon、startup/priming、历史状态和资源
   域。现有 generated-case convenience API 固定 10+60 s，不能直接拿来执行
   90+120 s。可复用更底层 `compile_case_draws`、settled seed、ensemble engine
   和 scorer，但必须传本合同的原边界，不能改 engine 全局常量。
3. 明确预测原点。当前 settled-state continuation 可用已到达且可见的历史；
   busy numerical resume 未完成时保持 UNKNOWN。若在 priming 后才封 continuation
   预测，只能声称条件于该 past state 的预测，不能声称 startup 已被前瞻预测。
   任何用于 E/M 的当前 startup 观察也必须标记观测来源，不能伪称模型预测。
4. 另封 fresh IDs、全部 source/model/training-index、input catalog、joint draw
   identity/weights 和全部候选预测，随后才允许开始对应独立 native collection。
5. 复用 `known_input_gate_v5` 检查 service/E2E、每任务 P95、queue-depth、相对
   baseline 的 ΔU sign 和所有 candidate pairs 的 rank。它允许**训练内容相同、
   新请求/新共驻**，严格拒绝训练 run/request 冒充独立；不能称 unseen-input
   泛化。input-disjoint v2 gate 是另一种更强范围，不能删训练 index 绕过它。
6. `twoFIFO_candidate_ranking_v1.rank_candidates` 继续复用 paired worlds 原
   `compare`。point gate 需要一套 prospectively declared point projection；
   当前 adapter 故意没有定义，也没有把 finite-strata quadrature 宣称为校准概率。
   所有 world 的 Q/T/L/P/M/E 不全 known 时不输出完整域赢家；条件均值不参与排名。
   point latency vector 的 P95、world P95 期望和 mixture P95 必须分清。

资源和评分仍由原 scorer 负责：M 是 startup→drain owned sampled peak，安全
因子一次；E 是 measurement board energy 加 startup/3600 一次，drain 能耗
不挪入 measurement。完整 offered cohort 中缺 RESULT/unfinished 不删除。
原 `rho`、`g(n)`、稳定性、显存预算和 complete-drain 是另外的 qualification
gate；预测误差通过不等于最终收益，也不新增 cap1/cap2。

## 当前可复核入口

```bash
python -m unittest -v tests.test_twoFIFO_prospective_contract_v1
python -m tools.predictor_validation_20261027.twoFIFO_prospective_contract_v1 \
  --objective configs/resource_spillover_objective_20261024.json \
  --public-plan artifacts/predictor-gpu-20261027/flow-validation-v13-ocr-audited/plan.json \
  --campaign artifacts/predictor-twoFIFO-source-rebound-v1-20261028-cpu09/campaign.json \
  --public-catalog artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl \
  --output NEW_EXCLUSIVE_OUTPUT
```

6 项 CPU tests 覆盖原完整 calendar/routes、四 owner 与任务域的区分、每 block
支持而非 pooled support、原等权/hash/route 缺口拒绝、端到端 HOLD 及 artifact
篡改拒绝。它们证明准备合同的逻辑边界，不是 GPU 预测精度。当前没有独立验证
误差、ΔU sign accuracy、rank accuracy 或最终分流收益数值。

独审后补充：audit 从原 catalog 重新推导完整 selected input manifest，拒绝
text SHA 被改、输入被删却同步 artifact receipt 的伪造；task/owner/support
summary 也从原公开计划重核。cpu01 prepared/author 保留为历史，cpu02 使用修复
后的源码收据；6 个对应篡改 subcases 已通过。
