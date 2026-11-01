# H600 固定路由整请求准入：服务机制试验 v1

这是独立预测验证完成之后的阶段三有限诊断。准备、封存、报告均为 CPU 操作；只有显式 `run` 会启动 GPU。它不构成最终四组分流实验，也不能证明六项效用或最优禁分流基线上的收益。

## 固定设计与范围

- 保持 `text#0`、`text#1`、`generalist_2#0`、`ocr#0` 四进程驻留，继续使用原始 `background-training-v7` 的公共输入域与不可变 worker context。
- 新公共到达种子固定 `627801`，H=600 s，warmup=0，bounded drain=300 s；text=1.85 rps，image/asr/ocr 各 .05 rps。三模式各保留全部 1200 请求，合计 3600 请求分母。
- 路由固定为 `shared_spill_low`：两专用分别约 5/12，G2 约 1/6；背景任务路由不变。每个请求的输入、到达时间、路由在三模式间完全相同。
- 只改变现有 native 整请求控制器：`shared_spill_low`、`admission1_spill_low`、`admission2_spill_low`，按此顺序执行。准入模式每 alias max_inflight=1，全局 max_active=1 或 2。相同 priming、warmup 和 native 采集器用于三模式。
- `prepare` 保留原 native 12 候选准备文件 `native-prepared-plan.json`，将最终执行 plan 限定到上述三模式，移除历史共享资源代理与省略启动比较声明，并校验其结构投影。不会改写原训练、独立验证或历史 native raw。

封存的身份分成两部分：基础 worker context 仍是共享训练所用的不可变执行身份；每个模式另有绑定资源配置及 admission/deployment/resources 源码的 `controller_sha256`。共享服务规律在准入下能否迁移是本试验待检验的假设，不会把共享模型称为经过准入训练的模型。

## 执行前的约束

`seal` 必须接收 selected 验证入口的 protocol、确切的 selected full16 模型、独立报告与 gate。它调用 selected 入口审计模型、封存源与严格实际证据，并使用原阈值重新计算 gate。四候选、完整 1200 请求分母是硬要求。

同一进程中，对同一直接输入只执行一次完整 selected 数值审计；缓存同时保留全部直接及递归 JSON 引用的 SHA 收据闭包、selected predictor 源码收据。每次复用仍重新读取并 hash 所有收据，任何模型、协议、raw、snapshot 或源码漂移均拒绝。新 CLI 进程需要重新完成首次审计，不通过布尔字段信任前次结论。

独立 gate=INCOMPLETE 时阻止准入封存/执行。完整 FAIL 保留原结论，后续只能通过 `run --diagnostic` 做有限机制诊断，不能升级为独立验证 PASS 或最终收益资格。该入口不读取独立结果调参、不重拟合、不换种子。

预测器直接调用冻结的数值 v2 全过程事件模拟器，不 monkeypatch 原模块。它只使用公共到达/输入/路由及服务模型，显式传入 UNKNOWN power/memory provider。启动、能量、显存、容量迁移、六项效用、ΔU 一律 UNKNOWN，不借用共享资源模型给准入模式打分。

每模式 CPU 预测保留完整队列和原 bounded drain。输出 `PREDICTED_COMPLETE`、`PREDICTED_DRAIN_INFEASIBLE` 或 `UNKNOWN_PREDICTION`，以及截止 drain 的模拟 handler permit 占用、完成数与未完成数。只有 PREDICTED_COMPLETE 会被 `run` 启动；其余模式保存明确 skip 记录，原 1200 分母与未完工预测继续保留。特别是 max1 如果预计排不完，不加长 drain、不减载、不丢请求，也不把缺测模式算作机制成功。

`run` 只能在全新输出目录执行一次，不自动重试。每模式启动前再次审计全部源文件；原始启动必须晚于完整源码/预测 seal。一个实际采集失败或证据不完整即停止剩余采集并保存 `INCOMPLETE_NO_RETRY`。

## 实际证据与报告

新 controller-aware adapter 直接验证 native raw、原 plan、单独 controller 身份和原始 invocation，再调用冻结的 `mechanism_audit.audit` 与严格单请求语义解析 `_request`。它不会调用或绕过旧 shared-only `map_flow_collection`。检查内容包括完整请求集合、实际路由与到达、READY PID/spec/clock、START/PHASE/RESULT、连续语义阶段、同 alias FIFO/非重叠、完整 priming/drain、清洁退出，以及 startup-through-drain 的 owned PID NVML 样本与原 2 s gap 限制。

实际语义轨迹标为 validation / not_for_training。报告包含：

- Worker START→handler END 的 service，scheduled offer→worker START 的等待，以及 scheduled offer→parent receipt 的 E2E。预测终点为 handler END，实际 parent receipt 尾部仍作为误差保留。
- 每任务完整请求群 P95、全过程 queue error、各项误差的实际配对覆盖率。组内有任何未知请求时，该组 P95 保持 UNKNOWN，不以已完成子集代替。
- measurement 与 whole_run 分开报告 observed permit peak、正准入等待以及等待区间中其他 alias 的 permit 证据。max1 需 measurement peak=1，max2 需 peak=2，且某等待区间内必须同时观测到至少 cap 个其他 alias 的 permit，即等待发生于已观测容量饱和期间，否则为 INSUFFICIENT_ACTIVATION。cap2 时仅另一个 alias 活跃和正时间戳间隔不能证明容量阻塞。priming 的 peak 不能替代 measurement 激活。
- 使用原 `flow_eligibility.native_stability`、原 catalog 及原稳定性阈值输出各模式稳定性结果。有限全完成/clean shutdown 不能代替原稳定性规则；rho 准入与容量迁移资格仍保持 UNKNOWN。

记录的 `[grant,parent_receipt)` 位于真实 permit 生命周期内部，peak 是观测下界。wait_start 在实际 ticket append 之前，grant 在 acquire 之后；不能重建精确全局 FIFO。模拟器按 alias 处理同时 offer、在 handler END 释放 permit，与真实 parent handling 释放的差异保留为预测误差。该审计不证明 SM/DRAM/L2 隔离、batch/公共权重缓存或 MPS 生效。

报告保留缺测模式与完整 3600 分母。只有三模式请求完整、两个准入模式在 measurement 均激活时，才能给出有限的 observed-activation 结论；不会产生自动准确度 PASS、六项效用排序或分流收益声明。单次诊断不是稳定性资格或最终收益确认。

## CPU 准备与后续命令

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest tests.test_h600_admission_pilot_v1 -q

python -m tools.predictor_validation_20261027.h600_admission_pilot_v1 prepare \
  --profile-dir artifacts/predictor-gpu-20261027/background-training-v7 \
  --output artifacts/predictor-gpu-20261027/h600-admission-pilot-v1
```

独立报告与 gate 产出并审查后，才运行以下封存命令（此命令仍为 CPU）：

```bash
python -m tools.predictor_validation_20261027.h600_admission_pilot_v1 seal \
  --plan artifacts/predictor-gpu-20261027/h600-admission-pilot-v1/plan.json \
  --model artifacts/predictor-gpu-20261027/background-statistical/gamma-h600-v2-full/model.json \
  --selected-protocol artifacts/predictor-gpu-20261027/flow-validation-h600-selected-v16/protocol.json \
  --independent-report artifacts/predictor-gpu-20261027/flow-validation-h600-selected-v16/report/report.json \
  --independent-gate artifacts/predictor-gpu-20261027/flow-validation-h600-selected-v16/gate.json \
  --output artifacts/predictor-gpu-20261027/h600-admission-pilot-v1/sealed
```

之后由主线程审核预测的 drain 可行性，按阶段顺序显式启动 `run --sealed-dir .../sealed --output .../runs --diagnostic`，并用 `report --sealed-dir .../sealed --raw .../runs/shared_spill_low/raw.json --raw .../runs/admission2_spill_low/raw.json --output .../report` 收集实际执行的模式。缺测模式仍留在最终报告中。以上路径中的 selected 模型以独立 seal 的精确 receipt 为准；不允许在运行时替换模型。

本轮实现准备不运行上述 GPU 命令、不进行新 fullfit、不提前封存或读取尚未完成的独立结果。
