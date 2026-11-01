# 固定 V4 conditional continuation

新模块 `tools/predictor_validation_20261027/admission_validation_continuation_v1.py` 提供 `prepare(output)`、`audit_protocol(path)`、`audit_root_review(protocol_path, review_path)` 和显式 `run(...)`。它只接续 V4，不调用 F8，不重新拟合服务模型，也不启动/重启 T8。

canonical 目录为 `artifacts/predictor-gpu-20261027/admission-validation-continuation-v1`。prepare 固定绑定原 T8 PID `85446`、Linux start ticks `602865199`、项目 cwd 和完整 `python -u -m ...admission_training_continuation_v1 run --protocol ... --review ...` 命令。PID 复用、另一个同模块进程、源代码变化或原进程无完成产物退出，均停止而不替代原运行。

prepare 只记录公开协议、原 T8 started/source/review 和预定路径，保存完整递归来源闭包。首次 audit 验证闭包完整性；同一精确协议 receipt 的后续等待循环只重哈希完整闭包，不重复解析历史 raw JSON 或运行数值拟合。协议变化不能命中缓存。

run 要求 canonical `root-review.json`，schema `gsdeploy.admission_V4_continuation.v1.main_thread_review`，decision `APPROVED_FOR_CONDITIONAL_V4_ONLY`，reviewer_role `main_thread`，created_ns 晚于 protocol。它必须绑定 protocol receipt，并逐项复制 protocol 的 source、training_protocol、training_review、validation_protocol、campaign、role_reservation、paths、training_process_identity；`final_gain_eligible=false`、`final_launch=false`，且 diagnostic_reason 为明确保留既有失败的非空理由。这是主任务内部来源审核，不新增用户审批。

run 排他锁与 exclusive started 使该接续器只尝试一次。原 T8 completed 出现后，只执行一次完整训练审核：原 8 个有序 completion、每个 1200 请求及精确 native/controller/review/source、严格训练 mapper、三个固定 manifest 及其原 mode protocol、原 upstream FAIL/INCOMPLETE。后续来源闭包持续重哈希，不按 V4 job 重复拟合。

固定输出为：

- `admission-validation-v1/sealed-v1`：原独立模块 CPU seal，使用 T8 continuation 的 resource-fit、quality-fit、capacity-fit 三个 manifest。
- `admission-validation-v1/execution-v1`：冻结 V runner 以 `diagnostic=True` 运行固定四条路由。
- `admission-validation-v1/report`：原独立报告、原 gate、容量附录及完整重算。

CPU 训练审核、seal 和 report 暂时隐藏 GPU 并将 BLAS/OpenMP 线程设为 1，结束或抛错后恢复原环境；显式 V4 GPU 调用继承主任务核实的设备环境。

若采集中途失败而原 run-state 可用，仍对严格完成的前缀生成 INCOMPLETE 报告，保留原四候选/4800 请求分母、原 error 和 native known；不调用 run 重试。来源变化或报告无法验证则保存 failed，停止接续。最终 completed 的 status 为 `DIAGNOSTIC_COLLECTED` 或 `INCOMPLETE_NO_RETRY`，而原 gate 单独保留在 independent_gate_status，绝不把采集完成解释为预测通过。

completed schema 为 `gsdeploy.admission_V4_continuation.v1.completed`，包括 source/protocol/root_review、training_completed、training_audit、seal_audit、run_state、report、gate、capacity_qualification、validation_audit、report_verification、source_receipts，以及 native_requested/native_known、independent_runs_launched、candidate_coverage、report_status、upstream_statuses。`training_launched=false`、`service_refitted=false`、`final_launched=false`、`not_for_training=true`、`final_gain_eligible=false`。实际 GPU 标志保留 run-state 原值。后续 F8 必须另行核验完整实际 4×1200 及六项效用，不能只依据本 completed 文件存在放行。

当前 CPU tests 使用明确的 fake collector/拟合审核，验证控制流和故障拒绝；不构成 GPU 或物理收益证据。它们覆盖进程身份与重复实例、协议重哈希、来源变更、root review、等待而不启动、一次性失败、全 8 项域/manifest、CPU 环境恢复、完整 V4 衔接及失败报告。
