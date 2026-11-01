# 全过程预测诊断与原资格规则的独立附加证据

`tools/predictor_validation_20261027/flow_eligibility.py` 提供原规划 rho 与原实测稳定性的附加报告，不修改已封存的预测器、runner、原始结果或效用报告。本附加报告单独不能授权最终四组收益结论。

原规划约束来自 `SystemConfig.validate_plan`：

`rho_m = sum_t routing_rps[t,m] * resolve_service(catalog,...)[m,t] / 1000 / capacity_factors[m][n_m]`

其中路线是候选预先声明的 alias 权重乘总 offered rate 后按 family 汇总，service 由原 resolver 获取，容量分母是原 `g_m(n)`，不直接以副本数代替。原阈值为 `0.8 + 1e-8`。静态逐请求路线必须能由同一 Router 及权重重放。实际完成数/H、实际 busy ratio、实际 service 和有限 forecast 的 service 总量/H 均不能替换该规划公式；也没有新增每 alias 阈值。

复用接口为 `original_rho(plan,candidate,catalog)`。`rho_threshold_satisfied` 只表示给定 catalog 下的原公式算术诊断；`rho_admission_passed` 另要求明确绑定当前 execution context、完整资源配置及训练来源的容量合同。旧 catalog 不含新执行身份和资源性能绑定时，此正式字段为 UNKNOWN，即使旧 catalog 的数值小于 0.8。

稳定性使用原 `gsdeploy.stability.reconstruct/classify` 和全部原 `DEFAULT_THRESHOLDS`，不修改门槛。仅适配 measurement_window 的字段名：提交积压使用 actual_submit_time_ns，完成终点使用父进程 completion_time_ns；scheduled 到达积压仅作原诊断。service-demand backlog 继续使用原 catalog 的 standalone s0，绝非推断剩余 GPU 工作。priming、warmup、runtime/telemetry errors、shutdown 与 bounded drain 全部保留。缺真实提交时间、全部 worker 的 shutdown 证据或完整 native collection，均不得默认通过。

```bash
python -m tools.predictor_validation_20261027.flow_eligibility seal \
  --plan PLAN.json --output NEW_ELIGIBILITY_SEAL.json
python -m tools.predictor_validation_20261027.flow_eligibility report \
  --sealed ELIGIBILITY_SEAL.json --raw SCORED_RAW.json \
  --raw OTHER_SCORED_RAW.json --output NEW_ADDENDUM.json
```

seal 绑定计划、catalog、原规则源码和执行来源，并记录单调时间。report 重新核验 SHA，严格要求 seal 先于各实际测量开始；晚于测量的封存不能补成前瞻规划证据。每项资格均显式记录，缺失不变成 True。

当前固定驻留 pilot 可比较预测与实际候选排序，即使正式容量／执行协议资格尚缺。最终四组实验仍须全部原规则及完整质量、显存、能量、cohort、执行协议与重复确认通过。统计预测的可计算性、局部精度通过和原资格分别报告。
