# 固定条件的原接口容量校准方案

当前第一批校准范围是固定 `text#0 + text#1 + generalist_2#0 + ocr#0`、共享执行、90 秒完整到达、text 1.85 rps 与 image/asr/ocr 各 0.05 rps，以及 balanced、skewed、spill_low、spill_high 四种提前声明的 alias 路线。两枚新训练 seed 627401/627402 各执行四候选，共八条新训练流。来源为 `continuous-training-v10/suite.json`；v9 原始验证结果及其拟合误差不进入数值估计。

`plan.phase=predictor_training`、`kind=prospective_training`、`split=training` 与 generator SHA 在 worker startup 之前封存。冻结的 native_flow 原始记录缺 split 时，校准工具根据该计划生成有审计链接的 training evidence wrapper；仅派生角色元数据，保留原始时钟、输出、状态和全部原 native raw。原始验证记录不能通过后加标签转换成训练记录。

每个完整 condition 单独产生 catalog：条件身份包含执行 context、所有驻留 alias、完整资源配置、alias route 权重、offered rates、窗口／drain／warmup、到达过程及已声明 known-input 子域。balanced 与 skewed 聚合成 family routing 后相同，原 profile_key 无法单独表达其差别；独立 catalog 的 measurement_context 明确加入完整 alias 条件，从而避免共用／覆写画像。

原服务接口与 rho 的数值接法为：

1. 以每个成功训练请求的真实 worker END−START 为 service，按 family/task 聚合。排队、准入等待和父进程响应尾部不混入 service。
2. `family_normalized_service_ms = empirical_worker_mean_ms * original_capacity_factor(n) / n`。原 catalog 的 capacity_factors 全部原样保留，另保存摘要防止改动。
3. 切换为原 `performance_mode=profiled`、`memory_mode=profiled`，填充训练源明确的 whole-plan service_time_ms 和 `safe_factor * max(raw_owned_sampled_peak)`。安全因子应用一次，不把采样峰值宣称为容量上界。
4. 按原 `resolve_service` 取服务；原 `rho = sum(planned_routing_rps * service_ms/1000)/g(n)` 及 `0.8+1e-8` 阈值不变。有限实际请求数/H 和后验 busy ratio 不作为规划到达率。

拟合工具为 `flow_capacity_catalog.py --suite ... --output NEW_DIR`，要求所有声明训练流完成、来源完整、每条件覆盖两个 seed，才生成数值。每个 catalog 保存 service 总时长／请求数、每 run 均值、输入数、drain handler 占比、g/n 系数、原始显存峰值、wrapper 与原 raw/plan/source SHA；`verify_catalog(path)` 重新核验整棵来源及归一化算术。wrapper 不能替代原 raw 的来源核验。

这批估计仅使用八条新训练流。原接口 profile key 含 seed、窗口和到达过程，因此初版只生成两枚训练 seed 对应的 key。新的公开确认 seed／trace 在另一次运行前公布后，才可生成独立版本的 prospective transfer 计划。由训练均值预测新 seed 的结果必须标明预测／转移，不能标成该新 seed 已被实测。确认运行继续独立核验父进程全过程、六项排序与原稳定性。

该批画像的输入分布是两条新训练 seed 实际采到的 known-input 子域，估计保留有限样本与冷／热状态及 drain 的限制；没有已校准置信界或稳态容量证明。90 秒 pilot 画像不自动变成最终 30 秒 warmup／600 秒测量／300 秒 drain 的匹配容量证据。

后续 admission1、admission2 必须分别收集相同 route/load 条件，不能继承 shared 数值；最低分别再八条训练流。原 family-average rho 也不独立证明全局 admission slot 容量充足。真实准入语义、完整未来到达队列预测以及原独立稳定性确认仍须全部保留。共同 batch、公共权重子矩阵或 cache 后端改变后，执行身份及其容量画像同样须重新建立。
