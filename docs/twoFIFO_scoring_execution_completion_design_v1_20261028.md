# TwoFIFO 评分层与执行层补齐设计（v1）

本文只定义下一阶段的接口和验收边界。它不修改冻结 runner、采集器或
`twoFIFO_forecast_ensemble_v1` 的现有行为，也不把 CPU 规划结果当作 GPU
收益证据。所有机制均须在独立预测排序通过后，使用相同公开 arrivals、输入和
固定 measurement/drain 合同做实测消融。

## 1. 多候选完整排序接口

现有 `compare(baseline, candidate)` 只能验证一对成对 finite ensemble。新增的
排序层应当是一个独立、纯 CPU 的 companion（建议文件名
`tools/predictor_validation_20261027/twoFIFO_candidate_ranking_v1.py`），接收
`{candidate_id: ensemble_document}` 和一个预声明的 `utility_tie_tolerance`。
它只消费已经生成的 ensemble，不拟合模型、不选择 draw、不读取 validation
labels。

输入校验必须逐候选验证：

* `schema`、完整 offered request 集、`original_window`、`original_drain_end_ns`、
  `score_comparison_contract` 和质量证据相同；候选可以改变 owner/route，但
  `(request_id, task, pool_id, arrival_ns)` 不能改变。
* draw 的 `(draw_id, block_id, weight)` 顺序和精确权重在所有候选间相同。
  一个 draw 代表同一 joint world；禁止每个候选另抽一套世界。
* 原始六项 `Q,T,L,P,M,E` 仍各占 `1/6`，调用已有 score bridge；不能按已知项
  重新分配权重。原 `rho`、`g(n)` 和稳定性规则保持在外层资格检查中，不能在
  排序器内新增全局 cap1/cap2 或候选数量上限。

输出应包含每候选的完整审计行：

```text
candidate_id
offered_requests
draw_count
known_utility_probability_mass
unknown_utility_probability_mass
per_draw_full_U             # known value or null, preserving draw identity
per_draw_term_status        # each Q/T/L/P/M/E known/unknown
complete_utility            # only when utility known on every draw
stability_status             # copied qualification, never inferred here
rank_status
```

排序规则是一个可审计的偏序，而不是把未知项强行变成数值：

1. 只有所有 paired draws 的 `full_U` 都是有限值时，才可给候选一个
   `complete_utility`（按原权重求期望）并进入普通排序。
2. 若某候选有 UNKNOWN world，或某个 GENERATED world 的 M/E/其它六项缺失，
   保留其原始概率质量，`complete_utility=null`。应输出条件均值仅作诊断，
   且字段名必须带 `conditional_on_known`；它不能参与最终排序。
3. 对任意候选对，输出逐 draw `delta_U`、已知质量、未知质量、正 ΔU 质量界和
   `candidate_better`。只有未知质量为零且期望值有限时，`candidate_better` 才
   是布尔值；否则为 null。可选的严格支配标记只在每个 draw 的 ΔU 都超过
   tie tolerance 且没有未知质量时为 `DOMINATES`，其余为
   `UNORDERED_UNKNOWN`/`TIED`。
4. 排名表必须保留并列和不可比较候选，不能依照条件均值、已知子集或 P95
   世界均值挑出“胜者”。`mean_of_world_P95_is_not_mixture_latency_P95`
   继续为真。完整 cohort 的 T 与 P95 终点仍是 scheduled arrival 到 RESULT
   reader receipt，drain 未完成请求不能从分母删除。

这样可以把任意数量候选接到同一评分层，同时不改变现有 `compare` 的二候选
审计结果；新的 companion 应以同一 `compare` 逻辑逐对交叉核验。

## 2. 六项评分与稳定性接线

评分层只接收 `actual_worker_world_score_v3.score_world` 或相同 schema 的完整
world。它不自己估计质量、启动能耗或显存。资源缺证据时 M/E 保持 UNKNOWN，
`full_U` 和绝对收益也保持 UNKNOWN；不能用零、条件均值或其他候选的资源值填补。
原 startup 能耗一次按 3600 s 摊销，M 为 startup→drain owned sampled peak，
安全因子只由原 scorer 应用一次。任何执行机制的额外准备、padding、拼接、cache
构建和残余 GEMM 必须进入该机制自己的 service/E/M 证据；没有独立 profile 时
整项机制 world 为 UNKNOWN。

稳定性是单独的资格 gate：排序器只复制已有
`stability_state_v2`、`rho`、`g(n)` 和原因，不计算新的阈值，也不把“预测可排序”
当成 stable。候选必须同时满足原稳定性、显存预算、完整六项和全 cohort drain
才可进入最终收益比较。

## 3. 执行机制 sidecar（不改冻结 runner）

建议新增 `twoFIFO_execution_mechanism_sidecar_v1.py` 及其设计用 JSON。sidecar
在候选的 `execution_context_sha256` 下声明机制配置和证据身份，并由一个新的
provider adapter 读取；冻结 runner 继续只接收已认证的 provider。

机制因子应至少有四个预声明值：

* `solo_baseline`：现有单请求、无共享权重的执行语义；
* `shared_batch`：同 owner、同模型/adapter 的共同 batch；
* `common_weight_submatrix`：最大公共连续权重块一次 GEMM，随后各自残余块；
* `usher_roomie_schedule`：USHER 投影的计算型/显存型工作类、phase/顺序和
  启动差作为调度诊断 sidecar。

每个机制 world 必须显式声明 `mechanism_id`、执行上下文、输入身份、后端版本、
证据收据、准备/执行/清理时间、资源范围和 `known`。缺任何字段应产生
`UnknownEvidence`，而不是经验折扣。

### 3.1 Batch/padding

批形成仍按公开到达和预声明 FIFO/最大 batch/最大等待规则。每个 batch 记录
成员 request IDs、实际 dispatch 时间、输入 token 宽度、padding token 数、
batch backend identity 和整批 RESULT 语义。若后端是整批返回，所有成员共享真实
batch completion/reader receipt；不能从一条 `generate` 回放中伪造逐请求提前完成。
配批等待、padding、batch prepare、shared tail、质量、startup、M 和 E 都要进入
同一机制 profile。batch2 没有精确 profile 时，只能 UNKNOWN，不能由 batch1 线性
外推。由于共同 batch 会改变 owner 结构，必须将 batch-aware group 标识接到
队列诊断，而不是把四 owner 单请求计划硬改成三 owner。

### 3.2 最大公共权重子矩阵与 cache

现有 `gsdeploy.backends.shared_linear` 只证明 CPU float32/64、字节相等的最大
连续矩形和代数正确性。执行 sidecar 可以引用其 `weight_identity`、矩形坐标和
残余 GEMM 数，但只有真实后端确认 dtype、layout、量化 scale/zero、adapter、
kernel 和输出语义一致后，才允许设置 `known=true`。

cache key 必须至少绑定 `(hardware_id, backend_id, model/adapter identity,
dtype/layout/quantization, common_rectangle, execution_context_sha256)`；权重
驻留复用、共同 GEMM、prefix/KV cache 和 L2 命中是不同机制，不能用一个
`cache_hit` 字段互相代替。缓存建立、驱逐、拼接、残余计算和并发同步成本必须
单独计入 service/M/E。缺少容量、生命周期或收益测量时保留 UNKNOWN。

### 3.3 USHER/Roomie 接线

计算型/显存型分类、projection work、phase、顺序和启动差可以先作为 sidecar
特征和候选排序诊断；它们不能直接改写 TwoFIFO 的 `rates`，也不能把静态
occupancy 当吞吐。只有同硬件、同输入、同后端的自然单路/共驻配对，具备 kernel
时间域和资源证据时，才能由 provider 返回机制特定 service/resource；否则返回
`UnknownEvidence`。任何 Roomie/USHER 消融都必须比较相同 arrivals/inputs、相同
固定 window/drain、相同六项和原稳定性 gate。

## 4. 消融与验收顺序

在预测排序 gate 通过以前，只做 CPU schema/contract tests。通过后，按以下预注册
矩阵分别采集并保持所有失败/HOLD：

1. `solo_baseline`；
2. baseline + `shared_batch`；
3. baseline + `common_weight_submatrix`；
4. baseline + `usher_roomie_schedule`；
5. 可选的 batch + common-weight 组合，只有前两项各自有独立 profile 才运行。

每组都要有同一公开 cohort 的完整 offered arrivals、single/paired/background
控制、输入长度和启动偏移，记录 worker/reader/collector FIFO、RESULT receipt、
startup→drain M、measurement E、质量和稳定性。最终只在每组完整六项且稳定时，
做四组公平分流收益：原执行禁分流、原执行允许分流、优化执行禁分流、优化执行
允许分流。任何机制未知、验证不完整或候选不可比较时，报告 HOLD/UNKNOWN，不能
用 CPU utility 或 projection proxy 代替收益。

## 5. 纯 CPU 测试建议

新增 companion 的 unittest 应至少覆盖：

* 三个以上候选的同 draw identity、不同 route 的完整排序；draw 顺序、block、
  精确权重不一致时拒绝；不同 arrival/input cohort、score contract、window 或
  drain 时拒绝；
* 一个候选含 UNKNOWN_WORLD、另一个含 GENERATED 但 M/E UNKNOWN：未知概率质量
  原样保留，conditional mean 不进入 rank，`complete_utility` 和最终 gain 为
  null；全 known 时逐对结果与现有 `compare` 完全一致；tie tolerance 和严格
  支配边界；
* T/P95 使用完整 offered cohort；unfinished、缺 RESULT、缺质量、缺 M/E 都不
  被删除或补零；均值 world P95 不被标成 mixture P95；
* 六项等权固定、`rho`/`g(n)`/稳定性字段只读复制，任何 cap1/cap2 或部分权重
  重分配输入均拒绝；
* batch 成员、padding 宽度、整批 receipt 语义、batch1→batch2 缺 profile 的
  UNKNOWN；跨 candidate/context 的 cache key 碰撞拒绝；公共矩形 dtype/layout/
  adapter 不一致拒绝，残余 GEMM/拼接开销不被删除；
* sidecar 证据收据、机制/后端/硬件身份篡改、重复 cache 生命周期、缺 kernel
  counter 或静态 occupancy 输入均保持 UNKNOWN，不能升级为物理 service rate。

这些测试应使用 `python -m unittest`，不调用 GPU、不读 validation labels，且不
修改冻结 runner/采集器。独立预测 gate、native review 和最终 GPU 消融仍按已有
continuation/coordinator 顺序执行。

