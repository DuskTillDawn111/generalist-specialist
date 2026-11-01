# v12：context 持有、待执行工作和 host 供给分别建模

本轮只使用已完成的训练采集与 CPU 重放。没有新增 GPU 请求，没有用独立验证结果拟合参数。冻结的 v7/v8 模型与诊断保持原样。

## 已确认的三重原因

证据为 `artifacts/predictor-first-contention-history-v8-20261028-diagnostic01/report.json`，SHA256 `5a33ec77ddfa7ffac6241f4be753dbbea7af8bf47d618ec1df1df74898bf2fb7`。

1. **训练和执行中的 current owner 含义不同。** 训练使用 GPU context envelope 中实际持有的 context；v8 执行器在只有一个 owner 有 ready kernel 时，把 `current_owner` 直接改为该 owner。它没有等待真实或预测 RESTORE，也没有独立保存 restore 时刻。capture02 的 SS-short 在相对首 ready 0.581313 ms 时，先前 ready owner 是 text#0，实际 held context 却是 text#1，age 9.385 μs、到 envelope 结束的余量 20.727 μs。训练器把这一类样本过滤，执行器又会产生这一类状态。
2. **独立 owner 的平均 host preparation 会改写联合顺序。** 实际两次 SS-short 均 text#1 首先出现 GPU 工作，text#0 分别晚 1.899450 ms、0.499001 ms。按独立 solo 均值生成的 prefix 则 text#0 先，text#1 晚约 0.446230 ms。修复首次 preparation 提前并没有识别共跑时的联合 host 供给分布。
3. **phase/frontier/current-owner 尚未证明足够描述残余。** `outside_language_forwards` 同时包括初始准备和生成尾部。capture01 的初始首次竞争是 text#0 outside + text#1 prefill；outside/outside 直到约 110.915 ms 的尾部才出现。capture02 初始 0.503111 ms 已出现 outside/outside。相同标签没有保存 context age、restore 时的条件、pending work 量、ready 等待年龄与 host 操作阶段。

因此不能通过补一个固定量子、把两次采集的经验余量填到所有格子，或仅追加相同 exact 表采集来宣布问题解决。

## 已执行的 preparation 修复及其边界

新源 `tools/predictor_validation_20261027/dag_preparation_host_release_v8.py` 把初始 prefix 与后续 issue-depth 分开。两次独立 light capture 的完整 runtime API 名称和 thread role 序列必须一致；初始 DAG API 按原始 ordinal 对齐到 natural 首 semantic 边界。pre-activity sync 独立保留，避免 OCR 约 100 ms 准备把最初的 sync 同时推迟。后段先扣除 preparation anchor，再重新拟合 depth 斜率。

冻结模型为 `artifacts/predictor-preparation-host-supply-v8-20261028-training01/model.json`，SHA256 `40198dc30429025ac33014f41b4b3b8097384e0decb65ac13b340cb1da1301f7`。独立重推和五项独立测试通过，见 `artifacts/predictor-preparation-host-independent-review-v8_1-20261028-cpu01/report.json`。

这仍是跨域 alignment 的条件假设。natural 没有观测首次 CUDA API，OCR 缺少 semantic anchor，保留了 light prefix transfer 假设。修复后的整 DAG 训练诊断仍是 14 solo jobs 数值、14 co-run jobs UNKNOWN；44 声明请求只有 14 个数值结果。solo 约 2.385% handler MARE 不能作为共驻准确度。

## 实际 v12 状态机

初始实现为 `tools/predictor_validation_20261027/held_context_scheduler_v12.py`。修订实现独立保存在 `held_context_scheduler_v12_1.py`，原 v12 与其 source replay 保持不变。

`HeldContextScheduler` 分别保存：

- `held_owner`、`held_knowledge`、`restore_start_ns`、`service_available`：context envelope 持有状态与年龄；
- `pending_since`、当前 frontier、每 owner 的剩余 reference work：待执行工作；
- `pending_switch`、绝对 SAVE/RESTORE/SERVICE 事件：已请求、尚未完成的切换；
- `review_deadline_ns`：下次政策判断的绝对时刻。

唯一 ready owner 不会直接改变 `held_owner`。ready 集合为空也不释放 context。host arrival、phase 更新、同一时刻重复调用不会重置 restore 时刻。切换请求先登记；SAVE_END 才清除旧 context，RESTORE_START 才设置新 context 和新的 age 起点，SERVICE_READY 才允许该 owner 的 reference work 前进。

在一个既有 review deadline 之前，host/frontier 更新不能把 deadline 向后延长。到达 deadline 后可以根据“已存活到现在”重新计算条件残余，但 age 仍从同一次 RESTORE 计算。phase 变化更新协变量，而不是抽取新的完整 grant。

v12_1 另外阻止远期 switch 绕过已有 deadline：若已有 review=20，host=5 时不能通过请求 save=100、restore=110 把 review 改为 150。旧 context 必须在已有识别范围内结束，或先在原 review 重新判断。初始 held 为 UNKNOWN 时也禁止通过一个任意 `hold` 超时伪装已识别 availability。

无政策参数时，实际调用严格抛出 `HELD_CONTEXT_AVAILABILITY_UNIDENTIFIED`。不会根据唯一 pending owner 选 context，不会采用默认量子或 equal share。

## 政策接口

```python
scheduler = HeldContextScheduler(resident_aliases, oracle=oracle)
decision = oracle.decide(query)
```

`HeldContextQuery` 包含 clock、held owner/knowledge、restore start、context age、service availability、pending frontier、每 owner remaining work、每 node pending age、已有 review deadline 和触发原因。

oracle 返回 `ContextAvailabilityDecision`：

- `hold(next_review_ns)` 保持 context，设置严格未来的 review；
- `switch(next_review_ns, target_owner, save_end_ns, restore_start_ns, service_ready_ns)` 声明绝对时刻；
- 不可识别时抛出 `UnknownEvidence`。

事件必须满足 `now <= save_end <= restore_start <= service_ready < next_review`。需要单独拟合 next-owner、handoff/service availability；仅有生存余量时不能据此臆造 switch target。

RESTORE_START 是 envelope 的观测边界；它并不证明同一时刻已获得完整 SM 资源。SERVICE_READY 与 reference-work rate 也必须由单独证据或明确假设支持。

## context-age 条件生存与 LOO

主线程负责从原始两次训练 run 的 context brackets 构造 age-entry / age-exit episodes：按实际 pending frontier 的变化切分，SAVE_END 为事件、frontier 改变为删失；held owner 不在 pending 集合时也保留它。不能再按“held != previous solo ready owner”删除样本。

采用 delayed-entry product-limit 时，risk set 必须对应 episode 的实际 attained age，按独立 capture 等权；不能把同一次 envelope 内多个 ready 点当成独立 capture。若 query 落在两个训练 run 均有 at-risk 支持的年龄范围内，条件残余分布可由 `S(a+r | history) / S(a | history)` 描述。尾部未识别的概率质量必须保留；不能补零或把 restricted mean 误称完整期望。

frontier 在生存期间变化属于 time-varying 条件。exchangeability、非信息删失、profile-domain → natural-domain 转移都仍是可检验假设。两 run 的 leave-one-run-out 只能给训练内可识别性和探索性误差；每折只有一个训练 run，不能冒称满足最终“两独立训练 run 支持”的放行标准。先比较原 phase/current-only 模型与 age 条件模型的 coverage、survival calibration 和 residual 排序，再决定需要哪些新增可观测协变量。没有支持的条件拒绝预测。

## 联合 host 情景

新的 host 供给情景应以一次训练 run 中同一 job 的所有 owner 为整体，保留 submit、handler-start、初始准备、首次 semantic/API 的联合顺序与间隔。不能分别求每 owner 均值后拼成一条从未观测到的联合轨迹。

需要把非阻塞 host 操作间隔与 GPU 阻塞同步依赖分开。直接把共跑实际 API 绝对时刻作为未来外生 release，会把原 GPU 等待固化到新调度里；这是重放而不是反事实预测。可执行 host DAG 应在 host thread 顺序中显式等待阻塞 API 的真实前驱完成，再消耗有独立证据的 host 间隔。

natural 和 light 的“packet01”是两次不同运行，不是同一随机实现。不能按同编号拼接跨域协方差。先保留 natural 联合 semantic/handler 场景和 light 联合 API-prefix 场景，再显式报告跨域 coupling 未识别；可以枚举允许的组合做稳健上下界，但不能悄悄按独立概率相乘。

LOO 按整个 run 留出，禁止拆同一 run 的 request 随机划分训练/验证。初始 owner 顺序、跨 owner first-ready 间隔、入竞争时 phase/frontier、请求全过程队列、P95 和六项效用的 candidate 排序需要一起检查。不同合理情景给出相反 ΔU 时，保留候选不可稳健区分的结论。

## source replay 的验证范围

`tests/test_held_context_scheduler_v12.py` 的五项测试包含真实 capture02 的 held/pending 反例、延迟 SAVE/RESTORE/SERVICE、空 ready/重复 host 调用不重置 age、既有 deadline 禁延长和未识别政策拒绝。source replay 使用显式 opt-in，普通预测器禁止注入观测 context 事件。

修订版本的 `tests/test_held_context_scheduler_v12_1.py` 保留上述测试，并增加远期 switch 绕 deadline 和 UNKNOWN 初始 hold 的反例，共七项通过。原始两 capture 的 source replay 检查前 5 ms 内 172 个 frontier 点，所有 paired-envelope holder/age 一致，其中 40 个点唯一 ready owner 与 held context 不同。

source replay 证明状态语义一致，不证明生存模型、自然域干扰预测、未来分流收益。v12 当前不支持 checkpoint；在完整 context/event/oracle 状态能够安全冻结和恢复之前，不能借用旧 scheduler checkpoint 宣称继续执行一致。

## 四任务补充证据

`artifacts/predictor-modality-host-preparation-v8-20261028-diagnostic02/report.json` 已核验原 image/ASR 四个 exact DAG，184,776 个 GPU nodes；原 `StaticDagPhaseBinding` 可以绑定这些原始 prefill/decode/outside 标签。没有凭 outside 标签猜测 image/audio encoder 的细粒度服务类别。

新 multimodal packet01 有六个 context-event keys，后两者在同一个已归属 OCR 的 opaque event globalPid 下稍后出现。原“四个 context”严格提取器已拒绝，没有丢掉附加 context 或伪造映射。将来需要显式支持一个 owner 对多个 context，并保持未配对 SAVE/RESTORE 的缺口，再训练多模态 availability。当前 modality preparation/context policy 保持 UNKNOWN。
