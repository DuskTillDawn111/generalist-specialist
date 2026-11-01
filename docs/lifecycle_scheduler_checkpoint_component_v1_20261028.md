# 生命周期 scheduler 模拟快照组件 v1

已实现独立可复用的 scheduler 组件导出、纯导入和事务恢复，主 `LifecycleDagPhaseModel` 尚未接入 resume。没有修改原 engine、forecast 或 GPU profile 闭包，也没有 native 观测同化、训练或 GPU 工作。

实现：`tools/predictor_validation_20261027/lifecycle_scheduler_checkpoint_v1.py`。

## 接口与边界

```python
checkpoint = export_scheduler_snapshot(
    scheduler, binding=binding, owner_last_ready=owner_last_ready)
checkpoint = json.loads(json.dumps(checkpoint, allow_nan=False))

# 纯校验、解码，目标对象不变。
detached = import_scheduler_snapshot(target_scheduler, checkpoint, binding=binding)

# 全部校验通过后替换 scheduler/oracle/mark 动态 dict。
owner_last_ready = restore_scheduler_snapshot(
    target_scheduler, checkpoint, binding=binding)
```

调用者必须独占已暂停的对象，暂停阶段固定为 `POST_DECISION_PRE_ADVANCE`。组件不调用 `snapshot()`（它会推进上下文）、`decide()`、`observe_*()`、`initialize()`、`start_restore()`、`update()` 或 `_budget()`。通过纯读字段和文件字节生成状态/静态身份；不推进时间、不消耗风险、不重抽 budget、不重复 observe。

`binding` 必须明确给出 execution/cohort/graph 的 SHA-256，原 now/deadline/drain、当前 clock 和 atomic stage。完整 cohort 的 hash 只是身份约束；本组件没有保存实际 arrivals/pending/running/DAG/资源状态，不能凭这个 hash 声称它们已恢复。

`owner_last_ready` 是独立必填映射，要求包含所有 owner、保留绝对 ns（允许 READY 尚在未来），恢复后作为独立返回值交给未来 engine adapter；不会偷偷写入无法访问的 v21 closure。这补齐旧 v7/v9 没有包含该字段的 schema 缺口，但不代表 engine 已使用它。

## 支持的真实实现

- 确切 v49 `LearnedLifecycleScheduler` 与其 v27 `LearnedLifecycleOracle`。
- 确切 v24 `CumulativeLifecycleHazard`：完整 sealed state 及 wrapper，包含 restore origin、observation、last update、risk compensation、consumed hazard、终止 SAVE、review、visible update count 和 hash；median 与 seeded budget 都支持。
- v44 `ObservableReferenceMark` 和 v56 `PhysicalContextMark` 的 typed lifecycle/clock，完整 resident 域不因恢复而重新过滤。
- controller candidate 为 v44 `ObservableSaveHazard` / v42 `ObservableContextCandidate`，或 v57 `ProjectedStateSaveHazard`。
- v14/v15 initial prior 与 v14 `HandoffCandidate` 作为明确无运行时累积状态的完整静态数据绑定，包含所有现有实例字段。

scheduler 保存 held owner/knowledge/restore/service、review、pending action、绝对 context events、pending_since、last frontier/clock、初始化选择、transition/query 历史和 v49 计数；oracle 保存 initial selection、restore key/history、hazard、current/last lifecycle、uncertainty 与 decisions。

所有这些对象使用确切类型与实例字段集合检查。未知子类、额外字段、未知 provider/mark 不会被忽略。新 typed JSON codec 不使用 pickle，也不采用旧 v7 checkpoint；精确保留 tuple/set/frozenset/deque/Counter 与冻结 dataclass，拒绝非有限数和重复键。

静态绑定包含实际 model/config/state 内容、原 public setup/workload/options、完整实现 import 闭包的直接字节 SHA，并核查组件现有 source receipts。恢复先检查 schema、hash、静态身份、时钟、完整 lifecycle、owner READY、sealed budget、accepted action 与绝对 context_events 一致性，再构造新对象 dict；提交出错会回退旧 dict。不会在验证失败后留下已修改的半个 scheduler。

## 明确尚不支持

- 初始 prior 已选出但尚未 SERVICE_READY 的 `context_events`；导出明确返回 `Initial pre-service context events are not yet supported`。普通已接受 pending switch 的绝对 RESTORE/SERVICE events 已支持并测试。
- 任意新 subclass、stateful provider 或额外实例字段。
- 主 engine closure 的 arrivals/cohort/queues/running/active/ready/receipt/READY events、DAG frontier/dependencies、原 validated rates/decision 和 next execution stage、queue diagnostics、资源/六项 prefix accounting。
- native 设备进度、实际 holder 识别、观测同化、预测准确率或 ΔU 收益。

所有 checkpoint 都显式带 `main_engine_resume_supported=False`、`physical_progress_observed=False`、`native_observation_assimilation_supported=False`。v21 主引擎仍拒绝把该组件作为 checkpoint 传入，测试也确认这一点。恢复后调用方必须继续原本已保存的 decision/advance 阶段；组件本身不替 engine 安排下一次 `decide()`。

## CPU 验证

`python -m unittest -v tests.test_lifecycle_scheduler_checkpoint_v1`：11 项通过。

- 对真实数值 v49/v24 的 median 与 seeded 路径，JSON 往返后累计 consumed hazard、原 restore age、review、下一决策和精确 SAVE 一致。
- 在 pending handoff 中暂停/恢复，保留绝对 RESTORE 时间；到达该时间后的决策/完整动态状态一致，没有新增 switch 或重复 observe。
- 将全部推进/观测/budget 方法替换为会抛错的 spy，export/import/restore 仍完成，且 source 状态不变。
- v56 保留全部四个 resident 概率；v57 hazard + v24 controller + v56 mark 往返后下一次真实更新相同。
- budget 和 pending time 被篡改后即使重新封 hash 也会拒绝；源文件或 graph binding 变化拒绝；纯 import 返回隔离副本且不修改目标。
- 显式保留 future owner_last_ready；未知字段/类型、初始 pre-service events、非有限值和重复键拒绝。
- 直接传给主 engine 仍拒绝，避免把组件测试通过误报成主引擎续算完成。

下一步是独立实现最小 engine-state adapter：保留完整 future cohort 和当前 FIFO、抽出原 engine closure 状态，保存一次 validated decision 后的推进阶段，与本组件一同原子导出/导入，再做整条轨迹对照。当前不复制四百行 engine，也不以重跑历史假装续算。
