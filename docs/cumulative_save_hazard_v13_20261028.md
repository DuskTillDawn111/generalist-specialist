# 累计 SAVE hazard budget 控制器

`tools/predictor_validation_20261027/cumulative_save_hazard_v13.py:CumulativeSaveHazardController` 是可组合的统计候选控制器，使用冻结 `SaveHazardCandidate` 的 piecewise rate。它不是物理执行门；六 context 的实际反例已经否证“kernel 只能在 owner envelope 内执行”的严格解释。本模块没有修改冻结的 `save_hazard_v13.py`。

每次 restore 固定一次预算，默认 median `log(2)`；也可显式声明 `SEEDED_UNIFORM` 和整数 seed，通过固定 SHA256 recipe 生成该 restore 的 uniform，再取 `-log(U)`。host/frontier 更新先积分旧 frontier 到更新 clock，只改变未来 rate；restore age、累计 hazard、预算不重置。各年龄 bin 精确分段积分，SAVE root 向上取整为整数 ns。补偿求和降低大量 host 刷新导致的舍入差异。

```python
from tools.predictor_validation_20261027.native_collect import receipt
from tools.predictor_validation_20261027.cumulative_save_hazard_v13 import CumulativeSaveHazardController

controller = CumulativeSaveHazardController(
    receipt('artifacts/predictor-save-hazard-v13-20261028-training01/age_additive-model.json'),
    allow_unvalidated_approximation=True,
    budget_policy='MEDIAN_LOG2',
    empty_frontier_policy='UNKNOWN',
)
result = controller.start_restore(owner, absolute_restore_ns, frontier, unique_restore_key)
result = controller.update(result['state'], absolute_update_ns, updated_frontier)
```

返回 `absolute_save_candidate_ns`、`next_review_ns`、sealed state、budget/consumed/remaining hazard、实际 restore age 和 uncertainty flags。rate 降低时保留旧的更早 review；旧 review 到达后只重算剩余预算的未来 crossing，不 renew grant。rate 提高时可提前 review。若一次 host 更新已经跨过累计预算，返回 `SAVE_BUDGET_REACHED` 和较早的 crossing，先处理该 SAVE candidate，不能跨过它应用新 frontier。terminal state 不允许 host 刷新重新授予预算；新的 restore 要显式调用 `start_restore` 并提供新的 key。

默认空 frontier 使该段 hazard 历史 `UNKNOWN`，之后不能声称有完整累计预算或绝对 SAVE。也可预声明 `NO_QUERY_PAUSE`：空段不查询、不增加累计 hazard，恢复时继续原预算和真实 restore age，并一直保留 `NO_QUERY_PAUSE_ASSUMPTION` 标记。这是统计模型的显式假设，不能解释为物理 GPU 在空段不 SAVE。两种策略均保留空段明细，不填补未测 hazard。

OOD、训练 age support gap 和不支持的尾部都传播到结果。数值候选可使用源模型在这些位置的 rate，但始终是 approximation，不声称 empirical support、概率校准或完整 finite distribution。`source_query_support_flags_clear` 只表示源查询标记，不证明 unseen feature combination 的条件准确率。

固定 source model receipt、hazard 实现 receipt、累计 solver 实现 receipt 和预算/空 frontier 配置一起产生 `controller.identity`。state 的内容 seal 与 controller identity 都会核验；跨模型/solver/config 的 state 会被拒绝。模型载入后使用私有内存快照；`assert_sources_unchanged()` 可在 campaign 边界核验文件，避免每个 DAG node 都重复 hash 模型。`provenance()` 返回这些收据和配置。

12 个 CPU 测试通过，覆盖独立 SciPy quadrature、常数解析解、年龄 bin、rate 升降、早 review、300 次 host 刷新、空段两种策略、seed、封印/来源/clock、OOD/tail 和 terminal SAVE 恢复语义。实际冻结训练模型的 text#0/decode smoke 给出 restore 后 163570ns 的 SAVE candidate；三次 host 刷新不改变预算或绝对 crossing，抵达时返回 terminal SAVE。无 GPU、无独立验证数据。

报告：`artifacts/predictor-cumulative-save-hazard-v13-20261028-cpu01/report.json`，SHA256 `f451038433509277295979c641ce1088f57327bd2ab54b4b2b2e34f731b49e41`。它仍需和 initial prior、next-owner、真实 handoff/prequeued 标签、gap/delay、host/joint 模型组合后接主 factory，再用完整原 cohort 检验。
