# 累计 hazard 的左截断条件初始化

`cumulative_save_hazard_v14.py:CumulativeSaveHazardController` 修正 initial-prior 接线：第一次 service query 已知 context 存活到当前 clock，不能从历史 restore 积分 median budget 并在观测之前触发 SAVE。冻结 v13 controller 和 `save_hazard_v13.py` 均保留。

接口与 v13 相同，仅新增：

```python
result = controller.start_restore(
    owner, original_restore_ns, frontier, unique_restore_key,
    observation_ns=current_clock_ns,
)
```

`observation_ns` 默认是 `restore_ns`，必须是整数且不早于真实 restore。输入表示调用者提供“已存活到 observation”这一条件；预算为从 observation 开始的剩余累计 hazard `log(2)`（或固定 seeded uniform 预算）。历史 restore 不移动，之后 age 仍是当前 clock 减原 restore。状态显式保存 origin restore、observation、preentry 区间，`preentry_hazard=None` 且未积分。该条件不是从物理执行门推导的。

初始化仅在 restore 的首次观测入口执行。后续每个 clock 都调用 `update`，包括空 frontier；不能为 host 更新重新调用 start/recondition。旧累计预算、真实年龄、observation 和更早 review 均保留。空段 `NO_QUERY_PAUSE` 必须预声明，并保留来源假设；默认 `UNKNOWN` 仍不填补未测 hazard。六 context 反例对应的非物理执行门约束继续保留。

17 个 CPU 测试通过：全部 12 个 v13 合约，加 88ms/1ms attained-age 条件初始化、entry=restore 默认等价、空段后 origin/假设保留和非法 clock。88ms 测试使用独立 tail age rate，区分“保留旧 restore 的条件剩余预算”和“把 restore 移到 observation”两种行为。实际冻结训练模型 smoke：age 1ms 的条件 SAVE 在 observation 后 221823ns；age 88ms 后 226087ns。两者都从已知存活的 observation 开始，三次 host 刷新不重抽预算且抵达时产生 terminal SAVE。

报告：`artifacts/predictor-cumulative-save-hazard-v14-20261028-cpu01/report.json`，SHA256 `f902517ac5bc346412677f84db9687071fd13755235b074091e210a88962420a`。这是实际模型 CPU 接口检查，未使用新 GPU 数据、不代表独立预测准确率。
