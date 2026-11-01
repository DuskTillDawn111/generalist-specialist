# 任务列生成前的显存下界筛选

MILP 生成某个任务的路由/副本列后，在质量和 FIFO 计算前，先求所有活跃模型状态的 `state_memory` 之和；超过预算则跳过，计入 `column_memory_pruned`。只有保留的列占用 `milp_max_columns`，已被剪去的列不会触发列数限制。

这是既有主问题显存约束的提前应用，没有改变部署、路由网格或完整评分器。对已验证 v1/v2，`state_memory` 是 `safe_factor * memory_multiplier * standalone_residency`，完整部署的预测显存不小于活跃状态之和。`profiled_memory=True` 时绝不使用 additive 剪枝，完整非加性画像仍决定显存。未验证代理也不使用这项状态界。

新增四项 `tests/test_column_memory_pruning_v6.py` 检查：v1 和 v2 都在调用 floor 前拒绝超预算列、与完整独立笛卡尔枚举同优；非加性实测显存反例不被误剪；被剪列不占保留列数上限。四项均通过。

原始七模型配置的保守列数统计如下，没有创建校准模型、没有求效用、没有启动 GPU。使用原 `safe_factor=1.110854` 和预算 49140 MiB，仅将 multiplier 取 1 作为保守统计；后续有效的 multiplier 不小于 1，因此只会进一步减少保留列。

| 网格 | 原始任务列 | 超活跃显存下界列 | 保留上界 |
| --- | ---: | ---: | ---: |
| K=4 | 9112 | 7262 | 1850 |
| K=8 | 178104 | 163530 | 14574 |

K=8 的保守保留数已低于默认 50000 列限制。这只解决该列数限制，不表示后续质量/FIFO 列构建或完整搜索一定在时间预算内完成，更不表示这些列能组成可行完整部署。

统计保存于 `artifacts/static-study-gpu-20261022/column-memory-counts/report.json`，包含每任务细分、原配置及实际载入问题哈希，原配置未修改。另将 MILP 的 floor 设为零作纯列生成 dry run（未评分），K=4/K=8 的实际生成数分别也是 1850/14574，剪枝数一致。

```bash
python scripts/count_static_memory_columns_v6.py --config artifacts/static-study-v6-20261021/inputs/system-config.json --output artifacts/static-study-gpu-20261022/column-memory-counts/report.json
```

此修正完成后冻结算法实现，不继续扩展数学搜索域。
