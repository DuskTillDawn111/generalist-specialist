# 静态搜索会话：保持计算结果不变的摘要缓存

本阶段只减少 CPU 重复计算，不改变效用、预测器、搜索界、质量记录或 GPU 运行方式。`gsdeploy/search_session.py` 从显式指定的冻结源码目录加载私有 Python 包；缓存只存在于该会话中，普通 `gsdeploy` 模块和可变输入的完整校验保持原样。

## 实现约束

1. 先深复制原始 JSON，并用冻结源码原有的 `SystemConfig` 和 v1/v2 校准门禁校验，再递归冻结嵌套字典、列表。新会话重新校验，不继承前一会话的缓存。
2. 冻结节点具有该冻结树独有的 token。原有 JSON 摘要函数在缓存未命中时原样执行；缓存命中返回之前得到的原始摘要。
3. 临时字典、列表使用递归结构和值作为 key，其中不可变子树用 token 表示。禁止使用临时对象的 `id`，以免 Python 复用对象地址而误命中。浮点 key 使用 `float.hex()`，区分正负零；非有限值仍被拒绝。
4. 仅替换私有模块中指向原始 `quality_digest` 的别名。退出会话清理私有模块命名空间，不修改公共模块、冻结源码文件或预测公式。
5. MILP、带界 DFS、plain DFS 都通过同一个会话包装器创建，各算法仍使用自己的原有搜索过程。plain DFS 仍关闭效用界剪枝。

## 瓶颈证据和等价性

CPU profile 位于 `artifacts/static-study-gpu-20261022/v4-prediction-profile/`，使用显式 v4 冻结源码：

| 工作 | 总时间 | 摘要相关时间 | 说明 |
|---|---:|---:|---|
| 赢家完整三场景预测 | 0.287 s | 0.253 s | 81 次 `quality_digest` |
| 原始模型池 K4 列生成 | 8.005 s | 5.872 s | 965 保留列，8014 内存剪枝列 |
| plain DFS 的 10 s 有界观测 | 10.127 s | 8.999 s | 35 次完整预测；TIME_LIMIT，不构成最优证书 |

`artifacts/static-study-gpu-20261022/search-session-equivalence/real-v4-report.json` 对五个真实配置候选完成独立私有模块之间的逐值比较：原赢家、纯专用、拥塞混合、G1 n2、G3 n2。完整预测序列化结果逐字节相同；列界和 partial 界通过 `float.hex()` 精确相等。未知场景继续为 `null`。原赢家效用仍为 `-0.30930287426747927`，纯专用为 `-0.34134155055108345`。缓存共 538 次命中、17 次未命中。

上述五次候选计时是单次工程观测，不能替代重复的算法效率实验。算法效率必须以同一冻结源码、同一模型域、网格、预算和会话机制的 fresh-process benchmark 为准。

## 三个入口

入口默认从 `--source-root/gsdeploy/search_session.py` 加载包装器；`--wrapper` 仅用于明确指定其他版本。正式运行应让入口脚本、包装器和数学源码都来自同一冻结快照。

```bash
python FROZEN/scripts/run_search_session.py \
  --source-root FROZEN --config CONFIG --output single.json \
  --algorithm milp --route-grid 4 --time-limit 120

python FROZEN/scripts/solve_static_session_study.py \
  --source-root FROZEN --config CONFIG --output six-roles \
  --algorithm milp

python FROZEN/scripts/compare_calibrated_pool_solvers.py \
  --source-root FROZEN --config CONFIG --output benchmark \
  --session --models generalist_2 text image ocr asr \
  --route-grid 4 --time-limit 120 --repeats 5
```

六角色入口保留 `gsdeploy.static_search_roles.v1` 契约，每个 `result.problem` 是包含默认字段的普通完整 JSON；纯通、纯专角色只裁剪原契约规定的三个模型字段，保留原任务负载、质量和画像记录。排除部署、固定部署后排除赢家路由、禁分流等限制写入 `search_restrictions`，结果可由原 `gsdeploy.confirmation._check_search` 直接验收。每完成一个角色即落盘。

效率入口每次用新的串行子进程运行三个算法，三者都启用会话缓存或都不启用。完整结果保留 TIME_LIMIT、gap、未知域、完整预测调用数、耗时和峰值 RSS，只有三者全有证书且效用一致才能标记 `benchmark_complete`。不要求并列最优时部署和路由标识完全相同。

`tests/test_search_session.py` 的七项检查覆盖冻结输入、临时对象 key、公共门禁隔离、v1/v2 完整预测与界精确相等、三算法与独立笛卡尔枚举一致，以及六角色和独立进程入口的实际调用。冻结 v5 的独立入口核验另保存为 `search-session-equivalence/v5-cli-synthetic-report.json`；合成测试不作为真实测量证据。
