# NVTX 区间索引及真实 SQLite 等价核验

`trace_export.parse_sqlite` 原先为每个 CUPTI kernel 扫描全部 NVTX ranges，复杂度为 `O(K×R)`。现在使用独立纯 CPU 模块 `nvtx_interval_index_v2.NVTXRangeIndex`，按完整 `globalTid` 分组，再建立按起点排序、记录子树最大终点 `maxend` 的平衡区间树。

查询保留原条件：同一线程，`range.start <= launch.start` 且 `launch.end <= range.end`。不能以时间重叠代替双端包含，也不把 kernel 的异步结束时间强塞进 host range。归属继续优先 GS_STAGE，其次选择最短 range；长度相等时保留原输入顺序。嵌套、交叉、重复范围及零时长 launch 均保留原语义，owner 返回原字典，使 kernel append 无需复制。

构建时间 `O(R log R)`、空间 `O(R)`；查询通过起点和 maxend 排除无关子树。测试在 255/2047/8191 个互不重叠区间上验证访问节点随树高增长，而不是每次遍历全部 ranges。当大量嵌套区间全部包含查询时，输出本来就可有 `R` 项，访问仍可为线性；没有声称所有输入均为常数时间。`owner_for_launch` 流式选最优，不对候选另排序。后续匹配 anchor 的代码原本只遍历 ranges 一次并查询字典，未发现第二处 `K×R`，保持不变。

## 两层真实证据

第一层：`nvtx_interval_audit_v2` 只读打开四个 structure-smoke SQLite，用独立逐项扫描 oracle 核对所有 55,891 个 kernel 的 owner，**0 差异**。oracle 和索引均复用相同 launch 查询结果，单独记录查询次数、节点数、完整扫描谓词次数及耗时；没有把缓存效果算成索引收益。

| trace | kernel | ranges | 索引查询秒 | 扫描 oracle 秒 | 索引访问/扫描谓词 |
|---|---:|---:|---:|---:|---:|
| text#0 | 19,423 | 3,375 | 0.200 | 3.551 | 1.35% |
| text#1 | 19,423 | 3,375 | 0.200 | 3.672 | 1.35% |
| generalist_2#0 | 11,552 | 1,275 | 0.103 | 0.793 | 2.99% |
| ocr#0 | 5,493 | 85 | 0.024 | 0.029 | 17.27% |

这些是本次 CPU 解析测量，不是 GPU 推理、训练或分流收益。

第二层：接入真实 `trace_export.py` 后，`nvtx_parser_equivalence_v2` 用旧 strict-review-v3 中保留的显式 anchors 重新解析，核对整个 JSON payload：kernels、NVTX ranges、phase union/sum、clock mappings、unmatched counts 等所有字段完全相等。四个完整解析耗时分别约 0.579、0.549、0.361、0.147 秒。旧 SQLite 和旧解析 JSON 均未改写。

新证据目录：

- `artifacts/predictor-gpu-20261027/nvtx-interval-index-equivalence-v2-20261028/report.json`：全 kernel 独立 owner oracle。
- `artifacts/predictor-gpu-20261027/nvtx-interval-parser-equivalence-v2-20261028-normalized01/report.json`：接入后的完整 JSON 等价。
- `artifacts/predictor-gpu-20261027/nvtx-interval-index-final-review-v2-20261028/report.json`：两层证据、历史 exporter 归档及当前源快照的综合回执。

首次完整对比的目录 `nvtx-interval-parser-equivalence-v2-20261028` 保留 FAIL：对比工具错误地把内存 tuple 与历史 JSON array 认作不同类型。修复为统一 JSON 序列化口径后写入新的 `normalized01`，不覆盖首轮记录。回归测试同时证明序列化类型处理正确且能检出真正的 clock offset 数值变化。

## 验证与后续冻结

```bash
python -m unittest tests.test_nvtx_interval_index_v2 tests.test_nsys_trace_export tests.test_native_profile_collection tests.test_kernel_profile_campaign_v2 tests.test_prediction_observed_v2 -q
```

本次 43 项通过；包括 1,200 组独立随机 oracle 查询、多进程同 correlation、多线程、跨 range 边界、nested/crossing/duplicates、最坏线性输出、SQLite 只读检查，以及集成解析器对逐项 oracle 的完整输出比较。

新 campaign 必须同时冻结 `trace_export.py` 和新增依赖 `nvtx_interval_index_v2.py`；只有 exporter 源回执不足以绑定实际解析行为。历史 campaign 不更新身份，不重跑 GPU。索引没有填补 SM/DRAM/L2 counter 或 occupancy/吞吐证据，也不改变预测 gate 或原六项评分。
