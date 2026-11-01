# v5 五模型 K4 真实静态搜索效率独立审计

审计输入为 `artifacts/static-study-gpu-20261022/five-model-v2-v5-K4-final/report.json` 及其 15 份独立进程原始记录。该实验使用真实画像拟合的冻结 v2 校准预测器，在 CPU 上搜索；本次审计没有重新求解或启动 GPU。

五轮中，MILP、带界 DFS、plain DFS 全部完成搜索、零 gap、无未知候选，15 次均选出完全相同的部署和路由，效用为 **-0.30930287426747927**。部署为 G2、image、ocr 各一副本，text/asr 路由至 G2，image/ocr 各至其专用模型。

## 效率结果

下表数值均由 15 份完整 JSON 重新计算，与原报告完全一致。

| 算法 | 会话构造与求解耗时中位数 | 独立进程总耗时中位数 | 完整三场景预测次数 | 最大峰值 RSS |
|---|---:|---:|---:|---:|
| MILP | 0.252850 s | 0.792042 s | 1 | 101956 KiB |
| 带界 DFS | 0.371656 s | 0.911831 s | 5 | 98624 KiB |
| plain DFS | 23.662888 s | 24.207291 s | 1696 | 98460 KiB |

完整预测次数在五轮内不变。plain DFS 的效用界计算、效用界剪枝次数均为零；带界 DFS 每轮为 909 个搜索状态，plain DFS 为 8659 个搜索状态。内存和容量可行性剪枝在 plain DFS 中仍保留。

采用“基线时间中位数 / MILP 时间中位数”的口径：

| 基线 | 会话与求解加速比 | 独立进程总耗时加速比 |
|---|---:|---:|
| 带界 DFS | 1.469871 倍 | 1.151241 倍 |
| plain DFS | 93.584852 倍 | 30.563125 倍 |

另保留逐轮配对比率，避免混淆两种统计量。plain DFS / MILP 的配对比率中位数为搜索 **94.184050 倍**、进程 **30.725106 倍**，逐轮范围分别为 93.172393–106.114820、30.438473–34.478318 倍。带界 DFS 对应配对中位数为 1.480470、1.170583 倍。

这些是同一主机上的五轮串行独立进程观测，没有构造置信区间。RSS 数值不支持 MILP 内存更低的结论。

## 范围与公平性

共同域是 G2 与 text/image/ocr/asr 四个专用模型，每个模型允许不部署或部署一个副本；每个任务使用分母 1..4 网格的并集，不添加容量边界 witness。共 17 个覆盖任务的部署、4097 个部署与路由标识；内存下界剔除一个五模型全驻留部署后，保留 1696 个标识的上界。共享负载 trace 包含 567 个请求。

三个算法共用同一冻结 v5 源码、输入、校准模型、任务质量记录、trace、网格、120 秒预算、100000000 状态上限、1000000 次 MILP 迭代上限及 1e-6 容差，均使用独立会话摘要缓存。顺序逐轮正反交替。对模型池的裁剪保留所有原始副本行和任务画像行；搜索允许的副本数单独约束为一副本。

该证据只覆盖上述有限预测域及 nominal 目标，不扩大为七模型、多副本、连续分流域或物理 GPU 全局最优结论。K8 或全模型池结果应引用各自独立的搜索证书。

## 严格审计与图表

新入口 `scripts/plot_static_search_efficiency.py` 读取已经存在的完整 benchmark 报告，不重新运行任何求解器。它执行以下核验后才允许绘图：

- 15 份原始运行文件、preflight 与报告内嵌记录逐值一致。
- 冻结数学模块、benchmark 入口、会话包装器 SHA 与运行记录一致；记录绝对输入路径及文件 SHA receipt。
- 调用冻结 benchmark 原有 `prepare` 和 `audit_domain`，重新执行校准门禁，重算原始/有效配置摘要、域、trace、预算及公平性字段。
- 所有运行的域、trace、预测器版本、会话、CPU affinity 一致；所有证书零 gap、无未决候选；赢家、预测效用、最优上界严格一致。
- 重新计算中位数、预测次数、峰值 RSS、逐轮同优标识，并拒绝与已有报告不一致的汇总。

输出目录为 `artifacts/static-study-gpu-20261022/five-model-v2-v5-K4-efficiency-figures/`，包括审计 JSON、输入 receipts、输出 manifest 和三组图，各有 PNG/SVG/PDF：

1. `cpu-search-wall-time`：会话与求解、独立进程总耗时，散点为全部五轮。
2. `cpu-full-prediction-calls`：完整三场景预测次数。
3. `cpu-milp-speed-ratio`：逐轮配对比率散点与中位数之比；图例明确区分口径。

每张图明确标示 15 次同优、共同效用、五次重复及有限域范围。绘图使用真实 CPU 搜索计时，未混入合成测试数据。

复现命令：

```bash
python scripts/plot_static_search_efficiency.py \
  --report artifacts/static-study-gpu-20261022/five-model-v2-v5-K4-final/report.json \
  --config artifacts/static-study-gpu-20261022/surrogate-v2/system-config.json \
  --output NEW_OUTPUT_DIRECTORY
```
