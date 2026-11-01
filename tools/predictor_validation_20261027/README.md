# 独立预测验证

CPU 分析入口，不训练、不运行 GPU、不把 CPU 构造数据或训练重构视为物理验证。

```bash
python -m tools.predictor_validation_20261027.validation --input MANIFEST.json --output NEW_REPORT.json
```

每个文件引用均为 `{"path":"相对 manifest 的路径或绝对路径","sha256":"文件原始字节 SHA256"}`。禁止覆盖输出。引用文件的 JSON 内容必须与其摘要匹配。

Manifest 结构：

```json
{
  "schema": "gsdeploy.independent_predictor_validation.v1",
  "model": {
    "artifact": {"path": "profiles.json", "sha256": "..."},
    "frozen_ns": 100,
    "training_sources": [
      {"run_id": "training-run-1", "artifact": {"path": "training.json", "sha256": "..."}}
    ]
  },
  "protocol": {
    "artifact": {"path": "protocol.json", "sha256": "..."},
    "frozen_ns": 100
  },
  "cases": [
    {
      "id": "validation-case-1",
      "run_id": "independent-validation-group-1",
      "split": "validation",
      "categories": ["service", "slowdown", "queue", "p95", "delta_U", "ranking"],
      "prediction": {"path": "sealed-prediction.json", "sha256": "..."},
      "observation": {"path": "observation.json", "sha256": "..."}
    }
  ]
}
```

`categories` 可省略，默认使用 protocol 的全部必需类别。不同实验可分别覆盖 service／slowdown 与 queue／p95／delta_U／ranking。`cases=[]` 明确输出 NOT_RUN。每个独立运行组只能出现一次，不能把同一运行拆成大量“独立”样本。训练来源不能与验证运行 ID 或原始文件摘要重叠。原生 segment profiles 的 `provenance.sources` 必须与 manifest 的训练文件路径集合完全一致。

Protocol 含固定的 `weights`（Q/T/L/P/M/E、和为 1）、`required_categories`、`utility_tie_tolerance`、`planned_cases` 与 `acceptance`。`planned_cases` 必须事先冻结所有验证 case，每项包含 `id`、`categories`、`expected_tasks`；含 delta_U／ranking 的 case 还须包含 `expected_candidate_ids`，至少两个。省略已计划 case、任务或候选会保持 INCOMPLETE，不能挑选成功实验得到 PASS。阈值必须先于验证观测冻结，不能看结果再调。示例规则结构为 `"delta_U_mae":{"operator":"<=","value":0.01}`。工具不替研究者编造可接受误差标准。

可检查的门槛有 service_mare、slowdown_mare、queue_mae、p95_mare、delta_U_mae、three_way_sign_accuracy、positive_gain_precision、false_positive_gain_count、pairwise_rank_accuracy、mean_selection_regret、max_selection_regret，以及 count、predicted_positive_count、rank_pair_count。可用后面几项声明最小验证数量，但不能把请求数当独立运行数。

Prediction 文件：

```json
{
  "schema": "gsdeploy.sealed_predictor_prediction.v1",
  "model_sha256": "模型原始文件摘要",
  "protocol_sha256": "协议原始文件摘要",
  "execution_context_sha256": "执行身份",
  "offered_cohort_sha256": "比较队列/负载身份",
  "created_ns": 201,
  "origin_ns": 200,
  "information_cutoff_ns": 200,
  "mode": "online_distribution",
  "actual_future_trace_used": false,
  "requests": [
    {"request_id":"r1","task":"text","pool_id":"public-input-1","service_ms":200,"wait_ms":100,"e2e_ms":300,"solo_ms":100}
  ],
  "p95_by_task": {"text":300},
  "queue_samples": [{"offset_ns":0,"depth":1}],
  "baseline_candidate_id": "no-spill",
  "candidates": [
    {"candidate_id":"no-spill","terms":{"Q":0.1,"T":0,"L":0.2,"P":0.3,"M":0.4,"E":0.5}},
    {"candidate_id":"spill","terms":{"Q":0.1,"T":0,"L":0.2,"P":0.3,"M":0.4,"E":0.5}}
  ]
}
```

Observation 对应结构：

```json
{
  "schema": "gsdeploy.predictor_validation_observation.v1",
  "evidence_kind": "measured",
  "run_id": "independent-validation-group-1",
  "execution_context_sha256": "执行身份",
  "offered_cohort_sha256": "比较队列/负载身份",
  "available_ns": 300,
  "requests": [
    {"request_id":"r1","task":"text","pool_id":"public-input-1","service_ms":250,"wait_ms":100,"e2e_ms":350,"solo_ms":100}
  ],
  "queue_samples": [{"offset_ns":0,"depth":1}],
  "candidates": [
    {"candidate_id":"no-spill","offered_cohort_sha256":"比较队列/负载身份","terms":{"Q":0.1,"T":0,"L":0.2,"P":0.3,"M":0.4,"E":0.5}},
    {"candidate_id":"spill","offered_cohort_sha256":"比较队列/负载身份","terms":{"Q":0.1,"T":0,"L":0.2,"P":0.3,"M":0.4,"E":0.5}}
  ]
}
```

以上数字仅解释格式。`measured` 是采集器声明，SHA 校验不替代硬件采集的真实性审计。CPU 示例必须使用 `cpu_fixture`。模型、协议冻结时间 ≤ prediction 创建时间 < observation 可用时间；输入截止时间不能晚于预测起点。所有时点须在同一可比较时钟域。

固定输入共驻 probe 可以逐请求比较。在线随机未来的请求身份与预测样本不同，不允许伪造逐请求匹配：用实际完整请求集合计算每任务 P95，并比较预测的任务分位数、预先声明时间网格上的队列深度与候选效用。队列 depth 必须双方使用同一口径（建议已到达但尚未完成，含执行者），offset 相对相同预测起点。

对于候选效用对照，`offered_cohort_sha256` 必须绑定相同的实际到达和输入集合；预测文件可在采集前绑定实验队列的 opaque commitment，但预测器不可读取随机在线未来。其训练输入只可包含已知分布和可见前缀。已知规划时间表必须使用 `mode=declared_plan` 并显式设置 `schedule_declared_before_origin=true`。不同资源候选可分别携带 `execution_context_sha256`，预测与实测必须逐候选匹配。效用从全部六项重算；任意缺项不补零。`baseline_candidate_id` 必须是声明的禁分流基线。只有完成禁分流候选优化后，才能把该比较称为“对最优禁分流”的增益。

`mode=trace_replay` 可用于解释给定真实并发轨迹下的服务误差，但不会通过在线物理验证。CPU fixture 同理。`requests`、`queue_samples` 和 `candidates` 可以留空或预测值使用 null，覆盖率保留这些 UNKNOWN；不能挑选成功样本后省略失败请求。

报告包含服务、等待、E2E、减速比、队列深度、完整每任务 P95 的误差；paired ΔU 的 MAE／偏差／正、零、负三类正确率／正收益精确率；候选两两排序正确率与选择后悔值。未覆盖必需任务、请求或候选时为 INCOMPLETE；未设置验收门槛时为 UNASSESSED；只有完整物理在线覆盖且冻结门槛全部通过才为 PASS。PASS 只表示本次独立验证协议通过，不表示物理最优、置信界已校准或最终分流一定正收益。
