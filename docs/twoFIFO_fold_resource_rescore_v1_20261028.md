# twoFIFO fold resource rescore companion v1

该 CPU companion 补充冻结 generated runner 的资源评分连接，不修改 collector、v5_2 engine、generated runner 或 training completion coordinator。入口只接受已完成的十二次训练报告，恢复一个既有 heldout case 的 full generated worlds，再调用原 `actual_worker_world_score_v3.score_world` 和原 `whole_score`。它不重新拟合响应模型，不读 case 的 labels.json，不启动 GPU、控制器或验证采集。

## 实际评分路径

`rescore_ensemble` 实例化真实 `FoldResourceTrainingProvider`，由其重新认证完整 native12 与三折资源模型。只有目标 block 排除后的两个 block、八次 donor 的 selected model 供值；target attempt 必须是 heldout 四次之一，每个响应 draw 的 block 必须属于这两个 donor block。评分居民/family roster 和 provider 的物理执行域必须一致。CPU 测试只替换 `resource.load_complete` 原生输入边界，真实 `fit_tables`、fold provider、world timeline 与 scorer 都执行。

每个 GENERATED_WORLD 保留整个原 world 与 query fingerprint，原 `score_world` 构造从 forecast origin 到 fixed drain 的完整 idle/active timeline。E 仅积分原测量窗重叠，原 scorer 将 donor startup joules 按 3600 秒加一次。M 包括 startup、显式过去累计 peak 和原 drain 内所有 slice 的 owned sampled peak，由原 scorer 应用一次安全因子。它不是 allocator 瞬时绝对上界。

Q/T/L/P/M/E 和 full-U 从完整 draw 集合重算。无资源支持为 UNKNOWN；transport UNKNOWN 保留原 reason、所有 request IDs、权重和分母。未知质量、资源或完成项不会产生 partial-U，UNKNOWN 权重不重归一化。quality 沿用冻结训练期望，既不冒充本次 native quality，也不宣称资源准确性或最终收益通过验证。

## 显式过去内存状态

缺少状态时 M 和 full-U 保持 UNKNOWN；绝不推断 fresh-after-priming。可选参数为有哈希 receipt 的 closed JSON：

```json
{
  "schema": "gsdeploy.twoFIFO_fold_resource_rescore.v1.causal_past_memory_state",
  "cutoff_ns": 0,
  "forecast_origin_ns": 0,
  "basis": "immutable_pre_case_history_only",
  "future_labels_used": false,
  "state": {
    "basis": "explicit_prior_attempt_history",
    "ordered_state_history_identity": "64-character history identity",
    "last_state_key": "64-character state identity",
    "prior_owned_sampled_peak_mib": 0
  },
  "sources": [{"path": "/absolute/past-evidence.json", "sha256": "receipt digest"}]
}
```

要求 cutoff≤forecast origin，forecast origin 精确等于原评分起点，hash 长度/数值非负有限，来源 receipt 不漂移。这里绑定声明与字节；历史状态的物理推导仍是调用方的显式假设，不是独立因果认证。来源必须是真正的 pre-case history，禁止把 heldout future M/E 填为该状态。结果显式记录此限制，保持所有 qualification 标志为 false。

## 归档 continuation

冻结 runner 只保存 compact forecast 与完整 forecast 的 SHA256。`restore_full_ensemble` 使用原 public calendar、保存的 settled seed、原 folded law 和固定八 draw 方案重新生成，逐 draw 验证 identity/block/weight/status/cohort 与 full forecast SHA，再要求重建 compact envelope 全等。缺 seed、root fit unqualified 或 transport 不可识别均保留原 UNKNOWN。它不选择新 latent、不缩小 cohort、不延长 drain。

`run_completed_case` 校验完成状态、原设计/source receipts、campaign 的精确 donor8/heldout4、目标 attempt/case，以及 cohort 相对计划起点的精确 arrivals 和固定 10s measurement + 60s drain。通过后才写 exclusive 新输出目录；标签提取与数值拟合均不调用。只对一个既有 case 显式执行，不接入冻结 coordinator，也不自动扩展到整个 campaign。

资源模型 campaign receipt 必须与 generated design 的 campaign receipt 完全相同，不能只匹配 IDs 或物理域。十二个资源 training-review receipts 必须各自在原 completed report.sources 中且没有字节漂移，避免同 campaign 混用另一 native run。非空 folded law 还须匹配 completed report 中记录的内容 digest。独立审阅先复现了两项跨来源混用缺口，修复后的作者 cpu02 与独审报告保留其负测证据；cpu01 为修复前历史，不覆盖。

```bash
python -m tools.predictor_validation_20261027.twoFIFO_fold_resource_rescore_v1 \
  --completed-report /absolute/completed-generated-report.json \
  --resource-model /absolute/completed-resource-model.json \
  --heldout-block training-block-3 \
  --attempt-id training-block-3-shard-1 --case-id ORIGINAL_CASE_ID \
  --output /absolute/new-exclusive-output
```

没有 `--past-memory-state` 时 M 未知是预期行为。完整 native12 gate 未完成前不得用本 CLI 做实测 continuation。

## 未来 caseM 协议：本版禁用

与 caseM 作者约定：当前 v3 是隔离候选，没有 scorer 可接受的认证 receipt。本版 `case_memory_candidate_merge_enabled=false`，不提供 float 注入 API，不替换 `provider.memory(active)`。

未来独立 merge 若获实现授权，必须先固定 closed schema/version，并验证 physical_execution_identity、heldout block、精确 donor8/model/source digests、public query/calendar fingerprint、forecast origin/cutoff 及 provenance、explicit memory origin/P0/shape provenance、support status/reasons、implementation 与独立 review receipts。唯一可评分量是未加安全因子的绝对 startup-through-current-fixed-drain owned sampled peak 或显式 UNKNOWN。通过审查后只交原 `whole_score`，不得用 active-state memory API 冒充 case-level 模型。

## CPU 验证

`tests/test_twoFIFO_fold_resource_rescore_v1.py` 包含 11 项作者测试：真实 provider/scorer、heldout future poison 不变、测量 E 与 startup3600/安全因子单次应用、drain peak、缺 past UNKNOWN、partial support/transport UNKNOWN 分母、物理域/fold/cohort/drain/score/compact 拒绝、过去状态未来/漂移/非法值拒绝、全 UNKNOWN 合同、compact fullworld 哈希/原 envelope 重放，以及 completed-case 精确绑定且不提标签不拟合。纯 CPU 合成资源表只证明连接与约束，不证明 native 准确性。
