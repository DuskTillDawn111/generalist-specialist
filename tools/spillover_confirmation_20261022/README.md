# 新负载独立确认工具

该工具与旧 `gsdeploy.confirmation` 的固定种子/负载协议分开。新增模块 `gsdeploy/spillover_confirmation.py`，CLI 为 `scripts/spillover_confirmation.py`；未改变旧确认模块或任何冻结快照。

prepare 只接受 `replay_transfer_v1` 的 `validated_transfer_spec` 和 `check_scoring` 全部通过的配置。它从合同读取 workload、时长、warmup、development/validation/confirmation seeds，并要求与 transfer 的种子用途一致且互不重叠。没有默认复用 42 或 43–45。本轮合同应固定 confirmation `[71,72,73]`、horizon 1800；协议本身支持其他预声明合同。

合同示例（workload 必须与已验证 config 完全一致）：

```json
{
  "schema": "gsdeploy.spillover_confirmation_contract.v1",
  "study_id": "asr-heavy-2.6-spillover-v1",
  "workload": {"text": 0.05, "image": 0.05, "ocr": 0.05, "asr": 2.6},
  "horizon_seconds": 1800,
  "warmup_seconds": 30,
  "seed_roles": {"development": [51], "validation": [61,62], "confirmation": [71,72,73]},
  "training_forbidden": true,
  "purpose": "spillover_independent_confirmation_not_training",
  "target_task": "asr",
  "winner_label": "free_search_winner",
  "required_roles": ["free_search_winner", "best_global_no_split", "best_same_deployment_no_split", "strongest_alternative_deployment", "best_one_specialist_max"]
}
```

候选文件为 `gsdeploy.spillover_confirmation_candidates.v1`，含 `config_sha256=quality_digest(config.as_dict())`、`study_contract_sha256=quality_digest(contract)`、按 required_roles 顺序排列的 `candidates`。每行包含 `label`、`plan`（deployment/routing）、`status`、`optimality_certified`，可保留额外搜索元数据。无可行计划时 `plan=null`，明确记录 INFEASIBLE/UNKNOWN/BUDGET/LIMIT 等原状态；不会凭空构造可测控制。one specialist max 的定义是**目标任务的专用家族副本总数不超过 1**，其他任务的专用模型不受该项限制。

```json
{"schema":"gsdeploy.spillover_confirmation_candidates.v1",
 "config_sha256":"...", "study_contract_sha256":"...",
 "candidates":[
   {"label":"free_search_winner", "plan":{"deployment":{"asr":2,"generalist_2":1,"ocr":1},"routing":{"asr":{"asr":1.7333333333333334,"generalist_2":0.8666666666666667},"text":{"generalist_2":0.05},"image":{"generalist_2":0.05},"ocr":{"ocr":0.05}}},"status":"OPTIMAL_FINITE_DOMAIN","optimality_certified":true},
   {"label":"best_global_no_split","plan":null,"status":"INFEASIBLE","optimality_certified":true},
   {"label":"best_same_deployment_no_split","plan":null,"status":"INFEASIBLE","optimality_certified":true},
   {"label":"strongest_alternative_deployment","plan":null,"status":"UNKNOWN","optimality_certified":false},
   {"label":"best_one_specialist_max","plan":null,"status":"INFEASIBLE","optimality_certified":true}
 ]}
```

示例只解释结构，不能替代真实搜索。准备时再次校验所有实际计划的完整约束、禁分流/固定部署/其他部署/单专用副本限制；不在此重新搜索或认证调用者的最优声明。相同计划按 SHA 去重，保留全部角色，按合同种子×唯一物理计划建立 jobs。

准备与校验：

```bash
CUDA_VISIBLE_DEVICES='' python scripts/spillover_confirmation.py prepare \
  --source-packet SOURCE_PACKET --config VALIDATED_TRANSFER_CONFIG \
  --candidates CANDIDATES_JSON --contract CONTRACT_JSON \
  --runtime-root FROZEN_V5_ROOT --checkpoint-report ABSOLUTE_CHECKPOINT_REPORT \
  --output NEW_CONFIRMATION_PACKET

CUDA_VISIBLE_DEVICES='' python scripts/spillover_confirmation.py validate --packet NEW_CONFIRMATION_PACKET
```

正式 run 必须显式执行；此说明没有启动 GPU：

```bash
python scripts/spillover_confirmation.py run --packet PACKET --job JOB_ID \
  --output NEW_JOB_DIRECTORY --execute

CUDA_VISIBLE_DEVICES='' python scripts/spillover_confirmation.py report \
  --packet PACKET --runs JOB1/run.json JOB2/run.json ... --output NEW_REPORT_JSON
```

运行包装器加载显式冻结 runtime 的 `static_study.execute_job`，保持 `phase=confirmation` 和合同自定义 purpose。v5 不识别新 transfer 模式，故包装器先构造仅用于收集的 `task_replica_proxy/standalone_proxy` config，并重算低层 packet 的 source_problem provenance；原 job identity、原确认 packet seal、原 transfer config 及所有权重保持绑定，不修改冻结执行器。执行身份额外绑定 purpose、study contract SHA、完整 scoring config SHA 和驻留检查点 SHA。该转换不能用于验证或搜索。

冻结执行器原始 `run.json` 保持写入一次。包装器另写 `spillover-collection-receipt.json`，绑定原 config、原模式、收集模式、原包 seal、runtime root、自定义用途及 raw-run SHA。报告拒绝缺少或错误的 sidecar。原生导入逐组检查硬件、执行身份、计划、transport、seed/horizon/warmup/drain、priming/cohort 和唯一允许的配置差；OCR 的 `GS_CHECKPOINT_FILES_JSON` 作为运行时派生环境字段处理，完整检查点 hash 仍必须一致。

报告直接 `evaluate_run` 重算真实质量、能量和六项效用；缺失/失败/不完整 U 拒绝入报，绝不补零。完整性按合同唯一物理方案×种子计算，不固定 15 组。部分报告保留缺失块，不生成未完成配对的区间/p。配对数学复用现有 `paired_comparisons`，独立单位仍是 seed，范围仅限实际声明候选；不宣称物理全组合最优。

准备把 source 输入、原 config、候选/合同文件、检查点 receipts、冻结执行源码、新包装器及调用侧源码放入 bindings。后续源码变更会使已准备包失效，正式 prepare 应在源码稳定后执行。所有包/报告/sidecar 输出拒绝覆盖。

CPU 验证：

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_spillover_confirmation -v
```

包含去重/新种子/用途、未验证 transfer 拒绝、种子重叠拒绝、身份与配置篡改拒绝、不完整 U、部分/全量报告，以及**实际未修改 frozen v5 execute_job** 入口的 CPU 模拟；后者仅替换检查点 I/O、GPU UUID 查询和 transport 推理。正向确认 fixture 对 transfer gate 和测量算术使用明确 mock，属于 CPU 协议测试，不是新负载验证或 GPU 证据。
