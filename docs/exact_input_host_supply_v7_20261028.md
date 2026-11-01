# 精确公开输入条件下的 host supply v7

v7 不修改旧 v5/v6 实现或 sealed03。旧失败 pilot 的正式结果仍是 FAIL；新的训练内检查不替换、追认或重评分该结果。

`dag_host_release_v7.py` 从原 v5 模型绑定的两个真实 isolated raw capture 重新读取严格审查、原 DAG、SQLite、公开内容及实际生成工作。参数按 `(alias, task, pool_id, public binding, DAG structure)` 精确分组，每组至少两个不同 raw SHA；同一个 raw 内增加请求不增加独立支持。参数先在每个 capture 内求均值，再等权合并 capture。两 text 副本也不互相补支持。

公开绑定保留原 light/natural context ID、同源 native projection、immutable content hash、tokenizer input_tokens、协议及 effective_generation、完整固定工作场景、DAG receipt 和 source SQLite receipt。OCR 明确保留非自回归类别，不填充不存在的 token/协议字段。provider 接受另经执行组件验证的 `public_bindings` 并逐字段比较。未知内容/生成配置/模板不回退到 family 均值。

当前只有两个 unique content，无法据此证明连续 token/shape 回归泛化。v7 使用精确输入分层，token/shape 是绑定条件；没有拟合或宣称未知输入的 shape 模型。八个现有 owner/input 组各有两个独立 capture，但这是 exact-known 支持，不是跨输入泛化。

两种 release scenario 保留：`front_loaded_supply`、`host_issue_depth_supply`。它们由 natural final-sync/同步 OCR call-return 上界与 light DAG critical path 形成条件可行分解。OCR call-return 是完成上界，不是精确 GPU 完成时刻。host submission、profiler transfer、未建模 GPU 序列化与资源争用仍不能由终点唯一识别。拟合终点的恒等性不能当预测精度；本模型不是物理 kernel 共驻干扰模型，不修改 kernel strict fit gate 或 numeric hypothesis 标记。

`dag_host_diagnostic_v7.py` 直接运行真实 v6 DAG 引擎及相同数值 duration/queue diagnostics 缓存。CPU 入口比较旧 family/task pooled 与新 exact-input，两 scenario 都完整保留。原 isolated captures 仅作拟合终点诊断；原 SSG 的两个 distinct natural captures 不拟合 host 参数，可作 host component 检查。其他 kernel/work 组件可能已使用 SSG light evidence，因此它不构成全流水线独立验证。每个 SSG job 保留原相对启动 offset、公开内容/固定工作核对、实际 peer overlap 和 request 级 service 误差，不把混合时段合成 solo 标签。

该入口只检查 handler service，不评估 post-handler lifecycle、完整 E2E 或六项 utility。数值不可得保持 UNKNOWN；strict residual/support 拒绝继续独立计数。真正 whole-capture host holdout 在仅两次 isolated capture 时无法保留最少两次训练支持，仍需第三次采集。SS 纯双专用、short/long 混合分配和 replicated SSG light 缺口保持未完成。

第一次诊断启动被 resolver 正确拒绝：five-bridge semantic fit 与原 two-capture provisional fit 的绑定 receipt 不同。在读取训练终点前退出，失败保留为 `dag-exact-input-host-v7-20261028-training-diagnostic01/failed-start.json`。后续 CPU 检查显式使用与 provisional 精确绑定的原 semantic fit；未绕过来源检查或改写原 fit。

初版 review 发现 provider 仅比较外层 binding/structure，未强制重推拟合均值。修后 `verify_model` 在构造时重新派发原 strict loader，从 source_host、两个实际 raw capture、context、DAG/SQLite 重建全部 observations、支持计数、public bindings 和均值，并与整个 model core 比较。原 v5 index 与新 v7 index 均逐内容重算，nested source/producer receipt 重新核验。刷新外层 model receipt 不能使伪造 mean、capture IDs 或 training index 获准。初版 training01/diagnostic02 保留为历史；修后新模型为 training02，实际诊断使用独立 diagnostic03，带完整生产源和模型证据 before/after 复核。
