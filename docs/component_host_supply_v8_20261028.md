# 按原执行组件固定选择 host predictor

`component_host_supply_v8.py` 将 host provider 的选择写入独立、带 receipt 的完整 component mapping；`forecast_gate_multi_native_v7.py` 使用该 mapping。原 v5/v6 文件、历史模型、sealed03 和失败评估均不修改。

目前精确支持域是：`original_text_ocr` 的 8 个已知输入/owner template 使用修后 training02 的 exact-input host v7；`original_image_asr` 的 4 个 image/ASR template 使用其独立、原生 context 下的原 v5 host model。image/ASR 不是 v7 查找失败后的回退路径。provider 在模拟开始前按 component 固定，任何 UNKNOWN 都继续拒绝，不另选模型。完整 component 集合必须恰好出现一次；缺失、重复、外来 component、错绑 public binding、替换原模型、v7 support 域不完整均拒绝。

selection 的 composite receipt 必须与主 bundle 的 composite receipt 完全相同，再校验其 execution id。v7 的 source-host/DAG/light/natural identities 必须等于该原组件，provider 重新推导训练参数和公开输入绑定。原 image/ASR provider 由 `MultiNativeProvider` 从原 captures/context/DAG/SQLite 已重建验证，不改写为 text/OCR context。每个输出请求另外保留 component id、provider kind、模型 receipt、两个原 context id，以及 `selection_changed_after_unknown=False`。

新版 gate 拒绝旧的全局 `host_supply` override。duration、generation、lifecycle、routing、resource、startup/priming 和六项等权评分规则沿用独立版本。评分先按 task 汇总质量分子/分母，L/P 包含排队和完整生命周期；E 加一次按 3600s 摊销的 startup/priming，M 对启动和完整 active/drain 的最大峰值施加一次安全因子。不会因为 host 条件分层而重调六项权重。

回归覆盖固定组件派发、UNKNOWN 不回退、错误/缺失/重复组件、错误模型/公开绑定、supported 域不足、全局 override 拒绝，以及重签伪 composite 并保留旧 id 的拒绝。30 项 host/generation/startup 相关测试通过。

实际 CPU 全流水线入口使用原四任务 12-template 公共输入和原 startup/generation 证据，不拟合新参数、不读取 validation labels、不运行 GPU。第一次 attempt01 在 134s 主动终止，保留其完整 source-stable process receipt，因为自审发现并修复 composite receipt 相等性检查；修后独立 final02 才是完整检查目标。

该工作验证执行接线和数值闭合，不构成未知输入泛化、独立准确率或部署收益。两种 host supply scenario 仍只是不能由 natural endpoint 唯一识别的条件分解；此次沿用原 bundle 已选的 `host_issue_depth_supply`，未按新结果选择 scenario。kernel 数值 hypothesis 的 strict 残差/支持拒绝保持完整，旧 pilot 正式 FAIL 不被替换。

修后实际结果位于 `artifacts/predictor-component-host-fourtask-cpu-diagnostic-v8-20261028-final02/`：260.17s 完成 12 请求、354217 DAG nodes、441943 events；369 个生产源在全过程 byte-identical，旧 361 源也逐 receipt 未变。独立核对通过 8:4 provider 派发、原 generation/startup/context 绑定、全部六项和 event 资源积分。训练诊断 U 为 -1.5452280656643664（旧诊断 -1.5433091371430834）；该变化是预测值变化，不是实测收益或准确率改进。严格 kernel/copy 拒绝继续完整保留。
