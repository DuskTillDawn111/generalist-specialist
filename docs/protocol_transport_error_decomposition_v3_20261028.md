# v3 误差拆解与后续瓶颈

完整分母的训练诊断表明，协议通知可组合后，主要误差仍在响应模型和 work transport。三个整 training-capture 留出、600 个 offered 请求的 reader E2E MAE 为 160.04ms。所有请求均计入；没有按成功子集删去 FIFO 冲突。

新增只读脚本 `protocol_transport_error_decomposition_v3.py` 重新执行相同固定场景，不改 β、bandwidth、候选选择或冻结源。每个请求验证精确加法：E2E 误差等于 START 相对 arrival 的误差、START→RESULT occurrence 响应误差、RESULT occurrence→reader 通知误差之和。公共 parent dispatch 与 handoff 尚不能分别归因。

|owner|reader E2E MAE (ms)|START／arrival 项 MAE (ms)|实际响应项 MAE (ms)|RESULT 通知项 MAE (ms)|
|---|---:|---:|---:|---:|
|generalist_2#0|213.43|0.34|213.38|0.10|
|ocr#0|244.95|0.62|244.40|0.05|
|text#0|115.46|0.33|115.35|0.13|
|text#1|110.22|0.47|110.11|0.16|

这些是各项绝对误差的均值，表内 MAE 不相加；逐请求有符号误差满足精确恒等式。宏级主要错误为 OCR native_ocr 244.40ms、G2 prefill 162.57ms、text#1 prefill 120.55ms、text#0 prefill 116.03ms、G2 最后 decode 99.38ms。结果提示下一步重点应是 reference work/residual 支持、当前 processed peer covariate 的解释能力，以及把 observed-path 方程迁移到新生成路径的假设；单独降低通知延迟不能修复这些响应误差。

宏表同时保留 donor wall 与 target wall 之差、generated wall 与 donor wall 之差、donor reference work 与 target 在同 fold β 下的 reference work 之差。它们是描述性分解，不能称为共驻干扰的因果贡献。query 表按 owner、来源 phase、false/true 分开记录 occurrence 相对 START、reader delay、processing delay；通知表按 owner/action 记录相同延迟误差。

source exact reconstruction 和 forward 初始化必须分开解释：前者使用整个 measured capture 的原始消息时钟验证递推相容性；后者把每个原 offered job 作为场景，从显式空站启动。每个场景的源 capture offset 和此前已发生的 measured 消息数已保留，但控制、priming 和 parent 背景事件仍未知。因此空站不是已知真实 source initial state，源时钟可以重建也不等于该初始化已被认证。donor work 原来已经受其 source processed peer path 影响，transport 后应接受新路径检验，不能把 source 的良好重构误报为 forward 准确。

最终拆解产物在 `artifacts/predictor-protocol-transport-error-decomposition-v3-20261028-cpu02/`。cpu01 被按实际 draw ordinal 补强 donor provenance 后的 cpu02 替代，旧产物保留。没有 GPU、独立 validation 标签、主预测资格或 ΔU 结论。
