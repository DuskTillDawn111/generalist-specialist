# Source-bound decoder 结构特征

`tools/predictor_validation_20261027/static_language_work_v1.py` 是独立 CPU 工具，不被冻结的 v6 collector/supervisor 导入。它将原先全部为空的结构字段推进到有配置、源码及本地 safetensors header 支撑的**名义七投影工作量**，不加载模型或权重 payload，不修改原 raw，不读取 validation labels。

实际 `worker-registry.json`、worker spec、checkpoint path、execution dtype/quantization 和 adapter manifest 必须一致。每个 decoder layer 的 q/k/v/o、gate/up/down 均按 config 的 H、FFN、attention/KV heads、head_dim 推出 `in/out`，再逐层核对 header。text 检查 32×7 个 F16 checkpoint 矩阵和 448 个 F32 LoRA A/B；G2 检查 36×7 组 GPTQ qweight/qzeros/scales/g_idx header，保留 input bit packing、group128 的维度轴，不能当作 dense `W[out,in]`。header SHA＋前后 stat 记录文件身份，本轮没有重哈希约 24GB 模型 payload。

实际声明源码：text 使用安装在 compact-standard 的 LlamaAttention/LlamaMLP；G2 的本地 MiniCPMO 明确构造 Qwen3ForCausalLM，安装 Qwen3 实现在 compact-dots；GPTQ ExLlama v1 实现在 compact-minicpm。工具检查七个 `nn.Linear` constructor 的维度表达式并记录 SHA。历史 worker 没有直接记录安装模块源码身份，因此该结果是现在可审查的声明拓扑证据，不冒充过去运行时的 affine 实测。

对实际 language BEGIN 的 B×S token rows=R，单个声明投影 `(in,out)`：

```
nominal dense GEMM FLOPs = 2 * R * in * out
dense-equivalent parameter bytes = in * out * activation_dtype_bytes
logical affine input/output operand bytes = R * (in + out) * activation_dtype_bytes
```

结果按全部层的七投影加总。所谓 logical operand bytes 是每个 affine 的操作数尺寸之和：会重复计算共享输入，既不是独占 tensor storage，也不是峰值、额外 GPU memory、DRAM 流量或 cache 命中。base 的 2-byte dense-equivalent 参数是假想稠密表示，不是 text INT8 或 G2 GPTQ4 的加载占用。

text LoRA 单列 `2 * R * rank * (in+out)` 两次 GEMM 的名义 FLOPs、448 个 F32 checkpoint 张量的逻辑字节及 F32-equivalent 操作数尺寸。它不 merge 到 base，也不把 adapter 当 BF16。实际 autocast/运行 dtype、转换、scale/add 的成本未测，保持 Unknown。

attention 乘积、真实 KV 长度/缓存字节、norm/RoPE、非线性、embedding、LM head、vision/audio 及量化/backend 开销均不在覆盖范围。`past_key_values_present=True` 不构成 KV 长度证据。runtime CB/SCB/GPTQ scales、物理 cache、DRAM、SM 服务和匹配量化精度的硬件峰值没有建立，分类明确保持 `unknown`。

`features_from_visible_begin()` 仅接受 parent 已完成 processing 的真实 semantic BEGIN，并读取该 BEGIN 中的实际 B/S；END、未来 BEGIN、缺失 shape 和时序错误均拒绝。`training_begin_features()` 只接受完整 training review，排除 priming、保留 OCR unsupported 行，不读取 worker response 输出或耗时来构造特征。每个 forward 的特征只能在该 BEGIN 后使用，不能让未来 decode shape 泄漏到早期路由。训练获得的 shape 模板没有自动获得 prospective 资格。

真实 CPU 产物位于 `artifacts/predictor-static-language-work-v1-20261028-cpu01/`：两个 binding、六份 BEGIN 特征（每 capture 50 measured，46 language 请求共 230 forward，4 OCR 保留 Unknown）、10 个 owner/phase/shape 条件及全部 45 对候选。优先序只使用 base dense-equivalent FLOPs/(dense-equivalent weights＋logical operands) 的相对 log contrast；没有裁剪任何候选，包括 Unknown 和上游需另判执行可行性的同 owner 对。prefill/decode 的对比来自实际 rows 差异，不能称为已测计算型/显存型或承诺降低干扰。

16 项 CPU 测试覆盖 GQA/head_dim、手算 FLOPs/bytes、LoRA 独立计算、全层 header/packing、缺失/越界 header、禁止 payload 读取、来源维度错误、binding 篡改、BEGIN 因果性、validation 拒绝及候选完整保留。下一步由 capture 留出验证这些结构量能否改善真实响应预测；仍须通过独立全过程排队/P95/ΔU 正负和候选排序 gate。
