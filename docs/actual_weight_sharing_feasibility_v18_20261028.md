# 实际 text/G2 权重共享可行性核查

本轮 CPU 核查尚未发现可认证的 text/G2 公共大线性子矩阵，不能把已有 NumPy 原型称为 Hugging Face 共执行后端。当前可以确定的完整权重共享机会是两个相同 text 副本；新的 shared_text 后端已经把它们放在同一权重 owner 内进行 HF batch。没有启动 GPU、下载模型或改变正在运行的 v16 依赖。

## 当前实际运行的模型，而非旧配置名

证据来自 `artifacts/static-study-v6-20261021/inputs/worker-registry.json`、实际 worker.json、它们声明的冻结 `source-qa` 加载代码及本地 safetensors header。不能用仓库旧 `compact_execution_memory_native.json` 中的 UnifiedQA/MiniCPM 版本代替当前部署。

| 项目 | 专用 text | G2 |
|---|---|---|
| checkpoint | nvidia/Llama3-ChatQA-1.5-8B，revision 3b98162e… | openbmb/MiniCPM-o-4_5-GPTQ，revision 02b54c54… |
| 语言实现 | LlamaForCausalLM | 本地 modeling_minicpmo.py 明确构造 Qwen3ForCausalLM |
| decoder / hidden / FFN | 32 / 4096 / 14336 | 36 / 4096 / 12288 |
| vocabulary | 128256 | 151748 |
| attention / KV heads | 32 / 8 | 32 / 8，另有 q_norm、k_norm |
| 执行表示 | bitsandbytes INT8，BF16 activation，threshold=0，lm_head跳过量化 | 官方 GPTQ4，group128，desc_act=false，ExLlama v1，FP16 activation |
| 适配器 | ChatQA专属 LoRA，r16、alpha32，七种投影，bias=none、无DoRA | 当前执行没有这个 ChatQA adapter |
| 本地 tensors | 323（291 F16、32 F32） | 2170（1414 F16、756 I32） |

所以“专用只是 G2 同一个基座上加 adapter”在当前部署中不成立。text 的 adapter 的确依赖 ChatQA 基座，但这个基座与 G2 的 Qwen3 不是同一组张量。维度4096相同也不说明训练后权重相同。两模型的 tokenization、RoPE参数、norm epsilon、FFN宽度及投影执行语义也不同。

ChatQA adapter 本地有448个F32张量，即32层×7投影×A/B。`PeftModel.from_pretrained` 保持 base layer 与 LoRA 路径分离并设为 eval；不得把它误认为 adapter 已经 merge 到 dense/GPTQ 权重。text#0/#1 同时绑定相同基座 revision、相同 adapter SHA 和执行策略，现有 shared_text 是此处直接可用的完整共享机会。

## 实际读了哪些字节，能证明什么

可复跑的只读审计脚本与输出在 `artifacts/weight-sharing-feasibility-v18-20261028-cpu01/`：

- 所有本地 safetensors header、tensor shape/dtype/data offsets，以及配置和加载源码的 SHA；大模型权重文件保留 header SHA 与前后 stat，**没有重新哈希整份约24GB权重**，不把历史下载 SHA 说成本次重新验证。
- text 的65条4096维 norm 与 G2 的73条对应类型完整小向量，逐tensor比较哈希，没有完整字节相同的跨模型向量。norm也不是大线性GEMM。
- 两模型 embedding、lm_head 的左上16×64 F16矩形，各1024元素字节比较：均无相同标量比特值。这只否定这两个特定样本矩形相同，不能证明所有偏移、所有层都不存在公共子矩阵。
- G2 的 q/k/v `g_idx` 在同层相同，它只是量化分组索引，不能当成共同 GEMM 的权重；`qzeros` 中的重复打包常数也必须结合 scales 和 unpack 规则解释。

没有进行穷举最大公共矩形搜索，没有读取 GPU 上加载后的 INT8 CB/SCB，没有测量 L2、SM、DRAM 或收益。因此正确结果是“尚无已认证、可用于当前跨模型共执行的大线性块”，不是“数学上证明不存在任何公共矩形”。

## 最小正确的层适配

PyTorch Linear 存储是 `W[out,in]`，计算 `Y=X@W.T+b`；现有 shared_linear.py 的 NumPy 原型采用 `W[in,out]`。任何适配必须明确转置与矩形轴，不能直接复制其坐标。

对分支 i，经逐字节或严格量化语义认证的公共矩形为 `W_i[O_i,I_i]=C`：

```
Z = concat_rows(X_0[:, I_0], X_1[:, I_1]) @ C.T
Y_i[:, O_i] += split_rows(Z, i)
Y_i += 对 W_i 公共矩形补集的互不重叠残余计算
Y_i += b_i                  # 每个分支恰好一次
Y_i += branch_specific_LoRA  # 若有，保持原转换和加法顺序
```

补集可按输出轴上/下及中部输入轴左/右四个不相交矩形划分。不得再执行原完整 Linear 后叠加公共结果，那会重复计算。所有输出列都必须有原始贡献；矩形外输出与输入不能丢失。bias不同不阻止共享乘积，但不能提前把一个分支bias加到公共Z。activation、SiLU gate×up、RMSNorm、attention 与 residual连接继续由各自模型执行，不因一个 GEMM 合并就跨模型复用。

若是“同基座＋不同 LoRA”，可共享基座 `XW0.T` 后分别计算 `scaling * (X A_i.T) B_i.T`；这不等价于有效权重 `W0+ΔW_i` 存在相同矩形的证明。当前两个 text 的 adapter也相同，直接整模型batch更简单。真实PEFT的F32 adapter计算、result dtype转换、eval dropout关闭及有序累加必须保留；不得静默merge、降精度或用浮点容差冒充字节一致。

输入允许不同 token 数，只在 ready 的本层 token rows 上合并并保留 request/token位置映射。预填充与解码不能为了配对强迫全模型同步到同一层。deadline/最大配批等待/暂存activation内存必须进入同一全过程队列预测。若内核需要alignment padding，只能增加数学上无贡献的零行/零列并在scatter时去除；保留完整原输入、attention mask、position和每请求KV，禁止截断。延迟收益需要覆盖 gather、concat、残余多次GEMM、scatter、等待与新workspace。

## 量化为什么不能只比较 qweight

GPTQ权重的有效值由 `qweight + qzeros + scales + g_idx + bits/group/packing/kernel规则` 联合定义，不能把I32打包值与F16参数直接比较。对当前GPTQ4，输入维的分组与bitpack边界、输出维qzeros的打包边界，以及ExLlama预处理/重排都影响可切片范围；任意矩形直接切qweight往往不是独立的合法量化层。相同g_idx或相同qzeros不意味着相同有效权重。

bnb INT8使用加载后的CB/SCB和activation量化scale。即使原F16矩形相同，完整权重行的scale不同也可能使INT8块不同。把输入列切成公共/残余块后重新各自量化activation，会改变原来基于完整输入行的scale；重算weight子块scale同样改变模型。正确适配必须使用原始完整行的scale、保持outlier策略，并在适当位置合并整数累加再按原语义dequant/bias/转换。当前threshold=0关闭其outlier分支，仍不消除scale和舍入边界要求。

现有text的bnb INT8/BF16与G2的GPTQ4/FP16没有可直接共用的量化算子合同。先dequant成另一种dtype再共算会形成新的执行精度/模型配置，须另建身份并重新验质量、时延、M/E；不能继承旧画像。因此本轮没有为当前跨模型路径写一个看似可运行、实际改变语义的adapter。

## 权重驻留、硬件cache和结果cache分别证明

1. 单份共同权重存储＋batched GEMM可减少重复权重驻留，可能改善硬件cache/权重带宽；对不同X，总MAC数通常不减少。当前 shared_text 已完成的是同权重owner与完整HF batch，其真实GPU效果仍待新画像。
2. 相同W但不同X不能复用乘积结果。计算结果cache必须同时认证精确input bytes、有效base/adapter、dtype、量化metadata、算子语义与shape/布局；模型state/position/attention/KV语义相关部分也要纳入身份。不能因为共享prefix文本，就假设跨模型hidden state相同。
3. KV/prefix cache是独立机制，要求模型、adapter、tokenizer、token IDs、position/rope、attention mask与缓存层状态完全兼容。ChatQA与Qwen3的KV不得互用。缓存占用、失效、生命周期和命中等待要重新画像。
4. 权重只存一份或一轮共batch，并不能证明GPU L2命中增加；只有实测硬件counter才支持此表述。本轮均UNKNOWN。

## 可执行的下一步

先利用已经实现的同text整模型batch，在既定预测gate后采新三owner画像。对跨text/G2，先以只读离线模块生成逐候选层的完整块身份证据（逻辑矩形、所有量化参数、packing/group边界和kernel合同）；没有块就不启用共享。若找到可观面积且共同算子兼容的真实块，再写独立Torch/HF层适配，CPU用原层参考验证残余、bias、LoRA、ragged rows、padding、cache失效和拒绝不兼容量化，之后再做独立GPU数值与性能核验。当前checkpoint差异较大，不能预先承诺这里一定有有用公共块。

若要构造同基座专用/通用模型来系统利用USHER共享，这是改变候选模型池的研究分支，需要训练/质量准入与新的执行身份；不应作为现有ChatQA/G2“本来就是同基座”的实现事实。
