# 同 owner ChatQA batch 后端 v2

新增 [shared_text_batch.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/backends/shared_text_batch.py) 和 [worker entry](/root/autodl-tmp/generalist-specialist-project/scripts/shared_text_batch_worker.py)。该实现实际把一批请求送进同一个模型的一次 `model.generate`，不是在单请求循环外贴 batch 标记。当前完成 CPU 假 tokenizer/model 和真实 frozen adapter 的契约测试，**未加载真实 GPU 模型，未证明质量等价、性能/显存或净效用收益，未接入既有冻结采集计划**。

## 给调度器的接口

```python
prepared = owner.prepare(sample)  # frozen adapter.build + executor.prepare
prepared.input_tokens            # 精确未补齐输入长度
prepared.compatibility_key       # 相同owner/协议/adapter/全部生成参数才可组批
execution = owner.execute_prepared([prepared1, prepared2], evaluate_quality=False)
execution.results                # 与顺序相同的逐请求原生结果dict
execution.metrics                # 同owner、调用次数、完整批事件、实际padding等
# owner.run_batch(samples, evaluate_quality=False) 为便捷入口。
```

owner 内单锁拒绝并发prepare/generate；同一个旧executor的candidate/hook状态不可被两线程并发修改。PreparedText 绑定owner令牌、sample/request、精确token及policy hash，执行前再检查；不同owner、不同compatibility key、policy/继承generation_config变更、输入篡改均拒绝。调度器在上层管理多逻辑alias、enqueue/arrival时间、配批窗口、长度bucket与微批选择；本后端不自行改变路由或增加隐藏等待。

`service_ms` 是共同批执行的wall时间，**不可逐请求相加**；包括张量装配、generate、输出恢复/finish和批后cache cleanup，已记录的逐请求prepare时间另列。worker_latency包含收到批后的prepare与执行；入队/配批等待需由上层时间戳贯通。不能拿新批的service字段直接套旧batch1容量/`g(n)/n`；需要新执行身份下匹配的训练、独立验证与容量映射。

## 模型和质量契约

复用 frozen `NativeExecutor.adapter.build` 构建完整ChatQA原prompt，逐样本调用原 `executor.prepare` 检查BOS、完整输入与context。保留相同checkpoint、revision、precision、QA PEFT adapter和质量协议；生成后的每行仍调用原 `adapter.finish`，原评分/evaluate_quality参数及结果schema不变。label/answers不进入build或token序列。

各行保留原token顺序；左侧用模型generation_config.pad_token_id补齐；模型未指定pad时使用当前原HF单请求的EOS fallback（第一个ChatQA终止符），不新增special token、不扩embedding。attention_mask准确标记左pad=0/原token=1。仅接受decoder-only Llama（当前ChatQA）按mask生成逐行position_ids；不手动传一次固定position_ids导致后续decode位置失真。

每批调用一次标准HF generate，精确传原max_new_tokens/top_p/top_k/temperature等显式参数；ChatQA双终止符沿用原逐token反向检查。每批从fresh cache开始，不注入past_key_values、共享prefix或外部cache，batch各行由HF独立mask和KV索引处理。两行同权重不等于共用KV。

当前只支持 deterministic greedy、beam1、return1；采样/beam不通过改参数“兼容”。padding会污染某些history/绝对长度处理器，因此遇repetition_penalty非1、no_repeat_ngram_size非0、min_length非0、forced token、额外stop字符串、assisted/watermark/static cache等不支持配置明确拒绝。native generate_options不能覆盖不同的已声明质量参数；未知生成控制也拒绝。当前冻结text协议为 `do_sample=false,num_beams=1,max_new_tokens=128`，未启用这些history控制，属于支持范围。其他质量政策必须先扩展并验证，不能偷改budget/beam/top_p。

输出按**补齐后的共同prompt宽度**切片，不按每行未pad长度切，以免把prompt尾部当答案。每行取首个原EOS（含EOS作为generated token），随后只允许HF fill pad并剔除回填；无EOS必须恰好达到原max_new_tokens，否则报错。拒绝prompt被修改、少/多行、空输出、超预算、EOS后非pad及无EOS短输出。原finish看到与batch1相同的终止符/生成数口径。

普通generate整批返回：短行先遇EOS仅表示该行语义结束，未提前对外返回。本实现所有请求在整个generate/恢复/cleanup完成后交付，`row_early_delivery=false`；不虚填短行的低延迟。批KV/峰显存、补齐计算、等待及新吞吐最终需实测。

## 显式运行入口

模块import和 `python scripts/shared_text_batch_worker.py --help` 不导入torch、不加载模型。`load_retained(spec, source_root, execute=False)` 默认拒绝，只有显式execute=True才调用原ProfileAdapter加载一份GPU模型。worker同样要求 `--execute`；环境须在加载前满足冻结allocator配置。工厂把frozen source-root挂到私有package namespace，避免当前gsdeploy已import导致误用新版backend；记录完整源码树hash、spec hash和执行身份，并于import/加载后再次核对。可以传 `--spec-sha256` 固定spec。

本次未执行下面的真实加载命令；后续仅在单独协议和GPU授权下使用：

```text
python scripts/shared_text_batch_worker.py --spec ABS_TEXT_SPEC --source-root ABS_FROZEN_SOURCE --spec-sha256 SHA --execute
```

wire由上层显式发送 `{op:"batch",batch_id:"B",requests:[{request_id:"r1",task_type:"text",sample:{...},submit_ns:...},...]}`。输出READY、START、BATCH_RESULT（results与requests顺序一一对应，保留request_id/pool_id/arrival_index/submit_ns）及shutdown后的BYE。异常输出ERROR和原始traceback后抛出、退出，不重试/回退/换模型/跳过行。此协议与原单alias RESULT不同，应由新运行时适配，不能直接替换旧worker命令并期待旧transport理解BATCH_RESULT。

## CPU 验证

[tests/test_shared_text_batch_v2.py](/root/autodl-tmp/generalist-specialist-project/tests/test_shared_text_batch_v2.py) 13项PASS：一次批generate、精确leftpad/mask、原参数不变、padded prompt slicing、两种EOS及长度终止混批、回填剔除、逐行schema/score对照、准备输入/owner/配置变更拒绝、unsupported质量控制拒绝、输出破坏/短输出拒绝、异常不替换、并发owner锁、context/BOS/task拒绝、import/help无torch加载、批wire/异常停止。

其中一项使用**真实frozen CompactAdapter + NativeExecutor.prepare**（注入CPU tokenizer/model）核对完整prompt与输出finish/schema、质量参数以及参考答案隔离。它验证软件契约，不能证明真实int8/PEFT GPU批执行和单请求数值/质量完全相同。

CPU命令：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m unittest discover -s tests -p test_shared_text_batch_v2.py -v`。未启动GPU、未修改frozen源码、未清除任何resource blocker。

本后端将完整同模型权重只加载一次并执行共同batch；它没有将 [shared_linear CPU算子](/root/autodl-tmp/generalist-specialist-project/gsdeploy/backends/shared_linear.py) 插入HF int8层，未实现异构公共子矩阵融合、prefill/decode抢占或跨请求prefix/cache复用。这些能力须各自有真实执行及成本证据后再进入预测。
