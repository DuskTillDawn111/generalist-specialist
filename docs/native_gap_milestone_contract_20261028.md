# Forward 外 gap 的真实证据与最小可组合合同

本审查只读取本轮 v6 natural/kernel training、合格 review、已有 SQLite 导出和本地源码。没有启动 GPU、没有重新 export、没有读取验证答案。当前四个新 capture 的执行身份与源码均未改变。

## 不能丢掉的 32.48%

natural 的 46 个语言请求每个有 entry、4 个 transition、exit，共 276 个 native_inference 内 forward 外区间。其总 wall 为 **15.129050305 s**，占 50 请求 START→READY 总 worker occupancy 46.578154846 s 的 **32.481%**。这些都是 disjoint host-clock labels，不是额外 GPU service。

| 区间 | natural 总 wall | 当前身份 |
|---|---:|---|
| native BEGIN→prefill BEGIN | 0.470904 s | entry，46 条 |
| prefill END→decode1 BEGIN | 9.947204 s | 第一 transition，46 条 |
| 其余 decode END→下个 decode BEGIN | 0.828985 s | 其余 transition，138 条 |
| 最后 decode END→native END | 3.881958 s | exit，46 条 |

G2 的 native 外围 gap 合计 9.583197 s；连同 native_inference 外的 handler gap 为 9.601227 s。其互斥 host_call_wall 合计 7.220846 s。只保留 forward 模型不能重构该请求过程。

## 三种 gap 的源码工作与依赖

冻结 worker `semantic_progress_worker_v1.py:110–124` 在 pre_sync 后调用 `executor.infer`，最后做 post_sync、reclaim、final_sync，再 poll。冻结 adapter `compact_native.py:431–458,510–526` 负责 build、prepare、generation 及 finish。shape/index 只描述 language forward；它们没有覆盖整个 generate 调用。

entry 包含协议 request build、prompt/processor/tokenizer、输入准备和搬运、generation/cache/mask 准备、MiniCPM input embedding，以及第一个 forward pre-hook。即使任务为 text，这个区间也有实际 CUDA kernel、copy、sync，不能统称 CPU prepare。G2 的 image/audio encoder 若位于 language hook 外也只能留作粗 native 工作；本轮 measured G2 都是 text，不能据此资格化多模态 operator。

transition 从一个 semantic END 到下一个 BEGIN：observer END/emit/NVTX-pop 后的 poll，generate 的 model-kwargs/cache/mask 更新、logits cast/process、argmax、输入拼接、停止判断、下一输入准备，再到下一个 pre-hook 的 poll。`DeviceProgressObserver._finish` 先 record end event，再发 host END，之后 poll；`before` 先 poll，再发下一个 host BEGIN。host END 不等待 forward CUDA 工作完成。

在已有 kernel trace 的 prefill→decode1 同域区间中：

| owner | NVTX gap wall | owner kernel execution overlap union | cudaStreamSynchronize API wall |
|---|---:|---:|---:|
| text#0 | 3.021781 s | 2.928468 s | 2.962065 s |
| text#1 | 1.970037 s | 1.887558 s | 1.936886 s |
| generalist_2#0 | 5.002877 s | 4.922622 s | 4.844011 s |

这些列互相重叠，**不能相加**。大量 kernel 的 launch 属于其他区间，却在这个 gap 执行：分别 13,118、8,936、16,426 行。加上真实 scalar DtoH/cudaStreamSynchronize，这与停止判断读取 GPU 标量、等待既有工作相符；没有 Python call stack，不能把每个 API 都指认到某一 Python 表达式。owner overlap union 也不是 SM 占用比例，更不表示其余 GPU 空闲；其他 owner 可以同时工作。自然域纳秒不能由 profiled 域直接替换。

exit 包含最后一轮 logits/selection/stop、generate 返回、`capture_generation` 的 `sequence_ids`（detach→cpu→tolist）、解码与 adapter.finish。G2 还包含以下明确的 TTS-bound 扫描。

## G2 exit 的精确计数与源码链

冻结 adapter `compact_native.py:520–522` 通过 `inner.chat(...)` 调用 MiniCPM。本地 pinned checkpoint `modeling_minicpmo.py` 和对应 HF dynamic-cache 文件当前 SHA256 相同：`d9272ddcd2722038959cd14544b86dd4b772ed87fca7d4b4c5837f6c164b1d97`。

1. `chat:1167–1177` 对 processor 输出调用 `.to(self.device)`。
2. `generate:879–891` 准备 embedding，调用 llm.generate；`_decode:802–813` 返回 outputs；`_decode_text:829–838` 在 CUDA result 上做过滤/标量检查后调用 tokenizer.decode。
3. 冻结 `capture_generation:164–183` 已把 llm output 的 sequence IDs 批量读回 CPU 作 observation，仍返回原 output；因此原 CUDA `outputs.sequences` 保留。
4. `chat:1207–1214` 把 CUDA input_ids 与 CUDA generated_ids `torch.cat`，保留 `outputs['full_sequences']`。
5. `chat:1218–1225` 不以 generate_audio 为条件，对 CUDA row 的每个 scalar 执行 `if x == tts_bos_token` / `elif x == tts_eos_token`。这些 Python truth tests 需要取回 GPU 比较结果。

这里“冻结”严格适用于 worker/adapter 的原 campaign source receipts。checkpoint 和 HF cache 的 modeling 文件不在原 context 的 277 项 execution_sources 内；本审查保留其当前 receipt、二者一致性、静态调用链和独立 trace 数量关系，**没有补造历史模型代码快照或逐 Python 调用点认证**。

每个 known input 有 9 个不同共跑/offset 条件，生成元数据均为 5 tokens：

| G2 input tokens | full sequence N | exit 唯一 kernel/请求 | long CompareEq kernel/请求 | DtoH/请求 | stream-sync/请求 |
|---:|---:|---:|---:|---:|---:|
| 389 | 394 | 823 | 793 | 794 | 794 |
| 4665 | 4670 | 9375 | 9345 | 9346 | 9346 |

所有 18 行严格满足 `DtoH=sync=2N+6`，`long CompareEq=2N+5`，`all kernel=2N+35`。G2 exit 合计 **91,782** 个唯一 kernel、**91,314** 次 copy（其中 **91,260 DtoH**）和 **91,260** 次 cudaStreamSynchronize。CUDA 上逐 scalar 的 TTS scan 预计贡献 `2ΣN=91,152` 次比较；剩余固定项包括退出 selection/stop、decode 与结果搬运等，不能未经 stack evidence 再分配。

同一短输入的 profiled exit wall 18.469–199.611 ms；同一长输入 195.274–709.903 ms。计数恒定而时间随共跑/状态变化。不能把扫描成本设为固定 1 个经验常数，也不能把它的等待与 forward GPU 等待独立相加。

## 计数 SQL 与身份

工具 `stage_native_gap_evidence_review_v1.py` 只以 SQLite `mode=ro` 打开 root 已导出文件，检查原 export/raw/rep/sqlite 收据及 READY PID。

gap 的 trace counterpart 使用同一 PID/globalTid 上唯一 `GS_STAGE:<request>:gpu:native_inference` 和每个 `prefill/decode` NVTX 范围：entry 为 native.start→first.start；transition 为 previous.end→next.start；exit 为 last.end→native.end。NVTX push/pop 和 worker occurrence 的放置不同，故 counterpart 不能冒充原 host gap 的精确同一时钟端点。

runtime API 只计完整落在 gap 中的行：

```sql
SELECT r.rowid, r.start, r.end, r.correlationId, s.value
FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
JOIN StringIds AS s ON s.id=r.nameId
WHERE r.globalTid=:owner_globalTid
  AND r.start>=:gap_start AND r.end<=:gap_end;
```

kernel 和 memcpy 按同 PID、correlationId 关联到这些 runtime 行，要求只有一个候选 API。kernel 主键为该 SQLite 的 `CUPTI_ACTIVITY_KIND_KERNEL.rowid`，copy 主键同理；不能把父 native 的 kernel 列表加到其 forward 子列表。本报告保留每个 gap 的全部唯一 kernel rowid、kernel name counts、copy row identity/bytes/kind/stream，以及 request/owner/前后 phase 身份。CUPTI 本地头文件定义 memcpy kind 1=HtoD、2=DtoH、8=DtoD。

owner kernel overlap 另按 kernel 的 device interval 与 NVTX gap 求交取 union；这和“在 gap 发射”是两个不同集合。API wall union、kernel union、copy 和 host wall 全部是不同观测域，禁止相加。EventQuery API 的返回语义/elapsed 数值独立认证仍未取得资格。

## 最小可组合 milestone 合同

第一步可以用粗 gap 节点覆盖全部 wall，无需先声称获得每个 GPU operator 的真实 service；但节点必须产生下一 publication/stop 分支，并与前向完成状态联合。

| 节点 | 输入与依赖 | 输出 milestone |
|---|---|---|
| entry_prepare | owner/task/input identity、已支持的输入长度/模板、冻结执行配置、当前已处理 peer prefix、allocator/cache 状态 | first forward 可发布；实际 BEGIN shape 及其独立可用通知 |
| advance_generation | previous forward identity、已发射 current-stream event、last-visible query、当前生成长度/已知 stop state、peer prefix | token/stop decision 可用；continue 时发布下一 forward；stop 时进入 finalization |
| finalize_native | 生成分支结束、完整序列长度、设备序列及相关依赖、CPU transfer/解码与 TTS-scan 机制身份 | native return；后续 post_sync 的依赖输入 |
| post_sync/reclaim/final_sync | 模拟中的未确认 device/frontier、cache 生命周期 | 同步/回收结束，再 RESULT flush 与 READY |

advance_generation 至少分开 prefill→decode1 与后续 decode transition；finalize 至少以 full sequence length 和 G2 scalar-TTS-scan 身份分层。本轮 5-forward 模板只是一条观测到的训练路径，未来 EOS/生成长度不能从验证结果预先注入。

每个节点需要独立 occurrence 与 parent-processed availability；BEGIN 迟到时保留 entry age。节点响应的动态 covariates 只从自身 cutoff 已可用的事件前缀构成。未知的将来 query、token/stop、peer BEGIN/END 必须由组合模型产生，不能从训练/验证轨迹回放。

已有粗 gap 总 wall 已含 CUDA scalar read 等待：可先拟合与前向联合的 host response milestone；**禁止独立抽一个 gap 总 wall，再加同一前向的 stream/GPU wait**。只有新增可识别的 host launch/decision 与 dependency-ready milestones，并建立跨流依赖证据后，才可进一步分拆等待。

## 下一步最少训练特征与缺口

现在可 CPU 提取 entry/transition/exit labels、前后 forward index/shape、input length、已实现的生成长度、full-sequence length、last-visible query、BEGIN/END lag、peer path、natural/profiler 域及精确 execution identity。只做 forward 拟合时不得声称完整过程已建模；应同时报告 gap 支持/残差。

当前 18/10/18 请求的两个 known inputs、单 capture 只支持工作量与机制诊断，不能提供跨 capture 方差。当前没有 token-selection/stop-decision/sequence-transfer/TTS-scan 细粒度边界，没有逐 Python API stack，没有全流依赖或真实 SM service。粗路径可以保留未知；细分需要未来新版本探针在不改变语义的条件下记录这些 publication、tensor device/shape、realized sequence length、同步与搬运关联及所有通知延迟。新探针本身是新执行身份，需独立开销/结果/work/reconstruction 检验。

输入和 peer 条件不匹配、生成模板停止行为不支持、异步 dependency 无法组合时返回 UNKNOWN。按 whole capture 重复、检查 baseline-nuisance-adjusted Jacobian 与留出稳定性；单 capture 的 276 gap 不能算 276 独立实验。

## 隔离优化候选：只把 TTS-bound scan 批量读回 CPU

`minicpm_tts_bounds_bulk_candidate_v1.py` 未被 frozen/current worker 导入，也未接入四个新 capture。候选仅替换上述扫描：

```python
ids = full_sequences[0].detach().cpu().tolist()
# 原 if/elif、i+1、最后 EOS teacher_forcing 排除、last-marker 语义不变
tts_bos_idx, tts_eos_idx = scan_integer_ids(ids, bos, eos, teacher_forcing)
```

原 `outputs['full_sequences']` 保持原 CUDA tensor、原值/shape/device/返回对象；不跳过 text 请求的 TTS 计算，也不跳过 audio/teacher-forcing 分支。整数序列上的等价性可严格验证，4 项 CPU 测试包括 0–6 长度穷举、相同 BOS/EOS 的 if/elif 优先级、重复 marker、最后 EOS teacher_forcing、一次 bulk read 且不改原 row。

这只资格化 CPU 整数算法。实际候选仍须独立核对 CUDA 批量 copy 的依赖与 blocking、所有 token IDs、TTS bounds、outputs/full_sequences alias/device、text/image/asr/audio/teacher-forcing/stream 分支、异常/空序列行为、峰值内存、观测开销与完整 RESULT/READY/work。一次 bulk wait 改变调度与共跑时序；必须建立新 execution identity 和新的训练/独立收益证据，不能继承当前 baseline 或把当前 trace 后处理成“优化后”结果。当前阶段没有部署或 GPU 试验资格。
