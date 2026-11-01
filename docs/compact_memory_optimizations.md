# 显存优化配置与待测入口（2026-10-17）

> 后续 GPU 进展：2026-10-17 已开始复测。r2 被显存比例拒绝；r3 新增 MiniCPM 原精度线性投影分批，正在独立新清单下测量。详见 [r3 修复与验证记录](../artifacts/compact-memory-repair-20261017-r3/README.md)；以下保留初次 CPU 准备说明。

本轮只改代码、做 CPU 验证和冻结新实验包，**未启动 GPU**。原 v5 配置、
`run-v5` 测量和 `compact-gpu-results/latest.json` 保留为历史证据。
通用常驻 4–8 GiB 是规划目标，尚无新配置的实测显存、速度、能耗或质量结论。

## 已实现

| 对象 | 修改 | 保留的计算与输入 |
|---|---|---|
| Qwen | 加载时关闭音频输出模块；语言输出投影只计算最后位置；主方案语言线性层 INT8 | 完整上下文、KV、原生 BF16 视觉/音频塔与输出头 |
| MiniCPM | 官方 GPTQ INT4 不变；原生 Qwen2 SDPA；最后位置输出投影；关闭语言隐藏层结果收集；视觉微批大小 2；加载后移除闲置 TTS/Vocos 输出模块 | 全部图像切片、音频编码器、投影器、重采样器、完整目录和 KV；不改官方权重文件 |
| Phi | 明确设置原生 `num_logits_to_keep=1`；图像特征提取按 2 个切片一批执行 | 所有切片、掩码和顺序、原生视觉隐藏层选取、图像压缩与音频模块 |
| 全部七个模型 | 启动前设置 `expandable_segments:True`；每次请求结束时，闲置预留超过 512 MiB 才执行真实 `empty_cache()` | 存活张量和持久工作缓冲区；清理耗时计入服务延时，整卡能耗包含清理 |

Phi 的固定 Transformers 版本原本已在生成时使用最后位置 logits；显式配置用于核验，
不能把它写成新增收益。MiniCPM 的 SDPA 只切换语言注意力，并记录实际类名；
音频编码器和 Phi 视觉编码器所需的隐藏层仍保留。
最后位置输出投影只用于本研究固定的 batch=1、贪心、无音频输出推理，完整前缀仍参与注意力。
CPU 小模型验证不替代量化权重和 GPU 内核验证，SDPA 的数值差异也需要重新评价质量。

每次请求记录缓存释放前后值和配置审计。NVML 整卡常驻与全过程峰值如实保存，
失败/中断作业也保留最终观察到的峰值，不从占用中扣除缓存。

另修复 Qwen 处理器会原地修改 `audio_kwargs` 的问题：传入嵌套选项的深拷贝，
使冻结请求标识保持一致。旧 ASR 结果仍作为诊断记录，不追认其质量准入。

## 两个独立配置

| 用途 | Study | Execution |
|---|---|---|
| 主方案：Qwen 语言 INT8 | `configs/study_memory.json` | `configs/compact_execution_memory.json` |
| 原精度对照：Qwen BF16 | `configs/study_memory_native.json` | `configs/compact_execution_memory_native.json` |

两套均启用上述实现优化；原精度对照用于区分量化与实现优化的效果。
新配置必须独立采集质量，不能继承 v5 的分数。四任务、完整输入、质量门槛、
五项等权效用和专用/通用常驻显存比 `[0.5,2]` 均保持原研究约定。

## 后续 GPU 顺序

新包每套 **90 个条件作业**：7 个 smoke、7 个新增 memory、7 个 quality、
21 个三轮 n=1/2/3 profile、48 个三轮低负载成本作业。
memory 阶段把每个模型支持任务的**完整冻结校准集**各跑一次，合计 1,440 个正式请求，
另有预热；不把重复运行当成新的质量样本。

smoke 的加载、输出或初步显存筛查失败立即停止后续阶段。完整 memory 阶段必须
全部完成、无失败请求、实测峰值在安全预算内，并通过全部 12 组显存比例，才能进入质量。
直接执行 quality/profile/cost 也会核验完整显存报告、作业收据、配置与样本指纹，不能绕过。
这只是早期显存筛查；最终准入仍依赖独立质量与同负载延时/能耗等完整证据。

CPU 准备包在 `artifacts/compact-memory-preparation-20261017/`；
其 `preparation-status.json` 列出冻结清单、校验结果和后续命令。
**当前未执行以下 GPU 命令**。后续上 GPU 时先运行主方案至 memory：

```bash
python scripts/run_compact_gpu_queue.py \
  --manifest artifacts/compact-memory-preparation-20261017/qwen-int8/experiments/experiment-manifest.json \
  --output artifacts/compact-memory-results-20261017/qwen-int8 \
  --through memory --execute
```

原精度对照把路径中的 `qwen-int8` 换成 `native`。先检查完整显存结果，
通过后再以同一清单和输出目录继续至 quality；成功收据可复用，失败证据不可覆盖。
若仍不满足显存或质量条件，继续修配置或换候选，不能带不合格模型进入系统矩阵。
各负载场景的绝对 QPS 仍需首轮实际容量确定，不在本轮预设部署结果。
