# Stage 2A real measurement smoke results

两个 smoke 均为真实 GPU execution；顺序运行，priming 后周期预热 5 s、measurement 20 s。
RTX 4090 / 49140 MiB，driver 595.71.05，进程级 OMP_NUM_THREADS=1。

## Arrival / routing / completion

| Run | Task | Target / actual offered rps | Target routing rps | Actual routing count | Completed | Completion rps |
| --- | --- | ---: | --- | --- | ---: | ---: |
| smoke_text | text | 4 / 4 | qwen=1, distilbert=3 | distilbert=60, qwen=20 | 80 | 4 |
| smoke_multitask | asr | 1 / 1 | qwen=0.2, whisper=0.8 | whisper=16, qwen=4 | 20 | 1 |
| smoke_multitask | lid | 2 / 2 | qwen=0.5, whisper=1.5 | whisper=30, qwen=10 | 40 | 2 |
| smoke_multitask | visual_qa | 0.5 / 0.5 | qwen=0.5 | qwen=10 | 10 | 0.5 |

## Latency (ms)

| Run | Task | E2E mean / P95 | Runtime mean / P95 | Worker mean / P95 | Schedule delay mean / P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| smoke_text | text | 18.544 / 49.900 | 18.383 / 49.733 | 17.932 / 49.265 | 0.160 / 0.202 |
| smoke_multitask | asr | 313.238 / 582.789 | 313.074 / 582.628 | 312.722 / 582.160 | 0.164 / 0.218 |
| smoke_multitask | lid | 139.275 / 383.563 | 139.098 / 383.406 | 93.103 / 274.011 | 0.177 / 0.249 |
| smoke_multitask | visual_qa | 370.770 / 1044.207 | 370.605 / 1044.034 | 194.050 / 350.753 | 0.164 / 0.192 |

## Definitions and validation

- E2E = complete − scheduled；runtime = complete − submit，包含 worker 队列；schedule delay = submit − scheduled。
- worker_latency_ms 保留原 handler 计时，不等于纯 GPU kernel latency。
- completion throughput = 正式窗口 scheduled 的请求、截至 drain 期限完成数 / 20 s。另保存窗口内完成吞吐。
- 本次两次 smoke 都在窗口内完成全部请求；无失败、无 incomplete、无 drain 后计入的 completion。
- 正式共 150 条原始记录均已复核 scheduled/submit/complete、路由计数、样本标识和均值一致性。
- measurement 测试 9/9；Stage 1/1.5 回归 27/27；旧 placement 测试 13/13；旧 benchmark 800 条 manifest 校验通过。
- Stage 1、旧 worker/protocol/launcher 的封存哈希校验通过。

## Files

- [Text config](../../joint_deploy/measurement/configs/smoke_text.json)
- [Multitask config](../../joint_deploy/measurement/configs/smoke_multitask.json)
- [Text raw JSON](smoke_text.json)
- [Multitask raw JSON](smoke_multitask.json)
- [Harness documentation](../../joint_deploy/measurement/README.md)

限制：每种模型仅一实例；有限样本 smoke 只验证 measurement 链路，不能据此判断最优 deployment/routing；本轮未验证 ResNet/PaddleOCR、多副本或 NVML 连续采样。
