# Qwen=2 vs Qwen=1：固定 ASR 单任务

此结果适用于 frozen Qwen ASR adapter 和 Stage-1 前 30 个 ASR 样本的固定输入流；不能扩展为所有 Qwen tasks/mixes 的容量结论。中/高负载没有因双副本结果而调整。

## medium: offered 1.452 req/s

三轮均值：吞吐提高 **是**；queue 下降 **是**；mean E2E latency 改善 **是**。

- Throughput: 1.244 → 1.389 req/s；paired gain 1.117；efficiency 0.558。
- Queue mean: 2403.9 → 311.4 ms；paired queue ratio 0.132。
- Mean E2E: 3172.2 → 1336.0 ms；latency ratio 0.427；P95 E2E 5099.8 → 2622.7 ms。
- Worker latency: 767.9 → 1024.2 ms；interference ratio 1.333。
- Post-warm idle VRAM: 17909 → 35670 MiB；increment 17761 MiB；memory ratio 1.992。Observed workload/drain peak: 18390 → 36198 MiB。
- Completion ratio: 1.000/1.000；stability: recoverable_congested×3 / steady×2, recoverable_congested×1；three pairs eligible: True。

吞吐/queue/latency 改变可直接由以上方向判断；低于 offered load 的 within-window throughput 和窗口结束 backlog 不能被 drain 后 100% completion 掩盖。

## high: offered 3.364 req/s

三轮均值：吞吐提高 **是**；queue 下降 **是**；mean E2E latency 改善 **是**。

- Throughput: 1.322 → 1.600 req/s；paired gain 1.216；efficiency 0.608。
- Queue mean: 22610.1 → 15066.4 ms；paired queue ratio 0.668。
- Mean E2E: 23353.5 → 16254.8 ms；latency ratio 0.697；P95 E2E 43243.4 → 30308.1 ms。
- Worker latency: 743.0 → 1188.0 ms；interference ratio 1.600。
- Post-warm idle VRAM: 17909 → 35670 MiB；increment 17761 MiB；memory ratio 1.992。Observed workload/drain peak: 18390 → 36198 MiB。
- Completion ratio: 1.000/1.000；stability: unstable×3 / unstable×3；three pairs eligible: True。

吞吐/queue/latency 改变可直接由以上方向判断；低于 offered load 的 within-window throughput 和窗口结束 backlog 不能被 drain 后 100% completion 掩盖。

Frozen high-load capacity surrogate assessment: **over-estimates real scaling**；usable capacity gain observed = **True**。

不声称两个 Qwen replicas 提供 2× real capacity；保留 worker latency interference、backlog/stability 与独立重复值。当前未修改 7B formulation，也未据结果执行 Generalist-vs-Specialist allocation search。
