# 三域训练画像桥接：首个独立 capture

保持原有四个物理 owner：`text#0 + text#1 + generalist_2#0 + ocr#0`。
首个 capture 已实际完成 metadata、light-kernel、natural 三域，各域 8 个
solo 训练请求（每 owner 短、长输入各一个），全部严格审核 PASS。
这些是新训练采集；之前 smoke 数据没有参加拟合。

实际证据根目录：
`artifacts/predictor-gpu-20261027/kernel-training-bridge-v3-20261028-prepared00`。

| 审核 | report SHA256 |
| --- | --- |
| metadata/strict-training-review-v3 | `409c2c3f6455e60862625ac2fe8133358539e060cf07aca180d16396ae03afa9` |
| light-kernel/strict-training-review-v3 | `c86920530485d022266ae12ac7533bedf0bf787eaf607a173030d88e68e7632f` |
| natural/strict-training-review-v3 | `db0d1d4869c15f1b576064e539c334aa243ce978193f2d7470f34ca186773313` |
| three-domain-bridge-review-v3 | `a6b5e31ddfebd6564be4c14b64cf12df0e25a8283ef997316d35d3c6ef3bd06a` |

## 实际配对结果

8 个请求的输入 receipt、输出、generation policy、实际 input/generated tokens
以及 semantic forward 形状在三域完全匹配。metadata 与 light 的 38 个语义
分组，其 CUDA launch order 和已观测静态启动签名逐项相同，实际总内核数均为
129,545。签名包含名称、grid/block、registers、静态/动态 shared memory、
cluster、stream role 及 SQLite 中可用的其他启动配置。

只有整组签名精确匹配才转移该组的算子归属，未使用 host gap 缩放或时长倍率。
共获得 38,894 个 metadata affine 算子到真实 light kernel 的精确关联：
四个 text 请求各 8,965 个，两个 G2 请求各 1,517 个。OCR 没有被语言 affine
observer 覆盖，因此其 affine 特征仍为 UNKNOWN。非 affine 内核同样不会被
伪造成 GEMM 特征。

| owner / input | metadata handler ms | light handler ms | natural handler ms |
| --- | ---: | ---: | ---: |
| text#0 short | 632.12 | 383.66 | 298.44 |
| text#1 short | 664.84 | 407.18 | 282.83 |
| G2 short | 280.38 | 202.07 | 162.16 |
| OCR short | 318.97 | 293.16 | 257.52 |
| text#0 long | 1157.28 | 908.73 | 846.69 |
| text#1 long | 1136.84 | 919.28 | 822.04 |
| G2 long | 930.24 | 855.40 | 773.37 |
| OCR long | 532.69 | 587.74 | 498.82 |

这是单次观测，不能用于给自然时序施加固定 profiler 修正系数。自然域没有
CUPTI 观测；匹配真实工作不代表其内核时序已被验证。板级功率和 owned NVML
采样峰值保留原始语义，不是单模型功率或瞬时显存峰值。SM/DRAM/L2 counters
仍为 UNKNOWN。

## 第二个独立 capture 准备

可执行目录是
`artifacts/predictor-gpu-20261027/kernel-training-bridge-v3-20261028-capture02-prepared01`。
准备时已对三个域运行 source/resource audit，均 PASS。此处只记录准备事实，
实际执行状态以各域 raw 和后续严格审核为准。

`repeat_training_bridge_prepare_v3.py` 复制冻结源与 selection，写入全新 plan、
context 和输出位置，保留原有大权重 stat manifest receipt，使相同域的
`execution_context_sha256` 与 capture1 **严格一致**：

- metadata: `245903d6adcf019dd527c13a2a6c41c247a9788ac416700605c32f4614e70706`
- light: `fc20ee9a472633fcc3d0f67dc706585cf8b7ae0f230ae249634817cfc57a7388`
- natural: `40dde9e69b8e9556353c8c5f649f0d84bc07af9cb5c7c627cc9287797e7c3f8b`

原 stat manifest 内容和实际大权重文件 stat 仍会被每次执行的 audit 检查。
此准备不复用任何测量、进程或 trace。详细准备来源写在
`repeat-preparation.json`。初始 `capture02-prepared00` 只做了 CPU 准备，
其大权重 manifest 路径使 context hash 不同，不作为这次匹配重复采集的入口。

准备后磁盘约 43.97 GiB 可用，保留 12 GiB 后约 31.97 GiB。首个三域实际数据
metadata 0.544 GiB、light 0.381 GiB、natural 0.001 GiB，映射等额外输出较小。
相同第二 capture 暂按 2 GiB 预留估算；这是实测外推，不是磁盘占用硬上界。

## 尚未证明的内容

首 capture 每域独立启动次数是 1，未满足 baseline estimator 默认 `min_runs=2`；
不同 job 或一次 capture 内的 repeat 不能代替独立启动。文本请求实际都只生成
5 个 token，长输入覆盖的是长 prefill，尚未覆盖长 decode。还缺共驻训练、
背景活动、G2 image/ASR、独立预测验证、准入/资源份额对照及共享 batch/weight
后端的真实收益。当前 PASS 证明采集与跨观测域标签配对成立，尚不证明减速、
P95、ΔU 符号、排序或端到端分流收益可靠。

新增 bridge 与 repeat-preparer 的 8 项纯 CPU 测试通过；相关严格审核测试合计
23 项通过。执行中的冻结源和已有 raw/report 均未改动。
