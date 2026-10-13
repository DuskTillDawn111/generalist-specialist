# Module 8E-4 — Minimal revision ablation

Campaign=PASS; R1=SUPPORTED; R2=SUPPORTED; interaction=SUPPORTED.

冻结 v1/v2、六个 workload、canonical routing、quality、candidate pool、objective、budget 与 seeds 均未变。用途标记 POST-VALIDATION ABLATION SET；ROUTING_REVISION_PENDING=true。

|比较|WIN|TIE|LOSS|
|---|---:|---:|---:|
|M10_vs_M00|0|6|0|
|M01_vs_M00|0|6|0|
|M11_vs_M00|2|4|0|
|M11_vs_M10|2|4|0|
|M11_vs_M01|2|4|0|

## Required final answers

1. 新增 timed runs=6；confirmation=0；retries=0；M00/M11 新增=0。
2. M10 vs M00={'WIN': 0, 'TIE': 6, 'LOSS': 0}。
3. M01 vs M00={'WIN': 0, 'TIE': 6, 'LOSS': 0}。
4. M11 vs M00={'WIN': 2, 'TIE': 4, 'LOSS': 0}，正式结果原样复用。
5. Replacement-only 是否足以产生 mix_02 improvement：False。
6. Diversified-only 有效 primary WIN 数=0。
7. R1 对已观察 full-v2 WIN 必要（仅本集、本预算）=True。
8. R2 对已观察 full-v2 WIN 必要（仅本集、本预算）=True。
9. 两者协同=SUPPORTED；joint-only workloads=['heldout_mix_02_high', 'heldout_mix_02_moderate']。
10. 没有 diversified 时 ranking 漏掉的有益 replacement：3 次；exact mechanism chain=['heldout_mix_02_high', 'heldout_mix_02_moderate']。
11. Ablation 比 full-v2 更好 workload=[]。
12. Full-v2 相对 v1 regression=0；相对 ablation=[]。
13. x=2 generated/Top-B/evaluated/accepted={'generated': 56, 'Top_B': 0, 'evaluated': 0, 'accepted': 0}；含四方法逐轮 candidate occurrences，不是独立样本数。
14. R1_REPLACEMENT_EFFECT=SUPPORTED；R2_DIVERSIFIED_TOPB_EFFECT=SUPPORTED；R1_R2_INTERACTION=SUPPORTED。
15. 足以冻结当前最终 static algorithm=True；该 freeze 不包含解决 routing。
16. 可以另行设计 Module 8F Dynamic Adaptation protocol=True；本阶段未执行 dynamic。

预算：new logical search evaluations=56<=72，new unique physical states=2，new timed runs=6<=16。
共享：E3 raw repetition references=210，E4 raw sharing=0；equal-budget=True；exact sharing=True；confirm independence=True。
所有有效 repetitions 保留 n/mean/std；复用的 raw 不计额外独立样本。不做显著性或普遍必要性推断。

本阶段停止；未修改 v1/v2，未执行 method v3 / routing revision / dynamic / resource sweep / second hardware。

最终独立审计 PASS：冻结源码/输入/模型资产/环境未改，GPU 已释放，预算、raw identity/hash、earliest donor、round commitments 与 confirmation role independence 均通过。M11 mechanism audit 直接复用 8E-3 原表并核对所有对应计数。
