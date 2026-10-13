# Module 8E-3 — Held-out real GPU validation

Campaign: COMPLETE; REVISION_VALIDATION_STATUS=SUPPORT; B2_GAP_STATUS=REDUCED

严格保留 8E-2 workload/intensity/method/routing/quality/pool/budget/seed/判定规则；ROUTING_REVISION_PENDING=true。

|Workload|M2 vs M1|M1 vs B2|M2 vs B2|mean delta ms|P95 delta ms|
|---|---|---|---|---:|---:|
|heldout_mix_01_moderate|TIE|LOSS|LOSS|0.0|0.0|
|heldout_mix_01_high|TIE|LOSS|LOSS|0.0|0.0|
|heldout_mix_02_moderate|WIN|LOSS|TIE|-17.164488443271082|-26.622561425280793|
|heldout_mix_02_high|WIN|LOSS|TIE|-3.676245824614199|-0.9519852231245096|
|heldout_mix_03_moderate|TIE|TIE|TIE|0.0|0.0|
|heldout_mix_03_high|TIE|LOSS|LOSS|0.0|0.0|

## Required final answers

1. 实际新增 timed runs：181；physical attempts：181；confirmation runs：40；retries=0。
2. M2 vs M1 win/tie/loss：2/4/0；未分类：0。
3. REVISION_VALIDATION_STATUS：SUPPORT；按冻结规则 W>L AND W>=1 AND activated AND SR<=1 AND NOT severe。
4. SWAP/replacement 被真实评估的 workload：['heldout_mix_01_moderate', 'heldout_mix_01_high', 'heldout_mix_02_moderate', 'heldout_mix_02_high', 'heldout_mix_03_moderate', 'heldout_mix_03_high']。
5. replacement closure 进入 Top-B：12 次；全部 targeted SWAP 进入 Top-B：12 次。
6. closure 真实接受：2 次；全部 targeted SWAP 真实接受：2 次。
7. M1 没测到而 M2 同预算测到的 objective-beneficial targeted state：3 次；全部成功与失败案例见 mechanism_evidence.json，incumbent 路径差异限制因果解释。
8. v2 是否缩小 B2 gap：是。
9. B2_GAP_STATUS：REDUCED。
10. 明显退化（steady mean>=10%）：[]；stability 退化：[]。
11. moderate/high：moderate={'WIN': 1, 'TIE': 2, 'LOSS': 0}，high={'WIN': 1, 'TIE': 2, 'LOSS': 0}；mechanism 激活分 workload 记录，不调整 reference-defined intensity。
12. x=2 generated/Top-B/evaluated/accepted：32/0/0/0。
13. failure/稳定性事件：{'steady': 0, 'recoverable_congested': 0, 'unstable': 0, 'runtime_resource_infeasible': 0, 'external_interruption': 0, 'model_or_harness_failure': 0}；无重复试到 steady。
14. equal-budget 静态与实际搜索契约：True；logical=66<=78，physical states=47<=72，attempts=181<=264。
15. sharing exact-compatible：True；shared raw references=77；confirmation seed/final 独立=True；round commitment firewall=True。
16. routing limitation 仍存在；本阶段没有验证或解决 routing revision。
17. 是否支持继续 ablation：True（仅下一阶段建议，未执行）。
18. 推荐下一阶段：ablation。

全部有效 repetition 原样保留；final n=2，search n=3，mean/std 为 equal-repetition sample statistics。raw sharing 不计额外独立样本；三 mixture 的 moderate/high 成对相关，不做 significance claim。

Frozen protocol audit valid=True；campaign error=None。

本阶段停止。未修改 v2，未执行 routing revision / ablation / dynamic / second hardware。

最终独立审计 PASS：再次校验冻结 artifacts、源码、模型权重、环境、GPU release、raw hashes、共享 donor、round commitment 时间与实际预算。Audit trace 按 count transition 区分 ADD/REMOVE 与 REPLICA_UP/DOWN，保留原始 integer SCALE_UP/DOWN 编码；primary 分类与方法 metrics 未改动。

补充计数与解释：

- 181 次 physical raw measurements 全部为 steady；失败日志中的 steady=0 表示失败事件表没有 steady 事件，不表示没有 steady 测量。
- 新增 search timed runs=141，confirmation runs=40；logical search evaluations=66/78，unique physical states=47/72，total timed runs=181/264。77 次 raw sharing references 不增加独立样本。
- Replacement closure 12 次进入 Top-B，10 次实际 runtime evaluated，2 次 accepted，8 次 rolled back；另外 2 次已进入 Top-B 但因冻结的剩余预算未评估。
- M1 未生成而 M2 实际测到的 objective-beneficial state 共 3 个：mix_01_high（未接受，因为该轮选择其他更优候选）、mix_02_moderate（接受）、mix_02_high（接受）。只有后两个对应最终 primary WIN；未接受案例也计入完整机制记录。
- Frozen equal-workload positive mean gap G1=17.09745%，G2=7.16865%，相对减少 58.07182%；B2_GAP_STATUS=REDUCED，未关闭全部 workload 的 gap。
- 两个 WIN 的 frozen planning memory 均增加 6206 MiB；实际 idle/peak VRAM、queue wait、worker latency、completion、throughput 与 quality compliance 见各方法 results 与 comparison CSV。
- moderate/high 均为 1 WIN / 2 TIE / 0 LOSS，各接受 1 次 replacement，均发生于 mix_02；不据此作 intensity 因果或显著性结论。
