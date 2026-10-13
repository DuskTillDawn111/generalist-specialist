# Module 8D-2 — Final Real Baseline Campaign

新增 valid timed runs = **101**；physical attempts = 101；复用引用 = **298**（106 个不同 raw）。Proposed 未重跑。已停止，未进入 ablation / dynamic / second hardware。

每个 condition 使用三次 search-protocol repetitions，另两次 independent final confirmation；B4 seed/final 各两次。所有有效 unstable 记录保留，无事后删样本。std 为 sample std(ddof=1)。跨 RNG 的共享 raw 已明确标记，不能当作额外独立测量。

## Table A — Proposed vs fixed baselines（mean E2E ms）

| Workload | Proposed | B0 | B1 | B2 | B3 |
|---|---:|---:|---:|---:|---:|
| demand_balanced | 44.702 | 44.702 | 169.772 | 37.412 | 44.702 |
| speech_heavy | 55.685 | 79.678 | 311.897 | 50.051 | 79.678 |
| vision_heavy | 32.325 | 32.325 | 162.960 | 29.436 | 32.325 |
| text_ocr_heavy | 21.319 | 21.319 | 88.130 | 19.934 | 21.319 |
| reasoning_vqa_heavy | 95.491 | 103.243 | 164.093 | 85.983 | 103.243 |
| no_vqa_phase | 21.355 | 22.931 | 156.533 | 21.003 | 29.184 |

完整 std/P95/stability/completion/throughput/queue/worker latency 与 signed improvement 见 static_baseline_comparison.csv。负 improvement 原样保留。

## Table B — Random baseline（三个 seed 全部报告）

| Workload | Proposed | Random-1103 | Random-2207 | Random-47 | across-seed mean±std / median | P win/tie/loss |
|---|---:|---:|---:|---:|---|---|
| demand_balanced | 44.702 | 44.702 | 44.702 | 44.702 | 44.702±0.000 / 44.702 | {'tie': 3} |
| speech_heavy | 55.685 | 56.557 | 56.557 | 55.685 | 56.266±0.504 / 56.557 | {'tie': 3} |
| vision_heavy | 32.325 | 32.325 | 32.325 | 32.325 | 32.325±0.000 / 32.325 | {'tie': 3} |
| text_ocr_heavy | 21.319 | 21.319 | 21.319 | 21.319 | 21.319±0.000 / 21.319 | {'tie': 3} |
| reasoning_vqa_heavy | 95.491 | 95.491 | 86.406 | 95.491 | 92.462±5.245 / 95.491 | {'tie': 2, 'loss': 1} |
| no_vqa_phase | 21.355 | 22.931 | 22.931 | 21.355 | 22.406±0.910 / 22.931 | {'win': 2, 'tie': 1} |

Proposed vs Random 总 win/tie/loss = **{'tie': 15, 'loss': 1, 'win': 2}**。按预冻结 NoiseAwareRuntimeObjective 双向比较，tie 表示未越过 frozen objective gate，不能等价为 latency 相同。Independent confirmation 单独保留且未反馈到搜索。

## Table C / D / E

deployment_comparison.csv：完整十二维 replica vector、N_G/N_S、Generalist/Specialist identity、empirical memory、actual idle/sample peak。
search_cost_comparison.csv：Proposed 与全部 Random 的 logical candidates、真实证据 state evaluations、new physical runs、reuse、accept/rollback、final objective。所有 job max states=6、top_b=2、iterations≤3、patience=1；缓存命中仍占 logical slot。
quality_compliance.csv：全部 21-path admission compliance 及 task-wise effective quality；排除的六条 path 均无流量。WER/CER 加权只是 frozen point-estimate reporting，不是新的 pooled quality evaluation。

## B2 强 baseline audit

B2 在以下 workload 的至少一个 latency metric 更好：**['demand_balanced', 'speech_heavy', 'vision_heavy', 'text_ocr_heavy', 'reasoning_vqa_heavy', 'no_vqa_phase']**。这些结果未截断或弱化。详见 b2_heuristic_audit.json 的逐 task queue 和逐 workload memory/latency；greedy routing 未做任何 runtime 调整。
- demand_balanced: P relative mean/P95 = -19.487% / -15.234%；B2/P empirical memory=25188/15310 MiB；B2 mean queue=0.262ms，per-task={'asr': 0.3034892389705882, 'image': 0.25625658119658123, 'lid': 0.30613536752136755, 'ocr': 0.24645661119440276, 'text': 0.23686169523809522, 'visual_qa': 0.32182779824561397}。
- speech_heavy: P relative mean/P95 = -11.256% / -11.493%；B2/P empirical memory=25188/18982 MiB；B2 mean queue=0.281ms，per-task={'asr': 0.29443066215781, 'image': 0.2624637619047619, 'lid': 0.3149704814814815, 'ocr': 0.25037328846153845, 'text': 0.2508534957957957, 'visual_qa': 0.3154390588235294}。
- vision_heavy: P relative mean/P95 = -9.814% / -4.838%；B2/P empirical memory=25188/15310 MiB；B2 mean queue=0.253ms，per-task={'asr': 0.30025675, 'image': 0.23625181914893617, 'lid': 0.2776285495495495, 'ocr': 0.25270193679653674, 'text': 0.2550290398136075, 'visual_qa': 0.34811363413413415}。
- text_ocr_heavy: P relative mean/P95 = -6.949% / -4.029%；B2/P empirical memory=25188/15310 MiB；B2 mean queue=0.243ms，per-task={'asr': 0.3415347440476191, 'image': 0.26670654285714285, 'lid': 0.2980753809523809, 'ocr': 0.24953053152739513, 'text': 0.22789411024265552, 'visual_qa': 0.3298285}。
- reasoning_vqa_heavy: P relative mean/P95 = -11.058% / -7.222%；B2/P empirical memory=25188/33288 MiB；B2 mean queue=0.271ms，per-task={'asr': 0.29841829166666667, 'image': 0.27072270175438595, 'lid': 0.351715054054054, 'ocr': 0.2490662856060606, 'text': 0.25786948655174075, 'visual_qa': 0.26941573345363773}。
- no_vqa_phase: P relative mean/P95 = -1.679% / -0.894%；B2/P empirical memory=7210/7210 MiB；B2 mean queue=0.253ms，per-task={'asr': 0.2969067785714286, 'image': 0.25989347163120563, 'lid': 0.2762396523297491, 'ocr': 0.25039107910193587, 'text': 0.2417379557823128}。

不能据本 campaign 声称 dynamic adaptation 优势；只比较已冻结 workload-specific deployment 与 deterministic greedy deployment/routing。

## Static Hybrid audit

- demand_balanced: P relative mean/P95 = 0.000% / 0.000%；search/confirmation objective outcome=tie/tie。
- speech_heavy: P relative mean/P95 = 30.113% / 61.365%；search/confirmation objective outcome=win/win。
- vision_heavy: P relative mean/P95 = 0.000% / 0.000%；search/confirmation objective outcome=tie/tie。
- text_ocr_heavy: P relative mean/P95 = 0.000% / 0.000%；search/confirmation objective outcome=tie/tie。
- reasoning_vqa_heavy: P relative mean/P95 = 7.509% / -0.743%；search/confirmation objective outcome=win/win。
- no_vqa_phase: P relative mean/P95 = 26.824% / 55.423%；search/confirmation objective outcome=win/win。

## Claim audit（六 workload 全部纳入，无综合总分）

- **A. PARTIAL** — Proposed > Seed-only；search={'tie': 3, 'win': 3}，confirmation={'tie': 3, 'win': 3}，效率证据 workloads=[]。
- **B. SUPPORTED** — Proposed > Qwen-centric；search={'win': 6}，confirmation={'win': 6}，效率证据 workloads=[]。
- **C. NOT_SUPPORTED** — Proposed > Specialist-greedy；search={'loss': 5, 'tie': 1}，confirmation={'loss': 5, 'tie': 1}，效率证据 workloads=[]。
- **D. PARTIAL** — Workload-specific deployment > Static Hybrid；search={'tie': 3, 'win': 3}，confirmation={'tie': 3, 'win': 3}，效率证据 workloads=[]。
- **E. PARTIAL** — LP-guided search > Random-neighbor under equal per-job budget；search={'tie': 15, 'loss': 1, 'win': 2}，confirmation={'tie': 15, 'loss': 1, 'win': 2}，效率证据 workloads=[]。

失败分类记录=[]。Genuine unstable 属于有效真实结果，不重试到 steady；resource infeasible 保留且 deployment 不改；external interruption 遵守既有零自动重试政策。

所有 baseline quality admissible=true。MODULE8E_READY=True，remaining blockers=[]。是否进入 8E 由后续授权决定，当前未启动。

结论限定冻结六 workload、三个随机 seeds、单 RTX4090、当前 bounds 和实验窗口；不做超出设计的 significance、全局最优或普遍优越性声明。

## Final execution validation and explicit budget audit

final_execution_validation.json = **PASS**：冻结 8D-1/8B/8C、原搜索与 runtime source、模型权重及环境全部保持一致；全部 raw identity、admission、预算、confirmation donor 与五张表核验通过。形式化 freeze 包含零流量 PaddleOCR-VL，runtime catalog 省略这个禁用模型；核验以完整向量及零流量等价逐项检查，未改变任何 routing。

新增 physical timed runs：B1=30、B2=30、B3=5、B4=36、B0=0，总计101。Baseline logical timed references=399，其中298次复用引用：273次来自8C、25次来自本campaign已完成的相同identity测量。106个不同复用donor中87个来自8C，19个来自本次101个new raw；不能将这些19个donor重复计作额外历史run。Proposed原8C的96次valid timed runs仅作reference，未重跑。

| Workload | Proposed used states | Random-1103 | Random-2207 | Random-47 | Frozen max per job |
|---|---:|---:|---:|---:|---:|
| demand_balanced | 3 | 3 | 3 | 3 | 6 |
| speech_heavy | 5 | 5 | 5 | 5 | 6 |
| vision_heavy | 3 | 3 | 3 | 3 | 6 |
| text_ocr_heavy | 3 | 3 | 3 | 3 | 6 |
| reasoning_vqa_heavy | 5 | 5 | 6 | 5 | 6 |
| no_vqa_phase | 5 | 3 | 3 | 5 | 6 |

严格公平指每个workload×RNG job与Proposed具有相同上限、停止规则及确认政策；实际消耗不要求相等。Proposed六组用了24个logical state evaluations（18个candidate）；三个Random（1103/2207/47）分别总计22/23/24个state evaluations（16/17/18个candidate）。reasoning/VQA-heavy Random-2207用了6个state，比Proposed的5个多1个，但仍在同一冻结上限内；其更好结果不能描述成用更少evaluation取胜。no-VQA两个Random提前停止于seed，消耗少但目标更差。没有跨两个workload的LP evaluation-efficiency证据。

LP claim E=PARTIAL仅代表两个no-VQA确认win；其余15个tie及reasoning/VQA-heavy的1个loss均保留。Random-2207在reasoning/VQA-heavy移除Phi，最终Qwen+S4：mean=86.406ms、P95=147.757ms、empirical memory=21516MiB；Proposed为95.491ms、157.304ms、33288MiB。Confirmation means=85.843/94.771ms，也确认Random更好。因此不能声称LP ranking在六workload普遍优于random。三个no-VQA Random final均N_G=0且steady。

B2所有六组primary mean/P95都更低，独立confirmation按冻结objective仍是五个Proposed loss、一个tie。B2逐task平均queue wait约0.228–0.352ms，completion=1，所有条件steady；当前窗口没有支持greedy routing出现明显拥塞弱点的证据。Proposed只在前四组具有memory优势；reasoning/VQA-heavy的latency与memory都由B2占优，no-VQA没有明确优势。Dynamic adaptation未实验，不能据此主张adaptation优势。

执行期间发生一次**GPU空闲、完整条件结束后的协调器切换**，仅为按冻结confirmation合同预留两个seed donor，未修改搜索/模型/路由/预算/测量协议。详见confirmation_donor_compatibility_freeze.json、coordinator_pause_record.json与coordinator_restart_record.json；timed runs interrupted/repeated/discarded=0。聊天中断也未中断后台timed window。OOM/resource infeasible/unstable/timed external interruption均为0，未重试或删样本。

MODULE8E_READY=true，remaining blockers=[]；Module8D-2已停止，未启动8E、dynamic或第二硬件。
