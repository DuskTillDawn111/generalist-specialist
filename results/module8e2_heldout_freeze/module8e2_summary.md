# Module 8E-2 — Held-out workload and validation protocol freeze

HELDOUT_GPU_CAMPAIGN_READY = true

已完成 workload/protocol 预注册；新 GPU runs=0，held-out runtime result reads=0。v1/v2、routing、quality policy、candidate pool 和 models 均未修改。

原六个 workload 永久列入 DEVELOPMENT_WORKLOAD_IDS；保存完整 record/lambda/hash 与来源，不可重命名、微调后用于 held-out 或最终 v2 advantage claim。

沿用 Module 8A 的统一 Qwen standalone s0 reference（ms 转 s）。Qwen OCR 只用于 demand 归一化，不作为可部署的质量合格 OCR path。全 task feasibility witness 固定为 Qwen + PaddleOCR；不根据 planning 更换 reference。

生成顺序证据：stage_preregistered.json → stage_workloads_hashed.json → stage_sanity_complete.json → stage_complete.json。生成与协议文件先落盘并 hash；单次 PCG64(20261004) Dirichlet([1]*6) 抽 3 个样本，再统一 affine floor alpha=0.01+0.94*draw；没有 rejection/resampling。

moderate L=0.4，high L=0.72，分别是既有 rho_max=0.8 的 50% / 90%。L=sum(lambda*s_ref)，lambda=L*alpha/s_ref。High 接近 reference planning 边界，不声称真实 GPU 或所有混合 deployment 饱和。

|Workload|L|Total rps|Seed max pressure|B2 max pressure|Planning|
|---|---:|---:|---:|---:|---|
|heldout_mix_01_moderate|0.40|2.599757|0.04867410100078358|0.031858|PASS|
|heldout_mix_01_high|0.72|4.679562|0.08761338180141046|0.057345|PASS|
|heldout_mix_02_moderate|0.40|3.247778|0.06186663450732366|0.055836|PASS|
|heldout_mix_02_high|0.72|5.846000|0.11135994211318259|0.100504|PASS|
|heldout_mix_03_moderate|0.40|3.367671|0.01971946728237347|0.013634|PASS|
|heldout_mix_03_high|0.72|6.061808|0.03549504110827224|0.024542|PASS|

各 workload 保存 exact lambda、raw request share、reference demand share、D_ref、seed、公式及 SHA-256。只进行 capability/quality/service-demand/memory/seed 检查；method candidates=0，enumeration=0，runtime evaluation=0。不可行按预注册规则保留样本并关闭 gate。

M1/M2 共用 cold MILP seed 与同一 frozen 12-model formal pool（11 个有效 quality-admitted family）、empirical capacity/memory、canonical routing、RuntimeEvaluator 和 NoiseAwareRuntimeObjective(1ms,5%)。Top-B=2、iterations<=3、unique states<=6（含 seed）、3 reps/state、patience=1、seed/final 各2 confirmations，predictor=None。重新检查源码 hashes、配置与 seeder/routing/accept/rollback/stopping AST 等价。

B2 严格沿用 8D-1：quality-admissible Specialist min s0，否则 Generalist min s0，model ID 解 ties，每个所选 model 单 replica；100% task traffic 到所选 model。其 fixed routing 与 M1/M2 canonical routing 的既有差异明确保留。

执行顺序按 workload 索引交错 method 的 seed / 每轮 Top-B state block，B2 按固定索引插入。未来 adaptive state 无法事前指定 vector，因此冻结 logical slot 绑定规则：每轮双方先独立 commit Top-B，再返回测量。跳过停止/空位，不转移预算，不根据结果改顺序。search slot 内3 reps 原子执行；GPU serial。

共享只允许 workload/deployment/routing/model/environment/protocol/phase/seed 全部 exact 相同的 raw measurement。跨 method 的共同 seed/search 可共享，confirmation 仅同 role 跨 method 共享；同 method seed/final 独立，即使 state 相同。共享仍分别记 logical evaluation 并计预算，不增加独立样本。

Primary 使用 independent final-confirmation aggregates，双向调用既有 objective 冻结 WIN/TIE/LOSS。steady > recoverable_congested > unstable；同 steady mean/P95/planning memory；同非 steady 按既有政策 TIE，不新增比较规则。P95 改善仍受既有5% mean gain gate。success_criteria.json 将 STRONG_SUPPORT/SUPPORT/NEUTRAL/NOT_SUPPORTED 和独立 B2_GAP_STATUS 的计数、退化阈值、mechanism 条件全部预注册。未完成或 protocol invalid 时不输出成功判断。

Mechanism 记录每轮 base/swaps/closures/Class P/R/Top-B/测量/接受/rollback，并对匹配 state 检查 v1 not generated / generated later / rank too low / budget or stopping missed；不增加 counterfactual GPU 测量。

最坏 logical search evaluations=78，confirmation role evaluations=30，总 aggregate requests=108，logical timed-run references=294。无共享最坏78个物理 state、294 timed runs；共同 seed 的严格共享保证最坏72个物理 state、216 search +48 confirmation=264 timed runs（timed window 15.4h，含10s warmup 16.13h，尚不含启动/drain/shutdown）。进一步 exact overlap 只降低物理测量。

ROUTING_REVISION_PENDING = true。Held-out 仅验证 neighborhood + candidate-selection revision，不能声称 routing 已解决。只有3个独立 mixture；其两档强度相关。任务数据/质量评估样本仍沿用已有冻结来源，不能声称 unseen task data 或第二硬件泛化。

Blockers: []

本阶段到此停止，没有开始 held-out campaign。
