# Module 15：Capacity-Constrained Three-Phase Existence Test

Gate：**THREE_PHASE_REAL_VALIDATED**。用户确认项目没有预定义E2E latency SLO，因此本模块不设置/宣称满足任何人为mean/P95阈值。冻结原quality/capability admission、14F实测safe memory、rho≤0.8、已测s0/g(x)/M(x)。E2E mean/P95只报告实测。

## 固定配置与CPU完整模型池竞争

G=**microsoft/Phi-4-multimodal-instruct@93f923e1a7727d1c4f446756212d9d3e8fcc5d81**，S=**openai/whisper-small@973afd24965f72e36ca33b3055d56a652f456b4d**；目标ASR，必需背景VQA恒为0.2 QPS。固定B1=17007.174054952MiB。M_G idle=12096.666667MiB / safe=13437.630012MiB；M_S idle=1616.000000MiB / safe=1795.139992MiB。G+S safe=15232.770004MiB，G+S2实测safe=17041.240243MiB>B1。

C_G=(0.8×g_G1−0.2×s_G,VQA)/s_G,ASR=0.679872738033QPS；C_S=0.8×g_S1/s_S,ASR=4.206853707655QPS，满足0<C_G<C_S。质量未修改：Phi ASR WER=0.019519856405654026；Whisper-small ASR WER=0.042180839129459274，均通过原WER≤0.042181；VQA沿用原语义准入。

资源优先目标是全池minimum measured safe memory；同最小cost的全部states再由原Module9 J-routing选择，不把replica count插在J之前，不强制G或S身份。完整已测E/H池参与，未测counts保持UNKNOWN，无linear extrapolation。独立枚举与LP确认：此预算中与正负载相关、覆盖ASR/VQA的已测驻留状态只有G1-only与G1+S1；LOW二者都可行但前者资源更小，MID/HIGH只剩后者容量可行。Qwen/Minicpm单G、Phi2、其它ASR S、Whisper-small2/3均因实测memory不满足同预算。详见measured_model_competition.csv；这不证明所有未测count物理不可能。

|阶段|ASR QPS|最优驻留|CPU G份额|CPU S份额|max rho|safe MiB|
|---|---:|---|---:|---:|---:|---:|
|LOW|0.169968185|{'phi4mm': 1}|100.000000%|0.000000%|0.224017469|13437.630012|
|MID|2.443363223|{'phi4mm': 1, 'whisper-small': 1}|0.000000%|100.000000%|0.464644296|15232.770004|
|HIGH|4.376821892|{'phi4mm': 1, 'whisper-small': 1}|8.021585%|91.978415%|0.765556719|15232.770004|

LOW<CG；CG<MID<CS；CS<HIGH<CS+CG。HIGH预测max rho=0.7655567194，距0.8约0.03444。背景、pool、budget完全相同，仅目标QPS变化。预设候选/点选择规则和实际point freeze在GPU前落盘。

原V3通过module9bc.frozen_v3.create/snapshot做CPU dry-run，原neighborhood/Top-B/J-ranking均未改；保留每阶段trace和J serving route。CPU dry-run不是完整GPU accept/rollback结果，也不是全局J最优参考；没有为V3编造runtime标签或额外GPU基线实验。

## 冻结GPU验证

每点三次独立worker进程，seeds15101/15202/15303，10s warmup、120s测量，既有periodic请求发生器、worker和stability_v2阈值不变；无容量profiling。真实路由统计来自measurement requests，priming不计入。

|阶段/repetition|实际ASR G份额|ASR mean ms|ASR P95 ms|system mean ms|system pooled P95 ms|postwarm safe MiB|after safe MiB|owned peak MiB|稳定性|通过|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|LOW/1|1.000000|1213.643081|2432.217873|738.947805|1937.215667|13512.427512|13536.866299|12186.000000|steady|True|
|LOW/2|1.000000|1144.119447|2496.495204|685.310959|2078.904038|13665.725358|13696.829268|12330.000000|steady|True|
|LOW/3|1.000000|1264.858807|2388.198840|818.032535|2075.282873|13487.988725|13510.205804|12162.000000|steady|True|
|MID/1|0.000000|224.541648|424.637443|223.479316|420.889033|15314.232627|15314.232627|13786.000000|steady|True|
|MID/2|0.000000|200.299081|386.967339|200.776020|386.170995|15467.530473|15467.530473|13924.000000|steady|True|
|MID/3|0.000000|205.653439|435.125229|207.216309|432.031835|15289.793840|15289.793840|13764.000000|steady|True|
|HIGH/1|0.079848|539.775730|1471.923043|565.210215|1734.204596|15334.227998|15338.671414|13808.000000|steady|True|
|HIGH/2|0.080000|542.154048|1663.805836|591.886465|2121.149254|15485.304136|15498.634384|13952.000000|steady|True|
|HIGH/3|0.079848|584.536000|1728.769228|610.972856|1875.591469|15309.789211|15312.010919|13784.000000|steady|True|

CPU memory和GPU safe计算使用同一个factor×owned idle；postwarm与postmeasurement两个idle窗口都核验，sampled owned peak单列并要求物理驻留未越预算，不把peak乘factor冒充idle。不是device overhead或其它进程双计。每task/route/request、worker PID、队列/尾部增长与stability原因完整保存在raw和gpu_validation_rows.json。

三阶段各3/3均steady且memory合规；LOW不驻留S，MID目标主要由S执行，HIGH三次均有实际G/S完成请求。已取得一个真实可复现容量约束三阶段实例。

完成后停止。仅此三个离散点、120秒窗口/三次重复的development证据，不宣称连续stable window、无定义的latency SLO、任意输入长度或未测副本的全局最优。历史源与全部保护文件hash核验通过。
