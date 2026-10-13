# Module 14C：Heavy pool re-freeze 与 CPU 单副本结构筛查

Gate：**HEAVY_STRUCTURAL_TRANSITION_SUPPORTED**。五任务 Heavy pool 冻结为18条实测 task-model paths、16个 unique checkpoint。E regime 永久保留；未修改 Module9、Module14A/14B。

本轮统一资格：通过原 frozen task quality/capability admission，且真实 s0 与 safe memory 均严格高于固定旧参考。无需质量严格提高，完全不使用14A/14B新增 HQ SLO、历史 Pareto 标签或后续 deployment/V3 outcome 决定入池。

原准入：ASR WER≤0.042181；Text accuracy≥0.925；Image top1 accuracy≥1.000；OCR exact match≥0.960；LID semantic/capability admission，无新增 accuracy threshold。

## 1. 五任务正式 Heavy candidates

|Task|实测 checkpoint|Q|s0 ms|safe MiB（task-path）|latency/旧参考|memory/旧参考|
|---|---|---:|---:|---:|---:|---:|
|asr|openai/whisper-large-v3|0.039488445|494.912528|8120.342413|3.384627|1.990741|
|asr|openai/whisper-large-v3-turbo|0.035001122|170.522292|4312.335054|1.166174|1.057190|
|asr|openai/whisper-medium|0.037693516|355.203280|4147.928669|2.429178|1.016885|
|image|torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1|1.000000000|7.690193|1501.874548|1.398771|2.238411|
|image|torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1|1.000000000|5.847736|677.620913|1.063646|1.009934|
|image|torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1|1.000000000|18.843878|953.112694|3.427517|1.420530|
|image|torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1|1.000000000|36.589595|1915.112219|6.655289|2.854305|
|lid|facebook/mms-lid-256|0.940000000|52.265767|5043.276957|1.929756|2.809406|
|lid|openai/whisper-large-v3-turbo|0.995000000|83.178565|4312.335054|3.071118|2.402228|
|lid|openai/whisper-medium|0.990000000|69.333216|3890.210551|2.559920|2.167079|
|ocr|baudm/parseq:abinet|0.970000000|12.936974|693.172868|2.132167|1.098592|
|ocr|baudm/parseq:parseq|0.965000000|10.435751|635.408462|1.719935|1.007042|
|ocr|baudm/parseq:parseq-patch16-224|0.980000000|11.786301|635.408462|1.942522|1.007042|
|ocr|baudm/parseq:trba|0.965000000|11.880107|793.149724|1.957982|1.257042|
|text|philschmid/roberta-large-sst2|0.970000000|8.642808|2052.858109|2.292942|2.464000|
|text|textattack/facebook-bart-large-SST-2|0.965000000|10.026380|2275.028900|2.660005|2.730667|
|text|textattack/roberta-base-SST-2|0.955000000|5.506414|1097.523708|1.460855|1.317333|
|text|textattack/xlnet-base-cased-SST-2|0.945000000|10.638476|1084.193460|2.822394|1.301333|

ASR Turbo / medium / large-v3 **全部保留**，未提前选 Turbo。Wav2Vec2 latency/memory 均低于旧 ASR参考，排除。
Image除14A测量的ConvNeXt-large、ViT-L/16 SWAG、SwinV2-B，还机械保留已测ConvNeXt-tiny（Q1.000，5.847736ms，safe677.620913MiB）；它虽名为tiny，也严格满足本轮定义，不能按名称删除。
LID包括MMS-lid-256（accuracy0.940也通过原语义准入）、Whisper-medium、Turbo。Text包括RoBERTa-base、RoBERTa-large、BART-large、XLNet-base。
OCR四个STRHub checkpoint全部入池。TrOCR-large-str exact-match0.925不通过原SLO，只保留14A历史characterization。PP-OCRv4/v5第一轮模型也不通过原OCR SLO。

Whisper-medium ASR/LID共享一个checkpoint、独立s0/quality。联合catalog按两条path最大实测idle3734MiB（safe4147.928669MiB）计一次residency；LID单独测得safe3890.210551MiB仍保留原表。该策略不推断未测capability。

## 2. Screen定义与限制

复用Module12/13原27个development contexts（9 workload×B1/B3/B6），原rho≤0.8、J-routing、quality/capability、Generalist pool与empirical校准不变。新checkpoint仅xS∈{0,1}。已有ConvNeXt-tiny、RoBERTa-base、Turbo分别沿用已测至3/2/2副本的g/memory；Generalists沿用原已测校准。E使用完全原始empirical count space。
另保存相同27个context的双方所有Specialist限0/1对照，Generalists不变，检查count空间影响。每个scope54行，共108行；所有数学结果有全局J误差≤1e-8秒的certificate或明确infeasible证据。
这是CPU数学结构诊断，**不是最终GPU deployment结果，不能宣称新Heavy multi-replica optimality**。未知g(2+)始终未知，没有线性外推、人工memory scaling或新counterfactual tuning。

## 3–7. Deployment与Generalist结构变化

27/27 context deployment或feasibility不同：19个双方可行且deployment不同；8个E可行而H不可行。E27/27可行，H19/27可行。
在19个可比context中，N_S减少14个，增加5个。Generalist share增加0个；新G/S split0个；新多任务G consolidation0个；N_G增加0个；新second-G（replica或不同identity）0个。N_G反而在3个B6低负载context减少。
Heavy身份集合随workload/budget变化：观察到3种集合。I2/I3会使用全部4个OCR checkpoint并增加Whisper-medium；不能把不同checkpoint的单副本队列说成同checkpoint已测g(2+)。
Generalist routing share只等于原VQA的workload fraction；no_vqa_phase为0。不得把8个H infeasible context算作routing增长或Generalist胜出。
单副本匹配对照：raw N_S减少2个，但剔除数值近零流量residency仅1个（mix_03_I1/B3）。reasoning_vqa_heavy/B6的另一个raw减少含原E近零flow tie，不能当稳健减少。G share/split/N_G增长在对照里仍均为0。大多数主screen N_S减少不能与count-space差异隔离。

## 8–9. Memory与latency归因

内存可严格解释8个B1 infeasible context：含VQA的最小真实模型task-cover memory下界为19622.124266MiB，高于B1的17007.174055MiB；该下界尚未施加任何latency/capacity约束。添加副本也不能修复最低residency覆盖不够的问题。no_vqa_phase/B1仍可行。
其余可比composition变化总体标记BOTH_WITH_EMPIRICAL_COUNT_SPACE_EFFECT。真实服务需求变大使I2/I3需要多个OCR及LID队列，memory同时改变可负担的replica/identity组合（B3从Qwen转Phi）。这些是实测成本与优化witness的机制解释；没有新latency-only/memory-only replay，不能声称完成独立因果分解。没有任何context被证明为latency-only G/S transition。
Module12既有latency/memory轴敏感性仅用于辅助解释：latency能触发split、memory更容易减S。那些旧global scaling阈值不能直接保证本轮真实、多候选Heavy pool也发生G/S split；本轮实际未观察到。

## 10. 最小replica profiling需求

下一阶段仅6个在H主screen optimum实际出现的新checkpoint需要首次补g(2)和真实M(2)：
- baudm/parseq:abinet @ v1.0.0:abinet-1d1e373e.pt（ocr）
- baudm/parseq:parseq-patch16-224 @ v1.0.0:parseq_small_patch16_224-fcf06f5a.pt（ocr）
- baudm/parseq:parseq @ v1.0.0:parseq-bb5792a6.pt（ocr）
- baudm/parseq:trba @ v1.0.0:trba-cfaed284.pt（ocr）
- facebook/mms-lid-256 @ edc73fd00996e671dfc59d16436a29b12b10588a（lid）
- openai/whisper-medium @ abdf7c39ab9d0397620ccaea8974cc764cd0953e（asr, lid）

Whisper-medium只测一份checkpoint的residency，但ASR与LID path latency/scaling必须分别记录。不得用其中一条path的latency覆盖另一条。
已有ConvNeXt-tiny、RoBERTa-base、Turbo也出现于H optimum，标记REPLICA_PROFILING_REQUIRED的证据已由历史g(2)满足，直接复用；本轮不要求重复GPU测量，不擅自扩大历史count bounds。
其余7个checkpoint标记NO_REPLICA_PROFILING_REQUIRED：Whisper-large-v3、RoBERTa-large、BART-large、XLNet-base、ConvNeXt-large、ViT-L/16 SWAG、SwinV2-B。ConvNeXt-large仅在人为限制所有旧S为1的补充对照有正flow；ViT和RoBERTa-large只出现在补充数值近零flow tie，均无主screen boundary证据。保留在冻结pool，不因为未选或大/慢而删掉。未知g(2)不能用来证明永久dominated。
本轮没有启动任何g(x>1)测量。补g(2)以后，是否继续g(3+)只能依新的实测boundary证据决定，禁止直接展开full replica frontier。

## 11. 是否进入正式Heavy-regime GPU experiment

值得先做上述6个checkpoint的最小replica补测，尚不具备直接启动正式GPU deployment campaign的条件。Gate SUPPORTED仅支持多个context的Heavy resource/composition sensitivity，不能当作Generalist routing/2G收益证据；补测并重新CPU screen后再决定正式campaign。本模块到此停止。

## Freeze与核查

**是否使用deployment/V3 outcome选模型：NO。** Pool由既有Q/s0/M和用户本轮统一门槛机械产生，在screen前冻结；没有新增搜索、改SLO或screen后换模型。
历史7848个文件SHA256前后均一致，原E27个J*均复现13B旧baseline（误差≤2e-8秒）。独立重算每行流量守恒、rho、safe memory与J，全部通过；最大certificate gap=2.24e-10秒。
主screen全部可行行通过原Module9 frozen operational router。补充对照有3次原router NUMERICAL_SOLVER_FAILURE（E mix_03_I2/B6；E/H mix_03_I3/B3），原失败witness完整保留，没有修router、修route或冒充最终可部署结果。

文件：heavy_candidate_pool.json；regime_e.json；regime_h.json；single_replica_structural_replay.csv；transition_summary.json；transition_attribution.json；replica_profiling_required.json；module14c_gate.json。
补充：screen_protocol_freeze.json、heavy_eligibility_audit.csv、single_replica_structural_replay_matched_counts.csv、27个完整witness、completion_audit.json、module14c_artifact_manifest.json。
