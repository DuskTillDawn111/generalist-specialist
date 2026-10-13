# Module 8A W1/W2/W3 convergence audit

冻结目录中的 s0 是 standalone worker latency；λ×s0/1000 的单位为参考 GPU 秒/秒，仅是服务成本 proxy。未运行新 search。

|Workload|Total rps|Total Qwen demand s/s|ASR share|LID share|ASR+LID|Whisper-small speech demand s/s|
|---|---:|---:|---:|---:|---:|---:|
|W1_balanced|2.350000|0.312393|31.41%|17.91%|49.32%|0.039359|
|W2_generalist_heavy|2.940000|0.380082|30.12%|18.04%|48.15%|0.046550|
|W3_specialist_friendly|2.120000|0.264604|29.66%|12.69%|42.35%|0.029320|

ASR 在三组中确实是最大的单项 task demand，但没有超过总 demand 的 50%。不能把“最大单项”写成“绝对多数”。Whisper-small 同时覆盖 ASR 与 LID；二者合计贡献及小模型 LID 成本共同解释其价值。

|Pair|Raw L1|Raw cosine|Qwen-demand L1|Qwen-demand cosine|
|---|---:|---:|---:|---:|
|W1_balanced / W2_generalist_heavy|0.033435|0.999360|0.025821|0.999428|
|W1_balanced / W3_specialist_friendly|0.147732|0.988994|0.149637|0.987341|
|W2_generalist_heavy / W3_specialist_friendly|0.128482|0.991033|0.127530|0.988554|

三组 request mix 本来就接近；W2 主要增加总量，不能凭 workload 名称认定覆盖了不同 regimes。Demand 空间的距离见上表：W1/W2 几乎同方向，W3 speech 比例略低，并未建立 vision/text/VQA 主导 regime。

## Offload explanation

Qwen ASR s0=654.078 ms。ASR specialists: whisper-small=190.166 ms, whisper_turbo=170.522 ms, distil_whisper=146.224 ms.
Whisper-small LID=27.084 ms，turbo LID=83.179 ms；Qwen LID=139.909 ms。turbo ASR 比 small 更快，但 small 的 LID 更快且内存更少。distil-whisper ASR 最快，但不覆盖 LID。

下面是每类 Specialist 对其全部覆盖任务的探索性 λ(sQ−sS)；它衡量总服务成本下降，**并非 canonical min-max objective 的排名**。在新增模型后 Qwen 剩余压力、分流约束、内存和并发干扰仍需由冻结 trace 解释。

|Workload|Specialist|Task(s)|Pressure relief ms/s|
|---|---|---|---:|
|W1_balanced|distilbert|text|32.696|
|W1_balanced|resnet50|image|53.646|
|W1_balanced|convnext_tiny|image|53.506|
|W1_balanced|paddleocr|ocr|35.432|
|W1_balanced|paddleocr_vl|ocr|3.646|
|W1_balanced|whisper-small|asr,lid|114.717|
|W1_balanced|whisper_turbo|asr,lid|95.226|
|W1_balanced|distil_whisper|asr|76.178|
|W1_balanced|roberta_sst2|text|31.306|
|W2_generalist_heavy|distilbert|text|42.913|
|W2_generalist_heavy|resnet50|image|65.717|
|W2_generalist_heavy|convnext_tiny|image|65.545|
|W2_generalist_heavy|paddleocr|ocr|43.405|
|W2_generalist_heavy|paddleocr_vl|ocr|4.467|
|W2_generalist_heavy|whisper-small|asr,lid|136.469|
|W2_generalist_heavy|whisper_turbo|asr,lid|112.420|
|W2_generalist_heavy|distil_whisper|asr|88.875|
|W2_generalist_heavy|roberta_sst2|text|41.089|
|W3_specialist_friendly|distilbert|text|32.696|
|W3_specialist_friendly|resnet50|image|53.646|
|W3_specialist_friendly|convnext_tiny|image|53.506|
|W3_specialist_friendly|paddleocr|ocr|35.432|
|W3_specialist_friendly|paddleocr_vl|ocr|3.646|
|W3_specialist_friendly|whisper-small|asr,lid|82.748|
|W3_specialist_friendly|whisper_turbo|asr,lid|71.642|
|W3_specialist_friendly|distil_whisper|asr|60.943|
|W3_specialist_friendly|roberta_sst2|text|31.306|

## Frozen search evidence and limits

### W1_balanced

原 seed pressure=0.312393；final canonical max pressure=0.158317。Final routing 将 ASR/LID 全部交给 Whisper-small，Qwen 保留其余四类。剩余 Qwen pressure=0.158317，Whisper pressure=0.039359。

Round 0: incumbent qwen=1；Top-B=[[["qwen", 1], ["whisper-small", 1]], [["qwen", 1], ["whisper_turbo", 1]]]；accepted=[["qwen", 1], ["whisper-small", 1]]。
Round 1: incumbent qwen=1, whisper-small=1；Top-B=[[["minicpm", 1], ["qwen", 1], ["whisper-small", 1]], [["phi4mm", 1], ["qwen", 1], ["whisper-small", 1]]]；accepted=null。

|Round|Candidate|Rank|Δz|Mean E2E ms|P95 ms|Status / reason|
|---|---|---:|---:|---:|---:|---|
|0|qwen=1, whisper-small=1|1|0.154076|114.824|210.507|ACCEPTED / mean_improved|
|0|qwen=1, whisper_turbo=1|2|0.154076|125.025|214.976|ROLLBACK / another_top_b_candidate_preferred|
|1|minicpm=1, qwen=1, whisper-small=1|1|0.064837|141.921|283.730|ROLLBACK / mean_worse|
|1|phi4mm=1, qwen=1, whisper-small=1|2|0.061429|130.473|221.769|ROLLBACK / mean_worse|

Stop=NO_RUNTIME_IMPROVEMENT；所有候选排名与 decision 原始字段保存于 convergence_evidence.json，source=/root/autodl-tmp/generalist-specialist/results/integer_replica_search_multiworkload/w1/search_trace.json。

### W2_generalist_heavy

原 seed pressure=0.380082；final canonical max pressure=0.197063。Final routing 将 ASR/LID 全部交给 Whisper-small，Qwen 保留其余四类。剩余 Qwen pressure=0.197063，Whisper pressure=0.046550。

Round 0: incumbent qwen=1；Top-B=[[["qwen", 1], ["whisper-small", 1]], [["qwen", 1], ["whisper_turbo", 1]]]；accepted=[["qwen", 1], ["whisper-small", 1]]。
Round 1: incumbent qwen=1, whisper-small=1；Top-B=[[["minicpm", 1], ["qwen", 1], ["whisper-small", 1]], [["phi4mm", 1], ["qwen", 1], ["whisper-small", 1]]]；accepted=null。

|Round|Candidate|Rank|Δz|Mean E2E ms|P95 ms|Status / reason|
|---|---|---:|---:|---:|---:|---|
|0|qwen=1, whisper-small=1|1|0.183019|105.740|200.185|ACCEPTED / mean_improved|
|0|qwen=1, whisper_turbo=1|2|0.183019|119.380|217.947|ROLLBACK / another_top_b_candidate_preferred|
|1|minicpm=1, qwen=1, whisper-small=1|1|0.080953|133.146|275.316|ROLLBACK / mean_worse|
|1|phi4mm=1, qwen=1, whisper-small=1|2|0.076575|124.818|218.356|ROLLBACK / mean_worse|

Stop=NO_RUNTIME_IMPROVEMENT；所有候选排名与 decision 原始字段保存于 convergence_evidence.json，source=/root/autodl-tmp/generalist-specialist/results/integer_replica_search_real_pilot/search_trace.json。

### W3_specialist_friendly

原 seed pressure=0.264604；final canonical max pressure=0.152537。Final routing 将 ASR/LID 全部交给 Whisper-small，Qwen 保留其余四类。剩余 Qwen pressure=0.152537，Whisper pressure=0.029320。

Round 0: incumbent qwen=1；Top-B=[[["minicpm", 1], ["qwen", 1]], [["qwen", 1], ["whisper-small", 1]]]；accepted=[["qwen", 1], ["whisper-small", 1]]。
Round 1: incumbent qwen=1, whisper-small=1；Top-B=[[["minicpm", 1], ["qwen", 1], ["whisper-small", 1]], [["phi4mm", 1], ["qwen", 1], ["whisper-small", 1]]]；accepted=null。

|Round|Candidate|Rank|Δz|Mean E2E ms|P95 ms|Status / reason|
|---|---|---:|---:|---:|---:|---|
|0|minicpm=1, qwen=1|1|0.118028|235.291|590.247|ROLLBACK / mean_worse|
|0|qwen=1, whisper-small=1|2|0.112068|105.840|171.293|ACCEPTED / mean_improved|
|1|minicpm=1, qwen=1, whisper-small=1|1|0.062603|135.021|281.401|ROLLBACK / mean_worse|
|1|phi4mm=1, qwen=1, whisper-small=1|2|0.059247|125.654|204.885|ROLLBACK / mean_worse|

Stop=NO_RUNTIME_IMPROVEMENT；所有候选排名与 decision 原始字段保存于 convergence_evidence.json，source=/root/autodl-tmp/generalist-specialist/results/integer_replica_search_multiworkload/w3/search_trace.json。

Empirical g(2)：Qwen=1.215925，Whisper-small=1.237807，ResNet50=1.978475。Qwen/Whisper 因子来自 ASR representative task；对 mixed tasks 的迁移仍为 surrogate。复制不提供新的任务覆盖；小模型卸载同时减轻两类 speech 负载，但实际接受仍以冻结 runtime 为准。三组 Top-B 中没有任何 x=2 deployment，因而 no accepted x=2 在此处主要表示未被选入实测，不能写成“实测副本更差”。

Frozen v2 只有 Qwen verified visual_qa，三个 λVQA>0，故 Qwen presence 是 coverage 约束。不能将其解释为在所有 Generalists 中以性能胜出。

相同 final 的证据解释：相近 mix、相同 mandatory coverage、同样 ASR/LID 卸载机会，并在有限 Top-B 的实际 runtime 决策中选择 small。不能仅凭 standalone cost 推导“small 必然全局最优”；其他 Specialist 未被全部实际评估，停止是预算内局部无改善而非穷举最优。
