# Module 14F：Memory Reconciliation + OCR x=3 Closure

Gate：**HEAVY_CHANGES_SPECIALIST_STRUCTURE_ONLY**。所有原模型identity、SLO、workload、budget、Module9、J、rho≤0.8、E/H pool均冻结；仅更新有实测来源的驻留语义，并加入四个指定OCR的g(3)/M(3)。

## Memory reconciliation

原mix_03_I3/B6 Heavy CPU_FEASIBILITY_CERTIFIED：**True**，针对旧表的CPU数学证书有效。CPU M总和=44212.000000MiB，safe总和=49113.075070MiB，budget=49140MiB，slack=26.924930MiB。14E OBSERVED_BUDGET_EXCEEDED：**三次均TRUE**。

|model identity|x|CPU M(x) MiB|CPU M_safe(x) MiB|
|---|---:|---:|---:|
|baudm/parseq:abinet@v1.0.0:abinet-1d1e373e.pt|2|1288.000000|1430.779894|
|baudm/parseq:parseq-patch16-224@v1.0.0:parseq_small_patch16_224-fcf06f5a.pt|2|1144.000000|1270.816925|
|baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt|2|1144.000000|1270.816925|
|baudm/parseq:trba@v1.0.0:trba-cfaed284.pt|2|1428.000000|1586.299448|
|convnext_tiny|3|1830.000000|2032.862738|
|facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a|1|4540.000000|5043.276957|
|openai/whisper-medium@abdf7c39ab9d0397620ccaea8974cc764cd0953e|1|3734.000000|4147.928669|
|phi4mm|2|24234.000000|26920.434752|
|roberta_sst2|1|988.000000|1097.523708|
|whisper_turbo|1|3882.000000|4312.335054|

|14E repetition|owned idle MiB|owned peak MiB|observed safe idle MiB|runtime workers|
|---|---:|---:|---:|---:|
|1|44980.000000|44982.000000|49966.210907|17|
|2|45698.000000|45700.000000|50763.804047|17|
|3|44928.000000|44928.000000|49908.446502|17|

CPU与GPU都使用同一safe factor=1.110853955254881乘owned idle residency。x2 OCR表正确加载且逐项精确符合GPU，实际17个进程与计划17个一致，无额外驻留/foreign PID。差额主要由MMS x1旧M1=4540MiB导致：它是在一次priming后、30条latency cohort前测的idle；GPU充分warmup后为5340/5904/5292MiB。14D原冻结M1但已有x1暖态ref为5688/6252/5292MiB。旧测量自身后续inference peak已经超过4540。checkpoint、FP32、框架、attention、TF32与threads一致，支持warming/input-state范围不匹配，不支持模型/框架漂移。Phi及其它模型有较小实测驻留波动；无法仅凭NVML把allocator cache机制当作已证实。数百至上千MiB差额不是数值容差。

修正semantics在新GPU前冻结：每个已测model/count使用历史M与各完整三次独立postwarm-idle观测组均值的最大值；E/H统一，原safe factor不变，不使用peak替代idle，不外推未测count，不降低cost来修复budget。原14E Heavy部署按修正表safe=50474.982019MiB，仍沿用B6=49140MiB，是否可行：False。

## OCR x3 empirical closure

|checkpoint|g(3)|M(3) idle MiB|M(3) safe MiB|x3 stability|
|---|---:|---:|---:|---|
|baudm/parseq:abinet@v1.0.0:abinet-1d1e373e.pt|2.564629662|1932.000000|2146.169842|['steady', 'steady', 'steady']|
|baudm/parseq:parseq-patch16-224@v1.0.0:parseq_small_patch16_224-fcf06f5a.pt|2.522694570|1716.000000|1906.225387|['steady', 'steady', 'steady']|
|baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt|2.436981496|1716.000000|1906.225387|['steady', 'steady', 'steady']|
|baudm/parseq:trba@v1.0.0:trba-cfaed284.pt|2.802829136|2142.000000|2379.449172|['steady', 'steady', 'steady']|

四个x3各3次独立repetition；辅助x2匹配参照同样fresh worker并交替顺序。g3=冻结g2×3个paired closed-loop throughput ratio中位数。5s warmup、30s capacity、30s单独低负载稳定性；每worker最多一个outstanding，drain不计throughput，无periodic-rate替代、无新x≥4测量。

## 原27 contexts重放

E可行19/27，H可行19/27，双方可比19。N_S reduction=7，increase=6；Generalist share increase=0；new G/S split=0；N_G increase=0；新增second G=0。已有2G驻留不解释为新增1G→2G。每行flow/memory/rho/J独立重算通过，CPU J证书gap≤1e-8秒；原Module9 acceptance/failure完整保留。这不是新GPU joint稳定窗口证明。

Gate=HEAVY_CHANGES_SPECIALIST_STRUCTURE_ONLY。ENDED_NOT_SUPPORTED_ON_FROZEN_27_CONTEXTS。

只生成两个独立候选研究manifest：HEAVY_MULTI_HOTSPOT、HEAVY_LOW_QPS_FIXED_COST。两者均PROPOSED_NOT_EXECUTED，未选择新rates/budget/hotspot，未派发CPU搜索或GPU实验。到此停止；即使x4有新active-optimum witness，也未自动扩展profiling或调原27 contexts。

## 与14C的闭环对照

原14个N_S reduction保留7个。保留：demand_balanced/B3, no_vqa_phase/B1, no_vqa_phase/B3, no_vqa_phase/B6, reasoning_vqa_heavy/B3, reasoning_vqa_heavy/B6, speech_heavy/B6。

原8个H infeasible恢复：[]；新H infeasible：[]。

统一的empirical postwarm-memory修正也作用于E：E可行数由27降为19，8个含VQA的B1 context都不再可行。它们即使忽略capacity，最小实测safe-memory task cover=17385.60496904239MiB也超过冻结B1=17007.17405495223MiB；不是只提高Heavy成本或修改budget。

重新优化的mix_03_I3/B6 Heavy在CPU修正表下可以选取另一合法部署；其joint GPU稳定性尚未重新验证，本模块没有把新CPU最优当作GPU通过。
