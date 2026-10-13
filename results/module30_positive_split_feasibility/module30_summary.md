# Module30 GPU 实测正分流可行性筛选

最终 Gate：**NO_POSITIVE_FEASIBILITY_CANDIDATE**。已停止；没有启动下一 Module 或独立 GPU 复现。

CPU 测前排名：① Qwen2.5-Omni + Whisper-small ASR；② Phi-4-multimodal + Whisper-small ASR。前者 G ASR 更快，后者有 Module15 真实分流画像；排名不代表已存在正效用。39 个已准入共同任务配对的依据、缺失画像和淘汰理由全部保留。

固定条件：ASR 1 QPS，Text/Image/OCR/LID/VQA 各 0.2 QPS；每组 G、Whisper-small、RoBERTa-base、ConvNeXt-tiny、PARSeq 各一副本。显存预算 34567.5534 MiB；目标路由 0/5/10%；候选各一条冻结 Poisson 轨迹及匹配输入。每个正式窗口 600 秒、30 秒预热、最长 60 秒排空。所有质量准入、20A 效用和稳定性阈值未改。

正式窗口实际记录：6/6；完整 6，可用于物理可行性比较的合法稳定窗口 4。前两个 C1 窗口受初版遥测控制锁影响，合法 U 为 NULL；原始完整 cohort 和算式保留。C1 的 10% 窗口与它们跨版本，禁止计算合法 ΔU。C2 三臂完整合法稳定：True；未执行窗口：[]。没有重复已经测量的窗口、换种子或新增点。

|组合|路由|合法 U|ΔU 对 S-only|G / S 实际 ASR 数|状态|
|---|---|---:|---:|---:|---|
|C1 qwen + Small|s_only|NULL|NULL|0 / 595|steady / eligible=False|
|C1 qwen + Small|split05|NULL|NULL|28 / 567|steady / eligible=False|
|C1 qwen + Small|split10|-1.2244|NULL|53 / 542|steady / eligible=True|
|C2 phi4mm + Small|s_only|-0.8558|NULL|0 / 572|steady / eligible=True|
|C2 phi4mm + Small|split05|-1.0770|-0.2212|21 / 551|steady / eligible=True|
|C2 phi4mm + Small|split10|-1.4809|-0.6250|52 / 520|steady / eligible=True|

最终六问：
1. **哪个组合最有潜力？** 测前首选 Qwen + Whisper-small ASR，但 C1 受遥测问题限制，没有完整同版本对照，不能由本轮确认其正收益潜力。现有合法匹配对照中最好的是 phi4mm + Whisper-small、split05，ΔU=-0.2212。
2. **是否实际观测到 G/S 优于 S-only？** 合格探索性正收益：否。只接受全对照合法稳定、双方实际执行及 ΔU>+0.01；初版受影响的完整算式差值 -0.2654730641914709 仅作技术诊断。
3. **主要物理变化是什么？** 下面给出同输入服务、等待与背景变化；native_latency_decomposition.csv 将目标 mean 的 S 保留请求执行/等待、G 转交代价、提交与采集残差逐项加总，严格等于实测 mean 差。P95 和系统效用另外按原定义计算。
   - C2/split05：S service 209.60→221.07 ms，S wait 38.31→45.39 ms；G service/wait 1737.17/63.81 ms。目标 mean/P95 249.93/526.20→324.66/674.39 ms；ΔU=-0.2212。
   - C2/split10：S service 209.60→213.93 ms，S wait 38.31→39.36 ms；G service/wait 1438.60/243.21 ms。目标 mean/P95 249.93/526.20→385.08/1475.74 ms；ΔU=-0.6250。
4. **背景任务是否损失？** 最差单任务变化为 C2/split10 visual_qa：mean +190.47 ms，P95 +1465.30 ms，ΔU 分项 -0.3102。所有五背景任务完整分项见下表；观察变化不等于已隔离 GPU 因果。

|组合/路由|背景任务|Δmean ms|ΔP95 ms|ΔU（L+P）|
|---|---|---:|---:|---:|
|C2/split05|text|+1.61|+1.56|-0.0095|
|C2/split05|image|-0.29|+5.16|-0.0053|
|C2/split05|ocr|+0.52|+2.04|-0.0065|
|C2/split05|lid|+8.18|+7.06|-0.0060|
|C2/split05|visual_qa|+109.76|+614.01|-0.1356|
|C2/split10|text|+1.92|+10.65|-0.0376|
|C2/split10|image|+0.15|-2.41|+0.0024|
|C2/split10|ocr|+0.14|+0.09|-0.0006|
|C2/split10|lid|-1.02|-9.96|+0.0043|
|C2/split10|visual_qa|+190.47|+1465.30|-0.3102|

5. **值得独立长窗口验证的候选？** 没有通过本轮门槛的候选。C1 技术上未定，不包装成负收益证明，也不因未定自动补测。
6. **无正收益的原因更接近什么？** 只对 C2 已完成的合法匹配窗口评估：以 S-only 的真实服务需求和等待识别负载点是否形成可释放排队，以 G/S 服务差与保留 S 同输入膨胀识别执行及共置代价。下述数值支持本轮归因；不推广到未测负载或未完成路由。
   - C2 S-only Specialist 完整 cohort 原生服务需求/600s=0.2067（非 GPU utilization），ASR service=209.60 ms，真实 wait=38.31 ms。若需求比远低于 1 且 wait 小，当前点的排队红利有限；分流执行差和共置外部代价更易压过它。

遥测审计：Module29 live request 无锁 deepcopy 的原方法已在 CPU 重现同型错误；修复采用 runtime.lock 一致快照、采样/日志串行化、fsync 后提交和原子文件。初版又发现 check 共用长 I/O 锁；v2 将控制状态与 I/O 锁分离。两个版本各做超过原故障时刻的长压；v2 另含慢传感器、完整/中止 cohort、并发写盘和错误注入。每次 GPU 原始记录与 durable journal 逐请求核对，技术失效 U 为 NULL。

限制：只有两个固定模型池、每池一个负载点、单匹配轨迹与小 G 样本；C1 缺合法同版本反事实。Module14F–29 全部历史文件与权重保存，Module29 Gate 不改，正式优化器不改。Stage D 仅在有合格正候选时输出 CPU 建议；独立长窗口验证另行执行。

关键文件：telemetry_fix_audit.md、candidate_pool_ranking.csv、frozen_gpu_manifest.json（原冻结）、frozen_gpu_manifest_v2.json（仅基础设施修正及未执行槽）、gpu_feasibility_runs.csv、system_utility_comparison.csv、worker_interference_analysis.csv、native_latency_decomposition.csv、module30_gate.json。

准备流程记录：原控制器在 C2_split05 启动前冷却超时后停止；另在 GPU 完全空闲时有界等待同一个冻结冷态条件，只允许补齐此前未测量的原槽，没有重复已测窗口、改阈值或扩展搜索。结果见 unmeasured_slot_completion_result.json，完整方法偏离与温度序列见 frozen_gpu_manifest_unmeasured_slot_completion.json。

## 原因判断

本轮没有探索性正收益。当前固定ASR=1QPS点的S-only原生服务需求仅约20.7%窗口，真实ASR等待约38ms；没有足够排队红利抵消慢G执行。Phi-ASR平均执行在两种分流中约1.44–1.74秒，S约0.21–0.22秒。G的平均剩余容量虽然很多，长ASR仍会阻塞同一串行worker的VQA。

因此原因更接近 **G/S性能差距 + 当前负载点S压力不足 + G共享队列对VQA的外部代价**。S同输入共置服务膨胀确实存在，但不是唯一或主要解释；10%同输入S服务增加约5.72ms（2.75%）、目标S等待增加约0.19ms。5%和10%的VQA效用损失分别约-0.1356/-0.3102，均大于任何背景改善。不是S等待下降却被其它项抵消：目标S等待在这两个窗口都没有下降。

C2三臂完整合法稳定，两比例均负收益；C1因两窗观察器影响缺少合法同版本反事实，不能声称已否定Qwen/Small组合所有比例。没有通过门槛的独立长窗口验证候选；Stage D未执行，正式优化器未改。全部结论限于本次固定池、负载和轨迹。
