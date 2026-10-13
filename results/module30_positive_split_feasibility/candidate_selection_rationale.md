# Module30 候选冻结依据（CPU-only）

排名第一：Qwen2.5-Omni + Whisper-small，ASR；排名第二：Phi-4-multimodal + Whisper-small，ASR。这只是优先测量顺序，尚无正收益认证。

完整已准入池的共同任务配对逐项见 candidate_pool_ranking.csv。Qwen 的真实 ASR 比 Phi 较快且已有六任务背景路径；Phi 配对在 Module15 有真实 G/S 请求和 S 等待，但不具备匹配 S-only 效用反事实。Module19 Phi/Small 高点两臂不稳定以及慢服务被保留；淘汰旧 4.3768QPS/8.02% 点，未淘汰整个模型配对。Qwen/Large ASR、Qwen/Medium LID 长窗口负结果直接降低优先级；其它配对缺乏完整共置画像，在最多两个组合上限内淘汰，未宣称数学上不可能。

两个组合均冻结 ASR=1 QPS；其余五任务各0.2QPS。Text→RoBERTa-base，Image→ConvNeXt-tiny，OCR→PARSeq，LID→Whisper-small，VQA→该 Generalist。五模型各一副本；预算34567.5534MiB，safe-factor与20A效用/refs及六任务质量准入原样使用。S额外承担LID时真实共享等待由新测量报告。两个组分别固定种子301001/301002，一条轨迹匹配三臂，输入池与顺序在GPU前冻结。

ASR点用合法历史 Small ASR 最大 session mean 792.1526310952381ms 核对；ASR1加LID0.2的历史需求约 0.7975694580707937，低于0.8，仅是筛选依据而非新组合稳定性保证。这个点同时规避Module19失败高点，并允许Poisson到达形成真实S排队。若S实际服务更快、队列压力不足，将保留负结果，不移动点。

分流比例预冻结5%/10%，约30/60个G请求；Phi在历史最慢ASR均值下10%加VQA需求仍有显著余量。实际冻结请求数见cpu_screening.json；没有根据收益换种子或提高样本数。历史极值不是逐请求上界。10%没有被包装成最优比例。不同模型副本的g/M表全部保留；Small x2 gain约1.238、x3约1.320，Phi x2约1.636，Qwen x2约1.216，MiniCPM x2<1；新增副本可能降低S压力，不能当同驻留路由收益，此轮固定x1。

最重要的缺失画像：当前GPU UUID、上述六任务共置条件下S-only和两种split的服务/等待/tail、S膨胀、G余量、VQA外部代价、低QPS背景mean/P95。历史数据已显示服务漂移，故不使用统一共置倍率，不预测正ΔU。两组各三次600秒探索，独立长窗口复现不自动执行。

## 历史合法性审阅补充

逐路径合法性单列于 historical_legal_path_audit.csv；原排名中的 service/queue 最大值含不稳定窗口，仅作为风险上界，不能当成合法排队红利。Module15 合法 Phi/Small 高点的 Small ASR mean 服务约224–232ms、mean 等待约158–184ms（其中两个窗口）；同时 G ASR 约1.76–1.91s。它有排队机会但没有匹配 S-only 的正 U 证据。Module19 更高 Small 服务时间与不稳定高点阻止把旧高负载直接移植；因此预冻结点采用保守1QPS，并明确该点可能无法形成足够队列压力。当前补充不改变两个模型池、负载、seed、轨迹、比例、运行顺序或淘汰决定，也不以不稳定等待预测收益。
