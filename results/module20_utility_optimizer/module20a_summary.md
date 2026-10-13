# Module 20A：Utility-Aware Joint Deployment and Routing Optimization — Offline Development

**Gate：UTILITY_MODEL_INSUFFICIENT；NO_POSITIVE_THREE_PHASE_CANDIDATE。** 两个维度独立判断。仅 CPU，未启动 GPU；原 V3、Module 19 所有负结果及历史文件均保持原样。

扫描全部 210 个预冻结六任务网格点：140 个有名义可行候选，70 个未找到；1118 个唯一候选、8820 次生成求解。没有正效用三阶段 witness；这不证明网格外或遗漏候选不存在。

## 冻结效用及证据边界

主权重 Q/T/L/P/M = 0/0.25/0.30/0.30/0.15；敏感性为预冻结 27 个权重组合，未按结果调整。L/P 均按冻结 offered QPS 加权后逐任务参考 ms 归一化。参考是 FULL_HEAVY_19 逐任务 standalone 服务均值的中位数，用同一物理标尺衡量 mean/P95，不冒充测得 tail reference 或 SLO。

参考 ms：`{"text": 10.026380266666667, "image": 27.71673651111111, "ocr": 11.880106522222222, "asr": 574.4954926166666, "lid": 76.25589069444445, "visual_qa": 160.11645761111112}`。T 为未完成及固定 offered QPS 窗口完成损失，drain 不抹除吞吐损失；持续增长/unstable 为硬拒绝。M 为 observed safe memory/budget；历史画像未提供部署预算时 U 留空。Q_norm=null，质量仅硬准入，无假质量数据。每项 norm、加权贡献、真实 pooled-request P95、未完成/queue/worker 证据均保存在 historical_utility_analysis.csv。

U 是成本项负和，不能把 U>0 当作收益；正效用指同负载可行对照的 delta_U>0.01。三阶段按自主选择事后检查；LOW 比可行 S-only、MID 比可行 G-only、HIGH 比最佳可行纯路由有正差值才计证据，没有用不可行对照制造效用增益。

## 旧算法诊断

|条件|CPU J GS/S ms|实测全系统 mean GS/S ms|实测 pooled P95 GS/S ms|真实 U(GS−S)|J/U 反向次数|
|---|---:|---:|---:|---:|---:|
|text_high|31.0313/31.4377|15.1434/11.3375|16.1201/14.2360|-0.163545|3/3|
|image_high|30.6226/30.4658|7.6177/6.5242|6.9514/6.4464|-0.018660|0/3|
|lid_mid|126.8671/127.3378|113.4520/102.2459|306.7395/185.2974|-0.108004|3/3|
|lid_high|178.4179/225.3736|218.2898/122.2218|886.8177/334.5321|-2.241160|3/3|

Text HIGH、LID MID/HIGH：同驻留中 J 偏向 GS，而三个重复的真实 U 都偏向 S。Image HIGH 不是 J 排序反向：原操作容量 rho≤0.8 硬门槛排除了固定 S-only，尽管 GPU 三次 steady，J 本身也可偏向 S。新优化器将 .8–1 的名义可行点标成操作余量不足/待验证，不能据非饱和吞吐改写 g 或宣称新的稳定容量。

误差同时涉及：(1) J 的 D/(1-rho) 队列代理与 deterministic-periodic 单快 worker 的实际排队不一致；(2) 系统目标遗漏逐任务参考归一化、tail、吞吐损失和显存取舍；(3) standalone service 在共置中不保持不变，特别是 Qwen LID、语音路径与背景任务。历史 path 的 service/s0、queue、CPU rho 与真实 worker 状态在 historical_path_diagnosis.csv 和 old_algorithm_diagnosis.csv。不能仅凭这些观察区分纯 GPU 干扰、CPU/frontend、样本差异与 runtime 状态，现有日志没有独立识别所有因果参数。

ASR HIGH 两臂 unstable 且未完成，U=null，只保留截尾诊断。Module17 的局部均值包络有原条件独立测量支持，但没有推广到新模型组合。standalone s0、实测 g/M、原始请求时序有独立画像来源；通用共置容量、service tail、统计效用界和跨任务质量标尺均没有。

## 新优化器与验收

第一层：全部准入模型和实测 count 做线性 MILP 候选生成，目标 G 比例 free/0/0.01/0.05/0.2/0.5/1，容量 cap 0.8/0.95，3 个预冻结显存种子权重；同时保留匹配历史候选。不同部署、副本和纯/混合路由共同竞争。MILP 内的比例用于保留候选，最终选择不受阶段限制。不是完整非线性效用优化证书。

第二层：固定周期到达、逐 worker FIFO、family 内 round robin，effective per-replica service=s0*n/g(n)，10 秒 warmup 后 120 秒测量。最大化 U_hat，另做相位及 1.25/2×服务压力场景。没有对 Module19 拟合校正系数；场景不是可信区间或实测保守上界。名义获选点须 rho<1；无法验证共置稳定性，全部显示 high uncertainty，stress-unstable 候选不标为认证稳定。

A：四个已知负分流条件，固定驻留候选都选择 S-only，12/12 开发配对 U 排序方向正确；27 个权重均保留该选择。Image HIGH 的 S-only 原 .8 操作门槛仍失败，所以这是条件性诊断候选，不能当成已认证基线。

B：保留并逐项报告 LOW 实测显存节省、画像副本容量收益，但没有已有独立、匹配、真实系统 U 正收益对照可以验证普遍 GS 获益排序。Module15/18 的部署 steady 和阶段行为不是反事实效用收益；不把它们充作独立胜利。所有可观察 Module19 部署对照包括正、负、同配置，均在 all_observed_deployment_contrasts.csv；完整 V3 仍未验证。

C：Module19 全部为开发集。按整个模型组合/workload 留出 Module16 六次运行；独立 matched ranking groups=0，因此排序准确率、反转数均 null。五个可评价 U 运行 MAE=15.342508；名义错误判为可行：['mid_rep3']。六次均无法在全部服务压力场景下保持可行；没有有限完整场景界覆盖率，不能将这种宽泛拒绝算作准确稳定性预测。该留出为回顾性，项目内已公开历史，不是盲测。

D：生成 42.084s，预测 12.044s，总 wall 66.078s（单次 CPU 记录，无统计加速声称）。FULL/COMMON 原始部署 Cartesian 状态分别 286,654,464/31,850,496，本算法共评估 1118 个唯一部署+路由候选。每网格 visited state 比例下降范围 99.99998116%–100.00000000%，只指省去显式枚举的状态；MILP 分支代价包含在计时，没有实测全枚举 wall speedup，也不保证全局最优。

E：27 个冻结权重下，77.57% 的有效网格选择与主权重为同一候选；权重依赖明显，不能宣称全网格稳定。敏感性只在生成候选集上排序，未冒充完整重优化。

F：全部 210 个点和 30 个 pool/budget/target 分组均保存。自然结构三阶段组合数=0；严格正效用 witness=0；名义最优 mixed 网格点=12。没有为追求三阶段改变权重、参考、模型池、预算或 QPS。

## 七项最终回答

1. 错预测发生在 Text HIGH、LID MID/HIGH 的 J→U 排序；Image HIGH 是保守操作容量门槛与真实运行可稳定的差异，ASR HIGH 是容量/共置不稳定漏判。
2. 新方法直接比较全系统效用贡献，并利用周期队列模型避免 J 的过度排队收益；纯路由和多副本/部署候选始终保留。开发样本改善不是预测器已修复的证据。
3. 尚不能证明独立效用排序正确：没有独立部署对照，且留出容量误判与效用误差明显。
4. 预冻结六任务网格未找到理论正效用三阶段配置；不能外推为不存在。
5. 省去的显式部署状态计数和实际 CPU 开销见上述 D；全枚举实测加速比未知，启发式覆盖有代价。
6. 新组合 service/tail/容量、相对 U 排序、共置因果误差和真正 GS 正收益需要新的匹配 GPU 对照，按配置/workload 留出；低 QPS 背景 tail 也需更多样本。
7. 不具备进入 Module20B 三阶段收益验证的证据；可作为另行授权校准/对照数据采集的开发输入。未创建虚假成功 witness，未自动启动 GPU。

完整性：15036 个受保护文件 hash 未变；7 项 CPU 科学不变量测试通过。冻结清单、候选完整输入、失败生成尝试及执行源码 hash 均落盘。
