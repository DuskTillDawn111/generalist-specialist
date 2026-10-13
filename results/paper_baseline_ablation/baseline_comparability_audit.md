# Paper Baseline and Ablation Evaluation：可比性审计

本轮复用 Module35 冻结的九个开发单元与 TEXT3 三池原始基线；新增固定部署诊断和同域 U-only Oracle 计时。完整八维矩阵见 `baseline_comparability_matrix.csv`。每条结果同时给出模型、副本、绝对QPS路由、比例、硬约束和来源。

所有主CPU有限域单元使用 A/B/C 同一池清单、49140 MiB、原质量准入、0/1副本、K=1完整端点路由、原20A FIFO120秒 phase=.37性能预测、原U权重与归一化、共同rho≤.8及完整drain约束。开发场景为LOW18、ASR32B、TEXT_HALF_MID18；后者实际Text=10QPS，保留原场景名。TEXT3是此前独立冻结的一种工作负载的三池验证；本轮诊断已知道其结果，不能将新增消融重新称为盲测。GPU持续稳定性不是CPU Gate。

| 方法 | 副本域 | 路由域 | 严格同搜索域 | 内部选择规则 |
|---|---|---|---|---|
| U_ALIGNED_DP | same binary0/1 | same complete K1endpointdomain | True | U |
| CPU_U_ORACLE | same binary0/1 | same complete K1endpointdomain | True | U |
| 20A_U_GRID | same binary0/1 | same complete K1endpointdomain | True | U |
| 20A_NATIVE_CONTINUOUS_PRESERVED | same binary0/1 | continuous; differs from K1Oracle | False | U |
| V3_J_GRID_CONTROL | same binary0/1 | same complete K1endpointdomain | True | V3 J routing/ranking plus native noiseaware meanP95memory 5%acceptance |
| V3_NATIVE_CONTINUOUS_PRESERVED | same binary0/1 | continuous; differs from K1Oracle | False | V3 J routing/ranking plus native noiseaware meanP95memory 5%acceptance |
| FASTEST_PATH_STATIC | same binary0/1 | same complete K1endpointdomain | True | fastestservice/static memory rule; no Usearch |
| MIN_MEMORY_COVER_FASTEST | same binary0/1 | same complete K1endpointdomain | True | fastestservice/static memory rule; no Usearch |

原生20A：原MILP连续路由及42种种子生成；最终由原U评分。历史20A本身还包含特定历史候选复用，本轮使用既有Module34/35域适配驱动，不能称为重新执行原20A全部历史研究流程。20A网格控制只改变MILP变量单位/整数性，不补候选。原生终点即使偶然落入K1，也不能据此把原生搜索时间或Regret作为严格同域比较；保留 `selected_plan_reference_gap`，正式 `regret` 置空。

V3：J用于路由和候选排序，正式接受逻辑仍为原多指标噪声门槛；不能描述为全局单标量J优化器。CPU使用冻结预测后端。原生连续V3与独立J网格控制分开记录。原生的域不同或数值失败保留，不能替换为控制版成功。V3内部门槛不同于统一输出合法性Gate。

静态基线：最快服务路径采用既有 `m34.cpu.static`；最小显存覆盖加最快路由采用既有实现，完整合法验证后报告U。其历史微秒时间只包括配置构造，最终四情形预测在时间之外；不能把该数字称为带验证的端到端优化时间。

Module33：预算34567.5534MiB、K=2、限定Mixed、Whisper-small仅LID、rho<1名义可行性，池SMALL6/LARGE7；与本轮49140MiB、K=1、A/B/C完整结构域不同。历史速度或Regret仅作背景，不混入同域均值。保留33 Gate。G-only Retention：P0/P1/P2池不同、LOW18与比例缩放开发负载；有效6单元最快启发式同最优，最低1/8负载缺LID预测样本、U=null，不是成功保持或0Regret。仅作为负结果背景，不并入主12单元统计。

严格组件消融的边界：冻结35的K1任务束DP没有候选扩展开关；A不能在保持全局部署搜索机制不变的情况下直接恢复20A MILP候选机制。A采用固定旧部署的候选覆盖诊断；只替换完整路由候选集合，旧生成器的完整候选和时间均保留，不能叫作全局单变量消融。B固定35完整部署，以同一DP核心限制到合法最快路由，保持驻留闲置模型显存计费，避免删模型产生多变量差异。它检验路由组件，不检验全局部署搜索；全局对照由既有最快路径基线提供。C无可关闭的共置修正，跳过。未开发新预测器或优化器。

GPU：直接复用34的A_TEXT10、B_TEXT10、C_ASR32B三个冻结同域配对，逐条核验35/最快/20A/V3完整配置等价、请求轨迹hash和内容、240秒窗口、30秒预热、49140MiB、原U和稳定性Gate。每个方法别名共享同一配置测量，不计作新的独立重复。复用34原配对顺序与多重校正。35 B→C跨会话收益保留为扩池背景，候选域与顺序控制不相同，不写为同域算法优势。GPU未新增测量。

求解效率：`solver_efficiency.csv`同时保存历史后端时间、新U-only Oracle三次冷缓存计时、固定部署路由子过程时间。U-only枚举去掉J诊断，预测公式与硬约束不变。状态单位不同：DP前缀label、Oracle完整路线leaf、MILP生成候选和内部节点不能互换。未记录的MILP内部节点、原生J调用数为null，不伪造。质量是历史能力准入，不等于重新测量质量分数。原20A `UTILITY_MODEL_INSUFFICIENT` Gate保持有效。

计数补充：V3网格全部12单元的实际J代理评价数已按冻结候选日志和共享缓存键重建，严格检查旧计数为同一缓存的两倍；见 `actual_J_count_audit.json`。原生J内部调用仍未知。20A/V3网格的同批次历史有限U+J枚举加速比另列 `historical_speedup_vs_finite_U_plus_J_oracle`，不与新U-only重复计时混为同一数字。
