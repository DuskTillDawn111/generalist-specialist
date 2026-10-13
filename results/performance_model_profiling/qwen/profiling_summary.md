# Module 2B: Qwen feature-directed profiling

状态 measurement_complete；33 runs，11 configurations（11 完整三重复），207 dataset rows。
自适应停止后应执行 33 轮；实际未执行配置：['A_u_0.50', 'A_u_0.60', 'A_u_0.70']；覆盖检查完成=True。
A 扫描在 u=0.3 触发停止；deferred balanced anchor=A_u_0.40。该 anchor 若作为 B 对照补跑，不计为缺失或重复数据。
A: Qwen-only，固定已有六任务比例；B: u=0.4 改变 demand mix（balanced复用A）；C: 固定驻留模型和 Qwen u/q，仅改变竞争路径的流量。
所有 features 由 performance_model/features.py 计算，实际观察到的 cohort routing rates 也单独经相同 extractor 转换。
均值±sample std 描述三轮离散，不是置信区间。worker是handler墙钟；u为standalone service demand，非实测GPU利用率。
C 是已有 offline resource fingerprint 按路由 req/s 加权求和的特征，单位为 fingerprint 单位×req/s；不等同于实测GPU压力或利用率。
异常/不稳定 6 轮；含 incomplete 3 轮、Qwen incomplete 1408 请求；Qwen request error 0；成功样本不足30的轮次 3。
**延迟仅统计成功请求。** incomplete 仍保留在分母和原始记录；受 drain 截断的 P95 不是全部 offered 请求的 P95。异常轮次照常纳入描述表，不静默删除。

| Condition | reps | u_Q | Qwen worker ms | Qwen mean E2E ms | Qwen P95 ms | Queue ms | Qwen window rps | unstable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A_u_0.10 | 3 | 0.1000 | 195.64 ± 30.04 | 256.08 ± 43.42 | 837.75 ± 181.52 | 59.98 ± 25.85 | 0.79 ± 0.00 | none |
| A_u_0.20 | 3 | 0.2000 | 214.27 ± 101.50 | 526.45 ± 506.29 | 2265.66 ± 2607.14 | 311.75 ± 404.95 | 1.58 ± 0.00 | none |
| A_u_0.30 | 3 | 0.3000 | 217.77 ± 52.70 | 1204.99 ± 1189.86 | 4710.66 ± 4233.17 | 986.78 ± 1139.99 | 2.33 ± 0.04 | A_u_0.30_rep_3 |
| A_u_0.40 | 3 | 0.4000 | 231.25 ± 127.71 | 5782.33 ± 9481.86 | 18249.43 ± 29941.39 | 5550.64 ± 9354.11 | 2.97 ± 0.35 | A_u_0.40_rep_1 |
| A_u_0.50 | 0 | 0.5000 | — | — | — | — | — | none |
| A_u_0.60 | 0 | 0.6000 | — | — | — | — | — | none |
| A_u_0.70 | 0 | 0.7000 | — | — | — | — | — | none |
| B_text_dominant | 3 | 0.4000 | 161.26 ± 101.71 | 52906.02 ± 61001.00 | 102890.48 ± 103389.07 | 16725.29 ± 17702.00 | 5.28 ± 1.41 | B_text_dominant_rep_1 |
| B_vision_dominant | 3 | 0.4000 | 236.00 ± 174.94 | 23210.95 ± 39853.60 | 53760.45 ± 92168.10 | 13887.46 ± 23939.23 | 2.96 ± 0.57 | B_vision_dominant_rep_1 |
| B_audio_dominant | 3 | 0.4000 | 489.96 ± 255.04 | 18864.92 ± 23763.45 | 47813.72 ± 44884.77 | 18364.66 ± 23504.18 | 1.80 ± 0.40 | B_audio_dominant_rep_1, B_audio_dominant_rep_2 |
| C_no_active_contender | 3 | 0.4000 | 154.50 ± 8.69 | 286.89 ± 9.41 | 893.32 ± 78.45 | 132.08 ± 16.16 | 3.17 ± 0.00 | none |
| C_distilbert_text | 3 | 0.4000 | 158.01 ± 6.18 | 305.25 ± 22.60 | 965.75 ± 47.01 | 146.96 ± 18.47 | 3.17 ± 0.00 | none |
| C_whisper_asr | 3 | 0.4000 | 168.77 ± 10.45 | 323.63 ± 68.01 | 1033.65 ± 127.73 | 154.54 ± 57.98 | 3.17 ± 0.00 | none |
| C_paddleocr_ocr | 3 | 0.4000 | 160.66 ± 8.61 | 305.27 ± 17.17 | 959.40 ± 26.54 | 144.33 ± 20.10 | 3.17 ± 0.00 | none |

## 必须回答的问题

1. **u 与 latency 是否稳定单调？** 以下为相邻点三轮均值方向；seed级值与逐任务曲线见JSON/图，不能从整体均值推出每任务或每seed单调。
   mean_e2e_latency_ms: [256.08175762931035, 526.4514036431034, 1204.987114383908, 5782.326398502873]; nondecreasing=True。
   p95_e2e_latency_ms: [837.7516267833331, 2265.657974483333, 4710.656504516665, 18249.4334742]; nondecreasing=True。
   mean_worker_queue_wait_ms: [59.9824641045977, 311.747803662069, 986.7774555074712, 5550.643801928161]; nondecreasing=True。
   mean_worker_latency_ms: [195.63692645402298, 214.26722804885057, 217.76779447068967, 231.2549706270115]; nondecreasing=True。
   各seed整体mean是否单调：[(1101, False), (2202, False), (3303, False)]。三轮均值的上升不代表稳定的逐轮单调关系；重复间变化必须一起解释。

2. **latency knee？** 整体三轮均值的上升候选区间：[(0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]；首次停止点 u=0.3。候选规则是相邻mean或P95翻倍且queue增加至少50ms，逐任务结果见JSON。重复间差异大时不能据此确定单一可靠knee；不以rho=0.8预设阈值。

3. **相同 u 下 mix 差异？** balanced=A_u_0.40，与B三种mix的逐任务mean/P95比较见图B和下表。整体Qwen均值含任务比例变化，需同时看同一task。

| Mix | task | mean E2E ms | P95 ms | worker ms | queue ms |
| --- | --- | ---: | ---: | ---: | ---: |
| A_u_0.40 | text | 5673.29 ± 9469.98 | 18093.03 ± 29987.71 | 70.88 ± 42.41 | 5601.99 ± 9427.54 |
| A_u_0.40 | image | 5727.74 ± 9274.64 | 18091.09 ± 29771.21 | 211.09 ± 120.20 | 5516.23 ± 9154.38 |
| A_u_0.40 | ocr | 5795.40 ± 9621.51 | 18322.57 ± 30064.18 | 155.41 ± 96.81 | 5639.57 ± 9524.60 |
| A_u_0.40 | asr | 6226.41 ± 9360.25 | 17870.64 ± 28124.45 | 1216.89 ± 769.29 | 5009.05 ± 8591.81 |
| A_u_0.40 | visual_qa | 5844.52 ± 9584.39 | 17590.43 ± 29388.24 | 188.06 ± 86.17 | 5656.03 ± 9498.33 |
| A_u_0.40 | lid | 5877.58 ± 9560.16 | 18188.17 ± 30096.00 | 373.92 ± 165.33 | 5503.21 ± 9394.76 |
| B_text_dominant | text | 52876.03 ± 60981.80 | 102559.21 ± 103051.61 | 102.68 ± 63.85 | 16730.17 ± 17706.24 |
| B_text_dominant | image | 52768.44 ± 60755.18 | 102605.39 ± 103193.97 | 304.34 ± 191.17 | 16555.62 ± 17492.80 |
| B_text_dominant | ocr | 53145.09 ± 61322.83 | 103080.98 ± 104293.87 | 232.31 ± 151.88 | 16953.07 ± 17906.21 |
| B_text_dominant | asr | 52584.83 ± 59341.05 | 100874.78 ± 100014.21 | 1831.58 ± 1325.16 | 16049.95 ± 16925.43 |
| B_text_dominant | visual_qa | 52954.69 ± 61089.77 | 102316.98 ± 102452.44 | 247.47 ± 121.46 | 16834.58 ± 17887.86 |
| B_text_dominant | lid | 53274.28 ± 61423.50 | 103113.98 ± 104174.88 | 541.48 ± 351.74 | 16557.15 ± 17559.30 |
| B_vision_dominant | text | 23100.52 ± 39797.14 | 53273.45 ± 91604.69 | 85.47 ± 68.19 | 13974.22 ± 24070.27 |
| B_vision_dominant | image | 23200.71 ± 39822.30 | 53541.35 ± 91878.25 | 253.22 ± 193.26 | 13849.22 ± 23870.94 |
| B_vision_dominant | ocr | 23189.39 ± 39875.89 | 53589.21 ± 92030.38 | 189.75 ± 155.93 | 13837.96 ± 23852.01 |
| B_vision_dominant | asr | 23025.25 ± 38506.65 | 51615.49 ± 86673.92 | 1450.71 ± 1224.34 | 13375.26 ± 23082.51 |
| B_vision_dominant | visual_qa | 23249.33 ± 39911.21 | 53724.88 ± 92241.31 | 212.30 ± 116.75 | 13959.48 ± 24072.24 |
| B_vision_dominant | lid | 23458.29 ± 40102.15 | 53685.26 ± 92113.48 | 433.99 ± 319.19 | 13915.91 ± 24007.55 |
| B_audio_dominant | text | 18745.76 ± 23716.79 | 47530.80 ± 44846.31 | 93.92 ± 48.16 | 18642.05 ± 23654.89 |
| B_audio_dominant | image | 18593.57 ± 23923.92 | 45858.03 ± 44405.20 | 286.17 ± 156.07 | 18306.93 ± 23774.23 |
| B_audio_dominant | ocr | 18899.22 ± 23748.00 | 47193.20 ± 43687.95 | 225.93 ± 146.01 | 18672.83 ± 23604.27 |
| B_audio_dominant | asr | 19212.37 ± 23600.75 | 47977.40 ± 44617.61 | 1662.18 ± 905.82 | 17546.83 ± 22718.77 |
| B_audio_dominant | visual_qa | 18503.07 ± 23905.64 | 44748.36 ± 42976.66 | 234.47 ± 97.07 | 18242.49 ± 23768.09 |
| B_audio_dominant | lid | 18939.07 ± 23802.56 | 47989.96 ± 45129.33 | 489.31 ± 236.32 | 18434.08 ± 23565.77 |

同task在四种mix间的range（条件按实测端点描述，不是routing/deployment选择）：
| Task | highest/lowest mean ratio | range ms | largest within-condition std ms | positive paired seeds |
| --- | ---: | ---: | ---: | ---: |
| text | 9.320 | 47202.74 | 60981.80 | 2/3 |
| image | 9.213 | 47040.70 | 60755.18 | 2/3 |
| ocr | 9.170 | 47349.68 | 61322.83 | 2/3 |
| asr | 8.445 | 46358.42 | 59341.05 | 3/3 |
| visual_qa | 9.061 | 47110.17 | 61089.77 | 3/3 |
| lid | 9.064 | 47396.70 | 61423.50 | 2/3 |

3/4. **实测结论与 q 的必要性：** 同task的条件均值相差 8.45–9.32 倍；2/6 个task的端点差在三个配对seed上同向，0/6 个task的条件range大于最大的条件内sample std。
q 应保留为待验证的解释变量；当前的重复间波动、不同mix的测量窗口和输入序列、以及截断样本限制了归因。这些数据不能单独证明 q 已充分解释差异。

5. **固定u/q后 C 能否解释额外变化？** 配对seed的C条件减去同驻留环境无活跃竞争者的结果：

| C condition | mean E2E delta ms | worker delta ms | queue delta ms |
| --- | ---: | ---: | ---: |
| C_no_active_contender | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| C_distilbert_text | 18.36 ± 18.09 | 3.51 ± 4.97 | 14.88 ± 18.20 |
| C_whisper_asr | 36.74 ± 60.11 | 14.27 ± 18.11 | 22.46 ± 42.02 |
| C_paddleocr_ocr | 18.38 ± 8.75 | 6.16 ± 7.03 | 12.25 ± 6.84 |
C_distilbert_text：3 个配对seed中 3 个E2E增加；单轮delta=[39.20406785689653, 6.706351224137961, 9.168493903448223] ms。
C_whisper_asr：3 个配对seed中 2 个E2E增加；单轮delta=[9.563614279310343, -4.979488013793116, 105.64391121724134] ms。
C_paddleocr_ocr：3 个配对seed中 3 个E2E增加；单轮delta=[13.982692055172436, 12.698977862069, 28.453530901724093] ms。
已齐全的竞争条件中，配对平均E2E变化范围为 18.36 至 36.74 ms；三轮均增加的条件：['C_distilbert_text', 'C_paddleocr_ocr']。这只衡量固定u/q下观察到的附加变化，尚未验证 C 的解释充分性或预测能力。

C仍存在调度/样本混杂：原harness按task生成arrival后routing。同task加入竞争流量会改变Qwen请求间隔和sample subsequence；JSON保留每seed的样本序列相等检查及音频时长。因此不宣称因果识别。

6. **哪些resource dimensions最一致？** 只对四个condition均值计算探索性相关性：

| Feature | Spearman | Pearson |
| --- | ---: | ---: |
| offline_energy_j_per_request | 1.000 | 0.972 |
| kernel_time_ms_per_request | 0.800 | 0.929 |
| kernel_fraction_conv | 0.738 | 0.000 |
| kernel_fraction_other | 0.738 | 0.014 |
| kernel_fraction_attn | 0.400 | 0.103 |
| kernel_fraction_gemv | 0.400 | 0.155 |
| kernel_fraction_elemcopy | 0.200 | 0.012 |
| kernel_fraction_gemm | 0.200 | 0.002 |
| kernel_fraction_rednorm | 0.200 | 0.006 |
| offline_device_vram_used_mib | 0.200 | 0.027 |
| offline_gpu_util_pct | 0.200 | 0.054 |
| offline_memory_util_pct | 0.200 | 0.032 |
| offline_power_w | 0.200 | 0.028 |
| offline_sm_clock_mhz | 0.200 | 0.017 |

四个点、多个共线维度不能识别真正资源瓶颈，相关性不当作predictor或显著性证据。

7. **单独u→L baseline是否不足？** B 的相同u存在上表中的大幅实测差异，单独u无法描述本批数据的全部变化；但重复间波动说明加入q/C也尚未被证明充分。A插值只在固定mix/context内使用，不能外推到不同mix/co-location。
   本次piecewise baseline初始化：True；合格稳定点：['A_u_0.10', 'A_u_0.20']。原始异常点仍保留在完整结果中。

8. **是否足以开始cross-deployment predictor？** 尚不足以支撑可信的首版跨deployment训练/验证；本批是Qwen探索性数据基础。C只有一个固定驻留环境、三条单一竞争路径及单一压力强度，缺独立deployment holdout、多竞争者组合、输入长度控制和更多压力强度；还需解释重复间漂移与截断。此次不训练。

## Data quality and boundaries

| Condition | Qwen runtime ms | Qwen schedule delay ms |
| --- | ---: | ---: |
| A_u_0.10 | 255.82 ± 43.41 | 0.26 ± 0.01 |
| A_u_0.20 | 526.21 ± 506.26 | 0.24 ± 0.03 |
| A_u_0.30 | 1204.74 ± 1189.82 | 0.24 ± 0.04 |
| A_u_0.40 | 5782.10 ± 9481.83 | 0.23 ± 0.04 |
| B_text_dominant | 16886.75 ± 17803.35 | 36019.26 ± 43339.16 |
| B_vision_dominant | 14123.66 ± 24114.26 | 9087.29 ± 15739.34 |
| B_audio_dominant | 18854.86 ± 23746.86 | 10.06 ± 16.98 |
| C_no_active_contender | 286.71 ± 9.41 | 0.18 ± 0.01 |
| C_distilbert_text | 305.11 ± 22.59 | 0.14 ± 0.01 |
| C_whisper_asr | 323.46 ± 68.01 | 0.18 ± 0.00 |
| C_paddleocr_ocr | 305.13 ± 17.17 | 0.14 ± 0.01 |

此表runtime和schedule delay与E2E使用同一成功请求群体，便于分解；原harness的task schedule delay统计包含所有已提交请求，定义保留在raw。主机sender排队也可能放大E2E，不能把全部增长归为GPU worker queue。
原始结果含source hashes、seed、arrival phase、request records、错误/incomplete/drain；低于30成功样本的路径标记tail_sample_shortfall。
实验环境固定OMP/MKL/OPENBLAS=1，三次全新进程。A按负载升序，B/C每个重复内固定seed随机condition顺序；不能排除时间漂移。
没有枚举deployment、没有训练MLP/GBDT等模型、没有guided search/routing optimization。
