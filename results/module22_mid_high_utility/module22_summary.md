# Module 22：Independent MID/HIGH Utility-Gain GPU Verification

**MID：`MID_POSITIVE_UTILITY`；HIGH：`HIGH_NO_BENEFIT`。**

本轮新 GPU session 12/12，正式测量 12 次。全部结果及未执行原因保留。LOW 未修复、未通过；不能宣称完整三阶段正效用已验证。

## 七个直接回答

1. **MID 增加 MMS 是否真实提高系统效用？** 本次三组可比配对均为正，平均 ΔU=3.461910，通过预冻结观测判据；n=3 的探索性结果不构成普遍收益保证。
2. **HIGH G/S 是否真实优于 S-only？** 没有通过本轮收益判据，平均 ΔU=-0.550405；保留全部负结果/低于收益门槛的结果。
3. **效用来源？** MID 平均 ΔU_mean=+0.983418、ΔU_P95=+2.507473、ΔU_throughput=+0.000185、ΔU_memory=−0.029167；HIGH 分别为 −0.118524、−0.431878、0、−0.000003。MID 的延迟改善超过新增显存成本；HIGH 损失主要来自 P95，与显存/吞吐基本无关。
4. **GPU 服务时间偏差？** HIGH Qwen LID 服务为 382.600 / 344.848 / 378.188ms，冻结 s0=139.909ms，高出 146.5%–173.5%；HIGH G/S 下 MMS 为 60.653 / 59.055 / 58.169ms，S-only 为 57.353 / 55.467 / 54.574ms，冻结 s0=52.266ms。所有路径的完整误差在下表及 CSV，不建立可迁移倍率。
5. **主要排队 worker？** MID G-only 对照的 Qwen 平均等待为 510.899 / 1050.776 / 595.210ms，三次均可恢复拥塞。HIGH G/S Qwen 为 489.849 / 295.717 / 232.949ms，前两次可恢复拥塞，第三次 steady；HIGH S-only 全部 worker steady。MMS 的卸载排队差距很小且不一致，见下方匹配请求归因。
6. **是否继续修复 LOW？** 模型池具有本次 MID 部署收益证据，但不建议继续当前 LOW 候选，也不建议延长当前 HIGH 8.456165% 候选。本轮不修复 LOW；没有证据表明增加样本或修正画像就会使 LOW 转正。
7. **进一步研究收益分流的依据？** 当前固定负载/模型池/分流点没有正效用分流的实测依据：三次差值均为负，且 S-only 全部 steady，没有容量可行性收益。画像与共享队列误差值得研究，但不等于已有有收益分流方案；其他负载/路由未验证，需要另行独立预注册，不能自动进入 Module23。

## 硬件预检与独立比较范围

当前设备是另一张 RTX4090，型号、49140MiB 总容量和 595.71.05 driver 与历史记录一致，物理 UUID 不同。第一次仅 NVML 预检被历史 UUID guard 拦截，**在任何模型/worker 启动前停止，没有产生 GPU repetition 或请求观测**。原清单、预检日志、失败及硬件更新均完整保留；首个正式 GPU session 之前仅更新硬件绑定，模型、请求、路由、预算、质量及效用没有改变。

全部十二次正式 session 使用 `GPU-31f2671b-99c5-ef57-be45-5c9a871e11f4`，所有新配对在同一物理卡上。不能以本轮 MID/HIGH 与历史 LOW 的跨卡绝对时延直接作因果比较。证据见 `hardware_preflight_amendment.json` 和 `frozen_experiment_manifest.prelaunch_v1.json`。

## 冻结实验与 Stage A

五模型池、B3=32019.2544MiB、质量硬准入、六任务 QPS 与 Module21 完全相同。Text/Image/OCR/ASR/VQA 为 4/2/2/0.05/0.2 QPS，MID/HIGH LID 为 3/12 QPS。
直接读取 Module21 fine CPU winner/control，无重新优化、比例调整或清零数值残余路由。MID Proposed 的 LID G 份额为 5.1283318e−8；HIGH 为 0.08456165089208065（8.456165089%）。
HIGH 两边模型/副本完全相同，非 LID 路由完全相同；唯一路由决策差异为 LID G/S vs MMS S-only。MID 对照不驻留 MMS，CPU 名义可行。
CPU 预测 MID ΔU=0.595139436；HIGH 冻结 fine ΔU=0.043003616。用户给定模拟参考 +0.04095 亦保留，但实际检验用真实请求和原效用，不把模拟数值当作结果。

同一配对重用 Module21 尚未执行的冻结 Poisson task/sample/offset/uniform 轨迹。三个 seed 21101/21202/21303，fresh workers，顺序 P/C、C/P、P/C。10s warmup、120s measurement、60s drain 上限、2s idle；不延长、不重试、不删 repetition。
MID 效用不通过不取消 HIGH。第一项持续 worker/system unstable 或未完成 cohort 停止该阶段剩余配对；运行/发生器/隔离/显存/释放安全错误停止 campaign。
成功判据在首个 GPU session 之前冻结：MID 三组正常可比 ΔU 均>0、Proposed steady；HIGH 三组正常可比方向为正、均值>0.01、两边 steady、每组均有真实 G/S 完成分流。95% CI 报告，不增加事后 CI gate。MID 三组均≤0、HIGH 三组均≤0.01 分别记 NO_BENEFIT；混合方向或无有效比较记 INCONCLUSIVE。

## 原定义与可比性

`U=−0.25*T_loss−0.30*L_norm−0.30*P_norm−0.15*M_norm`。质量为冻结准入硬约束。L/P 用冻结 offered QPS 权重和 Module20A 原任务参考；pooled mean/P95 单独报告。
`T_loss=1−min(1,completed/offered,completed_in_window/(fixed_offered_QPS*120))`；drain 不消除窗口吞吐损失。未完成/持续不稳定时 U=null，完成子集的效用分项只作截尾诊断。
safe memory 使用原 safe_factor×max(postwarm idle mean, postmeasurement idle mean)/预算；NVML owned peak 另核验。safe 是原安全余量画像，不是未采样瞬时峰值保证。
accepted 指 actual_submit 时间存在的 host 提交记录，不是独立 worker admission ACK。busy 是已完成 handler 的窗口 wall fraction，截尾时为下界，不等于 GPU SM utilization。

MID 对照的 Qwen 三轮为 recoverable_congested，完整 cohort 完成且队列清空；原 Module20A 允许已确认恢复的可恢复拥塞产生有限 U。持续 unstable/未完成 cohort 才是 U=null，绝未把失效对照改成正常收益。HIGH 前两轮 Proposed 同样记录可恢复局部拥塞，第三轮 steady；S-only 三轮全部 steady。

MID 配置保留原优化器背景路由极小数值残余，因此 JSON 的 non_LID_routes_identical=False；逐请求实测背景 worker 分配三组全部一致。HIGH 配置和逐请求的非 LID 分配均严格相同。配对的输入分布、预热轨迹、120s 窗口均已核验。

## 配对真实效用与分量

|阶段|rep|ΔU|ΔU_mean|ΔU_P95|ΔU_throughput|ΔU_memory|正常可比|可行性收益另计|
|---|---:|---:|---:|---:|---:|---:|---|---|
|MID|1|2.753150|0.731447|2.049274|0.000000|-0.027571|True|False|
|MID|2|4.802472|1.353084|3.480962|0.000556|-0.032129|True|False|
|MID|3|2.830108|0.865724|1.992184|0.000000|-0.027800|True|False|
|HIGH|1|-0.555367|-0.126876|-0.428491|0.000000|0.000000|True|False|
|HIGH|2|-0.465696|-0.112704|-0.352982|0.000000|-0.000010|True|False|
|HIGH|3|-0.630152|-0.115992|-0.514160|0.000000|0.000000|True|False|

|阶段|可比配对 n|均值 ΔU|SD|范围|nominal paired 95% CI|Gate|
|---|---:|---:|---:|---|---|---|
|MID|3|3.461910|1.161598|[2.753149807032445, 4.802471801976303]|[0.5763399521766686, 6.347479760315757]|MID_POSITIVE_UTILITY|
|HIGH|3|-0.550405|0.082340|[-0.6301518451254338, -0.4656964654416713]|[-0.7549487874399022, -0.34586149449166903]|HIGH_NO_BENEFIT|

主要统计单位是独立 GPU repetition 配对，最多 n=3；不把同 run 的请求当成独立部署实验。df=2 nominal CI 为探索性、未作多阶段校正，固定小池循环输入限制总体推广。窗口不足时只报告证据不足。

## HIGH 为什么没有真实收益

三次 ΔU 为 **−0.555367 / −0.465696 / −0.630152**，均值 **−0.550405**，nominal paired 95% CI **[−0.754949, −0.345861]**。用户给定模拟参考 +0.04095 没有出现；相对冻结 fine 预测 +0.043003616，实际均值误差为 **−0.593408757**。这只检验了当前固定点，不能推广到所有可能的 G/S 部署。

MMS worker 平均排队：G/S 为 **100.233 / 63.516 / 57.371ms**，S-only 为 **105.703 / 62.339 / 54.920ms**；P95 分别为 **429.190 / 249.826 / 219.738ms** 与 **392.552 / 248.154 / 214.693ms**。没有一致的 S-only 排队劣势；卸载后 MMS 的 P95 等待三次都略高。

逐 request_id、sample、offset 匹配两部署的完整 LID cohort，将 **mean 差异**精确分成转交 G 的请求和两边都由 S 处理的请求。它是 run 内算术归因，不把请求当作独立部署实验，也不对 P95 做错误的加性分解。

|rep|实际 G 完成份额|转交 G 的请求 mean E2E 增量 ms|两边同由 MMS 处理的请求 mean E2E 增量 ms|G 子集对 ΔU_LID_mean 的贡献|MMS 子集贡献|
|---|---:|---:|---:|---:|---:|
|1|7.6094%|+738.751|−3.190|−0.131055|+0.006872|
|2|8.6445%|+498.444|+4.448|−0.100453|−0.009474|
|3|8.5341%|+495.962|+6.127|−0.098675|−0.013065|

共享 MMS 子集在 G/S 中实际服务时间比对应 S-only 请求平均多 **3.300 / 3.386 / 3.741ms**；队列变化为 **−6.502 / +1.064 / +2.374ms**。第一轮卸载带来的少量队列改善，仍远小于转交 Qwen 的服务与等待代价。后两轮没有 MMS 平均改善。HIGH LID 的 mean+P95 效用平均差为 **−0.529069**，是总损失的主要来源；全部六任务均保留原权重，未因 QPS 小而忽略。

Qwen busy 为 **0.572 / 0.512 / 0.524**（G/S）对 **0.230 / 0.177 / 0.139**（S-only；精确值以 worker CSV 为准）。前两轮 G/S 的局部拥塞均最终恢复；第三轮 G/S 全部 steady 仍损失效用，所以结论不依赖截尾或失效对照。S-only 三轮均 steady，完整 cohort 全部完成，本轮没有额外的容量可行性收益。

上述 matched-request 差值包括所有运行条件和共置影响，只是实测误差与 mean 归因，**不是拟合/认证了可迁移的共置修正模型**。精确数据在 `task_and_split_diagnosis.json`、`per_task_paired_differences.csv` 和 `actual_path_service_vs_prediction.csv`。

## 每次完整系统结果

|run|offered/completed|窗口完成 QPS|pooled mean ms|pooled P95 ms|L_norm|P_norm|safe MiB|U|state|实际 LID G 完成份额|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
|mid_proposed_rep1|1386/1386|11.550000|42.061900|94.312154|1.181381|1.849925|28635.593|-1.043540|steady|0.000000|
|mid_control_rep1|1386/1386|11.550000|236.530209|1235.354781|3.619538|8.680837|22750.289|-3.796690|steady|1.000000|
|mid_control_rep2|1340/1340|11.141667|384.292802|2402.589413|5.683944|13.485230|22930.247|-5.860580|recoverable_congested|1.000000|
|mid_proposed_rep2|1340/1340|11.166667|37.124347|87.589255|1.173665|1.882023|29788.660|-1.058109|steady|0.000000|
|mid_proposed_rep3|1332/1332|11.100000|36.506288|86.826896|1.205721|1.946190|28677.806|-1.083253|steady|0.000000|
|mid_control_rep3|1332/1332|11.100000|249.544245|1450.611337|4.091466|8.586804|22743.624|-3.913361|recoverable_congested|1.000000|
|high_proposed_rep1|2491/2491|20.758333|152.943320|593.365961|2.316507|6.092258|28635.593|-2.656778|steady|0.076094|
|high_control_rep1|2491/2491|20.758333|119.134491|402.119261|1.893586|4.663955|28635.593|-2.101411|steady|0.000000|
|high_control_rep2|2445/2445|20.375000|88.067584|276.945703|1.521042|3.409535|29788.660|-1.618723|steady|0.000000|
|high_proposed_rep2|2445/2445|20.375000|117.656487|397.361647|1.896721|4.586143|29790.881|-2.084420|steady|0.086445|
|high_proposed_rep3|2453/2453|20.441667|109.748794|380.029890|1.851897|4.808031|28677.806|-2.132325|steady|0.085341|
|high_control_rep3|2453/2453|20.441667|80.489067|254.695071|1.465258|3.094164|28677.806|-1.502173|steady|0.000000|

## 实际服务相对画像

|阶段/arm/路径|s0 ms|各 run mean service ms|服务倍率范围|平均服务误差 ms|
|---|---:|---|---|---:|
|MID/proposed/asr/qwen#0|654.078|843.048 / 973.552 / 915.999|1.289–1.488|256.788|
|MID/proposed/visual_qa/qwen#0|144.507|187.617 / 178.627 / 194.916|1.236–1.349|42.546|
|MID/proposed/text/roberta_sst2#0|5.506|13.167 / 13.511 / 14.121|2.391–2.564|8.093|
|MID/proposed/image/convnext_tiny#0|5.848|15.760 / 16.083 / 15.370|2.628–2.750|9.890|
|MID/proposed/ocr/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0|10.436|20.793 / 19.277 / 19.603|1.847–1.992|9.455|
|MID/proposed/lid/facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a#0|52.266|55.179 / 55.299 / 55.286|1.056–1.058|2.989|
|MID/control/asr/qwen#0|654.078|682.822 / 884.848 / 784.549|1.044–1.353|129.995|
|MID/control/visual_qa/qwen#0|144.507|144.217 / 152.964 / 164.648|0.998–1.139|9.435|
|MID/control/lid/qwen#0|139.909|229.297 / 285.832 / 283.355|1.639–2.043|126.252|
|MID/control/text/roberta_sst2#0|5.506|13.266 / 12.450 / 13.503|2.261–2.452|7.567|
|MID/control/image/convnext_tiny#0|5.848|14.670 / 14.856 / 16.153|2.509–2.762|9.379|
|MID/control/ocr/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0|10.436|20.388 / 18.688 / 17.489|1.676–1.954|8.419|
|HIGH/proposed/asr/qwen#0|654.078|1366.645 / 1437.779 / 1152.844|1.763–2.198|665.011|
|HIGH/proposed/visual_qa/qwen#0|144.507|360.635 / 359.694 / 395.555|2.489–2.737|227.454|
|HIGH/proposed/lid/qwen#0|139.909|382.600 / 344.848 / 378.188|2.465–2.735|228.636|
|HIGH/proposed/text/roberta_sst2#0|5.506|14.637 / 13.866 / 15.405|2.518–2.798|9.130|
|HIGH/proposed/image/convnext_tiny#0|5.848|16.405 / 16.468 / 12.495|2.137–2.816|9.275|
|HIGH/proposed/ocr/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0|10.436|24.305 / 23.129 / 22.936|2.198–2.329|13.021|
|HIGH/proposed/lid/facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a#0|52.266|60.653 / 59.055 / 58.169|1.113–1.160|7.027|
|HIGH/control/asr/qwen#0|654.078|1384.751 / 1576.449 / 1178.576|1.802–2.410|725.847|
|HIGH/control/visual_qa/qwen#0|144.507|439.648 / 434.258 / 424.188|2.935–3.042|288.191|
|HIGH/control/text/roberta_sst2#0|5.506|14.916 / 14.498 / 14.935|2.633–2.712|9.276|
|HIGH/control/image/convnext_tiny#0|5.848|16.403 / 14.775 / 15.460|2.527–2.805|9.698|
|HIGH/control/ocr/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0|10.436|23.345 / 22.325 / 21.961|2.104–2.237|12.108|
|HIGH/control/lid/facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a#0|52.266|57.353 / 55.467 / 54.574|1.044–1.097|3.532|

## Worker 队列、busy 与稳定性

|run/worker|state|busy|queue mean ms|queue P95 ms|peak backlog|end backlog|tail slope rps|after drain|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|mid_proposed_rep1/qwen#0|steady|0.1234|119.263|471.799|3|0|-0.005916|0|
|mid_proposed_rep1/roberta_sst2#0|steady|0.0539|0.499|0.650|2|0|0.000068|0|
|mid_proposed_rep1/convnext_tiny#0|steady|0.0320|0.591|0.377|2|0|0.000225|0|
|mid_proposed_rep1/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0|steady|0.0405|0.457|0.394|3|0|-0.000492|0|
|mid_proposed_rep1/facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a#0|steady|0.1747|5.170|38.826|4|0|0.001368|0|
|mid_control_rep1/qwen#0|recoverable_congested|0.8244|510.899|1771.262|11|0|0.042949|0|
|mid_control_rep1/roberta_sst2#0|steady|0.0543|0.462|0.680|2|0|0.000123|0|
|mid_control_rep1/convnext_tiny#0|steady|0.0298|0.564|0.306|2|0|0.000069|0|
|mid_control_rep1/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0|steady|0.0398|0.623|0.298|3|0|-0.000514|0|
|mid_control_rep2/qwen#0|recoverable_congested|0.8864|1050.776|3079.084|13|3|-0.427163|0|
|mid_control_rep2/roberta_sst2#0|steady|0.0500|0.470|0.297|2|0|-0.000333|0|
|mid_control_rep2/convnext_tiny#0|steady|0.0321|0.544|0.380|2|0|0.000033|0|
|mid_control_rep2/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0|steady|0.0350|0.822|3.969|3|0|-0.000898|0|
|mid_proposed_rep2/qwen#0|steady|0.0889|67.522|516.794|3|0|-0.004769|0|
|mid_proposed_rep2/roberta_sst2#0|steady|0.0543|0.495|0.459|3|0|-0.000516|0|
|mid_proposed_rep2/convnext_tiny#0|steady|0.0347|0.743|0.585|2|0|-0.000178|0|
|mid_proposed_rep2/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0|steady|0.0361|0.746|3.062|3|0|-0.000646|0|
|mid_proposed_rep2/facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a#0|steady|0.1571|5.742|44.587|3|0|-0.000095|0|
|mid_proposed_rep3/qwen#0|steady|0.0859|54.779|144.858|2|0|-0.000517|0|
|mid_proposed_rep3/roberta_sst2#0|steady|0.0578|0.670|0.579|2|0|-0.000589|0|
|mid_proposed_rep3/convnext_tiny#0|steady|0.0313|0.615|0.667|2|0|0.000548|0|
|mid_proposed_rep3/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0|steady|0.0389|0.947|0.432|3|0|-0.001317|0|
|mid_proposed_rep3/facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a#0|steady|0.1530|6.029|49.188|3|0|-0.001111|0|
|mid_control_rep3/qwen#0|recoverable_congested|0.8572|595.210|1719.102|9|0|0.102988|0|
|mid_control_rep3/roberta_sst2#0|steady|0.0552|0.608|0.991|2|0|-0.000751|0|
|mid_control_rep3/convnext_tiny#0|steady|0.0328|0.724|1.419|2|0|0.000544|0|
|mid_control_rep3/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0|steady|0.0347|0.846|0.332|3|0|-0.001272|0|
|high_proposed_rep1/qwen#0|recoverable_congested|0.5721|489.849|1661.200|7|0|0.021199|0|
|high_proposed_rep1/roberta_sst2#0|steady|0.0599|0.505|0.855|2|0|0.000444|0|
|high_proposed_rep1/convnext_tiny#0|steady|0.0334|0.582|0.284|2|0|0.000168|0|
|high_proposed_rep1/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0|steady|0.0474|0.709|0.317|3|0|-0.000154|0|
|high_proposed_rep1/facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a#0|steady|0.6935|100.233|429.190|13|0|0.075000|0|
|high_control_rep1/qwen#0|steady|0.2301|243.429|1155.850|3|0|-0.011245|0|
|high_control_rep1/roberta_sst2#0|steady|0.0610|0.585|1.326|3|0|0.000049|0|
|high_control_rep1/convnext_tiny#0|steady|0.0334|0.478|0.285|2|0|0.000074|0|
|high_control_rep1/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0|steady|0.0455|0.679|0.268|3|0|-0.000245|0|
|high_control_rep1/facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a#0|steady|0.7097|105.703|392.552|13|0|0.037101|0|
|high_control_rep2/qwen#0|steady|0.1765|199.067|1044.221|3|0|-0.014059|0|
|high_control_rep2/roberta_sst2#0|steady|0.0582|0.568|0.375|3|0|-0.000437|0|
|high_control_rep2/convnext_tiny#0|steady|0.0319|0.675|0.308|2|0|-0.000635|0|
|high_control_rep2/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0|steady|0.0419|1.195|7.209|3|0|-0.001209|0|
|high_control_rep2/facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a#0|steady|0.6684|62.339|248.154|10|0|-0.030365|0|
|high_proposed_rep2/qwen#0|recoverable_congested|0.5120|295.717|1175.824|5|0|-0.038333|0|
|high_proposed_rep2/roberta_sst2#0|steady|0.0557|0.535|0.308|3|0|-0.000552|0|
|high_proposed_rep2/convnext_tiny#0|steady|0.0355|0.791|0.827|3|0|-0.000208|0|
|high_proposed_rep2/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0|steady|0.0434|1.276|6.882|3|0|-0.001584|0|
|high_proposed_rep2/facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a#0|steady|0.6501|63.516|249.826|10|0|-0.012514|0|
|high_proposed_rep3/qwen#0|steady|0.5240|232.949|895.386|5|0|-0.014717|0|
|high_proposed_rep3/roberta_sst2#0|steady|0.0630|0.762|3.113|2|0|-0.000945|0|
|high_proposed_rep3/convnext_tiny#0|steady|0.0254|0.500|0.337|2|0|0.000391|0|
|high_proposed_rep3/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0|steady|0.0455|1.471|1.520|3|0|-0.001220|0|
|high_proposed_rep3/facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a#0|steady|0.6442|57.371|219.738|7|0|-0.015499|0|
|high_control_rep3/qwen#0|steady|0.1394|139.609|1147.597|3|0|-0.001229|0|
|high_control_rep3/roberta_sst2#0|steady|0.0611|0.730|3.061|2|0|-0.000593|0|
|high_control_rep3/convnext_tiny#0|steady|0.0314|0.610|1.831|2|0|0.000441|0|
|high_control_rep3/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0|steady|0.0436|1.354|0.751|3|0|-0.001237|0|
|high_control_rep3/facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a#0|steady|0.6608|54.920|214.693|9|0|-0.014357|0|

Qwen 局部状态及 S-only 排队差距必须按以上实测记录判断；CPU rho 不等于 GPU 稳定性。HIGH 在同驻留条件下仍可能有实际显存 allocation 差异，该分项真实保留，不能假定 ΔM=0。

## 停止、遗漏与完整性

Stage stops：`{}`。
Global safety stop：`null`。
未执行：`[]`。

历史文件 15641 个，hash 未变=True；冻结实现未变=True；轨迹逐记录匹配=True；时间方程最大误差 ms=0.0；GPU release 全部验证=True。

共 22894 个正式请求全部 offered/submitted/completed，其中 22891 个在窗口内完成；仅 MID control rep2 的 3 个在 drain 中完成，吞吐损失保留。没有未完成或截尾 cohort。九项要求输出、72 行任务数据、57 个 fresh worker 及 12 次独立 U 重算均通过 `final_validation.json`。GPU 最终 used=1MiB，compute process list 为空。

模型池、效用、准入、预算及 QPS 未更改。未进入 Module23，完成后停止。

逐任务全指标在 `per_task_e2e.csv`，逐 worker 在 `worker_queue_and_service.csv`，各分项/公平性在 `system_utility_decomposition.csv`，精确统计在 `paired_utility_statistics.json`。原始记录与 telemetry 保留于 `raw_runs/`。
