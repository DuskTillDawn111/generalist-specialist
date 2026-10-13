# 质量、时间与干扰证据边界

原LID策略SEMANTIC_VALIDITY_ONLY，准确率是report-only、没有数值阈值。本报告不发明新阈值，不追溯剔除错误，不恢复合法U。推理成功率与correct、WER、闭集语义有效性分别报告。

|运行/worker/task|请求n|correct观测n/正确|accuracy|WER观测n/mean|闭集flag观测n|
|---|---:|---:|---:|---:|---:|
|S_only/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt/ocr|138|138/138|1.0|0/UNKNOWN|0|
|S_only/convnext_tiny/image|122|122/122|1.0|0/UNKNOWN|0|
|S_only/S/asr|124|0/0|UNKNOWN|124/0.02868903646717612|0|
|S_only/S/lid|7005|7005/7005|1.0|0/UNKNOWN|0|
|S_only/G/visual_qa|105|105/99|0.9428571428571428|0/UNKNOWN|105|
|S_only/roberta_sst2/text|113|113/101|0.8938053097345132|0/UNKNOWN|0|
|Dynamic/baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt/ocr|138|138/138|1.0|0/UNKNOWN|0|
|Dynamic/convnext_tiny/image|122|122/122|1.0|0/UNKNOWN|0|
|Dynamic/S/asr|124|0/0|UNKNOWN|124/0.02868903646717612|0|
|Dynamic/S/lid|6938|6938/6938|1.0|0/UNKNOWN|0|
|Dynamic/G/lid|67|67/61|0.9104477611940298|0/UNKNOWN|0|
|Dynamic/G/visual_qa|105|105/99|0.9428571428571428|0/UNKNOWN|105|
|Dynamic/roberta_sst2/text|113|113/101|0.8938053097345132|0/UNKNOWN|0|

G LID 61/67=91.0448%，名义95%Wilson[0.818072,0.958309]。这67次只有27个不同sample_id；按输入cluster的探索bootstrap区间[0.766667,1.000000]（10000次、分析RNG=300030，不改变负载种子）。两种区间都受路由选择、重复样本与未验证可交换性限制，不是独立GPU复现或总体质量保证。

Frozen轨迹与两次raw的task/pool_id/sample_id/offset完整对上，可获得LID语言标签，聚合见quality_LID_by_language.csv。Dynamic原预测文本被脱敏；6个incorrect的实际预测类、解析失败/错语言等错误子类型UNKNOWN，不能构造混淆矩阵。S/G路由子集不相同，也不能用另一run的同ID回答恢复本次未执行的S反事实。

**缺失字段澄清：G LID的67条worker_response均未保留valid_closed_set_output。** 历史统计explicit_invalid=0只表示没有记录到False，不证明67条全都有效；本Module记closed_set_validity=UNKNOWN并保持历史文件原样。VQA有单独有效性flag，不能借用到LID。

时间分解使用同一formal origin：scheduled→enqueued为调度/观测/日志等host时间；enqueued→send为host sender排队；send→native handler start为worker_queue_wait_ms（包括pipe/transport，非纯worker等待）；handler start→end为服务墙钟（包含CPU/GPU，不是kernel时间）；end→reader response为传播。collector commit晚于reader收到，只有已commit且按冻结harvest逻辑可见报告才能进EWMA。上述分量逐请求和e2e相加核对，router开销属于host分量，不能再加一次。

动态原decide mean/P95=29.94/46.54µs；instrumented observation+decision=916.15/1690.08µs；完整arrival=977.00/1795.35µs。S-only约99µs状态检查不等于原decide或同样的完整插桩开销。warmup/prime与正式窗分开，cohort延迟可含排空完成，但正式窗口吞吐不含排空。

worker pending/accepted waiting队列缺receipt ACK，均UNKNOWN。host pending含发送队列、pipe、执行和报告传播；native未开始handler数量也只能作transport-inclusive代理。native执行时间是收到响应后离线映射，不能用它或之后的完成时间回填过去路由。

403对S/G执行重叠复核，包括382个S LID、21个S ASR；G串行handler意味着VQA不能与同worker G-LID同时执行。4个VQA在G-LID执行时提交，等待mean287.24ms，其余101个mean19.11ms；这是串行时序关联，不能将差值当附加因果延迟。无新增匹配S-only/fixed控制，输入、时段、温度、CPU共享、策略选择和服务时长选择偏差未控制。
