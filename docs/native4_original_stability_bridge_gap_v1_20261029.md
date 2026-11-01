# Native4 → original stability schema：可行范围与原始字段缺口

2026-10-29。只读源审查；没有读取当前 live raw，没有实现桥，没有改变 pilot report。

原 [stability.reconstruct](/root/autodl-tmp/generalist-specialist-project/gsdeploy/stability.py:135) 可在独立投影 schema 上复用；阈值必须保持 [DEFAULT_THRESHOLDS](/root/autodl-tmp/generalist-specialist-project/gsdeploy/stability.py:7)。完整181原公共 cohort 均保留，包括未提交、错误、未启动、drain 后晚回包；不能只投影成功请求。

| 原字段/语义 | 可投影来源与不可补造的内容 |
| --- | --- |
| actual_submit_time_ns | [deployment.py:158](/root/autodl-tmp/generalist-specialist-project/gsdeploy/runtime/deployment.py:158) 的 host send-start。不是 write completed、ACK 或 worker START。 |
| completion_time_ns、status | [deployment.py:270](/root/autodl-tmp/generalist-specialist-project/gsdeploy/runtime/deployment.py:270) 的原 RESULT reader receipt 与冻结成功状态。late_response_after_drain 不得升级 completed；ERROR 不抵销 successful-completion backlog。 |
| worker_queue_wait_ms | [deployment.py:296](/root/autodl-tmp/generalist-specialist-project/gsdeploy/runtime/deployment.py:296) 直接保留 worker_response.queue_wait_ms；[semantic_progress_worker_v1.py:133](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/semantic_progress_worker_v1.py:133) 是 worker start_unix − parent submit_unix 原站间差，不能拿 enqueue wait 或剩余 GPU work 替换。 |
| full standalone s0 | [stability.py:55](/root/autodl-tmp/generalist-specialist-project/gsdeploy/stability.py:55) 要求 catalog 对每 model/task 有完整 service_time_ms。W(t) 是未成功完成请求的完整 solo s0 之和；不扣 active age，不能以 strict reference-work 或残余量替代。缺 s0 则加权 backlog UNKNOWN。 |
| original drain duration | [runner.py:85](/root/autodl-tmp/generalist-specialist-project/gsdeploy/runtime/runner.py:85) 在 runtime.drain 返回后记录 perf clock，drain_seconds = (return − measurement_end)/1e9。native4 的 [dispatch_case:221](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/twoFIFO_public181_resource_execute_v1.py:221) 调 drain 后直接等固定 deadline，没有保存返回时间。fixed120、observation_finalized、最后 RESULT 均不能认证原 drain_seconds，不能填0或120。 |
| timeout | [stability.py:205](/root/autodl-tmp/generalist-specialist-project/gsdeploy/stability.py:205) 检查原 error 的 drain timeout 字面证据。保留原错误与实际 fixed-cutoff 不同语义；不能重写原错误诱导分类。 |
| errors、shutdown | [stability.py:186](/root/autodl-tmp/generalist-specialist-project/gsdeploy/stability.py:186) 合计 measured/warmup/priming 错误；[stability.py:238](/root/autodl-tmp/generalist-specialist-project/gsdeploy/stability.py:238) 需要全部4个真实 worker 的 forced/exit_code/BYE。缺项不能填 clean、[]、0。 |

原 [classify](/root/autodl-tmp/generalist-specialist-project/gsdeploy/stability.py:293) 跳过 None 的 drain warning，因此直接调用所得 steady 不能当完整稳定性认证。桥必须另外输出字段覆盖与 partial/UNKNOWN 资格；不能让 full-U 补齐 stability 的原始缺字段。

未来可记录真实 drain-return 时间或可认证区间，再检查原分类在全区间是否不变；这不等于当前字段已存在。若其他已知警告已固定 recoverable 类别，可另外报告条件分类及不完整 reason 集，仍不得伪造原始 duration。当前不实现桥，不反馈或改写原 pilot report。
