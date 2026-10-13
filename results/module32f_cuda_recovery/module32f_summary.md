# Module32F CUDA恢复结果

Gate：`GPU_PREFLIGHT_RECOVERED`。本轮只进行启动修复与短时工程预检，U=NULL；未运行正式效用验证、2400秒窗口或优化器集成。

## 根因与修复

基础CUDA在干净独立进程中可用。旧Landlock独立及原组合均复现Error304，seccomp独立正常；全线程syscall证据显示CUDA线程命名打开自身`/proc/self/task/<tid>/comm`被EACCES拒绝。不是PARSeq独有、fork后CUDA复用、显存不足或并发context故障的证据。给worker自身task树追加WRITE_FILE，父supervisor只提供继承envelope，worker导入backend前再次收窄。历史文件写入仍拒绝。

冷缓存首次OCR另暴露原生Triton跨目录rename被Landlock ABI1拒绝EXDEV。仅以同目录临时文件+原子replace实现已有cache provider接口；不改变编译器选项、native算子、推理参数或权重。版本1失败完整保留，版本2在任何短时预检之前重新冻结工程源码；没有根据U或预检结果改策略。

## 回归与限制

历史1381次同状态决策逐字段完全相同，数值容差0。CPU2400秒规模3419请求以50倍时间压缩，真实发送/收集/遥测/日志线程；1x/8x历史的路由P95分别0.131480/0.184576ms。保留全部长尾，包括8x单次路由最大30.602787ms；验收门槛是P95≤1ms，不能声称每次≤1ms或2400秒墙钟浸泡通过。

PARSeq与原六模型逐个初始化、各支持路径原输入探测、六模型完整共置与清洁释放结果为True。共置安全驻留25776.255MiB，原B3预算34567.553MiB。质量仅核验原准入及闭集输出合法、原权重/推理/任务映射不变，不是重新评估质量分数。

短时两臂保持原工程seed320490和输入/到达序列、原六任务QPS、单副本驻留、300秒实际六任务调理/30秒warmup/180秒工程测量/60秒有界全队列排空。原control墙钟包含状态锁等待，检查及phase publish最大值均须<10ms；非空真实在线Dynamic P95≤1ms。全请求/等待/服务/E2E/队列/遥测/worker退出及原策略回放证据见raw_runs、gpu_measurements与gpu_acceptance，不按不同温度均值推断降频。

| 臂 | 工程通过 | 完成/计划 | Phi/Whisper ASR | 路由P95 ms | control check max ms | 实测峰值MiB |
|---|---|---|---|---|---|---|
| E0_S | True | 281/281 | 0/92 | None | 0.09628012776374817 | 23222.0 |
| E0_D | True | 281/281 | 1/91 | 0.16566345000000002 | 0.08838344365358353 | 23224.0 |

历史SHA及size/mtime/symlink前后核验：True（15100条输入，差异0）。全部活动freeze源核验：True。32D不可逐字节恢复的旧progress差异仍保留prior_incident_record，本轮不伪称恢复，也未修改历史Gate。SHA重读安排在GPU测量之外，避免人为干扰。

工程有效性与效用结论分开：恢复成功只支持后续验证的工程准备，不证明Dynamic正收益或独立复现。完成本Module后停止。
