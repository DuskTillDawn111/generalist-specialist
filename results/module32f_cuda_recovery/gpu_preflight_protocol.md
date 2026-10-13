# Module32F工程协议（GPU应用回归前冻结）

仅修复CUDA启动和工程验收。基础CUDA失败即停止，不重装驱动。逐层诊断采用干净exec；Landlock旧规则、seccomp单独、原组合与仅增加当前/proc/self/task WRITE_FILE的修复规则分别验证。CUDA thread comm是procfs线程命名控制，不是科学文件/临时输出；普通写入仍只准Module32F，设备/IPC例外保持。supervisor只为未来子PID提供procfs write-existing envelope，worker在原backend导入前强制交集收窄到自身task树，Python审计只允许自身task/*/comm；不修改历史权限或文件。

原subprocess.Popen+exec和顺序READY不改，不引入fork后CUDA复用，不同时创建六个context。控制器不导入torch、不初始化CUDA。先原PARSeq单模型，随后原六模型逐个独立fresh进程启动及每支持路径3次原始输入probe，每次清洁退出和显存释放；再六worker完整共置及原路径probe/驻留预算核验，清洁退出。其后S-only和Dynamic各180秒短时工程窗口，使用32D已冻结的同一180秒工程输入/到达流（seed320490），不是新筛选轨迹；300秒原S-only六任务真实调理、30秒原warmup、原60秒完整排空和热状态恢复均沿用。180秒是32D原工程风险预算周期，没有改策略参数；原policy逻辑使用当前窗口长度，不事后延长至产生分流。Dynamic必须有实际Phi和Whisper ASR执行才能完整验收，零分流不当作恢复协作验证成功。

前置38°C连续3次/间隔2秒/上限300秒，驻留预热后40°C连续3次/上限180秒；不改功率、频率、风扇、权重、推理、任务映射、QPS或预算。0.5秒记录实际SM/显存频率、温度、Pstate、功率、风扇和热降频标志，10秒实际fsync日志。原control perf_counter墙钟包括state_lock等待和错误状态检查，max(check,publish)<10ms；任一超限立即停止，保留完整planned cohort，不能以空样本0验收。Dynamic完整lock+projection+decide+enqueue P95≤1ms，原CPU 1381状态逐字段等价性及真实GPU因果状态回放保持。全部结果正负保留，无正式U计算、无长窗口、无策略调参。

诊断前/应用回归前/短时预检前/结束分别对历史目录+原权重/源码/输入做完整SHA；不在请求测量时运行大文件哈希。所有输出/缓存/临时文件在Module32F。32D旧progress非字节恢复例外只留记录，本轮不改32B–32E或其他历史科学输入及Gate。应用回归/预检失败停止本轮，不替换种子，不事后调策略、门槛或时长。
