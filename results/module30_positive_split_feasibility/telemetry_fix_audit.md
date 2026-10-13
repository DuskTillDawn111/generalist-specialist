# 遥测修复审计

Module29在1997.196239秒记录dictionary changed size during iteration；原始无堆栈，不能唯一识别具体触发语句。历史DurableMonitor对sender/collector正在修改的live request执行未加锁deepcopy；CPU普通字典并发实验重现同型异常，见legacy_race_reproduction.json。

新SafeMonitor在runtime.lock下深拷贝完整请求快照，退出锁后序列化/写盘；sample_lock串行后台与同步冷却采样、CPU状态、日志与phase切换，完成样本最后发布。先冻结完整planned cohort再启用日志；日志fsync成功后才提交persisted键，原子raw/progress使用唯一临时文件、fsync、replace与目录fsync。遥测/写盘异常保存堆栈并进入硬失败；不允许技术失败或不完整窗口算合法U。全请求期末与异常后均flush并核对journal；未到达槽也保留为incomplete。

新AuditedRuntime复用原native transport与推理，额外核验sender/collector/reader退出；先清洁关闭、按序接收RESULT/BYE/EXIT再标未完成，核验owned PID释放。CPU fixture验证200个请求无丢失且清洁退出。首次fixture协议前缀写错导致启动等待，修正为原READY|/RESULT|/BYE|协议后通过；初始日志保留，未启动GPU。最初默认Python缺scipy，改用项目已有gsdeploy环境；未安装依赖、失败日志保留。

2100秒CPU压力超过原故障时刻：两个持续请求字典更新线程、后台采样及同步采样并发；逐版本快照一致、JSONL完整、终态无重复/遗漏、线程退出。sensor异常、fsync失败与不退出collector注入均可见；写盘失败不提交persisted状态。数值结果见telemetry_stress_result.json。CPU soak使用假传感器，不冒充真实GPU画像；正式每次GPU同样检查遥测与journal完整性。

历史Module14F–29代码、错误、raw、Gate和权重均不覆盖；修复为module30/telemetry.py替换实现，旧冻结监测器保留为错误证据，当前Module不再调用其无锁请求拷贝。正式优化器不改。仅上述检查通过后才冻结当前NVML设备身份并开始GPU测量。

补充：在正式GPU前直接调用原Module29冻结DurableMonitor.sample，使用CPU传感器替身及持runtime.lock的更新线程，复现了原gpu_replication.py第51行未加锁outstanding deepcopy的dictionary changed size错误，完整堆栈见legacy_exact_method_reproduction.json。历史save被拦截、journal写入Module30 fixture，原文件SHA256未变。此定位证明原方法存在该竞态，原1997.196秒没有堆栈的单次异常仍不能唯一追溯到具体分支。

原生CPU运输的完整与强制中止窗口各80个请求，后者40完成/40未完成，全部80个终态保留且无重复；四线程320次原子写入无损坏，见cpu_harness_acceptance.json。原20A效用独立算式、journal不一致阻断U与unstable→NULL核验见cpu_analysis_acceptance.json。

## v2：修复初版控制路径锁阻塞

初版check和set_active使用整个采样锁，NVML/host/thermal I/O期间可能阻塞请求提交。这是本模块新增实现问题，不能当成模型的背景损失或分流收益。CPU慢传感器直接重现250ms check等待；完整第一窗口Text/Image/OCR平均schedule delay为34.5/45.3/54.7ms。原始时间戳与效用算式未篡改，但这些窗口不具备物理可行性比较资格。两个已启动窗口完整保留并计入6次预算，不补跑。

新SafeMonitorV2把控制状态和phase指针发布与串行采样/日志锁分离：check只读取短锁错误状态及线程安全Event；coherent_records先取得phase指针，退出状态锁后在runtime.lock下拷贝。所有I/O均在控制锁之外。NVML不能阻塞check或phase发布，request仍完整冻结并按原时间轨迹dispatch。

250ms传感器I/O期间12次check与phase发布均低于10ms，实测为微秒级。慢传感器下原生80请求完整/40完成40中止测试通过，全部80终态保留；完整窗口每请求schedule delay均低于20ms。重新运行2100秒带25ms传感器阻塞的并发压力，dict快照一致、日志无重复遗漏、线程退出、sensor/文件/退出故障注入可见，control check最大延迟低于10ms。精确值见对应v2 JSON。

仅上述检查通过且GPU空闲后，使用frozen_gpu_manifest_v2.json继续4个此前未执行槽。原配置、模型、seed、输入、分流比例及效用均不变；C1跨版本对照禁止，C2三臂可比较。原Module29 Gate和全部历史文件、初版manifest及数据保持不变。

## 精确复现与恢复长压证据

legacy_exact_method_reproduction.json 直接调用保留的 Module29 DurableMonitor.sample；CPU 并发写入使用原 runtime.lock，旧读者未锁，在 gpu_replication.py:51 的 deepcopy 原方法重现同型 RuntimeError，完整堆栈保留。原 1997.196 秒错误本身没有记录唯一堆栈，因此不声称还原当时线程的全部状态。

首轮 v2 慢传感器长压日志停于1920秒，进程已不在且没有通过结果，原因不明；它未被计为验收通过。日志、775MiB左右 journal及进程状态另存 interrupted_cpu_attempt/；cpu_pressure_interruption.json 记录恢复与仍未运行的四个冻结 GPU 槽。恢复后可见 cgroup 内存上限96GiB、OOM/oom_kill计数0，不能由此推定此前退出原因。

重新连续完成 v2 长压 2118.546294秒，7042340 次请求字典更新，74235 个采样，74301 条 journal，64个唯一终态。全部版本快照一致、无错误、线程退出；慢传感器下控制检查最大 0.070779ms。传感器错误、fsync失败不提交persisted、挂起collector退出故障均显式可见。只有该次完整通过后才开始剩余 GPU 测量。

C2最后5%槽曾在300秒启动前冷却检查失败，发生在任何模型加载、预热和正式窗口之前。原控制器停止记录完整保留。原任务规定最多六个正式窗口，目前五个完整；仅尝试补齐同一个此前未测量的第六槽，先额外空闲等待原≤38°C条件，最多900秒，原300秒检查、驻留≤40°C、模型池/副本/负载/seed/比例/输入/600秒/30秒预热/60秒排空均不改。此为明确记录的准备流程偏离，不是重复已测窗口；若原冷态条件仍不满足就终止，不调整温度阈值。详见 frozen_gpu_manifest_unmeasured_slot_completion.json（完成等待后写入）。

## 最终数据验收

额外空闲准备首末采样间隔 890.800秒，396个采样，最后3个均≤38°C，随后原300秒启动前及驻留≤40°C检查通过；无阈值放宽。第六窗口只执行一次，600秒，1162请求全完成。全轮6个正式600秒窗口、7038个正式请求全完成，warmup/正式输入轨迹以及native prime样本逐候选三臂完全相同。所有6个journal核对、native时间等式、线程/worker清洁退出及GPU释放通过；NVML owned-process显存全部数值有效。2个初版观察器窗口U仍为NULL，另4个合法U。详见final_data_acceptance.json。
