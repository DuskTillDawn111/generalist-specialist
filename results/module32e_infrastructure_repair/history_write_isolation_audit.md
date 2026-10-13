# Module32D历史误写调用链与32E隔离

32D cpu_validation.py导入m32d.monitor.IncrementalMonitor，继承module31a.campaign.TimedMonitor → module30.telemetry_v2.SafeMonitorV2 → module30.telemetry.SafeMonitor → CostMonitor。CPU初版没有运行GPU路径的configure()全局重定向；SafeMonitor.__init__使用其定义模块的OUT创建request_journals/CPU_CONCURRENT_1.jsonl和CPU_CONCURRENT_8.jsonl，journal_snapshot调用其定义模块的save('live_progress.json')，save内部仍引用Module30 common.OUT。即使调用方m32d.common.OUT已是Module32D，继承方法的全局命名空间仍是Module30。sample/collect→journal_snapshot→live_progress更新造成历史临时进度覆盖。详见历史cpu_validation.py、m32d/monitor.py、module31a/campaign.py、module30/telemetry_v2.py、module30/telemetry.py、module30/common.py，以及本目录prior_incident_record.json。

已知历史差异：Module30 live_progress.json原字节副本不存在，32D按未修改的C2_split05 checkpoint和窗口重建最终进度，原UI elapsed_seconds不可恢复；不能称为逐字节恢复。两个新CPU日志已在32D事故目录保留。32E不恢复、不修改这些历史文件，以本轮开始的明确现状作为哈希基线。保留原事故记录副本及其来源SHA，历史Gate、原始实验、权重不改。

32E监测器只继承无输出副作用的CostMonitor初始化，显式output_dir且要求位于Module32E，日志、进度、checkpoint均使用本轮common.save。没有修改旧模块全局OUT/save，没有调用旧监测器构造函数。CPU/GPU父进程启动前先安装继承到exec/fork的Landlock，只允许Module32E普通文件系统写入；原系统ABI1不涵盖truncate/metadata，额外seccomp禁止这些native syscall，Python audit检查路径/符号链接/FD的写入。仅保留/dev现有设备和/dev/shm IPC例外，所有普通缓存/tmp归本轮work目录。不能使用私有mount namespace（主机不允许unshare），没有修改历史目录权限。保护只限制本轮进程树，不限制无关主机进程，因此完整前后SHA仍必需。

保护自动测试实际针对Module30 progress与Module32D Gate：Python/native子进程写入、truncate、metadata、删除、重命名、symlink escape、旧save错误调用、旧输出目录初始化都被阻止；允许本轮正向写入。每项结果保留，不删除失败。真实回放/并发/GPU分别保留审计记录；每个GPU worker保存安装标记和正常退出后的违规清单。历史完整清单覆盖全部results/module*历史文件以及32D冻结源码、输入和模型权重；检测新增、删除、SHA、大小、mtime。32E新增输出不进入历史集合。
