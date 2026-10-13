# GPU工程环境记录

同一GPU UUID `GPU-31f2671b-99c5-ef57-be45-5c9a871e11f4`，RTX4090，驱动595.71.05；nvidia-smi CUDA13.2是驱动支持上限，PARSeq PyTorch2.14.0+cu130实际runtime13.0。可见物理设备节点为/dev/nvidia1，CUDA仅暴露一张卡映射cuda:0；完整设备和容器字段见container_environment.json/device_nodes.txt。未重装驱动、安装软件包、设置频率/风扇/功率。原subprocess.Popen+exec顺序READY，控制器不初始化CUDA，worker在导入后端之前安装自身隔离。

每臂前置温度≤38°C连续3次，驻留调理后≤40°C连续3次，原等待上限300/180秒。调理300秒有实际六任务工作，不在人为延时ASR中制造排队。原300/30/180/60秒工程协议与冷/热缓存加载条件详见冻结协议和raw日志。0.5秒实际NVML记录SM/显存频率、Pstate、功率、温度、风扇与hw/sw thermal slowdown；不以温度均值差推断降频。每臂均保持原450W功率限制，GPU_operating_state检查只在真实测量窗口核验全部采样的hw/sw thermal slowdown为Not Active，不宣称没有任何功率/空闲clock变化。峰值是NVML采样高水位，非瞬时allocator最大值，native worker额外保存allocator峰值。

| 臂 | 测量平均°C | 测量最高°C | 实际SM MHz范围 | check/publish联合max ms |
|---|---|---|---|---|
| E0_S | 47.83720930232558 | 62.0 | 2475.0–2730.0 | 1.6939453780651093 |
| E0_D | 48.53125 | 60.0 | 2475.0–2730.0 | 0.6457986310124397 |

控制探针沿用perf_counter墙钟，覆盖state_lock等待与原异常谓词，检查及phase发布最大值均须<10ms；原始每次wall/thread_cpu/state_wait/state_hold以及全部publish样本保存，不删长尾。路由开销为真实lock+增量projection+policy.decide+enqueue，另存P50/P95/P99/max。等待/服务时间使用同一worker Unix submit/start/end；控制器monotonic到达/完成及跨钟映射原始字段均保留，未把路由CPU开销混为GPU服务时间。

六模型退出BYE、exit code、线程及显存释放逐项由valid_session验收，最终nvidia-smi状态见nvidia_smi_after.txt。历史全量SHA扫描安排在测量前后，未在请求关键阶段占用CPU/磁盘。
