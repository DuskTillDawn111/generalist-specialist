# GPU工程环境报告

{
  "phases": {
    "load": 9
  },
  "measurement": {},
  "thermal_slowdown_samples": 0,
  "worker_guards": [],
  "clock_semantics": "NVML actual SM/memory clocks and thermal flags. Mean temperature alone is not a throttling test.",
  "pair_environment_validation": false,
  "prelaunch": {
    "time_ns": 4600439968961219,
    "unix_ns": 1791647744704203172,
    "uuid": "GPU-31f2671b-99c5-ef57-be45-5c9a871e11f4",
    "name": "NVIDIA GeForce RTX 4090",
    "driver": "595.71.05",
    "total_mib": 49140.0,
    "used_mib": 1.0,
    "gpu_util_percent": 0.0,
    "memory_util_percent": 0.0,
    "power_w": 17.76,
    "processes": [],
    "temperature_gpu_c": 33.0,
    "sm_clock_mhz": 210.0,
    "memory_clock_mhz": 405.0,
    "pstate": "P8",
    "power_draw_w": 17.76,
    "power_limit_w": 450.0
  },
  "release_verified": true
}

本轮在首个worker CUDA初始化失败后停止；仅有启动阶段0.5秒遥测和预启动温控记录，没有驻留预热、conditioning、请求测量或在线控制计时。无配对温差/频率可比性结论，无正式U。native worker错误、清洁关闭、GPU释放及全部启动样本见原始数据。
