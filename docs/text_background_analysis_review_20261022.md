# 背景诊断终态分析工具审查

审查对象为 `tools/spillover_background_20261022/analyze.py`。未修改已冻结的 prepare、runner、packet 或 GPU 原始文件。

修复了两个会阻断实际数据分析的问题：冻结 MemoryMonitor 仅输出 `unix_ns`，现在通过测量窗口 Unix/monotonic offset 检查起止覆盖；质量评分现在读取已绑定 `catalog.quality.references`，输入 JSONL 刻意不含答案，不能作为评分参考。

资源审计独立计算 safe-factor 峰值、窗口覆盖、whole-board 测量能量与启动能量，缺采样或数值不符不会成为 PASS。原生质量是否可计算单独报告，不新增绝对精度准入。服务、质量和资源结果不激活预测器，不产生总效用。

终态失败，包括加载/预热失败导致没有 measurement window，都会留下原始 receipt、错误及请求状态计数。不能验证的字段保持空缺；审计出错不会让整个 CLI 丢失失败记录。成功测量的 primary/support 分开、host/native 双时钟截止分开核验，晚返结果不补进截止内完成量。

context 同时报告两文本专用、OCR、G2 的实际 handler 重叠。另列 G2 的 text/image/asr handler 分层，供解释跨路径影响；G2 自身的不同任务仍串行。无返回或缺 handler 时间戳的已释放请求被列为未知活动，不能把零已观察重叠称为确定空闲。按路径、角色、专用/OCR 活动 context 输出服务汇总，每 input 的 OCR context 中位数及跨块同 context 描述性比值另列。原先声明的全体 input 跨块判据不变，同 context 比较不是新增筛选门。

验证：

```text
python -m unittest tests.test_spillover_background_analysis -v
7 tests, OK
```

测试包含真实已冻结首轮 21 条 native priming 的四任务质量重评分（全部 AVAILABLE）、错误使用输入 JSONL 的 UNKNOWN 回归、真实 MemoryMonitor 字段结构、缺能量/错误能量、加载失败 partial、重叠去重、未知活动、双时钟迟返、primary/support 混排下实际 admission 串行性。

这些是分析器验证，不能作为本次背景 GPU 批次已完成或通过的证据。终态后必须对实际 raw receipt 运行独立分析，并报告 coverage 与各项失败。
