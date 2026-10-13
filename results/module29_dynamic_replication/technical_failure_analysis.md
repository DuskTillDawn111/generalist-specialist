# Module 29 技术停止审计

第 5 次 G2 S-only 在 1997.196239/2400 秒出现遥测错误 `dictionary changed size during iteration`。错误由 telemetry collector 捕获；没有保存该异常的具体 Python 栈帧，不能唯一定位触发语句。

源码存在一致的竞态解释：gpu_replication.py 的 DurableMonitor.sample 在未持有 runtime.lock 时 deepcopy 正被 sender/collector 更新的请求字典；deployment_runtime.py 对这些字典的更新持有 runtime.lock。原始监测器来自 Module 28/28R。此为源码支持的候选原因，未用新 GPU 实验验证、未修补冻结代码。

冻结队列共 27395 个请求；中止前计划到达 22816，接受 22815；清洁关闭后已接受请求完成 22815。未来 4579 个计划槽和异常时尚未接受的 1 个请求全部保留；共 4580 个注册请求未完成。

五个 worker 均 BYE、exit=0、未强杀；GPU 释放已验证。未发现已接受请求错误、服务崩溃、GPU OOM 或温度停止证据。遥测异常使第 5 次会话及完整 2400 秒窗口无效；不得用它的完成前缀计算 U，也不能据此声称 S-only 物理不稳定。

冻结 failure_policy 为 ZERO GPU retries/replacements，并在技术故障时停止整轮。因此剩余四次未运行；不修复后重跑本轮、不换种子、不补 GPU。Gate 保持 REPLICATION_INVALID。第一组合法负收益保留，开发窗口不计入统计。
