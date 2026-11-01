# 完整 57 请求的联合 host/lifecycle 训练采集

入口是 `tools.predictor_validation_20261027/whole57_joint_training_v23.py`，包含可执行的 `prepare`、`audit`、`cohort`、`execute --execute`。准备和审计不启动 GPU；实际采集由主对话排程。CPU 导出/复核入口是 `whole57_joint_review_v23.py --run … --output …`。

训练 recipe 在采集器源码中事先固定：30 秒，text 1.85 rps、OCR 0.05 rps，三个 10 秒段的倍率为 0.5、2、0.5，共 56 个 text 和 1 个 OCR 请求；尾部 drain 120 秒。四种路由为原 balanced、skewed、spill_low、spill_high，来源明确标为预声明训练 recipe。未从失败验证结果选择路由。请求 ID 使用全新 `whole57-training-*` 前缀。prepare 只接受已审计的 training-only profile 的八个输入与来源，不读取 validation plan、请求 ID、输出标签或候选效用。

采集保持 text×2、G2、OCR 四 owner 驻留、原 batch=1、原共享资源和完整十二次 priming。复用 `unified_corun_collect_v8_1` 的单次整生命周期 Nsight、RLock 身份/板级监控、独占 GPU 锁、磁盘与时间 watchdog、所拥有进程组的清理；复用 `DeploymentRuntime` 与 `semantic_worker.py` 的真实 FIFO 发送和事件边界。完整 57 请求在第一次 enqueue 前登记；到达循环持续按公开 offset 入队，只有末尾 drain，管道堵塞不删除或延后登记未来请求。采集失败保留原计划 cohort，不补造实际已执行记录。

每次 light 执行保存 scheduled/enqueued/dispatch/pipe-write-completed/handler start/end、NVTX/CUDA API/kernel、parent reader/dequeue/processing、worker next-ready，均绑定同一次 raw/plan/context。CPU reviewer 检查完整训练身份、priming、四 READY PID/spec、FIFO 生命周期顺序、实际来源、watchdog、板级采样和整生命周期边界，再导出 native session 下首个 launch API/kernel 及原始 SQLite。natural 保留真实 host/semantic 与 lifecycle 边界；它没有 CUDA 轨迹，不能补上 first CUDA release。

host `perf_counter_ns`、Nsight 原始 session 坐标、session systemClockNs 和因果边界区间分别保留。没有把各 owner 的 median map 当作精确公共时钟，也没有宣称已校准的跨时钟误差上界。初始 READY 仍沿用原 worker payload/父启动边界；每个请求后的 next-ready 是带原始 perf 时间戳的直接观测。observed API wall 与 pre-sync 可包含 GPU 等待，不能直接称为反事实外生 host supply。

新上下文完整冻结实际 collector/reviewer、runtime/worker、输入及原 source/resource 证据。它是新训练域，不能自动转移到旧验证上下文。v17 的每 owner 只有一次请求的旧 bank 契约会拒绝重复 owner；这里保留整个 57 请求向量，后续拟合须使用新的重复请求契约，不能拆成各 owner 均值拼接。采集和 reviewer 均不计算 U、P95 或部署收益，不改变原六项等权效用和稳定性规则。

CPU 测试包含真实四子进程 stdin 管道堵塞下完整 57 请求和原始时钟链保留、第一条到达 wait 失败前全部登记、priming 失败保留完整计划、单 Nsight 启动命令、因果 host anchor、真实 controller/descendant process group 超时回收。CPU prepare 的八个 run 只是候选训练计划；独审通过后由主对话选择/顺序运行训练，再做独立验证。
