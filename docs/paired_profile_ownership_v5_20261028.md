# Paired profile v5:跨进程组的真实进程所有权

v4 training02 的 kernel smoke 四个 resident 正常 READY、正常退出，但 NVML 所有 worker 被当作 foreign，采样 owned 显存为零。失败发生于 priming：4 条 priming 记录、0 条 measured 请求。根因是 v1 collector 把 collector PGID 相等作为所有权必要条件；nsys 可以合法创建新的进程组。这份失败记录保留，不重试或覆盖。

隔离的新入口是 `paired_profile_campaign_v5.py`、`profile_training_supervisor_v2.py`、`profile_owned_collect_v2.py`。campaign/supervisor 除版本、import、schema、protocol 绑定改名之外与冻结版本相同。18GiB preflight、13GiB floor、6GiB increment、2400s timeout、单锁、完整依赖、no-retry 与双 smoke gate 保持原合同。正式 campaign/manifest 尚未 prepare，也未启动新 GPU 工作。

collector 的每个 NVML PID 必须在当前 collector→已声明 launcher→worker 的完整真实父子链上。链中每个 PID 都记录 start_ticks，并在本次验证结束前复读 start_ticks/ppid。launcher 首次绑定在真实 Popen.poll() 前后都确认原子进程尚未退出，再保存 `(pid,start_ticks)`；后续 sample 不能重新绑定一个复用 PID。monitor 持久化身份，READY 也通过同一验证并保留原始 ancestry。

PGID/session 作为证据保存，允许 nsys 或 worker 使用 setsid。未声明的同组 sibling 仍是 foreign；/proc 缺失、身份变化、退出、重设父进程造成的链不稳定都不能成为 owned。过去 membership、READY 的 PID 数字和环境变量均不单独构成所有权。

作者 CPU 测试 25 项通过，覆盖真实双层 setsid、同 PGID foreign、启动期无 READY 归属、死亡 launcher 首次绑定拒绝、已绑定 start_ticks 变化、锁与 detached cleanup、磁盘预算和依赖闭包。CPU 测试使用临时 fixture；正式画像未创建。独立复核及新的 natural/kernel smoke 仍是后续门槛。硬件显存是采样峰值，不能称为瞬时峰值。
