# 有限串行续跑 supervisor

工具：`tools/spillover_context_20261022/staged_continuation.py`。

已完成 CPU 准备并经root审阅启动续跑（2026-10-22 13:41 UTC）。请求目录为 `artifacts/static-spillover-study-20261022/staged-continuation-v1/`，其中 `request.json` 的 SHA256 为 `ed258fb25e70bda8f92ed30aa519d12aed5efd8d64b0a5758609fe610e49807e`。两份 packet 的冻结校验分别通过，声明 4 条 fine 条件和 2 条 strong 条件。实时阶段以 `execution/progress.json` 为准；首次核验为 `WAITING_FINE_TERMINALS`。

## 固定流程

1. 每 5 秒读取 fine 队列 JSON，不读取 `current` 指向的任何 live 文件。只接受 `ALL_DECLARED_DEVELOPMENT_ATTEMPTS_COLLECTED`、四个精确 job ID、原 packet/runner receipt、exit_code=0 和原 SHA 的已登记 `run.json`。RUNNING 时等待；队列停止、身份错误或终态缺失立即停止。
2. 调用原 `analyze_fine_development.py`。报告必须为 `ALL_FOUR_TERMINALS_AUDITED`，四个 job 全部 `VERIFIED_TERMINAL_EVIDENCE`，且 memory、energy、shutdown 均通过。无论 U 正负或 eligible 与否，都不用于决定是否继续。
3. 调用未修改的 `run_fine_development.py`，串行执行已声明 strong-controls packet 的 seed107、seed108 各一条。GPU 空闲/UUID、每条资源及退出检查仍由原 runner 执行。任何异常或非零退出立即停止，不重试。
4. strong 两条收齐后，调用原 `compare_development.py` 比较两 packet 的全部六条，再检查六条身份、独立审计、资源和回收，最后调用原 `plot_development.py`。

从 supervisor 执行开始（包括等待和四个阶段）设 7200 秒上限。SIGTERM/SIGINT 触发停止、向当前子进程组发 SIGTERM，最多等待 40 秒；仍未退出则向组发 SIGKILL。原 GPU runner 自身收到信号后负责终止其运行组并回收 native 工作进程。

每阶段前后重新校验冻结 SHA。冻结清单含两个 packet、全部 packet bindings、supervisor、四个被调用 context 工具、background/capacity 工具依赖以及 frozen execution-source-v5 全部 Python；共 840 个文件和 Python 解释器身份。383 份项目 Python 源码另存于 `source-snapshot/`。外部安装包内容未逐文件重新散列，原 native checkpoint 绑定及真实 GPU preflight 仍由原执行链完成。

输出固定在该目录的 `execution/`：`fine-terminal-audit/`、`strong-controls-gpu/`、`all-six-comparison/`、`figures/`，每个必须不存在才创建；各阶段日志及 `progress.json` 位于 execution 根目录。第二次执行同一请求因 execution 已存在而拒绝，不能重用目录隐式重试。

## 审阅后的启动命令

```bash
python tools/spillover_context_20261022/staged_continuation.py execute \
  --request artifacts/static-spillover-study-20261022/staged-continuation-v1/request.json \
  --request-sha256 ed258fb25e70bda8f92ed30aa519d12aed5efd8d64b0a5758609fe610e49807e
```

该命令已执行，勿重复运行同一请求。准备阶段仅运行原 runner 的只读 packet validation；正式supervisor会等待fine四条收齐，按上述门禁自动接着执行strong两条。

CPU 测试：`python -m unittest tests.test_staged_continuation -v`，6 项通过。覆盖 fine 未齐、fine 已停止、终态审计/资源失败均不启动 strong GPU；ineligible 不形成调度屏障；terminal receipt/runner SHA 改变拒绝；信号清理调用子进程组 SIGTERM。
