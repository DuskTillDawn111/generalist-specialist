# 独立 V4 固定运行入口

`tools/predictor_validation_20261027/admission_validation_run_v1.py` 只负责冻结验证模块之外的运行调度。`audit` 为 CPU 操作；`run` 明确启动 GPU，仅由主任务在 T8 训练、三项拟合和独立预测 seal 完成后调用。

固定顺序为 `admission2_balanced`、`admission2_skewed`、`admission2_spill_low`、`admission2_spill_high`，seed `628201`，每条 1200 请求。`audit` 首次调用冻结验证模块，完整重验三个拟合、固定服务模型和数值预测；随后建立递归 JSON 来源闭包，包括冻结执行快照对应的原始代码。每个 job 启动前及结束后重哈希整个闭包，避免每次重复拟合/数值 rollout，也不能沿用来源已变化的缓存证明。

预测有 UNKNOWN 时必须显式 `--diagnostic`。诊断运行采集全部四条预定路由，不跳过预测未知的候选，也不将 UNKNOWN 改成 PASS。原生 collector 使用原窗口及有界 drain；任一采集、语义映射或来源检查失败立即停止，记录 `INCOMPLETE_NO_RETRY`，4800 请求及四候选的预定分母保持不变。公共 target plan 目录中的 exclusive `independent-V4-attempt.json` 保留永久尝试声明，禁止通过新输出或新 seal 目录对相同固定计划再次运行。

mutable `run-state.json` 使用临时文件与原子替换更新；原始证据、claim、reservation、attempt 和 completion 仍使用 exclusive 创建，不覆盖历史文件。完整实际映射复用冻结独立验证模块，保持原稳定性和六项效用。采集完成仍只记 `COLLECTED_UNASSESSED` 或 `COLLECTED_DIAGNOSTIC`，资格由独立报告另行判断。

`report` 校验 seal 绑定的 claim、完成前缀和原始文件；只将已严格完成的实际日志交给原验证报告器，缺失路由保留在原报告的完整候选分母中。它调用 `verify_report` 重算报告后附上有限运行状态，不对未完成请求伪造耗时或质量，也不允许最终正收益资格。

CPU 控制测试采用明确的 fake collector 验证调度规则，不产生真实 GPU 证据。测试覆盖递归来源/原始快照变更、一次完整 audit、UNKNOWN 四路由诊断、失败停止、固定分母、同输出及换输出禁重试，以及首条运行中来源变更后停止。
