# v9 全过程独立预测验证：FAIL

四个共享执行候选使用同一个提前声明的 90 秒到达／输入 cohort，各 181 个请求，全部完成；预测覆盖 724/724。预测及门槛先于测量封存。本轮单次生成 `flow-pilot-v9/report/report.json`、`gate.json` 和 `eligibility-addendum.json`，未根据结果重拟合或调整门槛。

| 候选 | service MARE | 父进程 E2E MARE | 队列深度 MAE | 预测 comparison U | 实测 U |
|---|---:|---:|---:|---:|---:|
| shared_balanced | 5.84% | 6.51% | 0.238 | -1.81003 | -1.76009 |
| shared_skewed | 8.37% | 15.22% | 1.772 | -4.42310 | -3.87140 |
| shared_spill_low | 14.94% | 31.23% | 0.713 | -1.88530 | -1.61265 |
| shared_spill_high | 22.16% | 30.14% | 0.574 | -1.59184 | -1.99358 |

门槛结果为 FAIL，共 15 项失败、没有 UNKNOWN。shared_skewed 队列 MAE 超过原声明的 1 个请求；shared_spill_low/high 的 E2E MARE 超过 20%；shared_spill_high 的 service MARE 超过 20%。多个单任务 P95 误差亦超 20%，没有用任务间平均误差掩盖失败。

相对 balanced 的 ΔU 符号仅 1/3 正确，全部候选对排序仅 3/6 正确。低分流比例的预测 ΔU 为 -0.07527，实测为 +0.14743；高分流比例的预测为 +0.21819，实测为 -0.23350。当前模型会错误选择分流比例，不能进入最终分流收益确认。此结果也说明已知输入短探针的局部 service 通过，尚不足以确保连续队列及最终候选排序可靠。

预测的绝对 U 因启动能量未建模而保持 UNKNOWN；表中预测值为计划明确允许省略共同启动项的 comparison U。实际 U 包含每次测量的真实 startup 差。低／高分流相对基线的 startup 差分别仅 421.22 J、506.40 J，并已进入真实 ΔU；报告完整保留该差异。

原稳定性独立重构全部为 recoverable_congested，完整恢复且无 hard reasons。原 catalog rho 数值均处于原阈值以内；旧 catalog 缺当前执行身份／资源容量的适用性绑定，正式 rho 资格仍 UNKNOWN。因此所有最终收益资格均为 false。以上 U 与排序用于预测器诊断，不能视为已获最终资格的收益证明。

后续新增训练应来自提前声明的新训练流，覆盖原驻留、资源和 alias 路线条件，并与新的独立确认分开。本轮 holdout 不参与下一版拟合。容量校准须重算新训练的真实 worker service，经原 g(n)/n 转换接入原 resolver；不能只给旧数值添加身份标签，也不能把后验 busy ratio 替代原规划 rho。
