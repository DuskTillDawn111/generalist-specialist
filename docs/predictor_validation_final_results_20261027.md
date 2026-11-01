# 共驻画像与预测验证：本轮最终结论

本轮冻结的画像、训练、独立验证、机制诊断、四组对照和同域 CPU 搜索已完成，原流程认证及主线程直接收据核对均已通过。**预测器尚未通过准确度验收；整请求准入也未证明整体优于原共享执行。** 验证任务完成与算法科学目标达成分别报告。

## 完成范围与证据

| 工作 | 本轮结果 | 证据边界 |
|---|---|---|
| 共驻微画像 | 264 实际训练 job、690 测量请求；覆盖输入、组合、错位和背景 | 墙钟阶段、板级功率及 owned 离散显存；早期完整启动峰未知，物理 SM/DRAM/L2 互补未证 |
| 准入 T8 训练 | 8 job、9,600 原生及质量记录；容量/资源/质量画像及整 seed 留出 | 服务 Gamma 冻结；V/F 未回流训练 |
| 独立预测 | shared 4,800、admission2 V4 4,800 请求 | shared 原 gate FAIL6；V4 原 gate FAIL10，UNKNOWN0 |
| 机制诊断 | shared/max1/max2 pilot 共3,600；真实准入激活已观测 | 减少执行干扰与等待、尾部恶化同时出现；不是容量资格证明 |
| 最终 F8 对照 | 8 格、9,600 原生及质量记录；所有格正常退出、六项独立算术一致 | seed628301、同一计划到达/输入、单独立块；原 executor seal 认证及外层 wrapper 完成 |
| CPU 同域 DFS | shared 与 admission2 的普通/有界搜索分别一致 | 各有限四路线，0 剪枝；容量资格最优 UNKNOWN，不报告加速或全域最优 |

原始机器结果与图表见[完整中文汇总](../artifacts/predictor-gpu-20261027/campaign-final-summary-v2-final/results.zh.md)、[机器汇总](../artifacts/predictor-gpu-20261027/campaign-final-summary-v2-final/summary.json)、[主线程最终直接核对](../artifacts/predictor-gpu-20261027/campaign-final-root-direct-checkpoint-v1.json)。主线程最终核对不重新运行完整来源闭包认证；认证由原 executor/wrapper 执行。

## 预测器主要缺口

V4 四路服务 MARE 为6.25%、6.17%、9.44%、11.07%，均在原20%门槛内；全过程 queue MAE 为12.22、2.29、2.73、2.00个请求，均高于原1请求门槛。balanced/high 的 E2E 及若干任务 P95 也未通过。两种执行模式的 ΔU 符号3/3、候选两两排序6/6正确，但这些有限验证结果不能消除队列和尾部失败，也不能证明外域排序可靠。

预测器已经读取完整到达与输入。瓶颈是它生成的服务完成、准入交接和后继开工路径尚不准确，而不是缺少未来到达率。前序服务偏差能够沿繁忙链传递；模拟器在 handler END 立即释放 permit，而真实释放经过 reader/collector/锁处理，获准到 START 也有实际延迟。双方已有 strict FIFO 规则，不能把问题误写成模拟器漏了 FIFO。各时序差异与服务误差的因果贡献仍未分离，不能以统一常量补偿或归因到单一硬件因素。

详见[剩余瓶颈解释](../artifacts/predictor-gpu-20261027/predictor_remaining_bottlenecks_readonly_v1.md)和[独立误差图](../artifacts/predictor-gpu-20261027/admission-validation-v1/report/figures-v2/prediction-errors.png)。

## 最终分流对照

下表采用事前锁定 spill_high，对照同模式 balanced/skewed 两条声明禁分流路线中实测较优者。两模式该基线均为 balanced。数值是效用单位，不是提升百分比。

| 执行方式 | 强禁分流 U | 锁定 high U | 分流 ΔU |
|---|---:|---:|---:|
| shared | −1.509633 | −1.396036 | +0.113598 |
| admission2 | −11.718833 | −2.396205 | +9.322628 |

较大的 admission2 内部增益伴随明显更差的禁分流基线。同一 high 路线下 admission2−shared 为 **−1.000169**；因此本轮不能称准入整体优化成功。admission2 high 相对两个执行方式中较好的声明禁分流方案（shared balanced）仍低0.886571个效用单位。shared 的实测 low U=−1.331181，是事后包络结果，不替换锁定 high。保留全部八格，包括 skewed 与 low，不事后改选主比较。

八格原稳定性规则均通过，原容量资格仍全部 UNKNOWN；单块不提供运行级置信区间、显著性、重复收益或全部署最优证明。完整算术由[八格独立描述复核](../artifacts/predictor-gpu-20261027/four-group-final-eight-cells-descriptive-review-v1.md)核对；该较早快照的认证 PENDING 已被后来的原完成凭证和主线程最终核对更新。[最终对照图](../artifacts/predictor-gpu-20261027/campaign-final-summary-v2-final/figures-v1/four-group-comparison.png)与误差图已由主线程目视检查，均有 PDF 版本。

## 尚未实现的机制与后续优先级

计算型/显存型互补的物理归因及 MPS/有效资源份额保持 UNKNOWN。共同 batch、分 batch、公共权重最大子矩阵和跨请求 cache 复用后端保持 NOT_IMPLEMENTED；当前四个独立 owner、batch1 的执行没有获得这些收益，预测器也没有虚构复用折扣。这些后端不在本轮冻结有限方案内，不能冒称论文机制已全部落地。

下一版本先在 CPU 上对齐 runtime 与 simulator 的 ticket、真实 release、grant→START 和完成口径，并补齐缺失时刻的观测设计；然后区分逐 alias/前缀服务误差与队列路径偏差，评估条件服务模型、竞争输入和不确定性是否需要改进。新模型须使用新的训练/独立验证划分；本轮 V/F 保持独立评估身份。条件复用后端完成后再按独立协议测配批等待、额外计算/通信、显存及质量。本轮没有追加实验、重试、更换 seed 或修改原失败门槛。

六项沿用 `U=−(Q+T+L+P+M+E)/6`。T保留测量截止未完成比例；L/P为计划到达到parent receipt；M为startup-through-drain owned采样峰/49140×1.110854一次；E排除drain能耗，启动按3600秒摊销。原g(n)、rho与稳定性规则保留。


[最终子对话验收 v3](../artifacts/predictor-gpu-20261027/final-scope-acceptance-review-v3.md)确认有限证据交付完整，保留预测FAIL、容量UNKNOWN及条件后端缺口。原GPU流程与后处理终端均exit0；本轮任务收尾不表示预测器已合格或所有论文机制已实现。
