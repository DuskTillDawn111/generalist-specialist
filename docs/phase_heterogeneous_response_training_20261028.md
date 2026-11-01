# 固定 family/phase 响应设计：训练支持与失败边界

本轮仅使用三份完整训练 capture、150 请求、1002 macro 节点，在 CPU 上比较预先声明的全局 2 维、family/phase count 加全局 same 的 12 维、family/phase count 与 same 的 22 维设计。entry、OCR inference 和 READY tail 全部保留。没有读取独立验证标签，没有启动 GPU，没有升级主队列 provider。

训练报告：`artifacts/predictor-phase-heterogeneous-response-v2-natural-20261028-training01/report.json`，SHA256 `0abea54e63cb547598cca33a585b1094e597e897e1fc69c02f137c4230b833be`。v1 在不可辨识设计的数值求解期间中断，保留独立 interrupted01 回执，未把中断当作失败模型的有效输出；v2 为每个 full/fold 保存 checkpoint，并对严格结构失败停止无意义求解。

| 设计 | 完整训练 Jacobian | 全 capture 留出资格 | 结论 |
|---|---:|---|---|
| 全局 2 维 | 50/50 | full 和三个 fold 均通过 | 仍为经验条件响应基线 |
| 12 维 | 60/60 | 一个 fold 为 59/60，仅两 fold 通过 | 不可升级，不解释为生成预测改进 |
| 22 维 | 65/70 | 原始 piecewise 暴露有严格结构依赖 | 保留 FAIL，未删除列或 macro |

12 维失败 fold 的 heldout capture 为 `96f5d8d19b…`。剩下两份训练 capture 的 OCR entry 共 8 个节点、两个公开输入，peer-count 暴露路径全部为零。`count@ocr:entry` 因而完全没有训练支持。三次根解训练平均 NLL 都为 3.463268249，其他 11 个斜率一致，但这一斜率分别为 −1.161139、−0.382639、4.787323；最大根距离 5.948，最小奇异值 3.19e−17，调整 nuisance 后只有 11/12 秩。该 fold 的 heldout 条件 NLL 为 10.3067。优化器的 success 标志不能使这一自由方向获得资格。

22 维设计中，text READY tail、G2 READY tail、OCR inference、OCR READY tail 的 same 列在原始所有 piecewise 路径中严格为零，OCR entry 的 count 与 same 列严格相同。单 OCR owner 不存在另一个相同 OCR inference peer；native END 已清除 visible phase 后，也不存在 visible READY tail peer。这样的结构零不能通过“多采一些”解决。OCR entry 只有 entry peer 的混淆，则需要改变启动先后以产生可达的新暴露。

生成对照报告：`artifacts/predictor-phase-heterogeneous-generated-v1-natural-20261028-training01/report.json`，SHA256 `27fe7da53fb593c4fb406d284d5e5d3b5c8200273011a575ac5d064d1a0dcb3c`。12/22 维均明确跳过，未挑选成功 fold 生成指标。全局基线相同四次固定 draw 重现：150 个完整 ensemble，mean-point MAE 153.806 ms、RMSE 253.359 ms、有限四原子 CRPS 119.850 ms；160.038 ms 是随机 world 平均绝对误差。四样本 min/max 覆盖率 44.67%，不是 90% 预测区间。这个隔离参考仍假设 READY processing 后派发，且每个 job 初始 station 为空；单 owner 单请求场景与实际 two-FIFO 对照相同，不能用于繁忙队列资格。

描述性关联报告：`artifacts/predictor-response-residual-association-v1-natural-20261028-training01/report.json`，SHA256 `ca39cf9a80befcb028c9b1dd801105d9872b0291bb220e4b7e244c60dd66deb7`。旧全局模型的 text prefill、G2 prefill、OCR inference log-work 残差与 peer prefill 暴露的 Pearson r 分别约为 0.860、0.879、0.846。关联只能定位需要交叉控制的变量，不能证明物理因果；context/job 标识仅作诊断，未进入拟合或预测查询。

最小干预矩阵和逐 fold 原始支持证据：`artifacts/predictor-phase-response-identifiability-v1-natural-20261028-training01/report.json`，SHA256 `7c444dd54f561d295beb17c79ff2190ac5488a17ea3facf819f5463fef2eea39`。每份独立 capture 均应让同一个公开输入重复单路、不同 peer count 与 peer phase 条件，优先补 OCR entry 在语言 BEGIN 已处理之后启动的暴露，再交叉 OCR inference、G2/text prefill 与语言 prefill/decode。保留同时、own-first、peer-first，记录实际达到的 overlap；未达到目标相位的请求也保留。与实际 sender/worker busy、同 owner burst/HOL 的矩阵合并，固定 window/drain 和全部 offered 分母。

论文借鉴的边界必须保留：USHER/Roomie 的算子/阶段结构、互补性与资源需求可作为结构设计依据；当前静态 projection FLOP/逻辑字节描述只覆盖部分结构。Attention/KV、量化实际代价、真实 DRAM/cache、SM/带宽竞争、共同 batch 和公共权重 cache 执行后端仍缺物理资格。全局或 family/phase 自由斜率是经验校正，不能称为已实现论文物理模型，也不能称为已经降低真实共驻干扰。

38 个相关 CPU 反例通过，包括 owner-aware reference 在全局斜率下与旧参考完整 trace 等价，以及任一 draw 未知时保留整个请求 ensemble 未知。另对真实训练向量进行了全局退化检查：150 个 joint vector、1002 节点的 work 和所有 marks 精确相同。数值、结构和生成资格继续分开记录。
