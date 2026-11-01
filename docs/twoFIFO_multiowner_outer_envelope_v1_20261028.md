# 多路队列的有限速率保守包络

`twoFIFO_multiowner_outer_envelope_v1.py` 是独立 CPU 候选，没有修改正在运行的采集、冻结 runner 或自动接续。它处理 paired transport 只有可行区间时的完整请求集，避免为生成单点而任选不可观测的管道延迟。

前提是显式声明的封闭世界：所有潜在可见 owner 均在公开请求集内，初始 peer 状态为空、没有待处理协议消息，每个 owner 使用单一 stdout producer。每个实际宏 phase 必须有完整的 `(owner, phase, n, same)` 正有限速率表，其中 `0 <= same <= n <= owner_count-1`。协议 slacks 和完整请求向量在这一 draw 内固定。缺少必要证据时，所有 offered 请求保留并返回 UNKNOWN。

对宏工作量 w，任意可见状态切换下都有 `w/r_max <= duration <= w/r_min`。这也适用于宏内部每个 mark 的累计工作坐标。速率极值允许不可同时发生的竞争状态，因此这里给出保守外包络，不是可达极值。

paired START 对前一个实际 READY 单调。分别把 READY 下界、上界代入完整 paired 可行集，可得 START 的外界，再传播 READY 的工作时长界。sender FIFO 使用公开 arrival、enqueue 和 write 时间，按真实 enqueue 顺序推进；请求模板仍由公开 arrival ordinal 选取。

同一 owner 的 mark/RESULT/READY 发出顺序固定，reader 使用 `max(occurrence, previous_reader) + reader_slack`。该递推对全部 occurrence 时钟单调，因此 RESULT reader 和每请求 E2E 的界成立。共享 collector 改变未来可见状态，但全部有限速率已经包含其影响；此模块不输出 collector processing 时钟界。它不能覆盖另有共享 reader 线程、外部 producer、stdout 写阻塞反馈或额外准入控制的执行后端。

完整 cohort P95 对所有延迟坐标单调，所以逐请求包络也给出 P95 外界。固定 drain 边界使用 `receipt <= deadline`；存在潜在未完成请求时，原 fixed-drain P95 保持未知，另列未截断模型的 P95 界。不能从已完成子集补出尾延迟，也不能把跨 owner 的单个 latent 端点组合视为全局最优。

作者五项 CPU 测试检查九十个具体可行世界，包括三个 owner、排队、分数速率、reader backlog、与 arrival 不同的 enqueue 顺序，以及原两 owner 反例：u 为 0/10/20 时 P95 为 20/19.5/29。反例的内部值也在本外包络内；不能由此声称包络紧或实测预测准确。

下一步只在独审通过后作为训练诊断的另列输出，报告宽度和无法判定的比例。当前没有接入冻结生成 runner，也没有物理 GPU 速率界、M/E 资源界、潜变量概率或完整 ΔU。有限表的域内数学包络与真实执行的模型迁移有效性需要分别验证。
