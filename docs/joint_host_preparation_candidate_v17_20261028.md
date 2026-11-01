# 联合 host preparation：支持范围、候选与最小补数

当前训练证据足以证明不能用各 owner 的 solo 均值拼出共跑请求的 host 次序；它还不足以识别任意输入、路由和积压状态下的外生 CUDA 供给过程。本轮不修改正在运行的 v15/v16，不做 GPU 实验，也不把场景回放或 CPU 契约检查计为预测准确率。

## 已有证据究竟观测了什么

唯一数值来源是 `artifacts/predictor-joint-host-scenarios-v13-20261028-training03/observations.json`。共有 56 个完整训练 job：natural 28、unified-light-v8_1 28，各 44 个请求。每个 packet/domain 有自己的 execution_context SHA；不能把跨 context 的两次同名 job 当作同一个条件分布的两次独立重复。

natural 在共同的 scheduled job 原点下记录 submit、handler、pre_sync 和首个语义 Python 阶段，保留同 job 跨 owner 的相对时刻。首个语义阶段不是第一条 CUDA release；两个 OCR job 没有这一边界，必须保留 UNKNOWN。light 记录同一 Nsight 时钟的首个 kernel API/活动及 runtime prefix；它的公共 arrival 与 trace offset 没有已识别的同原点映射，API wall time 还可能包含 GPU 等待。两种域没有逐请求配对的观测依据。

具体反例：natural packet01 的 `SG-text#0-short` 中 G2 的语义起点比 text 早 55,703 ns；packet02 中 text 比 G2 早 7,205,710 ns。此处是训练内描述，不是泛化误差、因果结论或跨 packet 共同概率的估计。

新增 `joint_host_provider_v17.py` 只把上述完整观测做成可执行且可核验的 checkpoint/vector 接口。它按 domain、完整 workload（alias/task/pool/arrival）、execution_context 和可选 capture SHA 精确选择；核对 observations 与全部源文件 SHA；不均分场景概率、不拼 owner 边际、不配 natural/light、拒绝 counterfactual_release。它是诊断/候选构造的输入边界，**没有闭合部署队列预测缺口**。

## 可以独立验证的统计候选

对一个 natural joint job 的每个请求 i，完整保留同一场景的向量：

```
H_i = handler_i - scheduled_i
P_i = first_semantic_i - handler_i
B_i = (pre_sync_enter_i - handler_i, pre_sync_exit_i - handler_i)
```

H 包含 dispatch、OS 调度以及可能的 worker 等待，P 包含准备过程及可能的 pre_sync GPU 等待。它们不是已去除 GPU 等待的纯 CPU demand。不要把 sync wall time一减就当外生供给；同步与前驱关系仍应由执行 DAG 表达。

原 v8 host provider 中的 preparation 参考点是 `semantic - handler`；engine 将 `not_before_offset_ns` 加到预测的 `state.start_ns`。第一版统计候选只用同一个训练 joint job 的整组 P 替代每个 owner 的 solo mean preparation，保持原始 light API 相对语义/handler 位置、post-preparation depth step 与 terminal tail 的冻结值。若原模型每个 node 的 prefix offset 是 c_{i,n}，则：

```
preactivity prefix: R_i,n = 原冻结的 handler-relative offset
semantic-relative prefix: R_i,n = P_i + c_i,n
post prefix: R_i,n = P_i + c_i,anchor + (depth_n-depth_anchor)*原冻结step_i
terminal: 原冻结 after_predecessors tail_i
```

这是一种“联合 preparation 替换”的待验证统计假设，不是自然 API release 的因果识别。不得把 natural 的 `semantic - scheduled` 直接替进原 handler-relative offset；那会把原 transport/queue 延迟混入 preparation。也不能将新增 P 和原 prefix_release 中已有的 P 相加，必须先恢复原冻结的 API-relative offset。

P 整体抽取可保持 **P 的跨 owner 相关性**，但单独 P 不保持真实语义起点或 first-ready 的全序：H 的差异仍会改变次序。`preparation_vector` 明确输出这个限制。若再纳入 H，一个可检验的 empty-owner 训练起点假设是：

```
predicted_handler_i = max(arrival_i + H_i, predicted_owner_ready_i)
predicted_semantic_i = predicted_handler_i + P_i
```

这与 `max(arrival_i, owner_ready_i) + H_i + P_i` 不同，后者会把与已有排队重叠的 transport delay 再加一次。H 在持续积压、host contention、batch backend 或不同 resident layout 下的迁移没有现成证据，不能直接使用。上式也只是候选，pre_sync 依赖可能要求进一步拆解 P。

实施时应使用单独的 request/block-dependent release overlay，并保持原 work/DAG/edge/phase 不变。v16 的 `CompiledStaticProviders` 明确假定 release 仅依赖 `(alias, task, pool_id, node)`；不能把会随 block/request 改变的联合场景伪装成该静态坐标表。固定每个候选的 scenario assignment，并将它写进预测身份与续跑 checkpoint。

## 57 请求的 block scenario：现在必须标为 OOD

已有 bank 每个 job 只有一个请求/owner，最多三个 owner，请求结束后的连续供给与 block 间共享 host 状态未观测。它不能给完整 57 请求 workload 提供经训练支持的 joint scenario。不能依据预测后出现的排队或获益再挑 block，不能为凑匹配把同一 owner 的请求拆成互不相关的 solo 场景。

若预先定义 block，必须在查看结果前冻结：请求到达窗口或公开 request index、完整输入/路由/相对 arrival 签名、跨 block 的 owner-ready/host 状态、一个整体场景 ID、所有候选使用的共同场景抽样。匹配必须是完整条件组合；没有组合就 UNKNOWN。仅相同 owner 数、相似长度、任意平移 arrival 或把 natural 与 light 组合都不是精确匹配。输入 alias 变成 shared_text 的新三-owner布局更不能继承旧四-owner条件。

可以明确 opt-in 一个 block 重采样的敏感性候选，但它是对 block 间独立性、持续 host 状态和条件迁移的额外假设；当前两 packet/context 的样本也不支持稳定估计场景概率。应逐场景报告范围/候选排序，不自动赋 1/N 权重、平均成单一 P95，更不能称为最终分流证据。

## 最小的下一批训练数据

1. **统一同一次执行的时钟与身份。** 在一个真实共跑 job 内同时保存 scheduled、dispatch、reader receipt、handler、pre_sync enter/exit、语义边界、runtime API issue/return、kernel/copy dependency 与 owner-ready。记录跨时钟锚点及映射残差；每个 owner 独立做时钟平移会破坏相关性。同步等待作为 DAG 前驱约束保留。若现有权限只能得到语义边界，就只支持 preparation 候选，不声明 CUDA supply 已识别。
2. **对当前要比较的条件补整组数据。** 沿用实际原生共享执行，不增加任意全局一/二请求 cap。对专用双路/G2/OCR及新 shared_text backend 分别建身份；覆盖精确输入组合、同时与错峰到达、空闲与已有 backlog，以及实际 CPU/worker 配置。对 57 请求采用整段 training workload 或预声明多请求 block，保留完整到达及失败分母，观察相邻 block 连续 host 状态，而非再采独立 solo 均值。
3. **用控制变化分离等待与准备。** 固定输入和主机配置，在受控的 arrival offset/并发与 GPU 先前活动条件下重复完整 job，检查 pre_sync 等待是否随设备前驱变化；首个 API 的 CPU 准备与等待段不能靠 endpoint 回归分离。profiling overhead 用另行设计的对照衡量；两次 capture 的样本不能假配成 joint 数据。
4. **按精度而非任意次数停止训练。** 以 whole capture/job 为统计单元，检查 first-ready 次序概率、联合 preparation 分位数及候选差分排序的区间是否仍宽。跨 capture/context 差异先解释或分层，不能靠累计 owner 样本数掩盖只有少量完整 job 的事实。独立验证保留新的请求/arrival和capture，冻结后再运行，不把验证回灌训练。
5. **逐级核验再进入收益。** 先测联合 preparation/H 的次序和边界，再测全程 queue/parent receipt/P95 的覆盖与误差，随后测 ΔU 正负和候选排序。使用相同完整到达分母和原六项评分；本轮 CPU source/contract 检查不替代这些独立验证。

这些数据补齐前，可以把第一版 P 向量替换作为独立候选准备好，并在已支持训练条件上验证工程接线；不能强行让所有 57 请求输出 numeric 来替代缺失的可辨识性。
