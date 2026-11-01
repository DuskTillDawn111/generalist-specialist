# 精确公共线性块 CPU 原型 v2

该原型位于 [shared_linear.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/backends/shared_linear.py)，仅依赖 NumPy，无 torch/CUDA import，不加载模型，不接入现有 frozen 执行链。它提供可实际执行的 concat/GEMM/split/残差算子参考，**不是已经部署的同 owner HF generate、batch 调度或 GPU cache 机制**。

## API 和搜索定义

```python
from gsdeploy.backends.shared_linear import prepare
plan = prepare(W1, W2, max_search_cells=2_000_000)
(Y1, Y2), cost = plan.execute(X1, X2)
# Xi: [batch_i, in_i], Wi: [in_i, out_i], Yi: [batch_i, out_i]
# plan.reconstruct(): 重建两个原始权重，供审计。
# plan.audit(): 复核重建权重的dtype、shape和精确字节hash。
```

`weight_identity` 的 SHA256 包含 dtype（含 byte order）、shape、C-order 字节。原型接受有限 float32/float64、两权重相同 dtype；不同输入宽度、输出宽度、batch 行数以及不同公共块坐标均可。原始参数不可被静默量化/转 dtype；当前不支持量化 packing、bias、activation。

`find_max_common_contiguous` 返回两个权重中**面积最大的连续、轴对齐、精确字节相等矩形**；它不是任意非连续行列集合的全局最大公共子矩阵搜索，也不做近似相似性匹配。面积相同时按 `(left_row,left_col,right_row,right_col,height,width)` 确定唯一结果。

两个完全相同权重直接匹配全矩阵。其余情况枚举所有相对行列对齐，对每个精确相等网格用 histogram/stack 最大矩形算法；总 equality cell 数恰为 `W1.size * W2.size`。超过显式上限时，在枚举前抛 `SearchBudgetExceeded`，不把部分搜索结果叫做最大值。默认2,000,000，是小矩阵 CPU 机制原型上限，不适用于直接穷举真实大模型所有参数；真实部署需要先按算子语义和tensor身份筛选候选再搜索。

匹配块的两侧输入切片按 batch 维拼接，一次共同 GEMM，再拆回逐分支输出；每个权重的剩余部分被分成至多四个不重叠矩形分别计算并相加。无需执行含零遮罩的全矩阵 GEMM。无公共块时明确退回两个完整残差 GEMM。prepare 持有公共权重一份及独立残差的只读副本，外部原始数组后续修改不会改变 plan；每次 execute 先 CPU reconstruct/hash 审计，检测打包存储被更改。

数学上 `Xi @ Wi` 保持不变；浮点加法的 reduction 顺序改变，因此与原单独 BLAS GEMM 以容差比较，不宣称逐bit一致。公共块复用不减少必要乘加次数：`executed_macs == independent_macs`。它改变权重存储及算子组织，是否减少缓存流量/时间需真实后端另测。

## 明确计入的成本

execute 返回原独立/共同/残差 MACs、GEMM 调用数、逐输出累加元素数、concat复制字节、共同结果张量、split复制（当前为view，0）、最大单个残差结果、输出缓冲、原双权重字节、打包权重字节、打包复制字节、公共/残差各自存储与保存的权重字节。逐次审计会重建两份原权重，单列 `audit_reconstruction_bytes`。

这些是逻辑数组payload和算术计数，未统计 NumPy/BLAS 内部workspace、metadata、allocator/RSS、总峰内存或时间。审计还有hash读取/临时字节成本；不能把逻辑省下的参数字节直接解释为实测显存或缓存收益。共同块很小时可能引入1+8次GEMM、额外累加和concat，应保留这种成本而非假设必然获益。

## 验证

[test_shared_linear_v2.py](/root/autodl-tmp/generalist-specialist-project/tests/test_shared_linear_v2.py) 共11项：完整权重共用、不同宽度/不同坐标公共块且四残差、无匹配、35对小矩阵与独立穷举最大矩形oracle、固定tie、非连续输入、hash/dtype/shape/正负零、搜索限额拒绝、原数组修改隔离、打包权重篡改、非有限/形状错误，以及12组float32移位公共块输出对照。

CPU命令：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m unittest discover -s tests -p test_shared_linear_v2.py -v`。11/11 PASS。未运行GPU。

## 下一步真正同 owner batch 的接点

现有链路是 `static_study_worker.handle(request) -> ProfileAdapter.run(sample) -> NativeExecutor.infer(sample) -> _infer -> model.generate`。真实源码来自 frozen `artifacts/retained-replica-profiles-20261021/source-qa`；不要直接修改 frozen 文件或往 `generate` 外面套共享标记。

建议新增独立 text-only owner/runtime entry，加载一次相同 ChatQA checkpoint及QA adapter；上层双逻辑队列经已有batch grouping接口发送请求列表。新执行器逐样本调用 `CompactAdapter.build` 和原text准备逻辑，再以相同tokenizer/生成政策组成一个HF batch。生成前检查每请求完全相容的model/revision/quantization/adapter、terminator、采样/beam/预算；不相容请求不能强行同批。

最小改造点在 frozen `_infer` 的text通用分支：它于约549行取得prepared inputs，约584行校验ChatQA terminator，约618行真正调用 `self.model.generate`；新代码可复用其语义与数据，不能复用约619行的batch1 `sequence_ids`。需新多行输出恢复，保留padding后输入宽度与真实输入token数的区别，逐行识别EOS/生成长度并分别 `adapter.finish(request,sample,raw,...)`。decoder-only batch通常需左padding及正确attention_mask/position处理；先CPU接口测试，再独立真实正确性测试，不直接把这一约定视作当前模型已验证。

`ProfileAdapter.run` 原约961行包住单请求同步/缓存回收/计时，新batch entry应在批边界测共享执行时间，在逐请求事件中记录arrival、enqueue、batch_admit、generate_start/finish、result_emit。常规HF generate通常整批返回，单行遇EOS不等于当前wire已能提前交付，因此P95与吞吐不能虚填独立完成时间。暂不实现逐token调度时，明确整批交付语义。

当前executor存在请求间candidate及hook可变状态，新owner不可直接双线程调用原infer。shared_linear可独立验证准确公共块的重构与算子组织；把它接入HF量化线性层还需要bit/layout/scale/zero/adapter及真实kernel支持，当前原型没有该能力，也不代表int8/GPTQ跨格式合并已经解决。
