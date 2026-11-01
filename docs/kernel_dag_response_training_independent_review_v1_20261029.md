修复后的 node-keyed kernel lifetime compiler 与低内存拟合证书通过独立 CPU 审查，未发现尚未解决的阻断问题。证据在 `artifacts/predictor-kernel-dag-response-training-v1-20261029-cpu02-independent/`。独审没有修改作者源码，也没有启动 GPU、native collector 或真实训练 fit；作者的小型合成 CPU 求解器比较已运行。

旧版两处 P2 问题均有保留的精确源码和失败反例：同 hardware_id 改变 max_warps_per_sm 会改变特征，却不改变 bundle SHA；拟合期间外部可变 catalog/hardware/capture 被改写，会使旧数值行绑定新的身份。作者现在将完整 hardware hash 写入 bundle，consumer 强制校验；入口先执行 catalog 与所有 capture 的纯角色门禁，再 deepcopy(document, hardware, captures) 为同一快照，供数值路径和最终身份共同使用。13 项独立测试包含两反例及角色门禁早于 timing-label copy，修复后全部通过。旧 cpu01 报告保留，不覆盖。

编译器保留全 ordered offered registry 与精确 canonical GPU node bijection；重复 raw SHA 不能通过 capture/request 别名加权。每个 capture 独立 sweep，半开区间 [start,end) 与 endpoint 左极限正确；不同 node 即使 owner、request、name 相同也参与六个竞争特征。60 个随机 sweep 与直接 active-set oracle 一致，路径总时长等于完整响应。混 clock capture 全部 UNKNOWN 并保留分母，任何拒绝阻止 fit；不同 capture 无跨 capture 竞争。全部六列保留。

低内存 helper 共享原 profile、differential、local_minimum_certificate，保留三初值、trust-exact 与 LM root 方法及其容差。group-centered derivative 不创建 N×G dummy 矩阵。对 J=[Z,D]，中心化后 Z 与 Dc 正交，使用 min(sqrt(min count),sigma_min(Dc))/(1+||M||₂) 作为最小奇异值下界、||J||F 作为最大奇异值上界，保守比较原 NumPy 默认 rank 容差。界不充分即 false，projected rank 或成功 root 无法越过此门禁。120 个随机 dense oracle 验证上下界及无错误 full-rank 放行；6000 singleton node baseline 不能识别六 slope。作者的小型真实 CPU 求解与原 dense fit 的 beta、mu、variance、NLL、各 solver/root attempt 一致。

合跑作者 compiler13、helper5、consumer8 与独审13，共39项 unittest通过。98份完整实现闭包与测试 receipts 在测试前后未变化，原始 transcript 已封存。bundle 的 full hardware hash 与 consumer 的精确检查源码一并记录；consumer 的其他 propagation 合同由其独立审查负责。

此结果支持 declaration compiler 的 CPU 数值/身份合同。native_sources_authenticated 始终 false；declared measured 不是 native authentication，profiler lifetime 不是 SM service。尚无真实训练 bundle、独立整block预测结果、物理或部署资格。本审查没有触碰运行中的 baseline/controller。

复现：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/bin/python -m unittest tests.test_kernel_dag_response_training_v1 tests.test_kernel_dag_response_fit_v1 tests.test_kernel_dag_response_training_independent_v1 tests.test_kernel_node_clock_adapter_v11`。
