# Kernel 竞争特征 v2：训练身份与预测身份一致

v2 独立保存在 `tools/predictor_validation_20261027/kernel_competition_features_v2.py`，保留 v1 源码和旧审计证据。v1 的两个 CPU 反例已封存在 `artifacts/predictor-kernel-competition-features-v1-20261029-cpu01-independent-findings/report.json`：不同 device 的重叠 kernel 被计算为 peer，以及 owner a 使用 family b 的 profile 在训练侧被接受、预测侧被拒绝。v2 在生成特征前拒绝这两类输入。

v2 输入 schema 为 `gsdeploy.kernel_competition_training_capture.v2`。除 v1 的完整 offered request IDs、canonical kernels、owner PIDs、训练角色、profiler 与 context 外，必须提供：

```python
hardware_binding = {
    'hardware_id': catalog.hardware['hardware_id'],
    'device': 0,
    'device_identity_basis': 'canonical_profiler_device_index',
    'profiler_id': capture['profiler_id'],
    'execution_context_sha256': capture['execution_context_sha256'],
}
aliases = {'a': {'family': 'family-a', 'supported_tasks': ['text']}}
scenario = 'mean'
requests = [{
    'request_id': 'r1', 'alias': 'a', 'family': 'family-a',
    'task': 'text', 'pool_id': 'training-input', 'profile_id': 'exact-profile',
    'execution': {'batch_size': 1, 'backend': 'exact-backend', 'weight_reuse': False},
}]
```

`requests` 的顺序和 request IDs 必须与 offered registry 完全一致；aliases 与 owner PIDs 必须覆盖同一 resident roster。每个请求经真实 `KernelResourceProvider.plan` 校验 family/task/pool/profile/execution；每个 kernel 再核验 request→owner/profile/segment，并通过预测使用的 `_active_segments` 校验。所有 canonical launch 的 device 必须等于 capture 显式绑定的单一 device，避免跨 GPU 编造 SM 资源竞争。该 index 只在明确的 profiler/context 内解释，不能当作未验证的进程局部 CUDA ordinal。

这些仍是输入声明，`native_sources_authenticated=False` 保留。真实接入必须由独立 native intake 认证 device 到实际硬件的映射、请求与 worker 身份、完整 launch cohort、时钟与 profiler 域；仅填写这些字段不会获得物理资格。

六个静态 covariates、时间断点、半开区间、kernel 先后顺序、空隙与数学拟合路径保持原含义。训练用实际 kernel lifetime END 构建路径；预测只用 catalog 与显式 active set，不读取目标未来 END，也不调用 occupancy service。静态 mixed-block 冲突仍不是 achieved occupancy、吞吐或 compute/DRAM 分类。多 stream/context 或同 owner 重叠仍整 capture UNKNOWN；缺 trace 保留 offered 分母；结构 rank 不足时不调用 optimizer、不删列。v2 没有接入冻结 twoFIFO collector/runner，也没有运行 GPU 或新的数值求解。

CPU 验证命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest -v \
  tests.test_kernel_competition_features_v2 \
  tests.test_kernel_competition_features_v1 \
  tests.test_kernel_competition_features_v1_independent_counterexamples \
  tests.test_kernel_provider tests.test_kernel_sequence_audit_v2
```

v2 的 10 项新测试包含两个修复反例、硬件/context/profiler 绑定、请求轴错配、时间路径切换与空隙、整个 capture UNKNOWN、结构秩与无优化门禁。旧 v1 反例测试继续记录旧版本接受错误域的历史行为。
