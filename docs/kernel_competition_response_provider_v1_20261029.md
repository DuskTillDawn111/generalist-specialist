# Kernel 竞争响应时钟接入现有全过程引擎

新增 `tools/predictor_validation_20261027/kernel_competition_response_provider_v1.py`，没有修改冻结采集闭包、`KernelResourceProvider`、v2 特征器或 `FullHorizonModel`。这是 CPU 可执行的接口接入；真实逐 kernel 训练资格仍然缺失，不能把接口完成说成真实预测精度已通过。

`training_bundle(catalog, captures)` 直接调用 v2 的完整训练编译器与 `fit_response`。不另写似然求解器、不删六个特征，也不使用 macro fit 充当 kernel fit。bundle 显式绑定 catalog 内容 hash、hardware、execution context、单一 profiler、训练 capture 身份和完整实现源码闭包。输入的 native 身份仍只是声明，`native_sources_authenticated=False`。结构秩或数值求解失败完整保留；provider 在缺合格 fit 或缺 baseline 时保持 UNKNOWN。

`KernelCompetitionResponseProvider` 对每个原引擎生成的 active kernel set 调用同一个 v2 特征函数。既有拟合的响应时钟含义是：

```
reference work = integral exp(-beta * x(t)) dt
service_ms at current state = reference_ms * exp(beta * x(current))
```

因此，原 `FullHorizonModel` 的 remaining fraction 就是剩余 reference work 的比例。它继续处理所有未来到达、FIFO 排队、segment 完成和邻居切换，并在每个生成边界重定价；没有读取目标请求的未来 kernel END 来安排边界。执行中的 cursor 必须沿用原引擎要求的当前 segment 与 remaining fraction 证据，不能仅用已耗时猜测已完成比例。

`mu_group` 是基准 log-duration 均值。默认 `mean` reference 为 `exp(mu + sigma²/2)`；预声明的 quantile reference 为 `exp(mu + sigma * Phi_inverse(q))`，其中 q=0.5 才是 `exp(mu)`。这些是确定的 reference-clock 场景，并不等于随机系统的平均最终响应，也不是已拟合的联合请求分布。没有独立抽取每个 kernel 随机数、拼成一个假称已识别的 joint request draw。baseline nuisance 保留 owner/profile/segment/context/profiler 五个身份轴。

provider 的 `plan` 将 GPU segment 的工作单位改成相应拟合 reference duration，保证队列工作量与事件进度使用同一时钟。静态特征接口仍需原 catalog baseline 身份，因此 service 调用前仅将 active row 的 duration 字段转换回原 catalog 声明；真正的 family/task/input/profile/segment 和静态配置由原 v2 gate 再核验。初始化时快照 catalog；每个 kernel 边界不重建或 hash 整个 kernel 序列。

使用接口：

```python
from gsdeploy.forecasting.simulator import FullHorizonModel
from tools.predictor_validation_20261027.kernel_competition_response_provider_v1 import (
    training_bundle, KernelCompetitionResponseProvider,
)

bundle = training_bundle(catalog, complete_kernel_training_captures)
provider = KernelCompetitionResponseProvider(
    catalog, bundle,
    allow_unvalidated_profiled_response=True,
    scenarios={
        'mean': {'catalog_scenario': 'mean', 'reference_functional': 'mean'},
        'median': {'catalog_scenario': 'mean', 'reference_functional': 'quantile', 'quantile': .5},
    },
    non_gpu_provider=explicit_training_bound_host_or_gap_provider,
)
forecast = FullHorizonModel(aliases, provider).forecast(
    observed_queues, now_ns, measurement_end_ns, 'mean',
    future_arrivals=full_remaining_arrival_trace,
    route_future=declared_continuation_policy,
    drain_deadline_ns=drain_end_ns,
)
```

默认不允许这种未独立验证的 profiler response 使用，CPU fixture 还必须另外打开 `allow_cpu_fixture`。host/sync/cleanup 不自动按进度 1 处理，必须给显式、同 execution context 的 non-GPU response provider。其调用收到原 catalog active rows 与 catalog scenario；gap/host 真实服务仍由该 provider 负责。板级功率和 owned 显存继续分别交独立 provider；没有相应 provider 就 UNKNOWN。这里没有从 occupancy 推导速度、compute/DRAM 分类、功率或显存，也没有实现 shared batch、公共权重子矩阵、cache 命中或硬件份额控制。

当前真实训练还缺：

1. 同 catalog/context/profiler 下的完整共跑 kernel cohort，具有 v2 capture 的 offered 请求、device/owner PID、request→profile→segment 和精确静态 launch 配置绑定，由 native intake 独立认证。
2. 每个 owner 内可由当前串行 active-segment 引擎表达的顺序。已有 `ordered-kernel-profiles-v2-20261028-training01/02` 都指向同一个原 capture，不是两次独立启动；二者仅转换 8 个请求中的 6 个，两个 OCR 请求是 `NON_SERIAL_STREAM_DAG_REQUIRED`。不能丢掉 OCR 后声称四 owner 全域接入。
3. 足以在 baseline nuisance 后识别全部六个斜率的训练变化、数值求解与完整训练块留出。现有 synthetic rank rejection 是 rank 3 / 参数 8，不能删列或采用任意系数继续预测。
4. host/gap 的显式响应 law，以及 profiler lifetime 到真实无 profiler 请求端点的独立预测验证。当前 source-rebound native 宏采集没有逐 kernel END，不能直接产出这一 bundle。
5. 若要给计算型/DRAM 型实证标签，仍需对应配置的真实吞吐 counters；上述静态协变量不能替代。cache/共享权重收益仍需执行后端和独立测量。

CPU 验证：11 项新 adapter 测试 + 17 项 v2 特征测试，共 28 项通过。解析场景验证两个同时开始的 100/200 reference 时钟在 peer 倍率为 2 时于 200/300 ms 完成；50 ms 未来到达改变 active set，FIFO 第二请求随后进入；另测 cursor 剩余工作、mean/median、未知 fit、host gate、source/catalog 身份和边界性能契约。测试中的合格系数由显式 synthetic stub 提供，只验证接口和已知时钟积分，不构成真实拟合或预测精度证据。另一个测试使用真实 v2 结构秩门禁，确认不合格 fit 无法预测。
