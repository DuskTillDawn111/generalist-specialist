# 完整开发诊断 oracle

实现：`tools/spillover_context_20261022/development_oracle.py`。它独立于生产模型和冻结执行源，使用现有真实逐输入画像及开发拟合，供有限开发目标的自主搜索和 DFS 使用。主 `U` 始终为 UNKNOWN；显式 `diagnostic_extrapolation_only.U` 才是本轮可比较的数值目标，不具备独立验证或物理最优资格。

```python
from tools.spillover_context_20261022.development_oracle import DevelopmentOracle

oracle = DevelopmentOracle()
result = oracle.evaluate(
    {"deployment": deployment, "routing": routing},
    seed=107,
    diagnostic_extrapolation=True,
    include_requests=False,
    strict_context_checks=False,
)
diagnostic = result["diagnostic_extrapolation_only"]
score = diagnostic["U"]
planning_feasible = diagnostic["planning_feasible"]
oracle.verify_inputs()
sources = oracle.provenance()
```

`evaluate_plan` 是同接口别名。实例缓存原 baseline、任务到达/输入、域检查结果；缓存不改变数值。搜索方应保存源码和来源 SHA，终态再执行 `verify_inputs()`。请求完整逐行证据可启用 `include_requests=True`。`domain_summary()` 返回模型状态和已覆盖路径，不能替调用方声明有限搜索域。

算法输入接受原 catalog 中任何模型和已画像的 resident n，没有“必须专用/必须混合”的角色限制。路由必须按任务守恒、输入有限非负；未支持 resident n（即使空闲）属于无效 plan。有效模型池中的 G1/G3 等缺少局部服务拟合时，方案保留 UNKNOWN，不把它们用角色规则剪掉。

目前 exact local path states 包括 text n1/n2、G2 n1 的四任务、OCR n1、image n1、ASR n1/n2；个别路径训练输入不全。text1/2+G2+OCR、纯专及其邻域可以得到开发数值诊断，完整七池并没有覆盖，当前搜索不能证明七池最优。

计算顺序保持原语义：原 `make_trace` → 原 phase-local `SamplePool` → 原 `expand_routes` 和浮点 SWRR → 每 alias 独立 FIFO。测量开始假设暖机已排空，与原运行合同相同。服务点用 `b * exp(Xβ)`，b 是精确 family/task/n/input 的原独立画像成功服务时间中位数；默认 `family_self_other` 来自已有开发选择，未重新拟合或根据本轮结果选型。此点预测未做 smearing，仍是几何/中位尺度。

rho 在完整 input pool 上先对逐输入服务点做**算术平均**，再计算 `sum(q * arithmetic_point_service_ms / 1000) / n`。服务拟合 target 已经是 exact n 下实际 worker service，不再乘第二次 `n/capacity_factor`。构造 A_self/A_other 特征时仍沿用冻结原 baseline/capacity_factor。算术平均点预测不等于已校准真实服务算术均值；它只用于本轮明确标识的诊断规划限制。

Q 调原 `predict_quality`，通过实际逐输入和 SWRR 重放历史输出；它是质量损失，越低越好，不能冒充本次并发原生输出。T 是在正式窗口内未完成比例；L/P 是原任务到达率加权 mean/p95 latency 除原 reference。M 默认 `M_full`，safe factor 恰好施加一次。E 默认 `E_full_window` 加 `E_full_startup`，原归一化 `(window_watts + startup_joules/3600)/450`。六项 U 使用原 config weights，不添加质量绝对门槛。

`planning_feasible` 仅指开发服务点算术 rho≤0.8、拟合 M≤1 和确定性 FIFO 可在原 drain 内排空。`resource_feasible` 是兼容别名。两者都不替代原测量稳定性门禁，不是执行准入或资源安全证明；`actual_stability` 保持 UNKNOWN。

严格域保留原 local path/n/input 和冻结 A_self/A_other/M/H 凸包；资源候选逐实际特征检查凸包，同时保留冻结未见部署组合 UNKNOWN 规则。`strict_context_checks=False` 仅省去严格凸包计算；严格结果写 `NOT_CHECKED_DIAGNOSTIC_ONLY`，主 U 和 `supported_domain_development_arithmetic_U` 都为 null。测试确认这一选项不改变诊断目标。

诊断外推只对有 included local path/n 和精确 b 的输入计算数值，不填缺失 baseline、缺路径或缺系数。缺失源返回 UNKNOWN；已有源 SHA 不符立即报错。服务 UNKNOWN 会使该 alias 的 FIFO 后缀 UNKNOWN，不把可算前缀当完整六项评分。所有外推均独立标识 `valid_prediction=False`、`execution_admission=False`，不能用于认证。

107 和 108 都是已有结果被观察过的开发 seed；本 oracle 本身不读运行 raw 或队列，也不把这些 seed 称为独立验证。调用方必须同时保存完整有限域、全部候选 UNKNOWN/不可行情况、搜索及 DFS 的目标一致性，并将“有限开发诊断目标最优”与实测或物理最优区分。

验证：11 项 CPU 测试覆盖真实 trace/input/alias/Q、六项权重、算术 rho 与无二次 factor、FIFO 等待及 UNKNOWN 后缀、缺 baseline、不支持路径、域外数值隔离、非法 n/inf route、来源变化和 fast 模式数值不变。另有 6 类方案×2 seed 的完整 smoke 收据，保存在 `artifacts/static-spillover-study-20261022/development-complete-oracle-cpu-v1`；未据此选择模型或宣称最优。
