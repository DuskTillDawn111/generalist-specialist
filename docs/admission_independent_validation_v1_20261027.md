# admission2 独立预测验证：固定 V4

本轮独立验证固定 seed `628201`，四条 admission2 路由各 1200 个请求，共 4800 个请求。它与两条训练 seed `628101/628102` 分离，所有验证请求、结果和派生报告均标记 `not_for_training`，不能回流拟合或触发新一轮调参。既有 shared 独立验证失败保留。

服务时间预测仅转移已冻结的 shared Gamma full16 模型；这是待验证的跨控制器假设，不使用 admission 或验证结果重拟合服务规律。Q、E、M、启动开销和原 `g(n)/n` 容量来自独立 admission2 T8 训练协议。资源模型必须同时匹配路由、控制器、输入规律与执行上下文，缺失支持保持 UNKNOWN。

入口为 `tools/predictor_validation_20261027/admission_validation_v1.py`：

- `prepare(...)`：只读取公开输入、固定模型和三个训练协议，生成 prospective 公共计划及协议；不读取训练/验证结果。
- `audit_protocol(path)`：校验全部来源、原阈值、固定 seed、原始六项效用常量、到达过程、输入生成及路由；即使同时重哈希原生准备计划和公共计划，也不能改变这些约束。
- `seal(...)`：严格校验 T8 三个 fit manifest，使用对应路由 Q/E/M 和固定 Gamma 服务模型预测全过程；冻结预测及原容量预测，须早于每次独立运行启动。
- `check_sealed(folder)`：重新计算四条路由的全过程预测，拒绝仅重哈希后伪造的预测值。
- `report(...)`：独立解析真实原生日志、重算质量并调用原报告器生成实际六项指标、P95、ΔU 和候选排序；原 gate 与容量资格分别附录，不覆盖原结果。
- `verify_report(path)`：再次解析原生证据，并重算原报告、gate 和容量资格。

独立实际映射不调用训练映射器或 shared 映射器。它检查原生调用、完整 1200 请求、READY/START/PHASE 与工作进程时钟、原控制器机制、priming/warmup0、同 alias FIFO/无重叠、原窗口与有界 drain，以及从启动到排空的进程归属显存/板级功率采样。原始 raw、计分注释、调用和计划的哈希在解析前捕获，解析结束再次检查，拒绝读取中途变化。质量注释只能增加原 scorer 的证据，不能改写原生记录，质量总量另行重算。

资格沿用原 20% 服务均值误差门槛（预测均值作分母）、原 rho `0.8 + 1e-8` 与原生稳定性规则。均值拟合通过不能覆盖排队不稳定；缺少完整 Q/T/L/P/M/E 或请求证据不能升级资格。原 `g(n)/n` 不是全局 admission permit 容量证明，`global_permit_capacity_qualified` 保持未知。

该 V4 只有一个独立输入 seed，验证范围是预先声明的有限已知输入，不主张未见输入总体泛化。此阶段不具有最终正收益资格：`final_gain_eligible=false`。后续固定 F8 仍需单独报告 shared/admission2 × 四路由对照，保持原稳定性和六项效用。

本模块的 CPU 合成回归检查验证协议、映射、预测接口和拒绝规则；合成数值不构成真实 GPU 干扰、物理容量或收益证据。prepare 阶段不启动 GPU，不用缺失实际数据生成 fit/seal。
