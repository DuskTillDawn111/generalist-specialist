# Admission V4 同域 CPU 搜索核验

新增 `tools/predictor_validation_20261027/admission_search_check_v1.py`，只在 admission V4 的独立报告完成后，由主对话调用一次；不启动 GPU，不自动等待，不修改固定 F8 方案。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
python -m tools.predictor_validation_20261027.admission_search_check_v1 \
  --sealed-dir artifacts/predictor-gpu-20261027/admission-validation-v1/sealed-v1 \
  --validation-report artifacts/predictor-gpu-20261027/admission-validation-v1/report/report.json \
  --output artifacts/predictor-gpu-20261027/admission-same-domain-search-v1
```

入口为 `run(sealed_dir, validation_report, output)`。输出目录必须不存在。在解析和验证前捕获 report、validation-audit、seal-audit、predictions 的原始文件摘要。完整 `verify_report` 重新核验独立报告的原始证据、评分、原门槛、预测封存及三套训练清单一次；验证凭据必须指向同一 report，验证后四份入口文件也必须与原摘要一致。后续通过已审计的 receipt 读取对应清单，避免重复训练审计。每条路线使用自己确切的 admission/controller 资源模型、质量模型及启动能量/显存峰值覆盖值，无 shared fallback。plain DFS 和 bounded DFS 分别重新调用原 `admission_validation_v1.predict`，使用相同完整 V4 到达与输入、原六项效用、显存预算和 drain 限制。

已有封存预测仅供来源核验，搜索目标来自 fresh rollout。V4 观测仅用于既定适用性与稳定性准入，不用于拟合或选择效用，不读取 F8 实际结果。每一遍搜索后核验来源闭包，发生漂移则不写输出。

报告区分模型可行域与原资格可行域。原资格需同时满足独立数据完整性、统计服务适用性、统计 rho、原稳定性、全局 permit 容量、原 measured-profile resolver 适用性和原 catalog resolver rho。缺失项保持 UNKNOWN；任一已知不满足则排除。统计 transfer rho 不替代原 resolver 准入。所有路线均被明确排除时报告 EMPTY_DOMAIN，无最大值或最优证明。

上界沿用原六项非负成本的证明：已知训练质量的 Q 下界及 safe×该路线声明的 prephase peak/budget 的 M 下界，其余成本下界为零。严格 `upper < incumbent - 1e-12` 才剪枝，以保留所有容差内并列最优。模型和原资格分别维护 incumbent，禁止用不满足原资格的模型最优直接剪去真正的原资格最优。训练资源量只是经验估计，不是物理保证。

输出 `summary.json`、`plain-dfs.json`、`bounded.json`。`maximum_known_U` 描述已知候选；存在未决候选时 `certified_maximum_U` 和 `certified_all_tie_ids` 保持空。`original_gate_constrained_optimality_certified`、`physical_optimality_certified`、`final_gain_eligible` 固定为 false，global permit 容量证明固定 UNKNOWN。仅检查四条预留静态 admission 路线，不能推广到部署、复制数、动态路由、资源份额或 batch/cache 后端的全局最优。

CPU 测试：`python -m unittest tests.test_admission_search_check_v1 tests.test_admission_search_check_independent_review -v`，16 项通过；包括两遍 fresh rollout、原资格独立 incumbent、严格界与并列最优、UNKNOWN 与空域、路线模型缺失禁止 fallback、原 rho 不升级、来源漂移拒绝写结果，以及独立审查添加的验证后 report/audit 漂移、错误 report 验证凭据反例。

2026-10-27 06:16 UTC：主对话已通过独立的轻量 CPU 等待命令预约上述确切入口一次（PID92597，记录 `admission-same-domain-search-schedule-v1.json`）。命令绑定原 V2 PID91553/start603392070、协议和评审 SHA；只等原 V4 完整4800完成包，不采集或重试 GPU。运行前后重新核对已审源码 SHA，CUDA 为空、线程数1，搜索本身仍未开始；输出目录不得存在，失败不自动重试。

2026-10-27 10:24 UTC：原预约任务正常完成，输出三份结果，状态PASS。plain/bounded各4次fresh预测、最优及全部并列一致；模型内四路线唯一最优admission2_spill_high，U=-2.7305572882982583。0剪枝、0节省调用，不作加速声明。原资格域UNKNOWN，low/high尚未证资格，balanced/skewed由统计rho排除；不认证原门槛受限或物理最优。主线程直接核对与子对话独立小报告复核分别见artifacts/predictor-gpu-20261027/admission-search-final-direct-checkpoint-v1.json及admission-search-descriptive-review-v1.md。未重跑搜索。
