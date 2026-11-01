# 独立确认统计审计

当前状态：**准备完毕，真实 15/15 报告待生成、待执行审计。** 临时 CPU fixture 验证通过不代表真实确认完成。

在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES='' python tools/confirmation_statistics_audit_20261022/audit_statistics.py \
  --report artifacts/static-study-gpu-20261022/confirmation-report-v1/report.json \
  --verifier artifacts/static-study-gpu-20261022/confirmation-plot-source-v1/scripts/plot_static_confirmation.py \
  --output artifacts/static-study-gpu-20261022/confirmation-statistics-final-audit-v1.json
```

输出必须是新文件，既有文件拒绝覆盖。脚本先要求严格报告 `ALL_FROZEN_JOBS_VERIFIED`、15/15、无缺失，再调用指定冻结 verifier 核对包 seal、run receipts、名册和六项效用算术，随后读取全部 seed-level U 独立计算四组竞争方案的配对差、均值、SE、95% Student 区间、精确双侧符号翻转 p、Holm 及六项平均差，并逐字段比较原报告。输出绑定所有输入文件和审计器自身 SHA。

统计实现不依赖 `gsdeploy.study_reporting` 或 scipy。Student 临界值由 df=2 的解析 CDF 反解；SE 由无偏样本方差计算；精确 p 枚举 8 种符号，保留上游 1e-12 的并列判定容差；Holm 单独实现。比较容差为绝对/相对 1e-10。该脚本不重新运行原生导入、GPU 或模型，严格实测导入由上游冻结 reporter 负责。

真实输入通过后才输出 `REAL_15_OF_15_STATISTICS_VERIFIED` 与 `full_statistical_audit_complete=true`；fixture 必须显式 `--cpu-fixture`，且无论通过与否都不会标记真实完成。三个 seed 的最小双侧符号翻转 p=0.25，不能由本审计声称达到 0.05 显著性。

自检：

```bash
CUDA_VISIBLE_DEVICES='' python tools/confirmation_statistics_audit_20261022/self_test.py
```

五项测试使用临时文件并在结束时删除。完整 fixture 共五方案、三种子，包含正差、混合符号、零方差和全零差；原冻结统计实现仅在测试中作为独立对照。测试逐项篡改全部 comparison 字段，也校验已存在真实四组 partial 必须拒绝。`validation.json` 记录本次通过结果；正式待执行状态保留在 `status.json`。
