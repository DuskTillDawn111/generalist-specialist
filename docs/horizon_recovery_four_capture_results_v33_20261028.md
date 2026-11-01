# 四个自然训练 capture 的 horizon 风险诊断结果

已完成 12 个整 capture 留出诊断、各模型 reader-visible prefix 预测封存与独立标签 join、4 组三 horizon 固定 PAVA 诊断，以及 3 个全部四训练 capture 的最终候选冻结。此结果共享原 recipe/content，只是训练家族诊断；独立新执行结果另行报告。

四个 capture 共 5528 个 prefix，其中正积压 4370 个、空人口 1158 个。1/5/10 秒正积压目标类 0/1 分别为 425/3945、1096/3274、2045/2325。精确 recovery 被删失但完整观察 H 未清空时仍是已知正类。

| 留出 route | 正积压 | 被评分 | 覆盖率 | 1s Brier / baseline | 5s Brier / baseline | 10s Brier / baseline |
|---|---:|---:|---:|---|---|---|
| balanced | 1093 | 725 | 66.3% | 0.0595 / 0.1103 | 0.0810 / 0.1710 | 0.0927 / 0.2463 |
| skewed | 1115 | 144 | 12.9% | 0.1894 / 0.0721 | 0.1240 / 0.1106 | 0.1131 / 0.2478 |
| spill_low | 1076 | 1075 | 99.9% | 0.0387 / 0.1128 | 0.0923 / 0.2189 | 0.0830 / 0.2543 |
| spill_high | 1086 | 508 | 46.8% | 0.1062 / 0.2493 | 0.1695 / 0.3720 | 0.1454 / 0.2667 |

baseline 是训练类率，在与候选完全相同的坐标支持集上评分。支持外保持 UNKNOWN。skewed 只有 12.9% 支持，且 1s/5s Brier 劣于 baseline；不能宣称全域有效。其余三 route 也有覆盖差异，不能将 scored 子集误作全人口准确率。

| 留出 route | raw 三 H 违例 prefix 数 | 最大违例幅度 | 三 H raw 平均 Brier | 固定 PAVA 平均 Brier |
|---|---:|---:|---:|---:|
| balanced | 159 | 0.2545 | 0.0777 | 0.0762 |
| skewed | 81 | 0.3540 | 0.1421 | 0.1341 |
| spill_low | 142 | 0.2969 | 0.0713 | 0.0705 |
| spill_high | 280 | 0.3031 | 0.1403 | 0.1367 |

PAVA 为外部验证前固定的等权 L2 非增投影，不拟合、不依标签选择赢家。联合平均损失变小不意味着每个 H 均改善；原三分数和逐 H 结果全部保存。输出仍未校准，原 rho、六项效用和执行准入均未改变。

完整报告：`artifacts/predictor-observed-horizon-campaign-v33-20261028-four-natural-training01/report.json`；SHA256 `fcddb4010fa8bb07f2a4c771c64c97823cca8e83b8ddca9c7d02c023e8774fa3`。最终候选只绑定当时的四训练 context；新独立执行使用预声明目标扩展与实际 READY-only 资格核验后的纯 metadata rebind，所有 fit 数值和 provenance 不变。

这不是在线 rolling forecast、GPU kernel 服务预测、P95 或 ΔU 排序资格，也不构成分流收益证据。
