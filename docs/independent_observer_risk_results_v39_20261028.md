# 四组新执行的独立 reader-visible 风险结果

四组独立新执行均在第一条 priming 前通过完整 READY-only 资格核验，随后运行原 12 条 priming 和 57 条 measured cohort。使用四个原训练 capture 冻结的三个模型，扩展域绑定只修改 domain/model SHA；模型数值与训练 provenance 完全不变。所有 prediction 都先封存，随后独立导出 labels。

| route | 正积压 prefix | 被评分 | UNKNOWN | 1s raw/PAVA Brier | 5s raw/PAVA Brier | 10s raw/PAVA Brier |
|---|---:|---:|---:|---|---|---|
| balanced | 1092 | 1092 | 0 | 0.03629 / 0.04093 | 0.05855 / 0.05558 | 0.06547 / 0.06032 |
| skewed | 1115 | 1090 | 25 | 0.02919 / 0.02268 | 0.06454 / 0.06547 | 0.08709 / 0.08974 |
| spill_low | 1096 | 1096 | 0 | 0.04029 / 0.05154 | 0.09458 / 0.08925 | 0.08262 / 0.07336 |
| spill_high | 1085 | 1085 | 0 | 0.04203 / 0.05367 | 0.09294 / 0.08662 | 0.09265 / 0.08155 |

共 5528 个 prefix，4388 个当前正积压 prefix；4363 个得到三 H 支持（99.43%），25 个 UNKNOWN 均来自 skewed 的 text#1 最老等待年龄超出训练坐标范围。空积压人口 1140 个不制造低风险分数。

| H | raw Brier | 固定 PAVA Brier | 训练类率 baseline（同支持集） |
|---|---:|---:|---:|
| 1s | 0.03695 | 0.04221 | 0.09161 |
| 5s | 0.07765 | 0.07423 | 0.18821 |
| 10s | 0.08194 | 0.07623 | 0.24880 |

PAVA 是任何独立验证之前声明的等权 L2 非增投影。1s Brier 变差、5s/10s 变好；raw 与 PAVA 全部保留，不看标签选赢家。AUC、reliability bins、支持覆盖、原 horizon 违例与完整证据链均在每组评价文件。没有重拟合、阈值调整或删除失败/UNKNOWN。

这是 same recipe/content 的 fresh execution **causal receipt replay**：可以说明当前 host 可见积压的 horizon 清空风险在这四次新执行上的误差；不证明新输入/新计划泛化、概率校准、在线 rolling forecast、GPU kernel 服务进度、全过程 P95 或 ΔU 排序资格，也不构成分流收益证据。原六项评分、rho、执行准入不变。

执行前 66 项 CPU 检查通过，执行期间及最终核验 104 个冻结源字节均未变化。统一交付：`artifacts/predictor-independent-risk-validation-v39-20261028-delivery01/report.json`，SHA256 `30844517faf73e28e5b040cb1eea6a49737845425d0bfd6be34dac3e146ed183`。
