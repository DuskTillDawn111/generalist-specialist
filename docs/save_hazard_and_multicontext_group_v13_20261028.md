# 可训练 age-conditioned SAVE hazard 与六 context 统计归属

`tools/predictor_validation_20261027/save_hazard_v13.py` 已实现并实际拟合候选模型。它预测 RESTORE_START 到 SAVE_END 的 envelope 保存 hazard，不把 envelope 当 SM 服务，不是完整调度器。目标是在新 frontier 组合上使用已见可见特征组合预测，而不要求 exact-cell 永久存在。

固定年龄边界是 0、1、2、5、10、20、50、100、200、500µs，1、2、5、10、20、50ms 和之后的尾 bin。每 episode 按真实年龄暴露拆分；风险区间为 entry < t <= exit，边界 SAVE event 归左 bin。frontier 改变/请求边界仍为 censor，restore 年龄不重置。加性项是 holder、pending owner/task/pool/phase 集合指标和年龄 bin，无特征交互、无调参。到达后的响应文本、token 数或质量没有作为输入。

采用暴露单位 ms 的 piecewise-exponential Poisson 目标。capture 等权，capture 内 envelope 等权，episode-bin 权重固定为 1/(capture 数 × 该 capture 的 envelope 数)。同 envelope 的多段不是独立 capture。完整 exposure loss 为加权 Σ(exposure_ms × exp(η) − events × η)，另加固定 λ=1 的 0.5||β非截距||²，截距不惩罚。使用 SciPy L-BFGS-B；所有参数在训练内拟合。

实际输入是原 v12_4 model，通过 v12_5 的显式 `allow_v12_4_source_model=True` 对所有原 source 重推并逐字段比对，未使用 CPU fixture。保留 329093 episodes、35666 SAVE events、293427 censored episodes、17443207473ns 暴露；两个 capture 分别有 17688 / 18097 envelope。全源验证 32.71s，总流程 49.49s，9 次拟合全部收敛。

| 留出整个训练 capture | constant 完整 exposure loss | holder-only | age-additive |
|---|---:|---:|---:|
| 26305a33… | 0.29306653 | 0.28998505 | -0.53382279 |
| 98c4c261… | 0.27500191 | 0.27197847 | -0.55036822 |

越低越好。这里省略参数无关的 Poisson 常数，loss 可以为负；折外诊断不加训练正则项。两折所有原 episode 的暴露都参与评分，未知类别仍评分且标 OOD，没有移除数据；这次两折没有未见类别。该改善只支持这两个训练 capture 之间的诊断，不能声称独立预测准确率或概率校准。capture 内相关性和 censoring 假设尚未独立验证。

主模型 34 个参数、170 个原 holder/frontier 组合，预声明年龄网格产生 2720 次 candidate query。查询接口：

```python
import json
from tools.predictor_validation_20261027.save_hazard_v13 import SaveHazardCandidate

model = json.load(open('artifacts/predictor-save-hazard-v13-20261028-training01/age_additive-model.json'))
candidate = SaveHazardCandidate(model)
prediction = candidate.query(held_owner, pending_frontier, context_age_ns, horizon_ns)
```

返回 hazard_per_ms、conditional_CDF、restricted_candidate_CDF、暴露支持区间和 OOD reasons。exact frontier 未见本身不会禁止组合预测；新 owner/task/pool/phase 明确标 OOD，数值候选以未知系数零效应给出且不能当成已支持预测。age 支持使用训练暴露的实际并集，不填补 gap。CDF 条件是 frontier 保持不变，只积分到当前连通支持区间末端；超支持不报告完整 CDF，也不宣称完整 finite distribution。年龄 bin 尾部有部分暴露不意味着无限 tail 已支持。并集支持也不证明每个类别组合有年龄条件支持。

训练报告：`artifacts/predictor-save-hazard-v13-20261028-training01/report.json`，SHA256 `4ffb9c77f09803add5f2d0445b431b85bcecf92ee5e99003a18d82b814af184e`。`tests/test_save_hazard_v13.py` 的 10 项 CPU 测试通过，包括真实 329093 行完整计数、全部 source 收据和整 capture 分割。未启动 GPU。

## 六 context 能否按 scheduler process group 给 owner 候选

`multicontext_group_hypothesis_v13.py` 对两包全部六 context、全部 15 对 envelope、所有 kernel endpoints 进行了实际 CPU 检查。两包所有 paired envelope 两两无正长度重叠，原 12 个 priming 请求的 endpoints 仍唯一落对应 primary context。原四 context 的标准本来是控制 startup 的 temporal mapping 和独立 priming，对 extras 强制要求 native ID join 超出了原标准。

因此可以保留全部六个 opaque scheduler context keys，以同 scheduler globalPid 连接已归属 primary，建立显式可审的 transitive owner-group 统计假设；可用来训练多 context → owner 及 owner 内 restore，不应因此永远 UNKNOWN。它不能被提升为严格“只有自身 envelope 中才能执行 kernel”的规则。

| packet | 总 endpoints | group 候选一致 | uncovered | cross-owner | pairing |
|---|---:|---:|---:|---:|---|
| 01 | 2245454 | 2245045 | 409 | 0 | OCR primary 有 1 unpaired SAVE |
| 02 | 2245454 | 2245448 | 4 | 2 | 无未配对事件 |

packet02 的两个跨 owner endpoint 属于同一个 G2 kernel：profile:11、correlationId=3666105、84947133111–84947134135ns（1.024µs）。它完整落在 OCR primary envelope 84945430175–84948798192ns 内，距 restore ~1.703ms、距 SAVE ~1.665ms，距最近 G2 envelope ~1.706ms，不能用边界误差解释、不能忽略。保持这些反例和未覆盖事件，owner-group 可作为统计候选输入，严格执行支持仍有该明确缺口；不改标 G2，不裁剪 envelope，不发明 missing restore。

两包报告均保存所有异常明细和源码/原始 SQL 收据：

- `artifacts/predictor-multicontext-group-v13-20261028-packet01/report.json`：`d63a31b12f2337dfe757b9fe99602d10980e41cec8e429a2f80ad96c589b1863`。
- `artifacts/predictor-multicontext-group-v13-20261028-packet02/report.json`：`7080636abd858aa04e979ed8cdc2ebce123f3698eb1e47128c3f097ea9c165d0`。

本结论没有声称 native context ID 相等、OS namespace bridge 或 same-SM concurrency。六 context 新包尚未用于上面的双 capture SAVE hazard；该模型仍严格使用原 v12_4 全部 source。后续主路径还需要 next-owner、intra-owner/kernel-start 与 host/joint 模型接线和独立验证，不能把 hazard 损失改善直接当分流收益。
