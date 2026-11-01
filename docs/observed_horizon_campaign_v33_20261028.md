# 四个自然 capture 的风险诊断与最终候选封存

`observed_horizon_campaign_v33.py` 在四个 actual natural 原 context 全部资格通过后运行，不修改已冻结的 v30/v31/v32。固定使用 1/5/10 秒目标、40 棵 stump、learning_rate=.2、minimum_leaf=2。

先调用 v31 whole-capture 训练诊断，每次以三个完整 run 训练并留出第四个。随后将每个已冻结 fold 模型保存成单独文件，用 v32 对留出 capture 的 reader-visible prefixes 封存预测，再通过独立 evaluate 路径 join 标签。12 个 fold 的 coverage、UNKNOWN、Brier、AUC 和训练类率共同支持集 baseline 都保留。重放核验不会拟合模型或依据留出标签选参数。

在任何外部验证之前，预声明 v35 的三 horizon 联合输出：对同一 prefix 的 1/5/10 秒 `P(first clearance > H)` 原分数做固定等权 L2 PAVA 非增投影。三原模型不重训，raw score、违例最大幅度、投影调整平方和与投影后三分数并存。同一原 context、domain、cutoff 和 host-state 必须一致且三 H 全部支持，否则 joint UNKNOWN；不补零。空积压人口单列无分数。每个留出 capture 报告 raw 与固定 projection 在完全相同支持集上的 Brier，并保留单 H 原有更宽覆盖的诊断；不会看标签选择 raw/projection winner，不称已校准。

最后每个 H 使用全部四个 training capture 拟合一次最终候选，保存 3 个模型。最终候选不在自身训练结果上评分；四 run 共享 recipe/content、窗口相关，fold 结果只构成训练家族诊断，不证明独立输入泛化或概率校准。

```bash
python -m tools.predictor_validation_20261027.observed_horizon_campaign_v33 \
  --risk-domain <四actual成员通过的risk-domain.json> \
  --visible-report <natural balanced proof/report.json> \
  --visible-report <natural skewed proof/report.json> \
  --visible-report <natural spill_low proof/report.json> \
  --visible-report <natural spill_high proof/report.json> \
  --output <新输出目录>
```

未资格域在创建输出前直接返回 UNKNOWN。最终候选绑定当前四原 context 和完整 domain SHA。新独立 run 的 context 不能自动继承该域：必须另行预声明和严格比较执行投影、在评分前只使用 READY 信息核验实际运行配置，再由新版本显式扩展域绑定；训练数值参数不能因此重拟合。

`horizon_target_context_audit_v34.py` 已比较现有 v11 独立目标与 whole57 natural context。只移除 nonce、route 和明确声明的 training_only/validation_scope 标签后仍有 collector/schema、捕获/监视范围、task-domain、源集合等差异，因此输出 `UNKNOWN_CONTEXT_EXECUTION_PROJECTION_DIFFERS`。两域 290 个共同原源路径 SHA 均相同、execution/residents/threads 等配置相同，不足以证明完整投影等价。审计未读取目标 runtime、service、queue 或 labels，也未重写原 context SHA、扩展模型域或修改冻结模型。
