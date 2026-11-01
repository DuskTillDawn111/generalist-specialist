# 三个 horizon 的一致风险曲线

当前正积压人口的目标是首次 reader-visible 等待队列清空时间 `T`。对同一 cutoff，`P(T>1s) >= P(T>5s) >= P(T>10s)` 必须成立。三独立 stump 模型的原分数可能违背此约束。

v35 不重训 v31/v32。预声明固定等权 L2 PAVA，把 raw 三分数投影到 `1 >= z1 >= z5 >= z10 >= 0`，最小化 `sum((zH-rawH)^2)`。原分数已单调时不变；例如 `[0, .5, 1]` 投影为 `[.5, .5, .5]`，`[1, 0, 1]` 为 `[1, .5, .5]`。无标签决定 pooling 或权重。

联合输出保留三 raw score/status、模型 SHA、最大原违例幅度 `max(0, raw_j-raw_i; i<j)`、投影调整平方和和 projected scores。三个分数必须来自同一 id/run、原 context、显式 model domain、cutoff、host-state SHA 与 waiting count；任一 H UNKNOWN、未支持或身份不同，joint 输出 UNKNOWN，projected scores=None，不补零。空积压人口单列无分数。非法范围、NaN/Inf 或缺 horizon 直接拒绝。

`seal` 输入三个 v32 预测封存报告，核验精确冻结 source/model/domain/feature receipts，并从实际 feature prefix 使用冻结模型重放验证每行，随后才做固定 PAVA；整个步骤不读取 labels。独立 `evaluate` 再 join 同一原 cohort 的三 H 标签，验证身份与整 grid，对 raw 和 projection 在相同支持集上分别报告 Brier，不按标签选择 winner。v33 会对四个留出 capture 分别执行这两步。原来每个 H 较宽支持集的单模型诊断照常保留。

这保证候选风险曲线的 horizon 一致性，不证明分数已校准或具独立实测准确率，也不改变 rho、六项评分、执行准入或全过程 rolling forecast。

8 个 CPU 测试覆盖完整逆序、相邻 pooling、原单调不变、未知分数/域、不同 cutoff/context/host-state、空人口、数值边界和 raw 不变；与既有观察/风险测试合计 57 PASS。预声明 CPU 手算样例和精确源快照保存于 `artifacts/predictor-joint-horizon-risk-v35-20261028-predeclared-cpufixture01/`，不属于 GPU 准确率证据。
