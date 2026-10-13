# Module 3B：Real-runtime Closed-loop Search Pilot

状态：RUNTIME_EVALUATION_BUDGET_REACHED；decision audit=PASS。单 workload：W1_balanced。

1. Seed deployment：qwen，由本次 MILP 得到并重解 canonical routing。
2. 初始 bottleneck：['qwen']；z=0.312392558。
3. 每轮生成候选数：iteration 0 = 4, iteration 1 = 4。
4. LP Top-B：iteration 0: qwen+whisper, qwen+resnet50; iteration 1: qwen+resnet50+whisper, paddleocr+qwen+whisper。
5. 真实运行 states：qwen; qwen+whisper; qwen+resnet50; qwen+resnet50+whisper; paddleocr+qwen+whisper；每个 state 三次有效重复。
6. Accept/rollback：iteration 0 qwen+whisper: ACCEPTED (stability_improved); iteration 0 qwen+resnet50: ROLLBACK (another_top_b_candidate_preferred); iteration 1 qwen+resnet50+whisper: ACCEPTED (mean_improved); iteration 1 paddleocr+qwen+whisper: ROLLBACK (another_top_b_candidate_preferred)。
7. 最终 deployment：qwen+resnet50+whisper。
8. Weighted mean E2E：12997.344 → 84.679 ms；改善 99.35%。
9. Weighted task P95：27894.682 → 149.793 ms（不是 pooled system P95）。
10. Stability：recoverable_congested → steady；采用三次重复中的最差状态。
11. LP solves：8（含 seed canonical LP，另有 1 次 MILP；审计未重解 LP）。
12. Unique runtime evaluations：5 / 6。
13. 真实 run attempts：16 / 18；其中外部关机中断 1 次，不作 runtime 稳定性标签。
14. 重复的完整测量：0；关机中断后补测 1 次；cache hits=6（包含最终状态的显式缓存复用检查）。
15. Enumeration calls：0；未运行真实 exhaustive baseline。
16. 初步支持范围：闭环执行、预算、LP pruning、runtime 接受/回滚与可重建审计通过。本轮接受了真实改善。

## 重复波动与解释边界

Seed mean sample std=21975.910 ms；final mean sample std=7.744 ms。
Seed 的第三次 mean 约 38.4 秒，显著抬高其三次平均值；99.35% 是这批原始测量的样本内改善，不能解读为稳定、可泛化的常态加速比。
Latency 接受需至少 5% mean 改善；稳定性改善优先。未进行统计显著性检验；顺序运行和环境漂移仍可能影响结果。
所有 repetitions、worker errors、incomplete、drain timeout 与不稳定结果均保留。外部关机中断单独列账，不作模型 unstable；有效重复补足后聚合。三态使用既有 stability_v2 阈值，无 predictor 或人工候选补选。
历史 W1 的 15 次运行均 steady，是选 workload 的依据，不保证本次稳定。历史 raw 缺少模型权重内容哈希，无法认证 exact-cache identity；未冒充本轮缓存命中。
**Real exhaustive optimum = unavailable**。历史数据只覆盖 seed 与少量一跳配置，且没有完整来源哈希；未为 ground truth 新跑 31 个 deployments。
这是单 workload、小预算 pilot；不证明全局最优，也不构成 final evaluation。Module 2 predictor 继续为 optional，主实验 predictor=None。

已停止于 Module 3B；未进入多 workload、routing optimization、multi-replica、forecasting、predictor retraining 或 paper final benchmark。

## 关机中断修正

Host power-off during qwen+resnet50 repetition 1, no harness raw result. External interruption is not a model stability label.
原配置上限 6，修正后可测上限 5；实际物理运行尝试 16/18。
关机中断的原日志、进程元数据、适配器失败占位与旧 trace 存于 interruption_archive/；它没有进入三次有效重复的稳定性聚合。
