# Module 3A：LP-Guided Non-enumerative Deployment Search

离线测试：18 项通过。当前真实 5-model pool 的三个 workload 均仅做 dry run；GPU profiling=0。

| 模型数 | LP solves | Runtime evaluations（含 seed） | Visited | Iterations | 每轮最大候选数 | 全空间 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 11 | 8 | 8 | 4 | 5 | 32 |
| 10 | 26 | 9 | 9 | 4 | 10 | 1024 |
| 20 | 56 | 9 | 9 | 4 | 20 | 1048576 |
| 30 | 86 | 9 | 9 | 4 | 30 | 1073741824 |

5-model 独立 exhaustive baseline：J*=20.833333 ms；J_search=22.727273 ms；gap=9.090909%。只在 tests/small_space_baseline.py 中枚举 32 个状态，正式 search 未调用。

## 实现与边界

- 首步为 MilpDeploymentSeeder.solve；随后重解 canonical min-max LP，不使用 seed routing witness 指导 mutation。
- Bottleneck 按 LP pressure，pressure tasks 按瓶颈上的 demand contribution；ADD 仅考虑这些任务，REMOVE 用绝对 pressure 阈值及 coverage 检查。
- 一轮最多 M 个 one-hop candidates；coverage/memory/LP screening → delta_z 主排序 → Top-B。relief_per_memory 仅记录。
- Predictor=None 完整通过；不可用/异常 prediction 不阻止搜索；有值时只对相同 delta_z 作次排序。
- Runtime 稳定性三态优先；同为 steady 时比较 mean（绝对 ms tolerance）、weighted task P95、memory；同为非 steady 不宣称改善。
- 每轮最多接受一个 Top-B 候选；必须优于轮初 incumbent。其余测量保留为 ROLLBACK；visited/rejected 阻止重试。
- 测试覆盖无可行 seed、LP infeasible、solver error、canonical 与 2G 一致、rollback、budget、patience、iteration 和无候选停止。
- LP solve count 是实际 canonical LP 调用（含 seed 重解，不含 MILP 内部 relaxation）；重复候选缓存 LP。Runtime budget 含 seed；visited 含 seed 和已测量状态，不含仅做 LP 的候选。
- Synthetic latency 是人工离线 objective，扩展性表采用单任务异速模型池，不构成真实性能、全局最优或论文最终验证。O(M) 指邻域候选数，不是 MILP 时间复杂度。
- Canonical LP 在多最优解时不保证 routing 唯一；不扩展到其他 routing optimization。
- RuntimeEvaluator 接口完成；真实 GPU adapter 尚未启用，未来必须复用 joint_deploy/measurement。

Module 2 结论保留：cross-deployment regression PARTIAL；local bias calibration FAIL；ranking INSUFFICIENT COMPARABLE GROUPS。预测器不作为搜索正确性的必要依赖。

本阶段完成后停止；未进入 full real search、forecasting、multi-replica 或 final paper evaluation。

复现：`OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m deployment_search.validate`。
