# Paper Baseline and Ablation Evaluation

正式数据入口：`baseline_cpu_results.csv`、`ablation_cpu_results.csv`、`solver_efficiency.csv`、`gpu_baseline_results.csv`。结论、限制与八个研究问题见 `paper_ready_findings.md`。本轮是冻结算法的实验收尾，结束后停止。

- 12个主CPU单元，96条基线；原生20A/V3与有限网格控制分别保留。原生连续域的严格Regret为空，其选中方案有限成员参考差另列。
- A是固定旧部署的候选覆盖诊断；冻结实现没有独立候选扩展开关，不能称为保持全局DP不变的消融。
- B是固定35部署的路由组件诊断；同一DP核心仅限制任务束mask到合法最快路径。不能称为全局部署搜索开关消融。
- C没有可独立关闭的共置预测组件，明确跳过。不存在通过改rho、质量或显存制造收益的对照。
- GPU复用34三个同域冻结配置对的18个独立窗口。完整原始请求、轨迹与再审计结果均保留；同配置方法别名共享一次测量，不虚增样本数。
- 原历史U+J枚举计时保留，新U-only枚举计时单独输出；3次冷缓存中位数/min/max，完整U评价、4情形预测、任务束尝试和搜索状态单位分开。

每图均含PNG、PDF、CSV。`paper_figures/figureN_raw_data.json`列出可在本目录 `raw_data/`找到的完整原始文件和SHA256；共享数据不重复拷贝。GPU原始JSON保留历史路径字段，副本及轨迹均在 `raw_data/gpu/`。`artifact_sha256.json`为最终产物清单；`history_integrity_after.json`检查旧结果/代码SHA256与mtime。历史暂存缓存不纳入该清单。

复现CPU必须使用新目录，不能覆盖当前冻结结果。原始实验目录仍需存在，服务画像/质量准入从冻结清单读取。

```bash
export PAPER_EVAL_OUT=/root/autodl-tmp/generalist-specialist/results/paper_baseline_ablation_replay
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/gsdeploy/bin/python /root/autodl-tmp/generalist-specialist/results/paper_baseline_ablation/work/evaluate.py
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/gsdeploy/bin/python /root/autodl-tmp/generalist-specialist/results/paper_baseline_ablation/work/finalize.py
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/gsdeploy/bin/python /root/autodl-tmp/generalist-specialist/results/paper_baseline_ablation/work/count_audit.py
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python /root/autodl-tmp/generalist-specialist/results/paper_baseline_ablation/work/plot.py
```

`work/resume_timing.py`仅用于恢复缺失计时，不重测或覆盖已经完成的重复。首次计时进程在完整12单元完成前消失；已有原始记录保留。恢复驱动的一次缩进错误保留了日志；另一次恢复进程在完成一条重复后消失，原因未记录。随后将进程输出写入文件并仅恢复缺失重复，只修复运行与记录机制，优化器和科学协议未改变。相关事件见 `work/timing_resume_event.json`。实际计时批次属性保留，不把它们当作统计运行时间置信区间。

最终Gate区分实验产物已完成、A/B全局开关不可用、C跳过、GPU局部支持与不确定，不以一个成功状态掩盖这些边界。
