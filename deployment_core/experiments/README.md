# MILP seeder correctness validation

正式 seed 入口是 `deployment_core.seed`；本目录只保存验证实验。
`python -m deployment_core.experiments.validate_seeder` 对下面原有 19 个场景分别执行 MILP 和
exhaustive baseline，验证可行性、集合归属、最小显存目标和 routing 约束；另运行 guarded 20/30-model
synthetic 测试，结果写入 `results/deployment_seed/`。不重写已封存的 Stage 1.5 结果。

以下原有 Stage 1.5 sweep 仅为 exhaustive correctness baseline，不是正式 search path。

# Stage 1.5：Feasibility Sanity Sweep

运行固定的 19 个场景：6 个基准 workload scale、7 个指定 composition、6 个 rho_max。
仅调用 Stage 1 的 `screen_deployments`（也是 `generate_feasible_deployments` 的实现入口），
利用其计数和淘汰记录，避免第二次求解每个 LP。没有复制或修改 feasibility 算法。

```bash
cd /root/autodl-tmp/generalist-specialist
/root/miniconda3/envs/gsdeploy/bin/python -m deployment_core.experiments.stage1_5_sweep
/root/miniconda3/envs/gsdeploy/bin/python -m unittest discover -s deployment_core/tests -v
```

默认输出目录是项目根目录下的 `results/stage1_5/`，可用 `--output-dir` 指定另一目录。
每次运行会重写该目录下的五个报告文件：

- `load_intensity_sweep.json`
- `composition_sweep.json`
- `rho_sweep.json`
- `stage1_5_summary.json`
- `summary.md`

前三个文件记录每个场景的完整 arrival rates、rho_max、分层计数、可行集合、每个 routing
witness、utilization、淘汰原因以及验证失败详情。总 summary 保存数据来源和 profile 快照，
以及四类 routing 检查、VQA coverage 和两类集合单调性结果。

A/C 分别检查所有 15 对不同参数之间的集合包含关系，允许集合保持不变；不依赖数量比较。
D 独立审计每个返回的 witness：

- 按 Stage 1 `ROUTING_REL_TOLERANCE=1e-7` 检查每个任务守恒，绝对余量为零。
- 检查正 routing 只指向已部署且支持对应 task 的模型。
- 严格检查非负、有限值，不修复或裁剪被验证的 routing。
- 用原始 service time / 1000 重算 utilization，使用 `LP_TOLERANCE=1e-9` 检查上限，
  同时检查与 witness 保存的 utilization 一致。

任何 validation violation 或场景执行失败都会返回非零退出码，打印具体场景、deployment、
违反的条件或错误。场景失败不会被当成合法空集合，相关单调性/完整验证也不会标记通过。
若某个 witness 有 NaN/Inf，JSON 以字符串保存异常值作为证据，不生成非标准 JSON 数值。

Composition 的 specialist 分析只统计实际输出：含/不含 specialist 的可行数量，以及
“只加入该 specialist，且原组合因 Capacity 淘汰，新组合可行”的配对。具体配对保存在总
summary 中。空集报告为空集，不强制预期。LP witness 没有最小化 Qwen utilization，
因此不能把 witness 利用率当成最优卸载量，也不作 latency/throughput 推断。

新增测试包含同数量但不同集合的反例、rho 方向反例、错误 routing、非法 capability、负数、
NaN、伪造 utilization、漏请求、零负载泄漏、求解器错误及失败命令返回非零退出码。
真实 profile 的完整 sweep 作为集成测试运行；原 Stage 1 测试一起回归。
