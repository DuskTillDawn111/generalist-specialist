# 静态研究汇总入口

本轮正式执行现已完成：15/15条原生确认严格验证，最终汇总在
[`study-summary-final-v1/report.zh.md`](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/study-summary-final-v1/report.zh.md)。
独立原始算术复核390/390项通过，独立配对统计复核通过。
方法与结果解读见[研究说明](/root/autodl-tmp/generalist-specialist-project/docs/static_study_methods_and_results_20261022.md)。
下文保留汇总入口的接口和各阶段验证说明；其中“未生成”的表述指相应历史只读测试当时的状态。

独立脚本 `scripts/report_static_study.py` 的当前汇总 schema 为 v2，使用 Python 标准库，仅消费已有结果。
它不导入运行时、不加载模型、不启动 GPU、不搜索、不拟合，也不修改冻结 v4/v5。
正式 v5 六角色结果和效率结果完整后调用：

```bash
python scripts/report_static_study.py \
  --calibration-report /absolute/path/to/surrogate-v1/calibration-report.json \
  --validated-report /absolute/path/to/surrogate-v2/calibration-report.json \
  --search-dir /absolute/path/to/complete-six-role-search \
  --efficiency-report /absolute/path/to/three-algorithm-comparison/report.json \
  --output /absolute/path/to/new-study-summary
```

正式确认报告可选，用 `--confirmation-report /absolute/path/to/report.json` 添加。
输出目录必须不存在；生成 `report.json`、中文 `report.zh.md` 和输出 SHA `manifest.json`。

可补充一个或多个完整六角色搜索目录：

```bash
--additional-search-evidence /absolute/path/to/search-v2-v5-K4
```

额外证据逐文件核验，要求与主域使用相同冻结数学合同，只允许 K 和计算预算/停止参数不同，
且额外 K 不大于主 K、正副本域一致、双方均关闭 capacity witnesses。
K 使用分母 1..K 并集，因此在这些条件下 K4 候选属于 K8 域。
额外搜索结果、来源目录、bundle SHA、合同差异和纯部署候选完整嵌套保存，
**不会改写主搜索角色或已冻结 formal packet、也不会替换正式确认名单**。

v2 根据证书动态显示角色名称：没有纯通子域证书而预算耗尽时显示“纯通预算候选”，
不能写成“最优纯通”；其他角色同样处理。机器角色 ID 保持不变。
若额外 K4 的纯通候选好于主 K8 冻结 incumbent，报告保留两者并明确 K4 候选更强。
但 K4 的子域上界不能移用于 K8；排除纯通赢得全池预测最优必须使用主 K8 纯域上界，
并与全池已有可行方案目标值比较。即使排除了纯通赢得全池预测最优，
也不能因此证明纯通子域内当前候选最优。

输入遵循现有真实合同：

- v1 报告必须是 `gsdeploy.calibrated_surrogate.v1 / VALIDATION_FAILED`，完整保留失败结果、误差与排名检查。
- v2 必须是 `gsdeploy.calibrated_replay.v2 / VALIDATED_ON_HELDOUT_CONDITIONS`。
- 搜索目录必须包含 `solve_static_study.py` 写出的 `manifest.json`、`system-config.json`、
  `search-results.json` 和六角色单独 JSON。逐文件 SHA、bundle/单独结果一致性、七模型池、
  纯控制池、共同评分合同、v2 版本、六项效用算术和各角色方案定义均核对。
- 效率输入使用 `compare_calibrated_pool_solvers.py` 的 fresh-process 三算法合同，
  需要同目录 `preflight.json`、每轮 `run-N-{milp,dfs,plain_dfs}.json`。
  三算法进程失败也必须保留记录。逐文件与嵌入记录一致；重算同域合同、每轮目标/证书/方案身份、
  完成状态、墙钟与评价次数中位数、证书数和峰值 RSS。
- 如提供确认报告，核对其 packet/run SHA、原生确认阶段与执行身份、冻结搜索 bundle/候选/种子、
  六项实测效用、缺失作业、配对差值和精确符号翻转 p；不重跑原生导入器。
  Student 区间与 Holm 校正保留严格上游报告值，其假设和小样本限制仍适用。

主结果包括冻结七模型名称/角色、纯控制池、静态负载、六权重、K 与副本域、六角色预测 U/部署/
路由比例/六成本、证书/上界/gap/unknown/未解决候选、完整评价次数与墙钟。
有限预算未完成时保留 incumbent 和 gap；未知或无解角色不会被删除，也不填写零效用。

效率对比域单独报告，不能替代七模型全域证书。
只有三算法同域、同目标值、均有证书且来源/配置未变时，才输出
`certified_solve_speedup_rival_over_milp`；否则该值为 null，仅报告观测耗时比和各算法 incumbent/gap。
即使同优有证书，也只有两种竞争算法墙钟中位数都大于 MILP 时，
`milp_faster_than_both_at_certified_same_objective` 才为 true。
脚本不会因研究希望有优势而把预算时长比解释成求解加速。

未提供正式确认报告时明确标记 `NOT_DONE`，作业数、实测效用和配对统计为 null。
部分确认保留缺失种子，配对不完整时不允许推断统计量。
所有输出固定 `physical_global_optimality_certified=false`；不把代理模型证书写成物理全组合全局最优。

`python -m unittest tests.test_static_study_summary -v` 使用临时目录的合成 CPU fixture，
覆盖成功/预算耗尽/无方案/未确认/部分确认，以及 SHA、bundle、效用、种子、配对 p、
虚假证书/计时、缺失算法等篡改。测试输出明确 `CPU_FIXTURE_NOT_MEASURED`，无正式测量结果。
另对现有 v4 K4 六角色和 five-model pilot 做过只读合同解析，没有生成正式最终总报告。

v2 的九项 CPU fixture 测试另覆盖动态标签、额外更强纯候选、纯域上界排除范围、
主搜索文件/formal packet 不变，以及非嵌套网格/负载改变/来源 SHA 篡改拒绝。
真实 v5 K4/K8 只读核对结果：纯通预测 incumbent 分别为 -3.367339、-5.676148，
主 K8 纯通上界 -0.555303 低于全池可行 U=-0.309303；纯通子域仍无最优证书。
该核对没有生成新正式研究报告，没有改动正在运行的正式确认实验。

当前脚本、测试和本说明单独封存在
`artifacts/static-study-gpu-20261022/study-summary-source-v2/`。
先前 `study-summary-source-v1/` 和已有研究报告保持原样，不改变已有冻结执行快照。
