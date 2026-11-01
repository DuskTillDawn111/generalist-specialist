# 独立确认结果绘图

绘图入口 `scripts/plot_static_confirmation.py` 只消费
`report_static_confirmation.py` 严格导入、重算效用后的 report.json。
它独立检查 report 引用的 packet SHA、packet 自身 seal、六角色和候选/作业完整名册、
每条 run 的 SHA/原生确认身份，以及六项加权成本和 U 的一致性。
它不重新运行原生导入器，也不要求当前工作目录与冻结执行快照相同。
结果 manifest 保留 report、packet、每条 run、绘图脚本和所有输出文件的 SHA。

正式完成后调用：

```bash
python scripts/plot_static_confirmation.py \
  --report /absolute/path/to/strict-confirmation-report/report.json \
  --output /absolute/path/to/new-confirmation-figures
```

默认只接受 `ALL_FROZEN_JOBS_VERIFIED`。显式加入 `--allow-incomplete` 才允许部分结果，
所有图都标出未完成状态；该选项不放宽 SHA、身份、效用或名册验证。
尚无测量的候选保留占位，缺失种子列出；搜索无可行解的角色保留 NO PLAN 和搜索状态，
不会冒充尚待测量的作业。输出目录必须不存在，避免覆盖证据。

输出三个图组，每组包括 PNG、SVG、PDF：

1. `confirmation-utility`：每个候选的 43/44/45 种子 U 及已观测种子的算术均值。
2. `confirmation-costs`：Q/T/L/P/M/E 六项成本的已观测均值；它们之和等于负的平均 U。
3. `confirmation-selected-plan`：冻结选中方案的副本数、按任务的路由比例、完整六角色对照表。

`candidate-roster.json` 记录图中 C1 等编号对应的完整方案、角色、缺失和已观测种子。
均值不附带未经计算的误差条；统计区间及配对检验仍以严格确认报告为准。
所有图只比较冻结且已测的方案/种子，不能作为物理全组合全局最优证明。

当前入口只做过 CPU fixture 验证，没有生成或声称真实确认结果。
`python -m unittest tests.test_plot_static_confirmation -v` 在临时目录构造数据并删除，
检查九个格式文件、未完成/无方案角色、packet/run 篡改、错身份/阶段、重复/未知作业、
错误种子、效用/成本/计数和虚假全局最优声明等。
fixture 绘图必须传 `--cpu-fixture`，所有图及 manifest 明确标记
`CPU FIXTURE - NOT MEASURED`；该开关不绕过数据验证。

绘图脚本和测试另存于 `artifacts/static-study-gpu-20261022/confirmation-plot-source-v1/`，
独立 manifest 封存 SHA，不改变冻结 v4、算法或现有报告。
