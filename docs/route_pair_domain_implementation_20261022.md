# 可选细分流路由域实现与 CPU 闭环验证

本次只扩展有限搜索域，不修改真实服务预测、rho≤0.8、质量、六项效用、内存、稳定性或任何冻结实验源。未运行 GPU。实现已接入公开配置、MILP、有界 DFS、plain DFS、SearchSession、静态强对照候选生成和合成 benchmark。

## 精确定义

旧配置不需要新字段，默认 `route_pair_denominators_by_task={}`，旧域保持为共同分母 1..K 的全部非负模型族份额，K 仍限制为 1..16。

新配置示例：

```json
{
  "route_grid": 8,
  "route_pair_denominators_by_task": {"text": [32, 64]},
  "milp_capacity_witness": false
}
```

对 text 的每一对可执行模型族、每个指定分母 d，增加全部 u/d 与 (d-u)/d，u=1..d−1。这同时包含通用/专用、通用/通用、专用/专用，不按角色优待任何一对。其它任务不增加细网格；三族及以上组合继续使用原 K 域。纯方案及原副本选项全部保留，未强制混合或专用≥2。32、64 是族间份额分母，族内仍由原 SWRR 按副本均分。

例如专用×2、通用×1，族间份额 63/64 与 1/64，128 个 text 请求的静态 SWRR 计数是专用两个副本各 63，通用 2。系统无需修改 SWRR 即可执行这些权重。有限测量窗口可能让极小份额只有少数样本，实际覆盖必须另外审查，路由域支持不代表预测器已验证。

唯一候选生成器为 `gsdeploy/route_domain.py:fraction_rows`。所有网格首先以 Fraction 精确约分去重，再乘任务到达率；新增两族网格不会生成高分母的多族 composition。配置接受已知任务名及整数分母≥2，拒绝 bool、float、未知任务和非 list 形式。配置 load/as_dict/运行继承均保留此字段。

## 证书与 witness

三种算法的证书增加 `route_domain_schema=base_grid_plus_pair_grids_v1`、基础 K、新分母映射、pair extension 规则和精确去重说明；仍明确 `continuous_optimality=false`。相同配置传入三种算法，只有界和遍历方式不同。

容量边界 witness 的历史语义未改变：MILP 和 DFS 原先存在各自的 witness 行构造方式，新功能没有统一、重定义或取消这些行。需要严格同域比较时必须显式使用 `milp_capacity_witness=false`。新增细网格不能让调用方把启用各自 witness 的比较冒充为同一网格域。

## CPU 结果

新增 `tests/test_route_pair_domain.py` 的 9 项测试覆盖：旧域与独立笛卡尔枚举完全一致、精确去重与嵌套、三族域不扩张、配置兼容与拒绝错误值、完整域 oracle 与三个算法一致、保留纯路由及单专用控制、SearchSession 透传及旧冻结源兼容、SWRR 具体请求计数、旧证书显式拒绝新域。

完整 synthetic toy 有 133 个原始部署/路由方案，六项效用、完整人为 whole-board energy 证据。人为设置的唯一最优点是专用×2＋通用×1、通用份额 1/64；这是覆盖测试的输入设计，**不是现实研究发现，不能证明实际分流收益**。独立逐方案打分、MILP、有界 DFS、plain DFS 的最优 U 都为 −0.1696070075758196，各算法都完成并给出有限域证书。

单进程一次描述性计时：MILP 0.559 秒、DFS 0.182 秒、plain DFS 0.154 秒；MILP 完整预测 25 次、两个 DFS 各 133 次。这里 MILP 实际更慢，不能据此宣称效率优势；真实效率结论仍需同域、冷进程、多重复正式实验。

机器可审查记录：`artifacts/static-spillover-study-20261022/route-pair-domain-cpu-verification-v1/report.json`，包含源码 SHA、域参数、逐算法证书、计时口径和明确的 synthetic 标记；`oracle.json` 保存全部方案的 U，`synthetic-system-config.json` 保存完整测试问题。

验证命令及结果：

```text
python -m unittest tests.test_route_pair_domain tests.test_static_study_workflow tests.test_static_study_summary -q
22 tests, OK

python -m unittest tests.test_milp tests.test_static_dfs_v6 tests.test_milp_witness tests.test_search_session tests.test_spillover_target_tools -q
40 tests, OK
```

以上 22＋40 是初次验证。随后增加第 9 项旧冻结源回归：当前 SearchSession 读取旧 source config 时移除空的新字段，不向旧求解器传新 kwargs；非空新域以及手工 options 注入均明确 `SOURCE_ROUTE_DOMAIN_UNSUPPORTED`，避免默认旧流程因新代码无法运行。原 v1 CPU 记录保存的是首次源码 SHA；后续兼容修复 SHA 与验证另存在 `source-compatibility-amendment-v1.json`，不覆盖首次记录。

兼容修复后执行 `python -m unittest tests.test_route_pair_domain tests.test_search_session tests.test_static_study_workflow tests.test_static_study_summary -q`，30 项全部通过。

## 旧工具边界

下列旧工具仍只理解 K 网格，收到非空新域配置即以 `PAIR_ROUTE_DOMAIN_UNSUPPORTED_...` 显式拒绝，不能用其旧证书标签包装新路由：

- `tools/spillover_discovery_20261022/benchmark_multireplica.py`、`verify_full_domain.py`、`materialize_confirmation.py`、`solve_roles.py`，以及 `target_specialist.py` 的旧 membership 入口。
- `scripts/count_static_memory_columns_v6.py`、`report_static_study.py`、`solve_static_study.py`、`static_search_validation_v6.py`、`compare_calibrated_pool_solvers.py` 的旧域审计入口。

新域研究应使用公开 `DeploymentSystem`/`SearchSession` 与共享生成器，并采用新版本 domain-aware 结果整理及实验包。`scripts/run_search_session.py` 和 `scripts/solve_static_session_study.py` 通过已更新 SearchSession 使用完整新域。冻结旧源码不改，旧模型证据的失败准入状态不变。

这一步完成的是“算法可选择小份额且三种求解同域”的实现闭环。真实七模型池的服务证据、细份额下有限输入覆盖、预测器新域验证、自由最优搜索、强替代方案和 GPU 确认仍须独立完成。
