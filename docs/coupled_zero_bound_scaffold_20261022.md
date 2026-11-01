# 路由／负载耦合模式的零界安全占位

这一步只实现搜索安全接口，**没有新模型、拟合参数、服务函数、M/E预测、GPU数据或新的研究准入要求**。`route_load_replay_v1` 完整预测始终 UNKNOWN，不能生成正式可执行最优方案；当前 GPU 开发实验继续使用冻结 v5，不受影响。旧失败 `replay_transfer_v1` 不修改、不重封、不激活。

先前“surrogate_bounds返回None即可使用零界”的建议不完整，现已纠正。需要同时处理的危险旧回退为：

- `milp.py` 的 profiled_memory 旧模式列表可能为新模式使用 standalone state／column／master 显存界；`pressure_service→family_service→service_lower_ms` 可能用旧服务删除候选；diagnostic_mode 可能把未验证新资源当作只缺 E 的诊断。
- `optimizer.py` 的 profiled_memory 旧列表控制部署枚举显存预筛和 upper；`pressure_service→family_service→catalog standalone` 即使 separable_service关闭，路由递归仍累加旧 demand。
- `config.py:validate_plan` 旧逻辑可能把新模式原 additive memory 当成已验证完整值；完整预测与正式执行不能依赖这种回退。

现在 `gsdeploy/coupled_mode.py` 定义显式能力协议 `gsdeploy.coupled_zero_search_bounds.v1`。识别此新模式后，MILP/DFS 均禁用旧服务/pressure/显存/能耗非零预剪枝、容量 witness、MILP缺能耗诊断模式和跨部署 route feedback；只有原始固定历史 Q 可作相同任务路由的精确质量界。服务、pressure、M、E 下界全部为0，仅用于非负成本松弛，不是实际服务或可用容量。

配置为防止模式拼写或隐式升级，需要显式占位：

```json
{
  "performance_mode": "route_load_replay_v1",
  "memory_mode": "route_load_replay_v1",
  "route_load_replay_v1": {
    "schema": "gsdeploy.route_load_replay.placeholder.v1",
    "status": "UNIMPLEMENTED"
  }
}
```

缺占位、其它 schema 或伪造 VALIDATED 状态直接拒绝。符合占位的配置可审查有限搜索域，但 `resolve_service` 明确 `ROUTE_LOAD_MODEL_NOT_IMPLEMENTED`，输入schema不支持则 `UNSUPPORTED_ROUTE_LOAD_SCHEMA`。`validate_plan` 返回 memory/capacity 未验证、无实际rho/安全显存；公开 `predict` 保留 UNKNOWN，公开 `run` 拒绝执行。没有通过缺实现来默认删除模型或给有限域最优证书。

每种算法结果的 domain.coupled_bounds 包含模式、四项零界、禁止standalone回退、关闭witness/diagnostic以及“完整预测未实现”声明。现有模式仍使用原合同。

CPU测试 `tests/test_coupled_zero_bounds.py` 5项覆盖：

- 原standalone显存和服务故意极大，真实config/resolve_service/predict/run门禁仍为未知且拒绝执行，不能误认不可行或真实零资源。
- 明确验证每种搜索器所有相关零界和关闭项，UNKNOWN候选被保留，不能成为已不可行裁剪或闭合证书。
- 通过测试注入的非分离完整context oracle检查每个B≤C、DFS上界覆盖U、MILP/有界DFS/plainDFS同优，且不执行跨context route反馈。
- 存在可行incumbent但另有context未知时，三种算法保留未决而不给最优证书。

上述oracle是完整人为CPU whole-plan service/memory/energy profiles，并调用原公开 profiled `validate_plan` 与 `predict_scenarios` 核实真实完整路径。只有求解器评分回调注入此oracle；**生产占位仍全部UNKNOWN**。这验证搜索数学安全，不能证明新真实模型已实现、通过验证或可以预测真实分流。

后续新模型要单独提供版本化完整context服务及资源解析、精确输入回放、原rho语义、明确声明域和独立验证；若要强剪枝，还需显式可证明的非零下包络。不能把本占位改成旧常数服务重用后声称完成耦合建模。
