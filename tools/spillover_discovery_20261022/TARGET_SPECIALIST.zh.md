# 目标专用模型参数

四个现有工具现在均接受 `--target-specialist {asr,text,image,ocr}`，默认仍为asr：

- `solve_roles.py`：自由搜索仍覆盖原七模型完整测量域；仅 `best_one_specialist_max` 把指定目标的正副本数限制为1，仍允许0。其他模型的副本范围不变。目标字段不强制自由赢家一定混合。
- `materialize_confirmation.py`：确认物化核验目标专用至少2副本、该目标任务同时正流量路由至目标专用和通用；strong n1限制与选择的目标一致。结构未自然实现或证书不完整时仍拒绝。
- `benchmark_multireplica.py`：默认五模型子域中，所选目标的正副本范围为[1,2]，其余为[1]；显式 `--replica-options` 保持原样并接受既有测量范围核验。目标必须保留在模型子域。
- `verify_full_domain.py`：目标只是协议和参考证据元数据；仍使用所有七模型的全部测量副本数，不施加目标结构限制。

例如未来文本正式模型独立验证通过后，四步统一传入 `--target-specialist text`。当前失败/未验证预测器仍被原门禁拒绝；这个参数不能激活它们，也不能将有限text26开发候选直接包装成全池搜索赢家。

搜索包、物化报告、确认合同和候选包记录 `target_specialist`/`target_task`。两者必须一致。旧搜索包未记录目标时按历史asr解释，不能将旧asr strong n1证据重新标成text。确认consumer本来已按合同target_task核验strong n1，因此未改生产模块或冻结快照。

路由网格沿用实际求解器数学：每个任务的共同分母可以取1..K的并集，而非单独分母K。4/7和1/2都属于当前K8域；1/2也属于K7。物化逐条检查原始绝对路由流量守恒，并查找使该任务所有份额都为整数单位的共同分母d≤K，记录命中分母和单位，不改写或四舍五入实际路由。4/9不能以K8证书物化，4/7不能以K6证书物化。合同记录grid-only域，不接受额外capacity witness。默认K没有变化。

CPU测试：

```bash
python -m unittest tests.test_spillover_target_tools tests.test_replay_transfer tests.test_spillover_confirmation -v
```

新增测试使用明确标记的合成搜索/确认fixture：四目标结构替换、text strong n1实际参数传递、现有consumer对text n2伪装strong n1的拒绝、text合同物化、旧元数据错配、精确网格归属、FAILED门禁（含真实校验函数）均覆盖；无GPU、实际研究搜索或运行中raw读取。
