# 已知输入的独立探针验证与全过程验证接口（2026-10-27）

v6 统计模型的已知输入探针通过了预先声明的局部门槛；全过程队列、父进程收到结果的 P95、六项效用差和候选排序仍须另外验证。这里不宣称最终分流收益，也不把局部通过扩展到未见输入。

## 已完成的独立探针

工件：`artifacts/predictor-gpu-20261027/statistical-validation-v6/report/report.json`。v5 训练画像拟合统计模型后封存预测，再独立执行 v6 的 144 个批次、354 个请求；验证数据未用于重新拟合。

- 事先声明的已知输入子域为 177 个请求，预测覆盖 177/177；service 平均绝对相对误差 3.9009%，MAE 28.8258 ms，预测减实测偏差 +8.6497 ms。
- 两任务 P95 的平均绝对相对误差为 0.7769%。OCR 18 个请求：实测 2084.7820 ms、预测 2066.2672 ms；text 159 个请求：实测 2641.7596 ms、预测 2659.3467 ms。
- 已知输入子域满足事先声明的 100% 覆盖、service MARE ≤20%、任务 P95 MARE ≤20%，局部门槛为 PASS；9 个请求使用明确标记的阶段组合插值。
- 另 177 个未见输入仍为 UNKNOWN。完整 354 请求分母下只有 50% 覆盖，整体报告仍是 INCOMPLETE。

v3 精确上下文模型的独立报告仍是 354/354 UNKNOWN；没有改写旧封存预测或旧报告。v6 属于依赖正则化的经验模型，欠秩和活跃边界仍被记录；不能把预测误差小解释为物理系数已经识别。

探针阶段的延迟终点是 worker END。最终业务 E2E 的终点为父进程 completion_time_ns；两者之间的响应尾部不得从最终延迟中删除。上述 P95 通过仅限已知输入探针，不构成父进程全过程延迟通过，也不证明能量/显存预测已通过。

## 新增的全过程验证接口

`tools/predictor_validation_20261027/flow_validation.py` 提供 CPU 封存与评估命令，不启动 GPU：

```bash
python -m tools.predictor_validation_20261027.flow_validation seal \
  --plan PLAN.json --model MODEL.json --resources RESOURCES.json \
  --quality TRAINING_AGGREGATE_QUALITY.json --output NEW_SEALED_DIR
python -m tools.predictor_validation_20261027.flow_validation report \
  --sealed-dir SEALED_DIR --raw CANDIDATE_SCORED_RAW.json \
  --raw OTHER_CANDIDATE_SCORED_RAW.json --output NEW_REPORT_DIR
```

计划声明完整未来到达、输入 public identity、每请求静态路线、有限候选、队列采样时点、六项常数及执行源码摘要。预测不读取实际未来活动或答案；这是已知到达输入下的静态候选检验，不能据此声称在线策略已经完成物理验证。报告重新核验所有来源、候选与 cohort 身份；完整 offered 请求作为分母，未运行候选、未完成请求和缺失证据继续保留。

实际 service、START 等待与父进程响应尾部分开报告；队列深度指已到达而父进程尚未收到结果的请求数，包含正在执行的请求。L/P 使用父进程实际收到结果的时间。Q 使用每任务汇总的 numerator/denominator，再按 offered rate 加权；OCR/ASR 不取逐请求 CER/WER 的均值。M 为全部拥有进程原始采样峰值乘原 safe_factor 一次，包含 drain；E 仅含测量窗口板级积分与从 startup_begin 到测量开始的启动积分摊销，排除 drain 能量。原稳定性、rho 与显存资格单独保留，算术可计算不等于允许宣称收益。

预测器没有训练启动功耗时，绝对 U 保持 UNKNOWN。只有计划显式声明同驻留、同执行后端的共同启动比较组，才允许预测比较中省略共同启动项；实测 ΔU 仍包含每次运行的真实 startup 差，报告单列其差异。未声明时不生成完整 ΔU/排序结论。

下一步须先完成含全部四任务背景的独立全过程预测检查，再比较默认共享与整请求准入。MPS、共同 batch、公共权重子矩阵及缓存复用须有相应执行后端和独立真实证据，不能从当前统计模型引入假定折扣。最终四组长窗口对照尚未启用。
