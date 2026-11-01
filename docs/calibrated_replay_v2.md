# 独立逐输入服务回放 v2

实现位于 `gsdeploy/calibration_v2.py`，schema 为 `gsdeploy.calibrated_replay.v2`，performance/memory mode 和 catalog key 均为 `calibrated_replay_v2`。它不改变 v1 的常数服务、拟合或自动 driver；正式启用需要显式选择 v2、保留 v1 的验证结果并完成新的独立验证。当前准备和合成测试不代表真实 v2 已拟合或通过。

## 数学合同

每个 `(model,task,n,sample_id)` 的 baseline 是冻结 standalone 成功观测的 service 中位数；预测按运行时 `SamplePool` 的 `seed:samples:task` 打乱、循环顺序，先取样本，再交给相同 Router，最后对各 replica 执行 FIFO。样本选择独立于部署和路由。全局服务比例只用 development 请求的实际 worker service 总和除以相应 baseline 中位数总和。

rho 明确定义为完整冻结输入池的逐输入中位数等权均值乘流量、全局比例后除以 `n`；它是冻结预测器的运行约束，不是有限窗口实际忙碌时间估计。队列下界只能使用最小逐输入服务或同请求集的无等待回放，不能用 pool mean 代替。E 保留 v1 的原始 standalone mean 特征和非负整板 NNLS（最多三个系数）；M 保留 `max(multiplier>=1 * additive, development observed peak)`。

缺 sample/path 不回退 aggregate，也不读取 validation 的 worker 时间填预测。validation worker 时间仅用于误差比较。失败或未验证模型只供查看，正式 U、搜索界和 confirmation 均不得启用。

## 身份和独立性

原 v1 八条条件（包含已经查看的旧 holdout）显式标记为 `legacy_v1_development`。新两条条件来自 `v2-holdout-prepared/experiment-packet.json`，H600、seed46/47；43/44/45 始终保留给 formal confirmation。每个 source 独立绑定其原执行 packet/hash/runtime 版本。protocol 绑定模型结构、阈值、原 v1 protocol、逐输入 baseline、开发/验证角色以及数学实现的相对路径和内容摘要。fit 重建并验证这些合同，不允许查看 holdout 后修改阈值或结构。

Raw request/protocol hash 与 quality 协议收据分别保留，不假定相等。OCR 历史 raw 缺 hash 时，仅在 profile.model 等于 worker、job.spec 等于 worker、job 样本 payload 等于冻结 runtime input、媒体哈希和推理源哈希匹配时，允许 `historical_profile_input_binding`。对应 request/raw protocol hash 保持 null，协议等价明确未验证；正式 confirmation 必须评分真实输出。无法建立输入对应关系的路径为 UNKNOWN。

spec 内对冻结 baseline 重算独立摘要并与 protocol 对照；修改样本表或来源绑定、仅重算外层版本摘要，仍不能继续使用已验证状态。评分输入/quality、baseline、safe factor、完整 GPU/checkpoint context、负载变化均拒绝。

## CPU 命令与公共接口

在最终冻结执行源码 v4 中准备正式 protocol；工作区的 `replay-v2-prepared` 是中间 smoke revision，不能当作最终启用依据。

```bash
python scripts/fit_static_replay_v2.py prepare \
  --base-packet artifacts/static-study-gpu-20261022/execution-packet-v2.json \
  --legacy-v1-protocol artifacts/static-study-gpu-20261022/calibration-protocol-v1.json \
  --holdout-packet artifacts/static-study-gpu-20261022/v2-holdout-prepared/experiment-packet.json \
  --output artifacts/static-study-gpu-20261022/replay-v2-prepared-final
```

`prepare` 只读取 standalone、输入和执行收据，不读取实际校准请求，也不拟合。独立的 `fit --prepared ... --runs-root ... --output ...` 在全部十条完成后才拟合；缺失时返回 WAITING，不生成拟合配置；重复成功条件要求 `--runs` 显式选择。上述命令不会启动 GPU，输出 revision 不可覆盖。

下界和执行集成使用：

- `validated_replay_spec(data,rates)`：完整版本、状态和适用域门禁。
- `input_service_ms(spec,model,task,n,sample_id)`：已乘全局比例的逐输入服务。
- `replay_path_statistics(spec,model,task,n)`：`minimum_ms` 供安全下界，`mean_ms` 供 rho。
- `service_lower_ms(data,rates,model,task,n)`：经过正式门禁的最低服务，未知返回 None。
- `replay_auxiliary(...)`：E/M 低参数预测。

CPU fixture `tests.test_calibrated_replay_v2.fitted_replay_fixture(quality_mode=...)` 返回真实 v2 fit 产生的合成 `SystemConfig`，支持两任务、n1/n2、aggregate 或 per-sample quality；不是 GPU 实测证据。
