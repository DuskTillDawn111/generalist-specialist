matching owned producer 的 cpu02 已通过独立 CPU 契约审查。cpu01 发现的 generated-plan guard 缺口已由原作者修复；旧版本、prepared manifest 和缺陷复现均保留。此次独审没有修改作者源码，没有调用 native supervisor/collector，没有启动 GPU 或 fit。

cpu01 的 `execute` 只将 manifest、manifest.sources 和 claim 加入 SourceChangeGuard，manifest.sources 含 seed plans，却没有新生成的十五个 plans。独审在隔离临时文件及真实 execute/CPU stub supervisor 中证明，当前 plan、未来 plan，或首次完整 audit 期间的 plan 改写再恢复均可逃过 guard。独立加载作者保留的旧 execute 精确 bytes 后，三个期望 fail-closed 的断言全部失败、无测试错误；证据为 `artifacts/predictor-twoFIFO-matching-owned-producer-v1-20261029-cpu01-independent/report.json`。这是执行证据完整性缺口，未声称已经构造出最终 native semantic review 可接受的伪造。cpu01 不可作为本次修复后执行版本。

作者的修复版 execute SHA 为 `455e96738123f3996408de77dc0da7b4c192f6749953c620b3023f5f5c47f665`：先取得 receipt-checked manifest snapshot，首次完整 audit/plan 读取前建立包含全部 generated plans 的外层 guard；在 guard 内复核 manifest/plan，O_EXCL 创建 no-retry claim，再建立内层 claim guard。两层 guard 持续覆盖 supervisor 的 before/after_run，外层还核全部 future plans。独立反例现在全部通过，首次 audit 改写也在创建 claim/调用 supervisor 之前被拒绝。

新 prepared manifest 为 `artifacts/predictor-twoFIFO-matching-owned-producer-v1-20261029-cpu02-prepared/manifest.json`，SHA `a7b5853c8469fb888d0fab3a7142aa114226423a0f88e32a19eb4fe7c47fad06`。已实际调用纯 CPU `audit`，完整保留 15 attempts、3 blocks×5 shards、171 cases、390 measured 与270 priming，共660唯一请求；所有 claim 和 collection 均不存在。完整 receipts 在测试前后再次核验，未变化。新独审证据目录为 `artifacts/predictor-twoFIFO-matching-owned-producer-v1-20261029-cpu02-independent/`。

作者22项加独审13项共35项 unittest通过（43.174s）。独审新增覆盖：current/future generated plan change-and-restore、首次 audit watch 顺序、claim 在 supervisor 前持久化且 prelaunch 失败后禁止重试、lease 在读取 plan/构造 runtime 前拒绝、第七条 priming 失败保留全部18条及所有 measured rows、monitor close 异常仍关闭 worker并保存失败、独立 parent receive 时钟 FIFO、review拒绝停止后续执行、每个cleanup维度缺失均拒绝、CPU raw不能进入native身份、startup历史owned峰值不被case局部峰值替代、measurement energy排除drain，以及真实macro partition保留priming events同时排除priming labels。全部 runtime/supervisor 控制路径使用显式 CPU doubles；它们不能证明 CUDA/模型/监控硬件实测行为。

静态审查还确认：原生 review 要求新 training schema、完整 audited plan/context/pool/source绑定、executed-cases 精确相等、全 offered/priming 分母、连续历史和 checkpoint ledger；完成后才重解析原始输出与原 exact input/spec/protocol，核四 owner READY/干净 shutdown、supervisor claim/cleanup及每poll owned ancestry。CPU注入标记不可成为native资格。资源检查由 telemetry gate 将 measurement_windows 精确绑定每case端点，要求固定10s measurement与60s drain；startup能耗一次计入启动至首case（含priming），case能耗只计measurement，drain只诊断。显存使用从startup至相应drain的owned sampled cumulative peak，包含未活动resident及历史，未加安全系数，也不宣称连续峰值。

macro adapter先重新执行native review，使用完整attempt events；priming不进入训练标签。intake重新执行每份review、macro和资源算术，核三完整五shard blocks、390+270、十五startup及171资源窗口；leaveout单位是完整block，raw SHA/case/priming不算独立replication。报告明确旧fit接口尚不兼容，未拟合、未读validation labels，kernel/resource fit、native prediction accuracy与gain资格均为false。

没有找到修复版其余CPU审查路径的新阻断。此结论仅支持新producer准备合同及拒绝门禁，实际native evidence尚不存在；未证明真实overlap、12特征残差秩、数值root、整block留出支持、90+120s资源迁移、USHER/Roomie kernel响应或分流收益。运行中的GPU baseline/controller及冻结native/serial闭包均未由本独审修改。

复现命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/bin/python -m unittest tests.test_twoFIFO_matching_producer_v1 tests.test_twoFIFO_matching_producer_independent_v1
/root/miniconda3/bin/python -m tools.predictor_validation_20261027.twoFIFO_matching_execute_v1 audit --manifest artifacts/predictor-twoFIFO-matching-owned-producer-v1-20261029-cpu02-prepared/manifest.json
```
