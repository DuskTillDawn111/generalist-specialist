# 新画像到全过程队列预测的接入

`twoFIFO_source_rebound_joint_law_v1.py` 接受本轮 source-rebound 宏文档；不把它们改名成旧 schema。它核对完整 12 次执行、3 个 block、每 block 4 个分片、546 个 measured 请求及 216 个 history-only priming 请求，随后复用已有的请求向量、协议时钟和 paired transport 数学方法。宏行数由数据计算，不能固定为 546。每个请求的 entry/forward/OCR/tail、通知与 transport 支持集一起传递。

`twoFIFO_source_rebound_generated_v1.py` 连接新拟合入口、联合请求模型、原全过程 forecast 与 queue-envelope companion。原 draw 方案保持每个 donor block 四个固定 slot。每次整 block 留出只使用另两个 block 的 8 个 donor attempts；完整 150 cases / 546 requests / 1200 worlds 保留。每 case 先保存 prediction 和 queue envelope，再提取 reader-E2E 标签计算误差。

外层 `run` 通过 `audit_fit` 复用已经保存的一次数值拟合，并通过 `with_verified_bundle` 重新核验真实 exact12 intake；它不重新拟合。拟合 root、nuisance 或 OCR 支持未通过时，仍遍历完整案例，全部世界以 UNKNOWN 输出。队列范围不填补点预测或 M/E，也不参与效用排序。六项权重、rho、g(n) 和稳定性规则均保持原定义。

这份报告属于训练域整 block 留出诊断：完整 intake 和拟合审查已经读过训练域的留出标签，不能称为 prospective 独立预测验证。每次生成只使用公开 calendar、immutable pre-case snapshot 和 donor law。startup / intercase orchestration 尚未被预测；资源完整评分、忙态数值校正、校准恢复风险及物理 USHER/Roomie 支持仍有缺口。

本轮预声明入口：

* 拟合 manifest：`artifacts/predictor-twoFIFO-source-rebound-phase12-fit-v1-20261029-cpu02-prepared/manifest.json`
* 生成设计：`artifacts/predictor-twoFIFO-source-rebound-generated-v1-20261029-cpu01-prepared/design.json`
* 本轮采集最终报告的预期路径：`artifacts/predictor-twoFIFO-source-rebound-executor-v1-20261028-cpu11/run01/report.json`

全部 12 次原生采集和独立复核完成后，先使用拟合 manifest 的 `run --fit`，再执行：

```bash
python -m tools.predictor_validation_20261027.twoFIFO_source_rebound_generated_v1 run \
  --design artifacts/predictor-twoFIFO-source-rebound-generated-v1-20261029-cpu01-prepared/design.json \
  --fit-report artifacts/predictor-twoFIFO-source-rebound-phase12-fit-v1-20261029-native01/report.json \
  --output artifacts/predictor-twoFIFO-source-rebound-generated-v1-20261029-native01
```

上面的 native 输出尚未生成；不得从预声明或 CPU 测试推断真实预测精度或分流收益。

CPU 独立复核：拟合协调器 21 项测试通过，报告为 `artifacts/predictor-twoFIFO-source-rebound-phase12-fit-v1-20261029-cpu01-independent/report.json`；joint-law / generated 入口的作者与独立测试合计 9 项通过，报告为 `artifacts/predictor-twoFIFO-source-rebound-generated-v1-20261029-cpu01-independent/report.json`。前者的数值响应使用 mock，后者完整执行了 all-UNKNOWN 的 150-case 生成与持久化流程；均未运行真实 fit 或独立验证。

真实拟合入口的无拟合检查已返回 `WAITING_FOR_EXACT12_CONTINUATION_FINAL`，收据位于 `artifacts/predictor-twoFIFO-source-rebound-phase12-fit-v1-20261029-cpu03-waiting/report.json`。它未创建拟合 claim，未进入数值拟合。采集继续使用 cpu11 / run01；前三个接受分片包含 150 个 measured 请求和 54 个 priming 请求，第四个分片正在运行。这个 150 是当前请求数，完整设计的 150 cases 是另一项分母。

原生忙态适配器另经 35 项 CPU 测试和独审：`artifacts/predictor-twoFIFO-native-busy-state-v1-20261029-cpu03-independent/report.json`，源码 SHA `1951f3d0be119cd3f7a70d54c44ab0db3f97e6719a6136375ed77e96de7d5191`。它修复了真实 PHASE/QUERY 不带 PID 的兼容问题，以及历史回放泄露未来 registry 存在性的因果问题。当前只让已识别 settled 边界进入 v5_2；active work 残量与 collector in-flight 水位不足时，忙态仍为 UNKNOWN。该模块未接入冻结采集或当前生成 runner，不能算作忙态滚动校正已经完成。

## 2026-10-29 后续状态更正

上文“第四个分片正在运行”和 `WAITING_FOR_EXACT12_CONTINUATION_FINAL` 为当时的历史状态。cpu11/run01 已终止为 `STOPPED_NO_RETRY`：前四次执行通过复核，第五次在 supervisor 的 18 GiB **磁盘空间**预检失败，尚未创建 collection 或启动采集子进程。原报告、第五次 claim 和前四次数据均保留；旧设计继续绑定这一 STOPPED 报告，不能用于声称完成 exact12 拟合。

用户随后恢复了磁盘空间；只读实测 `/root/autodl-tmp` 可用 33,424,334,848 bytes，GPU 无计算进程。正在独立审查新的、仅适用于已证明未启动分片的恢复入口 `twoFIFO_prelaunch_recovery_v1.py` 及新 intake。恢复拟使用单独 run02 报告记录原四次加后八次的来源，不改旧停止报告、不降低磁盘门槛，不重跑已经启动的分片。只有完整采集、复核和显式新来源链通过，才能进入新的拟合及 generated 设计。

资源与评分路径的 CPU 进展：`twoFIFO_source_rebound_resource_rescore_v1.py` 已通过独立完整入口测试，报告为 `artifacts/predictor-twoFIFO-source-rebound-resource-rescore-v1-20261029-cpu01-independent/report.json`。测试真实执行 150-case / 546-request / 1200-UNKNOWN-world 的恢复、评分、预测持久化及逐 case 实测标签路径；认证与 native provider 构造边界使用 fixture。它没有验证真实预测误差或分流收益。实测完整六项效用仍要求原评分计划的全部任务有观测，缺任务时保持 UNKNOWN，不能调整原权重或用条件均值补齐。
