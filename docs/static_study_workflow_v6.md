# 静态六项研究：校准、联合搜索与独立 GPU 确认

2026-10-21 完成 CPU 准备；2026-10-22 已恢复真实 GPU 串行校准。
校准完成数以 `artifacts/static-study-gpu-20261022/calibration/queue-progress.json` 为准。
七模型全池与纯通、纯专结构控制已经用户确认。合成 CPU 验证、部分效用及旧饱和画像
均不能替代留出验证通过的联合评价器或独立 GPU 结果。

## 本轮产物

冻结输入在 `artifacts/static-study-v6-20261021/inputs/`，质量采用原生任务误差：
QA 为 1−EM、图像为 1−accuracy、OCR 为 CER、ASR 为 WER。
取消绝对质量阈值，不取消可执行能力、来源和输入/协议身份核验。
缺少质量或能耗不会填零；质量项非零时使用 `six_term_quality_v1`。

离线逐输入质量通过与 runtime 相同的种子、样本打乱与副本级加权轮询重放。
OCR/ASR 汇总错误数和参考长度，不平均每条错误率，不截断超过 1 的 CER/WER。
实际运行的 Q 必须重新评分实际 worker 输出，不能用聚合代理或历史逐输入输出替代。
历史配置回执不是重新读取全部权重所得哈希；派生 native binding 会保留这个区别。

## 准备、检查与执行

```bash
python scripts/prepare_static_experiments.py prepare \
  --inputs artifacts/static-study-v6-20261021/inputs \
  --output artifacts/static-study-v6-20261021/experiments-NEW
python scripts/prepare_static_experiments.py validate \
  --packet artifacts/static-study-v6-20261021/experiments-final-v2/experiment-packet.json
```

prepare/validate 不查询 GPU、不导入模型、不启动 worker。文件变更会使包失效。
job 身份绑定实际部署、分流、固定输入与轨迹、H、seed、预热规则、已部署 worker 和运行源码；
完整配置池与来源清单作为 provenance 单独保存，删去未部署候选不应改变同一条件的身份。

八个校准条件覆盖共享通用、多专用共驻、QA/ASR 不分流与 75% 给专用的配对、OCR/图像分流。
它们使用 H=600，正式条件 H=1800、确认种子 43/44/45 的身份另立，证据不能互换。
预热给每个部署 worker 的所有有负载可执行任务选择三个最大输入，再运行 30 秒预热窗口。
最大输入按媒体文件字节或 QA context+question UTF-8 字节衡量，这是明确的输入大小代理。

以下是单条件原生执行入口；当前串行校准使用已冻结的执行源快照：

```bash
python scripts/prepare_static_experiments.py run \
  --packet PATH/experiment-packet.json --job JOB_ID --output NEW_RUN_DIR --execute
```

执行前绑定当前权重、历史质量报告与冻结配置，PP-OCR 只绑定保留的 det/rec 目录，
不读取整个候选筛选目录。当前权重哈希仅证明当前执行文件；有历史 SHA 时核对它，
无历史 SHA 时不追认旧输出的全权重等价性。运行保存实际 GPU UUID、权重和执行身份。
新增未测条件通过显式 standalone transfer proxy 采集配置运行并监测资源，
该采集配置不转移搜索最优性证书。

## 导入实测并冻结评价器

```bash
python scripts/prepare_static_experiments.py freeze-evaluator \
  --packet PATH/experiment-packet.json --runs RUN_A/run.json RUN_B/run.json \
  --output NEW_FROZEN_EVALUATOR
```

只接受本包的完整原生校准运行，重新核验实际输出质量、能耗、窗口、预热、分流、GPU 和权重身份。
独立确认运行默认拒绝用于拟合；若要转为开发证据，应显式建立新的开发研究，原确认结论作废。
同条件重复运行汇总服务样本、平均电量，并取最大的实测安全显存。

首版冻结器只导入精确 `部署/分流/负载/H/seed` 的服务均值、整卡电量和显存。
服务均值用于常数服务 FIFO 预测，不能声称预测 P95 是实测 P95。
没有实现“由八点外推全部混合组合”的干扰模型；未测条件继续返回 unknown。
H600 校准不会自动提供 H1800 或新 seed 的精确证据。

冻结目录包含新 `system-config.json` 和 `evaluator-freeze.json`，后者记录版本哈希和导入来源。
后续必须从新配置建立新的 DeploymentSystem，丢弃旧缓存、反馈约束与最优性证书。

```bash
python scripts/prepare_static_experiments.py prepare \
  --inputs artifacts/static-study-v6-20261021/inputs \
  --config NEW_FROZEN_EVALUATOR/system-config.json --output NEW_SEARCH_PACKET
```

只有完整六项得分存在时才生成胜出与强竞争方案；证据不全时保留 gap 和等待状态。
竞争方案重新求解不同部署、相同部署的替代分流、全域不分流，重复方案合并，另留一个自适应名额。
运行预算不等于真实全域最优证明。

## 未测条件的补点队列

将待检候选写成 JSON 数组，每项含 `deployment` 与 `routing`：

```bash
python scripts/prepare_static_experiments.py queue \
  --packet PATH/experiment-packet.json --config NEW_FROZEN_EVALUATOR/system-config.json \
  --plans CANDIDATE_PLANS.json --output NEW_QUEUE_PACKET.json
```

输出是可 validate/run 的派生完整包，包含去重后的测量 job、完整评分/未知状态和来源绑定。
自适应补点属于开发校准；对其重新冻结后，应重新求解并另外准备独立确认数据。
未知候选不能按已测不可行剔除，有限域证书必须覆盖它们的有效上界。

## 少参数混合预测器 v1

实际 GPU 校准使用独立冻结的旧执行源，CPU 拟合与搜索使用当前代码。
固定六个训练条件、两个留出条件和误差阈值见
`artifacts/static-study-gpu-20261022/calibration-protocol-v1.json`。
留出 QA75/ASR75 不参与拟合、特征选择或阈值调整；独立确认数据不能用于拟合。

```bash
python scripts/fit_static_surrogate.py \
  --protocol artifacts/static-study-gpu-20261022/calibration-protocol-v1.json \
  --execution-packet artifacts/static-study-gpu-20261022/execution-packet-v2.json \
  --runs-root artifacts/static-study-gpu-20261022 \
  --output artifacts/static-study-gpu-20261022/calibrated-surrogate-v1
```

缺任何已声明条件时，仅输出 WAITING，不创建假校准配置。
脚本核验运行、原包及执行源码快照来源；输出不可覆盖的新配置、报告与来源哈希。
模型使用一个全局服务倍率、最多三个非负整卡功率回归系数、启动电量/worker 代理，
显存预测保持不小于独立驻留和。已完成独立画像的其他路径/副本可以明确标为外推；
缺失独立路径画像时仍为 unknown，不把外推称为已实测。

报告包含留出服务、整卡能耗、质量、mean/P95、六项效用及排序误差。
留出失败原样保留 `VALIDATION_FAILED`，不能调整 v1 阈值后复用留出结果宣称通过。
在该冻结预测器下的有限域搜索可以给数学最优证书，真实 GPU 全域最优仍未证明。

## 六角色搜索和独立确认

```bash
python scripts/solve_static_study.py --config FIT/system-config.json \
  --output NEW_SEARCH --time-limit 120 --route-grid 4
python scripts/solve_static_study.py --config FIT/system-config.json \
  --output NEW_SMALL_COMPARISON.json --compare-small --time-limit 10
python scripts/confirm_static_study.py prepare \
  --source-packet EXECUTION_PACKET.json --config NEW_SEARCH/system-config.json \
  --search NEW_SEARCH/search-results.json \
  --checkpoint-bindings artifacts/static-study-v6-20261021/cpu-checkpoint-bindings.json \
  --output NEW_CONFIRMATION_PACKET.json
python scripts/run_static_confirmation_queue.py --packet NEW_CONFIRMATION_PACKET.json \
  --output NEW_CONFIRMATION_RUNS --execute
python scripts/report_static_confirmation.py --packet NEW_CONFIRMATION_PACKET.json \
  --runs-root NEW_CONFIRMATION_RUNS --output NEW_CONFIRMATION_REPORT
```

正式运行从同一代码快照准备和执行，准备根目录变化会使完整运行身份核验失败。
搜索和确认准备均为 CPU-only，未验证或失败的代理无法进入正式比较。
六角色为自由搜索、最强不同部署、相同部署下最强替代路由、不分流、纯通、纯专；
物理部署和路由相同的角色按身份合并，每个确认种子只执行一次。
预算停止或未知域保留原状态、上下界和 gap，不能将当前 incumbent 写作已证明最优。

确认种子固定为 43/44/45，各自组成一个配对区组。输出和恢复绑定种子、完整作业 ID、
确认包、用途、GPU、当前权重、原始输出质量和整卡能耗。确认数据禁止进入拟合。
配对差值以种子而非请求作为独立单位；三个种子不能支持精确双侧 p<0.05 的结论。
搜索证书只属于冻结代理的有限域，实测报告只比较预先冻结并实际运行的候选。

## 本轮实际完成记录

2026-10-22 本轮已执行完整流程：v1留出失败保留，v2两个新留出通过，冻结七模型K8搜索证书gap=0，
五模型同域五轮MILP/两种DFS全部同优并有证书，正式五物理方案×三个种子15/15原生作业全部严格验证。
最终[综合报告](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/study-summary-final-v1/report.zh.md)、
[方法与实测解读](/root/autodl-tmp/generalist-specialist-project/docs/static_study_methods_and_results_20261022.md)、
[最终独立复审](/root/autodl-tmp/generalist-specialist-project/docs/static_confirmation_final_readonly_reaudit_20261022.md)
均已落盘。算术独立重算390/390项通过，配对区间/精确p/Holm独立审计通过。
三个种子均支持选中方案在已测候选中的最高U，但不构成统计显著性或物理全局最优证明。
纯通为预算候选，较强K4纯通候选另行披露且没有替换冻结正式名单。
