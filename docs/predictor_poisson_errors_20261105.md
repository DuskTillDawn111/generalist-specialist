# 冻结预测器：泊松到达下的预测误差

本次按请求补做两次固定 GPU 验证，合计 **199 个正式请求**；使用冻结预测器，不训练、不调参、不按结果更换种子。五模型全部驻留、各一副本，每任务按 50% G2 / 50% 专用模型进行原 SWRR 路由。

[五页科研误差图集](../artifacts/predictor-poisson-error-figures-20261105/poisson_prediction_errors.pdf) · [逐请求 CSV](../artifacts/predictor-poisson-error-figures-20261105/request-errors.csv) · [聚合时延 CSV](../artifacts/predictor-poisson-error-figures-20261105/aggregate-errors.csv) · [能耗/吞吐 CSV](../artifacts/predictor-poisson-error-figures-20261105/episode-errors.csv)

## 到达方式与训练分开

训练中的同时提交、错开提交、短突发及持续到达用于覆盖执行状态；不能用训练回放误差代替部署条件下的验证。本图只使用两次新的泊松到达实测。两轮计划和全部预测在首次 GPU 采集前一起封存。

泊松到达复用早期静态部署、分流和正式 GPU 对照所用的 `gsdeploy.trace.make_trace(..., "poisson")`：每任务独立指数间隔，不固定实际请求数，不事后调整间隔，不筛选种子。

到达族一致；本轮负载与窗口保持当前五模型范围：每任务 0.036977697 rps、每轮 649.040 秒、每任务期望 24 个请求。早期对照示例为每任务 0.074836084 rps、1800 秒，两者不是相同强度/时长的逐项性能对比。

此前五模型三候选确认的两种到达实际为 `stratified_jitter` 与 `paired_bursts`，其数据未混入这里，也没有改标为泊松。

## 主图：逐请求平均、最小与最大误差

`APE_i = 100 × |预测端到端时延_i − 实测端到端时延_i| / 实测端到端时延_i`。端到端时延从计划到达时刻算至完成，包含排队。

圆点为逐请求 APE 的算术平均，横线两端为最小值和最大值；每个请求权重相同，所有请求均纳入。图右侧直接写出 Mean / Min / Max / n。这不是带符号误差均值，不是置信区间，也不是平均时延的误差。

|任务|请求数|平均误差 %|最小误差 %|最大误差 %|
|---|---:|---:|---:|---:|
|ASR|58|7.5497|0.3299|74.9352|
|Image|50|11.5698|0.0691|44.5416|
|OCR|48|9.1759|1.0594|28.7118|
|Text|43|11.0086|0.0546|91.2925|
|All requests|199|9.6994|0.0546|91.2925|

![逐请求误差](../artifacts/predictor-poisson-error-figures-20261105/fig01_request_errors.png)

样本数是请求数，不是独立输入数或独立运行数。使用已有画像输入池，两轮只有两个到达历史，不据此给出总体显著性或未见输入泛化结论。

## 聚合误差与逐请求误差分别报告

图 3、4 先在每个任务/每轮分别计算预测和实测的平均时延、P95 时延，再计算相对误差。图中均值/最小/最大是在两轮聚合误差上统计，每任务 n=2。P95 时延的误差不是“逐请求误差的 P95”。

|任务|平均时延误差：均值 / 最小 / 最大 %|P95 时延误差：均值 / 最小 / 最大 %|
|---|---|---|
|ASR|7.0706 / 6.9422 / 7.1991|8.8262 / 5.7289 / 11.9234|
|Image|8.3401 / 4.7518 / 11.9283|9.6664 / 8.3205 / 11.0122|
|OCR|3.2304 / 2.0564 / 4.4044|5.7640 / 3.2926 / 8.2354|
|Text|8.3235 / 6.2693 / 10.3778|11.3137 / 6.5155 / 16.1118|

整段能耗与固定窗口吞吐同样报告两轮误差的均值、最小与最大。能耗是 GPU board 从启动至 observation_end，包含 priming、正式窗口、排空和尾部；不是整机能耗。吞吐是窗口内完成数除以同一窗口长度，不是最大稳定容量。

|指标|样本数|平均误差 %|最小误差 %|最大误差 %|
|---|---:|---:|---:|---:|
|GPU energy|2|2.6909|2.6381|2.7438|
|Throughput|2|0.0000|0.0000|0.0000|

## 五张独立图

|图|内容|文件|
|---|---|---|
|1|逐请求误差，按任务与总体|[PDF](../artifacts/predictor-poisson-error-figures-20261105/fig01_request_errors.pdf) · [SVG](../artifacts/predictor-poisson-error-figures-20261105/fig01_request_errors.svg) · [PNG](../artifacts/predictor-poisson-error-figures-20261105/fig01_request_errors.png)|
|2|逐请求误差，按 G2/专用执行路径|[PDF](../artifacts/predictor-poisson-error-figures-20261105/fig02_path_errors.pdf) · [SVG](../artifacts/predictor-poisson-error-figures-20261105/fig02_path_errors.svg) · [PNG](../artifacts/predictor-poisson-error-figures-20261105/fig02_path_errors.png)|
|3|任务平均时延误差|[PDF](../artifacts/predictor-poisson-error-figures-20261105/fig03_mean_latency_errors.pdf) · [SVG](../artifacts/predictor-poisson-error-figures-20261105/fig03_mean_latency_errors.svg) · [PNG](../artifacts/predictor-poisson-error-figures-20261105/fig03_mean_latency_errors.png)|
|4|任务 P95 时延误差|[PDF](../artifacts/predictor-poisson-error-figures-20261105/fig04_p95_latency_errors.pdf) · [SVG](../artifacts/predictor-poisson-error-figures-20261105/fig04_p95_latency_errors.svg) · [PNG](../artifacts/predictor-poisson-error-figures-20261105/fig04_p95_latency_errors.png)|
|5|整段 GPU 能耗与固定窗口吞吐误差|[PDF](../artifacts/predictor-poisson-error-figures-20261105/fig05_energy_throughput_errors.pdf) · [SVG](../artifacts/predictor-poisson-error-figures-20261105/fig05_energy_throughput_errors.svg) · [PNG](../artifacts/predictor-poisson-error-figures-20261105/fig05_energy_throughput_errors.png)|

所有图保留极端值，使用线性误差轴，不删点、不裁剪，不用候选总效用或 SLO 达标率替代预测误差。采用论文常见的简洁分面/点区间呈现；这里的区间严格是样本最小—最大值。PDF/SVG 为矢量，PNG 为 600 dpi。

## 验证边界与复现

正式请求数按两轮为 98 / 101。请求身份、路由、输入及预测前置封存检查通过。运行/输出审计问题记录为 `[]`；详细审计见 capture-audit.json。误差计算的三项测试通过，覆盖绝对误差不抵消、极值保留和非法分母拒绝。

混合到达训练是设计选择，本轮没有与“只用泊松训练”做对照，因此不能由此单独证明混合训练提升了鲁棒性。结果也不授予全部七模型、多副本、其他负载或未见输入的预测资格。

源码见 source/，封存协议与哈希见 protocol.json、sources.json、manifest.json。CSV 保留未四舍五入的请求级误差和正负方向，便于重画。两次固定实验已结束，不因某项误差大而自动重训或补采。
