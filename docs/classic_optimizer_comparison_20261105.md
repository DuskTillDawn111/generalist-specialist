# 经典优化算法对照：实际 SciPy 实现

固定 24 次 CPU 搜索已完成。主要对照直接使用 SciPy 1.15.3 的 `differential_evolution` 和 `dual_annealing`，另外保留相同编码的随机搜索校准。四种方法包含修复后的 BO–MADS，同一批次运行；每次上限 64 次完整预测，初始化计费。

|方法|7 模型效用增益中位数|14 模型效用增益中位数|7 / 14 模型耗时中位数（秒）|
|---|---:|---:|---:|
|BO–MADS|0.008439|0.008439|0.725 / 0.820|
|SciPy DE|0.002676|0.006796|0.145 / 0.219|
|SciPy Dual Annealing|0.004714|0.003078|0.150 / 0.222|
|Random|0.004460|0.000000|0.145 / 0.217|

BO–MADS 本批效用中位数最高，但耗时最高。7 模型退火 seed47、14 模型 DE seed47 都优于对应 BO–MADS；不声称每次或普遍获胜。之前增删交换在 14 模型池三次胜出的证据继续有效，新基线没有替代或删除这个强对照。

通用算法采用同一个五维混合配置编码，测试覆盖全部 33/117 个合法部署。编码、能力修复、目标和预算属于项目适配，实际搜索算子由 SciPy 执行。DE 在小预算下使用 10 个个体，DA 保留默认局部搜索；没有参数搜索。该实验是既有合成凸排队成本场景，不是实际服务器或真实 FIFO/GPU 验证，也不是 CherryPick/InferLine/MuxServe 复现。

11 项语义/预算测试通过；1,536 次付费预测、440 个内层结果核对通过，审计错误为 0。未新增 GPU、真实预测器调用或重训，无待运行自动实验。

- [完整方法来源、参数和适配边界](../artifacts/classic-optimizer-figures-20261105/README.md)
- [三页科研图集 PDF](../artifacts/classic-optimizer-figures-20261105/classic_optimizer_comparison.pdf)
- [逐种子数据 CSV](../artifacts/classic-optimizer-figures-20261105/results.csv)
- [固定实验协议](../configs/classic_optimizer_comparison_20261105.json)
- [原始结果与来源清单](../artifacts/classic-optimizer-comparison-20261105/plan.json)
- [验证收据](../artifacts/classic-optimizer-comparison-20261105/delivery-verification.json)

![效用与耗时](../artifacts/classic-optimizer-figures-20261105/fig01_quality_cost.png)

![收敛曲线](../artifacts/classic-optimizer-figures-20261105/fig02_anytime.png)
