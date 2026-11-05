# 相关论文算法对照完成报告

已新增 **CherryPick（NSDI 2017）** 与 **Proteus（ASPLOS 2024）** 的明确适配实现，完成一次固定 24 次 CPU 比较；经典 DE/DA 与设施增删交换继续作为补充，未被删除或改名。

- [完整结果、局限与科研图表](../artifacts/paper-baseline-figures-20261105/README.md)
- [运行前冻结的论文机制映射](paper_baseline_mapping_20261105.md)
- [三页矢量图集](../artifacts/paper-baseline-figures-20261105/paper_baseline_comparison.pdf)
- [逐次结果 CSV](../artifacts/paper-baseline-figures-20261105/results.csv)
- [封存实验协议及源码哈希](../artifacts/paper-baseline-comparison-20261105/plan.json)
- [原始评价和审计汇总](../artifacts/paper-baseline-comparison-20261105/summary.json)

**7 模型池 BO–MADS 三次均胜过 CherryPick-U；14 模型池 CherryPick-U 有两次胜出，效用中位数更高。** 14 模型增益中位数分别为 0.008439 与 0.011213，同批耗时中位数为 0.829 秒与 1.324 秒。CherryPick 找到了 BO–MADS 未在预算内充分评价的 g+x09 组合及连续分流，因此不能宣称 BO–MADS 全面领先。

Proteus-Q 保留原质量目标，只作不同目标的背景对照；Proteus-U-linear 是本项目把线性总效用代理放进联合 MILP 的适配，不是原论文算法。两者未带来初始解之外的总效用收益。U-linear 的拟合误差明显过大，明确降为方法适配诊断，不作为强主基线或优越性证据。

18 项机制与预算测试通过，1050 次付费预测、440 个内层结果审计无错误。BO–MADS 六次轨迹与先前封存结果相同。所有原生停止条件保留，未强制凑满预测预算；没有新增 GPU 工作、真实预测器重训或生产算法修改。实验仍为既有合成目录上的优化机制比较，不是两个论文系统的完整复现。
