# Module 15 preparation

状态：READY_FOR_CAPACITY_CONSTRAINED_CPU_SEARCH。项目没有预定义E2E latency SLO；已在搜索前报告。用户明确修改为容量约束测试，不设置人为延迟阈值。原quality/capability、rho≤0.8、14F实测safe memory均保留。E2E mean/P95只作为GPU指标。

完整现有E/H identity并集参加部署竞争；原V3对照不修改。主目标是最小实测驻留safe memory，同等最小资源成本状态中使用冻结J-routing。已核验原质量admission和真实Q/s0/g(x)/M(x)，未选择三阶段QPS、未运行GPU。
