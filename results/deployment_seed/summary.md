# MILP deployment seed validation

正式入口：`MilpDeploymentSeeder.solve(...)` / `python -m deployment_core.seed`。
一次稀疏 MILP 联合求整数实例计数 x 和 req/s routing r，仅返回一个 minimum-memory feasible seed。
约束复用 Stage 1：coverage、真实显存、workload conservation、capability/deployment linkage、standalone capacity。
目标仅为 Σ V_m x_m，mip_rel_gap=0；不加 secondary objective，不声明跨 solver 版本的同值解相同。
utilization=Σ r*s0、slack=rho_max*x−utilization、逐任务 demand=r*s0 均保留完整矩阵；不是实测 GPU utilization。
solver_time_ms 仅计 solver 调用，不包括数据读取/建模/审计；一次计时不代表普遍 scalability。

| Scenario | Feasible deployments (oracle) | Seed memory MiB | Oracle min MiB | Solver ms | Check |
| --- | ---: | ---: | ---: | ---: | --- |
| load_0.25 | 16 | 17574.0 | 17574.0 | 1.141 | PASS |
| load_0.50 | 9 | 19200.0 | 19200.0 | 7.740 | PASS |
| load_0.75 | 5 | 19804.0 | 19804.0 | 7.760 | PASS |
| load_1.00 | 3 | 20386.0 | 20386.0 | 8.237 | PASS |
| load_1.25 | 3 | 20386.0 | 20386.0 | 8.244 | PASS |
| load_1.50 | 2 | 20554.0 | 20554.0 | 0.942 | PASS |
| balanced | 4 | 19804.0 | 19804.0 | 0.935 | PASS |
| text-heavy | 5 | 19950.0 | 19950.0 | 7.560 | PASS |
| image-heavy | 4 | 19804.0 | 19804.0 | 0.946 | PASS |
| ocr-heavy | 4 | 19782.0 | 19782.0 | 0.919 | PASS |
| asr-heavy | 0 | None | None | 0.742 | PASS |
| vqa-heavy | 0 | None | None | 0.749 | PASS |
| lid-heavy | 8 | 19200.0 | 19200.0 | 0.876 | PASS |
| rho_0.50 | 2 | 20554.0 | 20554.0 | 0.925 | PASS |
| rho_0.60 | 3 | 20386.0 | 20386.0 | 8.351 | PASS |
| rho_0.70 | 3 | 20386.0 | 20386.0 | 8.292 | PASS |
| rho_0.80 | 3 | 20386.0 | 20386.0 | 8.401 | PASS |
| rho_0.90 | 5 | 19804.0 | 19804.0 | 8.168 | PASS |
| rho_0.95 | 5 | 19804.0 | 19804.0 | 7.208 | PASS |

19 scenarios: PASS。有解 17 个，无解 2 个。
所有 seeder 调用期间均将 enumeration/combinations/旧 Deployment object 构造替换为会抛错的 guard；baseline 仅在 guard 之外执行。

| Synthetic models | MILP variables | constraints | Solver ms | Enumeration | Check |
| ---: | ---: | ---: | ---: | --- | --- |
| 20 | 55 | 68 | 4.781 | not invoked | PASS |
| 30 | 85 | 98 | 5.376 | not invoked | PASS |

Synthetic 数据仅验证直接求解与调用路径，不是真实 serving 性能数据。

示例 seed：paddleocr, qwen, resnet50, whisper；显存 20386 MiB。
该 seed 仅用于可行初始化，不是最终性能最优 deployment。
正式接口允许零负载时全零计数；旧 exhaustive baseline 为历史兼容只枚举非空组合。19 个验证场景均为正负载。
数值校验使用原 Stage 1 守恒相对容差 1e-7、capacity 绝对容差 1e-9；约 1e-16 的负 slack 可由浮点误差产生，保留原计算值。
solver 只有明确 infeasible 状态返回 feasible=false；超时、数值错误、无效 witness 抛 SeedSolverError。
未修改 measurement、CUDA/PyTorch/model 环境；未运行性能 sweep、未实现性能优化或 guided search。

回归验证：42/42 tests 通过，含原 Stage 1/1.5 的 27 项测试和新增 15 项 seeder 测试。
保护文件 SHA256 核对：88 个 measurement/characterization 代码及封存结果文件均未改变。详见 verification.json 与 regression_tests.log。
