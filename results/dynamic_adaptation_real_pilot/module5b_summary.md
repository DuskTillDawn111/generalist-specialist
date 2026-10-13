# Module 5B：Real Dynamic Transition Pilot

当前证据：**PASS**；decision audit=PASS。

冻结 W1→W2→W3→W1，arrival rates 未改。Predictor=None、canonical routing、Module 3 的三态目标和 5% 接受门槛保持原样。D0/D1 不驱动主序列。

| Transition | Actual warm 起点 → 终点 | Action | D0 / cold / warm mean ms | Cold / warm P95 ms | Cold / warm evals | Gap | Saving eval |
|---|---|---|---|---|---|---|---|
| transition_w1_w2 | paddleocr+qwen+resnet50+whisper → distilbert+paddleocr+qwen+resnet50+whisper | SWITCHED | 58.752 / 55.911 / 43.181 | 93.855 / 73.418 | 6 / 5 | -22.77% | 16.67% |
| transition_w2_w3 | distilbert+paddleocr+qwen+resnet50+whisper → distilbert+paddleocr+qwen+resnet50+whisper | NO_SWITCH | 39.780 / 51.801 / 39.780 | 74.897 / 70.920 | 6 / 3 | -23.21% | 50.00% |
| transition_w3_w1 | distilbert+paddleocr+qwen+resnet50+whisper → distilbert+paddleocr+qwen+resnet50+whisper | NO_SWITCH | 45.168 / 59.988 / 45.168 | 96.460 / 79.759 | 6 / 3 | -24.71% | 50.00% |

## Confirmation 与状态

- transition_w1_w2：cold/warm independent confirmation mean=53.544/42.115 ms；gap=-21.35%；mean direction maintained=True；warm close=True。
- transition_w2_w3：cold/warm independent confirmation mean=52.143/39.635 ms；gap=-23.99%；mean direction maintained=True；warm close=True。
- transition_w3_w1：cold/warm independent confirmation mean=54.492/42.503 ms；gap=-22.00%；mean direction maintained=True；warm close=True。

## Reconfiguration 与搜索成本

| Transition / strategy | Accepted switches | Load / unload | VRAM moved MiB | Actual measured switches ms | Search physical attempts | LP solves | Iterations | First steady / final accepted eval |
|---|---:|---|---:|---|---:|---:|---:|---|
| transition_w1_w2 / D1 | 4 | 3 / 3 | 5792.0 | 1543.998, 3645.115, 2428.080, 3330.609 | 18 | 10 | 3 | 1 / 6 |
| transition_w1_w2 / D2 | 1 | 1 / 0 | 750.0 | 3343.868 | 6 | 8 | 2 | 1 / 3 |
| transition_w2_w3 / D1 | 4 | 3 / 4 | 6374.0 | 2208.351, 3654.540, 2435.959, 2259.494 | 18 | 10 | 3 | 1 / 6 |
| transition_w2_w3 / D2 | 0 | 0 / 0 | 0.0 | null; no switch | 3 | 5 | 1 | 1 / 1 |
| transition_w3_w1 / D1 | 4 | 3 / 4 | 6374.0 | 2558.512, 3826.838, 2595.137, 2356.578 | 15 | 10 | 3 | 1 / 6 |
| transition_w3_w1 / D2 | 0 | 0 / 0 | 0.0 | null; no switch | 3 | 5 | 1 | 1 / 1 |

累计 cold/warm state evaluations=18/11；节省 7 次（38.89%）。
累计 cold/warm models loaded=9/1；unloaded=11/0；accepted switches=12/1；Saving_switch=91.67%。
实际新执行 search physical attempts=75；confirmation physical attempts=12。Exact cache hits 与实际新 GPU state measurements 分开记录；evaluation 节省是算法访问 state 的计数，不冒充独立 GPU run 节省。

## 14 项核对

1. transition_w1_w2 actual warm：paddleocr+qwen+resnet50+whisper → distilbert+paddleocr+qwen+resnet50+whisper。
2. transition_w2_w3 actual warm：distilbert+paddleocr+qwen+resnet50+whisper → distilbert+paddleocr+qwen+resnet50+whisper。
3. W3→W1 action=NO_SWITCH；未强制 mutation。
4. 每个 transition 的 cold/warm evaluation 数见首表；均 <=6，含起点，3 completed repetitions/state。
5. Warm/cold final latency 的绝对差与 Gap 见 cold_vs_warm_metrics.json；mean/P95/stability 及 individual repetitions 全部保留。
6. Confirmation 见上表；2 independent seeds，与搜索隔离，无任何 post-confirmation decision 更新。
7. 装载/卸载累计与逐 transition 数见上表；D1 从同一 actual old 显式执行 seed reset，另列 local search switches 和 cold seed reset。
8. 真实 READY-based switch time 成功测量 13 次；所有无 switch 阶段时间=null。没有将 benchmark repetitions 计作 switch latency。
9. Warm MILP fallback：[False, False, False]。
10. Genuine unstable search repetitions=0；confirmation unstable=0。Completed instability 保留，不重标为外部中断。
11. External interruptions=0；全部 invalid attempts=0。无证据的缺失 raw 为 uncertain，不猜测原因。
12. Enumeration calls=0（runtime tripwire + audit）。
13. 累计节省 7 runtime evaluations（38.89%）。
14. 当前 pilot=PASS；close+cheaper transitions=3/3；confirmation close=3/3。

## 解释边界

Accepted-path worker activation 在 search 决策冻结后真实执行，保留未变化 workers。为避免 GPU 干扰，candidate profiling、D1 counterfactual 和 D2 activation 的进程 residency 分开；这是显式 phase 的动态 deployment pilot，没有持续到达请求下的零停机证明。Setup residency 和 teardown 另列且不计 accepted switch；全流程实际物理 load/unload 不能由 structural accepted-path totals 代表。
VRAM moved 来自冻结 catalog 的模型峰值 MiB，仅表示结构重配置量。Switch time 来自同一进程 monotonic clock，包含 accepted removals 和 loads 到所有 workers READY；不含 warmup、candidate profiling 或 confirmation。
冷搜索最终部署也给出从 actual old 到 final 的 endpoint structural sensitivity（见 README），避免只依赖 seed reset 的成本优势。
Close 门槛预先固定为 warm 同等或更好 stability 且 mean <= cold×1.05。所有统计仅是 pilot 描述，不声称统计显著性。Confirmation 数字不覆盖 search objective；排序接近时 direction 可以翻转。
已停止于 Module 5B；未进入 forecasting、change detection、prewarming、multi-replica、routing reoptimization 或 final paper benchmark。

## Warm mutation observations 与接受时点审计

- transition_w1_w2：actual starting deployment feasible=True；first accepted mutation={'operation': 'ADD', 'model': 'distilbert'}；accepted REMOVE=[]；cold/warm evaluations to final accepted state=6/3。
  Evaluated mutations: ADD distilbert: ACCEPTED (mean_improved); REMOVE paddleocr: ROLLBACK (mean_worse); REMOVE paddleocr: ROLLBACK (mean_worse); REMOVE resnet50: ROLLBACK (mean_worse)。
- transition_w2_w3：actual starting deployment feasible=True；first accepted mutation=None；accepted REMOVE=[]；cold/warm evaluations to final accepted state=6/1。
  Evaluated mutations: REMOVE distilbert: ROLLBACK (mean_worse); REMOVE paddleocr: ROLLBACK (mean_worse)。
- transition_w3_w1：actual starting deployment feasible=True；first accepted mutation=None；accepted REMOVE=[]；cold/warm evaluations to final accepted state=6/1。
  Evaluated mutations: REMOVE distilbert: ROLLBACK (mean_worse); REMOVE paddleocr: ROLLBACK (mean_worse)。

Final accepted-state evaluation index 按整批 Top-B 测完后的实际 accept 时点计算；首次 profile final 的 index 单独保存，二者不混用。上表中的 first steady 不变。
该 index 指 search accept decision；accepted-path physical activation 在 search 冻结后执行。其前已完成 evaluation 数另存 evaluations_before_physical_final_activation，不冒充在线 time-to-ready。
Derived-metric audit=PASS；confirmation 后 final deployment、objective、trace 和 switches 与 frozen decision 逐项相同。

Endpoint sensitivity（仅 structural，未执行该 alternative path）：
- D1: switches=3, load=1, unload=3, VRAM moved=2832.0 MiB；measured time=null。
- D2: switches=1, load=1, unload=0, VRAM moved=750.0 MiB；measured time=null。

## D0 / D1 / D2 final performance

| Transition | Strategy | Weighted mean ms | Weighted task P95 ms | Stability |
|---|---|---:|---:|---|
| transition_w1_w2 | D0 | 58.752 | 94.823 | steady |
| transition_w1_w2 | D1 | 55.911 | 93.855 | steady |
| transition_w1_w2 | D2 | 43.181 | 73.418 | steady |
| transition_w2_w3 | D0 | 39.780 | 70.920 | steady |
| transition_w2_w3 | D1 | 51.801 | 74.897 | steady |
| transition_w2_w3 | D2 | 39.780 | 70.920 | steady |
| transition_w3_w1 | D0 | 45.168 | 79.759 | steady |
| transition_w3_w1 | D1 | 59.988 | 96.460 | steady |
| transition_w3_w1 | D2 | 45.168 | 79.759 | steady |

本轮观察的是从已有良好部署继续搜索的价值。D2 在 T1 增加 DistilBERT 后拥有全部五模型；后续若 NO_SWITCH 且 D0 与 D2 相同，说明保留该部署已经足够，不证明每个 workload phase 都需要不同 specialist 组合。
性能 gap 与成本统计比较的是冻结小预算 cold search，不能解释为对全局最优的 gap，也不能将 accepted-path reset 成本解释成所有 cold 系统唯一的执行方案。
