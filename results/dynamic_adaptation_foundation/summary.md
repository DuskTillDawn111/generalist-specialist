# Module 5A：Dynamic Workload Adaptation Foundation

已实现 warm-start / ADD repair / fallback / NO_SWITCH / structural switching accounting。仅 synthetic 和 offline planning，GPU measurement calls=0；到此停止。

Cold 默认行为保持：initial_deployment=None → 原 MILP seed 和搜索；warm feasible 起点直接计算新 workload 的 canonical LP，MILP solves=0。
Repair 只生成与 missing capability / overloaded tasks 相交的 one-hop ADD。先恢复 feasibility，再减少 unserved rate / relaxed max pressure；恢复前不 REMOVE、不评价 infeasible runtime。
Repair 消耗 warm iterations；预算耗尽或没有可行修复才 fallback，记录 warm_start_fallback=true。fallback 后不增加 iteration/runtime budget。
当前 deployment 是正式 baseline；同一 phase visited states/rejected candidates 不重测，跨明确的新 phase 可重新访问旧 deployment。没有 forecasting 或自动 workload detection。
ReconfigurationCostModel 默认 StructuralReconfigurationCost；只累计 committed path 的 loads/unloads 和 VRAM movement，measured_switch_time_ms=null；没有从大小估算时间。

## Synthetic cold vs warm

A–D synthetic latency = arrival-rate weighted best-supported task service ms + per-model residence penalty；E 复用独立 diminishing-return/residence fixture。均仅人工 response surface，不能当作真实 GPU 效果。
两者共享 surface、objective、Top-2、最多 3 轮；warm budget=4、cold reference=6，并另报 cold budget=4 的 matched comparison。Cold 重配置成本包含 current → MILP seed。

| Transition | Warm action | Warm / cold / matched mean ms | Warm / cold / matched evals | Warm / cold LP | Warm loads/unloads | Cold loads/unloads | Warm / cold oracle gap ms |
|---|---|---|---|---|---|---|---|
| A_small_drift | NO_SWITCH | 267.170 / 267.170 / 267.170 | 3 / 4 / 4 | 3 / 4 | 0/0 | 1/1 | 0.000 / 0.000 |
| B_one_specialist_change | SWITCHED | 265.045 / 265.045 / 265.045 | 4 / 4 / 4 | 4 / 4 | 1/1 | 1/1 | 0.000 / 0.000 |
| C_missing_coverage | SWITCHED | 220.000 / 220.000 / 220.000 | 2 / 3 / 3 | 6 / 5 | 1/0 | 3/2 | 0.000 / 0.000 |
| C_capacity_violation | SWITCHED | 219.502 / 219.502 / 219.502 | 1 / 1 / 1 | 3 / 1 | 1/0 | 1/0 | 0.000 / 0.000 |
| D_phase_B | SWITCHED | 265.045 / 265.045 / 265.045 | 4 / 4 / 4 | 4 / 4 | 1/1 | 1/1 | 0.000 / 0.000 |
| D_phase_A_return | SWITCHED | 265.045 / 265.045 / 265.045 | 4 / 4 / 4 | 4 / 4 | 1/1 | 1/1 | 0.000 / 0.000 |
| E_10_model_pool | NO_SWITCH | 21.277 / 21.277 / 21.277 | 3 / 6 / 4 | 11 / 26 | 0/0 | 2/2 | unavailable / unavailable |
| E_20_model_pool | NO_SWITCH | 20.619 / 20.619 / 20.619 | 3 / 6 / 4 | 21 / 56 | 0/0 | 2/2 | unavailable / unavailable |
| E_30_model_pool | NO_SWITCH | 20.408 / 20.408 / 20.408 | 3 / 6 / 4 | 31 / 86 | 0/0 | 2/2 | unavailable / unavailable |

Case A：small drift 保持 NO_SWITCH。Case B：少量 one-hop ADD/REMOVE 达到 generalist+specialist-B。Case C 同时验证 coverage 和 pressure>rho_max 的 capacity repair。
Case D：A→B→A 可跨 phase 返回原 deployment；同一 phase follow-up 不产生来回切换；switch accounting 与所有 accepted transitions 一致。
Small-space oracle 只在 2/3-model synthetic catalog 的独立验证阶段调用；正式 cold/warm search 被 enumeration tripwire 包围。
完整 objective、LP、eval、mutation、VRAM movement 和 oracle gaps 见 cold_vs_warm_comparison.json。Warm 不保证获胜，尤其可能比 cold 需要更多 local mutations。

例如 C_capacity_violation：两者各 1 次 evaluation，warm LP=3、cold LP=1；B_specialist_change 两者 evaluation 都为 4。C_missing_coverage 则 warm eval=2、cold eval=3，warm loads/unloads=1/0、cold=3/2。不宣称 warm 在所有维度占优。

## Scalability

| Pool | Warm candidates per round | Warm evals / budget | Warm LP / linear bound | Warm action |
|---:|---|---|---|---|
| 10 | [10] | 3 / 4 | 11 / 31 | NO_SWITCH |
| 20 | [20] | 3 / 4 | 21 / 61 | NO_SWITCH |
| 30 | [30] | 3 / 4 | 31 / 91 | NO_SWITCH |

## Frozen real-workload planning

真实 workload 保持 Module 3C benchmark_workloads.json 速率；不复用历史 latency 作决策，不调用 runtime evaluator，不将 historical final 当作 oracle。
以下第二轮只是假设 first LP-selected ADD 被未来 runtime 接受后的一个局部 planning pass，不是实际接受，也不递归扩展完整状态图。

### W1_balanced -> W2_generalist_heavy

Current feasible=True；bottleneck=['qwen']；z=0.082274975。
Pressure tasks=['text', 'visual_qa']；capacity slack={'distilbert': 0.0, 'paddleocr': 0.7970269127990001, 'qwen': 0.7177250254478333, 'resnet50': 0.7973060673234444, 'whisper': 0.7534497464635}。
首轮 Top-B：ADD distilbert (delta_z=0.038235339); REMOVE paddleocr (delta_z=-0.046377617)。
+distilbert 自然产生=True，rank=1，Top-B=True。
-paddleocr 首轮已可生成=True；假设接受首个 ADD 后仍自然生成=True，后续 rank=1，Top-B=True。REMOVE 仍需 feasibility 筛选与未来 runtime 接受。
仅对 historical reference 计结构距离：warm 至少 2 mutations；cold seed=('qwen',) 至少 3 mutations。这是路径长度参考，不是算法会走向该部署的保证。
KEEP CURRENT / NO_SWITCH 是合法候选动作；planning 本身不能确认 latency improvement，committed switches=0。

### W2_generalist_heavy -> W3_specialist_friendly

Current feasible=True；bottleneck=['qwen']；z=0.060980471。
Pressure tasks=['ocr', 'visual_qa']；capacity slack={'distilbert': 0.7969845526222222, 'paddleocr': 0.0, 'qwen': 0.7390195289226666, 'resnet50': 0.7978008712844444, 'whisper': 0.7706799030946667}。
首轮 Top-B：ADD paddleocr (delta_z=0.033056676); REMOVE distilbert (delta_z=-0.035710971)。
+paddleocr 自然产生=True，rank=1，Top-B=True。
-distilbert 首轮已可生成=True；假设接受首个 ADD 后仍自然生成=True，后续 rank=1，Top-B=True。REMOVE 仍需 feasibility 筛选与未来 runtime 接受。
仅对 historical reference 计结构距离：warm 至少 2 mutations；cold seed=('qwen',) 至少 3 mutations。这是路径长度参考，不是算法会走向该部署的保证。
KEEP CURRENT / NO_SWITCH 是合法候选动作；planning 本身不能确认 latency improvement，committed switches=0。

### W3_specialist_friendly -> W1_balanced

Current feasible=True；bottleneck=['qwen']；z=0.064612461。
Pressure tasks=['text', 'visual_qa']；capacity slack={'distilbert': 0.0, 'paddleocr': 0.79757299004, 'qwen': 0.7353875391311111, 'resnet50': 0.7978008712844444, 'whisper': 0.7606414653805555}。
首轮 Top-B：ADD distilbert (delta_z=0.027609374); REMOVE paddleocr (delta_z=-0.037859279)。
仅对 historical reference 计结构距离：warm 至少 0 mutations；cold seed=('qwen',) 至少 3 mutations。这是路径长度参考，不是算法会走向该部署的保证。
KEEP CURRENT / NO_SWITCH 是合法候选动作；planning 本身不能确认 latency improvement，committed switches=0。
历史 final 与 current 相同，因此 NO_SWITCH 合理且可保留；即使 LP 找到 ADD，也必须由未来真实反馈决定，不能因为历史相同就强制 no-op。

### W1_balanced -> W3_specialist_friendly

Current feasible=True；bottleneck=['qwen']；z=0.058832163。
Pressure tasks=['text', 'visual_qa']；capacity slack={'distilbert': 0.0, 'paddleocr': 0.79757299004, 'qwen': 0.7411678370471112, 'resnet50': 0.7978008712844444, 'whisper': 0.7706799030946667}。
首轮 Top-B：ADD distilbert (delta_z=0.030908368); REMOVE paddleocr (delta_z=-0.037859279)。
仅对 historical reference 计结构距离：warm 至少 0 mutations；cold seed=('qwen',) 至少 3 mutations。这是路径长度参考，不是算法会走向该部署的保证。
KEEP CURRENT / NO_SWITCH 是合法候选动作；planning 本身不能确认 latency improvement，committed switches=0。
历史 final 与 current 相同，因此 NO_SWITCH 合理且可保留；即使 LP 找到 ADD，也必须由未来真实反馈决定，不能因为历史相同就强制 no-op。

## Scope and verification

Formal state S_t=(x_t,r_t,lambda_t)；新事件显式给 lambda_old/lambda_new，r_t 为 canonical。AdaptationHistory 的 phase_id 明确指定，不推断 phase。
Dry-run 返回 PLAN_ONLY 并保留原 deployment，不计 accepted switch；真实收益和 switching latency 尚未验证。
四个真实 dry-run 的 MILP calls=0；cold seed=qwen 只读取 Module 3C 已保存的 seed_deployment，用于 cold reference path 距离。
本阶段未进入真实 dynamic transition、forecasting、change-point detection、prewarming、replica、joint routing search 或 final paper benchmark。
测试结果和 no-GPU / enumeration 审计见 validation_manifest.json。
