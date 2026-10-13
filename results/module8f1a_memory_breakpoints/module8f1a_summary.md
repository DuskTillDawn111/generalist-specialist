# Module 8F-1A — Empirical Memory Breakpoint Audit

完成纯离线审计。static v2、routing、quality policy、candidate pool、replica evidence bounds 均未修改。新 GPU experiment / resource sweep / deployment optimizer 调用均为 0；不冻结最终 workload intensity。

主口径：全部 12 个当前冻结 workload 的 capability / quality / frozen planning-capacity feasible 集合交集；逐 workload、原始六项交集、heldout 六项交集及并集也完整保存。无 admitted edge 的 PaddleOCR-VL 保留在 12-model 原始枚举中，但按现有冻结 planning convention 在 feasible 集合中固定 x=0。

1. 实际枚举 **13,824** 个 vectors，理论状态数 3³ × 2⁹ = 13,824（含空向量和超过物理预算的向量）。
2. 历史独立 steady measurements n=453；epsilon_mem=0.110853955254881（11.085395525%），safety factor=1.11085395525488。使用全设备 sampled lifecycle peak；不是 5%/10% margin。positive error min/mean/median/p95/max = 0, 0.0524316771649, 0.0554836622452, 0.0875336999892, 0.110853955255。
3. 各 workload B_min（MiB）：

| Workload | B_min MiB | GiB |
|---|---:|---:|
| demand_balanced | 17007.174054952 | 16.608568413 |
| speech_heavy | 17007.174054952 | 16.608568413 |
| vision_heavy | 17007.174054952 | 16.608568413 |
| text_ocr_heavy | 17007.174054952 | 16.608568413 |
| reasoning_vqa_heavy | 17007.174054952 | 16.608568413 |
| no_vqa_phase | 3930.201293692 | 3.838087201 |
| heldout_mix_01_moderate | 17007.174054952 | 16.608568413 |
| heldout_mix_01_high | 17007.174054952 | 16.608568413 |
| heldout_mix_02_moderate | 17007.174054952 | 16.608568413 |
| heldout_mix_02_high | 17007.174054952 | 16.608568413 |
| heldout_mix_03_moderate | 17007.174054952 | 16.608568413 |
| heldout_mix_03_high | 17007.174054952 | 16.608568413 |

4. B_min_global = **17007.174054952 MiB**。
5. Frozen SG B_SG_global = **27980.189424960 MiB**；SG distribution 见 specialist_greedy_thresholds.json；所有 frozen fixed routes 的 capacity feasibility = True。
6. x=2 的 first-possible breakpoint（any workload）与全部 workload 交集 threshold：
   - qwen: 40255.125630526 MiB；交集 40255.125630526 MiB。
   - resnet50: 4601.157082666 MiB；交集 17678.129843926 MiB。
   - whisper-small: 5734.228117026 MiB；交集 18811.200878286 MiB。
7. 全枚举 unique memory breakpoints = **11,689**（含 0）；物理预算内 6,554；非零总数 11,688。
8. Full 49140 MiB 下全部 workload 交集 feasible deployments = **1,834**；并集 = 2,074。
9–12. 五个候选 budget、feasible-space 比例、SG 与 x=2：

| Budget | MiB | GiB | Feasible count | Fraction | All SG memory feasible | x=2 models |
|---|---:|---:|---:|---:|---|---|
| B1 | 17007.174054952 | 16.608568413 | 1 | 0.054525627% | False | none |
| B2 | 32019.254406267 | 31.268803131 | 612 | 33.369683751% | True | resnet50, whisper-small |
| B3 | 32652.441160762 | 31.887149571 | 639 | 34.841875682% | True | resnet50, whisper-small |
| B4 | 43736.541926295 | 42.711466725 | 1223 | 66.684841876% | True | qwen, resnet50, whisper-small |
| B5 | 49140.000000000 | 47.988281250 | 1834 | 100.000000000% | True | qwen, resnet50, whisper-small |

13. 原始预算 = [17007.17405495223, 32019.254406266693, 27980.189424959943, 43736.54192629518, 49140] MiB。实测最小正 memory increment=568.0 MiB，safe separation=630.965046585 MiB。发生 deterministic dedup 的点：B3 (empirically_tiny_spacing,order_conflict)；逐步 next-unused breakpoint 记录见 candidate_memory_budgets.json。
14. B1–B4 完全来自已有 empirical memory、历史误差、冻结 quality/capacity 与提前写定的规则，没有使用 Proposed runtime performance。B5=49140 MiB 是用户指定物理端点，**不是 empirical sum breakpoint**；“所有预算均为 empirical breakpoint”与“B5 精确等于 physical maximum”无法同时满足，已保留端点并明确记录例外，不伪造断点。
15. 可以进入 Module 8F-1B Workload-Intensity Freeze：memory audit 已完成，保守安全系数后仍有 1,834 个统一 feasible states，统一 minimum 上方 headroom=32132.825945048 MiB。最终 resource-sweep budget 全断点条件存在上述 physical-endpoint 例外；本阶段不进入 8F-1B、不冻结 intensity、不运行 sweep。

Physical sanity：所有候选 B_i≤49140 MiB；最大 safe feasible state=49128.627025102 MiB，剩余 11.372974898 MiB。epsilon_mem 未调整。

可复现：`/root/miniconda3/envs/gsdeploy/bin/python results/module8f1a_memory_breakpoints/run_audit.py`。部署 CSV 每个 workload 的 capacity_certificate 列指向 capacity_feasibility_certificates.json 中的实证 LP witness，可复核所有 feasible set。CSV/JSON 保留计算值，Markdown 显示小数不用于预算选择。历史 sampled maximum 仅为已有数据的误差上界，不保证未观测的峰值。

完成后停止。
