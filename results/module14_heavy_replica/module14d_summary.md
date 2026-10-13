# Module 14D：Minimal Heavy Replica Profiling + Empirical Structural Revalidation

Revised gate：**HEAVY_TRANSITION_EMPIRICALLY_SUPPORTED**。27 个原 development contexts、E/H pool、quality/capability SLO、任务映射、memory budgets、Module9、J-routing、rho≤0.8 均保持冻结。

## 必须回答的 11 项

1. 成功测得 7/7 条 task-path g(2)、M(2)，共6个checkpoint，每条x=2均3次独立repetition。原g(1)=1与M(1)保留；Turbo、RoBERTa-base、ConvNeXt-tiny既有副本表和raw证据完整性通过，未重测。
2. 真实不可行/失败 replica states：0。错误完整保留，不填估计值。首次STRHub启动的READY元数据兼容性错误单独保存在infrastructure_archive，未进入capacity计时，不解释为模型OOM或不可行；修复只补READY.tasks，不改handler、checkpoint、预处理、计时或并发。
3. 14C 的14个N_S reduction剩 **10/14**；消失 **4**。本轮总N_S reduction 10；排除近零flow驻留后 10。
4. Generalist routing share increase：**0** 个context。
5. 新G/S split：**0** 个context。
6. N_G increase：**0** 个context。
7. 新second G：**0** 个context。
8. 原8个Heavy infeasible：仍不可行 8；恢复 []；新增 []。E可行 27/27；H可行 19/27。每个不可行context保留最低实测task-cover memory下界。
9. x=3 profiling decision：**X3_PROFILING_REQUIRED**。只有completed x2 replay后，依据最优解使用已测x=2、仍并行使用同任务其它队列以及实测x1→x2正capacity增量生成需求；逐模型context witness见x3_profiling_decision.json。未运行任何新x≥3，未生成未测g/M。
10. Revised gate：**HEAVY_TRANSITION_EMPIRICALLY_SUPPORTED**。多个context持久N_S reduction符合用户gate；它本身不证明Generalist-sharing gain。
11. 自动进入正式bounded real-GPU Efficiency-vs-Heavy validation：**YES**。已执行的后续结果独立保存在14E目录；14F未启动。最终后续停止原因：**STOP_GPU_MEMORY_BUDGET_REJECTION**。

## 新实测点

|checkpoint/task path|g(2)|M(2) idle MiB|M(2) safe MiB|x2 repetitions|
|---|---:|---:|---:|---:|
|baudm/parseq:abinet / ocr|1.905592925|1288.000000|1430.779894|3|
|baudm/parseq:parseq-patch16-224 / ocr|1.878988990|1144.000000|1270.816925|3|
|baudm/parseq:parseq / ocr|1.807922326|1144.000000|1270.816925|3|
|baudm/parseq:trba / ocr|1.959827882|1428.000000|1586.299448|3|
|facebook/mms-lid-256 / lid|0.989708589|10998.000000|12217.171800|3|
|openai/whisper-medium / asr|1.084341601|7485.333333|8315.112140|3|
|openai/whisper-medium / lid|1.031670243|7478.666667|8307.706447|3|

## 协议与解释边界

Capacity沿用11B closed loop：5s warmup、30s timed capacity、30s独立lower-load stability validation，三组seed7101/7202/7303，fresh process，每worker1个outstanding。g(2)=原g(1)×三次paired capacity ratio中位数，计时窗口只统计成功完成响应，drain不计throughput。辅助x=1 reference只形成真实paired denominator，不覆盖冻结g(1)/M(1)。M(2)为postwarm idle实测均值，safe factor沿用原冻结值，另保留GPU high-water与完整request/telemetry。periodic offered-rate只用于独立稳定性，不用作service capacity。

Whisper-medium的ASR和LID分别实测，task-path gain均完整保留；同checkpoint独立worker共享模型身份与residency语义。原优化器是family scalar g(x)：本轮采用两个实测gain较小值、两个path实测residency较大值，checkpoint每副本只计一次驻留。它是保守的family transfer，不是未测mixed-task capacity证书；Module9、J和任务原s0不改，独立GPU验证承担实际混合并发核验。

27行context比较、54行CPU结果均有完整witness。原E的27个J*复现，逐行独立重算flow conservation、safe memory、rho与J通过，所有可行CPU结果global gap≤1e-8秒。未经测量的count保持unknown，不线性外推、不把empirical count上界当真实Specialist saturation。

## 消除的missing-replica artifacts

- mix_03_I1/B3
- mix_03_I2/B6
- mix_03_I3/B6
- reasoning_vqa_heavy/B6

## 后续执行与强制停止

{
  "GPU_validation_entered": true,
  "GPU_validation": {
    "at_utc": "2026-10-07T20:00:28.775617+00:00",
    "gate": "STOP_GPU_MEMORY_BUDGET_REJECTION",
    "comparisons": [
      {
        "context": "no_vqa_phase/B3",
        "all_sessions_valid": true,
        "E_steady_repetitions": 3,
        "H_steady_repetitions": 3,
        "H_unstable_or_failed_repetitions": 0,
        "CPU_NS_reduction": true,
        "CPU_G_share_increase": false,
        "safe_memory_budget_verified": true,
        "systematic_operational_contradiction": false,
        "H_mean_E2E_ms": {
          "n": 3,
          "mean": 27.09262362087735,
          "median": 26.893888978260875,
          "min": 26.021183477777775,
          "max": 28.362798406593406,
          "sample_std": 1.18338992405081
        },
        "E_mean_E2E_ms": {
          "n": 3,
          "mean": 19.393707456334162,
          "median": 19.293421065217384,
          "min": 17.505150644444445,
          "max": 21.38255065934066,
          "sample_std": 1.940644413348749
        }
      },
      {
        "context": "mix_03_I3/B6",
        "all_sessions_valid": true,
        "E_steady_repetitions": 0,
        "H_steady_repetitions": 0,
        "H_unstable_or_failed_repetitions": 3,
        "CPU_NS_reduction": false,
        "CPU_G_share_increase": false,
        "safe_memory_budget_verified": false,
        "systematic_operational_contradiction": false,
        "H_mean_E2E_ms": {
          "n": 3,
          "mean": 3654.7031357813426,
          "median": 3672.0804271043685,
          "min": 3615.060019798428,
          "max": 3676.9689604412306,
          "sample_std": 34.41884546542051
        },
        "E_mean_E2E_ms": {
          "n": 3,
          "mean": 275.6733000386439,
          "median": 271.69884560269105,
          "min": 269.6523248596092,
          "max": 285.6687296536315,
          "sample_std": 8.716565933846535
        }
      }
    ],
    "physical_sessions": 12,
    "continue_stable_overflow_search": false,
    "Generalist_sharing_gain_proven": false,
    "CPU_identity_counts_were_validated_as_fixed_deployments_not_GPU_optimality": true
  },
  "stable_overflow_search_entered": false,
  "stable_overflow_search": null,
  "stop_reason": "STOP_GPU_MEMORY_BUDGET_REJECTION"
}

## 14E 的实际停止证据

12个独立物理会话全部完成并正常释放GPU。no_vqa_phase/B3双方均3/3 steady；mix_03_I3/B6双方均3/3 unstable，原阈值检测到持续尾部backlog增长。所有请求最终drain完成，不等于steady系统。高负载Heavy实测safe idle memory为49908.447–50763.804MiB，三次都高于原B6=49140MiB；E也三次超过safe budget（49295.255–49461.883MiB）。实际GPU high-water未超过物理容量，无OOM，但不能因此放宽冻结safe-memory预算。

14E的狭义相对判据（E steady而H三次unstable）未触发，因为高负载双方都unstable；绝对CPU joint feasibility/stability在该高负载context未获真实GPU支持。停止依据为固定safe-budget的实测拒绝及重复稳定性失败；未运行14F、hotspot GPU、1G→2G，未移动任何hotspot或修改参数。

Whisper-medium和MMS不要求x3；4个OCR checkpoint有x3 may-change-active-optimum witness。该决定先于14E结果生成，只记录后续profiling需求；本轮强制停止后未执行x3。
