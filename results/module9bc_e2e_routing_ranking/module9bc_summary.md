# Module 9BC — E2E-aware routing and thin candidate ranking

Completed and stopped. 9A remains I0; 21/21 offline profiles remain characterization evidence.

Final gates: **ROUTING_SUPPORTED**, **RANKING_PARTIAL**.

Final default: **J_ROUTING + DELTA_J**, Top-B=2, frozen v2 generation and real GPU accept/rollback.

## Small matched routing validation

| State | R0 mean ms | J mean ms | Stability |
|---|---:|---:|---|
| C06 low_pressure_1G | 28.440 | 27.411 | steady → steady |
| C07 moderate_pressure_1G | 323.151 | 60.817 | unstable → steady |
| C03 pressured_RC_SG_like | 10881.000 | 8594.984 | unstable → unstable |
| C10 historical_multi_G | 30.787 | 26.387 | steady → steady |

Each exact deployment/workload/routing has three independent repetitions, alternating arm order. Same sample pool, arrival protocol, 10-second warmup, 210-second measurement, worker and runtime. Paired task P95 and predictions are in routing_gpu_results.csv; empirical paired statistics retain each repetition.

## Bounded ranking sanity

| Cell | V2 minus V3 mean ms | V3 better / worse | Logical states V2 / V3 |
|---|---:|---|---|
| C07 | 5.914 | True / False | 5 / 6 |
| C06 | 0.292 | False / False | 5 / 5 |

Both ranking arms use canonical routing, isolating evaluator/ranking changes; search budgets and early stopping are identical. No confirmations or heldout campaign were added.

## Required answers

1. J vs mean E2E: steady Spearman=0.928923, Kendall=0.842245; steady+recoverable Spearman=0.952295, Kendall=0.862349.

2. z comparator: steady Spearman=0.784904, steady+recoverable=0.831961. J improves these admitted-tier correlations and within-cell ordering; all-tier pooled correlation is not uniformly better. It is a congestion surrogate, not an exact M/M/1 runtime model.

3. Historical 40 states mean routing L1=0.399539 req/s; mean L1/Lambda=0.005143. Actual reassigned traffic fraction is half L1/Lambda. 27/40 changed.

4. Task redistribution: {"text": {"mean_fraction": 5.08543430570106e-16, "maximum_fraction": 6.8253847105576825e-15}, "image": {"mean_fraction": 9.423125003284257e-15, "maximum_fraction": 1.9026904932369797e-13}, "ocr": {"mean_fraction": 0.0002908224112798811, "maximum_fraction": 0.011632896451195245}, "asr": {"mean_fraction": 0.06100136661442725, "maximum_fraction": 0.17121763039594207}, "visual_qa": {"mean_fraction": 0.022142636695758937, "maximum_fraction": 0.47618147364049496}, "lid": {"mean_fraction": 0.007430227653237307, "maximum_fraction": 0.12463771245598784}}. Largest path shifts are ASR/VQA/LID; text/image are numerical zero; OCR mostly zero.

5. Real GPU mean E2E: paired state results above and paired_routing_statistics.json. Matched-steady equal-state mean improvement=2.714443637210028 ms; 2 states better, 0 worse. Unstable latency reductions are not steady-serving improvements.

6. Stability: 1 state-tier improvements, 0 degradations; retain all three repetition tiers per arm.

7. The chosen multi-G state is C10 (Phi + Qwen + 5S). Qwen VQA traffic rises from 0.195149 to 0.540989 req/s, while its 0.406275 req/s LID traffic moves to Whisper-small; Phi VQA falls from 0.531128 to 0.185288 req/s. Qwen total traffic therefore falls slightly (0.601425 to 0.540989 req/s), while its allocation becomes more task-specific. All three paired runs remain steady and improve mean E2E: 30.786857 to 26.387218 ms on average. This supports better allocation in this development deployment, without establishing general multi-Generalist superiority. Other historical multi-G usage is retained in routing_distance_and_task_shift.json.

8. ROUTING gate = ROUTING_SUPPORTED under the rule frozen before new GPU labels.

9. DeltaJ vs real candidate mean improvement: Spearman=0.872650, Kendall=0.716857; 30 known labelled records, with repeated cells/iterations clustered for descriptive CI.

10. Delta-z: Spearman=0.586546, Kendall=0.481271. Pooled DeltaJ correlation improves, but within-iteration accuracy is 0.823529 vs 0.882353; mixed evidence, no formula tuning.

11. Top-2 known-subset hit: DeltaJ=1.0, Delta-z=1.0; no gain. Known-label counts={'2': 11, '3': 2}; most iterations have only two measured candidates, making Top2=1 a ceiling. All nonsteady best-tier objective ties count as hits; no unmeasured outcomes imputed.

12. Known selected unstable rates: DeltaJ=0.4230769230769231, Delta-z=0.4230769230769231; no reduction. These rates apply to known-label restricted sets, not unmeasured full-pool winners.

13. Structural oracle/second-Generalist direction: C09 iter1 M5: exact generated=False, second-G ADD=True, ranks=[{"deployment": {"convnext_tiny": 0, "distil_whisper": 0, "distilbert": 1, "minicpm": 1, "paddleocr": 1, "phi4mm": 1, "qwen": 0, "resnet50": 1, "roberta_sst2": 0, "whisper-small": 1, "whisper_turbo": 0}, "new": 6, "old": 5}]; C09 iter1 M6: exact generated=False, second-G ADD=True, ranks=[{"deployment": {"convnext_tiny": 0, "distil_whisper": 0, "distilbert": 1, "minicpm": 1, "paddleocr": 1, "phi4mm": 1, "qwen": 0, "resnet50": 1, "roberta_sst2": 0, "whisper-small": 1, "whisper_turbo": 0}, "new": 6, "old": 5}, {"deployment": {"convnext_tiny": 0, "distil_whisper": 0, "distilbert": 1, "minicpm": 0, "paddleocr": 1, "phi4mm": 1, "qwen": 1, "resnet50": 1, "roberta_sst2": 0, "whisper-small": 1, "whisper_turbo": 0}, "new": 5, "old": 4}]; C12 iter1 M5: exact generated=False, second-G ADD=True, ranks=[{"deployment": {"convnext_tiny": 0, "distil_whisper": 0, "distilbert": 1, "minicpm": 1, "paddleocr": 1, "phi4mm": 1, "qwen": 0, "resnet50": 1, "roberta_sst2": 0, "whisper-small": 1, "whisper_turbo": 0}, "new": 6, "old": 5}]; C12 iter1 M6: exact generated=False, second-G ADD=True, ranks=[{"deployment": {"convnext_tiny": 0, "distil_whisper": 0, "distilbert": 1, "minicpm": 1, "paddleocr": 1, "phi4mm": 1, "qwen": 0, "resnet50": 1, "roberta_sst2": 0, "whisper-small": 1, "whisper_turbo": 0}, "new": 6, "old": 5}, {"deployment": {"convnext_tiny": 0, "distil_whisper": 0, "distilbert": 1, "minicpm": 0, "paddleocr": 1, "phi4mm": 1, "qwen": 1, "resnet50": 1, "roberta_sst2": 0, "whisper-small": 1, "whisper_turbo": 0}, "new": 5, "old": 4}]. Missing exact oracle vectors remain CANDIDATE_GENERATION_LIMIT; generated ADD ranks may also be affected by evaluator or budget. No neighborhood expansion was made.

14. Same frozen 6-state/3-iteration/Top2 limits: [{"cell": "C07", "V3_better": true, "V3_worse": false, "reason": "mean_improved", "reverse_reason": "mean_worse", "same_final_deployment": false, "mean_improvement_ms": 5.913772449770654, "V2_budget_used": 5, "V3_budget_used": 6}, {"cell": "C06", "V3_better": false, "V3_worse": false, "reason": "p95_worse", "reverse_reason": "latency_gain_below_min_relative_improvement", "same_final_deployment": true, "mean_improvement_ms": 0.2920230936720074, "V2_budget_used": 5, "V3_budget_used": 5}]. Actual used budgets can differ due to unchanged early stopping; caps and accept/rollback match exactly. Ranking-only sanity used canonical routes in both arms.

15. RANKING gate = RANKING_PARTIAL; finite measured-subset and bounded sanity evidence only.

16. V3_MINIMAL = J_ROUTING + DELTA_J; selection Case A. Unsupported components remain diagnostic alternatives. No component adopted merely to create a v3 label.

17. Uncached candidate J* evaluation median=3.707 ms, P95=4.184 ms; full generation/LP/J-ranking including seed median=68.200 ms.

18. Yes: the deterministic CPU implementation, bounded development validation and frozen default are sufficient to enter the next Specialist Saturation benchmark when requested. CPU candidate evaluation adds milliseconds and requires no GPU profiling. Ranking evidence remains partial; the combined J-routing + DeltaJ stack has component validation, not a new full end-to-end campaign. This module did not start the next benchmark.

19. No additional predictor/kernel feature/weight/neighborhood work is required. Preserve the OCR execution environment override in future launches; startup restoration is documented, with no source/package change. Any future benchmark must use the frozen default selected here, not assume J-routing success from CPU correlation.

## Evidence and limitations

J = sum_m D_m/(1-D_m/g_m) / Lambda, in seconds/request. All 130 replay states passed conservation, resident/quality/capability support, capacity domain, repeated solve equality and convex-gap certificates. The solver uses no random restart. An initial CPU preflight failure from objective conditioning was preserved separately; expressing the identical objective in milliseconds fixed numerical stopping without changing the formula.

Historical runtime audit deduplicates 192 physical repetitions into 40 exact states; three repeats are not treated as independent state observations. Correlation across load levels may be confounded, so within-cell and within-iteration ordering is also reported. Top-k uses only measured/known-label candidates; unmeasured outcomes stay unknown.

Accounting: {"physical_attempts": 88, "timed_runs": 87, "valid_timed_runs": 87, "routing_valid_runs": 24, "ranking_valid_runs": 63, "premeasurement_launch_failures": 1, "frozen_routing_max_timed_runs": 24, "frozen_ranking_max_timed_runs": 72, "selected_routing_states": 4, "selected_ranking_cells": 2, "recovery": "exact existing frozen environment restoration; one preserved premeasurement failure", "outcome_based_reruns": 0, "new_profiles": 0, "extra_states": 0}. One premeasurement PaddleOCR failure is preserved; exact existing frozen execution environment overrides restored the launch. No timed serving labels were discarded and no outcome-based repeats added. All original 8F/9A sources and artifacts reverified. No MPS, worker/runtime modification, new task, memory point, hardware, profiling, interference model, neighborhood, or future benchmark.

## Historical surrogate audit

| Tier subset | Predictor | States | Mean Spearman | Mean Kendall | Mean ordering accuracy | P95 Spearman |
|---|---|---:|---:|---:|---:|---:|
| steady | z | 15 | 0.7849 | 0.7432 | 0.8571 | 0.7021 |
| steady | J_seconds | 15 | 0.9289 | 0.8422 | 0.9048 | 0.8821 |
| steady_plus_recoverable | z | 18 | 0.8320 | 0.7411 | 0.8595 | 0.7801 |
| steady_plus_recoverable | J_seconds | 18 | 0.9523 | 0.8623 | 0.9183 | 0.9253 |
| all_development | z | 40 | 0.9695 | 0.8927 | 0.9327 | 0.9460 |
| all_development | J_seconds | 40 | 0.9501 | 0.8504 | 0.9122 | 0.9371 |

Full P95/stability metrics and within-cell ordering are in surrogate_audit.json.

## Ranking outcome attribution

[
  {
    "cell": "C07",
    "common_query_prefix_states": 4,
    "V2_states": 5,
    "V3_states": 6,
    "V3_winning_deployment_already_measured_by_V2": true,
    "previous_V2_winner_labels": [
      {
        "mean_e2e_ms": 168.50377747758628,
        "weighted_p95_ms": 463.68360918603526,
        "stability": "unstable"
      }
    ],
    "V3_winner_labels": {
      "mean_e2e_ms": 34.66569865185375,
      "weighted_p95_ms": 64.61396906640331,
      "stability": "steady"
    },
    "V3_winner_repetition_labels": [
      {
        "mean_e2e_ms": 36.709227068717574,
        "weighted_p95_ms": 74.7921675453861,
        "stability": "steady"
      },
      {
        "mean_e2e_ms": 34.393256092490276,
        "weighted_p95_ms": 62.56784132333443,
        "stability": "steady"
      },
      {
        "mean_e2e_ms": 32.89461279435341,
        "weighted_p95_ms": 56.481898330489415,
        "stability": "steady"
      }
    ],
    "previous_V2_winner_repetition_labels": [
      [
        {
          "mean_e2e_ms": 439.43247386178837,
          "weighted_p95_ms": 1275.7629273429754,
          "stability": "unstable"
        },
        {
          "mean_e2e_ms": 33.76863789917279,
          "weighted_p95_ms": 60.1696439248974,
          "stability": "steady"
        },
        {
          "mean_e2e_ms": 32.31022067179772,
          "weighted_p95_ms": 55.118256290232964,
          "stability": "steady"
        }
      ]
    ],
    "causal_limit": "When the winning deployment was already queried by V2, different final outcomes cannot establish exclusive discovery by the ranking. Independent measurements and unchanged accept/rollback can change transitions."
  },
  {
    "cell": "C06",
    "common_query_prefix_states": 4,
    "V2_states": 5,
    "V3_states": 5,
    "V3_winning_deployment_already_measured_by_V2": true,
    "previous_V2_winner_labels": [
      {
        "mean_e2e_ms": 26.900216239378814,
        "weighted_p95_ms": 39.2264939457495,
        "stability": "steady"
      }
    ],
    "V3_winner_labels": {
      "mean_e2e_ms": 26.608193145706807,
      "weighted_p95_ms": 39.37065300293477,
      "stability": "steady"
    },
    "V3_winner_repetition_labels": [
      {
        "mean_e2e_ms": 26.627756718755432,
        "weighted_p95_ms": 38.80461295062899,
        "stability": "steady"
      },
      {
        "mean_e2e_ms": 26.61941974271123,
        "weighted_p95_ms": 39.79715321627504,
        "stability": "steady"
      },
      {
        "mean_e2e_ms": 26.577402975653754,
        "weighted_p95_ms": 39.51019284190028,
        "stability": "steady"
      }
    ],
    "previous_V2_winner_repetition_labels": [
      [
        {
          "mean_e2e_ms": 26.922600510927953,
          "weighted_p95_ms": 38.23542241900152,
          "stability": "steady"
        },
        {
          "mean_e2e_ms": 26.87467622657001,
          "weighted_p95_ms": 39.82710774288281,
          "stability": "steady"
        },
        {
          "mean_e2e_ms": 26.903371980638482,
          "weighted_p95_ms": 39.616951675364184,
          "stability": "steady"
        }
      ]
    ],
    "causal_limit": "When the winning deployment was already queried by V2, different final outcomes cannot establish exclusive discovery by the ranking. Independent measurements and unchanged accept/rollback can change transitions."
  }
]

A better final measured state under the same available budget is a sanity outcome, not by itself a causal ranking certificate. In particular, a candidate queried in both arms can receive different repetition/stability labels and change acceptance. These limits do not alter the frozen formula or gate rule.
