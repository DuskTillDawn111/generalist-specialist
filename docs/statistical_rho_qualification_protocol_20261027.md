# Prospective statistical rho-profile qualification

The original rho rule is a statistical planning rule based on service profiles and g(n). It does not logically require a certified physical capacity upper bound or a measured profile under every random input seed. A source-bound prospective transfer interface is therefore scientifically legitimate, provided its statistical evidence and independent applicability check are explicit and it does not pretend to change the frozen resolver's evidence labels.

The new independent `flow_capacity_qualification.py` seals a fixed 20% mean-service applicability rule before target startup. Each active family/task must satisfy `abs(observed worker END−START mean − training worker mean)/training worker mean ≤ .2`, using every completed offered-cohort request including bounded drain. It additionally requires the exact bound trace/resources, complete clean native collection, fully covered ordered owned power/memory telemetry, the unchanged original native stability decision, and the unchanged original rho `.8 + 1e-8` threshold. Missing task support does not pass. The per-family/task checks prevent a high-volume text population from masking a poor low-rate task estimate.

The frozen v11 artifact is `artifacts/predictor-gpu-20261027/flow-validation-v11/capacity-qualification-protocol.json`, created at monotonic time `5998030788458373`, SHA256 `5320068af3b9982de50bfdb574d7baaa9e4656463ec57820b89f47d99dca1b18`. At sealing, the target directory had only its prepared plan, plan and transfer artifact; no target collection was opened. Reporting also requires the protocol timestamp to precede every actual target startup, and rechecks the complete transfer provenance and arithmetic. No v9 result was used.

When all requirements pass, the report can state `statistical_original_rho_planning_admission_passed=true`. In parallel it retains the original target resolver result, `old_target_resolver_admission=null`, `physical_capacity_bound_certified=false`, and `final_gain_eligible=false`. These fields answer different questions: usable independently checked statistical planning estimate; what the frozen measured-key resolver knows; and whether a physical bound or final utility improvement has been established.

This is one operating-run applicability check on a precisely matched H90 condition. It is not a confidence interval, a stationary capacity proof, a long-horizon qualification, a guarantee over unseen inputs, or a final six-term gain result. The 20% rule is a predeclared operational tolerance, not a probability bound. Its interaction with the .8 planning margin is understandable numerically, but neither the error tolerance nor the margin turns sampled service into a worst-case guarantee. Additional repetitions can strengthen the evidence without relabeling this single-run test.

Run after all declared target native raws are available:

```
python -m tools.predictor_validation_20261027.flow_capacity_qualification report \
  --protocol artifacts/predictor-gpu-20261027/flow-validation-v11/capacity-qualification-protocol.json \
  --raw NATIVE_RAW_1 --raw NATIVE_RAW_2 --raw NATIVE_RAW_3 --raw NATIVE_RAW_4 \
  --output NEW_REPORT_DIRECTORY
```

Three CPU checks cover the inclusive 20% boundary, independent rho/stability/telemetry failure, missing active-task support, and prospective timestamp enforcement. The implementation and frozen artifact remain unchanged after sealing.
