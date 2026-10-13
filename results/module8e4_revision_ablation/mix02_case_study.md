# Mix_02 mechanism case study

Static post-validation ablation; no revised algorithm or held-out claim after tuning.

## heldout_mix_02_moderate

- M00: Phi→Qwen exact replacement generated=False; records=[]; stop=NO_RUNTIME_IMPROVEMENT
- M10: Phi→Qwen exact replacement generated=True; records=[{"iteration": 1, "rank": 6, "Top_B": false, "runtime_evaluated": false, "accepted": false, "analytical_delta_z": 0.006031090827427643, "reason": ""}]; stop=NO_RUNTIME_IMPROVEMENT
- M01: Phi→Qwen exact replacement generated=False; records=[]; stop=NO_RUNTIME_IMPROVEMENT
- M11: Phi→Qwen exact replacement generated=True; records=[{"iteration": 1, "rank": 2, "Top_B": true, "runtime_evaluated": true, "accepted": true, "analytical_delta_z": 0.006031090827427643, "reason": "mean_improved"}]; stop=NO_RUNTIME_IMPROVEMENT

M00/M01 use only x±1, so cannot generate this two-count replacement in the same neighborhood. M10 generation and exact global rank are shown above; M11 generation and diversified Top-B admission are shown above. An unmeasured candidate is identified as beneficial only by a matching actually measured full-v2 state, with incumbent-path limitations.

## heldout_mix_02_high

- M00: Phi→Qwen exact replacement generated=False; records=[]; stop=NO_RUNTIME_IMPROVEMENT
- M10: Phi→Qwen exact replacement generated=True; records=[{"iteration": 1, "rank": 6, "Top_B": false, "runtime_evaluated": false, "accepted": false, "analytical_delta_z": 0.01085596348936975, "reason": ""}]; stop=NO_RUNTIME_IMPROVEMENT
- M01: Phi→Qwen exact replacement generated=False; records=[]; stop=NO_RUNTIME_IMPROVEMENT
- M11: Phi→Qwen exact replacement generated=True; records=[{"iteration": 1, "rank": 2, "Top_B": true, "runtime_evaluated": true, "accepted": true, "analytical_delta_z": 0.01085596348936975, "reason": "mean_improved"}]; stop=NO_RUNTIME_IMPROVEMENT

M00/M01 use only x±1, so cannot generate this two-count replacement in the same neighborhood. M10 generation and exact global rank are shown above; M11 generation and diversified Top-B admission are shown above. An unmeasured candidate is identified as beneficial only by a matching actually measured full-v2 state, with incumbent-path limitations.
