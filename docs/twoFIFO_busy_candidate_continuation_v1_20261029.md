# Distinct hypothetical busy continuations

`twoFIFO_busy_candidate_continuation_v1.evaluate` closes a small CPU composition
gap: each predefined candidate now generates its own complete v5_2 checkpoint
at the collector cutoff and resumes that checkpoint over the complete original
public calendar. The native guarded entry point is unchanged. This module does
not infer a native checkpoint, identify a posterior, or certify source evidence.

Each candidate uses the existing exact positive draw weight and joint
plans/rates/protocol-slacks/transport inputs, plus its own `initial_state`.
The latter must contain all three explicit original FIFO/protocol states, or
an existing checked `settled_initial_state`. A caller cannot insert an arbitrary
busy checkpoint. Missing scientific support uses `UnknownForecastInputs`.
Invalid contracts and implementation failures propagate as errors.

The necessary boundary screen compares known sender clocks, observed physical
START occurrence, and the complete modeled collector-processed RESULT/READY
pattern with all four clocks. Absent RESULT is collector-processing survival;
a physically finished handler, or even an already reader-received RESULT, may
remain compatible. Missing START never proves that a candidate has not started.
The engine does not model START notice transport, so this screen cannot establish
compatibility with the entire native observation likelihood. Native PHASE and
QUERY events have no authenticated mapping to generated semantic marks here;
they retain UNKNOWN mass. Prior history outside the modeled cohort does too.
No host elapsed time becomes GPU progress.

Each mismatch or unsupported slot retains its original weight as UNKNOWN.
There is no survivor renormalization. Reported finite weighted means describe
the supplied hypothetical catalog. They are not a native conditional forecast,
and the catalog is not claimed to cover every feasible residual state. UNKNOWN
mass makes the full finite-catalog expectation unavailable; the separately
labeled conditional-on-known mean does not replace it. Original six-term score,
full public denominator, future arrivals, and fixed-drain risk intervals are
reused. Missing resources keep full utility UNKNOWN.

Existing DAG modules already demonstrate the multiple-state mathematics:
`dag_busy_posterior_v8.condition_busy_scenarios` and
`dag_visible_busy_v10.condition_visible_phase_scenarios` preserve distinct
residual DAG trajectories and distinguish physical completion from reader
receipt. Their observer schema and training support differ from current
collector-processing twoFIFO observations, and their empirical survivor
renormalization is not copied here. `rolling_visible_queue_v1.forecast_suffix`
still requires a directly observed busy cursor and does not solve this gap.

Historical training evidence does not validate the new native application:

* `predictor-native-queue-phase-busy-v10-20261028-training01/report.json`
  reports `FAILED_ACTUAL_SOURCE_DIAGNOSTIC`, because joint reader support and
  target full context differ. Posterior and risk bounds are uncalibrated.
* `predictor-native-queue-busy-prefix-v10-20261028-training01/interruption-report.json`
  reports no completed cutoff result after at least 940 seconds wall time;
  the static encoding run was interrupted.
* `predictor-native-queue-busy-replay-v9-20261028-training02/failure-report.json`
  reports a DAG reader chronology failure and no completed cutoff result.
* `predictor-visible-dag-cutoff-replay-v8-20261028-final01/report.json`
  demonstrates an idle suffix forecast, but its started-busy cutoff remains
  UNKNOWN with `STARTED_RESIDUAL_UNKNOWN`.

Native eligibility still needs source-bound same-context joint candidate support,
START notification transport, PHASE/QUERY-to-profile/stream dependency mapping,
sender/collector inflight facts, dynamic output/EOS support, and independent
coverage/calibration evidence. Those facts cannot be supplied by CPU fixtures.

The eight new CPU tests include two distinct residual worlds with the same
visible START; a handler finished with reader RESULT arrived but collector
processing pending; exact checkpoint/full-rollout equality; future public work
and original score retention; mismatch and missing-donor mass; processed
RESULT/READY clock matching; opaque future raw bodies; producer identity;
and the original native guard. The related suite has 52 passing tests.

Receipt and synthetic outputs are in
`artifacts/predictor-twoFIFO-busy-candidate-continuation-v1-20261029-cpu01/`.
They are CPU composition evidence, not a real training fit or native validation.
