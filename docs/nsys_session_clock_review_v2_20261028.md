# Read-only Nsight session clock review

New output: `artifacts/predictor-session-clock-audit-v2-20261028/audit.json`.
All four original prepared01 smoke SQLite files contain
`TARGET_INFO_SESSION_START_TIME.utcEpochNs` and `systemClockNs`. Their
`utcEpochNs-systemClockNs` values agree within1ns across sessions. Their recorded
target display names also agree. Input hashes remain unchanged.

The installed official Nsight 2025.6.3 schema calls `systemClockNs` the target
system-clock timestamp at capture start. The User Guide's Time Synchronization
section explains that reports from the same target can use TSC values with
nanoseconds-level alignment. Its fallback UTC/NTP method has millisecond-scale
limitations. SQLite does not explicitly give the report alignment source,
device/host clock-calibration error, or a drift/error bound. Agreement of origin
metadata supports a common-clock mapping candidate; integer nanosecond fields
and a1ns origin-consistency spread alone do not establish1ns accuracy.

The retained worker source pushes NVTX before recording its perf start. At end,
it records perf time, emits and flushes PHASE, then pops NVTX. These endpoints
are causally nested; logging time affects their separation. Under a constant
offset `C`, each phase supplies `perf_end-NVTX_end <= C <= perf_start-NVTX_start`.
Intersecting these brackets gives:

| Owner | Bracket width | Median offset minus session origin | Largest old boundary residual |
|---|---:|---:|---:|
| text#0 | 9,359ns | -3,813ns | 187,206ns |
| text#1 | 8,979ns | -3,833ns | 49,939ns |
| generalist_2#0 | 9,256ns | -3,935ns | 48,324ns |
| ocr#0 | 77,579ns | -28,854ns | 38,790ns |

The session origin lies inside all four intervals. These are conditional
constant-offset brackets derived from the retained source ordering, not a
measured clock-drift or CUDA timestamp-calibration error bound. The report also
preserves conservative pairwise offset intervals; it never turns the maximum
boundary residual or the5ms parser validity tolerance into an alignment-accuracy
claim. In particular, the present evidence does not qualify1µs cross-process
kernel alignment. Changing a point origin alone would not establish that bound.

For further analysis, preserve `systemClockNs` as the metadata-supported origin
candidate, use actual same-target/TSC identity and exported timestamp-shift
settings, and verify independent tightly bracketed clock samples or a documented
calibration bound before asserting a numerical cross-process precision. Record
each capture's full clock basis and uncertainty. Within one process, a constant
offset cancels from CUPTI durations and relative kernel order; those quantities
do not acquire the cross-process offset uncertainty. Profiling perturbation is
a separate issue.

`tools/predictor_validation_20261027/session_clock_audit_v2.py` implements this
read-only audit without changing the frozen exporter or source traces. Three
CPU tests check interval intersection, incompatible brackets and malformed or
absent samples. Official documentation and worker/observer source hashes are
preserved in the report. No new profiler, clock calibration experiment, model,
or GPU work was started.
