# Unified native DAG intake: independent CPU review

The corrected `cpu02` intake passes 10 author tests and 7 independent tests.
The independent artifact report is
`artifacts/predictor-unified-kernel-dag-intake-v1-20261029-cpu02-independent/report.json`;
`verification.json` also binds the test transcript, source hashes, and the
pre-fix counterexample to the corrected implementation.

The review found and reproduced one fail-closed defect: reversed outside
intervals could cancel the endpoint deltas of a real overlapping outside
kernel. The old source and artifact remain preserved under `cpu01`; the author
now rejects noninteger identities and every nonpositive, reversed, or negative
native interval before attribution or sweeping. The independent regression
confirms rejection. Every one of the 371,244 actual SQLite kernels has a valid
positive integer interval, so this malformed-input defect did not invalidate
the original measured cohort.

The read-only artifact audit checks 547 distinct source/artifact receipts;
all 371,244 SQLite kernels and their occupancy declaration members; all 55,891
measured catalog/capture/path nodes; exact source, PID, stream, correlation,
context, device, timestamps, launch configuration, full hardware declaration,
and occupancy patch attribution. The measured/outside half-open union overlap
is zero. Every actual kernel belongs to device 0. OCR retains streams 7 and 13.

An independently rebuilt active-set sweep checks 87,847 intervals against the
six static covariates, including 168 intervals with another kernel of the same
owner. The static covariate function is shared with the implementation;
source attribution, SQL complement, active-set sweep, and path comparisons are
independent. The current whole-session outside sweep conservatively rejects
cross-device overlap as well; future multi-device support would need a device
domain filter. That does not affect this single-device capture.

Path covariate rank is six, but all 55,891 exact-node baseline groups have one
response each. The nuisance-adjusted derivative rank is therefore zero. No fit,
GPU launch, validation-label access, prediction-accuracy qualification, or
deployment-benefit claim is authorized by these diagnostics.
