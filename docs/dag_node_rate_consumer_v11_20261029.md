# Node-keyed dynamic DAG consumer v11 — CPU verification

The new `DagNodeRateCheckpointModel` is an explicit derivative of the retained
v10 checkpoint engine. It consumes `CheckpointNodeScheduler` and an actual
`KernelDagNodeClockAdapter`; it does not stop at a detached feature calculator.
The original v8/v10/serial predictor files are retained without edits.

## Node response and queue propagation

Each active GPU node has its own `(owner,node_id)` key and remaining reference
work. All other active nodes, including other streams of the same owner, enter
the unchanged six-column `competition_features` mapping used by the new DAG
training compiler. A separately supplied qualified law gives the node rate
`exp(-beta*x)`. This is progress in the fitted reference domain, not an SM share;
finite values above one are supported. Static occupancy constraints supply
covariates only and never an inverse-occupancy speed estimate.

The event engine preserves full public future arrivals, owner FIFO queues,
supplied request START bounds, DAG releases/dependencies, completion, parent
receipt and worker-ready events. A competitor arrival or completion changes the
active node set and immediately recomputes rates for all remaining work. A
pending host START is also a valid future transition when current rates are zero.
Same-clock closures run before the next positive service interval. Consumption
is clamped to remaining work under integer completion rounding; each node keeps
`reference = consumed + remaining` accounting. The supplied measurement and
fixed drain remain fixed, including unfinished request mass.

There is no global one-request or two-request admission limit in this consumer.
The existing executor's FIFO per owner remains an explicit execution assumption.
Copy/memset nodes require separate explicit reference durations and rate/next
transition laws; enabling independent copy alone supplies no numbers. Unknown
host/lifecycle, static allocation, node reference, or numerical-fit support stays
UNKNOWN rather than receiving a guessed default.

## Model binding and continuation

The adapter consumes the new `gsdeploy.kernel_dag_response_training.v1` bundle
directly. It binds the full training catalog, execution context, profiler, full
hardware definition, exact owner/profile/node reference groups, feature schema,
response domain and unchanged source receipts. It neither calls nor relaxes the
old serial compiler. A compiler output with failed structural rank can be loaded
but cannot produce a numeric rate or reference duration.

Catalog projection retains static launch fields only. Observed training START,
END, release and duration do not become prospective numeric node clocks.
Reference means or quantiles must be declared from the fitted lognormal law.
The cpu02 role correction requires an explicit training bundle split, explicit
training-only catalog with `validation_labels_read=False`, training profiles,
resolved acyclic flags and unique typed nodes before copying or hashing any
body. Legacy catalogs missing the explicit unread-label declaration are rejected
by this consumer; their original sealed files are not edited or silently
assigned a new role. The cpu01 sources remain snapshotted with their old report.
All static model tables are recursively immutable, with hashes computed once.
Per-event configuration checks do not rehash the complete catalog. Optional
non-GPU callbacks retain code/source/configuration identity checks.

The v11 checkpoint carries dynamic node work and scheduler/provider state,
while immutable model identities remain outside that state. Restoring validates
sources, contract, structural/lifecycle consistency and work conservation, then
replays the complete event prefix before committing restored state. Altering a
checksum alone cannot authorize changed consumed work or provider history.
The checkpoint is simulated model state, not a recovered native GPU cursor.

## CPU evidence and limits

Reproduce the author check with:

```bash
python artifacts/predictor-node-rate-dag-v11-20261029-cpu02/review.py
```

The suite exercises the consumer/adapter together using declared numerical CPU
stubs, checks the real compiler's rank-failed fixture bundle remains UNKNOWN,
and retains the related v10/v8/feature/response regression tests. It does not fit
measured data or access validation target endpoints.

This supplies a node-keyed executable forecasting consumer and its continuation
contract. It does not replace the separately frozen public prospective wrapper,
certify real OCR or cross-process clock support, authenticate native intake,
validate interference accuracy, estimate compute/DRAM complementarity, implement
joint batching or shared-weight execution, or claim a six-objective routing gain.
Those require the root workflow's independent evidence and integration gates.
