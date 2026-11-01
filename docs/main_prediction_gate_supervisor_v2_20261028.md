# Main prediction gate supervisor v2

The independent v1 CPU review reproduced two binding omissions using the real
v5 verifier and original v11 plan: a production receipt could point to a
preparation/review absent from the frozen protocol, and a correctly hashed
protocol with changed thresholds passed the supervisor. Normal v48 production
froze the correct evidence and thresholds, but the supervisor did not enforce
these prerequisites independently. Historical v1 and v48 sources remain frozen.

`main_prediction_gate_supervisor_v2.py` explicitly accepts only new campaign
schema `gsdeploy.projected_handler_artifact_binding_campaign.v55` and independent
review schema `gsdeploy.projected_handler_artifact_binding_independent_review.v55`.
It requires the original thresholds, zero utility tie tolerance, exact
preparation/review receipts and all prepared sources in the protocol. Sealed
cells must equal the reviewed preparation's verified cells, its training index
must be bound, and every protocol candidate must use its exact configuration.
The original v5 bindings plus original-plan preservation continue to enforce
request IDs/input/task identities, context, routes, queue grid and six weights;
the supervisor additionally rejects every incomplete numeric cell.

The execution mechanics are inherited unchanged from v1. Before each candidate
launch it requires 18 GiB free. While the child runs it checks every 0.2 seconds:
free space below 13 GiB, incremental consumption above 6 GiB, or elapsed time
above 2,400 seconds terminates only its launched process group. The timeout is
per candidate and includes waiting for the GPU lock, loading, priming, offered
requests, draining and shutdown. The v11 child alone acquires the GPU lock. An
independent CPU process test confirmed that terminating a launched child and
grandchild preserves a separate unrelated process session.

The new entry's complete `implementation_sources()` closure must be frozen via
v55's additional-execution-source manifest before the prediction protocol is
sealed. Preparation and execution remain distinct CLI subcommands. No GPU work
has been run by this supervisor. A successful collection still requires the
original observed adapter and main service/queue/P95/utility gate evaluation.

Nine v2 CPU tests pass: exact real-v5 fixture compatibility, strict evidence and
threshold refusal, schema/model/request/input/grid/UNKNOWN refusal, resource
boundaries, failure evidence retention, no retry, and source drift refusal.
Synthetic protocols exist only inside temporary test directories and do not
qualify forecasts or authorize GPU collection. Independent review of v2 remains
a separate prerequisite.
