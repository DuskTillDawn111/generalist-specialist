# Original twoFIFO owned training collector v3

v2 native captures are on `COLLECTOR_MEASUREMENT_CHECKPOINT_CONTAMINATION_HOLD`.
The frozen v2 source, capture, controller, reviews and macros are retained as
observational evidence. A native protocol audit does not qualify their response,
notification or energy distributions for production training. The full growing
raw was copied while holding the runtime lock immediately after the last arrival,
inside the ten-second measurement. Copy/serialization competes with sender,
collector and monitor execution; exact attribution of each observed delay remains
unidentified because v2 did not record checkpoint spans.

v3 is a separate source closure and collection identity. Its campaign/raw schemas
are `gsdeploy.original_twoFIFO_owned_training_campaign.v3` and
`gsdeploy.original_twoFIFO_owned_training_capture.v3`. It preserves all original
150 cases, 546 offered measured requests, 216 priming requests across twelve
startups, 13/13/13/11 case shards, fixed ten-second measurement plus sixty-second
drain, the exact original runtime/ownership monitor, and original supervisor
budget/disk gate. There are no caps, omitted offers, retries or new GPU leases.

Before startup, a durable raw contains all measured offered rows, complete public
case calendars and the full ordered public priming catalog. The eighteen original
priming operations run in order without full checkpoints between requests. After
priming, a full snapshot is durable. Before each case epoch, a detached prior
history snapshot includes actual priming and earlier measured rows, queued
runtime events, pending requests, READY and errors. The case preparation records
its exact public calendar, history index and returned durable checkpoint time.
Only after persistence returns and a new clock tick is reached is the case epoch
chosen. All arrivals then enqueue independently. There is no history/full-raw
copy or checkpoint from epoch through fixed drain deadline. Early completion
still waits to that deadline. The postcase snapshot begins strictly afterward.

`checkpoint-spans.json` is a canonical independent sidecar. Raw binds its fixed
filename, schema, execution identity and `time.perf_counter_ns` domain. The sidecar
records complete nested spans for checkpoint total, locked copy, serialization
and atomic fsync/replace, plus precase history snapshots; each has ordinal,
case/phase/stage and start/end. This avoids pretending that a raw snapshot can
contain the final time of its own write. Sidecar persistence itself returns before
choosing an epoch. The collector audits all recorded spans against closed case
intervals, and native review must independently authenticate and validate the
sidecar, public calendars, pre-epoch preparations and immutable histories. An
interrupted capture still follows original cleanup and saves failed evidence;
if its emergency checkpoint lies inside an unfinished case, it explicitly sets
`no_mutating_checkpoint_during_case=False` and cannot qualify.

CPU tests cover a 500 ms injected delay on every atomic save, full calendars and
identities before startup, all arrival offsets/history preservation, every
checkpoint outside both protected phases, real concurrent monitor/notification
threads while checkpoints hold both locks for 100 ms, fixed drain despite early
completion, closed-interval overlap rejection, injected CPU provenance, and the
new source freeze with unchanged public shards/budget. These are implementation
checks, not native timing qualification. The ownership monitor retains its actual
query cost and Python scheduling effects; v3 removes this identified checkpoint
intervention without claiming all observer overhead disappears. A new campaign
needs independent v3 review/macros/intake, and v2 evidence must never be relabeled
as v3 or mixed into its fitted response/transport/resource law.
