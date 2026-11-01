# Fixed campaign continuation v1

This small driver calls the original owned collector execute, new native review
and exact v2 macro extraction in the twelve frozen plan entries' order. It adds
no GPU lease, worker controller, retry, window extension or response model.

`prepare(campaign_path, output)` audits the existing campaign on CPU and writes
an exclusive manifest with all twelve plan receipts and its implementation
source closure. `run(manifest_path, output, execute=False)` defaults to reading
existing completed evidence. A new collection requires `--execute`. Each run
uses a new output directory, append-only timestamped event log and final report.
A campaign-level nonblocking file lock prevents concurrent drivers; the original
supervisor still owns the GPU experiment lease and cleanup.

An existing collection without `execution.json` returns WAITING without reading
its partial raw or launching another process. An existing final execution must
have exact plan/execution identity, original budget, successful watchdog exit,
clean descendants and final source audit. The driver redoes native review and
v2 macro extraction, saving its checks below its own run directory. Existing
`native-review01/report.json` and `macros-v2.json` are rederived and compared;
they are never overwritten. Every completed attempt must pass these checks
before the next attempt can start.

Failed execution, NOT_ELIGIBLE review, changed sources, different old macros,
missing final receipt after execute, or insufficient original 18 GiB filesystem
preflight stops the run. The unchanged original execute performs its own final
prelaunch filesystem gate, live lease and cleanup. A launch error is never
retried. A later driver invocation may inspect evidence already present; an
existing unfinished attempt still remains WAITING, including a previous crash.
