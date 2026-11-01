# Independent review of v11 node-rate consumer

Reviewed the four v11 consumer modules independently of their author. All
checks use explicitly declared synthetic CPU laws or supplied synthetic kernel
coefficients. No optimizer, native collection, GPU operation or validation
endpoint access is part of this review.

The initial adapter accepted six foreign or unresolved inputs after their
checksums were recomputed: a validation response bundle; a catalog not marked
training-only; a catalog marked as having read validation labels; a validation
profile; an unresolved profile; and a cyclic profile. Each still returned
numeric node rates. This was a source-role gate defect, independent of the
response formula. The exact old four source files were copied before repair;
independent tests execute the preserved adapter to retain all six counterexamples.

The author repaired the adapter to reject these roles before copying the
catalog, computing hashes or reading numerical payloads. The repair also
checks unique typed nodes and supported node kinds. The current independent
suite checks all six rejections and uses a poison label object to verify that
foreign label bodies are not copied or hashed first. The original compiler's
training-role gate remains unchanged. The CPU02 source is the reviewed target;
the CPU01 author evidence alone did not establish this property.

Independent checks exercise same-owner concurrent nodes, completion and future
arrival repricing using remaining reference work, three owners executing
concurrently, FIFO head-of-line blocking by a future host START, nonnegative
rates including values above one, rejection of incomplete/nonfinite/bool rates,
explicit copy/memset rate requirements, and complete checkpoint continuation.
Every atomic prefix resumes to the same result with its complete future cohort
and provider state. Re-signing a work ledger while conserving total work does
not bypass deterministic prefix replay; forged provider history is also rejected.
A real KernelDagNodeClockAdapter feeds the scheduler and event engine, with
same-owner multistream completion and JSON pause/resume equality. Training
observed future END, duration and release fields can change arbitrarily without
becoming numeric reference clocks in this adapter.

The response rate is fitted reference-work progress, not physical SM capacity.
The consumer permits finite values above one and imposes no global admission
cap of one or two requests. Copy enablement alone gives neither a duration nor
a service rate. Missing fitted/static/lifecycle/copy support remains UNKNOWN.

The report records exact source receipts before and after the test run, the
CPU02 author report, preserved CPU01 source receipts, test log and all limits.
Passing these checks qualifies the CPU consumer contract only: it supplies no
measured interference accuracy, native profiler authentication, OCR clock
support, compute/DRAM complementarity, cache or shared-weight execution,
batching implementation, independent predictor validation, or routing gain.
