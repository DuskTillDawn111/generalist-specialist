# Whole-attempt nominal resource structure sidecar

The new downstream-only file
`tools/predictor_validation_20261027/twoFIFO_static_resource_sidecar_v1.py`
connects the existing seven-projection structure to the new complete-attempt
macros. It changes neither collector nor response rates.

The public entry is:

```python
extract(training_review, static_inventory)
```

It invokes the real `twoFIFO_composable_macros_v2.extract(training_review)` and
accepts its `gsdeploy.twoFIFO_composable_response_macro_training.v2` output.
`checked_static_inventory()` verifies and reconstructs existing source/header
bindings. Worker context and READY spec hashes must match those bindings.
`attempt_events(raw)` indexes the whole attempt, including priming and earlier
cases, rather than resetting history at each case. Priming creates no labels.
The result schema is `gsdeploy.twoFIFO_static_resource_structure_sidecar.v1`.
All macro, raw/review, static binding and local implementation source receipts
are included and checked again before return.

The adapter reuses `features_from_visible_begin`, `descriptor` and `pair_state`.
It does not reimplement seven-projection arithmetic, fit new parameters or attach
a complementarity discount. Each piece describes its exact current macro BEGIN
only after parent processing. A missing shape stays unknown; macro shape fields,
later decode shapes, future END/query outcomes and completed exposure never fill
it. Every known BEGIN descriptor includes the existing complete
`nominal_projection_features`, including base nominal FLOPs/logical operand bytes,
LoRA accounting and its scope exclusions. Entry, OCR and READY tails remain in
the complete denominator with explicit unknown support.

Peer N/K/U accounting and pair identities are retained. All-peer aggregates stay
`None` if any peer pair is unknown. A known-pair sum of zero with K=0 describes
an empty known set, not zero physical demand. Nominal logical operand bytes are
explicitly distinguished from measured DRAM or L2 traffic.

Attention/KV length, attention FLOPs, KV bytes, measured DRAM/L2 bytes, achieved
compute/DRAM throughput, SM service, quantized runtime storage/scales, GPU memory
and cache gain remain `None`; physical classification stays `unknown`. This
sidecar does not qualify future shape templates, physical resource identity,
cache benefit, the native provider, prediction accuracy or routing gain.

`enrich(macros, events, bindings, owner_families, capture_sha256=...)` is a pure
CPU transformation for focused tests and explicit diagnostic use. It deliberately
does not authenticate native sources. Production extraction must use `extract`.
The collector has no reverse import of the new file, as checked from its complete
static source closure; the current collector freeze is unaffected.

Eight CPU counterexamples use the actual new macro adapter's request partition:
full attempt/priming conservation; future own/peer shape mutation cannot change
earlier pieces; missing current shape cannot be filled by later decode;
unsupported peers preserve N=K+U and unknown aggregates; schema/whole-attempt/
denominator rejection; real native extraction entry cannot silently use a pure
fixture; no collector reverse dependency; zero peer exposure has no division or
resource classification. Three synthetic sidecar artifacts retain 3 measured
requests, 18 priming history rows and 12 macro labels. Native extraction has not
been run because this packet is CPU-only and claims no native training evidence.

CPU report:
`artifacts/predictor-twoFIFO-static-resource-sidecar-v1-20261028-cpu01/report.json`.
