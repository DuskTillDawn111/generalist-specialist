# ChatQA public-length kernel template candidates

The CPU model emits complete numeric 3,829-kernel **prefill** candidates for new
integer input lengths between 414 and 4,514, batch one. These are explicit
hypotheses; the ordered topology, algorithm dispatch, new resource settings,
natural timing and complete request behavior have not been validated at a new
length. It does not predict the number of generated tokens or host gaps.

The final source-bound model is
`artifacts/predictor-chatqa-template-grid-model-v3-20261028-final02/report.json`
(SHA256 `6a0531370481734f8fc703616606af325d804287a23e5f1465a5f10e69e100e8`).
It contains 56 exact endpoint patterns and 15 numeric candidate files: lengths
512, 963, 1025, 2049 and 4096, each under `nearest_anchor`, `lower_domain` and
`upper_domain` dispatch hypotheses. Every observed field is reproduced exactly
at both actual training anchors, 414 and 4514.

The measured endpoint comparison contains 3,153 grid-only changes, 67 identical
signatures, 608 changes to the kernel algorithm/name and associated resources,
and one bool-reduction resource change. The model preserves separate endpoint
algorithms, register counts, shared memory, block sizes and driver attributes;
it never interpolates an algorithm name or static resource allocation.

| Rule or hypothesis | Endpoint evidence | Candidate at M=963 |
| --- | --- | --- |
| Torch elementwise `ceil(N / block_work)`, conditional `N=M*width` | 2,319 slots; installed headers | E.g. width4096, work512: grid.x7704 |
| bitsandbytes vector quantization `grid.x=rows=M` | 224 slots; installed Python and upstream0.45.5 source | grid.x963 |
| bitsandbytes dequantization `ceil(M*out_features/2048)` | 224 slots;512threads,4items/thread | out_features1024: grid.x482 |
| Bool reduction `min(512,floor_pow2(floor(M/4)))` | One slot; installed Reduce.cuh | block.x128, dynamic shared128bytes |
| Stable CUTLASS `8*ceil(M/64)` | 64 slots; two-shape hypothesis | grid.x128 |
| Flash attention `ceil(M/64)` | 32 slots; two-shape hypothesis | grid.x16 |
| Mean reduction `ceil(M/16)` | 65 slots; two-shape hypothesis | grid.x61 |
| Separate CUTLASS dispatch variants | Different endpoint tile/resources; threshold unknown | Short64tile x128; long128tile x64 |
| Separate Ampere GEMM variants | Different endpoint split-K/tile; threshold unknown | Short32tile y31/split-K16; long128tile y8/split-K7 |

There are 453 stable-grid slots whose launch arithmetic remains an empirical
hypothesis, including the CUTLASS/Flash/reduction cases above and constant grids.
The model records source receipts and 21 exact source-line citations for the
remaining arithmetic. An upstream version-tag source is not proof of exact
compiled-binary equivalence. Installed wheel headers are also not an independent
attestation of the loaded binary build. Only 896 slots combine the cited launch
arithmetic with an enclosing affine shape-width binding; all other
public-length-to-workload relations retain an explicit condition.

Both measured anchors are even. At the selected odd length, 2,565 stable-grid
slots have alternative integer/ceil bases that match both anchors but disagree
at the new point. One authentic odd input is therefore the first useful extra
training point, but cannot identify every algorithm threshold. A broader target
plan is 415,512,513,1024,1025,2048,2049,4096,4513, using authentic eligible
catalog inputs when available. These targets are not claims of existing inputs.

The real selected input is `tydiqa-new-native-31bfedbdfbc81089df511fec` with
**963 actual frozen ChatQA+BOS tokens**, not1025. It is the nearest eligible odd
input to1025 among21 actual catalog candidates. The selector excludes all four
bound prior training IDs, all four bound prior validation IDs, the runtime
prospective text reserve, and exact matching public contents.

`artifacts/predictor-chatqa-new-shape-training-selection-v3-20261028-final02/`
contains the selected public input, exact native request, real tokenizer IDs
hash, full eligible ranking, formatter/tokenizer receipts, and bound validation
reservation. Report SHA256 is
`4dba35c7fe196f0125372a1733ba45b25191070d894850de74c8c21111ee42e3`.
Content SHA256 is `4daeb163347535e4f986290d054c0451c3b3599ee148108732228b4eef83fb28`;
token-ID SHA256 is `744211715a9d2b1e6a68049e4757a41cdf6cafa597a8dc202d0368430eb2fa6c`.
Novelty is relative to the bound training selection, not all project history.

`instantiate_model(report,M,policy)` in `kernel_template_rules_v3.py` returns the
complete numeric sequence. `kernel_template_compare_v4.py` compares it against
a passing receipt-bound v3/v4 actual light capture, including exact SQLite launch
attributes and full order. Changed canonical contexts remain explicit diagnostic
comparisons and never qualify as same-context validation. The planned963-token
capture changes the wrapper/context, so this distinction applies to it.

```bash
python -m tools.predictor_validation_20261027.kernel_template_compare_v4 \
  --model artifacts/predictor-chatqa-template-grid-model-v3-20261028-final02/report.json \
  --light-review ACTUAL_PASSING_LIGHT_REVIEW \
  --selection artifacts/predictor-chatqa-new-shape-training-selection-v3-20261028-final02/report.json \
  --output NEW_COMPARISON_DIRECTORY
```

The selector, template builder and comparator only perform CPU work; they load
tokenizer assets and retained trace/source evidence, and never model weights.
