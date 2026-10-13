# Module 3B search trace

Predictor=None；全部候选来自 Module 3A one-hop + LP Top-B。

## Iteration 0

Current: qwen
Runtime: recoverable_congested, mean=12997.344 ms, P95=27894.682 ms

LP z=0.312392558; bottleneck=['qwen']; pressure tasks=['asr', 'lid', 'image', 'ocr', 'text', 'visual_qa']

| Mutation | z | delta_z | LP rank | Decision | Reason |
|---|---:|---:|---:|---|---|
| ADD distilbert | 0.277 | 0.036 | 4 | PRUNED_TOP_B |  |
| ADD paddleocr | 0.275 | 0.038 | 3 | PRUNED_TOP_B |  |
| ADD resnet50 | 0.257 | 0.056 | 2 | ROLLBACK | another_top_b_candidate_preferred |
| ADD whisper | 0.158 | 0.154 | 1 | ACCEPTED | stability_improved |

Top-B: qwen+whisper; qwen+resnet50

Accepted: qwen+whisper

- qwen+whisper: mean 119.293 ± 21.060 ms; P95 215.306 ms; steady; ACCEPTED (stability_improved).
- qwen+resnet50: mean 235.797 ± 37.969 ms; P95 628.907 ms; steady; ROLLBACK (another_top_b_candidate_preferred).

完整 routing、pressure、contribution、三次 raw paths 均保存在 search_trace.json。

## Iteration 1

Current: qwen+whisper
Runtime: steady, mean=119.293 ms, P95=215.306 ms

LP z=0.158317016; bottleneck=['qwen']; pressure tasks=['image', 'ocr', 'text', 'visual_qa']

| Mutation | z | delta_z | LP rank | Decision | Reason |
|---|---:|---:|---:|---|---|
| ADD distilbert | 0.123 | 0.036 | 3 | PRUNED_TOP_B |  |
| ADD paddleocr | 0.120 | 0.038 | 2 | ROLLBACK | another_top_b_candidate_preferred |
| ADD resnet50 | 0.102 | 0.056 | 1 | ACCEPTED | mean_improved |
| REMOVE whisper | — | unavailable | None | SKIP_TABU | already visited or rejected |

Top-B: qwen+resnet50+whisper; paddleocr+qwen+whisper

Accepted: qwen+resnet50+whisper

- qwen+resnet50+whisper: mean 84.679 ± 7.744 ms; P95 149.793 ms; steady; ACCEPTED (mean_improved).
- paddleocr+qwen+whisper: mean 91.627 ± 17.276 ms; P95 154.673 ms; steady; ROLLBACK (another_top_b_candidate_preferred).

完整 routing、pressure、contribution、三次 raw paths 均保存在 search_trace.json。
