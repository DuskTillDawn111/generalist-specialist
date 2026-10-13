# Module 4B Routing Search Trace

Termination: NO_RUNTIME_IMPROVEMENT

## Iteration 1

Current routing: `{"asr::distilbert": 0.0, "asr::paddleocr": 0.0, "asr::qwen": -0.0, "asr::resnet50": 0.0, "asr::whisper": 0.15, "image::distilbert": 0.0, "image::paddleocr": 0.0, "image::qwen": -0.0, "image::resnet50": 0.4, "image::whisper": 0.0, "lid::distilbert": 0.0, "lid::paddleocr": 0.0, "lid::qwen": -0.0, "lid::resnet50": 0.0, "lid::whisper": 0.4, "ocr::distilbert": 0.0, "ocr::paddleocr": 0.4, "ocr::qwen": 0.0, "ocr::resnet50": 0.0, "ocr::whisper": 0.0, "text::distilbert": 0.0, "text::paddleocr": 0.0, "text::qwen": 0.8, "text::resnet50": 0.0, "text::whisper": 0.0, "visual_qa::distilbert": 0.0, "visual_qa::paddleocr": 0.0, "visual_qa::qwen": 0.2, "visual_qa::resnet50": 0.0, "visual_qa::whisper": 0.0}`
Pressures: `{"paddleocr": 0.00242700996, "qwen": 0.06461246086888889, "resnet50": 0.0021991287155555556, "whisper": 0.03935853461944445}`; z=0.064612461

| LP rank | Task | Source → destination | Delta rps | Candidate z | Delta z | Top-B | Runtime mean / P95 ms | Decision / reason |
|---:|---|---|---:|---:|---:|---|---|---|
| 1 | ocr | paddleocr → qwen | 0.040000 | 0.068398389 | -0.003785928 | True | 61.408380 / 113.932222 | ROLLBACK: mean_worse |
| 2 | image | resnet50 → qwen | 0.040000 | 0.070196988 | -0.005584528 | True | 63.166133 / 119.931572 | ROLLBACK: mean_worse |
| 3 | lid | whisper → qwen | 0.040000 | 0.070208838 | -0.005596377 | False | not evaluated | PRUNED_TOP_B:  |
| 4 | asr | whisper → qwen | 0.015000 | 0.074423638 | -0.009811177 | False | not evaluated | PRUNED_TOP_B:  |

Accepted: None; analytical_runtime_mismatch=False

- image resnet50 → qwen candidate pressure: `{"paddleocr": 0.00242700996, "qwen": 0.07019698844977779, "resnet50": 0.001979215844, "whisper": 0.03935853461944445}`
- ocr paddleocr → qwen candidate pressure: `{"paddleocr": 0.002184308964, "qwen": 0.06839838881022223, "resnet50": 0.0021991287155555556, "whisper": 0.03935853461944445}`
- asr whisper → qwen candidate pressure: `{"paddleocr": 0.00242700996, "qwen": 0.07442363772738889, "resnet50": 0.0021991287155555556, "whisper": 0.03650604655261112}`
- lid whisper → qwen candidate pressure: `{"paddleocr": 0.00242700996, "qwen": 0.07020883824622223, "resnet50": 0.0021991287155555556, "whisper": 0.038275169224333334}`
