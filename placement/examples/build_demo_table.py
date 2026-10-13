"""Build SYNTHETIC interface fixtures, not estimates or research measurements.

The arbitrary numbers intentionally produce different winners for two workloads.
Do not use this fixture builder as an analytical performance model.
"""

from itertools import combinations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPECIALISTS = ("distilbert", "resnet50", "paddleocr", "whisper")
WORKLOADS = {
    "mixed": {"text": 10, "image": 5, "ocr": 3, "asr": 8, "visual_qa": 2, "lid": 1},
    "text_heavy": {"text": 24, "image": 1, "ocr": 1, "asr": 1, "visual_qa": 1, "lid": 1},
}


def main():
    rows = []
    for name, rates in WORKLOADS.items():
        (HERE / f"workload_{name}.json").write_text(
            json.dumps(rates, indent=2) + "\n", encoding="utf-8",
        )
        # Arbitrary fixture penalties/rewards; independent of existing profiles.
        preferred = "whisper" if name == "mixed" else "distilbert"
        for size in range(5):
            for subset in combinations(SPECIALISTS, size):
                mean = 200 + 20 * size - (100 if preferred in subset else 0)
                throughput = 29 if preferred in subset else 20
                rows.append({
                    "model_counts": {model: 1 for model in ("qwen", *subset)},
                    "arrival_rates": rates,
                    "performance": {
                        "mean_latency_ms": mean,
                        "p95_latency_ms": mean * 2,
                        "throughput_rps": throughput,
                        "completed_requests": throughput * 60,
                    },
                })
    data = {
        "schema_version": 1,
        "context": {
            "data_kind": "synthetic",
            "hardware_id": "demo-only-no-hardware-measurement",
            "routing_policy": "specialist-first; qwen fallback; one replica per model",
            "protocol": "synthetic 60-second window; deterministic interface fixture only",
            "latency_definition": "synthetic arrival-to-completion latency, including queue time",
            "note": "All performance numbers are invented. No scientific or deployment conclusion is supported.",
        },
        "rows": rows,
    }
    (HERE / "performance_table.synthetic.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
