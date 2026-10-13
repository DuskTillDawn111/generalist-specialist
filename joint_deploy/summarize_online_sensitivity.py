import csv
import json
from pathlib import Path


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

RESULT_DIR = (
    ROOT
    / "benchmark/results/joint/sensitivity"
)


CASES = [
    {
        "tag": "a010_w32",
        "alpha": 0.10,
        "window": 32,
        "path": (
            RESULT_DIR
            / "online_a010_w32.summary.json"
        ),
    },
    {
        "tag": "a025_w08",
        "alpha": 0.25,
        "window": 8,
        "path": (
            RESULT_DIR
            / "online_a025_w08.summary.json"
        ),
    },
    {
        "tag": "a025_w16",
        "alpha": 0.25,
        "window": 16,
        "path": (
            RESULT_DIR
            / "online_a025_w16.summary.json"
        ),
    },

    # Existing main formal experiment
    {
        "tag": "a025_w32",
        "alpha": 0.25,
        "window": 32,
        "path": (
            ROOT
            / "benchmark/results/joint/"
            "online_adaptive_formal_800.summary.json"
        ),
    },

    {
        "tag": "a025_w64",
        "alpha": 0.25,
        "window": 64,
        "path": (
            RESULT_DIR
            / "online_a025_w64.summary.json"
        ),
    },
    {
        "tag": "a050_w32",
        "alpha": 0.50,
        "window": 32,
        "path": (
            RESULT_DIR
            / "online_a050_w32.summary.json"
        ),
    },
]


rows = []


for case in CASES:

    path = case[
        "path"
    ]

    if not path.exists():
        print(
            "Missing:",
            path
        )
        continue

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        s = json.load(
            f
        )

    asr = s[
        "per_task"
    ][
        "asr"
    ]

    text = s[
        "per_task"
    ][
        "text"
    ]

    image = s[
        "per_task"
    ][
        "image"
    ]

    ocr = s[
        "per_task"
    ][
        "ocr"
    ]

    final_est = s[
        "final_service_estimate_ms"
    ]

    routing = s[
        "routing_by_task"
    ][
        "asr"
    ]

    row = {
        "tag":
            case["tag"],

        "alpha":
            case["alpha"],

        "window":
            case["window"],

        "wall_time_s":
            s[
                "wall_time_s"
            ],

        "throughput":
            s[
                "throughput_tasks_per_s"
            ],

        "text_accuracy":
            text[
                "accuracy"
            ],

        "image_accuracy":
            image[
                "accuracy"
            ],

        "ocr_accuracy":
            ocr[
                "accuracy"
            ],

        "asr_wer":
            asr[
                "corpus_wer"
            ],

        "asr_whisper":
            routing.get(
                "whisper",
                0
            ),

        "asr_phi4":
            routing.get(
                "phi4",
                0
            ),

        "asr_worker_wait_mean_ms":
            asr[
                "worker_queue_wait_mean_ms"
            ],

        "asr_worker_wait_p95_ms":
            asr[
                "worker_queue_wait_p95_ms"
            ],

        "asr_response_mean_ms":
            asr[
                "batch_response_time_mean_ms"
            ],

        "asr_response_p95_ms":
            asr[
                "batch_response_time_p95_ms"
            ],

        "final_whisper_asr_est_ms":
            final_est[
                "whisper"
            ][
                "asr"
            ],

        "final_phi4_asr_est_ms":
            final_est[
                "phi4"
            ][
                "asr"
            ],
    }

    rows.append(
        row
    )


output = (
    RESULT_DIR
    / "parameter_sensitivity.csv"
)


with open(
    output,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        rows
    )


print()
print(
    "===== PARAMETER SENSITIVITY ====="
)

header = (
    f"{'case':<11}"
    f"{'alpha':>7}"
    f"{'win':>6}"
    f"{'time(s)':>10}"
    f"{'task/s':>10}"
    f"{'W->ASR':>8}"
    f"{'P->ASR':>8}"
    f"{'WER':>10}"
    f"{'ASR wait95':>13}"
)

print(
    header
)

print(
    "-" * len(
        header
    )
)


for row in rows:

    print(
        f"{row['tag']:<11}"
        f"{row['alpha']:>7.2f}"
        f"{row['window']:>6d}"
        f"{row['wall_time_s']:>10.3f}"
        f"{row['throughput']:>10.3f}"
        f"{row['asr_whisper']:>8d}"
        f"{row['asr_phi4']:>8d}"
        f"{row['asr_wer']:>10.4f}"
        f"{row['asr_worker_wait_p95_ms']:>13.1f}"
    )


print()
print(
    "CSV:",
    output
)
