import json
import subprocess
import sys
from pathlib import Path
from statistics import mean


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

PYTHON = Path(
    "/root/miniconda3/envs/gsdeploy/bin/python"
)

PAIR_SCRIPT = (
    ROOT
    / "joint_deploy/run_active_pair_profile.py"
)

POOL = (
    ROOT
    / "benchmark/task_pools/"
    "mixed_formal_1200_6task.jsonl"
)

OUT_DIR = (
    ROOT
    / "benchmark/results/placement/"
    "active_gamma_v2"
)

GAMMA_OUT = (
    ROOT
    / "benchmark/results/placement/"
    "pairwise_interference_gamma_active_v2.json"
)

FAIL_OUT = (
    ROOT
    / "benchmark/results/placement/"
    "pairwise_interference_gamma_active_v2_failures.json"
)


# ============================================================
# Target service paths
# 11 service paths × 4 other resident models = 44 experiments
# ============================================================

TARGET_PATHS = [
    ("qwen", "text"),
    ("qwen", "image"),
    ("qwen", "ocr"),
    ("qwen", "asr"),
    ("qwen", "visual_qa"),
    ("qwen", "lid"),

    ("distilbert", "text"),
    ("resnet50", "image"),
    ("paddleocr", "ocr"),

    ("whisper", "asr"),
    ("whisper", "lid"),
]

ALL_MODELS = [
    "qwen",
    "distilbert",
    "resnet50",
    "paddleocr",
    "whisper",
]


# ============================================================
# Canonical co-resident saturation workload
# ============================================================

CO_TASKS = {
    "qwen": [
        "text",
        "image",
        "ocr",
        "asr",
        "visual_qa",
        "lid",
    ],

    "distilbert": [
        "text",
    ],

    "resnet50": [
        "image",
    ],

    "paddleocr": [
        "ocr",
    ],

    "whisper": [
        "asr",
        "lid",
    ],
}


# ============================================================
# Same-sample standalone references
# ============================================================

REFERENCE_FILES = {
    ("qwen", "text"):
        ROOT
        / "benchmark/results/joint/"
        "pure_qwen_formal_1200_6task.jsonl",

    ("qwen", "image"):
        ROOT
        / "benchmark/results/joint/"
        "pure_qwen_formal_1200_6task.jsonl",

    ("qwen", "ocr"):
        ROOT
        / "benchmark/results/joint/"
        "pure_qwen_formal_1200_6task.jsonl",

    ("qwen", "asr"):
        ROOT
        / "benchmark/results/joint/"
        "pure_qwen_formal_1200_6task.jsonl",

    ("qwen", "visual_qa"):
        ROOT
        / "benchmark/results/joint/"
        "pure_qwen_formal_1200_6task.jsonl",

    ("qwen", "lid"):
        ROOT
        / "benchmark/results/joint/"
        "pure_qwen_formal_1200_6task.jsonl",

    ("distilbert", "text"):
        ROOT
        / "benchmark/results/placement/"
        "single_text_specialist_formal_200.jsonl",

    ("resnet50", "image"):
        ROOT
        / "benchmark/results/placement/"
        "single_image_specialist_formal_200.jsonl",

    ("paddleocr", "ocr"):
        ROOT
        / "benchmark/results/placement/"
        "single_ocr_specialist_formal_200.jsonl",

    ("whisper", "asr"):
        ROOT
        / "benchmark/results/placement/"
        "single_asr_specialist_formal_200.jsonl",

    ("whisper", "lid"):
        ROOT
        / "benchmark/results/placement/"
        "single_lid_whisper_formal_200_6task.jsonl",
}


TARGET_COUNT = 30
CO_BACKLOG = 64
MIN_OVERLAP = 0.90


def read_jsonl(path):

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def save_json(path, obj):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def reference_map(
    worker,
    task,
):
    path = REFERENCE_FILES[
        (worker, task)
    ]

    if not path.exists():
        raise FileNotFoundError(
            f"Missing standalone reference: "
            f"{path}"
        )

    rows = read_jsonl(
        path
    )

    result = {}

    for row in rows:

        if (
            row.get("task_type")
            != task
        ):
            continue

        actual_worker = (
            row.get(
                "assigned_worker"
            )
            or row.get(
                "worker"
            )
        )

        if actual_worker != worker:
            continue

        result[
            row["pool_id"]
        ] = row

    if not result:
        raise RuntimeError(
            f"No standalone rows for "
            f"{worker}/{task}"
        )

    return result


def canonical_output(
    worker,
    task,
    co,
):
    return (
        OUT_DIR
        / (
            f"active_{worker}_{task}"
            f"_with_{co}_n{TARGET_COUNT}.jsonl"
        )
    )


def legacy_existing_output(
    worker,
    task,
    co,
):
    # Already completed before the batch runner
    if (
        worker == "qwen"
        and task == "visual_qa"
        and co == "resnet50"
    ):
        p = (
            ROOT
            / "benchmark/results/placement/"
            "active_qwen_visualqa_resnet50_n30.jsonl"
        )

        if p.exists():
            return p

    return None


def valid_existing_result(path):

    summary_path = Path(
        str(path)
        + ".summary.json"
    )

    if (
        not path.exists()
        or not summary_path.exists()
    ):
        return False

    try:

        summary = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

        return bool(
            summary.get(
                "valid_for_gamma"
            )
        )

    except Exception:
        return False


def run_experiment(
    worker,
    task,
    co,
    output,
):

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = Path(
        str(output)
        + ".log"
    )

    command = [
        str(PYTHON),
        str(PAIR_SCRIPT),

        "--pool",
        str(POOL),

        "--target-worker",
        worker,

        "--target-task",
        task,

        "--co-worker",
        co,

        "--co-tasks",
        ",".join(
            CO_TASKS[co]
        ),

        "--target-count",
        str(TARGET_COUNT),

        "--co-backlog",
        str(CO_BACKLOG),

        "--min-overlap",
        str(MIN_OVERLAP),

        "--output",
        str(output),

        "--timeout-s",
        "3600",
    ]

    print()
    print("=" * 76)

    print(
        f"RUN: {worker}/{task} "
        f"<- {co}"
    )

    print(
        "co workload:",
        CO_TASKS[co],
    )

    print(
        "output:",
        output,
    )

    print("=" * 76)

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log:

        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert (
            process.stdout
            is not None
        )

        for line in process.stdout:

            print(
                line,
                end="",
                flush=True,
            )

            log.write(
                line
            )

            log.flush()

        return_code = (
            process.wait()
        )

    if return_code != 0:
        raise RuntimeError(
            f"Experiment failed with "
            f"return code {return_code}. "
            f"See {log_path}"
        )


def calculate_gamma(
    worker,
    task,
    co,
    pair_path,
):

    summary_path = Path(
        str(pair_path)
        + ".summary.json"
    )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    if not summary.get(
        "valid_for_gamma"
    ):
        raise RuntimeError(
            "Pair result is not valid "
            "for gamma."
        )

    pair_rows = read_jsonl(
        pair_path
    )

    targets = [
        row
        for row in pair_rows
        if row.get(
            "profile_role"
        ) == "target"
    ]

    if len(targets) != TARGET_COUNT:
        raise RuntimeError(
            f"Expected {TARGET_COUNT} "
            f"target rows, got "
            f"{len(targets)}."
        )

    refs = reference_map(
        worker,
        task,
    )

    matched = []

    for target in targets:

        pool_id = target[
            "pool_id"
        ]

        # active profiler format:
        # target:<index>:<original_pool_id>
        parts = pool_id.split(
            ":",
            2,
        )

        if len(parts) != 3:
            raise RuntimeError(
                f"Bad target pool_id: "
                f"{pool_id}"
            )

        original_pool_id = (
            parts[2]
        )

        single = refs.get(
            original_pool_id
        )

        if single is None:
            raise RuntimeError(
                "Missing same-sample "
                "standalone row: "
                f"{original_pool_id}"
            )

        overlap = float(
            target[
                "active_overlap_ratio"
            ]
        )

        if overlap < MIN_OVERLAP:
            raise RuntimeError(
                f"Target overlap below "
                f"threshold: "
                f"{original_pool_id} "
                f"{overlap:.6f}"
            )

        matched.append({
            "pool_id":
                original_pool_id,

            "single_ms":
                float(
                    single[
                        "worker_latency_ms"
                    ]
                ),

            "pair_ms":
                float(
                    target[
                        "worker_latency_ms"
                    ]
                ),

            "overlap":
                overlap,
        })

    single_values = [
        x["single_ms"]
        for x in matched
    ]

    pair_values = [
        x["pair_ms"]
        for x in matched
    ]

    overlaps = [
        x["overlap"]
        for x in matched
    ]

    single_ms = mean(
        single_values
    )

    pair_ms = mean(
        pair_values
    )

    slowdown = (
        pair_ms
        / single_ms
    )

    gamma = (
        slowdown
        - 1.0
    )

    return {
        "worker":
            worker,

        "task":
            task,

        "co_resident":
            co,

        "single_ms":
            single_ms,

        "pair_ms":
            pair_ms,

        "gamma":
            gamma,

        "slowdown":
            slowdown,

        "source":
            pair_path.name,

        "protocol":
            "active_pair_profile_v1",

        "target_count":
            len(matched),

        "accepted_target_count":
            summary[
                "accepted_target_count"
            ],

        "accepted_fraction":
            summary[
                "accepted_fraction"
            ],

        "weighted_overlap_ratio":
            summary[
                "weighted_overlap_ratio"
            ],

        "minimum_sample_overlap":
            min(overlaps),

        "mean_sample_overlap":
            mean(overlaps),

        "co_resident_tasks":
            summary[
                "co_resident_tasks"
            ],

        "same_sample_reference":
            str(
                REFERENCE_FILES[
                    (worker, task)
                ].relative_to(
                    ROOT
                )
            ),
    }


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Verify references before burning GPU.
    print(
        "===== VERIFY STANDALONE REFERENCES ====="
    )

    for path_key in TARGET_PATHS:

        refs = reference_map(
            *path_key
        )

        print(
            f"{path_key[0]:12s} "
            f"{path_key[1]:10s} "
            f"{len(refs):3d} samples"
        )

    experiments = []

    for worker, task in TARGET_PATHS:

        for co in ALL_MODELS:

            if co == worker:
                continue

            experiments.append(
                (
                    worker,
                    task,
                    co,
                )
            )

    if len(experiments) != 44:
        raise RuntimeError(
            f"Expected 44 experiments, "
            f"got {len(experiments)}."
        )

    print()
    print(
        "Total experiments:",
        len(experiments),
    )

    rows = []
    failures = []

    for index, (
        worker,
        task,
        co,
    ) in enumerate(
        experiments,
        start=1,
    ):

        print()
        print(
            f"[{index:02d}/44] "
            f"{worker}/{task} "
            f"<- {co}"
        )

        output = canonical_output(
            worker,
            task,
            co,
        )

        existing = (
            output
            if valid_existing_result(
                output
            )
            else None
        )

        if existing is None:

            legacy = (
                legacy_existing_output(
                    worker,
                    task,
                    co,
                )
            )

            if (
                legacy is not None
                and valid_existing_result(
                    legacy
                )
            ):
                existing = legacy

        try:

            if existing is not None:

                pair_path = (
                    existing
                )

                print(
                    "SKIP GPU RUN: "
                    "valid existing result"
                )

            else:

                pair_path = (
                    output
                )

                run_experiment(
                    worker,
                    task,
                    co,
                    pair_path,
                )

            row = calculate_gamma(
                worker,
                task,
                co,
                pair_path,
            )

            # Replace any duplicate row
            # from a previous resume pass.
            rows = [
                old
                for old in rows
                if not (
                    old["worker"]
                    == worker
                    and old["task"]
                    == task
                    and old[
                        "co_resident"
                    ] == co
                )
            ]

            rows.append(
                row
            )

            rows.sort(
                key=lambda x: (
                    x["worker"],
                    x["task"],
                    x["co_resident"],
                )
            )

            save_json(
                GAMMA_OUT,
                rows,
            )

            print(
                "RESULT:",
                f"single="
                f"{row['single_ms']:.3f} ms,",
                f"pair="
                f"{row['pair_ms']:.3f} ms,",
                f"slowdown="
                f"{row['slowdown']:.4f},",
                f"gamma="
                f"{row['gamma']:.4f},",
                f"overlap="
                f"{row['weighted_overlap_ratio']:.4f}",
            )

        except Exception as e:

            failure = {
                "worker":
                    worker,

                "task":
                    task,

                "co_resident":
                    co,

                "error":
                    repr(e),
            }

            failures.append(
                failure
            )

            save_json(
                FAIL_OUT,
                failures,
            )

            print(
                "FAILED:",
                repr(e),
            )

            # Continue so one problematic
            # pair does not waste all of the
            # already-completed GPU runs.
            continue

    print()
    print("=" * 76)
    print(
        "===== ACTIVE GAMMA V2 COMPLETE ====="
    )

    print(
        "valid rows =",
        len(rows),
        "/ 44",
    )

    print(
        "failures =",
        len(failures),
    )

    print(
        "gamma file:",
        GAMMA_OUT,
    )

    if failures:
        print(
            "failure file:",
            FAIL_OUT,
        )

    print("=" * 76)


if __name__ == "__main__":
    main()
