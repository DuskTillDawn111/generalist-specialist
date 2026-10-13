import csv
import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

CHAR = (
    ROOT
    / "benchmark/results/characterization"
)

OUT = (
    CHAR
    / "c1_analysis_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. Service-path input files
# ============================================================

SERVICES = [
    # ---------------- Qwen ----------------
    {
        "path": "qwen/text",
        "model": "qwen",
        "task": "text",
        "nvml": CHAR / "standalone_v1/qwen_text_nvml_n30",
        "nvml_requests": 30,
        "nsys": CHAR / "qwen_nsys_v1/qwen_text_n30.nsys-rep",
        "nsys_requests": 30,
    },
    {
        "path": "qwen/image",
        "model": "qwen",
        "task": "image",
        "nvml": CHAR / "standalone_v1/qwen_image_nvml_n30",
        "nvml_requests": 30,
        "nsys": CHAR / "qwen_nsys_v1/qwen_image_n30.nsys-rep",
        "nsys_requests": 30,
    },
    {
        "path": "qwen/ocr",
        "model": "qwen",
        "task": "ocr",
        "nvml": CHAR / "standalone_v1/qwen_ocr_nvml_n30",
        "nvml_requests": 30,
        "nsys": CHAR / "qwen_nsys_v1/qwen_ocr_n30.nsys-rep",
        "nsys_requests": 30,
    },
    {
        "path": "qwen/asr",
        "model": "qwen",
        "task": "asr",
        "nvml": CHAR / "standalone_v1/qwen_asr_nvml_n30",
        "nvml_requests": 30,
        "nsys": CHAR / "qwen_nsys_v1/qwen_asr_n30.nsys-rep",
        "nsys_requests": 30,
    },
    {
        "path": "qwen/visual_qa",
        "model": "qwen",
        "task": "visual_qa",
        "nvml": CHAR / "standalone_v1/qwen_visual_qa_nvml_n30",
        "nvml_requests": 30,
        "nsys": CHAR / "qwen_nsys_v1/qwen_visual_qa_n30.nsys-rep",
        "nsys_requests": 30,
    },
    {
        "path": "qwen/lid",
        "model": "qwen",
        "task": "lid",
        "nvml": CHAR / "standalone_v1/qwen_lid_nvml_n30",
        "nvml_requests": 30,
        "nsys": CHAR / "qwen_nsys_v1/qwen_lid_n30.nsys-rep",
        "nsys_requests": 30,
    },

    # ---------------- Specialists ----------------
    {
        "path": "distilbert/text",
        "model": "distilbert",
        "task": "text",
        "nvml": CHAR / "specialist_nvml_v2/distilbert_text_nvml_n1000",
        "nvml_requests": 1000,
        "nsys": CHAR / "specialist_nsys_v1/distilbert_text_n30.nsys-rep",
        "nsys_requests": 30,
    },
    {
        "path": "resnet50/image",
        "model": "resnet50",
        "task": "image",
        "nvml": CHAR / "specialist_nvml_v2/resnet50_image_nvml_n1000",
        "nvml_requests": 1000,
        "nsys": CHAR / "specialist_nsys_v1/resnet50_image_n30.nsys-rep",
        "nsys_requests": 30,
    },
    {
        "path": "paddleocr/ocr",
        "model": "paddleocr",
        "task": "ocr",
        "nvml": CHAR / "specialist_nvml_v2/paddleocr_ocr_nvml_n1000",
        "nvml_requests": 1000,
        "nsys": CHAR / "specialist_nsys_v1/paddleocr_ocr_n30.nsys-rep",
        "nsys_requests": 30,
    },
    {
        "path": "whisper/asr",
        "model": "whisper",
        "task": "asr",
        "nvml": CHAR / "specialist_nvml_v1/whisper_asr_nvml_n30",
        "nvml_requests": 30,
        "nsys": CHAR / "specialist_nsys_v1/whisper_asr_n30.nsys-rep",
        "nsys_requests": 30,
    },
    {
        "path": "whisper/lid",
        "model": "whisper",
        "task": "lid",
        "nvml": CHAR / "specialist_nvml_v2/whisper_lid_nvml_n1000",
        "nvml_requests": 1000,
        "nsys": CHAR / "specialist_nsys_v1/whisper_lid_n30.nsys-rep",
        "nsys_requests": 30,
    },
]


# ============================================================
# 2. Kernel taxonomy
# Keep exactly the same taxonomy used in previous analysis.
# ============================================================

CATEGORIES = [
    "GEMV",
    "GEMM",
    "CONV",
    "ATTN",
    "REDNORM",
    "ELEMCOPY",
    "OTHER",
]


def classify_kernel(name):

    n = name.lower()

    if (
        "gemvx" in n
        or "gemv2" in n
        or "gemv_" in n
    ):
        return "GEMV"

    if (
        "softmax" in n
        or "attention" in n
        or "fmha" in n
        or "flash" in n
    ):
        return "ATTN"

    if (
        "gemm" in n
        or "sgemm" in n
        or "wmma" in n
        or "cublaslt::splitkreduce" in n
        or "cublaslt::epilogue" in n
    ):
        return "GEMM"

    if (
        "conv" in n
        or "cudnn" in n
    ):
        return "CONV"

    if (
        "reduce" in n
        or "layer_norm" in n
        or "layernorm" in n
        or "rms_norm" in n
        or "rmsnorm" in n
        or "batch_norm" in n
        or "batchnorm" in n
    ):
        return "REDNORM"

    if (
        "elementwise" in n
        or "copy" in n
        or "catarray" in n
        or "vectorized" in n
        or "gather" in n
        or "indexselect" in n
    ):
        return "ELEMCOPY"

    return "OTHER"


# ============================================================
# 3. Read NVML feature
# ============================================================

def read_nvml(base, request_count):

    summary_path = Path(
        str(base) + ".nvml_summary.json"
    )

    raw_path = Path(
        str(base) + ".nvml.jsonl"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            summary_path
        )

    if not raw_path.exists():
        raise FileNotFoundError(
            raw_path
        )

    x = json.loads(
        summary_path.read_text()
    )

    samples = sum(
        1
        for line in raw_path.read_text().splitlines()
        if line.strip()
    )

    return {
        "nvml_duration_s":
            float(x["duration_s"]),

        "nvml_samples":
            samples,

        "gpu_util_pct":
            float(x["gpu_util_pct"]["mean"]),

        "mem_util_pct":
            float(x["mem_util_pct"]["mean"]),

        "power_w":
            float(x["power_w"]["mean"]),

        "sm_clock_mhz":
            float(x["sm_clock_mhz"]["mean"]),

        "vram_used_mib":
            float(x["vram_used_mib"]["mean"]),

        "energy_j_per_req":
            float(
                x["estimated_energy_j"]
            ) / request_count,
    }


# ============================================================
# 4. Read Nsight kernel feature
# ============================================================

def read_kernel(rep, request_count):

    if not rep.exists():
        raise FileNotFoundError(rep)

    proc = subprocess.run(
        [
            "/usr/local/bin/nsys",
            "stats",
            "--report",
            "cuda_gpu_kern_sum",
            "--format",
            "csv",
            str(rep),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    lines = proc.stdout.splitlines()

    start = None

    for i, line in enumerate(lines):
        if (
            "Time (%)" in line
            and "Total Time" in line
            and "Instances" in line
            and "Name" in line
        ):
            start = i
            break

    if start is None:
        raise RuntimeError(
            f"Cannot find Nsight CSV header: {rep}"
        )

    reader = csv.DictReader(
        lines[start:]
    )

    category_ns = {
        x: 0
        for x in CATEGORIES
    }

    total_ns = 0
    instances = 0
    unique = 0

    for row in reader:

        try:
            ns = int(
                float(
                    row["Total Time (ns)"]
                )
            )

            count = int(
                float(
                    row["Instances"]
                )
            )

            name = row["Name"]

        except Exception:
            continue

        total_ns += ns
        instances += count
        unique += 1

        category_ns[
            classify_kernel(name)
        ] += ns

    result = {
        "kernel_gpu_ms_per_req":
            total_ns / 1e6 / request_count,

        "kernel_instances_per_req":
            instances / request_count,

        "kernel_unique":
            unique,
    }

    for cat in CATEGORIES:

        result[
            f"kernel_{cat}_pct"
        ] = (
            category_ns[cat]
            / total_ns
            * 100.0
            if total_ns
            else 0.0
        )

    return result


# ============================================================
# 5. Build 11-row service-path table
# ============================================================

service_rows = []

for spec in SERVICES:

    print(
        "Reading:",
        spec["path"],
        flush=True,
    )

    row = {
        "service_path": spec["path"],
        "model": spec["model"],
        "task": spec["task"],
    }

    row.update(
        read_nvml(
            spec["nvml"],
            spec["nvml_requests"],
        )
    )

    row.update(
        read_kernel(
            spec["nsys"],
            spec["nsys_requests"],
        )
    )

    service_rows.append(row)


service_csv = (
    OUT
    / "c1_service_path_features.csv"
)

with service_csv.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            service_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(service_rows)


features = {
    row["service_path"]: row
    for row in service_rows
}


# ============================================================
# 6. Pair measurement paths
# ============================================================

BASE = (
    CHAR
    / "c1_qwen_vs_whisper_v1"
)

OTHER = (
    CHAR
    / "c1_qwen_other_specialists_v1"
)

WHISPER = (
    CHAR
    / "c1_qwen_whisper_taskmatrix_v1"
)


VICTIM_TASKS = [
    "text",
    "image",
    "ocr",
    "asr",
    "visual_qa",
    "lid",
]


CONTENDERS = [
    (
        "distilbert/text",
        "distilbert",
        "text",
    ),
    (
        "resnet50/image",
        "resnet50",
        "image",
    ),
    (
        "paddleocr/ocr",
        "paddleocr",
        "ocr",
    ),
    (
        "whisper/asr",
        "whisper",
        "asr",
    ),
    (
        "whisper/lid",
        "whisper",
        "lid",
    ),
]


def read_jsonl(path):

    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def get_single(task):

    p = (
        BASE
        / f"qwen_{task}_single_n30.jsonl"
    )

    rows = read_jsonl(p)

    xs = [
        float(x["worker_latency_ms"])
        for x in rows
        if (
            x.get("worker") == "qwen"
            and x.get("task_type") == task
        )
    ]

    if len(xs) != 30:
        raise RuntimeError(
            f"Bad single count: "
            f"qwen/{task}: {len(xs)}"
        )

    return {
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
    }


def pair_file(
    victim_task,
    worker,
    co_task,
):

    if worker == "whisper":

        return (
            WHISPER
            / (
                f"qwen_{victim_task}"
                f"_vs_whisper_{co_task}_n30.jsonl"
            )
        )

    return (
        OTHER
        / (
            f"qwen_{victim_task}"
            f"_vs_{worker}_{co_task}_n30.jsonl"
        )
    )


# Numeric features to copy with victim/contender prefix.
FEATURE_COLUMNS = [
    "gpu_util_pct",
    "mem_util_pct",
    "power_w",
    "sm_clock_mhz",
    "vram_used_mib",
    "energy_j_per_req",
    "kernel_gpu_ms_per_req",
    "kernel_instances_per_req",
    "kernel_unique",
    "kernel_GEMV_pct",
    "kernel_GEMM_pct",
    "kernel_CONV_pct",
    "kernel_ATTN_pct",
    "kernel_REDNORM_pct",
    "kernel_ELEMCOPY_pct",
    "kernel_OTHER_pct",
]


# ============================================================
# 7. Build 30-row pair dataset
# ============================================================

pair_rows = []

for task in VICTIM_TASKS:

    baseline = get_single(task)

    victim_path = (
        f"qwen/{task}"
    )

    victim_feat = features[
        victim_path
    ]

    for (
        contender_path,
        worker,
        co_task,
    ) in CONTENDERS:

        p = pair_file(
            task,
            worker,
            co_task,
        )

        rows = read_jsonl(p)

        target_rows = [
            x
            for x in rows
            if (
                x.get("profile_role")
                == "target"
                and
                x.get("worker")
                == "qwen"
                and
                x.get("task_type")
                == task
            )
        ]

        if len(target_rows) != 30:
            raise RuntimeError(
                f"Bad pair target count: "
                f"{victim_path} <- "
                f"{contender_path}: "
                f"{len(target_rows)}"
            )

        pair_lat = [
            float(
                x["worker_latency_ms"]
            )
            for x in target_rows
        ]

        overlap = [
            float(
                x["active_overlap_ratio"]
            )
            for x in target_rows
        ]

        pair_mean = statistics.mean(
            pair_lat
        )

        pair_median = statistics.median(
            pair_lat
        )

        slowdown_mean = (
            pair_mean
            / baseline["mean"]
            - 1.0
        ) * 100.0

        slowdown_median = (
            pair_median
            / baseline["median"]
            - 1.0
        ) * 100.0

        contender_feat = features[
            contender_path
        ]

        row = {
            "victim_path":
                victim_path,

            "contender_path":
                contender_path,

            "victim_task":
                task,

            "contender_model":
                worker,

            "contender_task":
                co_task,

            "single_mean_ms":
                baseline["mean"],

            "single_median_ms":
                baseline["median"],

            "pair_mean_ms":
                pair_mean,

            "pair_median_ms":
                pair_median,

            "slowdown_mean_pct":
                slowdown_mean,

            "slowdown_median_pct":
                slowdown_median,

            "overlap_mean":
                statistics.mean(
                    overlap
                ),

            "overlap_min":
                min(overlap),
        }

        for col in FEATURE_COLUMNS:

            row[
                "victim_" + col
            ] = victim_feat[col]

            row[
                "contender_" + col
            ] = contender_feat[col]

        pair_rows.append(row)


pair_csv = (
    OUT
    / "c1_pair_dataset.csv"
)

with pair_csv.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            pair_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(pair_rows)


# ============================================================
# 8. Validation / preview
# ============================================================

print()
print("====================================")
print("C1 DATASET BUILD COMPLETE")
print("====================================")

print(
    "service paths =",
    len(service_rows),
)

print(
    "pair rows     =",
    len(pair_rows),
)

print(
    "service CSV   =",
    service_csv,
)

print(
    "pair CSV      =",
    pair_csv,
)

print()
print(
    "===== SERVICE PATH SUMMARY ====="
)

print(
    f"{'path':<20}"
    f"{'GPU%':>8}"
    f"{'MEM%':>8}"
    f"{'Power':>9}"
    f"{'GPUms':>9}"
    f"{'GEMV%':>8}"
    f"{'GEMM%':>8}"
    f"{'CONV%':>8}"
    f"{'ATTN%':>8}"
)

print("-" * 88)

for r in service_rows:

    print(
        f"{r['service_path']:<20}"
        f"{r['gpu_util_pct']:>8.1f}"
        f"{r['mem_util_pct']:>8.1f}"
        f"{r['power_w']:>9.1f}"
        f"{r['kernel_gpu_ms_per_req']:>9.1f}"
        f"{r['kernel_GEMV_pct']:>8.1f}"
        f"{r['kernel_GEMM_pct']:>8.1f}"
        f"{r['kernel_CONV_pct']:>8.1f}"
        f"{r['kernel_ATTN_pct']:>8.1f}"
    )

print()
print(
    "slowdown range =",
    round(
        min(
            x["slowdown_mean_pct"]
            for x in pair_rows
        ),
        2,
    ),
    "%",
    "to",
    round(
        max(
            x["slowdown_mean_pct"]
            for x in pair_rows
        ),
        2,
    ),
    "%",
)

print(
    "minimum overlap =",
    round(
        min(
            x["overlap_min"]
            for x in pair_rows
        ),
        4,
    ),
)
