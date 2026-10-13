import json
from pathlib import Path
from collections import defaultdict

PROFILE_PATH = Path(
    "benchmark/profiles/model_placement_profile.json"
)

OUT_DIR = Path(
    "benchmark/results/composition_motivation/plans"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

with PROFILE_PATH.open("r", encoding="utf-8") as f:
    profile = json.load(f)

models_profile = profile["models"]

# OCR-heavy workload
WORKLOAD = {
    "text": 80,
    "image": 80,
    "ocr": 560,
    "asr": 80,
}

BUDGET_GIB = 24.0
BUDGET_MIB = BUDGET_GIB * 1024


PLANS = {
    "d0_qwen": {
        "models": [
            "qwen",
        ],
        "executor": {
            "text": "qwen",
            "image": "qwen",
            "ocr": "qwen",
            "asr": "qwen",
        },
    },

    "d1_qwen_paddle": {
        "models": [
            "qwen",
            "paddleocr",
        ],
        "executor": {
            "text": "qwen",
            "image": "qwen",
            "ocr": "paddleocr",
            "asr": "qwen",
        },
    },

    "d2_qwen_resnet_paddle": {
        "models": [
            "qwen",
            "resnet50",
            "paddleocr",
        ],
        "executor": {
            "text": "qwen",
            "image": "resnet50",
            "ocr": "paddleocr",
            "asr": "qwen",
        },
    },

    "d3_qwen_distil_resnet_paddle": {
        "models": [
            "qwen",
            "distilbert",
            "resnet50",
            "paddleocr",
        ],
        "executor": {
            "text": "distilbert",
            "image": "resnet50",
            "ocr": "paddleocr",
            "asr": "qwen",
        },
    },

    "d4_all_specialists": {
        "models": [
            "distilbert",
            "resnet50",
            "paddleocr",
            "whisper",
        ],
        "executor": {
            "text": "distilbert",
            "image": "resnet50",
            "ocr": "paddleocr",
            "asr": "whisper",
        },
    },
}


for plan_name, cfg in PLANS.items():

    resident_models = cfg["models"]
    executor = cfg["executor"]

    # -----------------------------
    # Validate model names
    # -----------------------------
    for model in resident_models:
        if model not in models_profile:
            raise RuntimeError(
                f"{plan_name}: unknown model {model}"
            )

    # -----------------------------
    # Build fixed routing
    # -----------------------------
    routing = {}
    model_load_ms = defaultdict(float)

    for task_type, request_count in WORKLOAD.items():

        model = executor[task_type]

        if model not in resident_models:
            raise RuntimeError(
                f"{plan_name}: {model} handles "
                f"{task_type} but is not resident"
            )

        if task_type not in models_profile[model]["tasks"]:
            raise RuntimeError(
                f"{plan_name}: model {model} "
                f"does not have task profile {task_type}"
            )

        service_ms = float(
            models_profile[model]
            ["tasks"]
            [task_type]
            ["service_ms"]
        )

        routing[task_type] = {
            model: {
                "fraction": 1.0,
                "requests": float(request_count),
            }
        }

        model_load_ms[model] += (
            request_count * service_ms
        )

    # -----------------------------
    # Memory
    # Exact field confirmed from profile:
    # peak_process_vram_mib
    # -----------------------------
    memory_mib = sum(
        float(
            models_profile[model]
            ["peak_process_vram_mib"]
        )
        for model in resident_models
    )

    if memory_mib > BUDGET_MIB:
        raise RuntimeError(
            f"{plan_name}: memory {memory_mib} MiB "
            f"exceeds budget {BUDGET_MIB} MiB"
        )

    # -----------------------------
    # Standalone-profile prediction
    # Same logic as run_placement_plan.py
    # -----------------------------
    predicted_makespan_ms = max(
        model_load_ms.values()
    )

    total_requests = sum(
        WORKLOAD.values()
    )

    predicted_throughput_qps = (
        total_requests
        / (predicted_makespan_ms / 1000.0)
    )

    generalist_count = (
        1 if "qwen" in resident_models else 0
    )

    specialist_count = (
        len(resident_models)
        - generalist_count
    )

    best = {
        "models": resident_models,
        "model_count": len(resident_models),
        "generalist_count": generalist_count,
        "specialist_count": specialist_count,
        "memory_mib": memory_mib,
        "predicted_makespan_ms":
            predicted_makespan_ms,
        "predicted_throughput_qps":
            predicted_throughput_qps,
        "routing": routing,
        "model_load_ms": dict(model_load_ms),
    }

    result_row = {
        "budget_gib": BUDGET_GIB,
        "budget_mib": BUDGET_MIB,
        "max_models": len(resident_models),
        "max_generalists": generalist_count,
        "quality_preset":
            "motivation_fixed_routing",
        "subsets_enumerated": 1,
        "feasible_count": 1,
        "best": best,
        "top10": [best],
    }

    output = {
        "profile": str(PROFILE_PATH),
        "scenario":
            "ocr_heavy_composition_motivation",
        "quality_preset":
            "motivation_fixed_routing",
        "workload": WORKLOAD,
        "results": [
            result_row
        ],
    }

    output_path = (
        OUT_DIR / f"{plan_name}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 70)
    print("Plan:", plan_name)
    print("Models:", resident_models)
    print("Memory MiB:", memory_mib)
    print(
        "Predicted QPS:",
        round(predicted_throughput_qps, 3)
    )
    print("Routing:")
    for task_type in [
        "text",
        "image",
        "ocr",
        "asr",
    ]:
        print(
            f"  {task_type:>5} -> "
            f"{executor[task_type]}"
        )
    print("Saved:", output_path)

