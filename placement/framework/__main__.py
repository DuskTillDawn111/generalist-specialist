"""CPU-only entry point: python -m placement.framework --help."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .performance import TablePerformanceModel
from .schema import Hardware, MODEL_NAMES, Model, Workload
from .solver import Constraints, optimize

ROOT = Path(__file__).resolve().parents[2]


def load_catalog(path):
    """Reuse measured capabilities and memory; do not import old latency models."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        name: Model(name, frozenset(data["models"][name]["supported_tasks"]),
                    data["models"][name]["peak_process_vram_mib"])
        for name in MODEL_NAMES
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True, help="JSON object of task: req/s")
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--catalog", type=Path,
                        default=ROOT / "benchmark/profiles/model_service_profile_6task.json")
    parser.add_argument("--gpu-memory-mib", type=float, default=49140)
    parser.add_argument("--reserved-memory-mib", type=float, default=0)
    parser.add_argument("--objective", choices=("mean_latency", "throughput"), default="mean_latency")
    parser.add_argument("--min-throughput-rps", type=float, help="Default: total offered req/s")
    parser.add_argument("--max-p95-latency-ms", type=float)
    parser.add_argument("--allow-partial", action="store_true", help="Select only among table-covered candidates")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        workload = Workload(json.loads(args.workload.read_text(encoding="utf-8")))
        backend = TablePerformanceModel.from_json(args.table)
        hardware = Hardware(args.gpu_memory_mib, args.reserved_memory_mib)
        constraints = Constraints(
            workload.total_rps if args.min_throughput_rps is None else args.min_throughput_rps,
            args.max_p95_latency_ms,
        )
        result = optimize(load_catalog(args.catalog), workload, hardware, backend,
                          args.objective, constraints, args.allow_partial)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    output = {
        "schema_version": 1,
        "workload": asdict(workload),
        "hardware": asdict(hardware),
        "constraints": asdict(constraints),
        "performance_context": backend.context,
        **asdict(result),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result.best is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
