"""Read existing measurements; no model execution, guessed values or profiling."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .workload import TASKS, validate_number

DEFAULT_SOURCES = Path(__file__).with_name("catalog_sources.json")


@dataclass(frozen=True)
class ModelProfile:
    name: str
    supported_tasks: frozenset[str]
    memory_mib: float
    service_time_ms: Mapping[str, float]
    sources: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Model name must be a nonempty string")
        tasks = frozenset(self.supported_tasks)
        if not tasks or tasks - set(TASKS):
            raise ValueError(f"Invalid supported tasks for {self.name}")
        if set(self.service_time_ms) != tasks:
            raise ValueError(f"Service times must exactly cover capabilities for {self.name}")
        validate_number(self.memory_mib, f"{self.name}.memory_mib", positive=True)
        for task, value in self.service_time_ms.items():
            validate_number(value, f"{self.name}/{task}.service_time_ms", positive=True)
        object.__setattr__(self, "supported_tasks", tasks)
        object.__setattr__(self, "service_time_ms", MappingProxyType(dict(self.service_time_ms)))


@dataclass(frozen=True)
class Hardware:
    gpu_memory_mib: float
    name: str = "single GPU"
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        validate_number(self.gpu_memory_mib, "gpu_memory_mib")


def read_source(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    return json.loads(content), {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def read_manifest(path):
    path = Path(path).resolve()
    data, _ = read_source(path)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported catalog sources schema_version")
    return data, path.parent


def load_model_catalog(sources_path=DEFAULT_SOURCES) -> dict[str, ModelProfile]:
    manifest, base = read_manifest(sources_path)
    metadata, capability_source = read_source(base / manifest["capability_profile"])
    directories = [(base / p).resolve() for p in manifest["service_summary_dirs"]]
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("Service summary directories must be nonempty and unique")
    catalog = {}
    for name, memory_path in manifest["memory_snapshots"].items():
        info = metadata["models"][name]
        snapshot, memory_source = read_source(base / memory_path)
        if snapshot.get("return_code") != 0:
            raise ValueError(f"Unsuccessful memory profile: {memory_path}")
        memory = snapshot["peak_target_process_memory_mib"]
        if memory != info["peak_process_vram_mib"]:
            raise ValueError(f"Memory snapshot disagrees with metadata for {name}")
        memory_source["field"] = "peak_target_process_memory_mib"
        services, service_sources = {}, {}
        for task in info["supported_tasks"]:
            samples, weighted_ms, references = 0, 0.0, []
            for directory in directories:
                # Select by identity fields, not approximate numbers or a latest-file heuristic.
                matches = []
                for path in sorted(directory.glob("*.summary.json")):
                    summary, reference = read_source(path)
                    if summary.get("target_worker") == name and summary.get("target_task") == task:
                        matches.append((summary, reference))
                if len(matches) != 1:
                    raise ValueError(f"Expected one standalone summary for {name}/{task} in {directory}")
                summary, reference = matches[0]
                if summary.get("protocol") != "active_single_profile_v1":
                    raise ValueError(f"Unexpected standalone protocol: {reference['path']}")
                count = summary["target_count"]
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError(f"Invalid target_count: {reference['path']}")
                mean_ms = summary["mean_worker_latency_ms"]
                validate_number(mean_ms, f"{name}/{task}.mean_worker_latency_ms", positive=True)
                samples += count
                weighted_ms += mean_ms * count
                references.append({**reference, "field": "mean_worker_latency_ms",
                                   "target_count": count, "mean_worker_latency_ms": mean_ms,
                                   "protocol": summary["protocol"]})
            services[task] = weighted_ms / samples
            service_sources[task] = {"aggregation": "sample-count-weighted mean",
                                     "total_samples": samples, "summaries": references}
        catalog[name] = ModelProfile(name, frozenset(info["supported_tasks"]), memory, services, {
            "capability": {**capability_source, "field": f"models.{name}.supported_tasks"},
            "memory": memory_source,
            "service_time": service_sources,
        })
    if not catalog:
        raise ValueError("Model catalog is empty")
    return catalog


def load_hardware(sources_path=DEFAULT_SOURCES) -> Hardware:
    manifest, base = read_manifest(sources_path)
    data, reference = read_source(base / manifest["hardware_snapshot"])
    if data.get("return_code") != 0:
        raise ValueError("Unsuccessful hardware snapshot")
    return Hardware(data["gpu_baseline"]["total_mib"], data["gpu"],
                    {**reference, "field": "gpu_baseline.total_mib"})
