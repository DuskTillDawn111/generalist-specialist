from dataclasses import dataclass


@dataclass
class CandidateModel:
    id: str
    family: str
    role: str
    capabilities_verified: list[str]
    capabilities_pending: list[str]
    source: str
    source_revision: str | None
    parameter_count: float | None
    weight_size_gb: float | None
    measured_idle_vram_mib: float | None
    runtime_backend: str
    precision: str | None
    adapter_status: str
    license: str | None
