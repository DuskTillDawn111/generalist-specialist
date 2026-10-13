"""Replaceable performance acquisition, independent of the optimizer."""

from abc import ABC, abstractmethod
import json
from pathlib import Path

from .schema import Deployment, PerformanceResult, Workload


class MissingPerformanceError(LookupError):
    pass


class PerformanceModel(ABC):
    @abstractmethod
    def predict(self, deployment: Deployment, workload: Workload) -> PerformanceResult:
        """Predict serving metrics under the backend's fixed experimental context."""


class TablePerformanceModel(PerformanceModel):
    """Exact matching only; never interpolate or substitute standalone latency.

    Each table belongs to ONE hardware/routing/protocol context. A future
    functional backend implements the same two-argument predict interface.
    """

    def __init__(self, context: dict, rows: list[dict]):
        for name in ("data_kind", "hardware_id", "routing_policy", "protocol", "latency_definition"):
            if not isinstance(context.get(name), str) or not context[name].strip():
                raise ValueError(f"Missing table context: {name}")
        if context["data_kind"] not in {"synthetic", "measured"}:
            raise ValueError("data_kind must be synthetic or measured")
        self.context = dict(context)
        self._entries = {}
        for row in rows:
            deployment = Deployment(row["model_counts"])
            workload = Workload(row["arrival_rates"])
            key = (deployment.key, workload.key)
            if key in self._entries:
                raise ValueError(f"Duplicate performance point: {key}")
            self._entries[key] = PerformanceResult(**row["performance"])

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported performance table schema_version")
        return cls(data["context"], data["rows"])

    def predict(self, deployment, workload):
        try:
            return self._entries[(deployment.key, workload.key)]
        except KeyError:
            raise MissingPerformanceError(
                f"No exact table entry for {dict(deployment.key)} at {workload.arrival_rates}"
            ) from None
