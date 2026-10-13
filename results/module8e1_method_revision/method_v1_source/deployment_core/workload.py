"""Task arrival rates in requests/second, never sample counts or percentages."""

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

TASKS = ("text", "image", "ocr", "asr", "visual_qa", "lid")


def validate_number(value, name, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        domain = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {domain} number")


@dataclass(frozen=True)
class Workload:
    arrival_rate: Mapping[str, float]

    def __post_init__(self):
        unknown = set(self.arrival_rate) - set(TASKS)
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")
        for task, rate in self.arrival_rate.items():
            validate_number(rate, f"arrival_rate.{task}")
        object.__setattr__(self, "arrival_rate", MappingProxyType({
            task: float(self.arrival_rate.get(task, 0)) for task in TASKS
        }))

    @property
    def active_tasks(self):
        return tuple(task for task in TASKS if self.arrival_rate[task] > 0)


def parse_workload(value: str) -> Workload:
    rates = {}
    for item in value.split(","):
        if item.count("=") != 1:
            raise ValueError("Use comma-separated task=rate pairs, e.g. text=10,asr=2")
        task, rate = (part.strip() for part in item.split("="))
        if task in rates:
            raise ValueError(f"Duplicate task: {task}")
        try:
            rates[task] = float(rate)
        except ValueError:
            raise ValueError(f"Invalid arrival rate for {task}: {rate}") from None
    return Workload(rates)
