"""CPU-only public schemas; times in milliseconds at the API boundary."""
from dataclasses import dataclass
from typing import Any
@dataclass(frozen=True)
class PairPrediction:
 target_model: str
 target_task: str
 co_model: str
 co_task: str
 standalone_ms: float
 additional_delay_ms: float
 predicted_time_ms: float
 mu_hat: float
 alignment_count: int
 gamma: float
 policy: str
 confidence: dict[str,Any]
 def as_dict(self):
  from dataclasses import asdict
  return asdict(self)
