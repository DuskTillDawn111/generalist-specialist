from abc import ABC, abstractmethod
from .schema import PerformancePrediction


class PerformanceModel(ABC):
    """Offline planning contract P(x,r,lambda); live queues/telemetry are not inputs.

    A future runtime correction model must use a separate interface and must not
    be required by planning or deployment search. No runtime correction is implemented.
    """
    @abstractmethod
    def predict(self, deployment, workload, routing) -> PerformancePrediction:
        raise NotImplementedError
