"""Untrained controlled-load interpolation baseline; never extrapolate or invent points."""
from dataclasses import dataclass
import math
from pathlib import Path
from .base import PerformanceModel
from .schema import TaskPrediction, PerformancePrediction
from .features import curve_context


@dataclass(frozen=True)
class LoadPoint:
    model: str
    task: str
    standalone_service_demand: float
    context: dict
    mean_latency_ms: float | None
    p95_latency_ms: float | None
    throughput_rps: float | None
    source_result_paths: tuple[str,...]


def same_context(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(same_context(a[k],b[k]) for k in a)
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-10)
    return a==b


class PiecewiseLoadLatencyModel(PerformanceModel):
    def __init__(self,feature_extractor,profile_points=()):
        self.extractor=feature_extractor
        self.points=tuple(profile_points)
        for p in self.points:
            if not p.source_result_paths or any(not Path(s).is_absolute() for s in p.source_result_paths):
                raise ValueError('Measured load points require absolute source result paths')
            if not isinstance(p.context,dict) or not p.context:
                raise ValueError('Controlled feature context is required')
            if not math.isfinite(p.standalone_service_demand) or p.standalone_service_demand<0:
                raise ValueError('Invalid load coordinate')
            for value in (p.mean_latency_ms,p.p95_latency_ms,p.throughput_rps):
                if value is not None and (not math.isfinite(value) or value<0): raise ValueError('Invalid measured metric')

    def _path(self,features,task,model):
        u=features['per_model'][model]['standalone_service_demand']
        context=curve_context(features,model)
        points=sorted((p for p in self.points if p.model==model and p.task==task and same_context(p.context,context)),
                      key=lambda p:p.standalone_service_demand)
        xs=[p.standalone_service_demand for p in points]
        if len(set(xs))<2: return TaskPrediction()
        if len(xs)!=len(set(xs)): return TaskPrediction(status='duplicate_load_points_require_explicit_aggregation')
        if u<xs[0] or u>xs[-1]: return TaskPrediction(status='out_of_profile_range')
        left,right=next((a,b) for a,b in zip(points,points[1:]) if a.standalone_service_demand<=u<=b.standalone_service_demand)
        w=(u-left.standalone_service_demand)/(right.standalone_service_demand-left.standalone_service_demand)
        def interpolate(key):
            a,b=getattr(left,key),getattr(right,key)
            if w==0: return a
            if w==1: return b
            return None if a is None or b is None else a+(b-a)*w
        return TaskPrediction(*(interpolate(k) for k in ('mean_latency_ms','p95_latency_ms','throughput_rps')),status='interpolated_profile_baseline')

    def predict(self,deployment,workload,routing):
        f=self.extractor.extract(deployment,workload,routing)
        if any(n>1 for n in f['deployment'].values()):
            return PerformancePrediction({k:TaskPrediction(status='unsupported_replica_semantics') for k in f['workload']},status='unsupported_replica_semantics')
        predictions={}; path_status={}
        for task,rate in f['workload'].items():
            if rate==0:
                predictions[task]=TaskPrediction(throughput_rps=0.,status='inactive_task'); continue
            paths=[(m,r,self._path(f,task,m)) for m,r in f['routing'][task].items() if r>0]
            path_status[task]={m:p.status for m,r,p in paths}
            mean=sum(r/rate*p.mean_latency_ms for m,r,p in paths) if all(p.mean_latency_ms is not None for m,r,p in paths) else None
            throughput=sum(p.throughput_rps for m,r,p in paths) if all(p.throughput_rps is not None for m,r,p in paths) else None
            # Quantiles of mixtures cannot be averaged from component quantiles.
            tail=paths[0][2].p95_latency_ms if len(paths)==1 else None
            status='interpolated_profile_baseline' if mean is not None else next((p.status for m,r,p in paths if p.mean_latency_ms is None),'insufficient_profile_data')
            predictions[task]=TaskPrediction(mean,tail,throughput,status)
        system=sum(p.throughput_rps for p in predictions.values()) if all(p.throughput_rps is not None for p in predictions.values()) else None
        status='interpolated_profile_baseline' if all(p.mean_latency_ms is not None or f['workload'][k]==0 for k,p in predictions.items()) else 'insufficient_profile_data'
        return PerformancePrediction(predictions,system,status,dict(path_status=path_status,
            limitation='No accuracy claim; mixed-route P95 is null without a latency distribution. No automatic curve construction from routing sweeps.'))
