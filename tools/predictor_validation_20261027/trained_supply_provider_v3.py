"""Use fitted kernel regression inside supply scheduling; no target fallback."""
from __future__ import annotations

from collections import Counter

from gsdeploy.forecast_errors import UnknownEvidence
from .launch_supply_residual_v2 import LaunchSupplyResidualProvider


class TrainedKernelSupplyProvider(LaunchSupplyResidualProvider):
    def __init__(self,*args,baseline_resolver,**kwargs):
        super().__init__(*args,**kwargs)
        self.unsupported={};self.baseline_coverage={}
        for profile_id,profile in self.profiles.items():
            known=0;unknown=[];reasons=Counter();predicted_ns=0.
            for segment in profile['segments']:
                if segment['kind']!='gpu':continue
                result=baseline_resolver(segment)
                if result.get('known') is not True:
                    unknown.append(dict(segment_id=segment['id'],reasons=result.get('reasons',[])))
                    reasons.update(result.get('reasons',[]) or ['UNSPECIFIED_BASELINE_UNKNOWN'])
                    continue
                duration=result['duration_ns']/1e6
                if not 0<duration<float('inf'):raise ValueError('Positive finite trained kernel prediction required')
                known+=1
                predicted_ns+=result['duration_ns']
                segment['measured_profiled_duration_ms']=dict(segment['duration_ms'])
                segment['duration_ms']={scenario:duration for scenario in segment['duration_ms']}
                segment['timing_evidence']=dict(original=segment['timing_evidence'],
                    baseline_prediction={k:v for k,v in result.items() if k!='prediction'},measured_target_fallback_used=False)
                internal=self.internal.segments[profile_id,segment['id']]
                internal['duration_ms']=dict(segment['duration_ms']);internal['timing_evidence']=segment['timing_evidence']
            self.baseline_coverage[profile_id]=dict(known_kernels=known,unknown_kernels=len(unknown),
                                                  unknown_reason_counts=dict(reasons),
                                                  supported_kernel_predicted_sum_ms=predicted_ns/1e6,
                                                  partial_sum_is_whole_request_prediction=False,
                                                  measured_target_fallback_used=False)
            if unknown:self.unsupported[profile_id]=unknown

    def plan(self,row,aliases,scenario):
        profile=self.lookup.get((row['alias'],row['task'],row['pool_id']))
        if profile is not None and profile['id'] in self.unsupported:
            raise UnknownEvidence('FITTED_KERNEL_BASELINE_INCOMPLETE:'+profile['id']+':'+str(len(self.unsupported[profile['id']])))
        return super().plan(row,aliases,scenario)
