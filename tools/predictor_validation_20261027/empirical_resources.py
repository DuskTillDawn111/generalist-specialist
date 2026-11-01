"""Training-only board-power and owned-VRAM conditional resource evidence.

Full family-count contexts are measured directly, not composed from pairs.
Power is a time-weighted conditional mean under the training input distribution,
not causal attribution. Memory scenarios describe sampled batch peaks, not safe
capacity bounds or simultaneous sums of separate allocator process peaks.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

from .profiling_analysis import RESIDENTS,digest,map_native_collection

SCHEMA='gsdeploy.empirical_board_owned_resource_profiles.v1'
DEFINITION={'schema':SCHEMA,'scope':'Training conditional full-family-count board power and sampled owned VRAM',
 'power':'piecewise-linear integral through every measured sample; time-weighted mean per directly observed full context',
 'memory':'independent-batch all-owned sampled peaks conditional on actually observed context, augmented by run idle sampled peak',
 'scenarios':['mean','max'],'sampling_gap_limit_ns':2_000_000_000,
 'max_scope':'Empirical maximum, not a capacity bound or calibrated confidence bound',
 'allocator_scope':'Separate per-process allocator peaks are not simultaneous and never summed as actual board peak',
 'core_enabled':False}


def resource_context(active,aliases):
    counts=collections.Counter(aliases[a]['family'] for a in active)
    members=[{'family':family,'count':count} for family,count in sorted(counts.items())]
    return digest(members),members


def _at(samples,timestamp,field):
    for left,right in zip(samples,samples[1:]):
        if left['time_ns']<=timestamp<=right['time_ns']:
            ratio=(timestamp-left['time_ns'])/(right['time_ns']-left['time_ns'])
            return left[field]+ratio*(right[field]-left[field])
    raise ValueError('Telemetry does not bracket requested interval')


def integrate_samples(samples,left,right,field='watts'):
    """Trapezoid integral exact for the declared piecewise-linear sample model."""
    if not left<right:raise ValueError('Positive resource observation interval required')
    points=[(left,_at(samples,left,field)),*[(s['time_ns'],s[field]) for s in samples if left<s['time_ns']<right],(right,_at(samples,right,field))]
    return sum((b-a)*(pa+pb)/2 for (a,pa),(b,pb) in zip(points,points[1:]))


def fit_resources(raw_path):
    mapped=map_native_collection(raw_path);traces=mapped['traces']
    raw=json.loads(Path(raw_path).read_text())
    if not traces or any(t['split']!='training' for t in traces):
        raise ValueError('Training-only collection required; do not mine validation telemetry')
    samples=raw['board_samples'];times=[s['time_ns'] for s in samples]
    if len(times)<2 or times!=sorted(set(times)):raise ValueError('Ordered nonduplicate board telemetry required')
    gaps=[b-a for a,b in zip(times,times[1:])];maxgap=max(gaps)
    if maxgap>DEFINITION['sampling_gap_limit_ns']:raise ValueError('Board telemetry gap exceeds unchanged 2-second evidence limit')
    for sample in samples:
        if set(sample['owned_aliases'])!=set(RESIDENTS) or sample['foreign_pids']:
            raise ValueError('All owned resident workers and no foreign GPU processes required')
        for key in ('watts','owned_used_mib'):
            value=sample[key]
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Finite nonnegative observed resource samples required')
    aliases={a:{'family':f} for a,f in RESIDENTS.items()}
    rows=[r for t in traces for r in t['requests']]
    observation_left=raw['observation_start_ns'];observation_right=raw['observation_end_ns']
    cuts=sorted({observation_left,observation_right,*[r[k] for r in rows for k in ('handler_start_ns','handler_end_ns')]})
    contexts={};intervals=[]
    for left,right in zip(cuts,cuts[1:]):
        active={r['alias']:r for r in rows if r['handler_start_ns']<=left and right<=r['handler_end_ns']}
        key,members=resource_context(active,aliases)
        watt_ns=integrate_samples(samples,left,right)
        context=contexts.setdefault(key,{'members':members,'duration_ns':0,'watt_ns':0.,'interval_count':0,'interval_watts':[],'batch_peaks_mib':[]})
        context['duration_ns']+=right-left;context['watt_ns']+=watt_ns;context['interval_count']+=1
        context['interval_watts'].append(watt_ns/(right-left));intervals.append((left,right,key))
    idle_key=resource_context({},aliases)[0]
    idle_samples=[s['owned_used_mib'] for s in samples if observation_left<=s['time_ns']<=observation_right and not any(r['handler_start_ns']<=s['time_ns']<r['handler_end_ns'] for r in rows)]
    idle_peak=max(idle_samples) if idle_samples else None
    # Each trial contributes its complete owned-process sampled peak to every
    # full context actually seen in that trial. This is a conditional trial-peak
    # statistic, NOT instantaneous context memory nor a provable upper bound.
    for trace in traces:
        left,right=trace['observation_window_ns']['start'],trace['observation_window_ns']['end']
        local=[s['owned_used_mib'] for s in samples if left<=s['time_ns']<=right]
        if not local:continue
        seen={key for a,b,key in intervals if a<right and left<b}
        peak=max(local)
        for key in seen:contexts[key]['batch_peaks_mib'].append({'trace_id':trace['id'],'raw_sampled_peak_mib':peak})
    fitted={}
    for key,value in contexts.items():
        peaks=[p['raw_sampled_peak_mib'] for p in value['batch_peaks_mib']]
        if key==idle_key and idle_peak is not None and not peaks:peaks=[idle_peak]
        mean_memory=max(idle_peak,statistics.mean(peaks)) if peaks and idle_peak is not None else None
        max_memory=max(idle_peak,max(peaks)) if peaks and idle_peak is not None else None
        mean_power=value['watt_ns']/value['duration_ns']
        fitted[key]={'members':value['members'],'evidence':str(Path(raw_path).resolve())+':'+key,
            'scenarios':{'mean':{'watts':mean_power,'peak_memory_mib':mean_memory},
                         'max':{'watts':max(value['interval_watts']),'peak_memory_mib':max_memory}},
            'diagnostics':{'duration_ns':value['duration_ns'],'interval_count':value['interval_count'],
                'independent_batch_peaks':len(value['batch_peaks_mib']),'batch_peaks':value['batch_peaks_mib'],
                'mean_memory_scope':'Mean of independent full trial sampled peaks conditioned on this context appearing, max with run idle sampled peak',
                'max_power_scope':'Maximum interval-average watts, not unsampled instantaneous maximum'}}
    return {'schema':SCHEMA,'definition':DEFINITION,'definition_sha256':digest(DEFINITION),
        'execution_context_sha256':raw['execution_context_sha256'],'provenance':{'kind':'measured','split':'training','raw':str(Path(raw_path).resolve())},
        'source_receipts':[{'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in (raw_path,__file__)],
        'contexts':fitted,'core_enabled':False,'GPU_started':False,
        'telemetry':{'sample_count':len(samples),'max_gap_ns':maxgap,'gap_limit_ns':DEFINITION['sampling_gap_limit_ns'],
            'max_query_duration_ns':max((s.get('query_duration_ns',0) for s in samples),default=0),
            'time_scope':'NVML query midpoint; power/memory measurements are not atomically simultaneous; sampling and interpolation uncertainty not calibrated',
            'idle_sample_count':len(idle_samples),'raw_idle_sampled_peak_mib':idle_peak,
            'raw_run_owned_sampled_peak_mib':max(s['owned_used_mib'] for s in samples),
            'continuous_peak_known':False,'capacity_bound':False,'allocator_peaks_summed':False},
        'memory_formula':'Provider returns raw empirical MiB; caller retains original safe_factor*max(history,predicted_peak)/budget exactly once',
        'scope':'Input-distribution conditional descriptive resource model; independent held-out power/memory error not yet assessed'}


class EmpiricalResourceProvider:
    def __init__(self,document,execution_context_sha256):
        if document.get('schema')!=SCHEMA or document.get('execution_context_sha256')!=execution_context_sha256:
            raise ValueError('Matching explicit empirical resource execution context required')
        if document.get('provenance',{}).get('split')!='training':raise ValueError('Training-only resource profiles required')
        self.document=copy.deepcopy(document);self.execution_context_sha256=execution_context_sha256

    def _lookup(self,active,aliases,scenario):
        key,_=resource_context(active,aliases);value=self.document['contexts'].get(key)
        if value is None or scenario not in value['scenarios']:
            return None,'UNKNOWN_UNOBSERVED_FULL_FAMILY_RESOURCE_CONTEXT:'+key+':'+scenario
        return (value['scenarios'][scenario],value['evidence']),None

    def power(self,active,aliases,scenario):
        value,reason=self._lookup(active,aliases,scenario)
        return {'known':value is not None,'watts':value[0]['watts'] if value else None,
            'evidence':value[1] if value else reason,'scope':'Whole-board conditional training context power; idle residency included, not causal per-model attribution'}

    def memory(self,active,aliases,scenario):
        value,reason=self._lookup(active,aliases,scenario)
        peak=value[0]['peak_memory_mib'] if value else None
        return {'known':peak is not None,'peak_memory_mib':peak,'evidence':value[1] if value else reason,
            'scope':'Raw owned all-resident empirical conditional trial sampled peak; not instantaneous context peak or capacity bound',
            'capacity_bound':False,'safe_factor_applied':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);result=fit_resources(args.raw)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
