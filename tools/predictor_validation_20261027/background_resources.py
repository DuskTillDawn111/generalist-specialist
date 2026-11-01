"""Generalized all-task resource evidence using explicitly mapped training traces.

Family-count aggregation assumes task exchangeability for resource prediction.
The model records active family/task multisets and their conditional telemetry
separately so image/ASR omissions cannot silently inherit text-only evidence.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

from .empirical_resources import SCHEMA,DEFINITION as BASE_DEFINITION,integrate_samples,resource_context,EmpiricalResourceProvider
from .profiling_analysis import digest

DEFINITION={**BASE_DEFINITION,'generalized_tasks':True,
 'task_exchangeability_assumption':'Power/memory aggregated by full active family counts; different task mixes are pooled conditional observations, not task-invariant physics',
 'task_coverage_requirement':'Explicit family/task multiset coverage is recorded; unseen task mix remains unsupported for task-specific claims'}

TASK_CONDITIONAL_DEFINITION={
 'power':'Time-weighted mean from the directly observed full family/task multiset; family means are never substituted',
 'power_scenarios':['mean'],
 'memory':'Mean/max of full trial sampled owned peaks conditioned on the directly observed family/task multiset, augmented by run idle sampled peak',
 'memory_scenarios':['mean','max'],
 'scope':'Conditional empirical training statistics, not causal attribution or capacity/confidence bounds'}


def _task_key(active):
    members=[{'family':f,'task':task,'count':n} for (f,task),n in sorted(collections.Counter((r['family'],r['task']) for r in active.values()).items())]
    return digest(members),members


def fit_generalized_resources(raw,traces,residents,*,sources):
    """Fit generalized mapped TRAINING observations, no hardcoded task labels.

    residents maps alias->family or alias->{family,...}. Source receipts are
    immutable original raw+mapper provenance supplied by the actual mapper.
    """
    if not sources:raise ValueError('Explicit raw/mapper source receipts required')
    for source in sources:
        path=Path(source['path'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Changed generalized resource source')
    if raw.get('status')!='completed' or raw.get('backend')!='native' or raw.get('kind')!='measured':raise ValueError('Complete measured native collection required')
    if any(raw.get(k) for k in ('errors','runtime_errors','telemetry_errors')):raise ValueError('Error-free resource observation required')
    if not traces or any(t['split']!='training' for t in traces):raise ValueError('Training-only generalized resource traces required')
    aliases={a:{'family':v['family'] if isinstance(v,dict) else v} for a,v in residents.items()}
    rows=[r for t in traces for r in t['requests']];seen=set()
    for row in rows:
        if row['alias'] not in aliases or aliases[row['alias']]['family']!=row['family'] or not row.get('task'):
            raise ValueError('Generalized alias/family/task identity required')
        if row['request_id'] in seen:raise ValueError('Duplicate generalized request identity')
        seen.add(row['request_id'])
        if type(row['handler_start_ns']) is not int or type(row['handler_end_ns']) is not int or not row['handler_start_ns']<row['handler_end_ns']:
            raise ValueError('Complete actual observed handler interval required')
    samples=raw['board_samples'];times=[s['time_ns'] for s in samples]
    if len(samples)<2 or times!=sorted(set(times)) or max(b-a for a,b in zip(times,times[1:]))>2_000_000_000:raise ValueError('Ordered telemetry within unchanged 2-second gap limit required')
    for sample in samples:
        if set(sample['owned_aliases'])!=set(aliases) or sample['foreign_pids']:raise ValueError('All generalized residents owned, foreign-free telemetry required')
        if any(type(sample[k]) not in (int,float) or not math.isfinite(sample[k]) or sample[k]<0 for k in ('watts','owned_used_mib')):raise ValueError('Finite observed resources required')
    left,right=raw['observation_start_ns'],raw['observation_end_ns']
    if any(not left<=r['handler_start_ns']<r['handler_end_ns']<=right for r in rows):raise ValueError('Handlers outside frozen observation span')
    cuts=sorted({left,right,*[r[k] for r in rows for k in ('handler_start_ns','handler_end_ns')]})
    contexts,task_contexts,intervals={},{},[]
    for a,b in zip(cuts,cuts[1:]):
        active_rows=[r for r in rows if r['handler_start_ns']<=a and b<=r['handler_end_ns']]
        if len({r['alias'] for r in active_rows})!=len(active_rows):raise ValueError('Generalized independent alias overlap unsupported')
        active={r['alias']:r for r in active_rows};key,members=resource_context(active,aliases);task_key,tasks=_task_key(active)
        energy=integrate_samples(samples,a,b);watts=energy/(b-a)
        context=contexts.setdefault(key,{'members':members,'duration_ns':0,'watt_ns':0.,'interval_watts':[],'peaks':[],'task_context_ids':set()})
        context['duration_ns']+=b-a;context['watt_ns']+=energy;context['interval_watts'].append(watts);context['task_context_ids'].add(task_key)
        task=task_contexts.setdefault(task_key,{'members':tasks,'family_context_id':key,'duration_ns':0,'watt_ns':0.,'peaks':[]})
        task['duration_ns']+=b-a;task['watt_ns']+=energy;intervals.append((a,b,key,task_key))
    idle=[s['owned_used_mib'] for s in samples if left<=s['time_ns']<=right and not any(r['handler_start_ns']<=s['time_ns']<r['handler_end_ns'] for r in rows)]
    idle_peak=max(idle) if idle else None
    for trace in traces:
        a,b=trace['observation_window_ns']['start'],trace['observation_window_ns']['end']
        local=[s['owned_used_mib'] for s in samples if a<=s['time_ns']<=b]
        if not local:continue
        for key in {k for c,d,k,tk in intervals if c<b and a<d}:contexts[key]['peaks'].append(max(local))
        for tk in {tk for c,d,k,tk in intervals if c<b and a<d}:task_contexts[tk]['peaks'].append(max(local))
    fitted={}
    for key,c in contexts.items():
        peaks=c['peaks'];mean=max(idle_peak,statistics.mean(peaks)) if idle_peak is not None and peaks else None
        maximum=max(idle_peak,max(peaks)) if idle_peak is not None and peaks else None
        fitted[key]={'members':c['members'],'evidence':'generalized-training:'+key,
            'scenarios':{'mean':{'watts':c['watt_ns']/c['duration_ns'],'peak_memory_mib':mean},
                         'max':{'watts':max(c['interval_watts']),'peak_memory_mib':maximum}},
            'diagnostics':{'duration_ns':c['duration_ns'],'independent_batch_peak_count':len(peaks),
                           'task_context_ids':sorted(c['task_context_ids']),'task_exchangeability_unvalidated':len(c['task_context_ids'])>1}}
    task_report={key:{'members':c['members'],'family_context_id':c['family_context_id'],'duration_ns':c['duration_ns'],
        'mean_watts':c['watt_ns']/c['duration_ns'],'independent_batch_peak_count':len(c['peaks']),
        'mean_trial_sampled_peak_mib':statistics.mean(c['peaks']) if c['peaks'] else None,
        'max_trial_sampled_peak_mib':max(c['peaks']) if c['peaks'] else None,
        'family_pooled_mean_watts':fitted[c['family_context_id']]['scenarios']['mean']['watts'],
        'pooled_minus_task_mean_watts':fitted[c['family_context_id']]['scenarios']['mean']['watts']-c['watt_ns']/c['duration_ns'],
        'scope':'Descriptive training task-mix difference; no causal or held-out task-invariance claim'} for key,c in task_contexts.items()}
    return {'schema':SCHEMA,'definition':DEFINITION,'definition_sha256':digest(DEFINITION),
        'execution_context_sha256':raw['execution_context_sha256'],'provenance':{'kind':'measured','split':'training','sources':sources},
        'contexts':fitted,'core_enabled':False,'GPU_started':False,
        'task_conditional_definition':TASK_CONDITIONAL_DEFINITION,
        'task_conditional_definition_sha256':digest(TASK_CONDITIONAL_DEFINITION),
        'task_specific_resource_omission_audit':{'active_task_contexts':task_report,
            'family_contexts_with_multiple_task_mixes':sum(c['diagnostics']['task_exchangeability_unvalidated'] for c in fitted.values()),
            'known_tasks':sorted({r['task'] for r in rows}),'unknown_task_mix_may_not_transfer':True,
            'scope':'Family-count provider remains empirical pooled mean; use task-context coverage to gate generalization'},
        'telemetry':{'sample_count':len(samples),'max_gap_ns':max(b-a for a,b in zip(times,times[1:])),
            'max_query_duration_ns':max(s.get('query_duration_ns',0) for s in samples),'raw_idle_sampled_peak_mib':idle_peak,
            'raw_run_owned_sampled_peak_mib':max(s['owned_used_mib'] for s in samples),'capacity_bound':False,
            'timing_uncertainty':'NVML midpoint queries are not atomic; no calibrated timing/confidence bound'},
        'scope':'Raw owned sampled peaks and whole-board conditional power; no allocator peak sums, safe factor not yet applied'}


class TaskAwareResourceProvider(EmpiricalResourceProvider):
    """Reuse family-count means only inside an actually observed task domain."""
    def _task_covered(self,active,aliases):
        rows={}
        for alias,row in active.items():
            if not row.get('task'):
                return False,'UNKNOWN_RESOURCE_ACTIVE_TASK:'+alias
            rows[alias]={'family':aliases[alias]['family'],'task':row['task']}
        key,_=_task_key(rows)
        observed=self.document.get('task_specific_resource_omission_audit',{}).get('active_task_contexts',{})
        return key in observed,'UNKNOWN_UNOBSERVED_RESOURCE_FAMILY_TASK_CONTEXT:'+key

    def power(self,active,aliases,scenario):
        known,reason=self._task_covered(active,aliases)
        if not known:return {'known':False,'watts':None,'evidence':reason,'scope':'Unseen task mix cannot inherit pooled family resource values'}
        result=super().power(active,aliases,scenario)
        result['scope']+='; task-mix observed, family pooling remains empirical assumption'
        return result

    def memory(self,active,aliases,scenario):
        known,reason=self._task_covered(active,aliases)
        if not known:return {'known':False,'peak_memory_mib':None,'evidence':reason,'capacity_bound':False,'safe_factor_applied':False}
        return super().memory(active,aliases,scenario)


class TaskConditionalResourceProvider(EmpiricalResourceProvider):
    """Direct family/task-multiset statistics without family pooling.

    Max power is deliberately unknown: the task audit retained means, not
    interval-average maxima. Memory remains conditional trial-peak evidence.
    """
    def _conditional(self,active,aliases):
        rows={}
        for alias,row in active.items():
            family=aliases.get(alias,{}).get('family')
            if not family or not isinstance(row.get('task'),str) or not row['task']:
                return None,'UNKNOWN_RESOURCE_ACTIVE_TASK_OR_FAMILY:'+alias
            if row.get('family',family)!=family:
                return None,'UNKNOWN_RESOURCE_ACTIVE_FAMILY_MISMATCH:'+alias
            rows[alias]={'family':family,'task':row['task']}
        key,_=_task_key(rows)
        value=self.document.get('task_specific_resource_omission_audit',{}).get('active_task_contexts',{}).get(key)
        return value,'task-conditional-training:'+key if value is not None else 'UNKNOWN_UNOBSERVED_RESOURCE_FAMILY_TASK_CONTEXT:'+key

    def power(self,active,aliases,scenario):
        value,evidence=self._conditional(active,aliases)
        watts=value.get('mean_watts') if value is not None and scenario=='mean' else None
        if value is not None and scenario!='mean':
            evidence='UNKNOWN_UNRECORDED_TASK_CONDITIONAL_POWER_SCENARIO:'+scenario
        known=type(watts) in (int,float) and math.isfinite(watts) and watts>=0
        return {'known':known,'watts':watts if known else None,'evidence':evidence,
            'scope':'Direct full family/task conditional whole-board training mean; idle residency included; no family pooling or causal attribution'}

    def memory(self,active,aliases,scenario):
        value,evidence=self._conditional(active,aliases)
        field={'mean':'mean_trial_sampled_peak_mib','max':'max_trial_sampled_peak_mib'}.get(scenario)
        peak=value.get(field) if value is not None and field else None
        idle=self.document.get('telemetry',{}).get('raw_idle_sampled_peak_mib')
        valid=lambda v:type(v) in (int,float) and math.isfinite(v) and v>=0
        if value is not None and not value['members'] and peak is None and valid(idle):
            peak=idle
        known=field is not None and valid(peak) and valid(idle)
        if known:peak=max(peak,idle)
        if field is None:evidence='UNKNOWN_TASK_CONDITIONAL_MEMORY_SCENARIO:'+scenario
        return {'known':known,'peak_memory_mib':peak if known else None,'evidence':evidence,
            'scope':'Direct full family/task conditional raw owned trial sampled peak with idle floor; no family pooling; not instantaneous context memory or capacity bound',
            'capacity_bound':False,'safe_factor_applied':False}


def fit_background_collection(raw_path):
    from .background_profiles import map_background_collection
    mapped=map_background_collection(raw_path)
    raw=json.loads(Path(raw_path).read_text())
    context=json.loads(Path(raw['context']['path']).read_text())
    sources=[{'path':str(Path(raw_path).resolve()),'sha256':hashlib.sha256(Path(raw_path).read_bytes()).hexdigest()},
             {'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}]
    return fit_generalized_resources(raw,mapped['traces'],context['residents'],sources=sources)


def fit_from_packet(packet_path):
    packet=json.loads(Path(packet_path).read_text())
    return fit_generalized_resources(packet['raw'],packet['traces'],packet['residents'],sources=packet['sources'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group(required=True);group.add_argument('--input',type=Path);group.add_argument('--raw',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);result=fit_background_collection(args.raw) if args.raw else fit_from_packet(args.input)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
