"""Capture-balanced state-conditional availability policy from source DAG queues.

Budgets, next-owner labels and non-idle availability gaps are fitted separately.
No interpolation to unsupported exact input/frontier cells is allowed. A policy
with missing initial/handoff/target-budget support cannot silently forecast.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import copy
import json
from pathlib import Path
import statistics

from .context_envelope_diagnostic_v8 import IntervalUnion
from .conditional_availability_scheduler_v8 import cell_key
from .native_collect import receipt,save


def stable_key(frontier):
    return tuple((x[0],x[1],x[2],tuple(x[3])) for x in frontier)


def estimate(captures,raw_ids):
    if set(captures)!=set(raw_ids):return None
    medians={r:statistics.median(v) for r,v in captures.items()}
    checks=[]
    for held in raw_ids:
        train=next(r for r in raw_ids if r!=held);v=captures[held];pred=medians[train]
        checks.append(dict(train_raw_id=train,held_out_training_raw_id=held,
            sample_count=len(v),predicted_ns=pred,
            mean_absolute_relative_error=statistics.mean(abs(pred-x)/max(1,x) for x in v),
            median_absolute_error_ns=abs(pred-medians[held])))
    return dict(value_ns=round(statistics.mean(medians.values())),
        per_capture=[dict(raw_id=r,count=len(captures[r]),median_ns=medians[r],min_ns=min(captures[r]),max_ns=max(captures[r])) for r in raw_ids],
        leave_one_capture_out_training_checks=checks)


def derive(report_paths):
    if len(report_paths)!=2:raise ValueError('Two independent source-DAG training reports required')
    refs=[];sources=[receipt(__file__),receipt(Path(__file__).with_name('conditional_availability_scheduler_v8.py'))]
    datasets=[];raw_ids=[]
    for path in report_paths:
        ref=receipt(path);d=json.loads(Path(path).read_text())
        if d.get('schema')!='gsdeploy.recorded_DAG_pending_frontier.v8' or d.get('status')!='SOURCE_DAG_PENDING_FRONTIER_EXTRACTED':
            raise ValueError('Qualified source-DAG pending report required')
        for s in d['sources']:
            if receipt(s['path'])!=s:raise ValueError('Changed source-DAG evidence')
        if receipt(d['kernel_ready_observations']['path'])!=d['kernel_ready_observations']:
            raise ValueError('Changed exact kernel ready observations')
        rawref=next(s for s in d['sources'] if Path(s['path']).name=='raw.json')
        raw=json.loads(Path(rawref['path']).read_text());raw_id=rawref['sha256'];raw_ids.append(raw_id)
        refs.append(ref);sources.extend([ref,*d['sources'],d['kernel_ready_observations']])
        datasets.append((raw_id,d,raw))
    if len(set(raw_ids))!=2:raise ValueError('Independent actual raw captures required')
    budgets=defaultdict(lambda:defaultdict(list));gaps=defaultdict(lambda:defaultdict(list));choices=defaultdict(lambda:defaultdict(Counter))
    starts=defaultdict(lambda:defaultdict(Counter));censored=Counter()
    for raw_id,d,raw in datasets:
        rows={r['request_id']:r for r in raw['request_records']};jobs=defaultdict(list)
        for r in rows.values():jobs[r['profile_job_id']].append(r)
        observed=json.loads(Path(d['kernel_ready_observations']['path']).read_text());ranges=defaultdict(list)
        for r in observed['rows']:
            ready=r['certified_ready_bound_ns'];lo=min(ready,r['observed_start_ns']) if ready is not None else r['observed_start_ns']
            ranges[r['request_id'],r['phase']].append((lo,r['observed_end_ns']))
        indexes={k:IntervalUnion(v) for k,v in ranges.items()}
        for sample in d['samples']:
            if not sample['eligible_for_conditional_surrogate_budget']:continue
            a,b=sample['envelope_start_ns'],sample['envelope_end_ns'];state=[];changed=False;continued=False
            for row in jobs[sample['job_id']]:
                active=[]
                for (rid,phase),index in indexes.items():
                    if rid!=row['request_id']:continue
                    overlap=index.overlap(a,b)
                    if 0<overlap<b-a:changed=True
                    if overlap==b-a:
                        active.append(phase)
                        if row['model']==sample['alias']:
                            i=bisect_right(index.starts,b)-1
                            continued=continued or i>=0 and b<index.ends[i]
                if active:state.append([row['model'],row['task'],row['pool_id'],sorted(active)])
            if changed or not continued:
                censored['frontier_changed_or_owner_emptied_at_save']+=1;continue
            key=sample['alias'],stable_key(sorted(state))
            budgets[key][raw_id].append(sample['envelope_length_ns'])
        for handoff in d['handoff_samples']:
            state=stable_key(handoff['frontier']);owner=handoff['current_owner'];target=handoff['next_owner']
            if not state or target not in {v[0] for v in state}:
                censored['handoff_target_not_DAG_ready_at_save']+=1;continue
            reason='grant_expired' if owner in {v[0] for v in state} else 'yield'
            key=owner,reason,state
            choices[key][raw_id][target]+=1;gaps[owner,target,state][raw_id].append(handoff['availability_gap_ns'])
        for initial in d['initial_context_observations']:
            state=stable_key(initial['frontier']);target=initial['observed_context']
            if target not in {v[0] for v in state}:
                censored['initial_held_context_not_ready']+=1;continue
            starts[state][raw_id][target]+=1
    budget_groups=[];budget_index={};unsupported=[]
    for (owner,state),captures in sorted(budgets.items()):
        fitted=estimate(captures,raw_ids)
        if fitted is None:
            unsupported.append(dict(kind='grant_budget',owner=owner,frontier=state,reason='TWO_CAPTURE_SUPPORT_REQUIRED'));continue
        row=dict(owner=owner,frontier=state,**fitted);budget_groups.append(row);budget_index[owner,state]=row
    gap_groups=[];gap_index={}
    for (owner,target,state),captures in sorted(gaps.items()):
        fitted=estimate(captures,raw_ids)
        if fitted is None:
            unsupported.append(dict(kind='nonidle_availability_gap',owner=owner,target=target,frontier=state,reason='TWO_CAPTURE_SUPPORT_REQUIRED'));continue
        row=dict(owner=owner,target=target,frontier=state,**fitted);gap_groups.append(row);gap_index[owner,target,state]=row
    cells=[];categorical=[];deferred=[]
    for (owner,reason,state),counts in sorted(choices.items()):
        if set(counts)!=set(raw_ids):
            deferred.append(dict(current_owner=owner,reason=reason,frontier=state,missing='two-run next-owner support'));continue
        targets=sorted({t for c in counts.values() for t in c});weights={t:statistics.mean(counts[r][t]/sum(counts[r].values()) for r in raw_ids) for t in targets}
        target=max(targets,key=lambda t:(weights[t],t))
        categorical.append(dict(current_owner=owner,reason=reason,frontier=state,chosen_next_owner=target,
            capture_balanced_probabilities=weights,per_capture=[dict(raw_id=r,counts=dict(counts[r])) for r in raw_ids],
            deterministic_choice_is_prediction_hypothesis=True))
        budget=budget_index.get((target,state));gap=gap_index.get((owner,target,state))
        if budget is None or gap is None:
            deferred.append(dict(current_owner=owner,reason=reason,frontier=state,target=target,missing='target budget' if budget is None else 'nonidle gap'));continue
        cells.append(dict(current_owner=owner,reason=reason,frontier=state,next_owner=target,
            grant_budget_ns=budget['value_ns'],handoff_ns=gap['value_ns']))
    initial_groups=[]
    for state,counts in sorted(starts.items()):
        if set(counts)!=set(raw_ids):continue
        targets=sorted({t for c in counts.values() for t in c});weights={t:statistics.mean(counts[r][t]/sum(counts[r].values()) for r in raw_ids) for t in targets}
        target=max(targets,key=lambda t:(weights[t],t))
        initial_groups.append(dict(frontier=state,observed_held_context=target,capture_balanced_probabilities=weights,
            handoff_or_initial_wait_fitted=False))
    for s in sources:
        if receipt(s['path'])!=s:raise ValueError('Availability fit source changed')
    return dict(schema='gsdeploy.DAG_state_conditional_availability_fit.v8',status='TRAINED_CONDITIONAL_AVAILABILITY_COMPONENTS',
        sources=sources,source_DAG_reports=refs,training_only=True,reference_work_domain='SOLO_ONLY_REFERENCE_NS',
        grant_budget_groups=budget_groups,nonidle_availability_gap_groups=gap_groups,
        next_owner_groups=categorical,initial_held_context_groups=initial_groups,
        stateful_policy=dict(schema='gsdeploy.conditional_context_availability_policy.v8',training_only=True,
            reference_work_domain='SOLO_ONLY_REFERENCE_NS',whole_corun_multiplier_used=False,cells=cells),
        deferred_policy_cells=deferred,unsupported_component_cells=unsupported,censored_reasons=dict(censored),
        grant_budget_group_count=len(budget_groups),nonidle_gap_group_count=len(gap_groups),numeric_transition_cell_count=len(cells),
        initial_context_wait_available=False,solo_uncontended_policy_available=False,complete_full_horizon_policy_available=False,
        leave_one_capture_out_is_training_diagnostic=True,independent_prediction_validation=False,
        natural_profiler_timing_transfer_qualified=False,physical_quantum_identified=False,physical_SM_service_identified=False,
        unknown_cell_fallback=False,validation_labels_used=False,GPU_started=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',action='append',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);model=derive(a.report);save(out/'model.json',model)
    print(json.dumps({k:model[k] for k in ('status','grant_budget_group_count','nonidle_gap_group_count','numeric_transition_cell_count','censored_reasons')}))


if __name__=='__main__':main()
