"""Fit expiry, yield, isolated dispatch and contention-entry residual separately."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import json
from pathlib import Path
import sqlite3
import statistics

from .context_envelope_diagnostic_v8 import IntervalUnion,context_intervals
from .dag_availability_fit_v8 import derive as base_derive,estimate,stable_key
from .native_collect import receipt,save


def derive(paths):
    base=base_derive(paths);raw_ids=[];sources=[*base['sources'],receipt(__file__),receipt(Path(__file__).with_name('trained_dag_availability_v8.py'))]
    budgets={(g['owner'],stable_key(g['frontier'])):g for g in base['grant_budget_groups']}
    transition=defaultdict(lambda:defaultdict(Counter));gap_values=defaultdict(lambda:defaultdict(list))
    solo_values=defaultdict(lambda:defaultdict(list));initial_solo_values=defaultdict(lambda:defaultdict(list))
    entry_values=defaultdict(lambda:defaultdict(list));entry_gap=defaultdict(lambda:defaultdict(list));counts=Counter()
    for path in paths:
        d=json.loads(Path(path).read_text());rawref=next(s for s in d['sources'] if Path(s['path']).name=='raw.json');raw_id=rawref['sha256'];raw_ids.append(raw_id)
        raw=json.loads(Path(rawref['path']).read_text());rows={r['request_id']:r for r in raw['request_records']}
        observed=json.loads(Path(d['kernel_ready_observations']['path']).read_text())['rows'];intervals=defaultdict(list);kernels=defaultdict(list)
        for r in observed:
            ready=r['certified_ready_bound_ns'];start=r['observed_start_ns'];lo=min(ready,start) if ready is not None else start
            intervals[r['request_id'],r['phase']].append((lo,r['observed_end_ns']));kernels[r['request_id']].append((start,r['observed_end_ns']))
        indexes={k:IntervalUnion(v) for k,v in intervals.items()};jobs=defaultdict(list)
        kernel_unions={rid:IntervalUnion(v) for rid,v in kernels.items()}
        kernel_starts={rid:sorted(a for a,b in v) for rid,v in kernels.items()}
        for r in rows.values():jobs[r['profile_job_id']].append(r)
        sql=next(s for s in d['sources'] if Path(s['path']).suffix=='.sqlite')
        c=sqlite3.connect('file:'+sql['path']+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;ev,brackets,issues=context_intervals(c);c.close()
        mapping={w['alias']:tuple(w['context']) for w in d['controlled_temporal_attribution']['creation_windows']}
        all_brackets=sorted((a,b,alias) for alias,k in mapping.items() for a,b in brackets[k]);starts=[a for a,b,p in all_brackets]
        issue_times=sorted(p['timestamp'] for p in issues)
        def context_at(t):
            i=bisect_right(starts,t)-1
            return all_brackets[i] if i>=0 and t<all_brackets[i][1] else None
        for jobid,requests in jobs.items():
            members={r['model']:r for r in requests};local={p:{phase:ix for (rid,phase),ix in indexes.items() if rid==r['request_id']} for p,r in members.items()}
            pending_unions={p:IntervalUnion(interval for ix in ixes.values() for interval in ix.intervals) for p,ixes in local.items()}
            cuts=sorted({t for ixes in local.values() for ix in ixes.values() for a,b in ix.intervals for t in (a,b)})
            def state_at(t):
                result=[]
                for owner,row in sorted(members.items()):
                    phases=[]
                    for phase,ix in local[owner].items():
                        i=bisect_right(ix.starts,t)-1
                        if i>=0 and t<ix.ends[i]:phases.append(phase)
                    if phases:result.append([owner,row['task'],row['pool_id'],sorted(phases)])
                return stable_key(result)
            def covered(owner,a,b):
                return pending_unions[owner].overlap(a,b)==b-a
            previous=();initial=True
            for ci,t in enumerate(cuts):
                state=state_at(t);owners={v[0] for v in state};old_owners={v[0] for v in previous}
                if not state:previous=state;continue
                if len(state)==1:
                    owner=state[0][0];rid=members[owner]['request_id']
                    started=kernel_unions[rid].overlap(t,t+1)>0
                    future_index=bisect_right(kernel_starts[rid],t-1)
                    begin=t if started else kernel_starts[rid][future_index] if future_index<len(kernel_starts[rid]) else None
                    end=cuts[ci+1] if ci+1<len(cuts) else None
                    if begin is not None and (end is None or begin<end):
                        solo_values[state][raw_id].append(begin-t)
                        if initial:initial_solo_values[state][raw_id].append(begin-t)
                # Entry is learned from the remaining observed grant at this
                # exact arrival of competition, never another complete budget.
                if len(state)>=2 and len(previous)<2:
                    held=context_at(t)
                    if held and held[2] in owners:
                        a,b,owner=held
                        if old_owners and owner not in old_owners:
                            counts['entry_observed_context_not_previous_solo_owner']+=1
                        else:
                            reason='initial' if initial else 'contention_entry'
                            current=None if reason=='initial' else owner
                            entry_values[current,reason,state,owner][raw_id].append(b-t)
                            entry_gap[current,reason,state,owner][raw_id].append(0)
                initial=False;previous=state
            lo,hi=cuts[0],cuts[-1]
            for a,b,owner in all_brackets:
                if owner not in members or not lo<=b<hi:continue
                state=state_at(b);owners={x[0] for x in state}
                if not owners:continue
                reason='grant_expired' if owner in owners else 'yield'
                i=bisect_right(starts,b-1);target_interval=None
                for na,nb,target in all_brackets[i:]:
                    if na>hi:break
                    if target in owners:
                        target_interval=na,nb,target;break
                if target_interval is None:continue
                na,nb,target=target_interval
                if na<b or bisect_right(issue_times,na)-bisect_right(issue_times,b-1)>0:continue
                if not covered(target,b,na):continue
                transition[owner,reason,state][raw_id][target]+=1
                gap_values[owner,reason,state,target][raw_id].append(na-b)
    cells=[];deferred=[];solo=[];initial_solo=[];entry_groups=[];choice_groups=[]
    for name,table,target in [('uncontended',solo_values,solo),('initial_uncontended',initial_solo_values,initial_solo)]:
        for state,captures in sorted(table.items()):
            fitted=estimate(captures,raw_ids)
            if fitted is None:continue
            target.append(dict(frontier=state,dispatch_wait_ns=fitted['value_ns'],training_estimate=fitted))
    for (current,reason,state,target),captures in sorted(entry_values.items(),key=lambda v:str(v[0])):
        fitted=estimate(captures,raw_ids);gap=estimate(entry_gap[current,reason,state,target],raw_ids)
        if fitted is None or gap is None:continue
        cells.append(dict(current_owner=current,reason=reason,frontier=state,next_owner=target,
            grant_budget_ns=max(1,fitted['value_ns']),handoff_ns=gap['value_ns']))
        entry_groups.append(dict(current_owner=current,reason=reason,frontier=state,target=target,residual_grant=fitted,entry_wait=gap))
    for (owner,reason,state),captures in sorted(transition.items(),key=lambda v:str(v[0])):
        if set(captures)!=set(raw_ids):
            deferred.append(dict(current_owner=owner,reason=reason,frontier=state,missing='two independent transition captures'));continue
        targets=sorted({v for c in captures.values() for v in c});prob={v:statistics.mean(captures[r][v]/sum(captures[r].values()) for r in raw_ids) for v in targets};target=max(targets,key=lambda v:(prob[v],v))
        fitted_gap=estimate(gap_values[owner,reason,state,target],raw_ids);budget=budgets.get((target,state))
        choice_groups.append(dict(current_owner=owner,reason=reason,frontier=state,predicted_next_owner=target,capture_balanced_probabilities=prob,availability_wait=fitted_gap))
        if len(state)==1:
            # Executed by the explicit uncontended component; no fake quantum.
            continue
        if fitted_gap is None or budget is None:
            deferred.append(dict(current_owner=owner,reason=reason,frontier=state,target=target,missing='target grant budget' if budget is None else 'two-capture availability wait'));continue
        cells.append(dict(current_owner=owner,reason=reason,frontier=state,next_owner=target,
            grant_budget_ns=budget['value_ns'],handoff_ns=fitted_gap['value_ns']))
    for source in sources:
        if receipt(source['path'])!=source:raise ValueError('Stateful availability training source changed')
    return dict(schema='gsdeploy.trained_DAG_availability_model.v8',status='TRAINED_EXECUTABLE_AVAILABILITY_HYPOTHESIS',sources=sources,
        source_DAG_reports=base['source_DAG_reports'],grant_budget_groups=base['grant_budget_groups'],
        initial_and_contention_residual_groups=entry_groups,transition_choice_groups=choice_groups,deferred_cells=deferred,
        policy=dict(schema='gsdeploy.trained_DAG_availability_policy.v8',training_only=True,
            reference_work_domain='SOLO_ONLY_REFERENCE_NS',cells=cells,uncontended_cells=solo,initial_uncontended_cells=initial_solo),
        counts=dict(counts),independent_prediction_accuracy_validated=False,
        physical_SM_service_identified=False,natural_profiler_transfer_qualified=False,
        whole_corun_multiplier_used=False,unknown_cell_fallback=False,GPU_started=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',action='append',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);m=derive(a.report);save(out/'model.json',m)
    print(json.dumps(dict(status=m['status'],transition_cells=len(m['policy']['cells']),solo_cells=len(m['policy']['uncontended_cells']),initial_solo_cells=len(m['policy']['initial_uncontended_cells']),residual_groups=len(m['initial_and_contention_residual_groups']),deferred_cells=len(m['deferred_cells']))))


if __name__=='__main__':main()
