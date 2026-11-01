"""Fresh CPU enumeration versus a sound bound on a fixed full-flow route domain.

Both searches construct new explicit numerical-v2 FullHorizonModel instances, never consume sealed
predictions, native outcomes, or old objective tables. This verifies a finite
planning domain, not physical stability or deployment-wide optimality.
"""
from __future__ import annotations

import argparse
import collections
import inspect
import json
import math
from pathlib import Path
import statistics
import time

from gsdeploy.forecast_errors import UnknownEvidence
from .h600_event_simulator_v2 import FullHorizonModel
from .flow_validation import _provider, _resources, _score, _sources, _verify, validate_plan, cohort_sha
from .probe_validation import read, receipt, save
from .flow_eligibility import original_rho

TOLERANCE=1e-12
RHO_THRESHOLD=.8


def quality_lower(plan,candidate,quality):
    """Exact training-proxy Q when known, zero otherwise; no forecast needed."""
    by_task=collections.defaultdict(list)
    for request in plan['trace']:
        family=plan['aliases'][candidate['routes'][request['request_id']]]['family']
        value=quality.get(family,{}).get(request['task'])
        if (type(value) not in (int,float) or not math.isfinite(value) or value<0
                or request['task'] in ('text','image') and value>1):
            return {'lower':0.,'exact':False}
        by_task[request['task']].append(value)
    if any(not by_task[t] for t in plan['workload_rps']):return {'lower':0.,'exact':False}
    total=sum(plan['workload_rps'].values())
    return {'lower':sum(rate/total*statistics.mean(by_task[t]) for t,rate in plan['workload_rps'].items()),'exact':True}


def upper_bound(plan,candidate,quality):
    q=quality_lower(plan,candidate,quality)
    historical=candidate.get('predicted_prephase_peak_mib')
    memory=0. if historical is None else plan['memory_safe_factor']*historical/plan['memory_budget_mib']
    # M >= this declared model input, even when it is a training proxy rather
    # than a physical guarantee. All remaining predicted costs are nonnegative.
    return {'U_upper':-plan['weights']['Q']*q['lower']-plan['weights']['M']*memory,
            'Q_lower':q['lower'],'Q_exact_for_training_proxy':q['exact'],'M_lower':memory}


def evaluate(plan,candidate,model,resources,quality,catalog=None):
    """Fresh offered-cohort forecast; original admission stays a separate gate."""
    gate=original_rho(plan,candidate,catalog) if catalog is not None else {
        'status':'UNKNOWN','rho_admission_passed':None,'reason':'NO_SOURCE_BOUND_ORIGINAL_CATALOG'}
    try:
        provider=_provider(model,candidate['execution_context_sha256'])
        resource=_resources(resources,candidate['execution_context_sha256'])
        queues={a:[] for a in plan['aliases']};future=[]
        for request in plan['trace']:
            row={'request_id':request['request_id'],'task':request['task'],
                 'pool_id':request['pool_id'],'arrival_ns':request['offset_ns']}
            if row['arrival_ns']==0:queues[candidate['routes'][row['request_id']]].append(row)
            else:future.append(row)
        end=round(plan['window_seconds']*1e9)
        forecast=FullHorizonModel(plan['aliases'],provider,resource,memory_provider=resource,
            resources=candidate.get('resource_configuration'),max_requests=plan.get('max_forecast_requests',20000)).forecast(
                queues,0,end,future_arrivals=future,
                route_future=lambda row,pending,now:candidate['routes'][row['request_id']],
                require_energy=False,require_memory=False,
                drain_deadline_ns=end+round(plan['drain_seconds']*1e9))
        metric=_score(plan,forecast,quality,startup_joules=candidate.get('predicted_startup_joules'),
                      historical_peak=candidate.get('predicted_prephase_peak_mib'))
        counts=collections.Counter(a['family'] for a in plan['aliases'].values())
        rho={family:sum(r['predicted_remaining_service_ms'] for r in forecast['finishes'].values()
                      if plan['aliases'][r['alias']]['family']==family)/1000/plan['window_seconds']/count
             for family,count in counts.items()}
        comparable=metric['six_term_comparison_complete'] and (metric['absolute_utility_complete'] or bool(plan.get('startup_comparison_policy')))
        value=-metric['cost'] if comparable else None
        constraints={'memory_budget_satisfied':metric['memory_budget_satisfied'],'drain_complete':not forecast['incomplete_ids']}
        model_eligible=all(constraints.values()) if comparable else None
        planning_eligible=(False if model_eligible is False or gate['rho_admission_passed'] is False else
            True if model_eligible is True and gate['rho_admission_passed'] is True else None)
        return {'status':'COMPLETE' if comparable else 'UNKNOWN','comparison_U':value,
            'absolute_U':metric['full_U'],'terms':metric['terms'],'model_eligible':model_eligible,'planning_eligible':planning_eligible,
            'model_constraints':constraints,'finite_forecast_load_proxy_by_family':rho,
            'finite_forecast_load_proxy_below_point8':max(rho.values(),default=0)<=RHO_THRESHOLD+TOLERANCE,
            'finite_forecast_load_proxy_scope':'Accumulated finite forecast active service / window / resident count; diagnostic only, not original planned-rate resolver rho',
            'original_rho':gate,'cohort_size':len(forecast['cohort']),
            'incomplete_ids':forecast['incomplete_ids'],'unknown_reasons':metric['unknown_reasons'],
            'transition_count':forecast['transition_count'],'physical_eligibility':None}
    except (UnknownEvidence,ValueError) as exc:
        return {'status':'UNKNOWN','comparison_U':None,'absolute_U':None,'planning_eligible':None,
                'model_eligible':None,'original_rho':gate,
                'unknown_reasons':[type(exc).__name__+':'+str(exc)],'physical_eligibility':None}


def search(plan,model,resources,quality,*,bounded=False,catalog=None):
    began=time.perf_counter();best=None;records=[];values={};unknown=[];pruned=[];ineligible=[]
    candidates=plan['candidates']
    if bounded:candidates=sorted(candidates,key=lambda c:(-upper_bound(plan,c,quality)['U_upper'],c['id']))
    for candidate in candidates:
        bound=upper_bound(plan,candidate,quality)
        if bounded and best is not None and bound['U_upper']<best-TOLERANCE:
            pruned.append({'candidate_id':candidate['id'],'bound':bound,'incumbent_U':best})
            continue
        result=evaluate(plan,candidate,model,resources,quality,catalog)
        if result['comparison_U'] is not None and result['comparison_U']>bound['U_upper']+TOLERANCE:
            raise AssertionError('Nonnegative six-term upper bound violated')
        if result['model_eligible'] is None:unknown.append(candidate['id'])
        elif not result['model_eligible']:ineligible.append(candidate['id'])
        else:
            values[candidate['id']]=result['comparison_U']
            best=result['comparison_U'] if best is None else max(best,result['comparison_U'])
        records.append({'candidate_id':candidate['id'],'bound':bound,'result':result})
    ties=sorted(cid for cid,value in values.items() if best-value<=TOLERANCE) if best is not None else []
    return {'algorithm':'bounded_leaf_dfs' if bounded else 'plain_leaf_dfs','maximum_model_comparison_U':best,
        'all_tie_ids':ties,'candidate_count':len(candidates),'fresh_forecast_calls':len(records),
        'unknown_ids':sorted(unknown),'model_ineligible_ids':sorted(ineligible),
        'original_rho_admission_unknown_ids':sorted(r['candidate_id'] for r in records if r['result'].get('original_rho',{}).get('rho_admission_passed') is None),
        'bound_pruned':pruned,'records':records,'elapsed_seconds':time.perf_counter()-began,
        'physical_optimality_certified':False}


def run(plan_path,model_path,resources_path,quality_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    plan=validate_plan(read(plan_path));model=read(model_path);resources=read(resources_path);quality_doc=read(quality_path)
    quality=quality_doc.get('loss_by_family_task',quality_doc)
    if quality_doc.get('split','training')!='training':raise ValueError('Training-only quality required')
    if quality_doc.get('execution_context_sha256',plan.get('execution_context_sha256'))!=plan.get('execution_context_sha256'):
        raise ValueError('Training quality execution context mismatch')
    execution=[*plan.get('execution_sources',[]),*[plan[k] for k in ('runner_source','context','profile_plan','catalog') if k in plan]]
    if plan.get('prephase_memory_evidence'):execution.append(plan['prephase_memory_evidence']['source'])
    refs=[receipt(p) for p in (plan_path,model_path,resources_path,quality_path)]
    refs.extend([*_sources(model),*_sources(resources),*quality_doc.get('source_receipts',[]),*execution])
    tool_dir=Path(__file__).parent
    paths={Path(__file__),*[tool_dir/name for name in ('flow_validation.py','flow_metrics.py','probe_validation.py',
        'statistical_profiles.py','background_statistical.py','empirical_resources.py','background_resources.py','flow_eligibility.py',
        'profiling.py','validation.py')]}
    paths.update(Path(inspect.getfile(FullHorizonModel)).parents[1].rglob('*.py'))
    refs.extend(receipt(p) for p in sorted(paths))
    # Check every claimed receipt before deduplication. A current-code receipt
    # must never mask a stale training-source receipt for the same path.
    for ref in refs:_verify(ref)
    refs=list({ref['path']:ref for ref in refs}.values())
    catalog=read(plan['catalog']['path']) if plan.get('catalog') else None
    plain=search(plan,model,resources,quality,catalog=catalog)
    bounded=search(plan,model,resources,quality,bounded=True,catalog=catalog)
    for ref in refs:_verify(ref)
    same=plain['all_tie_ids']==bounded['all_tie_ids'] and plain['maximum_model_comparison_U']==bounded['maximum_model_comparison_U']
    unknown=bool(plain['unknown_ids'] or bounded['unknown_ids'])
    summary={'schema':'gsdeploy.full_flow_same_domain_search_check.v1',
        'status':'UNKNOWN' if unknown else 'PASS' if same else 'FAIL',
        'same_optimum_and_all_ties':same if not unknown else None,'same_search_output':same,
        'same_evaluated_unknown_ids':plain['unknown_ids']==bounded['unknown_ids'],
        'candidate_count':len(plan['candidates']),'request_count':len(plan['trace']),'cohort_sha256':cohort_sha(plan),
        'maximum_model_comparison_U':plain['maximum_model_comparison_U'],'all_tie_ids':plain['all_tie_ids'],
        'all_candidates_known':not plain['unknown_ids'],'no_model_feasible_candidate':not unknown and not plain['all_tie_ids'],
        'original_rho_threshold':RHO_THRESHOLD,'original_rho_gate_required_for_qualified_planning':True,
        'original_rho_admission_unknown_ids':plain['original_rho_admission_unknown_ids'],
        'original_gate_constrained_optimality_certified':False,
        'rho_scope':'Original planned-rate resolver rho recorded separately; finite forecast load proxy is diagnostic and cannot grant original admission',
        'physical_eligibility':None,'physical_optimality_certified':False,'GPU_started':False,'stored_prediction_or_outcome_used':False,
        'domain':'Fixed resident aliases and declared executor, finite plan static route candidates, complete declared arrival/input trace; model memory budget and full drain only, original-rho-constrained optimum not certified',
        'unsearched':['other deployments','replica counts','unlisted static routes','resource allocations','joint batch/cache executors','adaptive future policies'],
        'objective_scope':'Comparison U with common startup omitted only if declared in plan; absolute U may remain UNKNOWN',
        'upper_bound_proof':'U=-sum(w_i*c_i), with nonnegative weights and Q/T/L/P/M/E. Q exact for known training aggregate routes, else lower bound 0. M>=safe*declared_prephase_peak/budget, else lower bound 0. Other costs>=0. Strict U_upper<incumbent-1e-12 pruning preserves all ties within tolerance.',
        'bound_memory_scope':'Declared prephase model input may be a training empirical proxy, not a physical memory guarantee',
        'source_receipts':refs,'plain':{k:v for k,v in plain.items() if k!='records'},
        'bounded':{k:v for k,v in bounded.items() if k!='records'}}
    output.mkdir(parents=True,exist_ok=False)
    for name,value in [('summary',summary),('plain-dfs',plain),('bounded',bounded)]:save(output/(name+'.json'),value)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','model','resources','quality','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();result=run(a.plan,a.model,a.resources,a.quality,a.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_receipts','plain','bounded')}))


if __name__=='__main__':main()
