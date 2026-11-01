"""CPU finite same-predictor-domain DFS check for a declared routing domain.

No old outcome table is searched. Both algorithms invoke the new full-horizon
simulator afresh; the only branch bound uses nonnegative Q/T/L/P/M/E costs.
The deployment/executor are fixed by the explicit request template. This is not
an optimum certificate over unenumerated deployments or physical GPU outcomes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting import FutureWorkload, FullHorizonModel, ProfileProvider, score_forecast
from gsdeploy.resources import identity
from gsdeploy.routing import Router


def candidates(template, shares=(0., .125, .25, .5, .75, .875, 1.)):
    """One currently unstarted text request; all future tasks keep fixed routes."""
    rows = [r for queue in template['queues'].values() for r in queue]
    if len(rows) != 1 or rows[0]['task'] != 'text' or rows[0].get('head_age_ms', 0) or rows[0].get('forecast_cursor'):
        raise ValueError('Exactly one unstarted current text request required')
    aliases = template['aliases']
    specialists = [a for a,s in aliases.items() if s['family'] == 'text']
    generalists = [a for a,s in aliases.items() if s['family'].startswith('generalist') and 'text' in s['supported_tasks']]
    if not specialists or len(generalists) != 1:
        raise ValueError('At least one specialist and exactly one generalist required')
    items = []
    rate = template['future_workload']['rates']['text']
    for current_alias, share in itertools.product(specialists + generalists, shares):
        if type(share) not in (int,float) or not 0 <= share <= 1:
            raise ValueError('Invalid finite routing share')
        routes = {a: rate*(1-share)/len(specialists) for a in specialists if share < 1}
        if share:
            routes[generalists[0]] = rate*share
        items.append({'current_alias':current_alias, 'text_routing_rps':routes})
    return items


def _cohort(template, scenario):
    current = copy.deepcopy(next(r for q in template['queues'].values() for r in q))
    workload = FutureWorkload(**template['future_workload'])
    suffix = workload.suffix(template['phase_start_ns'], template['now_ns'], template['deadline_ns'], scenario_index=scenario)
    return current, suffix


def _quality_lower(template, candidate, scenario):
    """Exact Q for fixed routes without running service/queue simulation."""
    row, suffix = _cohort(template, scenario)
    routing = copy.deepcopy(template['continuation_routing_rps'])
    routing['text'] = candidate['text_routing_rps']
    router = Router(routing)
    bytask = {}
    for request, alias in [(row,candidate['current_alias']), *[(r,router.choose(r['task'])) for r in suffix['arrivals']]]:
        family = template['aliases'][alias]['family']
        quality = template['quality_loss'].get(family,{}).get(request['task'])
        if type(quality) not in (int,float) or not math.isfinite(quality) or quality < 0:
            return 0.
        bytask.setdefault(request['task'],[]).append(quality)
    rates = template['future_workload']['rates']; total=sum(rates.values())
    return sum(rate/total*statistics.mean(bytask[task]) for task,rate in rates.items() if task in bytask)


def upper_bound(template, candidate, scenarios, weights):
    # T,L,P >=0. M >=safe*observed_peak/budget. E >=0.
    # Use no predicted memory/power values or stored objective evaluations.
    historical = template.get('observed_peak_memory_mib')
    memory_floor = 0. if historical is None else template['memory_safe_factor']*historical/template['memory_budget_mib']
    quality = statistics.mean(_quality_lower(template,candidate,s) for s in scenarios)
    return -weights['Q']*quality - weights['M']*memory_floor


def evaluate(template, candidate, scenario, weights):
    provider = ProfileProvider(template['profiles'], template['execution_context_sha256'], allow_cpu_fixture=True)
    current, suffix = _cohort(template,scenario)
    aliases = template['aliases']; queues = {a:[] for a in aliases}
    queues[candidate['current_alias']] = [current]
    routing = copy.deepcopy(template['continuation_routing_rps']); routing['text'] = candidate['text_routing_rps']
    router = Router(routing)
    try:
        forecast = FullHorizonModel(aliases,provider,resources=template['resources']).forecast(
            queues,template['now_ns'],template['deadline_ns'],template.get('service_scenario','mean'),
            future_arrivals=suffix['arrivals'],route_future=lambda r,q,now:router.choose(r['task']),
            admission_state=template.get('admission_state'), drain_deadline_ns=template['drain_deadline_ns'])
        metric = score_forecast(forecast,template.get('completed',[]),aliases,template['future_workload']['rates'],
            template['latency_reference_ms'],template['quality_loss'],phase_start_ns=template['phase_start_ns'],
            now_ns=template['now_ns'],deadline_ns=template['deadline_ns'],memory_budget_mib=template['memory_budget_mib'],
            energy_reference_watts=template['energy_reference_watts'],memory_safe_factor=template['memory_safe_factor'],
            observed_peak_memory_mib=template.get('observed_peak_memory_mib'),
            observed_board_energy_joules=template.get('observed_board_energy_joules'),weights=weights)
        if not metric['six_term_comparison_complete'] or metric['full_U'] is None:
            return {'status':'UNKNOWN','U':None,'eligible':None,'reason':metric['unknown_reasons']}
        # Preserve original family-average rho planning threshold; no native
        # stationarity/stability evidence is manufactured from this simulation.
        horizon=(template['deadline_ns']-template['phase_start_ns'])/1e9
        counts={family:sum(s['family']==family for s in aliases.values()) for family in {s['family'] for s in aliases.values()}}
        rho={family:sum(r['predicted_remaining_service_ms'] for r in forecast['finishes'].values() if aliases[r['alias']]['family']==family)/1000/horizon/n for family,n in counts.items()}
        eligible=metric['memory_budget_satisfied'] and max(rho.values(),default=0)<=.8+1e-12 and not forecast['incomplete_ids']
        return {'status':'COMPLETE','U':metric['full_U'],'eligible':bool(eligible),'terms':metric['terms'],
                'rho_by_family':rho,'cohort_sha256':suffix['cohort_sha256'],
                'physical_stability':'UNKNOWN_NOT_NATIVE_EVIDENCE','transition_count':forecast['transition_count']}
    except UnknownEvidence as exc:
        return {'status':'UNKNOWN','U':None,'eligible':None,'reason':str(exc)}


def run_search(template, items, scenarios, weights, *, bounded=False):
    start=time.perf_counter(); best=None; values={}; records=[]; calls=0; unknown=[]; pruned=[]
    ordered=sorted(items,key=lambda c:(-upper_bound(template,c,scenarios,weights),identity(c))) if bounded else items
    for candidate in ordered:
        cid=identity(candidate); upper=upper_bound(template,candidate,scenarios,weights)
        if bounded and best is not None and upper < best-1e-12:
            pruned.append(cid); continue
        runs=[evaluate(template,candidate,s,weights) for s in scenarios];calls+=len(runs)
        if any(r['eligible'] is None for r in runs):
            unknown.append(cid); value=None
        elif not all(r['eligible'] for r in runs):
            value=None
        else:
            value=statistics.mean(r['U'] for r in runs)
            if value>upper+1e-12:
                raise AssertionError('Proved objective upper bound violated')
            values[cid]=value;best=value if best is None else max(best,value)
        records.append({'candidate_id':cid,'candidate':candidate,'runs':runs,'objective':value,'upper':upper})
    ties=sorted(cid for cid,value in values.items() if best-value<=1e-12) if best is not None else []
    return {'algorithm':'bounded' if bounded else 'plain_dfs','exact_max_objective':best,'tie_ids':ties,
            'winner_id':ties[0] if ties else None,'candidate_count':len(items),'evaluator_calls':calls,
            'bound_pruned_ids':pruned,'unknown_ids':unknown,'elapsed_seconds':time.perf_counter()-start,
            'records':records,'whole_declared_cpu_domain_certified':best is not None and not unknown}


def run(template_path, objective_path, output, *, scenarios=(0,1,2)):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    template=json.loads(Path(template_path).read_text()); objective=json.loads(Path(objective_path).read_text())
    weights=objective['weights']
    if set(weights)!=set('QTLPME') or any(abs(v-1/6)>1e-12 for v in weights.values()):
        raise ValueError('Frozen equal six-term objective required')
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError('Distinct nonempty future scenario ids required')
    items=candidates(template)
    dfs=run_search(template,items,scenarios,weights)
    bounded=run_search(template,items,scenarios,weights,bounded=True)
    certified=dfs['whole_declared_cpu_domain_certified'] and bounded['whole_declared_cpu_domain_certified']
    match=certified and dfs['tie_ids']==bounded['tie_ids'] and dfs['exact_max_objective']==bounded['exact_max_objective']
    summary={'schema':'gsdeploy.full_horizon_same_domain_search_check.v1','status':('PASS' if match else 'FAIL') if certified else 'UNKNOWN',
             'same_optimum_and_all_ties':match,'candidate_count':len(items),'scenarios':list(scenarios),
             'objective':objective,'memory_budget_mib':template['memory_budget_mib'],
             'rho_threshold':.8,'stability_scope':'family-average CPU planning rho; original native stability UNKNOWN',
             'profile_kind':template['profiles']['provenance']['kind'],'GPU_started':False,
             'physical_optimality_certified':False,
             'domain':'Fixed resident/executor and one current-route x finite static continuation text routing shares',
             'unsearched':['alternative deployments','replica counts','resource allocations','batch/merged weight executors','adaptive future policies'],
             'upper_bound_proof':'U=-(Q+T+L+P+M+E)/6; costs nonnegative, Q exact from known candidate routing, M>=safe*observed_peak/budget; others lower bounded by zero. Strict pruning preserves ties.',
             'input_receipts':[{'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in (template_path,objective_path,__file__)],
             'dfs':{k:v for k,v in dfs.items() if k!='records'},'bounded':{k:v for k,v in bounded.items() if k!='records'}}
    output.mkdir(parents=True)
    for name,value in [('summary',summary),('plain-dfs',dfs),('bounded',bounded)]:
        with (output/(name+'.json')).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--objective',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=run(args.template,args.objective,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in {'objective','input_receipts','dfs','bounded'}}))
