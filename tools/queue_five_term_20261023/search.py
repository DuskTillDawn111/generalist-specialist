"""New five-term policy simulations and finite DFS verification, never rescoring.

The frozen event environment and generic exhaustive/quotient traversal are
reused, but every visited leaf runs the new policy with Q/T/L/M/E=.2 and P=0.
Missing controlled background evidence remains UNKNOWN. Text-only diagnostics
cannot select the original mixed-workload policy or certify a GPU optimum.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.queue_coupled_utility_20261023.model import UnknownEvidence, finite
from tools.queue_coupled_utility_20261023.resources import load_provider, ResourceModel, key
from tools.queue_coupled_utility_20261023.search import SharedRateEnvironment, execute_search as frozen_search, identity
from tools.queue_five_term_20261023.policy import CoupledUtilityPolicy, FIXED_WEIGHTS, OBJECTIVE_SCHEMA, percentile, validate_weights
from tools.spillover_capacity_20261022.exposure import frozen_api

CPU_SEEDS = (413, 414)
PRIMARY = {'text':2, 'generalist_2':1, 'ocr':1}
DEFAULT_PARAMETERS = dict(allow_spill=True, scenarios=['mean','p90'], utility_margin=0., cooldown_ms=30000.)
TERMS = 'QTLME'


def read(value):
    return copy.deepcopy(value) if isinstance(value, dict) else json.loads(Path(value).read_text())


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def candidates(deployment=PRIMARY):
    for spill, scenarios, margin, cooldown in itertools.product(
        (False,True), (['mean'],['mean','p90']), (0.,.0005,.002), (0.,30000.)):
        yield dict(objective_schema=OBJECTIVE_SCHEMA, weights=dict(FIXED_WEIGHTS), deployment=copy.deepcopy(deployment),
                   allow_spill=spill, scenarios=list(scenarios), utility_margin=margin, cooldown_ms=cooldown)


def search(items, seeds, evaluate, *, optimized=False):
    for candidate in items:
        if candidate.get('objective_schema') != OBJECTIVE_SCHEMA:
            raise ValueError('Explicit new online objective identity required')
        validate_weights(candidate.get('weights'))
    return frozen_search(items, seeds, evaluate, optimized=optimized)


def coverage_audit(controlled, historical=None):
    """Inventory evidence without fitting, interpolation or provider substitution."""
    controlled_background = []
    for task, alias in (('image','generalist_2#0'),('asr','generalist_2#0'),('ocr','ocr#0')):
        controlled_background.append(dict(task=task, alias=alias, controlled_service_supported=False,
            controlled_shared_power_supported=False, reason='ControlledProvider explicitly rejects any non-text active task'))
    result = dict(status='FULL_CONTROLLED_WORKLOAD_UNIDENTIFIED', full_workload_candidate_selection_supported=False,
        controlled_schema=controlled.get('schema'), controlled_training_seeds=controlled.get('training_seeds'),
        controlled_background=controlled_background, interpolation_used=False, hybrid_provider_used=False,
        old_runtime_outcome_table_used=False, missing_is_infeasible=False,
        conclusion_zh='现有controlled模型只校准text任务。改变效用权重不能补齐背景共享速率与整板功率，因此完整混合负载保留UNKNOWN。',
        additional_evidence_needed=['Exact PRIMARY input-specific background baselines and supported actual-overlap progress states',
            'Text slowdown while image/asr/ocr execute, with visible-start/history semantics and held-out validation',
            'Background-aware whole-board power evidence and sufficient supported concurrency coverage'])
    if historical is not None:
        model = ResourceModel(historical)
        group = historical['deployments'].get(key(dict(sorted(PRIMARY.items()))))
        if group:
            backgrounds = []
            for encoded, data in group['service_strata']['task_context'].items():
                family, task, context = json.loads(encoded)
                if task != 'text':
                    backgrounds.append(dict(family=family, task=task, other_active_counts=context,
                        observations=len(data['values_ms']), seeds=data['seeds']))
            power = []
            for counts in itertools.product(range(2),range(2),range(3)):
                active = dict(zip(('generalist_2','ocr','text'),counts))
                prediction = model.predict_power(PRIMARY,active)
                power.append(dict(active_counts=active, known=prediction['watts'] is not None,
                                  status=prediction['status']))
            missing_service = []
            for family,task in (('generalist_2','image'),('generalist_2','asr'),('ocr','ocr')):
                for other_g,other_o,other_t in itertools.product(range(1 if family=='generalist_2' else 2),
                    range(1 if family=='ocr' else 2),range(3)):
                    context=dict(generalist_2=other_g,ocr=other_o,text=other_t)
                    prediction=model.predict_service(PRIMARY,family,task,'__coverage_inventory_not_a_real_input__',context)
                    if prediction['mean_ms'] is None:
                        missing_service.append(dict(family=family,task=task,other_active_counts=context,status=prediction['status']))
            result['historical_observational_inventory'] = dict(
                training_seeds=historical['training_seeds'], training_runs=len(group['runs']),
                background_task_contexts=backgrounds, power_states=power, missing_service_contexts=missing_service,
                context_semantics=historical['context_semantics'], causal_externality_identified=False,
                may_fill_controlled_actual_overlap_gap=False,
                reason_zh='历史记录含背景观测，但按host-pending分组且缺少部分task-context；不是actual-overlap的动态progress校准。不能直接拼接成完整controlled反事实环境。')
    return result


def evaluate_cohort(contract, resources, cohort, candidate, *, text_only=False, provider=None, memory_norm=None):
    validate_weights(contract['weights'])
    validate_weights(candidate['weights'])
    if candidate['objective_schema'] != OBJECTIVE_SCHEMA:
        raise ValueError('New five-term candidate required')
    if memory_norm is not None and (not finite(memory_norm) or memory_norm < 0):
        raise ValueError('Nonnegative conditional memory proxy required')
    provider = provider or load_provider(resources,candidate['deployment'])
    aliases=contract['aliases']
    workload={'text':contract['text_rate'], **{t:sum(v.values()) for t,v in contract.get('background_routing',{}).items()}}
    if text_only:
        workload={'text':workload['text']}
    horizon_ns=round(contract['horizon_seconds']*1e9)
    environment=SharedRateEnvironment(aliases,provider,horizon_ns)
    policy=CoupledUtilityPolicy(aliases,provider,contract['quality_loss'],dict(FIXED_WEIGHTS),contract['latency_refs_ms'],
        workload,scenarios=candidate['scenarios'],allow_spill=candidate['allow_spill'],utility_margin=candidate['utility_margin'],
        cooldown_ms=candidate['cooldown_ms'],exposure_rate_rps=contract['text_rate']/8,exposure_burst=1.,
        energy_reference_watts=450.,drift_ratio_limit=2.,maximum_pending_for_forecast=32)
    policy.start_phase(0,horizon_ns)
    api=frozen_api()
    _,routes=api['trace'].expand_routes(candidate['deployment'],contract['background_routing'])
    router=api['routing'].Router(routes)
    decisions=[]
    try:
        for request in cohort:
            if text_only and request['task']!='text':
                continue
            environment.advance(request['arrival_ns'])
            policy.observe_completed(environment.newly_completed)
            environment.newly_completed.clear()
            if request['task']=='text':
                policy.set_arrival(request['request_id'],request['arrival_ns'])
                result=policy.choose('text',request['pool_id'],request['arrival_ns'],environment.snapshot())
                alias=result['alias']
                decisions.append(dict(request_id=request['request_id'],alias=alias,spill=result['actual_spill_selected'],status=result['status']))
            else:
                alias=router.choose(request['task'])
            policy.observe_admission(dict(request,alias=alias))
            environment.enqueue(request,alias)
        environment.advance(horizon_ns)
        environment.drain()
        rows=list(environment.results.values())
        if not rows:
            raise UnknownEvidence('EMPTY_COHORT')
        terms=dict(Q=0.,T=1-sum(r['finish_ns']<=horizon_ns for r in rows)/len(rows),L=0.,P=0.,M=memory_norm,
            E=environment.energy_joules/(450.*contract['horizon_seconds']))
        metrics={}
        for task,rate in workload.items():
            task_rows=[r for r in rows if r['task']==task]
            if not task_rows:
                raise UnknownEvidence('MISSING_TASK_COHORT:'+task)
            lat=[(r['finish_ns']-r['arrival_ns'])/1e6 for r in task_rows]
            losses=[contract['quality_loss'][r['family']][task] for r in task_rows]
            if any(not finite(v) or v<0 for v in losses):
                raise UnknownEvidence('UNKNOWN_AGGREGATE_QUALITY:'+task)
            alpha=rate/sum(workload.values())
            terms['Q']+=alpha*statistics.mean(losses)
            terms['L']+=alpha*statistics.mean(lat)/contract['latency_refs_ms'][task]
            terms['P']+=alpha*percentile(lat)/contract['latency_refs_ms'][task]
            metrics[task]=dict(count=len(lat),mean_ms=statistics.mean(lat),p95_ms=percentile(lat))
        weighted=[t for t in TERMS if terms[t] is not None]
        value=-sum(FIXED_WEIGHTS[t]*terms[t] for t in weighted)
        return dict(status='COMPLETE_FIVE_TERM_CONDITIONAL_CPU_PROXY',U=value,eligible=True,
            objective_schema=OBJECTIVE_SCHEMA,weights=dict(FIXED_WEIGHTS),weighted_terms=weighted,terms=terms,
            P_evaluation_role='diagnostic_only',P_excluded_from_online_decision=True,
            proxy_objective_complete=all(terms[t] is not None for t in TERMS),physical_U=None,formal_U=None,
            original_stability='UNKNOWN_NOT_A_NATIVE_MEASUREMENT',quality_scope='Frozen family/task aggregate loss proxy',
            memory_scope='Empirical calibration resident peak constant; actual routing peak UNKNOWN' if memory_norm is not None else 'M_UNKNOWN_partial_objective',
            energy_scope='One whole-board measurement-window integral; startup omitted and physical counterfactual UNKNOWN',
            task_metrics=metrics,request_count=len(rows),spill_count=sum(d['spill'] for d in decisions),
            routes=dict(Counter(r['alias'] for r in rows)),decision_statuses=dict(Counter(d['status'] for d in decisions)),
            route_trace_sha256=identity([(r['request_id'],r['alias']) for r in rows]),
            completion_trace_sha256=identity([(r['request_id'],r['finish_ns']) for r in rows]),
            events=environment.event_count,active_rate_recomputations=environment.rate_changes,
            drain_finish_ns=environment.clock,conditional_proxy_only=True,future_arrivals_visible_to_policy=False,
            actual_new_policy_simulated=True,old_score_table_reused=False)
    except UnknownEvidence as exc:
        return dict(status='UNKNOWN',reason=str(exc),U=None,eligible=None,physical_U=None,formal_U=None,
            objective_schema=OBJECTIVE_SCHEMA,weights=dict(FIXED_WEIGHTS),P_evaluation_role='diagnostic_only',
            observed_requests=len(environment.results),events=environment.event_count,
            unsupported_not_infeasible=True,conditional_proxy_only=True,future_arrivals_visible_to_policy=False,
            actual_new_policy_simulated=True,old_score_table_reused=False)


def compare_domain(items,seeds,evaluate):
    dfs=search(items,seeds,evaluate)
    optimized=search(items,seeds,evaluate,optimized=True)
    reference={r['candidate_id']:r for r in dfs['records']}
    checks=[]
    for row in optimized['records']:
        other=reference[row['candidate_id']]
        if row['status']!=other['status'] or row['objective']!=other['objective'] or row['runs']!=other['runs']:
            raise ValueError('Independent exhaustive/quotient full-run mismatch:'+row['candidate_id'])
        checks.append(dict(candidate_id=row['candidate_id'],all_run_fields_identical=True))
    if dfs['winner_id']!=optimized['winner_id'] or dfs['tie_ids']!=optimized['tie_ids']:
        raise ValueError('Exhaustive/quotient full optimum/tie mismatch')
    summary=dict(candidate_count=len(items),dfs_oracle_calls=dfs['oracle_calls'],optimized_oracle_calls=optimized['oracle_calls'],
        actual_oracle_calls_saved=dfs['oracle_calls']-optimized['oracle_calls'],
        equivalent_candidates_skipped=optimized['proved_equivalent_candidates_skipped'],
        strict_bound_candidates_pruned=optimized['strict_bound_candidates_pruned'],
        nonempty_optimum_identified=bool(dfs['tie_ids']),same_nonempty_optimum_and_all_ties=bool(dfs['tie_ids']),
        empty_result_sets_agree_without_optimality_claim=not dfs['tie_ids'],
        whole_declared_cpu_domain_certified=dfs['whole_declared_cpu_domain_certified'],
        winner_id=dfs['winner_id'],tie_ids=dfs['tie_ids'],exact_max_objective=dfs['exact_max_objective'],
        unknown_candidate_count=len(dfs['unknown_ids']),
        unknown_run_reason_counts=dict(Counter(r['reason'] for row in dfs['records'] for r in row['runs'] if r['status']=='UNKNOWN')),
        dfs_seconds=dfs['elapsed_seconds'],optimized_seconds=optimized['elapsed_seconds'],
        observed_process_local_speedup=dfs['elapsed_seconds']/optimized['elapsed_seconds'] if optimized['elapsed_seconds'] else None,
        call_saving_source='Proved no-spill margin/cooldown equivalence; zero upper bound usually prunes nothing',
        timing_scope='Single nonexclusive process, DFS first and optimized second, actual simulations timed; no cold-start or GPU speedup claim',
        objective_cache_hits=0,full_run_crosschecks=checks,physical_U=None,physical_optimality=False)
    return dfs,optimized,summary


def run_study(output,template_contract,resource_path,base_packet,*,historical_resource_path=None,seeds=CPU_SEEDS):
    if tuple(seeds)!=CPU_SEEDS:
        raise ValueError('Declared fresh five-term CPU seeds413/414 required')
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    base,contract,resources=read(base_packet),read(template_contract),read(resource_path)
    contract['weights']=dict(FIXED_WEIGHTS)
    contract['objective_schema']=OBJECTIVE_SCHEMA
    contract['horizon_seconds']=base['config']['horizon_seconds']
    if contract['deployment']!=PRIMARY or contract['text_rate']!=1.85 or contract['horizon_seconds']!=600.:
        raise ValueError('Exact PRIMARY/text1.85/600s required')
    validate_weights(contract['weights'])
    provider=load_provider(resources,PRIMARY)
    samples=Path(base['inputs'])/'runtime-inputs.jsonl'
    if receipt(samples)['sha256']!=base['config']['catalog']['quality']['identity']['input_pool_sha256']:
        raise ValueError('Frozen input pool changed')
    output.mkdir(parents=True)
    inputs={name:receipt(value) for name,value in (('template',template_contract),('resources',resource_path),('base',base_packet)) if not isinstance(value,dict)}
    inputs['samples']=receipt(samples)
    sources=[receipt(__file__),receipt(ROOT/(CoupledUtilityPolicy.__module__.replace('.','/')+'.py')),
             receipt(ROOT/'tools/queue_coupled_utility_20261023/search.py'),receipt(ROOT/'tools/queue_coupled_utility_20261023/resources.py'),
             receipt(ROOT/'tools/queue_coupled_utility_20261023/model.py')]
    historical=read(historical_resource_path) if historical_resource_path is not None else None
    audit=coverage_audit(resources,historical)
    if historical_resource_path is not None and not isinstance(historical_resource_path,dict):
        inputs['historical_inventory_only']=receipt(historical_resource_path)
    save(output/'resource-coverage-audit.json',dict(audit,inputs=inputs))
    save(output/'objective-and-inputs.json',dict(objective_schema=OBJECTIVE_SCHEMA,weights=dict(FIXED_WEIGHTS),seeds=list(seeds),
        online_policy='New five-term CoupledUtilityPolicy, not frozen six-term routing with posthoc rescoring',
        inputs=inputs,sources=sources,P_evaluation_role='diagnostic_only',source_sha256=identity(sources)))
    workload={'text':contract['text_rate'],**{t:sum(v.values()) for t,v in contract['background_routing'].items()}}
    api=frozen_api()
    cohorts={}
    for seed in seeds:
        pool=api['workload'].SamplePool(samples,list(workload),seed)
        trace=api['trace'].make_trace(workload,600.,seed,base['config']['arrival_process'])
        cohorts[seed]=[dict(request_id=f'five-term-cpu{seed}:{i}',arrival_ns=round(arrival*1e9),task=task,pool_id=pool.next(task)['pool_id'])
                       for i,(arrival,task) in enumerate(trace)]
    items=list(candidates())
    save(output/'domain.json',dict(candidates=items,seeds=list(seeds),workload=workload,horizon_seconds=600.,
        objective_schema=OBJECTIVE_SCHEMA,weights=dict(FIXED_WEIGHTS),candidate_ids=[identity(c) for c in items],
        cohort_identities={str(seed):identity(rows) for seed,rows in cohorts.items()},source_scope='One PRIMARY deployment,24 policies only; no seven-model global certificate'))
    memory=resources.get('memory_evidence') or {}
    budget=base['config'].get('gpu_memory_mib')
    memory_norm=memory['safe_memory_mib']/budget if memory.get('measured') is True and memory.get('resource_valid') is True and finite(memory.get('safe_memory_mib')) and memory['safe_memory_mib']>=0 and finite(budget) and budget>0 else None
    studies={}
    for name,text_only in (('full_workload',False),('text_only_mechanism_diagnostic',True)):
        directory=output/name
        directory.mkdir()
        def evaluate(candidate,seed):
            return dict(evaluate_cohort(contract,resources,cohorts[seed],candidate,text_only=text_only,provider=provider,memory_norm=memory_norm),seed=seed)
        dfs,optimized,comparison=compare_domain(items,seeds,evaluate)
        comparison.update(workload={'text':workload['text']} if text_only else workload,
            scope='Text-only conditional mechanism diagnostic; never a full mixed-workload selector' if text_only else 'Original full mixed-workload coverage; missing evidence remains UNKNOWN')
        save(directory/'dfs.json',dfs)
        save(directory/'optimized.json',optimized)
        save(directory/'comparison.json',comparison)
        studies[name]=comparison
    full=studies['full_workload']
    result=dict(schema='gsdeploy.five_term_cpu_search.v1',objective_schema=OBJECTIVE_SCHEMA,weights=dict(FIXED_WEIGHTS),
        seeds=list(seeds),studies=studies,coverage=audit,
        native_parameter_selection=dict(source='Complete full-workload conditional CPU optimum' if full['whole_declared_cpu_domain_certified'] else 'Prespecified default; full-workload CPU optimum unidentified',
            full_workload_winner_id=full['winner_id'],parameters=next((c for c in items if identity(c)==full['winner_id']),DEFAULT_PARAMETERS),
            text_only_winner_used=False),physical_U=None,physical_optimality=False,GPU_started=False,
        actual_new_policy_simulated=True,old_outcome_tables_used=False,
        limitations=['Controlled resources support text-only activity, not the full image/asr/ocr shared environment.',
            'Aggregate Q, empirical constant M and window-only E are conditional proxies, not measured physical utility.',
            'No original stability certificate comes from CPU simulation; original native stability rules remain required.',
            'Real simulation count savings are from symbolic equivalence; no native GPU inference speedup is claimed.'])
    save(output/'summary.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--template-contract',type=Path,required=True)
    parser.add_argument('--resources',type=Path,required=True)
    parser.add_argument('--base-packet',type=Path,required=True)
    parser.add_argument('--historical-resources',type=Path)
    args=parser.parse_args(argv)
    result=run_study(args.output,args.template_contract,args.resources,args.base_packet,historical_resource_path=args.historical_resources)
    print(json.dumps(dict(status='COMPLETE',selection=result['native_parameter_selection'],studies=result['studies']),ensure_ascii=False))


if __name__=='__main__':
    main()
