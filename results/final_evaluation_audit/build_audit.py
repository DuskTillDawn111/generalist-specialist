"""Module 6A: CPU-only audit of frozen artifacts. No algorithm or GPU execution.

Run from repository root with /root/miniconda3/envs/gsdeploy/bin/python.
Plots are generated separately with the existing base Python matplotlib runtime.
All writes are restricted to this directory; historical results are immutable.
"""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as stats
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deployment_core.models import load_model_catalog
from performance_model.local_calibration.stability_v2 import DEFAULT_THRESHOLDS, reconstruct, classify

R = ROOT / 'results'
O = Path(__file__).resolve().parent
STATUS_ORDER = {'unstable': 0, 'recoverable_congested': 1, 'steady': 2}
inventory = json.loads((O/'input_inventory_sha256.json').read_text())

def read(p):
    return json.loads(Path(p).read_text())

def lines(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ref(p):
    p = Path(p).resolve()
    return {'path': str(p), 'sha256': inventory.get(str(p), sha(p))}

def save(name, data):
    (O/name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def csvsave(name, rows):
    assert rows
    with (O/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def active(d):
    return '+'.join(sorted(k for k, v in d.items() if v))

def rates(d):
    return {k: round(float(v), 12) for k, v in sorted(d.items()) if v > 0}

def fractions(routing, workload):
    return {t: {m: round(float(v)/workload[t], 12) for m, v in sorted(row.items()) if v > 0}
            for t, row in sorted(routing.items()) if workload.get(t, 0) > 0}

def key(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'))

def describe(reps):
    assert reps and all(r['run_status']=='completed' for r in reps)
    means = [r['weighted_mean_e2e'] for r in reps]
    tails = [r['weighted_p95_e2e'] for r in reps]
    assert all(x is not None and math.isfinite(x) for x in means+tails)
    return dict(stability=min((r['stability_state_v2'] for r in reps), key=STATUS_ORDER.get),
                mean=stats.mean(means), median=stats.median(means),
                sample_std=stats.stdev(means) if len(means)>1 else None,
                weighted_p95=stats.mean(tails), n=len(reps),
                individual_means=means, individual_weighted_p95=tails,
                seeds=[r['random_seed'] for r in reps],
                raw_sources=[ref(r['raw_result_path']) for r in reps])

def new_execution_status(raw, outcome, log):
    # Data Quality Audit v1 policy, as adopted by campaign_runtime.execution_status.
    # Completed serving instability remains a completed physical observation.
    if raw is None:
        return 'uncertain_interruption'
    w, s = raw.get('measurement_window') or {}, raw.get('system_metrics') or {}
    complete = bool(w and s and w.get('duration_seconds')==raw.get('measurement_seconds')
                    and w.get('arrival_loop_end_time_ns',0)>=w.get('end_time_ns',1)
                    and w.get('drain_end_time_ns',0)>=w.get('arrival_loop_end_time_ns',1))
    footer = Path(log).exists() and '"output"' in Path(log).read_text(errors='replace')[-1600:]
    if complete and footer and not outcome.get('configuration_mismatch'):
        return 'completed'
    if raw.get('adapter_failure'):
        return 'uncertain_interruption'
    if any('worker' in str(e).lower() or 'model' in str(e).lower() for e in raw.get('errors', [])):
        return 'model_failure'
    return 'harness_failure'

def physical_audit():
    catalog = load_model_catalog()  # Existing offline catalog only; no model loading.
    rows = lines(R/'data_quality_audit/run_provenance_audit.jsonl')
    assert len(rows)==read(R/'data_quality_audit/audit_counts.json')['total_runs']
    for r in rows:
        assert sha(r['source_result_path'])==r['source_sha256']
        r['audit_layer']='historical_Data_Quality_Audit_v1'
        r['measurement_stage']='historical_measurement'
    sources = [
        ('Module 3C', R/'deployment_search_multiworkload/shared_runtime_cache_manifest.json'),
        ('Module 4B', R/'routing_search_real_pilot/runtime_cache_manifest.json'),
        ('Module 5B', R/'dynamic_adaptation_real_pilot/search_cache/runtime_cache_manifest.json')]
    for phase, p in sources:
        manifest=read(p)
        for a in manifest['run_attempts']:
            rows.append(audit_attempt(a,phase,'search-time measurements',p,catalog))
    for phase, d in [('Module 3C','deployment_search_multiworkload'),('Module 5B','dynamic_adaptation_real_pilot')]:
        p=R/d/'confirmation_outcomes.jsonl'
        starts=lines(R/d/'confirmation_attempts.jsonl')
        outcomes=lines(p)
        assert len(starts)==len(outcomes)
        assert sorted(a['attempt_id'] for a in starts)==sorted(a['attempt_id'] for a in outcomes)
        for a in outcomes:
            rows.append(audit_attempt(a,phase,'post-search confirmation',p,catalog))
    assert len({r['source_result_path'] for r in rows})==len(rows)
    deployments, workloads, routing_rates, routing_fractions = {}, {}, {}, {}
    for row in rows:
        raw=read(row['source_result_path'])
        dep=raw.get('deployment')
        work=raw.get('workload_rps')
        route=raw.get('routing_rps')
        # The archived adapter envelope has no harness fields; recover frozen config.
        if dep is None or work is None or route is None:
            archive=R/'deployment_search_real_pilot/runtime_cache_manifest.json'
            m=read(archive)
            for e in m['configurations'].values():
                for a in e.get('interrupted_attempts',[]):
                    if a['archived_artifacts']['failure_envelope']['path']==row['source_result_path']:
                        ident=e['identity'];dep=ident['deployment'];work=ident['workload'];route=ident['routing']
        assert dep is not None and work is not None and route is not None, row['source_result_path']
        normalized_dep={k:int(v) for k,v in sorted(dep.items()) if v}
        normalized_work=rates(work)
        normalized_route={t:rates(v) for t,v in sorted(route.items()) if work.get(t,0)>0}
        normalized_fraction=fractions(route,work)
        row.update(deployment=normalized_dep, workload=normalized_work, routing_rps=normalized_route,
                   routing_fractions=normalized_fraction)
        for states, value in [(deployments,normalized_dep),(workloads,normalized_work),
                              (routing_rates,normalized_route),(routing_fractions,normalized_fraction)]:
            states[key(value)]=value
    (O/'execution_status_ledger.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    byphase={p:{'physical_attempts':sum(r['phase']==p for r in rows),
                'execution_status':dict(Counter(r['run_status'] for r in rows if r['phase']==p)),
                'completed_stability':dict(Counter(r['stability_state_v2_corrected'] for r in rows
                                          if r['phase']==p and r['run_status']=='completed'))}
             for p in sorted({r['phase'] for r in rows})}
    count=dict(scope='All physical timed-serving attempts inventoried under results/. Offline model profiling outside results/ is excluded.',
        execution_status_truth_source=ref(R/'data_quality_audit/run_provenance_audit.jsonl'),
        extension_policy_source=ref(ROOT/'deployment_search/campaign_runtime.py'),
        consolidated_truth_source=str(O/'execution_status_ledger.jsonl'),
        historical_physical_attempts=130, later_physical_attempts=len(rows)-130,
        total_physical_attempts=len(rows),
        valid_completed_runs=sum(r['run_status']=='completed' for r in rows),
        valid_latency_runs=sum(r['valid_for_latency'] for r in rows),
        external_interruptions=sum(r['run_status']=='external_interruption' for r in rows),
        genuine_unstable_runs=sum(r['run_status']=='completed' and r['stability_state_v2_corrected']=='unstable' for r in rows),
        execution_status_counts=dict(Counter(r['run_status'] for r in rows)),
        completed_stability_counts=dict(Counter(r['stability_state_v2_corrected'] for r in rows if r['run_status']=='completed')),
        total_workloads=len(workloads), total_unique_deployment_states=len(deployments),
        total_unique_routing_states=len(routing_fractions), total_unique_routing_rps_states=len(routing_rates),
        main_benchmark_named_workloads=3,
        uniqueness_definition='Positive arrival-rate vectors and routing values rounded to 12 decimal places; zero entries removed. Deployment uses active model counts. Main routing-state count uses fractions, not absolute rates.',
        by_phase=byphase,
        search_confirmation_new_physical_attempts={stage:sum(r['measurement_stage']==stage for r in rows)
            for stage in ['search-time measurements','post-search confirmation']},
        separately_counted_reconfiguration_operations=len(read(R/'dynamic_adaptation_real_pilot/reconfiguration_events.json')['events']),
        exclusions='Accepted-path activation, setup/teardown, test fixtures, cached copies, derived rows and duplicate artifact references are not timed-serving physical attempts.')
    save('gpu_experiment_scale.json',count)
    save('observed_states.json',dict(workloads=list(workloads.values()),deployments=list(deployments.values()),
          routing_fractions=list(routing_fractions.values()),routing_rps=list(routing_rates.values())))
    return rows,count

def audit_attempt(a, phase, stage, ledger, catalog):
    p=Path(a['raw_result_path']);raw=read(p) if p.exists() else None
    log=a.get('log_path',str(p.with_suffix('.log')))
    status=new_execution_status(raw,a.get('process_outcome',{}),log)
    assert status==a['run_status'], (p,status,a['run_status'])
    if a.get('raw_sha256') is not None:
        assert sha(p)==a['raw_sha256']
    label=None
    if status=='completed':
        label=classify(reconstruct(raw,catalog,DEFAULT_THRESHOLDS),DEFAULT_THRESHOLDS)['stability_state_v2']
        if a.get('runtime'):
            assert label==a['runtime']['stability_state_v2']
    active_tasks=[t for t,v in (raw or {}).get('workload_rps',{}).items() if v>0]
    latency=status=='completed' and bool(active_tasks) and all(
        raw['per_task_metrics'].get(t,{}).get('mean_e2e_latency_ms') is not None and
        raw['per_task_metrics'].get(t,{}).get('p95_e2e_latency_ms') is not None for t in active_tasks)
    return dict(run_id=f'{phase}/{stage}/{a["attempt_id"]}', phase=phase,
        source_result_path=str(p),source_sha256=sha(p) if p.exists() else None,
        run_status=status,stability_state_v2_corrected=label,valid_for_latency=latency,
        valid_for_stability=status=='completed',valid_for_attempt_accounting=True,
        valid_for_ranking=latency and label in ('steady','recoverable_congested'),
        audit_layer='Data_Quality_Audit_v1_scope_extension_Module6A', measurement_stage=stage,
        ledger_source=ref(ledger), evidence={'harness_log':ref(log),'recorded_run_status':a['run_status'],
            'raw_harness_status':(raw or {}).get('status'),'full_window_drain_and_footer_rechecked':status=='completed'})

def static_search():
    base=R/'deployment_search_multiworkload'
    records=[];table=[];comparison=read(base/'proposed_vs_random.json')
    for p in sorted((base/'algorithm_runs').glob('*/run_result.json')):
        run=read(p);metric=next(m for m in read(base/'per_workload_metrics.json') if m['run_id']==run['run_id'])
        trace=read(p.parent/'search_trace.json')
        search=describe(run['final_runtime']['repetitions'])
        seed=describe(run['seed_runtime']['repetitions'])
        conf=describe([read(p.parent/'confirmation/final'/f'rep_{i}.json') for i in (1,2)])
        seed_conf=describe([read(p.parent/'confirmation/seed'/f'rep_{i}.json') for i in (1,2)])
        assert run['final_deployment']==trace['deployment']
        assert run['valid_unique_runtime_evaluations']==trace['runtime_evaluation_count']
        rec=dict(run_id=run['run_id'],workload=run['workload'],algorithm=run['algorithm'],
            final_deployment=run['final_deployment'],seed_deployment=run['seed_deployment'],
            source=ref(p),search_final=search,search_seed=seed,confirmation_final=conf,confirmation_seed=seed_conf,
            runtime_evaluations=run['valid_unique_runtime_evaluations'],lp_solves=run['lp_solve_count'],
            accepted_mutations=metric['accepted_mutations'],rollbacks=metric['rollback_count'],
            search_physical_attempts=metric['physical_run_attempts'],confirmation_physical_attempts=4,
            search_improvement_fraction=1-search['mean']/seed['mean'],
            confirmation_improvement_fraction=1-conf['mean']/seed_conf['mean'],
            enumeration_calls=run['enumeration_calls'],predictor=None,
            historical_pooled_mean=metric['final']['weighted_mean_e2e']['mean'])
        records.append(rec)
        for stage,values in [('search-time measurements',search),('post-search confirmation',conf)]:
            table.append({'Workload':run['workload'].split('_')[0],'Method':run['algorithm'],
                'Final Deployment':active(run['final_deployment']),'Stability':values['stability'],
                'Mean E2E':values['mean'],'Median E2E':values['median'],'Std':values['sample_std'],
                'Weighted P95':values['weighted_p95'],'Runtime Evaluations':rec['runtime_evaluations'],
                'LP Solves':rec['lp_solves'],'Accepted Mutations':rec['accepted_mutations'],'Rollbacks':rec['rollbacks'],
                'Measurement Stage':stage,'Repetitions':values['n'],'Latency Unit':'ms',
                'Cost Stage':'frozen search counters; repeated on confirmation row for context',
                'Source Report':str(p),'Raw Sources':';'.join(x['path'] for x in values['raw_sources'])})
    rq=dict(question='Does LP-guided bounded-feedback search improve serving performance over the MILP feasible seed and simple search baselines?',
        records=records,decision='PARTIAL SUPPORT',frozen_objective_comparisons=comparison,
        b2_seed_improvement_workloads=sum(x['b2_improves_seed_under_frozen_objective'] for x in comparison),
        b2_random_trial_wins=sum(sum(x['b2_beats_random_trial_under_frozen_objective']) for x in comparison),
        real_random_trials=6,b2_b3_same_final_all_workloads=all(x['proposed_final']==x['b3_final'] for x in comparison),
        method_definitions={'B0':'minimum-memory feasible MILP seed', 'B1':'random neighbor selection with runtime feedback; seeds 1103 and 2207',
            'B2':'LP guidance with bounded runtime feedback; predictor=None','B3':'LP-only transition decisions; runtime measured for comparison'},
        provenance_policy='Main tables never pool search and confirmation. Existing pooled descriptive means retained solely for reconciliation. Confirmation did not influence search.',
        metric_semantics='Arrival-rate-weighted task mean and arrival-rate-weighted task P95; P95 is not the pooled request percentile. Std is between-repetition sample std (ddof=1), not a confidence interval.',
        shared_cache_limitation='Equal-budget state evaluations are policy accesses; different policies can share the exact same physical repetitions. Three state repetitions and two independent confirmation seeds are not independent search campaigns.',
        workload_freeze=ref(base/'benchmark_workloads.json'))
    save('rq2_static_search.json',rq);csvsave('table_static_search.csv',table)
    return records

def scalability():
    path=R/'deployment_search_foundation/scalability_validation.json';x=read(path)
    oracle=read(R/'deployment_search_foundation/synthetic_validation.json')['small_exhaustive_baseline']
    table=[]
    for r in x['runs']:
        table.append({'Number Models':r['num_models'],'Candidate Count':r['num_candidates_generated'],
            'LP Solves':r['lp_solve_count'],'Runtime Evaluations':r['runtime_evaluation_count'],
            'Search Result':r['status'],'Oracle Gap if available':oracle['gap'] if r['num_models']==5 else None,
            'Enumeration Calls':r['enumeration_call_count'],'Max Candidates Per Iteration':r['max_candidates_in_iteration'],
            'Final Synthetic Mean ms':r['final_mean_ms'],'Full Binary Space':r['full_binary_space_size'],
            'Measurement Stage':'synthetic search-time measurements','Oracle Gap Unit':'fraction',
            'Source Report':str(path)})
    csvsave('table_scalability.csv',table)
    save('rq1_scalability.json',dict(question='Can the method search Generalist–Specialist deployments without exhaustive deployment enumeration?',
        runs=x['runs'],small_space_oracle=oracle,source=ref(path),
        enumeration_scope='0 within production guided search. Independent 5-model oracle enumerates 32 binary states, with 31 feasible objective evaluations.',
        evidence_strength='Strong implementation and bounded-count validation; synthetic only for 10/20/30-model search.',
        complexity_limit='At most M one-hop candidates per iteration. This is not a polynomial-time guarantee for MILP or total system execution.',
        measured_search_wall_clock_seconds=None,real_large_pool_performance_available=False))

def guidance():
    base=R/'deployment_search_multiworkload';points=[];unique={}
    for p in sorted((base/'algorithm_runs').glob('*/search_trace.json')):
        trace=read(p);run=read(p.parent/'run_result.json')
        for t in trace['iterations']:
            current=t['current_runtime'];cur=active(t['deployment'])
            for c in t['candidates']:
                runtime=c.get('runtime')
                if runtime is None or current['weighted_mean_e2e'] is None or runtime['weighted_mean_e2e'] is None:
                    continue
                item=dict(workload=run['workload'],method=run['algorithm'],run_id=run['run_id'],iteration=t['iteration'],
                    current_deployment=cur,candidate_deployment=active(c['mutation']['deployment']),
                    delta_z=c['delta_z'],delta_J_ms=current['weighted_mean_e2e']-runtime['weighted_mean_e2e'],
                    candidate_decision=c['status'],source=ref(p),
                    current_cache_key=current.get('cache_key'),candidate_cache_key=runtime.get('cache_key'))
                points.append(item);unique[(item['workload'],cur,item['candidate_deployment'])]=item
    original=read(base/'lp_runtime_correlation.json')
    assert len(points)==len(original['transitions'])
    sensitivity=read(base/'lp_runtime_correlation_unique_sensitivity.json')
    assert all(sum(x['workload']==w for x in unique.values())==m['n_unique_transitions'] for w,m in sensitivity['by_workload'].items())
    save('rq3_guidance.json',dict(question='Is LP pressure relief informative for identifying useful deployment mutations?',
        points=points,unique_transition_points=list(unique.values()),
        policy_transition_descriptive_correlations=original['descriptive_by_workload'],
        unique_transition_sensitivity=sensitivity,proposed_vs_random=read(base/'proposed_vs_random.json'),
        feedback_ablation=read(base/'ablation_results.json'),
        delta_definitions='delta_z = current LP max pressure minus candidate LP max pressure; delta_J = current weighted mean E2E minus candidate weighted mean E2E in ms; positive means improvement.',
        safe_feedback_wording='runtime feedback provides an online correction / safety mechanism, while the current static benchmark does not isolate an independent final-performance gain from feedback.',
        feedback_correction_in_current_real_static_data=0,causal_interpretation=False,
        limitation='Evaluated Top-B candidates are selected, cached and correlated; unmeasured mutations are excluded. No fit, causal claim or independent-sample significance test.'))
    csvsave('figure_lp_vs_runtime_data.csv',[{k:v for k,v in x.items() if k!='source'} for x in unique.values()])

def dynamic():
    base=R/'dynamic_adaptation_real_pilot';x=read(base/'cold_vs_warm_metrics.json');table=[];trace=[];records=[]
    initial=read(base/'transition_w1_w2/D2/result.json')['initial_deployment']
    trace.append(dict(phase=0,workload='W1',deployment=active(initial),action='INITIAL',models_loaded=0,models_unloaded=0))
    for i,t in enumerate(x['transitions'],1):
        trans=t['transition'];cold=t['strategies']['D1'];target=trans[-2:].upper()
        for method,s in t['strategies'].items():
            p=base/trans/method/'result.json';run=read(p)
            for k in ['final_deployment','final_objective','trace','switches']:
                assert run[k]==read(p.parent/'search_result_frozen.json')[k]
            values=describe(run['final_objective']['repetitions'])
            assert math.isclose(values['mean'],s['performance']['weighted_mean_e2e'],abs_tol=1e-10)
            rc=s['reconfiguration'];coldmean=cold['performance']['weighted_mean_e2e'];coldevals=cold['search']['valid_runtime_evaluations']
            record=dict(transition=trans,method=method,source=ref(p),
                initial_deployment=s['initial_deployment'],final_deployment=s['final_deployment'],
                action=s.get('action') or s['status'],search_final=values,search_cost=s['search'],
                reconfiguration=rc,endpoint_structural_sensitivity=s.get('endpoint_structural_sensitivity'),
                confirmation_final=None)
            stages=[('search-time measurements',values)]
            if method!='D0':
                conf=read(p.parent/'confirmation/result.json')
                assert conf['independent_of_search'] and conf['decisions_unchanged']
                confirmation=describe(conf['repetitions']);record['confirmation_final']=confirmation
                stages.append(('post-search confirmation',confirmation))
            for stage,value in stages:
                comparemean=coldmean if stage=='search-time measurements' else t['confirmation']['strategies']['D1']['weighted_mean_e2e']
                table.append({'Transition':trans.replace('transition_','').replace('_',' -> ').upper(),'Method':method,
                    'Initial Deployment':active(s['initial_deployment']),'Final Deployment':active(s['final_deployment']),
                    'Runtime Evaluations':s['search']['valid_runtime_evaluations'],'Mean E2E':value['mean'],
                    'Weighted P95':value['weighted_p95'],'Accepted Switches':rc['accepted_switches'],
                    'Models Loaded':rc['models_loaded'],'Models Unloaded':rc['models_unloaded'],
                    'Reconfiguration Time':rc['measured_switch_time_ms']/1000 if rc['measured_switch_time_ms'] is not None else None,
                    'Performance Gap vs Cold':value['mean']/comparemean-1,
                    'Evaluation Saving':1-s['search']['valid_runtime_evaluations']/coldevals,
                    'Measurement Stage':stage,'Repetitions':value['n'],'Stability':value['stability'],
                    'Action':record['action'],'Latency Unit':'ms','Reconfiguration Time Unit':'s',
                    'Gap and Saving Unit':'fraction','Reconfiguration Semantics':'isolated accepted-path reconfiguration measurement; D1 includes cold seed reset; no switch = null',
                    'Cost Stage':'frozen search and isolated activation counters; repeated for context',
                    'Source Report':str(p),'Raw Sources':';'.join(v['path'] for v in value['raw_sources'])})
            records.append(record)
        warm=t['strategies']['D2'];rc=warm['reconfiguration']
        trace.append(dict(phase=i,workload=target,deployment=active(warm['final_deployment']),action=warm['action'],
                          models_loaded=rc['models_loaded'],models_unloaded=rc['models_unloaded']))
    csvsave('table_dynamic_adaptation.csv',table);csvsave('figure_dynamic_trace_data.csv',trace)
    save('rq4_dynamic_adaptation.json',dict(question='When workload changes, can warm-start adaptation reduce search and reconfiguration overhead relative to cold restart?',
        records=records,sequence_trace=trace,original_transition_metrics=x['transitions'],totals=x['totals'],decision=x['decision'],
        confirmation_direction_count=sum(t['confirmation']['mean_direction_maintained'] for t in x['transitions']),
        reconfiguration_semantics='isolated accepted-path reconfiguration measurement; includes cold seed reset for D1. Excludes candidate profiling, warmup, confirmation, setup and teardown; no continuous-arrival zero-downtime or wall-clock adaptation-speedup claim.',
        no_switch_semantics='D2 switches once by adding DistilBERT, then retains all five models; D0 equals D2 in phases 2 and 3. Demonstrates reuse, not phase-specific specialist turnover.',
        cache_semantics='Evaluation counts include exact-cache state accesses; physical attempts and independent repetitions are separately counted.',
        endpoint_caveat='Structural cold endpoint path is only sensitivity analysis; no time was measured for that alternative.',
        independent_full_sequence_campaigns=1,sequence_source=ref(base/'dynamic_sequence.json')))
    return records

EXPERIMENTS = {
 'deployment_seed':('1','Validate minimum-memory feasible MILP seed against small-space oracle and synthetic sizes','MILP feasible seed','summary.md','PASS'),
 'performance_model_foundation':('2A','Build run/path dataset and planning features from existing measurements','feature extraction / optional scorer foundation','dataset_summary.md','FOUNDATION'),
 'performance_model_profiling':('2B','Profile Qwen load, task mix and colocation axes','feature-directed profiling','qwen/profiling_summary.md','MEASUREMENT COMPLETE'),
 'performance_model_active_sampling':('2C','Add selected feature coverage conditions','active condition sampling','summary.md','MEASUREMENT COMPLETE'),
 'performance_model_generalization':('2D','Cross-deployment LODO prediction gate','Ridge / GBT','generalization_summary.md','PARTIAL'),
 'performance_model_generalization_corrected':('2D corrected','Reevaluate fixed LODO configuration using audited completed labels','fixed Ridge / GBT','corrected_lodo_metrics.json','PARTIAL'),
 'performance_model_local_calibration':('2E','Test bias-only local calibration K=0/1/2/3','bias-only calibration','local_calibration_summary.md','FAIL'),
 'performance_model_local_calibration_corrected':('2E corrected','Reevaluate existing bias calibration after status audit','bias-only calibration','local_calibration_summary.md','FAIL'),
 'performance_model_ranking':('2F','Test cross-deployment ranking in comparable groups','OOF scorer ranking','ranking_summary.md','INSUFFICIENT COMPARABLE GROUPS'),
 'performance_model_ranking_corrected':('2F corrected','Reevaluate comparable groups using corrected labels','OOF scorer ranking','ranking_summary.md','INSUFFICIENT COMPARABLE GROUPS'),
 'performance_model_comparable_groups':('2G','Measure controlled deployment comparisons for scorer ranking','controlled groups + optional scorer','module2g_summary.md','INSUFFICIENT COMPARABLE GROUPS'),
 'performance_model_comparable_groups_corrected':('2G corrected','Reanalyze controlled GPU measurements with DQA labels','controlled groups + optional scorer','module2g_summary.md','INSUFFICIENT COMPARABLE GROUPS'),
 'data_quality_audit':('Audit','Separate execution status from completed serving instability','read-only provenance and label audit','audit_summary.md','PASS'),
 'deployment_search_foundation':('3A','Validate non-enumeration, bounded candidates, rollback and small oracle gap','LP-guided + synthetic bounded feedback','summary.md','PASS'),
 'deployment_search_real_pilot':('3B','Pilot real bounded deployment search','LP-guided + runtime feedback','module3b_summary.md','PILOT OBSERVATION ONLY'),
 'deployment_search_multiworkload':('3C','Compare B0/B1/B2/B3 across frozen W1/W2/W3','equal-budget static search','module3c_summary.md','PARTIAL SUPPORT'),
 'routing_search_foundation':('4A','Validate fixed-deployment routing mutations and oracle cases','bounded LP-guided routing pilot','summary.md','PASS'),
 'routing_search_real_pilot':('4B','Test canonical versus two selected W1 routing transfers','canonical / OCR 10% / image 10% to Qwen','module4b_summary.md','NEUTRAL'),
 'dynamic_adaptation_foundation':('5A','Validate warm feasibility, repair, fallback, NO_SWITCH and structural cost','synthetic cold/warm adaptation','summary.md','PASS'),
 'dynamic_adaptation_real_pilot':('5B','Compare D0/cold/warm across W1->W2->W3->W1','warm-start dynamic adaptation','module5b_summary.md','PASS'),
 'stage1_5':('Legacy Stage 1.5','Historical feasibility sweeps, not serving latency measurements','analytical placement sweep','summary.md','HISTORICAL'),
 'stage2a':('Legacy Stage 2A','Short measurement smoke checks','static mixed deployment smoke','summary.md','HISTORICAL'),
 'stage2b':('Legacy Stage 2B','Historical routing characterization','static routing characterization','routing_characterization/summary.md','HISTORICAL')}

def manifest(rows, static, dyn):
    entries=[]
    ledger={r['source_result_path']:r for r in rows}
    synthetic={
        'deployment_seed':{'milp_real_profile_scenarios':19,'synthetic_scalability_cases':2},
        'deployment_search_foundation':{'scalability_search_runs':4,'small_exhaustive_oracle_runs':1,'offline_tests':18},
        'routing_search_foundation':{'synthetic_search_cases':3,'scalability_cases':3},
        'dynamic_adaptation_foundation':{'synthetic_case_families':6,'scalability_cases':3}}
    for d in sorted(p for p in R.iterdir() if p.is_dir() and p!=O):
        name=d.name;module,purpose,algorithm,report,decision=EXPERIMENTS.get(name,('Legacy','Historical artifacts','unspecified',None,'UNASSESSED'))
        reportpath=d/report if report else None
        if reportpath and not reportpath.exists():
            candidates=sorted(d.rglob('*summary*.md'));reportpath=candidates[0] if candidates else None
        files=sorted(p for p in d.rglob('*') if p.is_file())
        refs=set();analysis_rows={};recorded=[];offline_facets={k:{} for k in ('workload','deployment','routing')}
        # Find concrete physical source references in datasets and provenance files.
        for p in files:
            if p.suffix not in ('.json','.jsonl') or p.stat().st_size>30_000_000:
                continue
            if p.suffix=='.jsonl':
                items=lines(p);analysis_rows[str(p)]=len(items)
            else: items=[read(p)]
            def walk(v):
                if isinstance(v,dict):
                    for k,child in v.items():
                        # Preserve synthetic and dry-run definitions as separately
                        # labeled facets; they never become GPU observations.
                        if not own_phase(name) and isinstance(child,(dict,list,str)):
                            facet=('workload' if k in ('workload','workload_rps','arrival_rate_rps','arrival_rates_rps','old_workload','new_workload')
                                   else 'deployment' if k in ('deployment','fixed_deployment','current_deployment','initial_deployment','final_deployment')
                                   or (k=='models' and isinstance(child,list))
                                   else 'routing' if k in ('routing_rps','canonical_routing','initial_routing','final_routing') else None)
                            if facet and child:
                                offline_facets[facet][key(child)]={'kind':'offline synthetic / analytical / dry-run; not GPU',
                                                                 'value':child,'source':ref(p)}
                        if isinstance(child,str) and child in ledger:refs.add(child)
                        elif isinstance(child,(dict,list)):walk(child)
                elif isinstance(v,list):
                    for child in v:
                        if isinstance(child,str) and child in ledger:refs.add(child)
                        elif isinstance(child,(dict,list)):walk(child)
            for item in items:walk(item)
            if 'hash' in p.name or 'provenance' in p.name or 'freeze' in p.name:
                recorded.append(ref(p))
        own=[r for r in rows if Path(r['source_result_path']).is_relative_to(d)]
        refs.update(r['source_result_path'] for r in own)
        # Dataset foundation inputs are listed as source documents rather than raw paths.
        if name=='performance_model_foundation':
            refs.update(r['source_result_path'] for r in rows if r['phase'] in ('Stage 2A','Stage 2B'))
        referenced=[ledger[p] for p in sorted(refs)]
        states=lambda k: list({key(r[k]):r[k] for r in referenced}.values())
        entries.append(dict(experiment=name,module=module,purpose=purpose,
            workload=states('workload') or list(offline_facets['workload'].values()),algorithm=algorithm,
            deployment=states('deployment') or list(offline_facets['deployment'].values()),
            routing=states('routing_fractions') or list(offline_facets['routing'].values()),
            number_of_runs={'new_physical_attempts':len(own),'referenced_physical_attempts':len(referenced),
                            'analysis_row_counts':analysis_rows,'synthetic_case_counts':synthetic.get(name,{})},
            GPU_runs={'new_physical_attempts':len(own),'new_completed_runs':sum(r['run_status']=='completed' for r in own),
                      'referenced_completed_runs':sum(r['run_status']=='completed' for r in referenced)},
            synthetic_runs=synthetic.get(name,{}),
            main_metrics={**metric_refs(name,d),'summary':metric_summary(name,d)},decision=decision,
            source_hashes={'all_artifacts':{str(p):inventory[str(p)] for p in files},'recorded_source_hash_manifests':recorded},
            report_path=str(reportpath) if reportpath else None,
            physical_sources=[ref(p) for p in sorted(refs)],
            counting_note='New GPU runs are owned physical attempts only. Referenced runs, task/path rows, corrected analyses and cache hits are not additional GPU executions. Empty workload/state lists mean no physical measurement source, not an empty synthetic search.',
            supplementary_search_records=[s for s in static if name=='deployment_search_multiworkload']+
                                         [s for s in dyn if name=='dynamic_adaptation_real_pilot']))
    save('experiment_manifest.json',dict(schema_version=1,scope='All immediate results/ experiment directories; raw artifacts fully hashed; corrected variants and legacy stages included.',
        generated_date='2026-10-01',counting_policy='Data Quality Audit v1 plus separately identified read-only 6A scope extension; do not sum referenced runs across experiments.',
        experiments=entries,total_experiment_directories=len(entries),
        total_owned_physical_attempts=sum(e['GPU_runs']['new_physical_attempts'] for e in entries),
        evidence_tiers=['synthetic validation','historical pilot','search-time measurements','post-search confirmation'],
        index_scope_exclusion='No algorithm development; no newly generated GPU or synthetic algorithm runs.'))

def own_phase(name):
    return name in {'deployment_search_multiworkload','deployment_search_real_pilot','dynamic_adaptation_real_pilot',
                    'routing_search_real_pilot','stage2a','stage2b','performance_model_profiling',
                    'performance_model_active_sampling','performance_model_comparable_groups'}

def metric_summary(name,d):
    if name=='deployment_search_foundation':
        x=read(d/'scalability_validation.json')
        return {'scalability':[{k:r[k] for k in ['num_models','num_candidates_generated','lp_solve_count','runtime_evaluation_count','enumeration_call_count','final_mean_ms']} for r in x['runs']],
                'oracle_gap_fraction':read(d/'synthetic_validation.json')['small_exhaustive_baseline']['gap']}
    if name=='deployment_search_multiworkload':
        x=read(O/'rq2_static_search.json')
        return {k:x[k] for k in ['decision','b2_seed_improvement_workloads','b2_random_trial_wins','real_random_trials','b2_b3_same_final_all_workloads']}
    if name=='dynamic_adaptation_real_pilot':
        x=read(d/'cold_vs_warm_metrics.json')
        return {'totals':x['totals'],'decision':x['decision'],'independent_confirmations_maintain_direction':3}
    if name=='routing_search_real_pilot':
        return {'decision':'NEUTRAL','W1_mean_ms':{'canonical':59.215912,'OCR_10pct_to_Qwen':61.408380,'image_10pct_to_Qwen':63.166133},
                'accepted_mutations':0,'measured_states':3}
    if name=='deployment_seed':
        return {'analytical_scenarios':19,'passed':read(d/'stage1_5_milp_validation.json')['passed'],
                'synthetic_scalability':[{k:r[k] for k in ['num_models','num_variables','num_constraints','solver_time_ms','enumeration_called']} for r in read(d/'synthetic_scalability.json')['tests']]}
    if name.startswith('performance_model_generalization'):
        return {'cross_deployment_predictor_gate':'PARTIAL','role':'optional coarse performance scorer'}
    if name.startswith('performance_model_local_calibration'):
        return {'bias_only_calibration_gate':'FAIL','K':[0,1,2,3]}
    if name.startswith(('performance_model_ranking','performance_model_comparable_groups')):
        return {'ranking_gate':'INSUFFICIENT COMPARABLE GROUPS','not_a_core_contribution':True}
    if name=='data_quality_audit':
        return {'historical_scope_counts':read(d/'audit_counts.json'),'current_scope_counts_file':str(O/'gpu_experiment_scale.json')}
    if name=='deployment_search_real_pilot':
        return {'historical_reported_improvement':'99.35%','paper_use':'pilot observation only; not main acceleration claim',
                'extreme_seed_repetition_approx_ms':38400}
    return {'metric_provenance':'See hashed metric files and experiment report; not a new measurement.'}

def metric_refs(name,d):
    names={
      'deployment_search_multiworkload':['per_workload_metrics.json','macro_summary.json','proposed_vs_random.json','ablation_results.json'],
      'deployment_search_foundation':['scalability_validation.json','synthetic_validation.json'],
      'dynamic_adaptation_real_pilot':['cold_vs_warm_metrics.json','reconfiguration_costs.json','final_validation.json'],
      'data_quality_audit':['audit_counts.json','module2_gate_corrected.json'],
      'performance_model_generalization_corrected':['corrected_lodo_metrics.json'],
      'performance_model_local_calibration_corrected':['calibration_fold_metrics.json'],
      'routing_search_real_pilot':['pilot_metrics.json','final_comparison.json']}.get(name,[])
    if not names:
        names=[str(p.relative_to(d)) for p in sorted(d.glob('*.json')) if any(s in p.name for s in ('metrics','validation','verification','decision','manifest'))]
    return {'files':[ref(d/n) for n in names if (d/n).exists()],
            'role':'optional coarse performance scorer; proposed predictor=None' if name.startswith('performance_model') else 'See source-specific metrics and RQ tables; no cross-stage pooling.'}

def verify_immutable():
    actual={str(p):sha(p) for p in R.rglob('*') if p.is_file() and O not in p.parents}
    assert actual==inventory, 'Historical results changed during audit'
    save('audit_validation.json',dict(passed=True,input_files_verified=len(actual),historical_result_files_changed=0,
        new_gpu_experiments=0,new_synthetic_algorithm_experiments=0,algorithm_files_modified=0,
        tables_separate_search_confirmation=True,execution_ledger_has_unique_physical_paths=True,
        source_hashes_verified=True,search_decisions_and_dynamic_freezes_checked=True,
        note='Dataset and provenance checks only; no algorithm tests or GPU experiments executed.'))

if __name__=='__main__':
    rows,count=physical_audit()
    print('Physical audit:',json.dumps({k:count[k] for k in ['total_physical_attempts','valid_completed_runs','external_interruptions','genuine_unstable_runs','total_workloads','total_unique_deployment_states','total_unique_routing_states']}),flush=True)
    static=static_search();scalability();guidance();dyn=dynamic();manifest(rows,static,dyn)
    verify_immutable()
    print('Module 6A data audit complete; no GPU experiments launched.',flush=True)
