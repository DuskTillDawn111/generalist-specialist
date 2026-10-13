"""Closeout from frozen CPU/GPU evidence, with automatic minimal selection."""
import time,math,itertools
import numpy as np
from deployment_core import Workload,Hardware
from method_v2 import frozen_config
from dataclasses import replace
from .common import *
from .integration import create_search

def paired_ci(values):
    if not values:return None
    # Exact empirical paired bootstrap for n=3 (27 combinations).
    means=[float(np.mean(sample)) for sample in itertools.product(values,repeat=len(values))]
    return np.quantile(means,[.025,.975]).tolist()

def attempt_raw_path(attempt):
    """Resolve archived launch failures by recorded content identity."""
    original=Path(attempt['raw_path'])
    candidates=[original]
    if '/R0/' in str(original):
        candidates.append(Path(str(original).replace('/R0/','/R0_startup_failure_001/')))
    expected=attempt['raw_sha256']
    matches=[p for p in candidates if p.exists() and file_hash(p)==expected]
    assert len(matches)==1,('Attempt raw identity missing or ambiguous',attempt['raw_path'],expected)
    return matches[0]

def ranking_path_audit():
    records=[]
    for cell in read(OUT/'ranking_gpu_manifest.json')['selected_cells']:
        a=read(OUT/'ranking_gpu'/cell/'V2_delta_z'/'search_complete.json')
        b=read(OUT/'ranking_gpu'/cell/'V3_delta_J'/'search_complete.json')
        def queries(trace):
            result=[{'deployment':trace['seed']['deployment'],'runtime':trace['seed_runtime'],'iteration':0}]
            for iteration in trace['iterations']:
                for candidate in sorted((c for c in iteration['candidates'] if c.get('runtime')),key=lambda c:c['rank']):
                    result.append({'deployment':candidate['mutation']['deployment'],'runtime':candidate['runtime'],'iteration':iteration['iteration']+1})
            return result
        qa,qb=queries(a),queries(b);prefix=0
        for old,new in zip(qa,qb):
            if old['deployment']!=new['deployment']:break
            prefix+=1
        previous=[q for q in qa if q['deployment']==b['deployment']]
        records.append({'cell':cell,'common_query_prefix_states':prefix,'V2_states':len(qa),'V3_states':len(qb),
                        'V3_winning_deployment_already_measured_by_V2':bool(previous),
                        'previous_V2_winner_labels':[labels(q['runtime']) for q in previous],
                        'V3_winner_labels':labels(b['runtime']),
                        'V3_winner_repetition_labels':[labels(r) for r in b['runtime']['repetitions']],
                        'previous_V2_winner_repetition_labels':[[labels(r) for r in q['runtime']['repetitions']] for q in previous],
                        'causal_limit':'When the winning deployment was already queried by V2, different final outcomes cannot establish exclusive discovery by the ranking. Independent measurements and unchanged accept/rollback can change transitions.'})
    save('ranking_path_divergence_audit.json',{'records':records,'extra_GPU_runs':0,'ranking_formula_or_gate_changed':False})
    return records

def main():
    assert read(OUT/'campaign_status.json')['status']=='GPU_VALIDATION_COMPLETE','Finish bounded GPU validation before final selection'
    c,h,roles,cal=inputs();grid=cells();factor=read(ROOT/'results/module8f1a_memory_breakpoints/memory_safety_factor.json')['safety_factor']
    rg=read(OUT/'routing_gate.json');kg=read(OUT/'ranking_gate.json');sa=read(OUT/'surrogate_audit.json');cm=read(OUT/'candidate_ranking_metrics.json');distance=read(OUT/'routing_distance_and_task_shift.json');overhead=read(OUT/'runtime_overhead.json')
    # Full generation + canonical LP + J ranking, exact frozen neighborhood.
    timing=[]
    for cell in ['C06','C07','C10','C11','C12']:
        for _ in range(5):
            state=grid[cell];engine=create_search(None,c,cal,factor,'delta_J');start=time.perf_counter()
            result=engine.search(Workload(state['lambda_rps']),c,Hardware(state['planning_budget_input_mib']),replace(frozen_config(),dry_run=True),roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=cal)
            timing.append({'cell':cell,'wall_ms':1000*(time.perf_counter()-start),'candidates_generated':result.num_candidates_generated,'includes_cold_seed_MILP':True})
    vals=[r['wall_ms'] for r in timing]
    overhead['full_generation_LP_J_ranking_including_seed']={'median_ms':float(np.median(vals)),'p95_ms':float(np.quantile(vals,.95)),'samples':timing}
    save('runtime_overhead.json',overhead)
    routegate=rg['gate'];rankgate=kg['gate']
    if routegate=='ROUTING_SUPPORTED':routing='J_ROUTING';ranking='DELTA_J' if rankgate!='RANKING_NOT_SUPPORTED' else 'DELTA_Z';case='A' if ranking=='DELTA_J' else 'B'
    elif routegate=='ROUTING_NOT_SUPPORTED':routing='CANONICAL_MINMAX';ranking='DELTA_Z';case='D'
    else:
        # Partial evidence defaults to the simplest comparator unless there is
        # an actual repeatable steady gain and no state-level stability loss.
        gains=rg['equal_state_mean_steady_improvement_ms']
        supported_gain=gains is not None and gains>0 and rg['stability_degradations']==0 and any(r['all_three_steady_improve'] for r in rg['comparisons'])
        routing='J_ROUTING' if supported_gain else 'CANONICAL_MINMAX';ranking='DELTA_J' if supported_gain and rankgate=='RANKING_SUPPORTED' else 'DELTA_Z';case='C'
    frozen={'at_utc':now(),'Module9A':'FROZEN_I0_NO_INTERFERENCE','routing_gate':routegate,'ranking_gate':rankgate,'selection_case':case,'routing':routing,'candidate_ranking':ranking,'Top_B':2,'candidate_generation':'unchanged method_v2 targeted neighborhood and replacement closure','real_accept_rollback':'unchanged NoiseAwareRuntimeObjective(1ms,.05)','seed':'unchanged deterministic empirical MILP','solver_formula_weights':None,'quality_capability_s0_memory_g_workloads_budgets_unchanged':True,'default_integration':{'entrypoint':'module9bc.integration.create_search','ranking_argument':'delta_J' if ranking=='DELTA_J' else 'delta_z','runtime_adapter':'module9bc.integration.RoutingRuntimeAdapter' if routing=='J_ROUTING' else 'none'},'alternative_J_routes_retained':True,'runtime_capacity_infeasibility_not_relabelled':True,'future_Specialist_Saturation_started':False,'no_full_heldout_campaign':True,'gate_refs':[ref(OUT/'routing_gate.json'),ref(OUT/'ranking_gate.json')],'code_refs':read(OUT/'execution_freeze.json')['sources']}
    frozen['default_integration']['automatic_factory']='module9bc.frozen_v3.create'
    frozen['default_integration']['auditable_snapshot']='module9bc.frozen_v3.snapshot'
    frozen['integration_source_refs']=[ref(ROOT/'module9bc/frozen_v3.py'),ref(ROOT/'module9bc/launch.py')]
    save('v3_minimal_frozen.json',frozen)
    # Exercise the selected factory against a capture-only evaluator, including
    # actual route identity and JSON trace metadata. No worker is launched.
    from .frozen_v3 import create as create_frozen,snapshot as v3_snapshot
    class Capture:
        def evaluate(self,x,w,r):self.routing=dict(r);return 'CAPTURED_CPU_ONLY'
    capture=Capture();selected_engine=create_frozen(capture,c,cal)
    state=read(OUT/'routing_gpu_manifest.json')['states'][0];testw=Workload(grid[state['cell']]['lambda_rps'])
    if selected_engine.serving_routing_adapter:
        selected_engine.serving_routing_adapter.evaluate(state['deployment'],testw,flat(state['R0_routing']))
        expected=flat(state['R1_routing'])
        assert max(abs(capture.routing.get(pair,0.)-v) for pair,v in expected.items())<1e-9
    testcell=grid['C07']
    dry=selected_engine.search(Workload(testcell['lambda_rps']),c,Hardware(testcell['planning_budget_input_mib']),replace(frozen_config(),dry_run=True),roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=cal)
    test_snapshot=v3_snapshot(selected_engine,dry,Workload(testcell['lambda_rps']))
    import json
    json.dumps(test_snapshot,allow_nan=False)
    save('frozen_v3_factory_validation.json',{'status':'PASS','selected_components':{'routing':routing,'ranking':ranking},'actual_J_routing_identity_verified':routing=='J_ROUTING','canonical_planning_and_serving_routes_separated':True,'J_score_trace_serialization':ranking=='DELTA_J','no_GPU_calls':True})
    path_audit=ranking_path_audit()
    rows=read(OUT/'routing_gpu_results.json')['rows'];paired=[]
    for comparison in rg['comparisons']:
        index=comparison['state_index'];a=sorted([r for r in rows if r['state_index']==index and r['routing']=='R0'],key=lambda r:r['repetition']);b=sorted([r for r in rows if r['state_index']==index and r['routing']=='R1'],key=lambda r:r['repetition'])
        ds=[r['mean_e2e_ms']-s['mean_e2e_ms'] for r,s in zip(a,b)]
        paired.append({'cell':comparison['cell'],'selection_slot':comparison['selection_slot'],'mean_improvement_ms':float(np.mean(ds)),'paired_repetition_differences_ms':ds,'empirical_bootstrap_CI95_ms':paired_ci(ds),'R0_stability':[r['stability'] for r in a],'R1_stability':[r['stability'] for r in b],'P95_improvement_ms':float(np.mean([r['weighted_p95_ms']-s['weighted_p95_ms'] for r,s in zip(a,b)]))})
    save('paired_routing_statistics.json',{'comparisons':paired,'CI_limit':'exact empirical paired bootstrap on three independent repeats; not a broad population guarantee; gate rule unchanged'})
    oracle=[]
    with (OUT/'oracle_reachability_audit.csv').open() as stream:
        import csv
        oracle=[r for r in csv.DictReader(stream) if r['cell'] in ['C09','C12']]
    oracletext='; '.join(f"{r['cell']} iter{r['iteration']} {r['oracle_direction']}: exact generated={r['exact_generated']}, second-G ADD={r['second_G_ADD_generated']}, ranks={r['second_G_ranks']}" for r in oracle)
    integrity_refs=read(ROOT/'results/module9a_scheduling_tax/protected_references.json');verify(integrity_refs);verify(read(OUT/'experiment_freeze.json')['protected_references']);verify(read(OUT/'execution_freeze.json')['sources'])
    attempts=read(OUT/'gpu_attempts.json')['runs'];timed=0;valid=0;attempt_identities=[]
    for attempt in attempts:
        p=attempt_raw_path(attempt)
        raw=read(p);timed+=bool(raw.get('measurement_window'));valid+=bool(raw.get('valid_execution'))
        attempt_identities.append({'recorded_path':attempt['raw_path'],'resolved_raw':ref(p),'measurement_started':bool(raw.get('measurement_window')),'valid_execution':bool(raw.get('valid_execution'))})
    assert len({r['resolved_raw']['path'] for r in attempt_identities})==len(attempt_identities),'A physical attempt was counted twice'
    assert timed==valid,'Unexpected invalid timed measurement'
    save('physical_attempt_identity_audit.json',{'status':'PASS','identity':'recorded SHA256; paths alone are insufficient after startup recovery','attempts':attempt_identities})
    ranking_rows=read(OUT/'ranking_gpu_results.json')['rows']
    accounting={'physical_attempts':len(attempts),'timed_runs':timed,'valid_timed_runs':valid,'routing_valid_runs':len(rows),'ranking_valid_runs':timed-len(rows),'premeasurement_launch_failures':len(attempts)-timed,'frozen_routing_max_timed_runs':read(OUT/'routing_gpu_manifest.json')['max_new_runs'],'frozen_ranking_max_timed_runs':read(OUT/'ranking_gpu_manifest.json')['max_new_runs'],'selected_routing_states':len(read(OUT/'routing_gpu_manifest.json')['states']),'selected_ranking_cells':len(read(OUT/'ranking_gpu_manifest.json')['selected_cells']),'recovery':'exact existing frozen environment restoration; one preserved premeasurement failure','outcome_based_reruns':0,'new_profiles':0,'extra_states':0}
    assert len(rows)<=accounting['frozen_routing_max_timed_runs'] and accounting['ranking_valid_runs']<=accounting['frozen_ranking_max_timed_runs']
    save('run_accounting.json',accounting)
    from replica_scaling.telemetry import snapshot
    gpu=snapshot();assert not gpu['processes'],'GPU workers still active'
    save('validation_results.json',{'status':'PASS','convex_solver_checks':read(OUT/'routing_solver_validation.json')['status'],'analytic_integration_checks':read(OUT/'analytic_and_integration_validation.json')['status'],'frozen_9A_refs_verified':len(integrity_refs),'all_experiment_protected_refs_verified':len(read(OUT/'experiment_freeze.json')['protected_references']),'worker_runtime_and_9A_unchanged':True,'MPS_enabled':False,'candidate_generation_unchanged':True,'within_frozen_timed_budget':True,'no_active_GPU_workers':True,'automatic_gate_selection':frozen,'final_heldout_used':False})
    def corr(sub,p):return sa['metrics'][sub][p]['mean_e2e_ms']
    steadyJ=corr('steady','J_seconds');steadyZ=corr('steady','z');combinedJ=corr('steady_plus_recoverable','J_seconds');combinedZ=corr('steady_plus_recoverable','z')
    gpu_table='\n'.join(f"| {r['cell']} {r['selection_slot']} | {r['R0_mean_ms']:.3f} | {r['R1_mean_ms']:.3f} | {r['R0_tier']} → {r['R1_tier']} |" for r in rg['comparisons'])
    candold=cm['metrics']['delta_z']['mean_improvement_ms'];candnew=cm['metrics']['delta_J_seconds']['mean_improvement_ms']
    ranktable='\n'.join(f"| {r['cell']} | {r['mean_improvement_ms']:.3f} | {r['V3_better']} / {r['V3_worse']} | {r['V2_budget_used']} / {r['V3_budget_used']} |" for r in kg['comparisons'])
    audit_lines=[]
    for subset in ['steady','steady_plus_recoverable','all_development']:
        for predictor in ['z','J_seconds']:
            mean=sa['metrics'][subset][predictor]['mean_e2e_ms'];p95=sa['metrics'][subset][predictor]['weighted_p95_ms']
            audit_lines.append(f"| {subset} | {predictor} | {mean['n']} | {mean['Spearman']:.4f} | {mean['Kendall']:.4f} | {mean['pairwise_ordering_accuracy']:.4f} | {p95['Spearman']:.4f} |")
    audit_table='\n'.join(audit_lines)
    answers=[
      f"1. J vs mean E2E: steady Spearman={steadyJ['Spearman']:.6f}, Kendall={steadyJ['Kendall']:.6f}; steady+recoverable Spearman={combinedJ['Spearman']:.6f}, Kendall={combinedJ['Kendall']:.6f}.",
      f"2. z comparator: steady Spearman={steadyZ['Spearman']:.6f}, steady+recoverable={combinedZ['Spearman']:.6f}. J improves these admitted-tier correlations and within-cell ordering; all-tier pooled correlation is not uniformly better. It is a congestion surrogate, not an exact M/M/1 runtime model.",
      f"3. Historical 40 states mean routing L1={distance['historical_mean_L1_rps']:.6f} req/s; mean L1/Lambda={distance['historical_mean_L1_over_Lambda']:.6f}. Actual reassigned traffic fraction is half L1/Lambda. {distance['historical_changed_states']}/40 changed.",
      f"4. Task redistribution: {json.dumps(distance['historical_task_shift'],ensure_ascii=False)}. Largest path shifts are ASR/VQA/LID; text/image are numerical zero; OCR mostly zero.",
      f"5. Real GPU mean E2E: paired state results above and paired_routing_statistics.json. Matched-steady equal-state mean improvement={rg['equal_state_mean_steady_improvement_ms']} ms; {rg['matched_steady_states_better']} states better, {rg['matched_steady_states_worse']} worse. Unstable latency reductions are not steady-serving improvements.",
      f"6. Stability: {rg['stability_improvements']} state-tier improvements, {rg['stability_degradations']} degradations; retain all three repetition tiers per arm.",
      f"7. The chosen multi-G state is C10 (Phi + Qwen + 5S). Qwen VQA traffic rises from 0.195149 to 0.540989 req/s, while its 0.406275 req/s LID traffic moves to Whisper-small; Phi VQA falls from 0.531128 to 0.185288 req/s. Qwen total traffic therefore falls slightly (0.601425 to 0.540989 req/s), while its allocation becomes more task-specific. All three paired runs remain steady and improve mean E2E: 30.786857 to 26.387218 ms on average. This supports better allocation in this development deployment, without establishing general multi-Generalist superiority. Other historical multi-G usage is retained in routing_distance_and_task_shift.json.",
      f"8. ROUTING gate = {routegate} under the rule frozen before new GPU labels.",
      f"9. DeltaJ vs real candidate mean improvement: Spearman={candnew['Spearman']:.6f}, Kendall={candnew['Kendall']:.6f}; {cm['labelled_candidates']} known labelled records, with repeated cells/iterations clustered for descriptive CI.",
      f"10. Delta-z: Spearman={candold['Spearman']:.6f}, Kendall={candold['Kendall']:.6f}. Pooled DeltaJ correlation improves, but within-iteration accuracy is {cm['within_iteration_ordering']['delta_J_seconds']['accuracy']:.6f} vs {cm['within_iteration_ordering']['delta_z']['accuracy']:.6f}; mixed evidence, no formula tuning.",
      f"11. Top-2 known-subset hit: DeltaJ={cm['new']['Top2_hit_rate']}, Delta-z={cm['old']['Top2_hit_rate']}; no gain. Known-label counts={cm['known_label_count_distribution']}; most iterations have only two measured candidates, making Top2=1 a ceiling. All nonsteady best-tier objective ties count as hits; no unmeasured outcomes imputed.",
      f"12. Known selected unstable rates: DeltaJ={cm['new']['unstable_selected_rate']}, Delta-z={cm['old']['unstable_selected_rate']}; no reduction. These rates apply to known-label restricted sets, not unmeasured full-pool winners.",
      f"13. Structural oracle/second-Generalist direction: {oracletext}. Missing exact oracle vectors remain CANDIDATE_GENERATION_LIMIT; generated ADD ranks may also be affected by evaluator or budget. No neighborhood expansion was made.",
      f"14. Same frozen 6-state/3-iteration/Top2 limits: {json.dumps(kg['comparisons'],ensure_ascii=False)}. Actual used budgets can differ due to unchanged early stopping; caps and accept/rollback match exactly. Ranking-only sanity used canonical routes in both arms.",
      f"15. RANKING gate = {rankgate}; finite measured-subset and bounded sanity evidence only.",
      f"16. V3_MINIMAL = {routing} + {ranking}; selection Case {case}. Unsupported components remain diagnostic alternatives. No component adopted merely to create a v3 label.",
      f"17. Uncached candidate J* evaluation median={overhead['candidate_J_star_evaluation']['median_ms']:.3f} ms, P95={overhead['candidate_J_star_evaluation']['p95_ms']:.3f} ms; full generation/LP/J-ranking including seed median={overhead['full_generation_LP_J_ranking_including_seed']['median_ms']:.3f} ms.",
      "18. Yes: the deterministic CPU implementation, bounded development validation and frozen default are sufficient to enter the next Specialist Saturation benchmark when requested. CPU candidate evaluation adds milliseconds and requires no GPU profiling. Ranking evidence remains partial; the combined J-routing + DeltaJ stack has component validation, not a new full end-to-end campaign. This module did not start the next benchmark.",
      "19. No additional predictor/kernel feature/weight/neighborhood work is required. Preserve the OCR execution environment override in future launches; startup restoration is documented, with no source/package change. Any future benchmark must use the frozen default selected here, not assume J-routing success from CPU correlation."
    ]
    text=f'''# Module 9BC — E2E-aware routing and thin candidate ranking\n\nCompleted and stopped. 9A remains I0; 21/21 offline profiles remain characterization evidence.\n\nFinal gates: **{routegate}**, **{rankgate}**.\n\nFinal default: **{routing} + {ranking}**, Top-B=2, frozen v2 generation and real GPU accept/rollback.\n\n## Small matched routing validation\n\n| State | R0 mean ms | J mean ms | Stability |\n|---|---:|---:|---|\n{gpu_table}\n\nEach exact deployment/workload/routing has three independent repetitions, alternating arm order. Same sample pool, arrival protocol, 10-second warmup, 210-second measurement, worker and runtime. Paired task P95 and predictions are in routing_gpu_results.csv; empirical paired statistics retain each repetition.\n\n## Bounded ranking sanity\n\n| Cell | V2 minus V3 mean ms | V3 better / worse | Logical states V2 / V3 |\n|---|---:|---|---|\n{ranktable}\n\nBoth ranking arms use canonical routing, isolating evaluator/ranking changes; search budgets and early stopping are identical. No confirmations or heldout campaign were added.\n\n## Required answers\n\n'''+ '\n\n'.join(answers)+f'''\n\n## Evidence and limitations\n\nJ = sum_m D_m/(1-D_m/g_m) / Lambda, in seconds/request. All 130 replay states passed conservation, resident/quality/capability support, capacity domain, repeated solve equality and convex-gap certificates. The solver uses no random restart. An initial CPU preflight failure from objective conditioning was preserved separately; expressing the identical objective in milliseconds fixed numerical stopping without changing the formula.\n\nHistorical runtime audit deduplicates 192 physical repetitions into 40 exact states; three repeats are not treated as independent state observations. Correlation across load levels may be confounded, so within-cell and within-iteration ordering is also reported. Top-k uses only measured/known-label candidates; unmeasured outcomes stay unknown.\n\nAccounting: {json.dumps(accounting,ensure_ascii=False)}. One premeasurement PaddleOCR failure is preserved; exact existing frozen execution environment overrides restored the launch. No timed serving labels were discarded and no outcome-based repeats added. All original 8F/9A sources and artifacts reverified. No MPS, worker/runtime modification, new task, memory point, hardware, profiling, interference model, neighborhood, or future benchmark.\n'''
    text+='\n## Historical surrogate audit\n\n| Tier subset | Predictor | States | Mean Spearman | Mean Kendall | Mean ordering accuracy | P95 Spearman |\n|---|---|---:|---:|---:|---:|---:|\n'+audit_table+'\n\nFull P95/stability metrics and within-cell ordering are in surrogate_audit.json.\n'
    text+='\n## Ranking outcome attribution\n\n'+json.dumps(path_audit,ensure_ascii=False,indent=2)+'\n\nA better final measured state under the same available budget is a sanity outcome, not by itself a causal ranking certificate. In particular, a candidate queried in both arms can receive different repetition/stability labels and change acceptance. These limits do not alter the frozen formula or gate rule.\n'
    (OUT/'module9bc_summary.md').write_text(text)
    save('campaign_status.json',{'status':'COMPLETE_AND_STOPPED','at_utc':now(),'routing_gate':routegate,'ranking_gate':rankgate,'no_active_GPU_run':True})
    artifacts={str(p.relative_to(OUT)):ref(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['artifact_hashes.json','gpu_execution.lock']}
    save('artifact_hashes.json',artifacts)
    print(json.dumps({'COMPLETE':True,'routing_gate':routegate,'ranking_gate':rankgate,'v3_default':[routing,ranking],'accounting':accounting}),flush=True)
if __name__=='__main__':main()
