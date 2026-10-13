"""Close out the fixed benchmark, including numerical and planning failures."""
import numpy as np
from .common import *
from .planning import flat,serial
from deployment_core import Workload
from deployment_core.routing.e2e_surrogate import evaluate_surrogate

def main():
    assert read(OUT/'benchmark_status.json')['status']=='GPU_VALIDATION_COMPLETE','Finish authorized runnable states first'
    c,h,roles,cal=inputs();primary=read(OUT/'primary_hotspot_selection.json');points=read(OUT/'hotspot_points.json')['points'];planning=read(OUT/'planning_results.json')['rows'];gpu=read(OUT/'gpu_results.json')['rows']
    aggregate=read(OUT/'gpu_aggregate.json')['runtime'];target=primary['target'];cs=primary['C_Sonly'];ca=primary['C_ALL']
    rc0=next(r for r in planning if r['point']=='H0' and r['method']=='RC_SG')
    v20=next(r for r in planning if r['point']=='H0' and r['method']=='v2')
    assert rc0['deployment']==v20['deployment'] and rc0['routing']==v20['routing'],'Exact-state alias identity differs'
    for rep in gpu:
        raw=read(rep['raw_path']);requests=[r for r in raw['request_records'] if r['task']==target]
        S=sum(roles[r['model_family']]=='specialist' for r in requests);G=len(requests)-S
        duration=raw['measurement_window']['duration_seconds']
        rep.update(observed_target_S_requests=S,observed_target_G_requests=G,
                   observed_target_S_rps=S/duration,observed_target_G_rps=G/duration,
                   observed_f_S=S/len(requests),observed_f_G=G/len(requests),
                   predicted_worker_rho=rc0['rho'],safe_memory_mib=rc0['safe_memory_mib'],
                   N_G=rc0['N_G'],N_S=rc0['N_S'],specialist_replicas=rc0['specialist_replicas'])
    save('gpu_results.json',{'rows':gpu,'independent_physical_measurements':len(gpu),'exact_state_aliases_are_not_extra_observations':True})
    csvsave('gpu_results.csv',gpu)
    results=[];transitions=[]
    for row in planning:
        measured=[r for r in gpu if r['point']==row['point'] and row['method'] in r['methods_sharing_exact_label']]
        record={**row,'physical_repetitions':len(measured),'stability':aggregate['stability_state_v2'] if measured else None,
                'mean_e2e_ms':float(np.mean([r['mean_e2e_ms'] for r in measured])) if measured else None,
                'weighted_p95_ms':float(np.mean([r['weighted_p95_ms'] for r in measured])) if measured else None,
                'per_repetition_stability':[r['stability'] for r in measured],
                'per_task_mean_ms':{k:float(np.mean([r['per_task_mean'][k] for r in measured])) for k in measured[0]['per_task_mean']} if measured else None,
                'per_task_p95_ms':{k:float(np.mean([r['per_task_p95'][k] for r in measured])) for k in measured[0]['per_task_p95']} if measured else None,
                'GPU_measurement_status':'MEASURED_EXACT_SHARED_STATE' if measured else row['status']}
        results.append(record)
    csvsave('method_comparison.csv',results);save('method_comparison.json',{'rows':results,'RC_SG_v2_share_physical_observations':True,'V3_metrics_not_imputed':True})
    for method in ['RC_SG','v2','V3_MINIMAL']:
        rows=[r for r in results if r['method']==method]
        valid=[r for r in rows if r['physical_repetitions']==3]
        split=next((r for r in rows if r.get('f_G') is not None and r['f_G']>1e-9),None)
        two=next((r for r in rows if r.get('N_G') is not None and r['planning_feasible'] and r['N_G']>=2),None)
        unstable=next((r for r in valid if r['stability']=='unstable'),None)
        rejected=next((r for r in rows if not r['planning_feasible']),None)
        steady=[r['target_rate'] for r in valid if r['stability']=='steady']
        transitions.append({'method':method,'lambda_split':split['target_rate'] if split else None,'lambda_split_point':split['point'] if split else None,
                            'lambda_2G':two['target_rate'] if two else None,'lambda_unstable':unstable['target_rate'] if unstable else None,
                            'first_method_rejected_load':rejected['target_rate'] if rejected else None,'first_rejection_reason':rejected['status'] if rejected else None,
                            'max_tested_steady_target_rate':max(steady) if steady else None,'measured_hotspot_points':len(valid),
                            'breakpoint_scope':'first observed frozen H0-H3 point, not an exact continuous breakpoint; unmeasured points remain unknown'})
    csvsave('phase_transitions.csv',transitions);save('phase_transitions.json',{'rows':transitions})
    checks=read(OUT/'mechanism_planning_checks.json')['points'];detail=read(OUT/'planning/H0_V3_MINIMAL.json');h0=points[0]
    sroute=flat(primary['specialist_only_witness']['routing'])
    for pair in list(sroute):
        if pair[0]==target:sroute[pair]*=h0['target_rate']/cs
    forced_S=evaluate_surrogate(primary['specialist_only_witness']['deployment'],sroute,Workload(h0['lambda_rps']),c,cal)
    canonical=next(r for r in planning if r['point']=='H0' and r['method']=='v2')
    # Reasons for a pre-exhaustion split are explicit; no arbitrary fraction
    # threshold is introduced to redefine the user's mechanism check.
    attribution={'primary_known_space_feasible_deployments':read(OUT/'cpu_validation.json')['primary_memory_and_coverage_feasible_deployments'],
                 'deployment_attribution':'Only one memory-and-coverage-feasible deployment. RC-SG and v2 have identical canonical routes; no independent method superiority follows from their shared labels.',
                 'V3_attribution':'No returned frozen J routing, hence no V3 GPU performance comparison. The rejected solver iterate is diagnostic only; no fallback or tolerance change.',
                 'frozen_J_failure':detail['frozen_J_solver_diagnostic'],
                 'H0_forced_S_only_surrogate':forced_S,'H0_canonical_surrogate_seconds':canonical['J_seconds'],
                 'H0_pre_exhaustion_split':{'S_fraction':canonical['f_S'],'G_fraction':canonical['f_G'],'S_majority':canonical['f_S']>canonical['f_G'],
                                           'true_S_ceiling_not_exceeded':h0['target_rate']<cs,
                                           'forced_S_only_exceeds_method_rho_max':forced_S['z']>.8,
                                           'PREMATURE_GENERALIST_ROUTING':False,
                                           'reason':'Frozen rho_max=.8 requires some G routing at H0 despite rho1 Specialist headroom; canonical minmax additionally equalizes Phi/Whisper pressure. This is an explained pre-exhaustion split, not evidence of Specialist exhaustion.'},
                 'H1_H2_H3_rejection':{'cause':'frozen canonical seed/feasibility rho_max=.8','global_method_target_ceiling_rps':read(OUT/'cpu_validation.json')['independent_enumeration_LP_optima']['ALL_rho0.8'],
                                      'C_Sonly_rho1':cs,'C_ALL_rho1':ca,'workload_points_moved':False,'oracle_substituted_as_method':False},
                 'multi_G':'A second G cannot fit this primary memory budget. Larger-budget multi-G envelopes lack complete target-Specialist replica evidence and were excluded by the frozen mechanical eligibility rule.',
                 'mechanism_planning_checks':checks,'real_exhaustion_to_steady_overflow_validated':False}
    save('attribution.json',attribution)
    # Neither of the user's positive/no-window conditions holds: a certified
    # planning window exists, but H2/H3 cannot be measured through frozen methods.
    gate={'gate':'CURRENT_POOL_OVERFLOW_NOT_VALIDATED','CURRENT_POOL_OVERFLOW_SUPPORTED':False,'CURRENT_POOL_NO_OVERFLOW_WINDOW':False,
          'certified_planning_overflow_window_exists':True,'replica_audit_primary':'PASS','H2_H3_SPECIALIST_EXHAUSTED_planning':[r['point'] for r in checks if r['point'] in ['H2','H3'] and r['SPECIALIST_EXHAUSTED']],
          'H2_H3_real_joint_steady_extension_measured':False,'blocking_conditions':['Frozen rho_max=.8 makes H1-H3 method-planning infeasible','Frozen Module9 J solver rejects H0 convex certificate gap (12.243 ns > 10 ns)'],
          'Task_Rich_required':False,'Task_Rich_need_determined':False,'no_task_rich_started':True,'no_Module9_changes':True,
          'decision':'Do not equate method rejection with absence of an overflow window. No physical exhaustion-to-overflow support is claimed. Adding tasks is not justified by this diagnostic; resolving frozen method limits requires a separate user decision.'}
    save('module10_gate.json',gate)
    attempts=read(OUT/'gpu_attempts.json')['runs'];assert len(attempts)==len(gpu)==read(OUT/'gpu_manifest.json')['max_timed_runs']==3
    for a in attempts:
        verify([a['raw_ref']]);raw=read(a['raw_path']);assert raw['valid_execution'] and raw['measurement_window']
    verify(read(OUT/'input_freeze.json')['protected_references']);verify(read(OUT/'execution_freeze.json')['sources'])
    from replica_scaling.telemetry import snapshot
    assert not snapshot()['processes']
    save('validation_results.json',{'status':'PASS_WITH_FROZEN_METHOD_LIMITATIONS','CPU_checks':read(OUT/'cpu_validation.json'),'hotspot_rates_unchanged':True,
                                    'Module9_and_worker_runtime_hashes_unchanged':True,'MPS_enabled':False,'new_replica_profiles':0,'physical_GPU_runs':len(gpu),
                                    'timed_repetitions_per_measured_final_state':3,'shared_RC_v2_labels_not_independent':True,'outcome_based_reruns':0,'no_active_GPU_workers':True,
                                    'V3_failure_not_hidden_or_relabelled_as_capacity':True,'planned_but_infeasible_hotspots_not_labelled_unstable':True})
    rows=read(OUT/'capacity_envelope.json')['rows'];table=[]
    for con in contexts():
        rates=[]
        for k in sorted(read(OUT/'input_freeze.json')['eligible_tasks']):
            r=next(a for a in rows if a['context']==con['id'] and a['target']==k and a['mode']=='S_ONLY')
            rates.append(f"{r['capacity_rps']:.6f}"+('†' if not r['envelope_certified'] else ''))
        table.append('| '+con['id']+' | '+f"{con['memory_budget_mib']:.3f}"+' | '+' | '.join(rates)+' |')
    captable='\n'.join(table)
    pointtable='\n'.join(f"| {p['point']} | {p['target_rate']:.6f} | {p['above_C_Sonly']} | {p['formal_GPU_point']} |" for p in points)
    comparison='\n'.join(f"| {r['point']} | {r['method']} | {r['status']} | {r['stability'] or 'unmeasured'} | {r['mean_e2e_ms'] if r['mean_e2e_ms'] is not None else '—'} | {r['weighted_p95_ms'] if r['weighted_p95_ms'] is not None else '—'} |" for r in results)
    outcomes={r['method']:r for r in results if r['point']=='H0'}
    answers=[
      '1. Legal G/S overlap: ASR, image, LID, OCR, text. VQA has no admitted Specialist and was excluded mechanically.',
      '2. C_Sonly for all six frozen budgets and both I1/I2 backgrounds is tabulated below; full precision, maximizing deployments/routes/pressure and conditional scope are in capacity_envelope.csv. Values marked † are maxima only within current evidence-supported counts, not certified true exhaustion capacities.',
      '3. Yes, artificial/uncertified count caps exist in larger-memory contexts. Both local +1 and optimistic alternate background-memory allocations were audited. The primary context passes: its single Whisper-small is memory-constrained, not manually capped at x=1; an evidence-supported second replica cannot fit with mandatory backgrounds.',
      '4. No replica capacity profiling was needed for the mechanically selected certified context. No unprofiled g/count was extrapolated into any envelope or method. Other tasks/budgets remain visibly uncertified rather than being silently called exhausted.',
      f"5. Primary = {primary['context']['id']}, ASR, frozen mix_03 I1, safe B={primary['context']['memory_budget_mib']:.6f} MiB (planning input={primary['context']['planning_budget_input_mib']:.6f} MiB). No certified multi-G context exists; the largest legal R selects this context before any new hotspot GPU label.",
      f"6. C_Sonly={cs:.9f}, C_ALL={ca:.9f}, W={primary['W']:.9f} req/s, R={primary['R']:.9f}. Exact primary MILP optimum independently matches enumeration+LP over the 6,912 evidence-supported count vectors; only one vector fits memory and all-task coverage.",
      '7. H0-H3 and the unrun H4 are listed above. All non-target task rates remain exactly the frozen I1 values. No point was moved after results.',
      f"8. H0 RC-SG/v2 route {canonical['f_S']:.6%} of ASR to Whisper-small and {canonical['f_G']:.6%} to Phi. Specialist is the majority but not exclusive. H1 is unmeasured because frozen method planning fails. V3 has no valid returned route at H0; its rejected numerical iterate is not a method outcome.",
      f"9. First observed configured split for RC-SG/v2 is H0={h0['target_rate']:.9f} req/s, below C_Sonly. It is explained by frozen planning headroom/minmax load balancing and must not be presented as an exhaustion transition. V3 lambda_split is unavailable.",
      '10. No second G appears. The primary budget cannot hold two G replicas with backgrounds. Higher-budget multi-G analytical witnesses are not eligible without the missing Specialist replica audit evidence; they were not substituted into this benchmark.',
      f"11. RC-SG: H0 measured tier={outcomes['RC_SG']['stability']}, mean={outcomes['RC_SG']['mean_e2e_ms']:.6f} ms, weighted task P95={outcomes['RC_SG']['weighted_p95_ms']:.6f} ms. H1-H3 are method-planning infeasible. No real post-exhaustion outcome can be inferred.",
      '12. v2 returns the same unique deployment and canonical route; its three repetitions are exact reused RC-SG labels. It has no feasible neighborhood candidate. H1-H3 have no feasible seed; this is distinct from observed runtime instability.',
      '13. V3: H0 frozen J solve has valid constraints and solver_success=true but convex gap=1.224330937e-8 seconds exceeds the frozen 1e-8-second certificate tolerance; the frozen method rejects it. H1-H3 have no feasible seed under rho_max=.8. No V3 GPU mean/P95 or fallback route was fabricated.',
      '14. Stable interval superiority cannot be established: RC-SG/v2 share the same single measured point; V3 is unmeasured. phase_transitions.csv records first observed unstable load if present, separately from planning/solver rejection. Sparse-grid/censored transitions are not continuous breakpoint estimates.',
      '15. No. H2/H3 satisfy all four Specialist-exhaustion planning conditions (including replica audit), but no frozen method admits them through the .8 planning gate; there is no real joint steady extension measurement.',
      '16. The current 6-task pool has a legitimate certified planning overflow window. It has not yet demonstrated the requested physical exhaustion-to-overflow mechanism under the frozen methods. The result is CURRENT_POOL_OVERFLOW_NOT_VALIDATED, not CURRENT_POOL_NO_OVERFLOW_WINDOW.',
      '17. A Task-Rich Benchmark is not justified as a necessary next step by these results. The concrete obstructions are frozen solver numerical acceptance and planning headroom, not proof that the task pool lacks overflow. Module9 was left untouched and no new task/benchmark started.'
    ]
    report=f'''# Module 10 — Specialist exhaustion and Generalist overflow\n\nCompleted and stopped under the supplied freeze. Gate: **CURRENT_POOL_OVERFLOW_NOT_VALIDATED**. A certified planning window exists; the required physical overflow extension could not be tested through frozen methods. This is neither a positive mechanism claim nor a no-window claim.\n\n## Frozen hotspot points\n\n| Point | ASR req/s | Above C_Sonly | Formal GPU point |\n|---|---:|---|---|\n{pointtable}\n\n## Method outcomes\n\n| Point | Method | Planning/search status | Real tier | Mean E2E ms | Task-weighted P95 ms |\n|---|---|---|---|---:|---:|\n{comparison}\n\nOnly 3 new valid timed GPU runs were used, shared by exact RC-SG/v2 final-state identity. Three distinct seeds, unchanged 10-second warmup and 210-second measurement, original pool/worker/runtime. Numerical/structural failures have no fake latency labels. No outcome reruns.\n\n## Required answers\n\n'''+ '\n\n'.join(answers)+f'''\n\n## All-budget Specialist-only envelopes\n\n| Context | Safe budget MiB | ASR | image | LID | OCR | text |\n|---|---:|---:|---:|---:|---:|---:|\n{captable}\n\nAll rates are req/s. † = provisional maximum within known replica evidence; not a certified true Specialist exhaustion envelope. There is no global no-overflow conclusion from these uncertified contexts. All CPU solves use D<=g at the actual planning capacity ceiling. The separate frozen method policy uses rho_max=.8 and is never folded silently into C_Sonly.\n\n## Attribution and limitations\n\nThe primary safe budget equals its mandatory minimum admitted background deployment. Whisper-small x=2 (whose g and memory are already measured) does not fit; absent target-Specialist alternatives and missing higher counts also fail the optimistic background-memory cover. Thus x_S=1 arises from memory, not a imposed software cap. No identity was deleted, speed reduced, memory inflated, quality changed, or model pool changed. Larger budgets are marked uncertified even when a locally full maximizing state cannot fit a +1: another deployment could free memory for an unprofiled count.\n\nAt the fixed H0, a forced S-only route has predicted max rho={forced_S['z']:.9f}, above the method's .8 headroom policy but below the true planning capacity ceiling 1. Canonical routing therefore produces an explained pre-exhaustion split. Global method target capacity is only 4.347876367 req/s, below H1-H3, despite C_ALL=5.576645355. J routing is blocked independently by the frozen convex-gap acceptance check. The rejected J iterate is diagnostic only. This exposes method limits without changing them.\n\nAn initial CPU MILP lexicographic-tie conditioning failure was retained; normalizing memory coefficients and a fixed 1e-6 req/s solver-feasibility tolerance repaired the audit only. Primary capacities are independently verified by LP and repeated MILP. A CPU-only driver serialization correction occurred before any GPU attempt; no worker or serving label existed at that time. No Module9 source/artifact, native worker, runtime, MPS, hardware, quality, task, budget, or hotspot-point change. All protected references reverified.\n'''
    (OUT/'module10_summary.md').write_text(report)
    save('benchmark_status.json',{'status':'COMPLETE_AND_STOPPED','at_utc':now(),'gate':gate['gate'],'no_active_GPU_workers':True})
    save('artifact_hashes.json',{str(p.relative_to(OUT)):ref(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['artifact_hashes.json','gpu_execution.lock']})
    print(json.dumps({'COMPLETE_AND_STOPPED':True,'gate':gate['gate'],'physical_GPU_runs':len(gpu),'no_Module9_changes':True}),flush=True)
if __name__=='__main__':main()
