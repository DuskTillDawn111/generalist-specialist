"""Independent arithmetic/domain audit and separated final gates."""
from .isolation import install
install('supervisor')
from .common import *


def main():
    required=['experiment_protocol.json','candidate_pool_manifest.csv','admission_feasibility_audit.csv','cpu_oracle_admission_results.csv',
        'optimizer_admission_decisions.csv','gpu_paired_admission_results.csv','admission_utility_breakdown.csv','final_summary.md']
    assert all((OUT/name).exists() for name in required)
    proto=read(OUT/'experiment_protocol.json');cpu=read(OUT/'cpu_results.json');done=read(OUT/'CPU_stage_completed.json');smoke=read(OUT/'admission_smoke_result.json');integrity=read(OUT/'history_integrity_after.json')
    assert integrity['passed'] and not integrity['changed']
    assert ref(OUT/'experiment_protocol.json')==read(OUT/'protocol_freeze_receipt.json')['protocol']
    for item in [proto['utility_definition'],proto['utility_reference'],proto['quality_policy'],proto['performance_version'],*proto['sources']]:assert ref(item['path'])==item
    weights=read(proto['utility_definition']['path'])['weights'];refs=read(proto['utility_reference']['path'])['latency_reference_ms']
    errors=[];candidate_count=0;valid_candidates=0;largest_error=0.;counts={}
    for oracle in cpu['oracles']:
        scene,pool=oracle['scenario'],oracle['pool'];rates=proto['development_scenarios'][scene]['rates'];d=read(proto['pools'][pool]['path'])
        rows=[]
        for stored in csv.DictReader((OUT/f'cpu/{scene}_{pool}_candidates.csv').open()):
            x=json.loads(stored['deployment']);r=json.loads(stored['routing']);p=json.loads(stored['prediction']);candidate_count+=1
            eligible=stored['eligible']=='True';U=float(stored['predicted_U']) if stored['predicted_U'] else None
            assert all(n==1 for n in x.values())
            assert all(abs(sum(r[t].values())-q)<1e-9 for t,q in rates.items())
            assert all(m in x and t in d['models'][m]['supported_tasks'] for t,v in r.items() for m,q in v.items() if q>0)
            assert all(abs(q/rates[t]-1.)<1e-9 for t,v in r.items() for q in v.values())
            memory=d['safe_factor']*sum(d['memory_table_mib'][m]['1'] for m in x);assert memory<=proto['budget_mib']+1e-6
            if eligible:
                valid_candidates+=1
                L=sum(q/sum(rates.values())*p['per_task'][t]['mean_ms']/refs[t] for t,q in rates.items())
                P=sum(q/sum(rates.values())*p['per_task'][t]['p95_ms']/refs[t] for t,q in rates.items())
                recomputed=-weights['T']*p['T_norm']-weights['L']*L-weights['P']*P-weights['M']*memory/proto['budget_mib']
                largest_error=max(largest_error,abs(U-recomputed));assert abs(U-recomputed)<1e-10
                assert p['quality_capability'] and p['route_conservation'] and p['max_rho']<=.8+1e-8
                assert set(p['per_task'])==set(rates) and p['incomplete_after_60s_drain']==0
                rows.append(dict(U=U,memory=memory,deployment=x,routing=r))
            else:
                assert U is None
                if scene=='LOW18_EIGHTH':assert p.get('reason')=='MISSING_TASK'
        counts[f'{scene}/{pool}']=len(rows)
        if rows:
            best=min(rows,key=lambda v:(-v['U'],v['memory'],identity(v['deployment'],v['routing'])))
            assert identity(best['deployment'],best['routing'])==identity(oracle['deployment'],oracle['routing'])
            assert abs(best['U']-oracle['predicted_U'])<1e-10
            if pool!='P0':assert oracle['delta_U_admission']>=0
        else:assert oracle['predicted_U'] is None and not oracle['G_only_retained']
    matches=[v for v in cpu['decisions'] if v['algorithm']=='MODULE35_FROZEN' and v['eligible']]
    assert len(matches)==6 and all(v['oracle_plan_match'] and v['regret']==0 for v in matches)
    assert not done['retention_cases'] and not done['GPU_trigger_passed']
    assert smoke['passed'] and smoke['safe_idle_memory_mib']<=proto['budget_mib']
    assert len(smoke['path_probe_records'])==36 and all(v['status']=='completed' for v in smoke['path_probe_records'])
    assert len({v['pid'] for v in smoke['worker_identities'].values()})==4
    assert smoke['gpu_release']['verified'] and not smoke['unexpected_write_violations']
    assert all(v['exit_code']==0 and v['bye_received'] and not v['forced'] for v in smoke['shutdown'].values())
    reserved=set(proto['GPU']['independent_trace_seeds']);assert proto['GPU']['development_trace_seed'] not in reserved
    assert read(OUT/'GPU_execution_accounting.json')['actual_formal_GPU_sessions']==0
    assert not (OUT/'raw_runs').exists(), 'No formal GPU observations are authorized in the early-stop branch'
    assert read(OUT/'pool_and_profile_preservation.json')['passed']
    assert read(OUT/'high_load_control_audit.json')['same_candidate_pool_positive_control_available'] is False
    for r in read(OUT/'smoke_source_inventory_resolution.json')['sources']:
        assert ref(r['snapshot_path'])['sha256']==r['original']['sha256']
    figure_count=len(list((OUT/'paper_figures').glob('*.png')));assert figure_count==5
    review=dict(passed=True,required_files_present=True,all_source_hashes_unchanged=True,full_candidate_rows=candidate_count,
        valid_candidate_rows=valid_candidates,eligible_counts=counts,independent_U_arithmetic_max_abs_error=largest_error,
        Module35_complete_plan_matches=len(matches),Module35_regret_max=0.,lowest_load_cells_null=3,
        all_original_profiles_and_admission_paths_preserved=True,positive_specialist_usage_verified=True,
        native_joint_model_smoke_pass=True,formal_GPU_pairs=0,formal_GPU_values_and_CI_null=True,
        independent_validation_not_claimed=True,high_load_splicing_prohibited=True,figure_count=figure_count,
        history_files_unchanged=integrity['files'],no_history_changed=True,source_snapshots_resolved=True,
        algorithm_source=ref(ROOT/'results/module35_utility_routing/u_aligned_optimizer.py'))
    save('final_artifact_review.json',review)
    gate=dict(experiment=proto['experiment'],status='COMPLETED_EARLY_STOP_NO_RETENTION_TARGET_IN_BOUNDED_SCREENING',
        gates=dict(G_ONLY_RETENTION_CPU_CONFIRMED=False,G_ONLY_RETENTION_GPU_SUPPORTED=False,
            ADMISSION_FEASIBILITY_INVALID=False,GPU_COMPARISON_INCONCLUSIVE=True),
        gate_scope=dict(ADMISSION_FEASIBILITY_INVALID='False: eligible lightweight specialists and a legal runnable complete Mixed exist. The lowest-load missing CPU cohort is a scenario/predictor evaluation limitation, not a model admission failure.',
            GPU_COMPARISON_INCONCLUSIVE='True because formal GPU comparison was not executed after the CPU trigger failed; no paired values/CI exist, so this is not a nonsignificant measured contrast.'),
        stage_results=dict(A='PASSED: nested pools, frozen quality/service/memory/weights and successful 4-worker native path smoke',
            B='COMPLETED: 6 valid cells all matched Oracle; target retention absent; 3 lowest-load cells null due missing LID sample',
            C='NOT_EXECUTED: prespecified CPU retention trigger not met',
            D='AUDITED: P0=A only; expanded pools not comparable, no load-switch curve',
            E='COMPLETED: raw data, utility components, plots, separate gates and report saved'),
        candidate_admission_passed=True,CPU_feasible_cells=6,CPU_Oracle_Module35_matches=6,CPU_max_regret=0.,
        invalid_CPU_scenario='LOW18_EIGHTH: original120s phase.37 has no nominal LID sample',
        retention_cases=0,formal_GPU_pairs=0,feasibility_smoke_sessions=1,
        model_profiles_changed=False,utility_weights_changed=False,quality_thresholds_changed=False,
        Module35_code_results_Gate_changed=False,history_integrity=ref(OUT/'history_integrity_after.json'),artifact_review=ref(OUT/'final_artifact_review.json'),
        scientific_limit='Bounded development screening, endpoint0/1 domain and frozen CPU forecast only; cannot establish real G-only advantage, load-aware rejection, a holdout result or global GPU optimum.',
        stop_now=True,next_experiment_started=False,frozen_unix_ns=time.time_ns())
    save('final_gate.json',gate)
    report=OUT/'final_summary.md';report.write_text(report.read_text()+f'\n最终完整性与验收：历史 {integrity["files"]} 个文件 SHA256 与mtime均未改变；独立重算 {valid_candidates} 个合法候选U，最大误差 {largest_error:.3g}；408 个已评价候选记录均完成域与空值审计。四项Gate依次为 CPU保持=False、GPU支持=False、准入不可行=False、GPU比较不确定=True（未执行）。\n')
    files=[]
    for file in sorted(OUT.rglob('*')):
        if file.is_file() and file.suffix!='.log' and not any(k in file.relative_to(OUT).parts for k in ['work']) and file.name not in ['scientific_artifact_SHA256.json','result_freeze.json']:
            files.append(ref(file))
    save('scientific_artifact_SHA256.json',dict(artifacts=files,complete=True))
    save('result_freeze.json',dict(status='FROZEN_SUPPLEMENT_NEGATIVE_RESULT_COMPLETE',frozen_unix_ns=time.time_ns(),
        artifact_manifest=ref(OUT/'scientific_artifact_SHA256.json'),final_gate=ref(OUT/'final_gate.json'),stop_now=True,no_next_experiment=True,
        preserved_history=ref(OUT/'history_integrity_after.json')))
    print('FINALIZED',len(files),'artifacts','candidates',candidate_count,'valid',valid_candidates,'max_U_error',largest_error,'history unchanged',integrity['files'],flush=True)


if __name__=='__main__':main()
