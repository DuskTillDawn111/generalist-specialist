"""Mechanical first missing frontier and fixed-cost plan, before new GPU labels."""
from collections import defaultdict
from candidate_pool_service_demand.prepare import targets
from replica_scaling.telemetry import snapshot
from .common import *

def main():
    assert not (OUT/'input_freeze.json').exists(),'Never replace the premeasurement freeze'
    c,hw,roles,cal=inputs();old=read(OLD/'input_freeze.json');verify(old['protected_references'])
    refs=old['protected_references']+[ref(p) for d in ['module11','results/module11_multihotspot','replica_scaling','integer_replica_pilot'] for p in (ROOT/d).rglob('*') if p.is_file() and p.suffix!='.pyc']
    refs=list({r['path']:r for r in refs}.values());verify(refs)
    for r in refs:
        s=Path(r['path']).stat();r.update(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    ctx=read(ROOT/'results/module8f2_resource_static/contexts/C01.json');gpu=snapshot()
    previous=read(ROOT/'results/module10_specialist_saturation/execution_freeze.json')['GPU']
    assert all(gpu[k]==previous[k] for k in ['uuid','name','driver','total_mib'])
    assert not gpu['processes'] and gpu['used_mib']<=128
    fits=read(OLD/'capacity_witnesses.json')['rows'];struct=read(OLD/'structural_witnesses.json')['rows']
    budgets=read(OLD/'budget_structure_audit.json')['budgets'];bybudget={b['budget']:b for b in budgets}
    evidence=defaultdict(lambda:{'contexts':set(),'boundaries':set(),'reasons':set(),'witnesses':[]})
    demand=defaultdict(lambda:defaultdict(float))
    for f in fits:
        for p,v in f['routing'].items():
            k,m=p.split('::');demand[m][k]+=v*c[m].service_time_ms[k]/1000
        for m,n in f['deployment'].items():
            if not n or n!=cal.evidence_bounds[m]:continue
            if f['rho'].get(m,0)<.8-1e-8:continue
            if roles[m]=='generalist' and n>=2:continue
            screen=next(s for s in bybudget[f['budget']]['missing_count_screen'] if s['model']==m)
            if not screen['can_fit_memory_screen']:continue
            key=(m,n+1);a=evidence[key];a['contexts'].add(f['budget']+'/'+f['set_id']);a['boundaries'].add('alpha_'+f['mode'])
            a['reasons'].add('ACTIVE_CAPACITY_AT_MEASURED_FRONTIER_AND_NEXT_COUNT_MEMORY_SCREEN_CAN_FIT')
            a['witnesses'].append({'budget':f['budget'],'set_id':f['set_id'],'boundary':f['mode'],'rho':f['rho'][m],
                'global_memory_screen_mib':screen['minimum_coverage_memory_screen_mib']})
    for s in struct:
        for a in s['local_specialist_expansion_alternatives']:
            if a['g_measured'] or not a.get('memory_screen_feasible'):continue
            m=a['model'];key=(m,a['next_count']);e=evidence[key]
            e['contexts'].add(s['budget']+'/'+s['set_id']);e['boundaries'].add('near_alpha_1G')
            e['reasons'].add('NEAR_BOUNDARY_EXTRA_SPECIALIST_REPLICA_CAN_FIT_AND_MAY_CHANGE_OPTIMUM')
    # Do not hide alternate admitted Specialist identities: a two-G window cannot
    # be certified if its target Specialist's first missing count fits after a
    # legal memory reallocation. This is directly tied to the recorded window.
    windows=read(OLD/'windows.json')['rows']
    for w in windows:
        if w['W_2G']<=0:continue
        for screen in bybudget[w['budget']]['missing_count_screen']:
            m=screen['model']
            if roles[m]!='specialist' or not screen['can_fit_memory_screen'] or not set(w['tasks'])&c[m].supported_tasks:continue
            e=evidence[(m,screen['first_missing_count'])];e['contexts'].add(w['budget']+'/'+w['set_id'])
            e['boundaries'].update(['alpha_S','alpha_1G','alpha_2G'])
            e['reasons'].add('CONDITIONAL_2G_WINDOW_DEPENDS_ON_UNMEASURED_TARGET_SPECIALIST_FRONTIER')
    required=[]
    for (m,x),e in sorted(evidence.items()):
        # Count table prefixes must be contiguous. A required x=2 for a model
        # whose measured frontier is 2 is already evidenced, not a new run.
        if x<=cal.evidence_bounds[m]:continue
        primary=cal.model_evidence[m].get('task') or min(c[m].supported_tasks,key=lambda k:(-demand[m][k],k))
        required.append({'model':m,'current_max_measured_x':cal.evidence_bounds[m],'required_x':x,
            'affected_contexts':sorted(e['contexts']),'affected_boundary':sorted(e['boundaries']),
            'reason':sorted(e['reasons']),'primary_task':primary,'related_admitted_tasks':sorted(c[m].supported_tasks),
            'active_constraint_witnesses':e['witnesses'],'only_next_frontier_not_full_sweep':True})
    assert {r['model'] for r in required}==set(c)-{'qwen'},[r['model'] for r in required]
    csvsave('replica_evidence_required.csv',required);save('replica_evidence_required.json',{'rows':required})
    pool=OUT/'frozen_samples.jsonl';pool.parent.mkdir(parents=True,exist_ok=True)
    pool.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for k in sorted(old['background']) for r in targets(k)))
    seeds=read(ROOT/'results/module8e2_heldout_freeze/method_comparison_freeze.json')['search_seeds']
    ordering=read(ROOT/'results/module8f3_structural_dominance/local_audit_results.json')['fastest_unique_specialists']
    assert len(ordering)==5 and all(roles[m]=='specialist' for m in ordering)
    residents=[]
    for g in ['phi4mm','qwen']:
        for n in range(6):
            x={g:1,**{m:1 for m in ordering[:n]}};mem=factor()*sum(cal.memory_table_mib[m][v] for m,v in x.items())
            assert mem<=hw.gpu_memory_mib
            residents.append({'set_id':g+f'_plus_{n}S','deployment':x,'N_G':1,'N_S':n,'S_order_prefix':ordering[:n],
                'safe_memory_mib':mem,'G_workload_task':'visual_qa'})
    for n in [0,5]:
        x={'phi4mm':1,'qwen':1,**{m:1 for m in ordering[:n]}};mem=factor()*sum(cal.memory_table_mib[m][v] for m,v in x.items())
        if mem<=hw.gpu_memory_mib:residents.append({'set_id':f'phi_qwen_plus_{n}S','deployment':x,'N_G':2,'N_S':n,
             'S_order_prefix':ordering[:n],'safe_memory_mib':mem,'G_workload_task':'visual_qa'})
    save('resident_set_manifest.json',{'frozen':True,'at_utc':now(),'sets':residents,'specialist_order':ordering,
        'order_source':ref(ROOT/'results/module8f3_structural_dominance/local_audit_results.json'),
        'memory_budget_mib':hw.gpu_memory_mib,'repetitions':3,'seeds':seeds,
        'low_load_protocol':{'Generalist_total_rate_rps':.05/max(c[g].service_time_ms['visual_qa']/1000 for g in ['phi4mm','qwen']),
            'Generalist_nominal_pressure_ceiling':.05,'requests_per_G_phase_per_repetition':30,
            'specialist_nominal_pressure_each_when_active':.001,
            'S_task_rule':'largest frozen background service demand, tie task ID',
            'idle_and_active_S_phases':'same G arrival rates and same seeded G sample stream; phase order alternates by seed',
            'idle_observation_seconds':10,'measurement_phase_seconds':'30 / fixed Generalist total rate',
            'no_queueing_rule':'retain measured backlog/queue evidence; congested phases excluded from idle-cost interference inference, never retune rate'},
        'CUDA_context_count':'NVML compute PIDs only; exact CUDA contexts per PID unavailable',
        'shared_G_workload':'visual_qa quality-admitted on both resident Generalists; no new task'})
    plan={'at_utc':now(),'GPU':gpu,'registry':ctx['registry'],'models':ctx['models'],'required_initial_frontier':required,
         'sample_pool':ref(pool),'source_pool':ref(Path(ctx['input_pool'])),'seeds':seeds,'repetitions':3,
         'warmup_seconds':5,'measurement_seconds':30,'drain_timeout_seconds':60,'startup_timeout_seconds':300,
         'shutdown_timeout_seconds':15,'telemetry_interval_seconds':.5,'idle_observation_seconds':10,
         'high_rate_rule':'(previous empirical g + 1.2) / frozen s0 seconds; 1.2 is original 2.2x high-load minus g(1), fixed paired offered rate for previous and new count',
         'g_point_rule':'g_new = unchanged g_previous * median of three paired within-window throughput(new)/throughput(previous) ratios at matched rate and seed',
         'sequential_termination':{'user_authorized':True,'no_count_or_model_unrelated_to_breakpoint':True,
             'memory':'measured safe residency plus minimum admitted background cover exceeds every affected frozen budget; retain measured raw count',
             'no_repeatable_positive_increment':'all three full, capacity-pressured matched pairs have throughput(new)<=throughput(previous); a mixed-sign result is inconclusive and cannot stop the frontier',
             'low_load_unstable':'all three lower-load full windows unstable or repeatable count-specific load/runtime failure; high-load overload alone never excludes a count',
             'higher_counts_after_true_terminal':'terminal-evidenced, not artificial missing; never fill their g/M tables'},
         'primary_task_rule':'historical measured scaling task if present; otherwise largest summed service demand in frozen 11A capacity witnesses',
         'other_task_probe':'3 serial samples per replica on every other admitted path; separate from capacity window',
         'capacity_scope':'same frozen family scalar / representative-path transfer convention; probes do not establish mixed-task capacity',
         'protective_stop':read(ROOT/'results/replica_scaling_real/replica_scaling_plan.json')['protective_stop'],
         'no_retries':True,'no_science_threshold_changes':True,'no_formal_hotspot_campaign':True,
         'history_hardware_note':'Historical Module7 scaling may originate on a different GPU UUID; keep old points immutable; every new adjacent pair measured on current frozen Module10 hardware.',
         'environment_overrides':{**read(ROOT/'results/module8e2_heldout_freeze/method_comparison_freeze.json')['execution_environment_overrides'],
             'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}}
    save('measurement_plan.json',plan)
    save('input_freeze.json',{'at_utc':now(),'protected_references':refs,'measurement_plan':ref(OUT/'measurement_plan.json'),
        'resident_set_manifest':ref(OUT/'resident_set_manifest.json'),'samples':ref(pool),
        'normalized_directions_preserved':ref(OLD/'normalized_directions.json'),'budgets_and_sets_preserved':ref(OLD/'input_freeze.json'),
        'Module9_and_old_calibration_updates':False,'hotspot_results_updates':False,'GPU_authorized_scope':'Module11B characterization only'})
    print(json.dumps({'required_counts':[(r['model'],r['required_x']) for r in required],'resident_sets':len(residents),'initial_profile_sessions':6*len(required),'fixed_cost_sessions':3*len(residents)}),flush=True)

if __name__=='__main__':main()
