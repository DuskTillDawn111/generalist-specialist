import ast, csv, json, math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from module8c_final_real_search.common import ROOT, B, P, OUT as C, inputs, read, ref, verify_freeze
from deployment_search.runtime_cache import save_json, digest
from deployment_search.result import jsonable
from deployment_core import Workload
from deployment_core.milp_seeder import MilpDeploymentSeeder, validate_seed
from deployment_core.replica_calibration import ReplicaPlanningPolicy
from deployment_core.replicas import replica_statistics
from deployment_search import IntegerGuidedDeploymentSearch, SearchConfig
from deployment_search.routing import solve_canonical_routing

OUT = ROOT / 'results/module8d_baseline_freeze'
TASKS = ('text','image','ocr','asr','visual_qa','lid')


def freeze_file(name, value):
    path = OUT / name
    if path.exists():
        assert read(path) == jsonable(value), f'Immutable baseline definition changed: {path}'
    else:
        save_json(path, value)
    # The definition, random seeds and aggregate are hashed before planning.
    from deployment_search.runtime_cache import file_hash
    sidecar = path.with_suffix(path.suffix + '.sha256')
    expected = file_hash(path) + '\n'
    if sidecar.exists():
        assert sidecar.read_text() == expected
    else:
        sidecar.write_text(expected)
    return path


def csvwrite(name, rows, columns=None):
    columns = columns or list(rows[0])
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,columns,extrasaction='ignore');writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def build():
    verify_freeze()
    assert read(C/'campaign_status.json')['status']=='complete'
    OUT.mkdir(exist_ok=True)
    catalog, hardware, roles, cal = inputs()
    policy = ReplicaPlanningPolicy(catalog,'empirical','empirical',cal)
    bounds,_=policy.bounds(hardware)
    full_ids=sorted(read(P/'candidate_pool_v2_1.json')['formal_model_ids'])
    admission=read(B/'quality_admission_matrix.json')
    admitted={(r['task'],r['model']):r for r in admission['rows'] if r['quality_admissible']}
    matrix=read(B/'final_benchmark_workload_matrix.json')
    workloads={p['name']:p['lambda_rps'] for p in matrix['proposals']}
    protocols={name:read(C/name/'protocol.json') for name in workloads}
    protocol=next(iter(protocols.values()))
    for p in protocols.values():
        for key in ('search_config','measurement_settings','search_seeds','confirmation_seeds','confirmation_order','objective'):
            assert p[key]==protocol[key]
    history=ROOT/'deployment_search/multiworkload.py'
    historical_random=next(ast.literal_eval(n.value) for n in ast.parse(history.read_text()).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RANDOM_SEEDS' for t in n.targets))
    assert list(historical_random.values())==[1103,2207]
    random_seeds=[1103,2207,47]
    sources={key:ref(path) for key,path in {
        'candidate_pool':P/'candidate_pool_v2_1.json','quality_policy':B/'quality_policy.json',
        'quality_admission':B/'quality_admission_matrix.json','service_demand':P/'service_demand_v2_1.json',
        'workload_matrix':B/'final_benchmark_workload_matrix.json',
        'capacity':ROOT/'results/replica_capacity_calibration/replica_capacity_factors.json',
        'memory':ROOT/'results/replica_capacity_calibration/replica_memory_table.json',
        'bounds':ROOT/'results/replica_capacity_calibration/evidence_replica_bounds.json',
        'quality_matrix':ROOT/'results/quality_workload_audit/quality_matrix.json',
        'admission_quality_matrix_csv':ROOT/'results/quality_workload_audit/quality_matrix.csv',
        'environment':C/'environment_manifest.json','model_sources':C/'model_source_hashes.json',
        'historical_random_policy':history}.items()}
    definitions={'module':'8D-1','frozen':True,'definition_inputs':sources,
       'definition_order':'Rules frozen before CPU deployment generation and reuse/outcome inspection',
       'outcome_data_used_for_definition':False,
       'B0':{'name':'Seed-only','seed':'Exact Module 8C quality-constrained empirical integer MILP seed, sourced from pre-GPU planning_preflight',
             'routing':'Exact seed canonical LP routing','deployment_search':False},
       'B1':{'name':'Qwen-centric + Quality Repair','deployment_semantics':'FIXED_ALWAYS_ON_FULL_SIX_TASK_REPAIR',
             'qwen_count':1,'default':'Every admitted Qwen task exclusively routes to Qwen',
             'repair':'Cover remaining six-task universe with minimum-cardinality admitted Specialist set; tie-break by total frozen s0 on repaired tasks, then sorted model IDs',
             'inactive_tasks':'Retain fixed deployment; zero traffic; do not remove Qwen or repairs'},
       'B2':{'name':'Specialist-greedy + Minimal Generalist','active_task_rule':'Specialist argmin s0 among admitted paths; if absent, Generalist argmin s0; exact ties by model ID',
             'replicas':'One replica for each distinct selected model','routing':'100% task traffic to selected model; not canonical redistribution',
             'inactive_tasks':'Excluded from selection; N_G=0 allowed if all active tasks have Specialist coverage'},
       'B3':{'name':'Static Hybrid','aggregate_lambda':'Per-task math.fsum of six frozen lambda values divided by six; no rounding',
             'deployment':'Existing quality-constrained empirical integer MILP seeder at aggregate lambda; minimum empirical memory then replica count',
             'routing':'Canonical empirical LP separately for each original workload, fixed deployment',
             'inactive_tasks':'Retain static deployment; no deployment pruning','real_runtime_search':False},
       'B4':{'name':'Random-neighbor + Same Runtime Budget','constructor':{
             'class':'IntegerGuidedDeploymentSearch','selection_policy':'random','transition_policy':'runtime','predictor':None},
             'candidate_set':'Exactly the existing x±1 neighborhood and feasibility/tabu/memory/coverage/canonical-capacity filters',
             'prioritization':'Existing random.Random(seed).sample(sorted(feasible,key=model),len(feasible)); first top_b; no delta_z/LP rank selection',
             'LP_allowed':'Same bottleneck/neighborhood definition, feasibility and canonical routing; delta_z may be logged but not used for selection',
             'random_seeds':random_seeds,'search_config':protocol['search_config'],
             'search_repetitions':3,'measurement_seeds':protocol['search_seeds'],
             'confirmation_repetitions_per_seed_and_final':2,'confirmation_seeds':protocol['confirmation_seeds'],
             'confirmation_order':protocol['confirmation_order'],'objective':protocol['objective'],
             'runtime_evaluator':'Existing ReplicaRuntimeEvaluator; no-VQA uses the same frozen transport-compatibility runner as Module 8C',
             'cached_evaluations_count_against_search_budget':True,
             'reporting':'Report all three fixed RNG seeds individually and their predeclared equal-weight mean/median/range; no best-of-seeds or best-of-baselines',
             'fairness_unit':'One workload × one RNG seed has exactly Proposed budget; three-seed totals are explicitly 3×, never pooled into a larger search'},
       'measurement_protocol':{'settings':protocol['measurement_settings'],'measurement_seeds':protocol['search_seeds'],
             'fixed_baseline_independent_confirmation_seeds':protocol['confirmation_seeds'],
             'B0_to_B3':'3 timed repetitions + 2 independent confirmation repetitions per fixed state; no search',
             'B4':'≤6 valid unique states including seed ×3, seed and final separately ×2 confirmations; same max_iterations/patience as Proposed',
             'metric':'Existing equal-repetition arrival-weighted mean E2E and weighted task P95; unchanged stability/objective',
             'harness_by_workload':{n:'module8c_runtime_compat.runtime' if n=='no_vqa_phase' else 'integer_replica_pilot.runtime' for n in workloads},
             'runtime_context_by_workload':{n:ref(C/n/'runtime_context.json') for n in workloads},
             'execution_environment_overrides':read(C/'startup_recovery.json')['runtime_environment_overrides'],
             'source_hash_gate':'Recheck environments, source revisions/weights, handlers, precision/backend and all frozen inputs before any future measurement'},
       'reuse_policy':{'all_12_identity_conditions_required':True,'exact_floats_no_rounding_no_tolerance':True,
             'measurement_seed_and_phase_contract_required':True,'forbid_seed_final_confirmation_overlap_within_B4_run':True,
             'donor_selection':'Required search seeds 7101/7202/7303 and confirmation seeds 8101/8202; priority source Module8C then historical, seed-role evidence before final-role evidence; lexicographic paths; never latency values',
             'seed_evidence_shared_between_random_rng_seeds':True,
             'correlation_reporting':'Shared raw runs are one measurement, not independent additional repetitions',
             'unknown_B4_future_states':'Reuse only after full exact identity check; no guessed trajectories or budget extension'},
       'quality_reporting':'Task-wise routing-fraction weighted frozen audit metrics, report-only; no cross-task universal average, no runtime objective changes',
       'no_GPU_experiments_in_this_module':True}
    freeze_file('baseline_definitions.json',definitions)
    freeze_file('random_seed_freeze.json',{'random_seeds':random_seeds,'source':ref(history),
       'historical_random_policy':historical_random,'third_seed_origin':'47 from user example; frozen before any new real baseline results',
       'project_has_historical_two_seed_campaign_not_three_seed_global_policy':True,
       'RNG':'Python random.Random; new RNG initialized to this seed for each workload/run',
       'measurement_random_seeds_are_separate':protocol['search_seeds'],'additional_random_seeds_allowed':False})
    aggregate={task:math.fsum(r[task] for r in workloads.values())/len(workloads) for task in TASKS}
    freeze_file('aggregate_reference_workload.json',{'lambda_rps':aggregate,'lambda_sha256':digest(aggregate),
       'source_workload_matrix':sources['workload_matrix'],'operation':'Per-task arithmetic mean of all six frozen workloads, including no-VQA zero',
       'selected_without_runtime_results':True})
    def full(x):return {m:x.get(m,0) for m in full_ids}
    def canonical(x,rates):
        result=solve_canonical_routing(x,Workload(rates),catalog,.8,replica_bounds=bounds,capacity_mode='empirical',calibration=cal)
        assert result is not None
        return result
    def nested(route):return {t:{m:route.get((t,m),0.) or 0. for m in full_ids} for t in TASKS}
    def solve_seed(rates):
        result=MilpDeploymentSeeder().solve(catalog,Workload(rates),hardware,.8,replica_bounds=bounds,
              capacity_mode='empirical',memory_mode='empirical',calibration=cal)
        assert result.feasible
        assert not validate_seed(result,catalog,Workload(rates),hardware,.8,bounds,planning_policy=policy)
        return result
    # Current repair universe has exactly one task and one legal Specialist.
    repairs={t for t in TASKS if (t,'qwen') not in admitted}
    assert repairs=={'ocr'}
    repair_options={t:sorted(m for (k,m) in admitted if k==t and roles[m]=='specialist') for t in repairs}
    assert repair_options=={'ocr':['paddleocr']}
    b1_selected={t:'qwen' if (t,'qwen') in admitted else repair_options[t][0] for t in TASKS}
    b1_x={m:int(m in set(b1_selected.values())) for m in catalog}
    static_seed=solve_seed(aggregate);static_x=dict(static_seed.deployment)
    static_route=canonical(static_x,aggregate)
    freeze_file('static_hybrid_planning_witness.json',{'seed':static_seed,'aggregate_canonical':static_route,
       'full_deployment':full(static_x),'aggregate_source':ref(OUT/'aggregate_reference_workload.json')})
    states=[];random_initial=[];quality_rows=[]
    def add_state(baseline,name,x,route,origin,selected=None,seed=None):
        rates=workloads[name];x=dict(x);rr=nested(route)
        assert policy.deployment_memory(x)<=hardware.gpu_memory_mib
        pressure={m:math.fsum(rr[t][m]*catalog[m].service_time_ms.get(t,0)/1000 for t in TASKS)/policy.capacity(m,x[m]) if x[m] else 0 for m in catalog}
        assert max(pressure.values())<=.8+1e-9
        for t in TASKS:
            assert math.isclose(math.fsum(rr[t].values()),rates[t],rel_tol=1e-7,abs_tol=1e-12)
            for m,value in rr[t].items():
                if value>0:assert (t,m) in admitted and x[m]>0
        state={'baseline':baseline,'workload':name,'random_seed':seed,'deployment':full(x),
           'lambda_rps':rates,'routing_rps':rr,'routing_fractions':{t:{m:v/rates[t] if rates[t]>0 else 0. for m,v in rr[t].items()} for t in TASKS},
           'planning_memory_mib':policy.deployment_memory(x),'max_pressure':max(pressure.values()),
           **replica_statistics(x,roles),'selected_task_model':selected,'origin':origin}
        state['measurement_state_sha256']=digest({k:state[k] for k in ('deployment','lambda_rps','routing_rps')})
        effective={}
        for t in TASKS:
            entries=[dict(model=m,fraction=v/rates[t],metric=admitted[t,m]['quality_metric'],quality=admitted[t,m]['measured_quality'],
                          secondary_cer=admitted[t,m]['secondary_cer'],quality_source=admitted[t,m]['source'])
                     for m,v in rr[t].items() if v>0]
            effective[t]={'active':rates[t]>0,'metric':entries[0]['metric'] if entries else next(r['quality_metric'] for r in admission['rows'] if r['task']==t and r['quality_measured']),
                'routing_fraction_weighted_audit_metric':math.fsum(e['fraction']*e['quality'] for e in entries) if entries else None,
                'routing_fraction_weighted_secondary_cer':math.fsum(e['fraction']*e['secondary_cer'] for e in entries) if entries and all(e['secondary_cer'] is not None for e in entries) else None,
                'model_components':entries,'status':'AUDIT_POINT_ESTIMATE_NOT_DEPLOYMENT_QUALITY_MEASUREMENT' if entries else 'INACTIVE_NO_EFFECTIVE_QUALITY'}
        state['effective_task_quality_summary']=effective
        states.append(state)
        return state
    for name,rates in workloads.items():
        prior=read(C/name/'planning_preflight.json')
        sx=dict(prior['seed']['deployment'])
        sr={tuple(k.split('::')):v for k,v in prior['canonical']['routing'].items()}
        check=solve_seed(rates);cr=canonical(sx,rates)
        assert check.deployment==sx and cr.routing==sr
        add_state('B0',name,sx,sr,{'pre_GPU_seed_source':ref(C/name/'planning_preflight.json'),'CPU_reproduced_exactly':True})
        br={(t,m):rates[t] if b1_selected[t]==m else 0. for t in TASKS for m in catalog}
        add_state('B1',name,b1_x,br,{'rule':'FIXED_ALWAYS_ON_FULL_SIX_TASK_REPAIR'},b1_selected)
        selected={}
        for t in TASKS:
            if rates[t]<=0:continue
            options=[m for k,m in admitted if k==t and roles[m]=='specialist']
            if not options:options=[m for k,m in admitted if k==t and roles[m]=='generalist']
            selected[t]=min(options,key=lambda m:(catalog[m].service_time_ms[t],m))
        bx={m:int(m in selected.values()) for m in catalog}
        br={(t,m):rates[t] if selected.get(t)==m else 0. for t in TASKS for m in catalog}
        add_state('B2',name,bx,br,{'rule':'ADMITTED_SPECIALIST_ARGMIN_S0_ELSE_GENERALIST_ARGMIN_S0'},selected)
        b3route=canonical(static_x,rates)
        add_state('B3',name,static_x,b3route.routing,{'aggregate_seed':ref(OUT/'static_hybrid_planning_witness.json')})
        for seed in random_seeds:
            # Tripwire: the random branch must never call the LP rank function.
            with patch('deployment_search.search.rank_candidates',side_effect=AssertionError('LP ranking forbidden for Random')) as rank:
                dry=IntegerGuidedDeploymentSearch(selection_policy='random',transition_policy='runtime',random_seed=seed,predictor=None).search(
                    Workload(rates),catalog,hardware,SearchConfig(**dict(protocol['search_config'],dry_run=True)),
                    roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=cal)
                assert rank.call_count==0 and dry.runtime_evaluation_count==0
            assert dry.seed.deployment==sx and dry.iterations[0]['canonical'].routing==cr.routing
            add_state('B4_SEED',name,sx,sr,{'rule':'EXACT_PROPOSED_SEED','final_status':'UNKNOWN_UNTIL_REAL_RANDOM_SEARCH'},seed=seed)
            candidates=[c for c in dry.iterations[0]['candidates'] if c.selected_top_b]
            candidates.sort(key=lambda c:c.rank)
            import random
            feasible=[c for c in dry.iterations[0]['candidates'] if c.canonical is not None]
            expected=random.Random(seed).sample(sorted(feasible,key=lambda c:c.mutation.model),len(feasible))[:2]
            assert [c.mutation.deployment for c in candidates]==[c.mutation.deployment for c in expected]
            for c in candidates:
                state=add_state('B4_INITIAL_CANDIDATE',name,c.mutation.deployment,c.canonical.routing,
                       {'iteration':0,'random_selection_order':c.rank,'mutation':jsonable(c.mutation),'no_LP_rank_called':True},seed=seed)
                random_initial.append(state)
    freeze_file('baseline_deployments.json',{'states':states,'fixed_B1_deployment':full(b1_x),
       'fixed_B3_deployment':full(static_x),'B4_final_deployments':'UNDEFINED_BEFORE_RUNTIME_SEARCH',
       'no_runtime_results_used_for_B1_B2_B3_or_B4_initial_candidates':True})
    freeze_file('baseline_routing_rules.json',{'B1_task_to_model':b1_selected,'B1_repairs':repair_options,
       'B2_task_to_model_by_workload':{s['workload']:s['selected_task_model'] for s in states if s['baseline']=='B2'},
       'B0_B3':'Frozen explicit canonical fractions per workload; no routing recomputation for reuse',
       'B4':'Current canonical route for each random-selected feasible state; never prioritizes by LP score',
       'states':[{k:s[k] for k in ('baseline','workload','random_seed','measurement_state_sha256','lambda_rps','routing_rps','routing_fractions')} for s in states]})
    freeze_file('quality_path_audit.json',{'status':'PASS','global_admissible_path_count':21,
       'excluded_verified_paths':[{k:r[k] for k in ('task','model','admission_reason')} for r in admission['rows'] if r['capability_verified'] and not r['quality_admissible']],
       'states':[{'baseline':s['baseline'],'workload':s['workload'],'random_seed':s['random_seed'],
                  'effective_task_quality_summary':s['effective_task_quality_summary']} for s in states],
       'all_positive_routes_admitted':True,'B4_future_quality_reporting':'Compute same task-wise weighted frozen metrics for each visited/final routing; no fabricated pre-search final quality',
       'WER_CER_note':'Fraction-weighted frozen corpus point estimates are reporting diagnostics, not a fresh pooled transcript CER/WER measurement',
       'no_cross_task_quality_average':True,'runtime_objective_unchanged':True})
    freeze_file('random_initial_planning_audit.json',{'status':'PASS','runtime_evaluations':0,'GPU_runs':0,
       'LP_rank_call_count':0,'random_choice_matches_existing_RNG_branch':True,
       'seed_and_seed_canonical_exactly_Proposed':True,'initial_candidates':random_initial,
       'later_trajectory':'Not simulated or inferred; depends on real runtime feedback'})
    return definitions, states
