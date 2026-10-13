"""CPU-only calibration reports; no serving or GPU imports."""
import itertools
import json
import math
from pathlib import Path
import unittest
from unittest.mock import Mock
from candidate_pool_service_demand.dry_run import load_completed_catalog
from deployment_core import Workload, Hardware, MilpDeploymentSeeder
from deployment_core.candidate_pool_v2 import mandatory_presence
from deployment_core.milp_seeder import validate_seed
from deployment_core.replica_calibration import ReplicaPlanningPolicy, load_replica_calibration, reference
from deployment_core.replicas import replica_statistics
from deployment_search import IntegerGuidedDeploymentSearch, SearchConfig, solve_canonical_routing
from deployment_search.feedback import bottleneck_feedback
from deployment_search.integer_foundation import binary_compatibility, model
from deployment_search.result import jsonable, CandidateEvaluation
from deployment_search.neighborhood import Mutation, missing_coverage
from deployment_search.scoring import rank_candidates
from .tests import competition, fixture

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/replica_capacity_calibration'


def write(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(jsonable(data),indent=2,allow_nan=False)+'\n')


def candidate_record(candidate,current,catalog,roles,policy):
    mutation,canonical=candidate.mutation,candidate.canonical
    m,old,new=mutation.model,current[mutation.model],mutation.deployment[mutation.model]
    increment=policy.memory(m,new)-policy.memory(m,old)
    return {'operation':mutation.operation,'model_id':m,'role':roles[m],
        'old_replica_count':old,'new_replica_count':new,'deployment':mutation.deployment,
        'effective_capacity_before':policy.capacity(m,old),'effective_capacity_after':policy.capacity(m,new),
        'capacity_increment':policy.capacity(m,new)-policy.capacity(m,old),
        'empirical_capacity_increment':policy.calibration.capacity_factors[m][new]-policy.calibration.capacity_factors[m][old],
        'memory_increment_mib':increment,'memory_saving_mib':max(0.,-increment),
        'total_memory_mib':policy.deployment_memory(mutation.deployment),
        'feasibility':canonical is not None,'status':candidate.status,'reason':candidate.reason or None,
        'delta_z':candidate.delta_z,'candidate_max_pressure':canonical.max_pressure if canonical else None,
        'effective_normalized_pressure':canonical.model_pressure if canonical else None,
        'ranking':candidate.rank,'Top_B':candidate.selected_top_b,**replica_statistics(mutation.deployment,roles)}


def planning_run(row,catalog,hardware,roles,records,calibration,mode,initial=None,memory_mode='empirical'):
    policy=ReplicaPlanningPolicy(catalog,mode,memory_mode,calibration)
    evaluator=Mock();evaluator.evaluate.side_effect=AssertionError('Runtime/GPU forbidden in 7C-2')
    result=IntegerGuidedDeploymentSearch(evaluator).search(Workload(row['arrival_rates_rps']),catalog,hardware,
        SearchConfig(dry_run=True),roles=roles,capacity_mode=mode,memory_mode=memory_mode,
        calibration=calibration,initial_deployment=initial,
        replica_bounds=calibration.evidence_bounds if memory_mode=='additive' else None)
    evaluator.evaluate.assert_not_called()
    if not result.seed.feasible or result.canonical is None or result.runtime_evaluation_count:
        raise RuntimeError('Expected feasible CPU-only planning state')
    current,canonical=result.deployment,result.canonical
    candidates=[candidate_record(c,current,catalog,roles,policy) for c in result.iterations[0]['candidates']]
    if len(candidates)>2*len(catalog):raise AssertionError('Neighborhood exceeds 2M')
    demand={m:math.fsum(v for (k,name),v in canonical.task_contributions.items() if name==m) for m in catalog}
    feedback=bottleneck_feedback(canonical,current)
    ordered=sorted((c for c in candidates if c['ranking'] is not None),key=lambda c:c['ranking'])
    choices=[]
    for task in feedback.pressure_tasks:
        up=[c for c in candidates if c['operation']=='SCALE_UP' and task in catalog[c['model_id']].supported_tasks]
        general=[c for c in up if c['role']=='generalist'];special=[c for c in up if c['role']=='specialist']
        choices.append({'pressure_task':task,'direct_generalist_specialist_candidates':bool(general and special),
            'generalist_plus_one':general,'specialist_first_replica':[c for c in special if c['old_replica_count']==0],
            'specialist_additional_replica':[c for c in special if c['old_replica_count']>0]})
    return {'workload':row['name'],'workload_rps':row['arrival_rates_rps'],'status':result.status,
        'capacity_mode':mode,'memory_mode':memory_mode,'blocked':False,'planning_only':True,
        'start_kind':'additional_fixed_state_one_hop_diagnostic' if initial is not None else 'minimum_memory_MILP_seed',
        'integer_seed_replica_vector':result.seed.deployment,'deployment':current,
        'memory_usage_mib':policy.deployment_memory(current),'gpu_capacity_mib':hardware.gpu_memory_mib,
        **replica_statistics(current,roles),'canonical_routing':canonical.routing,
        'family_total_demand':demand,'task_model_demand':canonical.task_contributions,
        'effective_capacity':{m:policy.capacity(m,current[m]) for m in catalog},
        'effective_normalized_pressure':canonical.model_pressure,
        'average_per_replica_pressure_diagnostic':{m:demand[m]/current[m] if current[m] else None for m in catalog},
        'pressure_definition':'D_m/g_m(x_m)' if mode=='empirical' else 'D_m/x_m',
        'inactive_pressure_values':'zero placeholders; pressure only defined for active families',
        'max_pressure':canonical.max_pressure,'bottleneck_families':feedback.bottleneck_models,
        'pressure_tasks':feedback.pressure_tasks,'all_replica_mutation_candidates':candidates,
        'feasible_candidates':[c for c in candidates if c['feasibility']],
        'ordered_candidates':ordered,'Top_B':[c for c in ordered if c['Top_B']],
        'generalist_specialist_competition':choices,
        'mandatory_presence':mandatory_presence(records,Workload(row['arrival_rates_rps'])),
        'generalist_composition_note':'Qwen presence is mandatory due to unique verified VQA coverage, not a performance choice.',
        'candidate_count':len(candidates),'one_hop_bound':2*len(catalog),
        'evidence_bounds':calibration.evidence_bounds,'effective_bounds':result.replica_bounds,
        'bound_provenance':result.bound_provenance,'MILP_solves':result.milp_solve_count,
        'LP_solves':result.lp_solve_count,'runtime_evaluations':0,'GPU_measurements':0,'workers_started':0,
        'exhaustive_enumeration_used':False,'complete_trace':result.to_dict()}


def complete_neighbor_diagnostic(state,catalog,hardware,roles,calibration,mode):
    """All bounded one-hop mutations at a fixed state, even if guidance prunes them.

    Diagnostic only: no transitions, runtime evaluations or vector enumeration.
    The production guided neighborhood remains unchanged.
    """
    policy=ReplicaPlanningPolicy(catalog,mode,'empirical',calibration)
    bounds,_=policy.bounds(hardware)
    current=state['deployment'];w=Workload(state['workload_rps'])
    guided={(a['model_id'],a['new_replica_count']) for a in state['all_replica_mutation_candidates']}
    candidates=[]
    for m in sorted(catalog):
        for direction in (1,-1):
            new=current[m]+direction
            if not 0<=new<=bounds[m]:continue
            x=dict(current,**{m:new})
            candidate=CandidateEvaluation(Mutation('SCALE_UP' if direction>0 else 'SCALE_DOWN',m,x))
            candidate.memory_mib=policy.deployment_memory(x)
            if candidate.memory_mib>hardware.gpu_memory_mib:
                candidate.status,candidate.reason='REJECTED','memory'
            elif missing_coverage(x,catalog,w):
                candidate.status,candidate.reason='REJECTED','coverage'
            else:
                candidate.canonical=solve_canonical_routing(x,w,catalog,replica_bounds=bounds,capacity_mode=mode,calibration=calibration)
                if candidate.canonical is None:
                    candidate.status,candidate.reason='REJECTED','capacity'
                else:
                    candidate.delta_z=state['max_pressure']-candidate.canonical.max_pressure
                    candidate.status='DIAGNOSTIC_FEASIBLE'
            candidates.append(candidate)
    for i,candidate in enumerate(rank_candidates([a for a in candidates if a.canonical is not None]),1):candidate.rank=i
    output=[]
    for candidate in candidates:
        record=candidate_record(candidate,current,catalog,roles,policy)
        record['included_by_production_pressure_task_guidance']=(record['model_id'],record['new_replica_count']) in guided
        output.append(record)
    if len(output)>2*len(catalog):raise AssertionError('All-direct one-hop diagnostic exceeds 2M')
    return {'scope':'All bounded direct mutations at one fixed state; supplemental diagnostic, not production search or a state transition',
        'candidates':output,'candidate_count':len(output),'one_hop_bound':2*len(catalog),'runtime_evaluations':0,
        'comparison_note':'A zero delta_z indicates that extra capacity does not relieve the current bottleneck; it must not be called useful merely because g increases.'}


def small_discrete_oracle(calibration):
    # Validation-only 27-vector oracle; production search never imports it.
    c={'qwen':model('qwen',3.,asr=1000.,image=1000.),
       'whisper-small':model('whisper-small',1.5,asr=500.),'resnet50':model('resnet50',1.,image=1000.)}
    cal=fixture(c,calibration,memory2={'qwen':5.5,'whisper-small':3.1,'resnet50':2.1})
    h=Hardware(9.);p=ReplicaPlanningPolicy(c,'empirical',calibration=cal);cases=[]
    for loads in ({'asr':.9,'image':1.2},{'asr':2.1,'image':.5},{'asr':0.,'image':0.},{'asr':20.,'image':20.}):
        w=Workload(loads);values=[]
        for counts in itertools.product(range(3),repeat=3):
            x=dict(zip(sorted(c),counts));memory=p.deployment_memory(x)
            if memory>h.gpu_memory_mib:continue
            r=solve_canonical_routing(x,w,c,capacity_mode='empirical',calibration=cal)
            if r is not None:values.append((memory,sum(counts),x))
        seed=MilpDeploymentSeeder().solve(c,w,h,capacity_mode='empirical',calibration=cal)
        optimum=min(((v[0],v[1]) for v in values),default=None)
        observed=(seed.objective_value,seed.secondary_objective_value) if seed.feasible else None
        errors=validate_seed(seed,c,w,h,.8,{m:2 for m in c},planning_policy=p) if seed.feasible else []
        passed=seed.feasible==bool(values) and observed==optimum and not errors
        if not passed:raise AssertionError('Discrete MILP differs from small independent oracle')
        cases.append({'loads_rps':loads,'oracle_optimum_memory_then_count':optimum,'seed':seed,
            'independent_witness_errors':errors,'passed':passed,'vectors_examined':27})
    return {'passed':True,'model_types':3,'replica_bound':2,'scope':'validation-only oracle; never called by production search',
        'cases':cases,'GPU_measurements':0}


def run_regression():
    loader=unittest.TestLoader();suite=unittest.TestSuite()
    for directory in ('deployment_core/tests','deployment_search/tests'):
        for path in sorted((ROOT/directory).glob('test_*.py')):
            suite.addTests(loader.loadTestsFromName('.'.join(path.relative_to(ROOT).with_suffix('').parts)))
    for name in ('candidate_pool_service_demand.test_service_demand','joint_deploy.measurement.tests.test_measurement',
                 'replica_scaling.test_replica_scaling','replica_capacity_calibration.tests'):
        suite.addTests(loader.loadTestsFromName(name))
    with (OUT/'cpu_regression.log').open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    value={'status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'log':reference(OUT/'cpu_regression.log'),'GPU_measurements':0,
        'synthetic_campaign_wrappers':'Included CPU-only 7B A-D and 12/20/30 model tests; no real campaigns invoked.'}
    write('cpu_regression_validation.json',value)
    if not result.wasSuccessful():raise AssertionError('CPU regressions failed; inspect log')
    return value


def order(state):return [(c['model_id'],c['new_replica_count']) for c in state['ordered_candidates']]
def top(state):return [(c['model_id'],c['new_replica_count']) for c in state['Top_B']]


def run():
    c,h,roles,records,_=load_completed_catalog();cal=load_replica_calibration(c)
    policy=ReplicaPlanningPolicy(c,'empirical',calibration=cal);bounds,provenance=policy.bounds(h)
    sources=dict(cal.source_hashes)
    for name in ('candidate_pool_v2/candidate_models.json','candidate_pool_v2/measured_vram.json',
        'candidate_pool_v2/compatibility_matrix.json','candidate_pool_v2_service_demand/updated_service_demand_catalog.json',
        'deployment_search_multiworkload/benchmark_workloads.json'):
        sources[name]=reference(ROOT/'results'/name)
    write('replica_capacity_factors.json',{'capacity_mode':'empirical','frozen':True,'models':{
        m:{'g':cal.capacity_factors[m],'marginal_capacity':{n:cal.capacity_factors[m][n+1]-cal.capacity_factors[m][n] for n in range(cal.evidence_bounds[m])},
           'evidence':cal.model_evidence[m]} for m in c},'source_hashes':sources,
        'aggregation':'Frozen high-load paired throughput-gain mean, three valid repetitions, full precision',
        'transfer_scope':'Qwen/Whisper ASR-only measured factors applied to family demand as a planning surrogate; mixed-task transfer unvalidated.'})
    write('replica_memory_table.json',{'unit':'MiB','frozen':True,'models':{m:{'M':cal.memory_table_mib[m],
        'single_anchor':'frozen v2 measured idle VRAM; exact binary compatibility',
        'observed_7c1_memory':{k:v for k,v in cal.model_evidence[m].items() if k.startswith('observed_7c1')},
        'double_vs_additive_difference_mib':cal.memory_table_mib[m][2]-2*c[m].memory_mib if 2 in cal.memory_table_mib[m] else None}
        for m in c},'source_hashes':sources,'scope':'Idle-memory planning surrogate; heterogeneous co-residency / peak not guaranteed.'})
    write('evidence_replica_bounds.json',{'gpu_capacity_mib':h.gpu_memory_mib,'replica_bounds':bounds,
        'models':provenance,'source_hashes':sources,'meaning':'Evidence limits do not imply physical inability to host more instances.'})
    rows=json.loads((ROOT/'results/deployment_search_multiworkload/benchmark_workloads.json').read_text())['workloads']
    dry_runs=[]
    for i,row in enumerate(rows,1):
        empirical=planning_run(row,c,h,roles,records,cal,'empirical')
        linear=planning_run(row,c,h,roles,records,cal,'linear')
        additive=planning_run(row,c,h,roles,records,cal,'linear',memory_mode='additive')
        warm=dict.fromkeys(c,0);warm.update(qwen=1,resnet50=1);warm['whisper-small']=1
        warm_e=planning_run(row,c,h,roles,records,cal,'empirical',initial=warm)
        warm_l=planning_run(row,c,h,roles,records,cal,'linear',initial=warm)
        value={'module':'7C-2','source_hashes':sources,'empirical':empirical,'linear':linear,
            'original_linear_additive_memory_baseline_with_same_evidence_bounds':additive,
            'comparison':{'controlled_settings':'Identical evidence bounds, discrete memory, demands, workload, Top-B and seed policy; only capacity mode differs.',
                'same_seed':linear['deployment']==empirical['deployment'],'candidate_order_changed':order(linear)!=order(empirical),
                'linear_order':order(linear),'empirical_order':order(empirical),
                'linear_Top_B':top(linear),'empirical_Top_B':top(empirical),'Top_B_changed':top(linear)!=top(empirical)},
            'additional_fixed_state_diagnostic':{'reason':'Compare G second replica, new S first replica, and tested S second replica at one fixed feasible state; no transitions.',
                'initial_deployment':warm,'empirical':warm_e,'linear':warm_l,
                'candidate_order_changed':order(warm_e)!=order(warm_l),
                'all_direct_empirical':complete_neighbor_diagnostic(warm_e,c,h,roles,cal,'empirical'),
                'all_direct_linear':complete_neighbor_diagnostic(warm_l,c,h,roles,cal,'linear')}}
        write(f'w{i}_calibrated_dry_run.json',value);dry_runs.append(value)
        print(row['name'],json.dumps(value['comparison']),flush=True)
    binary=binary_compatibility();milp=small_discrete_oracle(cal)
    real_validations=[]
    for v in dry_runs:
        w=Workload(v['empirical']['workload_rps'])
        seed=MilpDeploymentSeeder().solve(c,w,h,capacity_mode='empirical',calibration=cal)
        errors=validate_seed(seed,c,w,h,.8,bounds,planning_policy=policy)
        if errors:raise AssertionError(str(errors))
        real_validations.append({'workload':v['empirical']['workload'],'seed':seed,'errors':errors})
    write('empirical_milp_validation.json',dict(milp,real_pool_seed_validation=real_validations))
    write('linear_vs_empirical_validation.json',{'binary_compatibility':binary,'synthetic_competition':competition(cal),
        'marginal_gain_full_precision':{m:cal.capacity_factors[m][2]-1. for m in ('qwen','whisper-small','resnet50')},
        'real_pool_comparisons':[{'workload':v['empirical']['workload'],**v['comparison']} for v in dry_runs],
        'GPU_measurements':0,'source_hashes':sources})
    regression=run_regression()
    finalize()


def finalize():
    """Validate and freeze already-produced outputs, without repeating tests/runs."""
    from .report import write_summary
    import platform
    import sys
    import numpy
    import scipy
    c,h,roles,records,_=load_completed_catalog();cal=load_replica_calibration(c)
    before=json.loads((OUT/'before_hashes.json').read_text())
    changes=[name for name,ref in before['frozen_inputs'].items() if reference(ref['path'])['sha256']!=ref['sha256']]
    if changes:raise AssertionError('Frozen inputs changed: '+str(changes))
    dry_runs=[json.loads((OUT/f'w{i}_calibrated_dry_run.json').read_text()) for i in range(1,4)]
    sources=dry_runs[0]['source_hashes']
    sources=dict(sources,measurement_environment_manifest=reference(ROOT/'results/replica_scaling_real/environment_manifest.json'))
    source_files=sorted(set(ROOT.glob('deployment_core/*.py'))|set(ROOT.glob('deployment_search/*.py'))|set(ROOT.glob('replica_capacity_calibration/*.py'))|
        {ROOT/'deployment_core/README.md',ROOT/'deployment_search/INTEGER_REPLICAS.md',ROOT/'replica_capacity_calibration/README.md'})
    write('source_hashes.json',{'frozen_inputs_checked':len(before['frozen_inputs']),'frozen_input_changes':changes,
        'evidence_and_planning_inputs':sources,'before_manifest':reference(OUT/'before_hashes.json'),
        'current_source_hashes':{str(p.relative_to(ROOT)):reference(p) for p in source_files},
        'authorized_algorithm_changes':[name for name,ref in before['algorithm_before'].items() if reference(ref['path'])['sha256']!=ref['sha256']],
        'environments_modified':False,'GPU_measurements':0,'workers_started':0,
        'CPU_runtime':{'python':platform.python_version(),'executable':sys.executable,'numpy':numpy.__version__,'scipy':scipy.__version__},
        'disk_space_recovery':'Moved one pip HTTP cache response (1.23 GB) and its metadata to /tmp/gs_7c2_pip_cache_backup; environments/models/results untouched.'})
    regression=json.loads((OUT/'cpu_regression_validation.json').read_text())
    if regression['status']!='passed' or regression['failures'] or regression['errors']:raise AssertionError('Tests not passed')
    if reference(regression['log']['path'])['sha256']!=regression['log']['sha256']:raise AssertionError('Test log hash drift')
    validation=json.loads((OUT/'linear_vs_empirical_validation.json').read_text())
    validation['cases']={
        'A':{'status':'passed','validation':'Exact R=1 seed/routing/search serialization and five historical binary traces'},
        'B':{'status':'passed','qwen_second_marginal_capacity':cal.capacity_factors['qwen'][2]-1.,'linear_second_marginal_capacity':1.},
        'C':{'status':'passed','resnet_g2':cal.capacity_factors['resnet50'][2],'relative_difference_from_linear':abs(cal.capacity_factors['resnet50'][2]/2.-1.)},
        'D':{'status':'passed','linear_winner':validation['synthetic_competition']['linear']['winner'],
            'empirical_winner':validation['synthetic_competition']['empirical']['winner']}}
    validation['CPU_test_evidence']=regression
    write('linear_vs_empirical_validation.json',validation)
    milp=json.loads((OUT/'empirical_milp_validation.json').read_text())
    if not milp['passed'] or any(v['errors'] for v in milp['real_pool_seed_validation']):raise AssertionError('MILP validation failed')
    bounds,_=ReplicaPlanningPolicy(c,'empirical',calibration=cal).bounds(h)
    checked=0
    for v in dry_runs:
        states=[v['empirical'],v['linear'],v['original_linear_additive_memory_baseline_with_same_evidence_bounds'],
            v['additional_fixed_state_diagnostic']['empirical'],v['additional_fixed_state_diagnostic']['linear']]
        for state in states:
            p=ReplicaPlanningPolicy(c,state['capacity_mode'],state['memory_mode'],cal)
            if state['GPU_measurements'] or state['workers_started'] or state['runtime_evaluations']:raise AssertionError('Unexpected runtime activity')
            if state['candidate_count']>2*len(c):raise AssertionError('Too many one-hop candidates')
            if any(n>bounds[m] for m,n in state['deployment'].items()):raise AssertionError('Count outside evidence bound')
            if p.deployment_memory(state['deployment'])!=state['memory_usage_mib']:raise AssertionError('Memory mismatch')
            if state['memory_usage_mib']>h.gpu_memory_mib:raise AssertionError('Memory infeasible')
            for a in state['all_replica_mutation_candidates']:
                if any(n>bounds[m] for m,n in a['deployment'].items()):raise AssertionError('Candidate exceeds bounds')
                if not math.isclose(p.deployment_memory(a['deployment']),a['total_memory_mib'],abs_tol=1e-9):raise AssertionError('Candidate memory mismatch')
                if a['feasibility'] and a['total_memory_mib']>h.gpu_memory_mib:raise AssertionError('Accepted overbudget')
            checked+=1
    write_summary(cal,dry_runs,regression,OUT)
    required=('replica_capacity_factors.json','replica_memory_table.json','evidence_replica_bounds.json',
        'empirical_milp_validation.json','linear_vs_empirical_validation.json','w1_calibrated_dry_run.json',
        'w2_calibrated_dry_run.json','w3_calibrated_dry_run.json','calibration_summary.md')
    if not all((OUT/name).exists() for name in required):raise AssertionError('Missing required artifact')
    write('calibration_validation.json',{'status':'passed','required_artifacts':list(required),'planning_states_checked':checked,
        'source_hashes_checked':len(before['frozen_inputs']),'frozen_inputs_changed':0,'pending_model_scaling_factors_filled':0,
        'real_pool_ordering_changed':[v['empirical']['workload'] for v in dry_runs if v['comparison']['candidate_order_changed']],
        'GPU_measurements':0,'real_searches':0,'workers_started':0,'CPU_regression':regression})
    artifacts=[p for p in OUT.iterdir() if p.is_file() and p.name not in ('artifact_manifest.json','finalize.log')]
    write('artifact_manifest.json',{'module':'7C-2','frozen':True,'status':'complete',
        'artifacts':{p.name:reference(p) for p in sorted(artifacts)},'GPU_measurements':0,'real_searches':0,'frozen_input_changes':0})
    print(json.dumps({'status':'complete','regression':regression['tests_run'],'outputs':str(OUT)}),flush=True)


if __name__=='__main__':run()
