"""One MILP seed, one-hop mutations, then six stratified farthest-point conditions."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from deployment_core.models import load_hardware
from deployment_core.workload import Workload,TASKS
from deployment_core.deployment import Deployment
from deployment_core.milp_seeder import MilpDeploymentSeeder
from deployment_core.seed import seed_as_dict
from deployment_core.candidate_generator import missing_coverage,deployment_memory_mib
from deployment_core.routing_solver import solve_capacity_lp
from joint_deploy.measurement.schema import MeasurementConfig
from joint_deploy.measurement.load_generator import periodic_arrivals,phases_for
from performance_model.resources import source
from performance_model.stability import DEFAULT_THRESHOLDS
from .common import *

BASE=dict(zip(TASKS,(2.,1.,1.,.3,.5,1.)))


def one_hop(seed):
    for model in sorted(seed):
        x=dict(seed); x[model]=1-x[model]
        yield x,{'operation':'add' if x[model] else 'remove','model':model,'delta':1 if x[model] else -1}


def feasible(x,workload,catalog,hardware):
    if not any(x.values()): return False,'empty deployment'
    d=Deployment(tuple(m for m,n in x.items() if n))
    if missing_coverage(d,catalog,Workload(workload)): return False,'coverage'
    if deployment_memory_mib(d,catalog)>hardware.gpu_memory_mib: return False,'memory'
    witness=solve_capacity_lp(d,catalog,Workload(workload),.8)
    return witness is not None,'capacity LP feasible' if witness else 'capacity LP infeasible'


def choose(pool,existing,space,budget=6):
    if budget!=6: raise ValueError('This acquisition round has a fixed six-condition budget')
    selected=[]; counts=Counter(); used=set()
    # Category quotas and context caps are design constraints, never performance scores.
    for category in ('clearly_stable','near_boundary','likely_unstable')*2:
        eligible=[c for c in pool if c['expected_category']==category and c['candidate_id'] not in used and counts[c['deployment_id']]<2]
        if not eligible: raise ValueError('Insufficient local candidates for declared quotas')
        population=existing+[c['feature_vector'] for c in selected]
        winner=max(eligible,key=lambda c:(nearest(c['feature_vector'],population,space),c['candidate_id']))
        winner=dict(winner,selection_score=nearest(winner['feature_vector'],population,space),selection_rank=len(selected)+1,
            selection_rationale='Maximum block-balanced normalized distance to existing runs plus already selected conditions, within two-per-risk-category and at-most-two-per-context constraints. No performance objective.')
        selected.append(winner);used.add(winner['candidate_id']);counts[winner['deployment_id']]+=1
    return selected


def make_plan(output=OUT):
    output=Path(output).resolve()
    if (output/'active_sampling_runs/raw').exists() and list((output/'active_sampling_runs/raw').glob('*.json')):
        raise ValueError('Do not regenerate an acquisition plan after measurements exist')
    ex=extractor(); hardware=load_hardware(); catalog=ex.catalog
    workload=Workload(BASE); seed=MilpDeploymentSeeder().solve(catalog,workload,hardware)
    if not seed.feasible: raise ValueError('Base workload has no seed')
    prior=[]; references=[]
    for path,stage,role in existing_sources():
        d=json.loads(path.read_text());f=ex.extract(d['deployment'],d['workload_rps'],d['routing_rps'])
        if 'qwen' in f['per_model']: prior.append(feature_vector(f))
        references.append(dict(**source(path),stage=stage,role=role,deployment_id=deployment_id(d['deployment'])))
    base_r={k:{m:BASE[k] if m=='qwen' else 0. for m in catalog} for k in TASKS}
    mix=ex.extract({'qwen':1},BASE,base_r)['per_model']['qwen']['task_demand_mix']
    vision={k:(mix[k] if k not in ('asr','lid') else 0.) for k in TASKS}; total=sum(vision.values());vision={k:v/total for k,v in vision.items()}
    mixes={'reference_six_task':mix,'text_vision':vision}
    policy=dict(min_requests_per_path=30,min_measurement_seconds=120,max_measurement_seconds=300,
        likely_unstable_max_measurement_seconds=180,warmup_seconds=10,max_drain_seconds=30,
        safe_run_timeout_seconds=900,interrupt_grace_seconds=45,severe_pending_per_model=256,
        rho_max=.8,stability_thresholds=dict(DEFAULT_THRESHOLDS),
        safe_stop_semantics='Original harness bounds host queues, drain and worker startup. Backlog labels are computed after each bounded window, not by a live queue observer. No higher load is added after observing instability; repetitions remain bounded and retained.')
    mutations=[];pool=[]
    for x,mutation in one_hop(seed.deployment):
        ok,reason=feasible(x,BASE,catalog,hardware)
        mutations.append(dict(deployment=x,mutation=mutation,feasible=ok,reason=reason))
        if not ok or not x.get('qwen') or mutation['operation']!='add': continue
        contender=mutation['model']
        for task in sorted(catalog[contender].supported_tasks):
            for mix_name,q in mixes.items():
                for category,u in (('clearly_stable',.10),('near_boundary',.25),('likely_unstable',.35)):
                    for co_u in (.10,.20):
                        r={k:{m:0. for m in catalog} for k in TASKS}
                        for k in TASKS: r[k]['qwen']=u*q[k]/(catalog['qwen'].service_time_ms[k]/1000)
                        r[task][contender]=co_u/(catalog[contender].service_time_ms[task]/1000)
                        w={k:math.fsum(row.values()) for k,row in r.items()}
                        valid,why=feasible(x,w,catalog,hardware)
                        if not valid: raise ValueError('Generated condition failed feasibility oracle: '+why)
                        f=ex.extract(x,w,r)
                        if any(v['standalone_service_demand']>.8+1e-9 for v in f['per_model'].values()): raise ValueError('Explicit routing exceeds capacity surrogate')
                        nominal=math.ceil(30/min(v for row in r.values() for v in row.values() if v>0))
                        cap=policy['likely_unstable_max_measurement_seconds'] if category=='likely_unstable' else policy['max_measurement_seconds']
                        duration=max(policy['min_measurement_seconds'],min(cap,nominal))
                        cfg=MeasurementConfig(x,w,r,warmup_seconds=10,measurement_seconds=duration,
                            drain_timeout_seconds=policy['max_drain_seconds'],max_pending_per_model=policy['severe_pending_per_model'],
                            routing_source='bounded_feature_directed_active_sampling')
                        cfg.validate({m:p.supported_tasks for m,p in catalog.items()})
                        c=dict(candidate_id='candidate_'+identity(cfg.as_dict()),configuration_id=configuration_id(cfg.as_dict()),
                            deployment_id=deployment_id(x),source_seed=seed.deployment,mutation=mutation,
                            deployment=x,workload=w,routing=r,config_template=cfg.as_dict(),features=f,feature_vector=feature_vector(f),
                            expected_category=category,expected_category_rationale='Sampling prior from Qwen u=0.10 stable runs and backlog at u=0.30; transferred to new mix/context, not a predictor or guaranteed observed label.',
                            mix_name=mix_name,co_runner={'task':task,'model':contender,'standalone_demand':co_u},nominal_measurement_seconds=nominal,
                            measurement_seconds=duration,feasibility={'oracle':'coverage + catalog memory + solve_capacity_lp; explicit routing separately checked','passed':True})
                        pool.append(c)
    space=fit_distance_space(prior+[c['feature_vector'] for c in pool])
    for c in pool: c['nearest_existing_feature_distance']=nearest(c['feature_vector'],prior,space)
    selected=choose(pool,prior,space)
    for i,c in enumerate(selected,1):
        c['name']=f'AS_{i:02d}'; c['runs']=[]
        for rep,seed_value in enumerate(SEEDS,1):
            run_id=f"{c['name']}_rep_{rep}";cfg=dict(c['config_template'],random_seed=seed_value)
            path=output/'active_sampling_runs/configs'/f'{run_id}.json';save(path,cfg)
            c['runs'].append(dict(run_id=run_id,repetition_id=rep,seed=seed_value,config=str(path),config_sha256=source(path)['sha256'],
                result=str(output/'active_sampling_runs/raw'/f'{run_id}.json'),log=str(output/'active_sampling_runs/logs'/f'{run_id}.log'),
                initial_arrival_phases=phases_for(c['workload'],seed_value),
                expected_task_counts=dict(Counter(k for _,k in periodic_arrivals(c['workload'],c['measurement_seconds'],seed_value)))))
    protected={str(p.relative_to(ROOT)):source(p)['sha256'] for folder in ('deployment_core','performance_model','joint_deploy/measurement','joint_deploy') for p in (ROOT/folder).glob('*.py')}
    report=dict(seed=seed_as_dict(seed,catalog,workload,hardware,.8),mutations=mutations,local_mutation_count=len(mutations),
        feasible_local_deployment_count=sum(m['feasible'] for m in mutations),candidate_condition_count=len(pool),
        considered_but_not_selected_conditions=len(pool)-len(selected),candidate_conditions=pool,
        distance_space=space,distance_definition='Equal weight per varying feature block; min/max fitted on existing + candidate features, frozen before measurements; no IDs, latency, labels or uncertainty in score. Missing values omitted, never guessed.',
        prior_sources=references,scope='One seed and its immediate binary add/remove neighbors only. Finite prescribed load/mix/co-load conditions; no recursive graph expansion.')
    save(output/'candidate_pool.json',report)
    plan=dict(conditions=selected,condition_budget=6,repetitions=3,maximum_runs=18,seeds=list(SEEDS),policy=policy,
        environment={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},protected_sources=protected,
        acquisition_score_uses_deployment_id=False,stop_after_budget=True)
    save(output/'selected_conditions.json',plan)
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=OUT);args=parser.parse_args()
    p=make_plan(args.output_dir)
    print(json.dumps([dict(name=c['name'],deployment=c['deployment'],category=c['expected_category'],mix=c['mix_name'],seconds=c['measurement_seconds'],score=c['selection_score']) for c in p['conditions']],indent=2))
