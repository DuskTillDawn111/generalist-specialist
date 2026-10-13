"""Freeze Module 2G workload groups, one-hop candidates, canonical routing and OOF scorer."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from deployment_core.models import load_hardware, load_model_catalog
from deployment_core.milp_seeder import MilpDeploymentSeeder
from deployment_core.routing_solver import solve_capacity_lp
from deployment_core.deployment import Deployment
from deployment_core.candidate_generator import deployment_memory_mib, missing_coverage
from deployment_core.workload import TASKS, Workload
from joint_deploy.measurement.load_generator import phases_for, periodic_arrivals
from joint_deploy.measurement.schema import MeasurementConfig
from performance_model.features import FeatureExtractor
from performance_model.generalization.features import FoldPreprocessor, planning_vector
from performance_model.profiling.plan import ROOT, save
from performance_model.resources import load_resource_profiles
from .canonical import solve_canonical_routing

OUT=ROOT/'results/performance_model_comparable_groups'
SEEDS=(7101,7202,7303)
RHO_MAX=.8
WORKLOADS={
    'W1_balanced':{'text':.8,'image':.4,'ocr':.4,'asr':.15,'visual_qa':.2,'lid':.4},
    'W2_generalist_heavy':{'text':1.5,'image':.7,'ocr':.7,'asr':.25,'visual_qa':.35,'lid':.7},
    'W3_specialist_friendly':{'text':1.0,'image':.5,'ocr':.5,'asr':.15,'visual_qa':.2,'lid':.3},
}
POLICY=dict(rho_max=RHO_MAX,one_hop='flip exactly one absent/present model bit from the MILP seed; no recursive expansion or all-combinations call',
    candidate_budget=5,repetitions=3,seeds=list(SEEDS),warmup_seconds=10,drain_timeout_seconds=60,max_pending_per_model=10000,
    min_requests_per_active_task=30,measurement_seconds='ceil(30 / minimum_positive_task_rate) + 10, bounded at >=120; no workload scale sweep',
    routing='canonical load-balancing LP: minimize z=max_m sum_k r[k,m]*s0[k,m], subject to conservation, capability, deployment, rho_max; not optimal routing',
    selection='deterministic farthest-point in planning feature space; candidate pool is exactly seed plus feasible one-hop mutations; no measured labels or performance scores',
    predictor='Fixed Module 2D-style C GBT, log(mean E2E/s0), fitted before new measurements separately excluding each evaluated deployment from historical steady rows. Frozen prediction artifacts are copied into this plan.',
    ranking='only within exact workload and compatible protocol; strict all repetitions steady; extended all repetitions steady/recoverable; common seed+phase blocks >=2',
    no_scale_sweep=True,no_performance_selection=True,no_new_data_in_predictor=True)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:16]
def deployment_id(x): return 'dep_'+identity({m:n for m,n in sorted(x.items()) if n})
def configuration_id(x,w,r): return 'cfg_'+identity(dict(deployment={m:n for m,n in sorted(x.items()) if n},workload=w,routing=r))

def feature_vector(f):
    out={}
    for m in sorted(f['per_model']):
        p=f['per_model'][m]
        out.update({f'{m}/u':p['standalone_service_demand'],**{f'{m}/q/{k}':p['task_demand_mix'][k] for k in TASKS},
            **{f'{m}/memory':p['model_memory_mib'],f'{m}/instances':p['instances']}})
        for k,v in p['colocation_pressure'].items(): out[f'{m}/C/{k}']=v['value']
    out.update({'deployment/total_instances':f['deployment_features']['total_model_instances'],
                'deployment/total_memory':f['deployment_features']['total_resident_memory_mib']})
    return out

def distance(a,b,space):
    vals=[]
    for k,s in space.items():
        if not s['varying'] or a.get(k) is None or b.get(k) is None: continue
        vals.append(((a[k]-b[k])/(s['max']-s['min']))**2)
    if not vals: raise ValueError('No varying planning dimensions')
    return math.sqrt(sum(vals)/len(vals))

def distance_space(vectors):
    keys=sorted({k for v in vectors for k in v}); out={}
    for k in keys:
        vals=[v[k] for v in vectors if v.get(k) is not None]
        out[k]=dict(min=min(vals) if vals else None,max=max(vals) if vals else None,varying=bool(vals) and max(vals)>min(vals),available=len(vals))
    return out

def one_hop(x,catalog):
    yield dict(x),dict(operation='seed',model=None,delta=0)
    for m in sorted(catalog):
        y=dict(x); y[m]=1-int(x.get(m,0)); yield y,dict(operation='add' if y[m] else 'remove',model=m,delta=1 if y[m] else -1)

def feasible(x,w,catalog,hardware):
    if not any(x.values()): return False,'empty deployment'
    d=Deployment(tuple(m for m,n in x.items() if n))
    if missing_coverage(d,catalog,Workload(w)): return False,'coverage'
    if deployment_memory_mib(d,catalog)>hardware.gpu_memory_mib: return False,'memory'
    try: witness=solve_capacity_lp(d,catalog,Workload(w),RHO_MAX)
    except Exception as e: return False,'capacity solver error: '+str(e)
    return witness is not None,'capacity LP feasible' if witness else 'capacity LP infeasible'

def fit_scorer(rows,ex,resource_keys,target_deployment):
    label_rows=[json.loads(s) for s in (ROOT/'results/performance_model_local_calibration/stability_v2_labels.jsonl').read_text().splitlines()]
    states={r['run_id']:r['stability_state_v2'] for r in label_rows}
    train=[r for r in rows if r['deployment_id']!=target_deployment and states[r['run_id']]=='steady']
    vec=[]; y=[]; ids=[]
    for r in train:
        v,s0,_=planning_vector(r,ex,'C',resource_keys); label=r['labels']['mean_e2e_latency_ms']
        if label is None or not math.isfinite(label) or label<=0: continue
        vec.append(v); y.append(math.log(label/s0)); ids.append(r['run_id']+'::'+r['task']+'::'+r['target_model'])
    if len(vec)<20: raise ValueError('Insufficient historical steady rows for frozen scorer')
    pre=FoldPreprocessor().fit(vec,list(vec[0])); model=GradientBoostingRegressor(n_estimators=50,learning_rate=.05,max_depth=2,min_samples_leaf=10,loss='huber',random_state=20260930)
    model.fit(pre.transform(vec),y)
    return pre,model,dict(target_deployment=target_deployment,training_deployment_ids=sorted({r['deployment_id'] for r in train}),training_row_ids=ids,training_rows=len(vec),target='log(mean E2E/s0)',model='fixed C GBT',params={'n_estimators':50,'learning_rate':.05,'max_depth':2,'min_samples_leaf':10,'loss':'huber','random_state':20260930},preprocessor=pre.audit(vec))

def predict_candidate(candidate,pre,model,ex,resource_keys):
    paths={}
    for k,row in candidate['routing'].items():
        for m,rate in row.items():
            if rate>0:
                v,s0,_=planning_vector(dict(deployment=candidate['deployment'],workload=candidate['workload'],routing=candidate['routing'],task=k,target_model=m),ex,'C',resource_keys)
                paths[k,m]=float(s0*math.exp(float(model.predict(pre.transform([v]))[0])))
    task={}
    for k,rate in candidate['workload'].items():
        if rate>0: task[k]=sum(candidate['routing'][k][m]/rate*paths[k,m] for m in candidate['routing'][k] if candidate['routing'][k][m]>0)
    return sum(candidate['workload'][k]*task[k] for k in task)/sum(candidate['workload'].values()),{f'{k}::{m}':v for (k,m),v in paths.items()},task

def make_plan(output=OUT):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    if (output/'comparison_plan.json').exists() and (output/'raw_runs').exists() and list((output/'raw_runs').glob('*.json')): raise ValueError('Plan is frozen after runs exist')
    catalog=load_model_catalog();hardware=load_hardware();ex=FeatureExtractor(catalog,load_resource_profiles(catalog));keys=sorted({k for p in ex.resource_profiles.values() for k in p['values']})
    old_rows=[json.loads(s) for s in (ROOT/'results/performance_model_active_sampling/dataset_rows.jsonl').read_text().splitlines()]
    label_path=ROOT/'results/performance_model_local_calibration/stability_v2_labels.jsonl'; label_hash=sha(label_path)
    groups=[]; all_candidates=[]; freeze_predictors=[]
    for name,w in WORKLOADS.items():
        seed=MilpDeploymentSeeder().solve(catalog,Workload(w),hardware,RHO_MAX)
        if not seed.feasible: raise ValueError(name+' MILP seed infeasible')
        mutations=[]; pool=[]
        for x,mutation in one_hop(seed.deployment,catalog):
            ok,reason=feasible(x,w,catalog,hardware); mutations.append(dict(deployment=x,mutation=mutation,feasible=ok,reason=reason))
            if not ok: continue
            routing,util,z=solve_canonical_routing(x,w,catalog,RHO_MAX)
            f=ex.extract(x,w,routing); vec=feature_vector(f)
            cfg=dict(deployment=x,workload_rps=w,routing_rps=routing,warmup_seconds=10,measurement_seconds=0,random_seed=0,
                     routing_source='module2g_canonical_load_balancing_routing',drain_timeout_seconds=60,shutdown_timeout_seconds=10,startup_timeout_seconds=300,max_pending_per_model=10000)
            min_rate=min(v for v in w.values() if v>0); duration=max(120,math.ceil(30/min_rate)+10);cfg['measurement_seconds']=duration
            MeasurementConfig(**cfg).validate({m:p.supported_tasks for m,p in catalog.items()})
            pool.append(dict(deployment=x,mutation=mutation,feasible=True,feasibility_reason=reason,deployment_id=deployment_id(x),
                workload=w,routing=routing,canonical_utilization=util,canonical_z=z,features=f,feature_vector=vec,configuration_id=configuration_id(x,w,routing),
                config_template=cfg,selection_score=None,selection_rationale='Seed or feasible one-hop mutation; all five in this workload are within budget.'))
        # Exact pool is seed + four additions for this catalog; still use the frozen feature space audit.
        if len(pool)>5: raise ValueError('Unexpected one-hop pool exceeds fixed budget')
        # Farthest score is computed even when no truncation is needed.
        prior=[feature_vector(ex.extract(r['deployment'],r['workload'],r['routing'])) for r in old_rows]
        space=distance_space(prior+[c['feature_vector'] for c in pool])
        selected=[]
        for c in sorted(pool,key=lambda c: c['configuration_id']):
            pop=prior+[s['feature_vector'] for s in selected]
            c['selection_score']=min((distance(c['feature_vector'],p,space) for p in pop),default=0.)
            selected.append(c)
        selected=selected[:5]
        for c in selected:
            pre,model,meta=fit_scorer(old_rows,ex,keys,c['deployment_id']); pred,paths,task=predict_candidate(c,pre,model,ex,keys)
            meta['artifact_id']=identity(dict(target=c['deployment_id'],training=meta['training_row_ids']))
            freeze_dir=output/'predictor_freeze'; freeze_dir.mkdir(exist_ok=True)
            artifact=freeze_dir/(meta['artifact_id']+'.pkl'); artifact.write_bytes(pickle.dumps({'preprocessor':pre,'model':model},protocol=4)); meta['artifact_path']=str(artifact);meta['artifact_sha256']=sha(artifact)
            c['frozen_prediction']=dict(mean_objective_ms=pred,path_mean_ms=paths,task_mean_ms=task,predictor=meta)
            freeze_predictors.append(meta);all_candidates.append(c)
        groups.append(dict(workload_group=name,workload=w,seed_deployment=seed.deployment,
            seed_routing={f'{k}::{m}':v for (k,m),v in seed.routing_rps.items()},seed_objective=seed.objective_value,
            mutations=mutations,all_feasible_candidate_count=len(pool),selected_candidates=[{k:v for k,v in c.items() if k not in ('features',)} for c in selected],
            distance_space=space))
    plan=dict(schema_version=1,stage='Module 2G Comparable-Group Profiling',workload_groups=WORKLOADS,groups=groups,policy=POLICY,
              predictor_freeze=freeze_predictors,label_source={'path':str(label_path),'sha256':label_hash},
              catalog_sources={m:p.sources for m,p in catalog.items()},hardware=hardware.source,seeds=list(SEEDS),
              max_runs=sum(len(g['selected_candidates'])*len(SEEDS) for g in groups),
              freeze_statement='This JSON and its hash are written before any Module 2G raw run. Predictor fits use only prior Module 2D/2E source data and exclude target deployment. No Module 2G labels are read.',
              no_enumeration=True)
    # Configs and run manifests are created before measurements, with a single plan hash after complete content.
    plan_path=output/'comparison_plan.json';save(plan_path,plan);plan_hash=sha(plan_path)
    save(output/'workload_groups.json',dict(schema_version=1,groups=WORKLOADS,definitions={
        'W1_balanced':'Six tasks, balanced lower load selected below prior Qwen overload region.',
        'W2_generalist_heavy':'Higher six-task load with Qwen-only seed load below rho_max; no scale sweep.',
        'W3_specialist_friendly':'Six tasks with lower audio/lid rates and specialist-compatible canonical routing; fixed workload, not an intensity sweep.'},
        planning_only=True,plan_sha256=plan_hash))
    for g in groups:
        for c in g['selected_candidates']:
            cdir=output/'raw_runs'/'configs';cdir.mkdir(parents=True,exist_ok=True)
            c['runs']=[]
            for rep,seedv in enumerate(SEEDS,1):
                run=f"{g['workload_group']}_{c['configuration_id']}_rep_{rep}"
                cfg=dict(c['config_template'],random_seed=seedv)
                cp=cdir/(run+'.json');save(cp,cfg)
                c['runs'].append(dict(run_id=run,repetition_id=rep,seed=seedv,config_path=str(cp),config_sha256=sha(cp),
                    result_path=str(output/'raw_runs'/(run+'.json')),log_path=str(output/'raw_runs'/(run+'.log')),initial_arrival_phases=phases_for(c['workload'],seedv),
                    expected_task_counts=dict(Counter(k for _,k in periodic_arrivals(c['workload'],cfg['measurement_seconds'],seedv)))))
    # Update plan with configs would alter the freeze hash; write a separate manifest and retain frozen hash.
    manifest=dict(plan_sha256=plan_hash,groups=groups,created_before_measurements=True,run_count=sum(len(c['runs']) for g in groups for c in g['selected_candidates']))
    save(output/'run_manifest.json',manifest)
    save(output/'comparison_plan_hash.json',dict(path=str(plan_path),sha256=plan_hash,contents_frozen=True,run_manifest_sha256=sha(output/'run_manifest.json')))
    return plan,manifest

if __name__=='__main__':
    p,m=make_plan();print(json.dumps(dict(plan_sha256=sha(OUT/'comparison_plan.json'),groups=[(g['workload_group'],len(g['selected_candidates'])) for g in m['groups']],runs=m['run_count']),indent=2))
