"""Small controlled Qwen feature-axis design using existing offline profiles."""
import json
import math
from pathlib import Path
from deployment_core.models import load_model_catalog
from deployment_core.workload import TASKS
from performance_model.features import FeatureExtractor
from performance_model.resources import load_resource_profiles,source,ROOT
from joint_deploy.measurement.schema import MeasurementConfig
from joint_deploy.measurement.load_generator import periodic_arrivals,phases_for
from collections import Counter

OUT=ROOT/'results/performance_model_profiling/qwen'
SEEDS=(1101,2202,3303)
BASE=dict(zip(TASKS,(2.,1.,1.,.3,.5,1.)))


def save(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n'); tmp.replace(path)


def extractor():
    catalog=load_model_catalog()
    return FeatureExtractor(catalog,load_resource_profiles(catalog))


def choose_contenders(ex):
    # Select diverse service paths, not combinations of deployments.
    candidates=sorted(pair for pair in ex.resource_profiles if pair[1]!='qwen')
    dimensions=sorted(k for k in next(iter(ex.resource_profiles.values()))['values']
                      if k.startswith('kernel_fraction_') or k in ('offline_gpu_util_pct','offline_memory_util_pct','offline_power_w'))
    dimensions=[k for k in dimensions if all(ex.resource_profiles[p]['values'].get(k) is not None for p in candidates)]
    spans={k:(min(ex.resource_profiles[p]['values'][k] for p in candidates),max(ex.resource_profiles[p]['values'][k] for p in candidates)) for k in dimensions}
    spans={k:v for k,v in spans.items() if v[1]>v[0]}
    vectors={p:{k:(ex.resource_profiles[p]['values'][k]-lo)/(hi-lo) for k,(lo,hi) in spans.items()} for p in candidates}
    def distance(a,b): return math.sqrt(sum((vectors[a][k]-vectors[b][k])**2 for k in spans))
    first=min(candidates,key=lambda p:(ex.resource_profiles[p]['values']['kernel_time_ms_per_request'],p))
    selected=[first]
    while len(selected)<3:
        selected.append(max((p for p in candidates if p not in selected),key=lambda p:(min(distance(p,q) for q in selected),p)))
    return selected,dict(method='Lowest standalone kernel time/request anchor, then greedy farthest-first Euclidean distance in min-max normalized offline features; no deployment combination enumeration.',
        rationale='NVML device averages and kernel name taxonomy do not reliably identify pure compute vs memory bottlenecks. Use feature-space diversity fallback, not causal resource-class labels.',
        dimensions=list(spans),normalization={k:list(v) for k,v in spans.items()},
        candidates=[dict(task=k,model=m,normalized_features=vectors[k,m]) for k,m in candidates],
        selected=[dict(task=k,model=m) for k,m in selected],contender_standalone_demand=.2)


def make_plan(output=OUT):
    output=Path(output).resolve(); ex=extractor()
    audit=ROOT/'results/performance_model_foundation'
    audits={name:source(audit/name) for name in ('dataset_summary.md','feature_coverage.json','dataset_manifest.json')}
    coverage=json.loads((audit/'feature_coverage.json').read_text())
    selection,selection_meta=choose_contenders(ex)
    base_r={k:{m:(BASE[k] if m=='qwen' else 0.) for m in ex.catalog} for k in TASKS}
    anchor=ex.extract({'qwen':1},BASE,base_r)['per_model']['qwen']['task_demand_mix']
    mixes={'balanced':anchor,
        'text_dominant':dict(zip(TASKS,(.55,.08,.08,.10,.09,.10))),
        'vision_dominant':dict(zip(TASKS,(.06,.25,.23,.10,.26,.10))),
        'audio_dominant':dict(zip(TASKS,(.06,.06,.06,.50,.06,.26)))}
    configs=[]
    def add(name,family,u,mix,models,contender=None):
        x={m:int(m in models) for m in ex.catalog}
        r={k:{m:0. for m in ex.catalog} for k in TASKS}
        for k,q in mix.items(): r[k]['qwen']=u*q/(ex.catalog['qwen'].service_time_ms[k]/1000.)
        if contender:
            k,m=contender; r[k][m]=.2/(ex.catalog[m].service_time_ms[k]/1000.)
        workload={k:math.fsum(row.values()) for k,row in r.items()}
        # At least ~30 requests on every positive service path, including target paths.
        nominal=math.ceil(30/min(v for row in r.values() for v in row.values() if v>0))
        duration=max(60,min(900,nominal))
        expected=ex.extract(x,workload,r)
        runs=[]
        for rep,seed in enumerate(SEEDS,1):
            cfg=MeasurementConfig(x,workload,r,warmup_seconds=10,measurement_seconds=duration,
                random_seed=seed,routing_source='feature_directed_profiling_explicit_matrix',drain_timeout_seconds=60)
            cfg.validate({m:p.supported_tasks for m,p in ex.catalog.items()})
            run_id=f'{name}_rep_{rep}'
            config_path=output/family/'configs'/f'{run_id}.json'
            save(config_path,cfg.as_dict())
            runs.append(dict(run_id=run_id,repetition=rep,seed=seed,config=str(config_path),config_sha256=source(config_path)['sha256'],
                result=str(output/family/'raw'/f'{run_id}.json'),log=str(output/family/'logs'/f'{run_id}.log'),
                initial_arrival_phases=phases_for(workload,seed),expected_task_counts=dict(Counter(k for _,k in periodic_arrivals(workload,duration,seed)))))
        configs.append(dict(name=name,experiment_family=family,target_model='qwen',target_feature={'load_axis':'u_Q','task_mix_axis':'q_Q','colocation_axis':'C_Q'}[family],
            deployment=x,workload=workload,routing=r,expected_features=expected,expected_u=expected['per_model']['qwen']['standalone_service_demand'],
            expected_task_mix=expected['per_model']['qwen']['task_demand_mix'],chosen_co_runner=dict(task=contender[0],model=contender[1]) if contender else None,
            selection_rationale='Controlled feature condition; see family design and contender_selection in this plan.',
            repetitions=3,nominal_measurement_seconds=nominal,measurement_seconds=duration,runs=runs))
    for i in range(1,8): add(f'A_u_{i/10:.2f}','load_axis',i/10,mixes['balanced'],{'qwen'})
    for name in ('text_dominant','vision_dominant','audio_dominant'): add('B_'+name,'task_mix_axis',.4,mixes[name],{'qwen'})
    resident={'qwen'}|{m for k,m in selection}
    add('C_no_active_contender','colocation_axis',.4,mixes['balanced'],resident)
    for k,m in selection: add(f'C_{m}_{k}','colocation_axis',.4,mixes['balanced'],resident,(k,m))
    hashes={str(p.relative_to(ROOT)):source(p)['sha256'] for folder in ('joint_deploy/measurement','performance_model') for p in (ROOT/folder).glob('*.py')}
    hashes.update({str(p.relative_to(ROOT)):source(p)['sha256'] for p in (ROOT/'joint_deploy').glob('*worker.py')})
    plan=dict(stage='Module 2B Qwen feature-directed profiling',target_model='qwen',audit_sources=audits,
        existing_qwen_u_range=coverage['model_load_ranges']['qwen'],existing_controlled_curve_groups=coverage['curve_audit']['groups_with_two_distinct_loads'],
        feature_extractor='performance_model.features.FeatureExtractor.extract',protocol_hashes=hashes,
        seeds=list(SEEDS),configs=configs,unique_configurations=len(configs),maximum_repetitions=sum(c['repetitions'] for c in configs),
        family_design=dict(load_axis='Qwen only, existing six-task base composition; u=0.1..0.7 ascending. Complete 3 reps at a point before overload stop of higher loads.',
            task_mix_axis='u=0.4, Qwen only. Balanced reuses A_u_0.40 three runs; 3 dominant mixtures computed via r_kQ=u*q_kQ/s0_kQ.',
            colocation_axis='Qwen u/mix match balanced anchor. All selected contenders remain resident in all C conditions; only one contender path active at demand=0.2. Own no-active control isolates residency from traffic.'),
        balanced_rationale='Existing six-task base composition has repeated formal coverage, spans text/vision/audio, and provides provenance-aligned anchor; no claim of best/most stable mix.',
        anchor_aliases={'B_balanced':'A_u_0.40'},contender_selection=selection_meta,
        measurement_policy=dict(min_requests_per_positive_service_path=30,min_seconds=60,max_seconds=900,warmup_seconds=10,max_drain_seconds=60,
            drain_overload_seconds=30,safe_run_timeout_seconds=2400,completion_ratio_threshold=.98,
            backlog_rule='10 equal bins; second-half mean backlog increase >=max(10, 2% offered), >=75% adjacent increases, window-end backlog >=half growth threshold'),
        limitations=['C uses original per-task periodic arrivals and weighted routing. Adding contender traffic for the same task can change Qwen sample subsequence/arrival gaps despite identical target mean rates; record and audit, no causal claim.',
            'Independent process restarts each repetition; same seeds across conditions; B/C condition order randomized within repetition to reduce time confounding.',
            'Existing routing sweep is not a matched Qwen-only load curve; no existing sufficient matched condition is repeated. A balanced anchor is reused across A/B.'],
        execution_environment={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    save(output/'profiling_plan.json',plan)
    return plan

if __name__=='__main__':
    plan=make_plan()
    print(json.dumps(dict(configurations=plan['unique_configurations'],max_runs=plan['maximum_repetitions'],selected=plan['contender_selection']['selected'],
        nominal_measurement_hours=sum(c['measurement_seconds']*3 for c in plan['configs'])/3600),indent=2))
