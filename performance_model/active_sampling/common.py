import hashlib
import json
import math
from pathlib import Path
from performance_model.profiling.plan import ROOT, extractor, save
from performance_model.dataset import input_results

OUT=ROOT/'results/performance_model_active_sampling'
SEEDS=(4101,5202,6303)


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:16]


def deployment_id(x):
    return 'dep_'+identity({m:n for m,n in sorted(x.items()) if n})


def configuration_id(data):
    keys=('deployment','workload_rps','routing_rps','warmup_seconds','measurement_seconds','drain_timeout_seconds')
    return 'cfg_'+identity({**{k:data[k] for k in keys},'arrival_process':'deterministic_periodic'})


def existing_sources():
    sources=list(input_results())
    sources += [(p,'module2b',p.parents[1].name) for p in sorted((ROOT/'results/performance_model_profiling/qwen').glob('*/raw/*.json'))]
    paths=[p.resolve() for p,_,_ in sources]
    if len(paths)!=len(set(paths)): raise ValueError('Duplicate source paths')
    return sources


def feature_vector(features):
    q=features['per_model']['qwen']
    return {'own_load/u':q['standalone_service_demand'],
        **{'mix/'+k:v for k,v in q['task_demand_mix'].items()},
        **{'pressure/'+k:v['value'] for k,v in q['colocation_pressure'].items()},
        **{'deployment/'+k:v for k,v in features['deployment_features'].items()},
        'target/model_memory_mib':q['model_memory_mib'],'target/instances':q['instances']}


def fit_distance_space(vectors):
    out={}
    for key in sorted({k for v in vectors for k in v}):
        values=[v[key] for v in vectors if v.get(key) is not None]
        out[key]={'min':min(values) if values else None,'max':max(values) if values else None,
                  'available':len(values),'varying':bool(values) and max(values)>min(values)}
    return out


def distance(a,b,space):
    blocks={}
    for key,scale in space.items():
        if not scale['varying'] or a.get(key) is None or b.get(key) is None: continue
        blocks.setdefault(key.split('/')[0],[]).append(((a[key]-b[key])/(scale['max']-scale['min']))**2)
    if not blocks: raise ValueError('No comparable varying features')
    return math.sqrt(sum(sum(v)/len(v) for v in blocks.values())/len(blocks))


def nearest(vector,population,space):
    return min(distance(vector,v,space) for v in population)
