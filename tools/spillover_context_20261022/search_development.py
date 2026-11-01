"""Autonomous finite-domain search of an explicit real-data development objective.

The production U remains UNKNOWN. The solver callback's U is *only* the named
diagnostic numerical objective; a solver certificate never transfers to reality.
No GPU, runtime-result reads, production changes, or invented complete oracle.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.spillover_context_20261022.development_oracle import DevelopmentOracle
from gsdeploy.coupled_mode import MODE,PLACEHOLDER_SCHEMA
from gsdeploy.milp import JointMilpOptimizer
from gsdeploy.optimizer import UtilityOptimizer
from gsdeploy.route_domain import fraction_rows
from tools.spillover_context_20261022.development_bounds import DevelopmentBounds

OPTIONS={'generalist_2':[1],'text':[1,2],'ocr':[1],'image':[1],'asr':[1,2]}

def identity(plan):
    return json.dumps({'deployment':dict(sorted(plan['deployment'].items())),
        'routing':{t:{m:round(q,14) for m,q in sorted(row.items()) if q>0}
                   for t,row in sorted(plan['routing'].items())}},sort_keys=True,separators=(',',':'))

def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value

def write(path,value):
    with Path(path).open('x') as stream:json.dump(clean(value),stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')

def domain(catalog,denominator=32,K=2,options=None):
    options=options or OPTIONS;count=0;deployments=0;routes_by_deployment=[]
    for ns in itertools.product(*[[0]+v for v in options.values()]):
        x={m:n for m,n in zip(options,ns) if n};nr=1
        for t in ['text','image','ocr','asr']:
            supported=[m for m in x if t in catalog['models'][m]['supported_tasks']]
            nr*=len(list(fraction_rows(supported,K,[denominator] if t=='text' and denominator else [])))
        if nr:count+=nr;deployments+=1;routes_by_deployment.append(dict(deployment=x,complete_plans=nr))
    return dict(replica_options=options,zero_residency_state=True,K=K,
        route_pair_denominators_by_task={'text':[denominator]} if denominator else {},
        complete_deployments=deployments,complete_plans=count,by_deployment=routes_by_deployment,
        all_task_routes_free=True,role_winner_prespecified=False,
        omitted_pools={'generalist_1':'UNKNOWN exact development path/n service',
                      'generalist_3':'UNKNOWN exact development path/n service'},
        full_seven_pool_optimality=False,continuous_optimality=False)

class Adapter:
    """Explicit translation for the existing numerical search engines only."""
    def __init__(self,oracle,seeds,log):
        self.oracle=oracle;self.seeds=seeds;self.log=log;self.cache={};self.calls=0;self.hits=0;self.oracle_seconds=0.

    def __call__(self,data,x,r,*args,**kwargs):
        self.calls+=1;plan=dict(deployment=x,routing=r);key=identity(plan)
        if key in self.cache:self.hits+=1;return self.cache[key]
        started=time.perf_counter();runs=[]
        for seed in self.seeds:
            raw=self.oracle.evaluate(plan,seed=seed,diagnostic_extrapolation=True,strict_context_checks=False)
            d=raw['diagnostic_extrapolation_only'] or {}
            runs.append(dict(seed=seed,formal_U=raw['U'],diagnostic_U=d.get('U'),terms=d.get('terms'),
                rho_by_family=d.get('rho_by_family'),resource_feasible=d.get('planning_feasible',d.get('resource_feasible')),
                memory_budget_pass=d.get('memory_budget_pass'),rho_threshold_pass=d.get('rho_threshold_pass'),
                drain_complete=d.get('drain_complete'),strict_arithmetic_U=raw.get('supported_domain_development_arithmetic_U'),
                unknown_reasons=sorted(set(v['kind'] for v in raw['unknown_reasons']))))
        self.oracle_seconds+=time.perf_counter()-started
        known=all(run['diagnostic_U'] is not None for run in runs)
        feasible=all(run['resource_feasible'] is True for run in runs)
        u=sum(run['diagnostic_U'] for run in runs)/len(runs) if known else None
        terms={t:sum(run['terms'][t] for run in runs)/len(runs) for t in 'QTLPME'} if known else None
        record=dict(plan=plan,identity=key,runs=runs,diagnostic_objective=u,diagnostic_feasible=feasible,
                    formal_U=None,physical_optimality=False)
        self.log.write(json.dumps(clean(record),allow_nan=False)+'\n');self.log.flush()
        p=dict(U=u,constraints_eligible=feasible,performance_unknown=not known,memory_unknown=False,
            reason='EXPLICIT_REAL_DATA_DEVELOPMENT_DIAGNOSTIC_ONLY',diagnostic_partial_U=None,
            scenario_U_low=u,scenario_delay_costs=[sum(self.oracle.config['weights'][t]*terms[t] for t in 'TLP')] if known else [None],
            M_norm=terms['M'] if known else None,Q_norm=terms['Q'] if known else None,
            Q_contribution=-self.oracle.config['weights']['Q']*terms['Q'] if known else None,
            formal_U=None,development_only=True,execution_admission=False,runs=runs)
        self.cache[key]=p
        return p

def search(output,algorithm,seeds=(107,),time_limit=1200,denominator=32,K=2,options=None,bounds='zero'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    source_paths=[Path(__file__),ROOT/'tools/spillover_context_20261022/development_bounds.py',ROOT/'gsdeploy/optimizer.py',ROOT/'gsdeploy/milp.py',ROOT/'gsdeploy/route_domain.py',ROOT/'gsdeploy/quality.py',ROOT/'gsdeploy/coupled_mode.py']
    source_before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    oracle=DevelopmentOracle();c=oracle.config;catalog=copy.deepcopy(oracle.catalog)
    # Keep only the declared local numerical domain; missing G1/G3 are explicit
    # unresolved full-pool candidates, never assigned made-up bad scores.
    options=options or OPTIONS
    catalog['models']={m:catalog['models'][m] for m in options}
    catalog['performance_mode']=MODE;catalog['memory_mode']=MODE
    catalog[MODE]=dict(schema=PLACEHOLDER_SCHEMA,status='UNIMPLEMENTED')
    spec=domain(catalog,denominator,K,options);write(output/'domain.json',spec)
    context=dict(horizon=c['horizon_seconds'],seed=seeds[0],arrival_process=c['arrival_process'],
        drain_seconds=c['drain_seconds'],energy_reference_watts=c['energy_reference_watts'],
        startup_amortization_seconds=c['startup_amortization_seconds'],search_policy='nominal',
        milp_capacity_witness=False,route_pair_denominators_by_task=spec['route_pair_denominators_by_task'],
        milp_time_limit_seconds=time_limit,search_max_states=1000000,milp_max_iterations=100000,
        milp_max_columns=100000)
    # For seed aggregation the unchanged per-task Q bound from a single seed
    # need not bound mean Q. Zero is the explicit safe shared objective floor.
    klass=JointMilpOptimizer if algorithm=='milp' else UtilityOptimizer
    if algorithm!='milp':context.update(dfs_bound_pruning=algorithm!='plain_dfs')
    with (output/'evaluations.jsonl').open('x') as log:
        adapter=Adapter(oracle,list(seeds),log)
        solver=klass(catalog,c['workload'],c['gpu_memory_mib'],c['latency_reference_ms'],c['weights'],**context)
        envelope=DevelopmentBounds(oracle,list(seeds)) if bounds=='derived' else None
        if envelope:envelope.install(solver,algorithm)
        elif len(seeds)>1:
            if algorithm=='milp':solver.floor=lambda *args:0.
            else:solver.upper=lambda *args:0.
        with patch('gsdeploy.milp.predict_scenarios',side_effect=adapter),patch('gsdeploy.optimizer.predict_scenarios',side_effect=adapter):
            result=solver.solve(K=K,replica_options=options)
    oracle.verify_inputs()
    source_after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    if source_before!=source_after:raise ValueError('Search mathematical source changed during run')
    result=dict(algorithm=algorithm,seeds=list(seeds),objective='mean_of_explicit_extrapolated_development_U',
        interpretation='Any solver certificate applies only to this finite diagnostic mathematical objective; formal U and physical/full-seven-pool optimum remain UNKNOWN.',
        result=result,bound_variant=bounds,bound_contract=envelope.contract() if envelope else None,
        wall_seconds=time.perf_counter()-started,oracle_seconds=adapter.oracle_seconds,
        adapter_calls=adapter.calls,unique_complete_evaluations=len(adapter.cache),cache_hits=adapter.hits,
        oracle_provenance=oracle.provenance(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mathematical_sources_before=source_before,mathematical_sources_after=source_after,
        timing_contract='Fresh oracle and bounds caches per independent process/run, no precomputed utility table. Wall includes init, hashes, setup, real oracle and solver. CPU threads externally limited to1.',
        domain=spec,formal_U=None,formal_optimality_certified=False)
    write(output/'report.json',result)
    summarize(output/'evaluations.jsonl',output/'top-controls.json')
    print(json.dumps(dict(output=str(output),algorithm=algorithm,seeds=list(seeds),wall_seconds=result['wall_seconds'],
        status=result['result']['status'],evaluations=len(adapter.cache),winner=result['result']['winner']),default=str),flush=True)
    return result

def summarize(log_path,output):
    rows=[json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()]
    rows=[r for r in rows if r['diagnostic_feasible'] and r['diagnostic_objective'] is not None]
    rows.sort(key=lambda r:(-r['diagnostic_objective'],r['identity']))
    winner=rows[0] if rows else None
    tests={'no_text_split':lambda p:len(p['routing']['text'])==1,
        'pure_generalist':lambda p:all(m.startswith('generalist_') for m in p['deployment']),
        'pure_specialist':lambda p:all(not m.startswith('generalist_') for m in p['deployment']),
        'text_replicas_at_most_one':lambda p:p['deployment'].get('text',0)<=1,
        'different_deployment':lambda p:winner and p['deployment']!=winner['plan']['deployment']}
    controls={name:next((r for r in rows if test(r['plan'])),None) for name,test in tests.items()}
    write(output,dict(top10=rows[:10],controls=controls,feasible_evaluations=len(rows),
                     controls_are_evaluated_subset_only=True,requires_complete_plain_DFS_for_domain_optimum=True))

def aggregate(logs,output):
    bykey={}
    for logfile in logs:
        for line in Path(logfile).read_text().splitlines():
            r=json.loads(line);key=r['identity'];bykey.setdefault(key,dict(plan=r['plan'],identity=key,runs={}));
            for run in r['runs']:bykey[key]['runs'][run['seed']]=run
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with (output/'evaluations.jsonl').open('x') as stream:
        for row in bykey.values():
            runs=[row['runs'][s] for s in [107,108] if s in row['runs']];known=len(runs)==2 and all(r['diagnostic_U'] is not None for r in runs)
            vals=[r['diagnostic_U'] for r in runs] if known else []
            row.update(runs=runs,diagnostic_objective=sum(vals)/2 if known else None,
                diagnostic_feasible=known and all(r['resource_feasible'] is True for r in runs),
                diagnostic_min=min(vals) if vals else None,diagnostic_max=max(vals) if vals else None,
                formal_U=None,physical_optimality=False)
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    summarize(output/'evaluations.jsonl',output/'top-controls.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--algorithm',choices=['milp','dfs','plain_dfs'],default='plain_dfs');parser.add_argument('--seeds',type=int,nargs='+',default=[107])
    parser.add_argument('--time-limit',type=float,default=1200);parser.add_argument('--denominator',type=int,default=32);parser.add_argument('--K',type=int,default=2)
    parser.add_argument('--bounds',choices=['zero','derived'],default='zero')
    parser.add_argument('--aggregate-logs',type=Path,nargs='+')
    a=parser.parse_args()
    if a.aggregate_logs:aggregate(a.aggregate_logs,a.output)
    else:search(a.output,a.algorithm,a.seeds,a.time_limit,a.denominator,a.K,bounds=a.bounds)
