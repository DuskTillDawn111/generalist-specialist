"""Finite deployment+policy search, with separate diagnostic and measured scopes.

No GPU calls. Diagnostic exactness refers only to the declared frozen numerical
model. A measured table can certify only its declared candidate/seed domain.
UNKNOWN leaves always prevent a formal whole-domain certificate.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.spillover_context_20261022.development_oracle import DevelopmentOracle,FAMILIES
from tools.spillover_context_20261022.development_bounds import DevelopmentBounds
from tools.spillover_context_20261022.search_development import OPTIONS,domain as static_domain
from gsdeploy.route_domain import fraction_rows
from tools.queue_refined_20261023.policy import QueueAwarePolicy
from tools.queue_refined_20261023.guarded_policy import RateLimitedPolicy
from tools.queue_refined_20261023.replay import simulate,metrics,ns

TASKS=('text','image','ocr','asr')
DYNAMIC=('queue_specialist_only','queue_quality','queue_latency')
TERMS='QTLPME'

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def identity(candidate):return hashlib.sha256(canonical(candidate).encode()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def finite(x):return type(x) in (int,float) and math.isfinite(x)

def candidates(catalog,workload,denominator=32):
    """7508 static leaves plus three causal policy leaves per covering x.

    The dynamic background reference uses its same-task specialist when
    resident, otherwise the first supported family in sorted order. Only text
    is adaptive; every candidate may choose zero generalist spill.
    """
    for nspec in itertools.product(*[[0]+v for v in OPTIONS.values()]):
        x={m:n for m,n in zip(OPTIONS,nspec) if n}
        supported={t:sorted(m for m in x if t in catalog['models'][m]['supported_tasks']) for t in TASKS}
        if any(not vv for vv in supported.values()):continue
        rows={t:list(fraction_rows(supported[t],2,[denominator] if t=='text' and denominator else [])) for t in TASKS}
        for routes in itertools.product(*(rows[t] for t in TASKS)):
            r={t:{m:float(q)*workload[t] for m,q in row if q>0} for t,row in zip(TASKS,routes)}
            yield dict(kind='static_swrr',deployment=x,routing=r)
        reference={t:{t if t in supported[t] else supported[t][0]:workload[t]} for t in TASKS}
        for strategy in DYNAMIC:
            yield dict(kind='causal_queue',deployment=x,reference_routing=reference,strategy=strategy,
                       overdue_remaining_ms=5.,adaptive_tasks=['text'],
                       spill_guard=(dict(schema='queue_aware_token_bucket_v1',task='text',family='generalist_2',rate_rps=workload['text']/8,burst=1.)
                           if strategy!='queue_specialist_only' and 'text' in x and 'generalist_2' in x else None))

def domain_contract(o,items):
    return dict(schema='gsdeploy.queue_joint_search_domain.v1',candidates=len(items),
        static=sum(c['kind']=='static_swrr' for c in items),dynamic=sum(c['kind']=='causal_queue' for c in items),
        deployment_count=len({canonical(c['deployment']) for c in items}),
        static_domain=static_domain(o.catalog),dynamic_strategies=list(DYNAMIC),
        dynamic_background='Same-task resident specialist else lexicographic supported resident; text alone adaptive',
        candidate_ids_sha256=hashlib.sha256('\n'.join(sorted(identity(c) for c in items)).encode()).hexdigest(),
        full_seven_pool_optimality=False,omitted_families=['generalist_1','generalist_3'],
        omitted_reason='Missing exact development service evidence; UNKNOWN, not infeasible',
        formal_U=None,physical_optimality=False)

class DiagnosticOracle:
    def __init__(self,seeds):
        self.o=DevelopmentOracle();self.seeds=list(seeds);self.contexts={};self.bound=DevelopmentBounds(self.o,self.seeds)
    def context(self,c):
        key=canonical(c['deployment'])
        if key in self.contexts:return self.contexts[key]
        o=self.o;x=c['deployment'];plan=dict(deployment=x,routing=c['reference_routing'])
        A,M,H,missing=o._context(plan)
        if missing:raise ValueError('Unknown frozen reference context '+canonical(missing))
        services={};loss={}
        # Potential services are frozen BEFORE any routing decisions. Actual
        # realized route pressure is never fed back into their generation.
        for m,n in x.items():
            services[m]={};loss[m]={}
            for t in o.catalog['models'][m]['supported_tasks']:
                bs=o._baselines(m,t,n)
                if (m,t,n) not in o.domain or not bs or any(v is None for v in bs.values()):continue
                lm=statistics.mean(math.log(v) for v in bs.values());other=sum(v for f,v in A.items() if f!=m)
                values={}
                for sid,b in bs.items():
                    feat=[float(f==m) for f in FAMILIES]+[A[m],other,M,H[m],(math.log(b)-lm)*other]
                    values[sid]=b*math.exp(sum(feat[i]*v for i,v in zip(o.columns,o.beta)))
                services[m][t]=values
                record=o.api['quality']._path(o.catalog['quality'],m,t)
                totals=[o.api['quality'].sample_totals(t,v) for v in record['samples'].values()]
                loss[m][t]=sum(a for a,b in totals)/sum(b for a,b in totals)
        value=(services,loss,dict(reference_plan=plan,A=A,M=M,H=H));self.contexts[key]=value;return value
    def upper(self,c):
        if c['kind']=='static_swrr':return self.bound.upper(c['deployment'],c['routing'],None)
        # New dynamic proof: actual per-request latency >= own fixed potential
        # service, irrespective of dispatch. P95 monotonic under pointwise
        # lower bounds. Q,T,E >=0. NNLS memory pressure/intercept omitted.
        services,loss,_=self.context(c);o=self.o;values=[];rates=o.config['workload'];total=sum(rates.values());x=c['deployment']
        for seed in self.seeds:
            _,raw=o._cohort(seed);per={t:[] for t in TASKS}
            for a,t,sid in raw:
                eligible=[m for m in x if t in services.get(m,{}) and sid in services[m][t]]
                if t!='text':eligible=list(c['reference_routing'][t])
                elif c['strategy']=='queue_specialist_only' and 'text' in eligible:eligible=['text']
                # Missing potential path gives no positive lower bound.
                own=min((services.get(m,{}).get(t,{}).get(sid,0.) for m in eligible),default=0.)
                per[t].append(own)
            delay=sum(rates[t]/total/o.config['latency_reference_ms'][t]*(o.config['weights']['L']*statistics.mean(v)+o.config['weights']['P']*percentile(v)) for t,v in per.items())
            values.append(-delay-o.config['weights']['M']*self.bound.memory_norm(x))
        return statistics.mean(values)
    def evaluate(self,c,seed):
        o=self.o
        if c['kind']=='static_swrr':
            raw=o.evaluate(dict(deployment=c['deployment'],routing=c['routing']),seed=seed,
                           diagnostic_extrapolation=True,strict_context_checks=False)
            d=raw.get('diagnostic_extrapolation_only') or {}
            return dict(seed=seed,U=d.get('U'),terms=d.get('terms'),eligible=d.get('planning_feasible'),
                        formal_U=None,status='STATIC_DEVELOPMENT_MODEL',rho_by_family=d.get('rho_by_family'))
        services,loss,context=self.context(c);x=c['deployment'];aliases={f'{m}#{i}':dict(family=m,supported_tasks=o.catalog['models'][m]['supported_tasks']) for m,n in x.items() for i in range(n)}
        allowed=aliases
        if c['strategy']=='queue_specialist_only' and 'text' in x:allowed={a:v for a,v in aliases.items() if v['family']=='text'}
        policy=QueueAwarePolicy(allowed,services,loss,o.config['weights'],o.config['latency_reference_ms'],
            mode='latency' if c['strategy']=='queue_latency' else 'quality_aware',overdue_remaining_ms=c['overdue_remaining_ms'])
        if c.get('spill_guard'):policy=RateLimitedPolicy(policy,c['spill_guard'])
        _,routes=o.api['trace'].expand_routes(x,c['reference_routing']);router=o.api['routing'].Router(routes)
        _,raw=o._cohort(seed);cohort=[dict(request_id=f'seed{seed}-{i:06d}',arrival_ns=ns(a),task=t,pool_id=sid) for i,(a,t,sid) in enumerate(raw)]
        def choose(req,snapshot):
            if req['task']!='text':return dict(alias=router.choose(req['task']))
            return policy.choose(req['task'],req['pool_id'],req['arrival_ns'],snapshot)
        try:requests,decisions=simulate(cohort,aliases,services,choose)
        except (ValueError,KeyError,TypeError) as exc:return dict(seed=seed,U=None,formal_U=None,eligible=None,status='UNKNOWN',reason=str(exc))
        d=metrics(o,requests);rate={t:{} for t in TASKS};counts=Counter((r['task'],r['family']) for r in requests);totals=Counter(r['task'] for r in requests)
        for (t,m),count in counts.items():rate[t][m]=o.config['workload'][t]*count/totals[t]
        # Resource diagnostics use realized route exposure, a posthoc numerical
        # extrapolation, not prospective capacity certification or service refit.
        A,M,H,missing=o._context(dict(deployment=x,routing=rate));demand=sum(A.values()) if all(v is not None for v in A.values()) else None
        features=dict(intercept=1.,raw_resident_over_budget=M,sum_baseline_pressure=demand,workers=sum(x.values()))
        rv={k:o._resource(name,features,x,False)['diagnostic_value'] for k,name in [('M',o.memory_name),('W',o.window_name),('S',o.startup_name)]}
        terms=d['terms'];terms['M']=rv['M']*o.catalog['safe_factor'] if finite(rv['M']) else None
        terms['E']=(rv['W']+rv['S']/o.config['startup_amortization_seconds'])/o.config['energy_reference_watts'] if finite(rv['W']) and finite(rv['S']) else None
        known=all(finite(v) and v>=0 for v in terms.values())
        u=-sum(o.config['weights'][t]*terms[t] for t in TERMS) if known else None
        rho={m:sum(r['environment_service_ms'] for r in requests if r['family']==m)/1000/o.config['horizon_seconds']/n for m,n in x.items()}
        eligible=(max(rho.values())<=.8+1e-8 and terms['M']<=1+1e-8 and d['drain_complete']) if known else None
        return dict(seed=seed,U=u,formal_U=None,terms=terms,eligible=eligible,status='DYNAMIC_DEVELOPMENT_MODEL',
            rho_by_family=rho,routing_realized=rate,counts_by_task_family=d['counts_by_task_family'],
            reference_context=context,service_frozen_before_route=True,resource_scope='Realized exposure in unvalidated NNLS M/E extrapolation',
            actual_stability='UNKNOWN',physical_admission=False)

def percentile(v):
    vv=sorted(v);q=(len(vv)-1)*.95;a=math.floor(q);b=math.ceil(q);return vv[a]+(vv[b]-vv[a])*(q-a)

def search(items,seeds,evaluate,upper,*,algorithm='plain_dfs',log=None):
    """No objective cache: each visited candidate/seed invokes evaluator anew.

    bounded uses only supplied proved complete-candidate upper bounds, then
    seed aggregation bound U<=0. Equality is evaluated to preserve all ties.
    """
    if algorithm not in {'plain_dfs','bounded'}:raise ValueError('Unknown algorithm')
    start=time.perf_counter();leaves=0;calls=0;pruned=0;unknown=[];best=None;values_by_id={};bound_seconds=0.
    ordered=[(c,0.) for c in items]
    if algorithm=='bounded':
        bs=time.perf_counter();ordered=[(c,upper(c)) for c in items];ordered.sort(key=lambda v:(-v[1],identity(v[0])));bound_seconds=time.perf_counter()-bs
    for c,ub in ordered:
        cid=identity(c)
        if algorithm=='bounded' and best is not None and ub<best-1e-12:pruned+=1;continue
        leaves+=1;runs=[];values=[];state='COMPLETE'
        for seed in seeds:
            r=evaluate(c,seed);calls+=1;runs.append(r)
            if r.get('eligible') is False:
                state='INELIGIBLE'
                if algorithm=='bounded':break
                continue
            if r.get('eligible') is not True or not finite(r.get('U')):
                if state!='INELIGIBLE':state='UNKNOWN'
                unknown.append(cid)
                if algorithm=='bounded':break
                continue
            if r['U']>1e-12:raise ValueError('U<=0 aggregation proof violated')
            values.append(r['U'])
            if algorithm=='bounded' and best is not None and sum(values)/len(seeds)<best-1e-12:state='SEED_BOUND_PRUNED';break
        value=statistics.mean(values) if state=='COMPLETE' and len(values)==len(seeds) else None
        if state=='COMPLETE' and value is not None:
            values_by_id[cid]=value
            best=value if best is None else max(best,value)
        record=dict(candidate_id=cid,candidate=c,runs=runs,status=state,objective=value,bound=ub)
        if log:log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
    # One exact maximum anchors tolerance, making ties traversal-independent.
    # Return the selected candidate's value, which may be just below the max.
    ties=sorted(cid for cid,v in values_by_id.items() if best-v<=1e-12) if best is not None else []
    winner=ties[0] if ties else None
    return dict(algorithm=algorithm,objective=values_by_id[winner] if winner else None,exact_max_objective=best,winner_id=winner,tie_ids=ties,
        candidates=len(items),visited_leaves=leaves,oracle_calls=calls,bound_pruned=pruned,unknown_ids=unknown,
        conditional_known_domain_optimality=best is not None,whole_declared_domain_certified=best is not None and not unknown,
        elapsed_seconds=time.perf_counter()-start,bound_seconds=bound_seconds,
        timing_scope='One process-local search; report cold end-to-end separately. No objective cache.',formal_U=None)

def measured_table_search(document,algorithm='plain_dfs'):
    """Exact comparison of explicit GPU candidate table; never an extrapolator.

    Required table: candidates, seeds, records [{candidate_id,seed,utility:{
    Q_norm,..,E_norm,U,eligible,utility_complete,...}}]. Missing/UNKNOWN rows
    remain UNKNOWN. Proven infeasibility must be explicit, not eligible=False.
    """
    items=document['candidates'];seeds=document['seeds'];records={}
    weights=document.get('weights',{t:1/6 for t in TERMS})
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Nonempty unique seed list required')
    if len({identity(c) for c in items})!=len(items):raise ValueError('Duplicate candidate identities')
    if set(weights)!=set(TERMS) or any(not finite(v) or v<0 for v in weights.values()):raise ValueError('Six finite nonnegative weights required')
    declared={identity(c) for c in items}
    for row in document['records']:
        key=(row['candidate_id'],row['seed'])
        if key[0] not in declared or key[1] not in seeds:raise ValueError('Measurement outside declared candidate/seed domain')
        if key in records:raise ValueError('Duplicate candidate/seed measurement')
        records[key]=row
    def evaluate(c,seed):
        row=records.get((identity(c),seed),{});v=row.get('utility',{});known=all(finite(v.get(t+'_norm')) and v[t+'_norm']>=0 for t in TERMS)
        verified=all(v.get(k) is True for k in ['cohort_verified','constraints_verified','energy_verified','memory_verified','quality_identity_verified'])
        arithmetic=known and finite(v.get('U')) and math.isclose(v['U'],-math.fsum(weights[t]*v[t+'_norm'] for t in TERMS),rel_tol=0.,abs_tol=1e-10)
        eligible=True if v.get('eligible') is True and v.get('utility_complete') is True and verified and arithmetic else (False if row.get('proven_infeasible') is True else None)
        return dict(U=v.get('U') if eligible is True else None,eligible=eligible,seed=seed,
                    status='MEASURED' if eligible is True else 'UNKNOWN_OR_PROVEN_INFEASIBLE')
    result=search(items,seeds,evaluate,lambda c:0.,algorithm=algorithm)
    result.update(scope='Declared measured candidate table only',formal_U=result['objective'],physical_full_pool_optimality=False,
        oracle_call_kind='Lookup of existing measurements; no inference is executed and lookup wall time is not GPU measurement efficiency.')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--algorithm',choices=['plain_dfs','bounded'],default='bounded');parser.add_argument('--seeds',type=int,nargs='+',default=[107,108]);parser.add_argument('--table',type=Path)
    args=parser.parse_args();cold=time.perf_counter();args.output.mkdir(parents=True,exist_ok=False)
    source_paths=[Path(__file__),Path(__file__).with_name('policy.py'),Path(__file__).with_name('replay.py'),Path(__file__).with_name('guarded_policy.py')]
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    if args.table:report=measured_table_search(json.loads(args.table.read_text()),args.algorithm)
    else:
        oracle=DiagnosticOracle(args.seeds);items=list(candidates(oracle.o.catalog,oracle.o.config['workload']));write(args.output/'domain.json',domain_contract(oracle.o,items))
        with (args.output/'evaluations.jsonl').open('x') as log:report=search(items,args.seeds,oracle.evaluate,oracle.upper,algorithm=args.algorithm,log=log)
        oracle.o.verify_inputs();report.update(oracle_provenance=oracle.o.provenance(),objective_scope='Unvalidated finite diagnostic model; all native/full U UNKNOWN',formal_U=None,
            bound_contract=dict(static=oracle.bound.contract(),dynamic='Own fixed potential service floor for L/P; zero Q/T/E; nonnegative NNLS residency+worker memory floor. Different proof; no static FIFO bound on dynamic routes.'))
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=sources[str(p)] for p in source_paths):raise ValueError('Search source changed during execution')
    report['source_fingerprints']=sources
    report['cpu_timing_exclusive']=False;report['cpu_timing_note']='GPU native batch and other CPU agents may run concurrently; wall times descriptive, not exclusive benchmark.'
    report['cold_end_to_end_seconds']=time.perf_counter()-cold;report['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();write(args.output/'report.json',report);print(json.dumps(report,allow_nan=False))
if __name__=='__main__':main()
