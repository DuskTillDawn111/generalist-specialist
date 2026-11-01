"""CPU closed-loop routing diagnostic; never a native shared-GPU experiment.

All policies share the frozen arrivals, inputs and latent point-model service
table. The simulator alone sees completion events. Routing sees only pending
requests already enqueued, observed completions and frozen service estimates.
Dynamic M/E are uncalibrated, so full U always remains UNKNOWN.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict,deque
import copy
import hashlib
import heapq
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.spillover_context_20261022.development_oracle import DevelopmentOracle,FAMILIES

DEPLOYMENT={'text':2,'generalist_2':1,'ocr':1}
STRATEGIES=['no_split_swrr','fixed_1of32_swrr','queue_latency','queue_quality','queue_specialist_only']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
def ns(seconds):return int(round(seconds*1_000_000_000))
def percentile(values,q=.95):
    values=sorted(values);a=(len(values)-1)*q;lo=math.floor(a);hi=math.ceil(a)
    return values[lo]+(values[hi]-values[lo])*(a-lo)

def frozen_tables(oracle):
    """Evaluate all potential paths at one predeclared no-split context.

    G2/text has zero offered load in this context. Its potential service is
    an explicitly extrapolated diagnostic, never a measured counterfactual.
    No refitting and no post-hoc per-policy native service reads occur here.
    """
    o=oracle;c=o.config
    plan=dict(deployment=DEPLOYMENT,routing={'text':{'text':c['workload']['text']},
        'image':{'generalist_2':c['workload']['image']},'asr':{'generalist_2':c['workload']['asr']},'ocr':{'ocr':c['workload']['ocr']}})
    A,M,H,missing=o._context(plan)
    if missing:raise ValueError('Frozen reference context missing: '+str(missing))
    table={};loss={};details=[]
    for family,n in DEPLOYMENT.items():
        table[family]={};loss[family]={}
        for task in o.catalog['models'][family]['supported_tasks']:
            baseline=o._baselines(family,task,n)
            if (family,task,n) not in o.domain or baseline is None or any(v is None for v in baseline.values()):
                details.append(dict(family=family,task=task,status='UNKNOWN_EXACT_PATH'));continue
            logmean=statistics.mean(math.log(v) for v in baseline.values());other=sum(v for m,v in A.items() if m!=family)
            values={}
            for sid,b in baseline.items():
                features=[float(f==family) for f in FAMILIES]+[A[family],other,M,H[family],(math.log(b)-logmean)*other]
                values[sid]=b*math.exp(math.fsum(features[i]*v for i,v in zip(o.columns,o.beta)))
            table[family][task]=values
            # Aggregate historical loss is the only quality input to routing.
            record=o.api['quality']._path(o.catalog['quality'],family,task);totals=[o.api['quality'].sample_totals(task,v) for v in record['samples'].values()]
            loss[family][task]=sum(a for a,b in totals)/sum(b for a,b in totals)
            details.append(dict(family=family,task=task,status='FROZEN_POINT_PROXY',mean_ms=statistics.mean(values.values()),
                quality_aggregate_loss=loss[family][task],context_extrapolation=family=='generalist_2' and task=='text'))
    return table,loss,dict(reference_plan=plan,context=dict(A=A,M=M,H=H),paths=details,
        service_equation='b_perinput * exp(frozen19run context coefficients), n already in baseline; no additional capacity factor',
        no_refit=True,native_policy_service_read=False,shared_gpu_interference_simulated=False,
        ideal_estimation='Policy estimates equal simulator point services by default; ideal model diagnostic only')

class ClosedLoop:
    def __init__(self,aliases,actual_service_ms,estimate_service_ms=None):
        self.aliases=aliases;self.actual=actual_service_ms;self.estimates=estimate_service_ms
        self.pending={a:deque() for a in aliases};self.last={a:None for a in aliases}
        self.events=[];self.serial=0;self.completed=[];self.results={}

    def snapshot(self):
        fields=['request_id','task','pool_id','enqueued_ns','actual_submit_ns']
        return {a:dict(outstanding=[{k:r[k] for k in fields} for r in rows],last_completed_ns=self.last[a]) for a,rows in self.pending.items()}

    def _start(self,alias,now):
        row=self.pending[alias][0]
        value=self.actual[self.aliases[alias]['family']][row['task']][row['pool_id']]
        if not math.isfinite(value) or value<=0:raise ValueError('Unknown/nonpositive simulator service')
        self.serial+=1;finish=now+ns(value/1000)
        heapq.heappush(self.events,(finish,self.serial,alias,row['request_id']))
        self.results[row['request_id']].update(start_ns=now,finish_ns=finish,environment_service_ms=value)

    def advance(self,now):
        observed=[]
        while self.events and self.events[0][0]<=now:
            finish,serial,alias,rid=heapq.heappop(self.events);head=self.pending[alias].popleft()
            if head['request_id']!=rid:raise ValueError('FIFO head mismatch')
            self.last[alias]=finish;self.completed.append(rid);observed.append(rid)
            if self.pending[alias]:self._start(alias,finish)
        return observed

    def enqueue(self,request,alias):
        if alias not in self.aliases or request['task'] not in self.aliases[alias]['supported_tasks']:raise ValueError('Invalid chosen alias')
        # Match observable host submission: queued requests are submitted to
        # the worker pipe immediately. Internal worker start is never exposed
        # through the policy snapshot.
        row=dict(request,enqueued_ns=request['arrival_ns'],actual_submit_ns=request['arrival_ns'])
        self.results[row['request_id']]=dict(request,alias=alias,family=self.aliases[alias]['family'])
        self.pending[alias].append(row)
        if len(self.pending[alias])==1:self._start(alias,request['arrival_ns'])

def simulate(cohort,aliases,services,choose,*,estimated_services=None):
    """Generic causal event engine, with choose receiving no environment data."""
    sim=ClosedLoop(aliases,services,estimated_services);decisions=[]
    for request in cohort:
        observed=sim.advance(request['arrival_ns']);snapshot=sim.snapshot()
        decision=choose(copy.deepcopy(request),copy.deepcopy(snapshot))
        alias=decision.get('alias')
        audit=dict(request_id=request['request_id'],arrival_ns=request['arrival_ns'],task=request['task'],pool_id=request['pool_id'],
            queue_snapshot=snapshot,newly_observed_completed_ids=observed,
            observed_completed_ids=list(sim.completed),decision=decision)
        # Route decision is now fixed, before the environment service is read.
        sim.enqueue(request,alias);decisions.append(audit)
    sim.advance(10**30)
    return [sim.results[r['request_id']] for r in cohort],decisions

def metrics(oracle,requests):
    c=oracle.config;per_task={};rates=c['workload'];total=sum(rates.values());quality=oracle.catalog['quality'];Q=0.
    for task in rates:
        rows=[r for r in requests if r['task']==task];lat=[(r['finish_ns']-r['arrival_ns'])/1e6 for r in rows]
        nums=[];dens=[]
        for r in rows:
            score=oracle.api['quality']._path(quality,r['family'],task)['samples'][r['pool_id']]
            a,b=oracle.api['quality'].sample_totals(task,score);nums.append(a);dens.append(b)
        loss=sum(nums)/sum(dens);Q+=rates[task]/total*loss
        per_task[task]=dict(requests=len(rows),mean_ms=statistics.mean(lat),p95_ms=percentile(lat),historical_loss=loss,
            quality_error_numerator=sum(nums),quality_reference_denominator=sum(dens))
    terms=dict(Q=Q,T=1-sum(r['finish_ns']<=ns(c['horizon_seconds']) for r in requests)/len(requests),
        L=sum(rates[t]/total*per_task[t]['mean_ms']/c['latency_reference_ms'][t] for t in rates),
        P=sum(rates[t]/total*per_task[t]['p95_ms']/c['latency_reference_ms'][t] for t in rates),M=None,E=None)
    return dict(terms=terms,U=None,Q_T_L_P_partial_cost=sum(c['weights'][t]*terms[t] for t in 'QTLP'),
        partial_cost_is_not_full_U=True,per_task=per_task,drain_complete=all(r['finish_ns']<=ns(c['horizon_seconds']+c['drain_seconds']) for r in requests),
        counts_by_task_family={t:dict(Counter(r['family'] for r in requests if r['task']==t)) for t in rates},
        dynamic_memory_energy_calibrated=False,physical_stability='UNKNOWN',quality_scope='historical frozen sample_totals scored after all decisions; not native measured accuracy')

def run(output,seeds=(107,108)):
    from tools.queue_aware_20261023.policy import QueueAwarePolicy
    output=Path(output);output.mkdir(parents=True,exist_ok=False);o=DevelopmentOracle();services,loss,context=frozen_tables(o)
    source_paths=[Path(__file__),Path(__file__).with_name('policy.py')];sources={str(p):sha(p) for p in source_paths}
    write(output/'frozen-simulation-contract.json',dict(deployment=DEPLOYMENT,strategies=STRATEGIES,seeds=list(seeds),
        service_ms=services,aggregate_quality_loss=loss,context=context,sources=sources,formal_U=None,
        no_future_service_or_reference_passed_to_policy=True,dynamic_policy_claim='CPU ideal-point-model diagnostic; no GPU/physical improvement claim'))
    aliases={f'{m}#{i}':dict(family=m,supported_tasks=o.catalog['models'][m]['supported_tasks']) for m,n in DEPLOYMENT.items() for i in range(n)}
    results=[]
    for seed in seeds:
        trace,raw=o._cohort(seed);cohort=[dict(request_id=f'seed{seed}-request{i:06d}',arrival_ns=ns(a),task=t,pool_id=sid) for i,(a,t,sid) in enumerate(raw)]
        cohort_sha=hashlib.sha256(json.dumps(cohort,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        for strategy in STRATEGIES:
            share=1/32 if strategy=='fixed_1of32_swrr' else 0.
            routing={'text':{m:q for m,q in [('text',1.85*(1-share)),('generalist_2',1.85*share)] if q},
                'image':{'generalist_2':.01},'asr':{'generalist_2':.01},'ocr':{'ocr':.01}}
            _,expanded=o.api['trace'].expand_routes(DEPLOYMENT,routing);router=o.api['routing'].Router(expanded)
            mode='quality_aware' if strategy in ['queue_quality','queue_specialist_only'] else 'latency'
            eligible={a:v for a,v in aliases.items() if v['family']=='text'} if strategy=='queue_specialist_only' else aliases
            policy=QueueAwarePolicy(aliases=eligible,service_ms=services,quality_loss=loss,weights=o.config['weights'],
                latency_refs_ms=o.config['latency_reference_ms'],mode=mode)
            def choose(req,snapshot):
                if req['task']!='text' or strategy in ['no_split_swrr','fixed_1of32_swrr']:
                    return dict(alias=router.choose(req['task']),status='FROZEN_SWRR_BACKGROUND' if req['task']!='text' else 'FROZEN_SWRR',trigger='STATIC_ROUTE')
                return policy.choose(task=req['task'],pool_id=req['pool_id'],now_ns=req['arrival_ns'],pending_snapshot=snapshot)
            try:
                requests,decisions=simulate(cohort,aliases,services,choose)
                result=dict(seed=seed,strategy=strategy,status='COMPLETE_CPU_DIAGNOSTIC',cohort_sha256=cohort_sha,
                    metrics=metrics(o,requests),requests=len(requests),decisions=len(decisions))
                stem=f'{strategy}-seed{seed}'
                for suffix,rows in [('requests',requests),('decisions',decisions)]:
                    with (output/f'{stem}-{suffix}.jsonl').open('x') as stream:
                        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            except Exception as exc:
                result=dict(seed=seed,strategy=strategy,status='FAILED_RETAINED',error=f'{type(exc).__name__}: {exc}',U=None,cohort_sha256=cohort_sha)
            results.append(result);write(output/f'{strategy}-seed{seed}-report.json',result)
    o.verify_inputs()
    if any(sha(p)!=s for p,s in sources.items()):raise ValueError('Replay/policy source changed during run')
    write(output/'report.json',dict(results=results,context=context,oracle_provenance=o.provenance(),sources=sources,
        formal_U=None,shared_gpu_interference_simulated=False,production_changed=False,seeds_are_development_not_independent=True))
    files={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    write(output/'manifest.json',dict(files=files,sources=sources,inputs=o.provenance()['inputs']))
    print(json.dumps([dict(seed=r['seed'],strategy=r['strategy'],status=r['status'],metrics=r.get('metrics')) for r in results]))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[107,108]);a=p.parse_args();run(a.output,a.seeds)
