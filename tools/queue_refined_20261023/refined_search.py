"""Finite, per-workload CPU policy screening in one frozen mean-service world.

Historical empirical means are the environmental service estimates; policy risk
quantiles are routing estimates, never latent GPU truth. Missing M/E leave full
U UNKNOWN. The exact/DFS claim applies only to this partial diagnostic domain.
"""
from __future__ import annotations
import argparse
from collections import Counter,deque
import copy
import hashlib
import heapq
import itertools
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.spillover_capacity_20261022.exposure import frozen_api
from tools.queue_refined_20261023.calibration import ServiceModel,visible_context
from tools.queue_refined_20261023.refined_policy import RefinedPolicy
from tools.queue_refined_20261023.policy import QueueAwarePolicy
from tools.queue_refined_20261023.guarded_policy import RateLimitedPolicy
from tools.queue_refined_20261023.replay import metrics,ns
from tools.queue_refined_20261023.search import search,identity

PARTIAL_TERMS='QTLP'
CPU_SEEDS=(213,214)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
def finite(value):return type(value) in (int,float) and math.isfinite(value)

def finite_candidates(entries):
    """17 per exact x: twelve mixed, four same-mode/risk no-spill, one legacy latency spill policy."""
    for e in entries:
        common=dict(deployment=e['deployment'],model_sha256=e['model_sha256'],label=e['label'])
        for mode,margin,risk in itertools.product(('conservative','global_incremental'),(0.,250.,750.),(.5,.9)):
            yield dict(**common,strategy='calibrated_'+mode,mode=mode,margin_ms=margin,risk_quantile=risk,allow_spill=True)
        for mode,risk in itertools.product(('conservative','global_incremental'),(.5,.9)):
            yield dict(**common,strategy='calibrated_specialist_only',mode=mode,margin_ms=0.,risk_quantile=risk,allow_spill=False)
        yield dict(**common,strategy='legacy_queue_latency',mode='legacy',margin_ms=0.,risk_quantile=None,allow_spill=True)

def structural_groups(items,catalog):
    """A quotient of the declared domain; no observed objective enters keys.

    The reduction is restricted to the pure-specialist resident case. Any
    uncertainty in the sufficient conditions keeps that candidate a singleton.
    """
    groups={}
    for c in items:
        x=c['deployment'];supported=[m for m in x if 'text' in catalog['models'][m]['supported_tasks']]
        reducible=(c['mode'] in ('conservative','global_incremental')
            and not any(m.startswith('generalist_') for m in x)
            and bool(supported) and all(m=='text' for m in supported))
        if reducible:
            key=('PURE_SPECIALIST_CALIBRATED',c['label'],json.dumps(x,sort_keys=True),c['model_sha256'],c['mode'],c['risk_quantile'])
        else:key=('SINGLETON',identity(c))
        groups.setdefault(key,[]).append(c)
    result=[]
    for members in groups.values():
        members.sort(key=identity);representative=members[0]
        proof=(dict(schema='queue_refined.pure_specialist_equivalence.v1',
            conditions=['No resident generalist family','All resident families supporting text equal text',
                'Identical deployment, frozen model, workload, quality, backgrounds, mode and risk quantile'],
            proof_steps=['The margin branch applies only to generalist candidates, absent here.',
                'allow_spill=False excludes only non-text families supporting text; none exist.',
                'No G2/text guard is constructed in either version.',
                'At identical snapshots all candidate scores and deterministic alias ties coincide.',
                'On the same frozen arrivals/input cohort, identical choices imply identical mean-environment contexts/services and FIFO completions; induction preserves every next snapshot and decision.',
                'Therefore route traces, post-route quality, Q/T/L/P and diagnostic feasibility coincide for every seed.'],
            scope='Exact deterministic CPU proxy and policy map at identical snapshots only; no identical real GPU observations or wall-clock route trace guarantee',
            objective_used_to_form_group=False,representative_rule='Smallest original-domain candidate_id') if len(members)>1 else None)
        result.append(dict(representative=representative,representative_id=identity(representative),
            members=members,member_ids=[identity(c) for c in members],proof=proof))
    return sorted(result,key=lambda g:g['representative_id'])

def quotient_bounded_search(items,seeds,evaluate,upper,catalog,*,log=None):
    """Evaluate one representative per proved group, then expand full ties.

    This is symbolic domain reduction before objective evaluation, not an
    objective cache or a timing experiment based on a precomputed score table.
    """
    started=time.perf_counter();group_started=time.perf_counter();groups=structural_groups(items,catalog)
    group_seconds=time.perf_counter()-group_started;representatives=[g['representative'] for g in groups]
    internal=io.StringIO();bounded=search(representatives,seeds,evaluate,upper,algorithm='bounded',log=internal)
    records={r['candidate_id']:r for r in map(json.loads,internal.getvalue().splitlines())}
    values={};unknown=[];expanded_bound_prunes=0
    for group in groups:
        rid=group['representative_id'];record=records.get(rid)
        if record is None:expanded_bound_prunes+=len(group['members'])
        for c in group['members']:
            cid=identity(c);expanded=copy.deepcopy(record) if record is not None else dict(status='BOUND_PRUNED_REPRESENTATIVE',runs=[],objective=None)
            expanded.update(candidate=c,candidate_id=cid,representative_id=rid,
                oracle_evaluated_for_this_identity=cid==rid and record is not None,
                structural_equivalence=group['proof'],objective_inherited_from_proved_equivalent=cid!=rid and record is not None)
            if record is not None and record['status']=='COMPLETE' and finite(record.get('objective')):values[cid]=record['objective']
            if rid in bounded['unknown_ids']:unknown.append(cid)
            if log:log.write(json.dumps(expanded,allow_nan=False)+'\n');log.flush()
    maximum=max(values.values()) if values else None
    ties=sorted(cid for cid,value in values.items() if maximum-value<=1e-12) if maximum is not None else []
    winner=ties[0] if ties else None
    result=dict(bounded);result.update(algorithm='quotient_bounded',inner_algorithm='bounded',candidates=len(items),representatives=len(groups),
        representative_ids=[g['representative_id'] for g in groups],proven_equivalent_skips=len(items)-len(groups),
        group_construction_seconds=group_seconds,elapsed_seconds=time.perf_counter()-started,
        objective=values[winner] if winner else None,exact_max_objective=maximum,winner_id=winner,tie_ids=ties,unknown_ids=sorted(set(unknown)),
        expanded_candidate_bound_prunes=expanded_bound_prunes,objective_cache_hits=0,
        timing_scope='Group construction, proved bound generation, real representative simulations and full-domain expansion; no precomputed objective table or objective cache',
        groups=[dict(representative_id=g['representative_id'],member_ids=g['member_ids'],proof=g['proof']) for g in groups],
        equivalence_scope='Declared deterministic CPU mean environment only, not a native GPU repeatability guarantee')
    return result

class MeanEnvironment:
    """Causal FIFO with service frozen at pre-enqueue dispatch context.

    One deterministic structural environment is reused by every policy. Routes
    can change its observed context, as expected for a shared resource proxy.
    No model sees future arrivals; no conditional service is reconstructed from
    the policy's risk quantile or future request finishes.
    """
    def __init__(self,aliases,model):
        self.aliases=aliases;self.model=model;self.pending={a:deque() for a in aliases};self.last={a:None for a in aliases}
        self.events=[];self.serial=0;self.results={};self.completed=[]
    def snapshot(self):
        fields=('request_id','task','pool_id','enqueued_ns','actual_submit_ns')
        return {a:dict(outstanding=[{k:r[k] for k in fields} for r in rows],last_completed_ns=self.last[a]) for a,rows in self.pending.items()}
    def _start(self,alias,now):
        row=self.pending[alias][0];rid=row['request_id'];finish=now+ns(row['_latent_ms']/1000);self.serial+=1
        heapq.heappush(self.events,(finish,self.serial,alias,rid));self.results[rid].update(start_ns=now,finish_ns=finish)
    def advance(self,now):
        while self.events and self.events[0][0]<=now:
            finish,serial,alias,rid=heapq.heappop(self.events);row=self.pending[alias].popleft()
            if row['request_id']!=rid:raise ValueError('FIFO event mismatch')
            self.last[alias]=finish;self.completed.append(rid)
            if self.pending[alias]:self._start(alias,finish)
    def enqueue(self,request,alias):
        family=self.aliases[alias]['family'];context=visible_context(self.snapshot(),alias,self.aliases)
        estimate=self.model.predict(family,request['task'],request['pool_id'],context=context,quantile=.5)
        service=estimate.get('mean_ms')
        if not finite(service) or service<=0:raise ValueError('UNKNOWN_ENVIRONMENT_SERVICE:'+family+':'+request['task'])
        row=dict(request,enqueued_ns=request['arrival_ns'],actual_submit_ns=request['arrival_ns'],_latent_ms=service)
        self.results[row['request_id']]=dict(request,alias=alias,family=family,environment_service_ms=service,
            environment_context=context,environment_evidence_status=estimate['evidence_status'])
        self.pending[alias].append(row)
        if len(self.pending[alias])==1:self._start(alias,request['arrival_ns'])

class ScreeningStudy:
    def __init__(self,base_packet,model_specs,text_rate,seeds=CPU_SEEDS):
        self.base=copy.deepcopy(base_packet);self.config=copy.deepcopy(base_packet['config']);self.config['workload']['text']=float(text_rate)
        self.api=frozen_api();self.catalog=self.config['catalog'];self.seeds=list(seeds)
        if any(not finite(self.config['weights'].get(t)) or self.config['weights'][t]<0 for t in PARTIAL_TERMS):raise ValueError('Nonnegative finite partial weights required for zero bound')
        if self.seeds!=list(CPU_SEEDS):raise ValueError('Fresh CPU-only seeds213/214 are generated traces; no reserved or native result reads')
        self.samples=Path(base_packet['inputs'])/'runtime-inputs.jsonl'
        if sha(self.samples)!=self.catalog['quality']['identity']['input_pool_sha256']:raise ValueError('Frozen sample pool fingerprint mismatch')
        self.entries=[];self.models={};self.cohorts={}
        for spec in model_specs:
            path=Path(spec['model_path']).resolve();document=json.loads(path.read_text());model=ServiceModel(document)
            x=document['deployment'];label=spec['label'];model_sha=sha(path)
            if any(e['label']==label for e in self.entries):raise ValueError('Duplicate model label')
            aliases={f'{m}#{i}':dict(family=m,supported_tasks=self.catalog['models'][m]['supported_tasks']) for m,n in x.items() for i in range(n)};model.validate_aliases(aliases)
            reference={}
            for task,rate in self.config['workload'].items():
                supported=sorted(m for m in x if task in self.catalog['models'][m]['supported_tasks'])
                if not supported:raise ValueError('Uncovered task in deployment')
                reference[task]={task if task in supported else supported[0]:rate}
            loss={}
            for m in x:
                loss[m]={}
                for task in self.catalog['models'][m]['supported_tasks']:
                    p=self.api['quality']._path(self.catalog['quality'],m,task);vv=[self.api['quality'].sample_totals(task,v) for v in p['samples'].values()]
                    loss[m][task]=sum(a for a,b in vv)/sum(b for a,b in vv)
            legacy=spec.get('legacy_service_ms');legacy_receipt=None
            if spec.get('legacy_contract_path'):
                legacy_path=Path(spec['legacy_contract_path']).resolve();ld=json.loads(legacy_path.read_text());legacy_receipt=dict(path=str(legacy_path),sha256=sha(legacy_path))
                if 'request_records' in ld:raise ValueError('Legacy input must be a frozen estimate contract, not a raw runtime result')
                if 'contexts' in ld:
                    options=[v for v in ld['contexts'] if v['candidate']['deployment']==x]
                    if len(options)!=1:raise ValueError('Missing exact legacy deployment context')
                    legacy=options[0]['service_ms']
                else:
                    if ld.get('deployment')!=x:raise ValueError('Legacy replica-state mismatch')
                    legacy=ld['service_ms']
            entry=dict(label=label,deployment=x,model_sha256=model_sha,model_path=str(path),aliases=aliases,
                reference_routing=reference,quality_loss=loss,legacy_service_ms=legacy,legacy_receipt=legacy_receipt)
            self.entries.append(entry);self.models[label]=model
        self.bylabel={e['label']:e for e in self.entries};self._own_floor_cache={};self._upper_cache={}
        for seed in self.seeds:
            trace=self.api['trace'].make_trace(self.config['workload'],self.config['horizon_seconds'],seed,self.config['arrival_process'])
            pool=self.api['workload'].SamplePool(self.samples,list(self.config['workload']),seed)
            cohort=[dict(request_id=f'seed{seed}-request{i:06d}',arrival_ns=ns(a),task=t,pool_id=pool.next(t)['pool_id']) for i,(a,t) in enumerate(trace)]
            self.cohorts[seed]=cohort
    def own_mean_upper_by_seed(self,c):
        """Proved only for this deterministic conditional-mean environment.

        Every latency >= its own service >= min possible dispatch-context
        mean after the same integer-nanosecond quantization as _start.
        Sample mean and empirical linear P95 are coordinate-monotone.
        Q/T >=0; M/E are absent from this partial objective, not estimated zero.
        """
        e=self.bylabel[c['label']];model=self.models[c['label']];x=e['deployment']
        cachekey=(c['label'],c['allow_spill'])
        if cachekey in self._upper_cache:return dict(self._upper_cache[cachekey])
        contexts=[dict(other_active_bin=b,g2_visible_outstanding=g) for b in ('0','1','2plus') for g in (False,True)]
        results={}
        for seed,cohort in self.cohorts.items():
            floors={t:[] for t in self.config['workload']}
            for r in cohort:
                task,pool=r['task'],r['pool_id'];key=(c['label'],task,pool,c['allow_spill'])
                if key not in self._own_floor_cache:
                    supported=[m for m in x if task in self.catalog['models'][m]['supported_tasks']]
                    if task!='text':supported=list(e['reference_routing'][task])
                    elif not c['allow_spill'] and 'text' in supported:supported=['text']
                    values=[];unknown=False
                    for family in supported:
                        for context in contexts:
                            value=model.predict(family,task,pool,context=context,quantile=.5).get('mean_ms')
                            if not finite(value) or value<=0:unknown=True
                            # _start rounds seconds to integer nanoseconds.
                            # Raw mean_ms can be slightly larger than that
                            # realized own service, which would invalidate an
                            # upper bound at sub-nanosecond boundaries.
                            else:values.append(ns(value/1000)/1e6)
                    self._own_floor_cache[key]=0. if unknown or not values else min(values)
                floors[task].append(self._own_floor_cache[key])
            total=sum(self.config['workload'].values());cost=0.
            for task,v in floors.items():
                if not v:continue
                alpha=self.config['workload'][task]/total/self.config['latency_reference_ms'][task]
                ordered=sorted(v);p=(len(v)-1)*.95;lo=math.floor(p);hi=math.ceil(p);p95=ordered[lo]+(ordered[hi]-ordered[lo])*(p-lo)
                cost+=alpha*(self.config['weights']['L']*statistics.mean(v)+self.config['weights']['P']*p95)
            results[seed]=-cost
        self._upper_cache[cachekey]=dict(results);return results

    def upper(self,c):return statistics.mean(self.own_mean_upper_by_seed(c).values())

    def evaluate(self,c,seed):
        e=self.bylabel[c['label']];model=self.models[c['label']];aliases=e['aliases'];rate=self.config['workload']['text'];x=e['deployment']
        if c['deployment']!=x or c['model_sha256']!=e['model_sha256']:raise ValueError('Candidate/model identity mismatch')
        if c['mode']=='legacy':
            if not e['legacy_service_ms']:return dict(seed=seed,U=None,eligible=None,status='UNKNOWN_LEGACY_ESTIMATE',formal_U=None)
            policy=QueueAwarePolicy(aliases,e['legacy_service_ms'],e['quality_loss'],self.config['weights'],self.config['latency_reference_ms'],mode='latency')
        else:
            policy=RefinedPolicy(aliases,model,e['quality_loss'],self.config['weights'],self.config['latency_reference_ms'],self.config['workload'],
                mode=c['mode'],margin_ms=c['margin_ms'],risk_quantile=c['risk_quantile'],allow_spill=c['allow_spill'],
                horizon_seconds=self.config['horizon_seconds'],energy_reference_watts=self.config['energy_reference_watts'])
        if c['allow_spill'] and 'text' in x and 'generalist_2' in x:
            policy=RateLimitedPolicy(policy,dict(schema='queue_aware_token_bucket_v1',task='text',family='generalist_2',rate_rps=rate/8,burst=1.))
        _,routes=self.api['trace'].expand_routes(x,e['reference_routing']);router=self.api['routing'].Router(routes)
        env=MeanEnvironment(aliases,model);decision_sha=hashlib.sha256();max_pending=0;spills=0
        try:
            for request in self.cohorts[seed]:
                env.advance(request['arrival_ns']);snapshot=env.snapshot();max_pending=max(max_pending,sum(len(v['outstanding']) for v in snapshot.values()))
                if request['task']=='text':decision=policy.choose(request['task'],request['pool_id'],request['arrival_ns'],snapshot)
                else:decision=dict(alias=router.choose(request['task']),status='FROZEN_BACKGROUND')
                alias=decision.get('alias')
                if alias not in aliases:raise ValueError('UNKNOWN_POLICY_ALIAS')
                spills+=request['task']=='text' and aliases[alias]['family']=='generalist_2'
                compact=dict(request_id=request['request_id'],alias=alias,status=decision['status'],scores=[dict(alias=a['alias'],score=a.get('score'),completion=a.get('predicted_completion_ms')) for a in decision.get('audit',[])])
                decision_sha.update(json.dumps(compact,sort_keys=True,allow_nan=False).encode());env.enqueue(request,alias)
            env.advance(10**30)
        except (ValueError,TypeError,KeyError) as exc:return dict(seed=seed,U=None,eligible=None,status='UNKNOWN_RETAINED',reason=str(exc),formal_U=None)
        requests=[env.results[r['request_id']] for r in self.cohorts[seed]];o=SimpleNamespace(config=self.config,catalog=self.catalog,api=self.api);d=metrics(o,requests)
        partial=-sum(self.config['weights'][t]*d['terms'][t] for t in PARTIAL_TERMS)
        rho={m:sum(r['environment_service_ms'] for r in requests if r['family']==m)/1000/self.config['horizon_seconds']/n for m,n in x.items()}
        planning=max(rho.values(),default=0)<=.8+1e-8 and d['drain_complete']
        return dict(seed=seed,U=partial,eligible=planning,status='PARTIAL_MEAN_ENVIRONMENT_DIAGNOSTIC',formal_U=None,
            partial_objective=partial,full_U=None,terms=d['terms'],metrics=d,
            diagnostic_rho_by_family=rho,diagnostic_drain_and_rho_feasible=planning,memory_feasibility='UNKNOWN',physical_stability='UNKNOWN',
            spilled_text_requests=spills,max_visible_pending=max_pending,decision_sha256=decision_sha.hexdigest(),
            cohort_sha256=digest(self.cohorts[seed]),environment='Same frozen conditional empirical mean at pre-enqueue dispatch context; FIFO deterministic; no actual shared-GPU truth',
            policy_quantile_is_not_environment_service=True,policy_cpu_overhead_simulated=False,ongoing_requests_repriced_after_dispatch=False,
            interference_limit='Dispatch-context mean proxy does not reprice in-flight service when later requests arrive; global externality forecast is not validated by this environment.')

def _pick(rows):
    if not rows:return None
    maximum=max(r['objective'] for r in rows)
    return min((r for r in rows if maximum-r['objective']<=1e-12),key=lambda r:r['candidate_id'])

def _rank(rows):
    remaining=list(rows);ordered=[]
    while remaining:
        maximum=max(r['objective'] for r in remaining)
        tied=sorted((r for r in remaining if maximum-r['objective']<=1e-12),key=lambda r:r['candidate_id'])
        ordered.extend(tied);ids={r['candidate_id'] for r in tied};remaining=[r for r in remaining if r['candidate_id'] not in ids]
    return ordered

def summarize(rows):
    feasible=_rank([r for r in rows if r['status']=='COMPLETE' and r['objective'] is not None])
    best=lambda fn:_pick([r for r in feasible if fn(r['candidate'])])
    modes={m:best(lambda c:c['mode']==m and c['allow_spill']) for m in ('conservative','global_incremental')}
    risk_matches={}
    for mode,row in modes.items():
        risk_matches[mode]=best(lambda c:row is not None and c['label']==row['candidate']['label'] and c['strategy']=='calibrated_specialist_only' and c['mode']==mode and c['risk_quantile']==row['candidate']['risk_quantile'])
    shared_risk={}
    for q in (.5,.9):
        winners={m:best(lambda c:c['mode']==m and c['allow_spill'] and c['risk_quantile']==q) for m in ('conservative','global_incremental')}
        shared_risk[str(q)]=dict(modes=winners,matched_controls={m:best(lambda c:row is not None and c['label']==row['candidate']['label'] and c['strategy']=='calibrated_specialist_only' and c['mode']==m and c['risk_quantile']==q) for m,row in winners.items()})
    primary_rows=[r for r in feasible if r['candidate']['label']=='primary' and r['candidate']['mode'] in ('conservative','global_incremental') and r['candidate']['allow_spill']]
    both=[r for r in primary_rows if all(v.get('spilled_text_requests',0)>0 for v in r['runs'])]
    any_spill=[r for r in primary_rows if any(v.get('spilled_text_requests',0)>0 for v in r['runs'])]
    primary=_pick(both or any_spill or primary_rows)
    matched=best(lambda c:primary is not None and c['label']=='primary' and c['strategy']=='calibrated_specialist_only' and c['mode']==primary['candidate']['mode'] and c['risk_quantile']==primary['candidate']['risk_quantile'])
    shortlist=dict(primary_calibrated_spill=primary,same_risk_primary_specialist=matched,
        primary_legacy_latency=best(lambda c:c['label']=='primary' and c['strategy']=='legacy_queue_latency'),
        strongest_different_deployment=best(lambda c:c['label']!='primary'),
        primary_spill_evidence='BOTH_CPU_SEEDS_POSITIVE' if both else ('ONE_OR_MORE_CPU_SEEDS_POSITIVE' if any_spill else 'NO_POSITIVE_SPILL_OBSERVED'),
        selection_rule='Among diagnostic-feasible primary calibrated candidates prefer spill on both seeds, then any seed, then best zero-spill; rank by partial QTLP within each tier. Same-risk specialist, legacy latency, strongest different deployment are independent controls.',
        full_U=None,native_validation_required=True)
    shortlist.update(calibrated_spill=shortlist['primary_calibrated_spill'],risk_matched_no_spill=shortlist['same_risk_primary_specialist'],legacy_latency=shortlist['primary_legacy_latency'],strong_other_deployment=shortlist['strongest_different_deployment'])
    return dict(finite_domain_winner=_pick(feasible),tie_rule='Exact maximum anchors absolute1e-12 tie set; smallest candidate_id within that set, matching search.search',top10=feasible[:10],top_per_mode=modes,risk_matched_specialist_controls=risk_matches,shared_risk_groups=shared_risk,native_shortlist=shortlist,
        structure_controls=dict(pure_specialist=best(lambda c:all(not m.startswith('generalist_') for m in c['deployment'])),
            other_mixed=best(lambda c:'generalist_2' in c['deployment'] and 'image' in c['deployment']),
            legacy_queue_latency=best(lambda c:c['strategy']=='legacy_queue_latency')),
        diagnostic_planning_feasible=len(feasible),formal_U=None,full_six_term_optimality=False)

def run_study(output,base_packet,model_manifest,loads=(1.85,1.89),seeds=CPU_SEEDS):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);base=json.loads(Path(base_packet).read_text());manifest=json.loads(Path(model_manifest).read_text());specs=manifest['models']
    files=[Path(__file__),Path(__file__).with_name('refined_policy.py'),Path(__file__).with_name('calibration.py'),Path(__file__).with_name('search.py'),Path(__file__).with_name('policy.py'),Path(__file__).with_name('guarded_policy.py'),Path(__file__).with_name('replay.py'),Path(base_packet),Path(model_manifest)]
    files += [Path(base['inputs'])/'runtime-inputs.jsonl']
    files += [Path(e['model_path']) for e in specs]+[Path(e['legacy_contract_path']) for e in specs if e.get('legacy_contract_path')]
    bindings={str(p.resolve()):sha(p) for p in files};allreports=[]
    for rate in loads:
        load_dir=output/f'text{rate:g}';load_dir.mkdir();study=ScreeningStudy(base,specs,rate,seeds);items=list(finite_candidates(study.entries))
        write(load_dir/'domain.json',dict(candidates=items,candidate_count=len(items),candidate_ids=[identity(c) for c in items],load=study.config['workload'],seeds=list(seeds),
            scope='51 deployment/policy candidates with 3 exact x. Partial QTLP screening only; M/E UNKNOWN.',formal_U=None))
        # Each search reconstructs a fresh environment and policy for every
        # callback. Frozen-model distribution caches only speed arithmetic;
        # they do not cache objectives or request outcomes.
        with (load_dir/'plain-evaluations.jsonl').open('x') as log:plain=search(items,seeds,study.evaluate,lambda c:0.,algorithm='plain_dfs',log=log)
        # New ServiceModels avoid cross-search distribution-cache timing bias.
        bounded_study=ScreeningStudy(base,specs,rate,seeds)
        with (load_dir/'bounded-evaluations.jsonl').open('x') as log:bounded=quotient_bounded_search(items,seeds,bounded_study.evaluate,bounded_study.upper,bounded_study.catalog,log=log)
        rows=[json.loads(l) for l in (load_dir/'plain-evaluations.jsonl').read_text().splitlines()]
        checks=[]
        for row in rows:
            vals=[r['U'] for r in row['runs'] if finite(r.get('U'))]
            if len(vals)==len(seeds):
                bounds=bounded_study.own_mean_upper_by_seed(row['candidate']);mean=sum(vals)/len(vals);upper=statistics.mean(bounds.values())
                single=[dict(seed=r['seed'],upper=bounds[r['seed']],objective=r['U'],violation=max(0.,r['U']-bounds[r['seed']])) for r in row['runs']]
                checks.append(dict(candidate_id=row['candidate_id'],upper=upper,mean=mean,violation=max(0.,mean-upper),per_seed=single))
        if any(c['violation']>1e-12 or any(s['violation']>1e-12 for s in c['per_seed']) for c in checks):raise ValueError('Own conditional-mean L/P floor violated')
        same=plain['winner_id']==bounded['winner_id'] and ((plain['objective'] is None and bounded['objective'] is None) or (plain['objective'] is not None and bounded['objective'] is not None and abs(plain['objective']-bounded['objective'])<1e-12))
        if not same:raise ValueError('Bounded and DFS disagree')
        # Independent grid identity audit, without calling finite_candidates.
        expected=set()
        for e in study.entries:
            for mode in ('conservative','global_incremental'):
                for margin in (0.,250.,750.):
                    for risk in (.5,.9):expected.add((e['label'],'calibrated_'+mode,mode,margin,risk,True))
            for mode in ('conservative','global_incremental'):
                for risk in (.5,.9):expected.add((e['label'],'calibrated_specialist_only',mode,0.,risk,False))
            expected.add((e['label'],'legacy_queue_latency','legacy',0.,None,True))
        actual={(c['label'],c['strategy'],c['mode'],c['margin_ms'],c['risk_quantile'],c['allow_spill']) for c in items}
        if actual!=expected or len(rows)!=len(items) or {r['candidate_id'] for r in rows}!={identity(c) for c in items}:raise ValueError('Finite-domain identity audit failed')
        full_by_id={r['candidate_id']:r for r in rows}
        expanded=[json.loads(l) for l in (load_dir/'bounded-evaluations.jsonl').read_text().splitlines()]
        expanded_by_id={r['candidate_id']:r for r in expanded};equivalence_audit=[]
        if set(expanded_by_id)!=set(full_by_id):raise ValueError('Quotient expansion lost declared identities')
        for group in bounded['groups']:
            representative=full_by_id[group['representative_id']];reference={r['seed']:r for r in representative['runs']}
            for member_id in group['member_ids']:
                member=full_by_id[member_id]
                for run in member['runs']:
                    ref=reference[run['seed']]
                    if run.get('U')!=ref.get('U') or run.get('eligible')!=ref.get('eligible') or run.get('decision_sha256')!=ref.get('decision_sha256'):
                        raise ValueError('Declared structural equivalence failed full DFS audit')
                computed=expanded_by_id[member_id]
                if computed.get('oracle_evaluated_for_this_identity'):
                    for run in computed['runs']:
                        ref=reference[run['seed']]
                        if run.get('U')!=ref.get('U') or run.get('eligible')!=ref.get('eligible') or run.get('decision_sha256')!=ref.get('decision_sha256'):
                            raise ValueError('Fresh representative evaluation differs from full DFS')
            equivalence_audit.append(dict(representative_id=group['representative_id'],members=len(group['member_ids']),
                independent_DFS_all_seed_U_feasibility_and_decision_hash_equal=True))
        if plain['tie_ids']!=bounded['tie_ids'] or plain['exact_max_objective']!=bounded['exact_max_objective']:raise ValueError('Quotient full tie set or exact maximum differs from DFS')
        comparison=summarize(rows)
        if (comparison['finite_domain_winner'] or {}).get('candidate_id')!=plain['winner_id']:raise ValueError('Summary tie rule differs from exact search')
        report=dict(load=study.config['workload'],plain_dfs=plain,bounded=bounded,quotient_bounded=bounded,same_winner=True,equivalence_audit=equivalence_audit,
            objective_scope='Negative weighted QTLP only; full original U and M/E UNKNOWN',formal_U=None,full_six_term_optimality=False,
            bound='Own-service floor: minimum frozen environmental mean quantized through ns(mean_ms/1000)/1e6 across eligible families and all6 visible contexts; monotone sample L/P, Q/T zero floors. Finite partial model only, not native GPU bound',bound_checks=checks,
            independent_identity_count=len(expected),bound_violations=0,zero_upper_bound_candidate_pruning_possible=0 if plain['exact_max_objective'] is None or plain['exact_max_objective']<=0 else None,numeric_bound_leaf_checks=len(checks),unknown_bound_leaf_checks=len(items)-len(checks),comparison=comparison,
            visited_status_counts={name:dict(Counter(json.loads(l)['status'] for l in (load_dir/f'{name}-evaluations.jsonl').read_text().splitlines())) for name in ('plain','bounded')},
            efficiency_scope='Small-domain CPU selection arithmetic only; structural quotient plus derived own-mean bound compared with full DFS, zero bound cannot prune candidates when incumbent is nonpositive; no speedup claimed unless observed. Model caches cold per search, objects instantiated before timer; no objective cache.',
            cpu_timing_exclusive=False,models=[dict(label=e['label'],deployment=e['deployment'],model_path=e['model_path'],model_sha256=e['model_sha256']) for e in study.entries])
        write(load_dir/'report.json',report);allreports.append(dict(load=rate,report_path=str(load_dir/'report.json'),winner=plain['winner_id'],partial_objective=plain['objective']))
    if any(sha(p)!=s for p,s in ((Path(p),s) for p,s in bindings.items())):raise ValueError('Sources changed during screening')
    write(output/'report.json',dict(per_load=allreports,no_cross_load_utility_ranking=True,bindings=bindings,formal_U=None,GPU_used=False,
        unmodeled=['Physical interference','M','E','Native policy computation overhead'],reserved_results_read=False,cpu_only_trace_seeds=list(seeds),native_results_for_cpu_seeds_read=False))
    print(json.dumps(allreports,allow_nan=False));return allreports

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--base-packet',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--loads',type=float,nargs='+',default=[1.85,1.89]);p.add_argument('--seeds',type=int,nargs=2,default=list(CPU_SEEDS));a=p.parse_args();run_study(a.output,a.base_packet,a.models,a.loads,a.seeds)
