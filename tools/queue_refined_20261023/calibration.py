"""Deployment-isolated empirical service distributions from training native runs.

No quality predictions, references, or held-out seeds enter this model. Context
is dispatch-time host-visible pipeline activity, never hidden worker activity.
Predictions are selected-path diagnostics and not counterfactual certificates.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right,bisect_left
from collections import Counter,defaultdict
import hashlib,json,math,random
from pathlib import Path
import statistics

SCHEMA='gsdeploy.queue_refined_service_calibration.v2'
DEVELOPMENT_SEEDS=(113,114)
CALIBRATION_SEEDS=(201,202)
ALLOWED_SEEDS=DEVELOPMENT_SEEDS+CALIBRATION_SEEDS
CALIBRATION_CLAIM='NEW_RANDOMIZED_CALIBRATION_TRAINING_NOT_EFFICACY_VALIDATION'

def deployment_identity(deployment):
    if not isinstance(deployment,dict) or not deployment or any(not isinstance(f,str) or not f or type(n) is not int or n<=0 for f,n in deployment.items()):
        raise ValueError('Explicit positive integer deployment required')
    return dict(sorted(deployment.items()))

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def finite(x):return type(x) in (int,float) and math.isfinite(x)
def key(*parts):return json.dumps(parts,separators=(',',':'),ensure_ascii=False)
def sha(path):
    p=Path(path).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(path,doc):
    with Path(path).open('x') as f:json.dump(doc,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def visible_context(snapshot,target_alias,aliases):
    """Only presence of visible pending work; no worker starts/service truth."""
    active=sum(bool(v.get('outstanding',[])) for alias,v in snapshot.items() if alias!=target_alias)
    generalist=any(bool(v.get('outstanding',[])) for alias,v in snapshot.items()
                  if aliases.get(alias,{}).get('family')=='generalist_2')
    return dict(other_active_bin='0' if active==0 else '1' if active==1 else '2plus',
                g2_visible_outstanding=generalist)

def normalize_context(context):
    context=context or {}
    if context.get('other_active_bin') not in ('0','1','2plus') or type(context.get('g2_visible_outstanding')) is not bool:
        return None
    return dict(other_active_bin=context['other_active_bin'],g2_visible_outstanding=context['g2_visible_outstanding'])

def context_key(context):
    c=normalize_context(context)
    return key(c['other_active_bin'],c['g2_visible_outstanding']) if c else None

def _raw_records(raw,receipt,*,training_phase):
    seed=raw['transport_config']['random_seed']
    if raw.get('status')!='completed' or raw.get('backend')!='native':raise ValueError('Only completed native training')
    contract=raw['policy_contract'];aliases=contract['aliases'];deployment=deployment_identity(contract['deployment'])
    if dict(Counter(a['family'] for a in aliases.values()))!=deployment:raise ValueError('Training alias deployment mismatch')
    records=[]
    for r in raw['request_records']:
        if r.get('status')!='completed':raise ValueError('Incomplete training request')
        value=r.get('worker_response',{}).get('service_ms')
        if not finite(value) or value<=0:raise ValueError('Missing positive actual worker service')
        alias=r['model'];family=raw['replica_to_family'][alias]
        if family!=aliases[alias]['family']:raise ValueError('Training alias family mismatch')
        context=visible_context(r['routing_decision']['snapshot'],alias,aliases)
        # Whitelist actual service/context/identity only; never quality truth.
        records.append(dict(family=family,task=r['task'],pool_id=r['pool_id'],context=context,
            service_ms=value,seed=seed,request_id=r['request_id'],raw_sha256=receipt['sha256'],
            legacy_predicted_service_ms=contract['service_ms'][family][r['task']][r['pool_id']],
            deployment=deployment,strategy=contract['strategy'],training_phase=training_phase))
    return records

def extract_development(report_path,deployment=None):
    """Whitelist complete development113/114 before opening any raw file."""
    report_path=Path(report_path).resolve();report=json.loads(report_path.read_bytes())
    if report.get('seeds')!=[113,114] or len(report.get('observations',[]))!=14:
        raise ValueError('Only complete original development14 at seeds113/114 is permitted')
    selected=deployment_identity(deployment) if deployment is not None else None
    records=[];sources=[sha(report_path)]
    for ob in report['observations']:
        if ob.get('seed') not in DEVELOPMENT_SEEDS:raise ValueError('Reserved seed forbidden before raw read')
        path=Path(ob['raw']['path']).resolve()
        if not path.is_relative_to(report_path.parent.parent):raise ValueError('Raw outside declared development directory')
        receipt=sha(path)
        if receipt['sha256']!=ob['raw']['sha256']:raise ValueError('Raw SHA mismatch')
        raw=json.loads(path.read_bytes());sources.append(receipt)
        seed=raw['transport_config']['random_seed']
        if seed not in DEVELOPMENT_SEEDS or seed!=ob['seed']:raise ValueError('Nondevelopment or mismatched seed')
        rows=_raw_records(raw,receipt,training_phase='development')
        if selected is None or deployment_identity(raw['policy_contract']['deployment'])==selected:records.extend(rows)
    return records,sources

def extract_completed_calibration(packet_path,progress_path,deployment=None):
    """Consume only terminal, SHA-bound calibration201/202, never held-out runs.

    Packet phase/seeds/probe declarations are checked before any raw is opened.
    All four jobs must have completed; partial or ongoing batches fail closed.
    """
    packet_path=Path(packet_path).resolve();progress_path=Path(progress_path).resolve()
    packet=json.loads(packet_path.read_bytes());state=json.loads(progress_path.read_bytes())
    if packet.get('schema')!='gsdeploy.refined_calibration_packet.v1' or packet.get('phase')!='calibration' or packet.get('seeds')!=list(CALIBRATION_SEEDS):
        raise ValueError('Only explicitly declared calibration201/202 permitted')
    if packet.get('no_retries') is not True:raise ValueError('Calibration packet must declare no retries')
    if state.get('status')!='COMPLETE' or state.get('packet')!=sha(packet_path) or state.get('no_retries') is not True:
        raise ValueError('Calibration requires COMPLETE SHA-bound no-retry batch')
    jobs=packet.get('jobs',[]);conditions=[]
    for job in jobs:
        identity=job['identity'];contract=job['policy_contract'];probe=contract.get('probe',{})
        if job.get('seed') not in CALIBRATION_SEEDS or identity.get('seed')!=job['seed'] or job.get('phase')!='calibration' or identity.get('phase')!='calibration':raise ValueError('Reserved or wrong-phase calibration job')
        if identity.get('measurement_claim')!=CALIBRATION_CLAIM or identity.get('schema')!='gsdeploy.refined_calibration_job.v1':raise ValueError('Calibration training claim missing')
        if job['id']!=digest(identity) or identity.get('policy_contract_sha256')!=digest(contract):raise ValueError('Calibration identity/contract SHA mismatch')
        if identity.get('deployment')!=contract.get('deployment'):raise ValueError('Calibration deployment identity mismatch')
        if set(probe)!={'schema','probability','namespace'} or probe.get('schema')!='randomized_text_probe_v1' or probe.get('namespace')!='queue-refined-calibration-20261023' or type(probe.get('probability')) not in (int,float) or probe['probability'] not in (0.,1/16):raise ValueError('Undeclared randomized calibration probe')
        prereg=identity.get('predeclaration',{})
        if not prereg.get('path') or sha(prereg['path'])!=prereg:raise ValueError('Calibration preregistration SHA mismatch')
        conditions.append((job['seed'],probe['probability']))
    if sorted(conditions)!=[(201,0.),(201,1/16),(202,0.),(202,1/16)]:raise ValueError('Complete declared four-job probe required')
    terminals=state.get('jobs',[])
    if len(terminals)!=4 or len({t['job_id'] for t in terminals})!=4:raise ValueError('Four distinct terminal calibration jobs required')
    terminal_by_id={t['job_id']:t for t in terminals}
    if set(terminal_by_id)!={j['id'] for j in jobs}:raise ValueError('Calibration terminal job identity mismatch')
    selected=deployment_identity(deployment) if deployment is not None else None
    records=[];sources=[sha(packet_path),sha(progress_path)]
    for job in jobs:
        terminal=terminal_by_id[job['id']]
        if terminal.get('status')!='completed' or terminal.get('exit_code')!=0:raise ValueError('Unsuccessful calibration terminal')
        path=Path(terminal['raw']['path']).resolve()
        if not path.is_relative_to(progress_path.parent):raise ValueError('Calibration raw outside declared batch')
        receipt=sha(path)
        if receipt!=terminal['raw']:raise ValueError('Calibration raw SHA mismatch')
        raw=json.loads(path.read_bytes());sources.append(receipt)
        if raw.get('experiment_job_id')!=job['id'] or raw.get('executed_measurement_identity')!=job['identity'] or raw.get('executed_measurement_id')!=job['id'] or raw.get('experiment_phase')!='calibration' or raw.get('measurement_claim')!=CALIBRATION_CLAIM:raise ValueError('Raw calibration phase/claim/identity mismatch')
        if raw.get('policy_contract')!=job['policy_contract'] or raw['transport_config']['random_seed']!=job['seed']:raise ValueError('Raw calibration contract/seed mismatch')
        # Check the input-independent, one-draw-per-text-arrival stream itself.
        probe=job['policy_contract']['probe'];stream=hashlib.sha256(f"{probe['namespace']}:{job['seed']}:measurement".encode()).hexdigest()
        rng=random.Random(int(stream,16));index=0
        for r in raw['request_records']:
            if r['task']!='text':continue
            index+=1;coin=rng.random();observed=r['routing_decision'].get('decision',{}).get('probe',{})
            if observed.get('spec')!=probe or observed.get('stream_sha256')!=stream or observed.get('draw_index')!=index or observed.get('coin')!=coin or observed.get('requested_probe')!=(coin<probe['probability']):raise ValueError('Raw randomized probe stream mismatch')
        rows=_raw_records(raw,receipt,training_phase='calibration')
        if selected is None or deployment_identity(job['policy_contract']['deployment'])==selected:records.extend(rows)
    return records,sources

def fit(records,*,sources=(),input_prior=16.,context_prior=32.,label='development113114'):
    if not records:raise ValueError('Empty calibration data')
    deployment=deployment_identity(records[0].get('deployment'))
    if any(deployment_identity(r.get('deployment'))!=deployment for r in records):raise ValueError('Mixed deployments forbidden: fit one exact resident deployment at a time')
    if input_prior<=0 or context_prior<=0:raise ValueError('Positive shrinkage pseudo counts required')
    groups={x:defaultdict(list) for x in ('family','family_task','family_task_context','family_task_input','family_task_input_context')}
    counts=Counter();run_ids=set()
    for r in records:
        if r['seed'] not in ALLOWED_SEEDS:raise ValueError('Reserved seed forbidden for fitting')
        if r['seed'] in CALIBRATION_SEEDS and r.get('training_phase')!='calibration':raise ValueError('Calibration seed requires explicit calibration training provenance')
        if r['family'] not in deployment:raise ValueError('Observation family outside deployment')
        if not finite(r['service_ms']) or r['service_ms']<=0:raise ValueError('Invalid service observation')
        f,t,p=r['family'],r['task'],r['pool_id'];c=context_key(r['context'])
        for name,k in [('family',key(f)),('family_task',key(f,t)),('family_task_context',key(f,t,c)),
                       ('family_task_input',key(f,t,p)),('family_task_input_context',key(f,t,p,c))]:
            groups[name][k].append(float(r['service_ms']))
        counts[str(r['seed'])]+=1;run_ids.add(r['raw_sha256'])
    return dict(schema=SCHEMA,label=label,deployment=deployment,training_seeds=sorted({r['seed'] for r in records}),
        observations=len(records),training_runs=len(run_ids),observations_by_seed=dict(counts),sources=list(sources),
        shrinkage=dict(input_prior=input_prior,context_prior=context_prior),
        strata={name:{k:sorted(v) for k,v in values.items()} for name,values in groups.items()},
        context_semantics='At dispatch: number of OTHER aliases with visible outstanding work binned0/1/2plus; G2 visible outstanding indicator. Pipeline pending is not actualGPU concurrency.',
        training_target='Actual completed worker_response.service_ms; excludes queue wait; measurement cohort only.',
        quality_used=False,per_input_true_answers_used=False,reserved_data_used=False,
        limits=['Selected-path historical fit only; no counterfactual service proof.',
            'Applicable only to exact declared resident deployment/replica counts; no cross-deployment service transfer.',
            'Multiple repeated requests of same32-input pool are correlated; sample counts are not independent evidence counts.',
            'Same observed context can conceal GPU contention and actual worker-start variation.',
            'Residual conditions on service survival but supplied head age is host-inferred, not directly observed true service age.'])

def empirical(values,weight):
    return [(v,weight/len(values)) for v in values] if values and weight>0 else []

def mix(parent,values,prior):
    if not values:return parent
    w=len(values)/(len(values)+prior)
    return [(v,p*(1-w)) for v,p in parent]+empirical(values,w)

def quantile_of(distribution,q):
    if not finite(q) or not 0<q<1:raise ValueError('Quantile must be in(0,1)')
    total=math.fsum(w for _,w in distribution);s=0
    for value,weight in sorted(distribution):
        s+=weight
        if s>=q*total:return value
    return max(v for v,_ in distribution)

class ServiceModel:
    def __init__(self,document):
        if document.get('schema')!=SCHEMA or document.get('quality_used') is not False or document.get('reserved_data_used') is not False:
            raise ValueError('Unsupported or leaked calibration document')
        if any(s not in ALLOWED_SEEDS for s in document['training_seeds']):raise ValueError('Reserved training seed')
        self.deployment=deployment_identity(document.get('deployment'))
        self.document=document;self.strata=document['strata'];self.prior=document['shrinkage']
        self._cache={}

    def validate_aliases(self,aliases):
        if not isinstance(aliases,dict) or dict(Counter(spec['family'] for spec in aliases.values()))!=self.deployment:
            raise ValueError('Service model deployment does not match resident alias family/replica counts')
        return True

    def _distribution(self,family,task,pool_id,context):
        cachekey=key(family,task,pool_id,context_key(context))
        if cachekey in self._cache:return self._cache[cachekey]
        g=self.strata;ft=g['family_task'].get(key(family,task),[])
        # Cross-task transfer is explicitly UNKNOWN even if family data exist.
        if not ft:return [],dict(family_task=0,input=0,input_context=0,task_context=0),None
        distribution=empirical(ft,1.)
        c=context_key(context);input_values=g['family_task_input'].get(key(family,task,pool_id),[])
        task_context=g['family_task_context'].get(key(family,task,c),[]) if c else []
        input_context=g['family_task_input_context'].get(key(family,task,pool_id,c),[]) if c else []
        distribution=mix(distribution,task_context,self.prior['context_prior'])
        distribution=mix(distribution,input_values,self.prior['input_prior'])
        distribution=mix(distribution,input_context,self.prior['context_prior'])
        support=dict(family_task=len(ft),input=len(input_values),input_context=len(input_context),task_context=len(task_context))
        # Weighted strata overlap deliberately for shrinkage; compress equal
        # values and precompute sums to make head residual queries logarithmic.
        merged=defaultdict(float)
        for value,weight in distribution:merged[value]+=weight
        ordered=sorted(merged.items());values=[];cum=[];mom=[];square=[]
        total=math.fsum(w for _,w in ordered);cw=cm=cs=0.
        for value,weight in ordered:
            weight/=total;cw+=weight;cm+=weight*value;cs+=weight*weight
            values.append(value);cum.append(cw);mom.append(cm);square.append(cs)
        dist=dict(values=values,cumulative=cum,moment= mom,squared=square)
        result=(dist,support,normalize_context(context));self._cache[cachekey]=result
        return result

    @staticmethod
    def _quantile(dist,q):
        if not finite(q) or not 0<q<1:raise ValueError('Quantile must be in(0,1)')
        i=min(bisect_left(dist['cumulative'],q*dist['cumulative'][-1]),len(dist['values'])-1)
        return dist['values'][i]

    def predict(self,family,task,pool_id,context=None,quantile=.9):
        distribution,support,c=self._distribution(family,task,pool_id,context)
        if not distribution:return dict(mean_ms=None,quantile_ms=None,conservative_ms=None,support=support,evidence_status='UNKNOWN_FAMILY_TASK',context=c)
        mean=distribution['moment'][-1]/distribution['cumulative'][-1]
        q=self._quantile(distribution,quantile)
        return dict(mean_ms=mean,quantile_ms=q,conservative_ms=max(mean,q),support=support,
            evidence_status='HISTORICAL_SELECTED_PATH' if support['input_context'] else 'SHRUNK_MISSING_INPUT_OR_CONTEXT',
            context=c,quantile=quantile,scope='Empirical selected-path distribution, not a calibrated confidence bound.')

    def residual(self,family,task,pool_id,age_ms,context=None,quantile=.9):
        if not finite(age_ms) or age_ms<0:raise ValueError('Finite nonnegative age_ms required')
        if not finite(quantile) or not 0<quantile<1:raise ValueError('Quantile must be in(0,1)')
        distribution,support,c=self._distribution(family,task,pool_id,context)
        if not distribution:return dict(conditional_mean_ms=None,quantile_ms=None,conservative_ms=None,
            survival_probability=None,effective_survivor_support=0,evidence_status='UNKNOWN_FAMILY_TASK',age_ms=age_ms,support=support)
        i=bisect_right(distribution['values'],age_ms)
        total=distribution['cumulative'][-1]
        before=distribution['cumulative'][i-1] if i else 0.
        sw=total-before;survival=sw/total
        if i<len(distribution['values']) and sw>0:
            beforemoment=distribution['moment'][i-1] if i else 0.
            mean=max(0.,(distribution['moment'][-1]-beforemoment)/sw-age_ms)
            target=before+quantile*sw
            index=min(bisect_left(distribution['cumulative'],target),len(distribution['values'])-1)
            q=distribution['values'][index]-age_ms
            # This is the empirical conditional excess, not max(mean-age,5).
            tail_span=max(0.,self._quantile(distribution,.95)-self._quantile(distribution,.5))
            conservative=max(mean,q,tail_span)
            status='CONDITIONAL_EMPIRICAL_TAIL'
            # Weighted mixture includes overlapping strata: count is only a
            # mixture effective mass diagnostic, never independent sample N.
            bs=distribution['squared'][i-1] if i else 0.
            effective=sw*sw/(distribution['squared'][-1]-bs)
        else:
            # No supported survival event: provide a conspicuous tail-scale
            # fallback, flag unknown extrapolation, never a tiny expired floor.
            mean=None;q=None;conservative=max(self._quantile(distribution,.95),age_ms*.25)
            effective=0;status='TAIL_EXTRAPOLATION_UNVERIFIED'
        return dict(conditional_mean_ms=mean,quantile_ms=q,conservative_ms=conservative,
            survival_probability=survival,effective_survivor_support=effective,evidence_status=status,
            age_ms=age_ms,support=support,context=c,quantile=quantile,
            conservative_tail_safeguard='At least historical q95-minus-median tail span within support; beyond support at least historicalq95 or age/4. This is an unvalidated risk heuristic, not a residual-mean identity.',
            scope='E[S-a|S>a] empirical estimate when supported. Input age must be audited host-inferred head age; beyond support no mean or quantile is claimed.')

def cross_seed(records):
    folds=[]
    for train_seed,test_seed in ((113,114),(114,113)):
        model=ServiceModel(fit([r for r in records if r['seed']==train_seed],label=f'crossseed-train{train_seed}'))
        by=defaultdict(list)
        for r in records:
            if r['seed']!=test_seed:continue
            p=model.predict(r['family'],r['task'],r['pool_id'],r['context'])
            by[(r['family'],r['task'])].append(dict(actual=r['service_ms'],prediction=p,legacy=r.get('legacy_predicted_service_ms')))
        groups=[]
        for(f,t),rows in sorted(by.items()):
            known=[r for r in rows if r['prediction']['mean_ms'] is not None]
            groups.append(dict(family=f,task=t,test_requests=len(rows),known=len(known),
                MAE_mean_ms=statistics.mean(abs(r['actual']-r['prediction']['mean_ms']) for r in known) if known else None,
                mean_signed_error_actual_minus_prediction_ms=statistics.mean(r['actual']-r['prediction']['mean_ms'] for r in known) if known else None,
                actual_over_mean_prediction_median=statistics.median(r['actual']/r['prediction']['mean_ms'] for r in known) if known else None,
                q90_empirical_coverage=sum(r['actual']<=r['prediction']['quantile_ms'] for r in known)/len(known) if known else None,
                conservative_empirical_coverage=sum(r['actual']<=r['prediction']['conservative_ms'] for r in known)/len(known) if known else None,
                legacy_MAE_ms=statistics.mean(abs(r['actual']-r['legacy']) for r in rows if finite(r['legacy'])) if any(finite(r['legacy']) for r in rows) else None,
                missing_input_context=sum(r['prediction']['support']['input_context']==0 for r in rows)))
        folds.append(dict(train_seed=train_seed,test_seed=test_seed,groups=groups,
            interpretation='Whole-seed diagnostic holdout, correlated repeated inputs and sharedGPU context; not independent sample confidence or counterfactual validation.'))
    return folds

def coverage(records):
    groups=defaultdict(list);seeds=defaultdict(set)
    for r in records:
        k=key(r['family'],r['task'],r['pool_id'],context_key(r['context']));groups[k].append(r);seeds[k].add(r['seed'])
    cells=[dict(cell=k,requests=len(v),seed_count=len(seeds[k]),raw_count=len({r['raw_sha256'] for r in v}),
                insufficient=len(v)<16 or len(seeds[k])<2) for k,v in sorted(groups.items())]
    tasks=defaultdict(set);family_tasks=set()
    for r in records:tasks[r['task']].add(r['pool_id']);family_tasks.add((r['family'],r['task']))
    missing=[]
    for f,t in sorted(family_tasks):
        for p in sorted(tasks[t]):
            for active in ('0','1','2plus'):
                for busy in (False,True):
                    k=key(f,t,p,context_key(dict(other_active_bin=active,g2_visible_outstanding=busy)))
                    if k not in groups:missing.append(dict(family=f,task=t,pool_id=p,context=dict(other_active_bin=active,g2_visible_outstanding=busy)))
    return dict(observed_cells=cells,insufficient_cells=[r for r in cells if r['insufficient']],
        missing_input_context_cells=missing,
        unmeasured_regions=['G1/G3 all task/input/context regions absent from development14.',
            'No measurements for unchosen family/task/input routes; family/task shrinkage cannot establish counterfactual service.',
            'High concurrency/G2 busy contexts may be sparse; visible pipeline bins cannot separate actualGPU overlap.',
            'Service tail beyond observed maximum has no empirical survivors and remains unverified.',
            f"Only{len({r['raw_sha256'] for r in records})} training runs across{len({r['seed'] for r in records})} seeds in this exact deployment; repetition of32 inputs is not equivalent to independent training diversity."])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development-report',type=Path,required=True);p.add_argument('--deployment',type=json.loads,required=True,help='Exact resident deployment JSON; never pool across deployments');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    records,sources=extract_development(a.development_report,a.deployment)
    a.output.mkdir(parents=True,exist_ok=False)
    doc=fit(records,sources=sources);save(a.output/'model.json',doc)
    save(a.output/'cross-seed-validation.json',cross_seed(records));save(a.output/'coverage.json',coverage(records))
    with(a.output/'training-records.jsonl').open('x') as stream:
        for row in records:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    save(a.output/'manifest.json',dict(inputs=sources,generator=sha(Path(__file__)),model=sha(a.output/'model.json'),
        cross_seed=sha(a.output/'cross-seed-validation.json'),coverage=sha(a.output/'coverage.json'),training_records=sha(a.output/'training-records.jsonl'),
        quality_used=False,reserved_data_used=False,GPU_used=False))
    print(json.dumps(dict(output=str(a.output.resolve()),observations=len(records),training_seeds=doc['training_seeds'])))

if __name__=='__main__':main()
