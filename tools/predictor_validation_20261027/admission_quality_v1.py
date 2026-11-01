"""Prospective admission2 quality proxy, strictly isolated by controller/route.

CPU preparation reads no outcomes. Fitting audits all eight original training
runs and independently recomputes their immutable scored annotations. No shared
quality model, validation wrapper, input-conditioned model selection or GPU.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
from pathlib import Path
import time

from . import admission_capacity_v1 as capacity
from . import quality_input_conditioned_v1 as arithmetic
from . import training_run_audit as audit
from .native_flow import checked, receipt, save
from .probe_validation import read
from .profiling import digest

SCHEMA='gsdeploy.admission2_training_quality.v1'
POLICY=dict(training_seeds=list(capacity.SEEDS),candidate_ids=list(capacity.ROUTES),
    stratum=['controller_sha256','candidate_id','family','task'],
    estimator='Original family/task training proxy: sum native error numerators / sum reference denominators',
    prediction='Mean routed family/task proxy within each public task, original offered task-rate weighted Q',
    cv='Omit the entire heldout seed across all four routes; same-route remaining seed only',
    pooling='Same exact route/controller/condition only; never shared, pilot, validation or another route',
    missing='Missing positive-weight family/task support makes task and Q UNKNOWN; no fallback or renormalization',
    scorer='Unchanged native score_result and sample_totals; never mean request CER/WER',
    scope='Empirical training output-distribution proxy; repeated inputs are not independent quality examples',
    model_selection=False,independent_validation_sources_used=[],final_gain_eligible=False)


def sources():
    import gsdeploy.quality,gsdeploy.tasks
    return [receipt(p) for p in (__file__,arithmetic.__file__,audit.__file__,
        gsdeploy.quality.__file__,gsdeploy.tasks.__file__)]+capacity.sources()


def prepare(campaign_path,output):
    if Path(output).exists():raise FileExistsError(output)
    campaign=capacity.training.verify(campaign_path)
    value=dict(schema=SCHEMA+'.protocol',created_ns=time.perf_counter_ns(),policy=POLICY,
        campaign=receipt(campaign_path),controller_binding=campaign['controller_binding'],
        source_receipts=capacity.unique_refs([*sources(),*campaign['source_receipts']]),
        training_outcomes_read=False,independent_outcomes_read=False,GPU_started=False)
    value['protocol_sha256']=digest(value);save(output,value);return value


def verify_protocol(path):
    value=read(path)
    if (value.get('schema')!=SCHEMA+'.protocol' or value.get('policy')!=POLICY
            or type(value.get('created_ns')) is not int
            or value.get('training_outcomes_read') is not False
            or value.get('independent_outcomes_read') is not False
            or value.get('GPU_started') is not False
            or value.get('protocol_sha256')!=digest({k:v for k,v in value.items() if k!='protocol_sha256'})):
        raise ValueError('Changed prospective admission quality protocol')
    capacity.unique_refs([value['campaign'],*value['source_receipts']])
    campaign=capacity.training.verify(value['campaign']['path'])
    required=[*sources(),*campaign['source_receipts']]
    if (value['controller_binding']!=campaign['controller_binding'] or
            not {(r['path'],r['sha256']) for r in required}<=
                {(r['path'],r['sha256']) for r in value['source_receipts']}):
        raise ValueError('Quality protocol omitted or changed controller/sources')
    return value


def load_runs(protocol_path):
    protocol=verify_protocol(protocol_path)
    # This loader calls training.map_training separately for all eight native
    # sources and verifies exact roles, routes, source receipts and controller.
    bundle=capacity.load_campaign(protocol['campaign']['path'])
    for run in bundle['runs']:
        if not protocol['created_ns']<run['startup_begin_ns']:
            raise ValueError('Quality protocol must precede every training startup')
        native=read(checked(run['raw_source']))
        scored_path=Path(run['raw_source']['path']).with_name('scored-raw.json')
        scored_ref=receipt(scored_path);scored=read(scored_path)
        if (scored.get('native_raw')!=run['raw_source'] or
                {k:v for k,v in scored.items() if k not in ('native_raw','quality_evidence')}!=native or
                set(scored)!=set(native)|{'native_raw','quality_evidence'}):
            raise ValueError('Scored annotation must bind and preserve the original native training raw')
        if any(r.get('split') not in (None,'training') for r in native['request_records']):
            raise ValueError('Validation outputs cannot become training quality')
        plan=run['plan'];catalog=read(checked(plan['catalog']))
        evidence=scored['quality_evidence']
        expected=[plan['catalog'],receipt(capacity.training.native_flow.__file__),
            receipt(audit.score_result.__code__.co_filename),receipt(audit.sample_totals.__code__.co_filename)]
        capacity.unique_refs(evidence['source_receipts'])
        if evidence['source_receipts']!=expected:
            raise ValueError('Exact frozen native quality scorer/catalog sources required')
        verification=audit.verify_quality_annotation(scored,plan,catalog)
        rows=[]
        for record in native['request_records']:
            rid=record['request_id'];totals=evidence['totals_by_request'][rid]
            if record['worker_response'].get('task')!=record['task']:
                raise ValueError('Training native response task mismatch')
            rows.append(dict(request_id=rid,family=plan['aliases'][record['model']]['family'],
                task=record['task'],pool_id=record['pool_id'],numerator=totals['numerator'],
                denominator=totals['denominator'],split='training',
                controller_sha256=run['controller_sha256'],candidate_id=run['candidate_id'],seed=run['seed']))
        run.update(quality_rows=rows,scored_source=scored_ref,quality_audit=verification)
        capacity.unique_refs([run['raw_source'],scored_ref,plan['catalog'],*evidence['source_receipts']])
    refs=capacity.unique_refs([receipt(protocol_path),*protocol['source_receipts'],*bundle['source_receipts'],
        *[r['scored_source'] for r in bundle['runs']]])
    return dict(protocol=protocol,runs=bundle['runs'],source_receipts=refs)


def merge_route(runs,cid,protocol_ref,omitted_seed=None):
    selected=[r for r in runs if r['candidate_id']==cid]
    if not selected or any(r['seed']==omitted_seed for r in selected):
        raise ValueError('Same-route retained training quality required; heldout seed prohibited')
    first=selected[0]
    if any((r['condition'],r['generation'],r['controller_sha256'])!=
            (first['condition'],first['generation'],first['controller_sha256']) for r in selected):
        raise ValueError('Cannot pool quality across controller/route conditions')
    totals={};unique=collections.defaultdict(set);count=0
    for run in selected:
        for row in run['quality_rows']:
            if (row.get('split')!='training' or row.get('controller_sha256')!=first['controller_sha256']
                    or row.get('candidate_id')!=cid or row.get('seed')!=run['seed']):
                raise ValueError('Only exact training route/controller quality rows allowed')
            n,d=row['numerator'],row['denominator']
            if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (n,d))
                    or n<0 or d<=0 or (row['task'] in ('text','image') and n>d)):
                raise ValueError('Finite native quality error/reference totals required')
            f,t=row['family'],row['task'];target=totals.setdefault(f,{}).setdefault(t,
                dict(numerator=0.,denominator=0.,request_count=0))
            target['numerator']+=n;target['denominator']+=d;target['request_count']+=1;count+=1
            unique[(f,t)].add(row['pool_id'])
    losses={}
    for f,tasks in totals.items():
        for t,value in tasks.items():
            value['unique_input_count']=len(unique[(f,t)])
            losses.setdefault(f,{})[t]=value['numerator']/value['denominator']
    document=dict(schema=SCHEMA+'.model',policy=POLICY,kind='training_aggregate_proxy',split='training',
        controller_binding=first['plan']['controller_binding'],candidate_id=cid,
        execution_context_sha256=first['plan']['execution_context_sha256'],
        condition=first['condition'],generation=first['generation'],catalog=first['plan']['catalog'],
        omitted_seed=omitted_seed,training_seeds=sorted(r['seed'] for r in selected),
        loss_by_family_task=losses,totals_by_family_task=totals,request_count=count,
        protocol=protocol_ref,runs=[dict(seed=r['seed'],raw=r['raw_source'],scored_raw=r['scored_source'],
            plan=r['plan_source'],quality_audit=r['quality_audit']) for r in selected],
        source_receipts=capacity.unique_refs([protocol_ref,*sources(),*[ref for r in selected for ref in
            (r['raw_source'],r['scored_source'],r['plan_source'],r['plan']['catalog'])]]),
        target_outcomes_used=False,independent_validation_sources_used=[],GPU_started=False,final_gain_eligible=False)
    document['model_sha256']=digest(document);return document


def predict(document,plan,candidate):
    identity=capacity.controller_identity(plan,candidate)
    if (document.get('schema')!=SCHEMA+'.model' or document.get('policy')!=POLICY
            or document.get('split')!='training' or document.get('target_outcomes_used') is not False
            or document.get('candidate_id')!=candidate['id']
            or document.get('controller_binding',{}).get('controller_sha256')!=identity
            or document.get('execution_context_sha256')!=candidate.get('execution_context_sha256')
            or document.get('catalog')!=plan['catalog']
            or document.get('condition')!=capacity.condition(plan,candidate)
            or document.get('generation')!=capacity._generation(plan,document['condition'])
            or document.get('model_sha256')!=digest({k:v for k,v in document.items() if k!='model_sha256'})):
        raise ValueError('Quality model requires exact admission route/controller/public law')
    capacity.unique_refs(document['source_receipts']);checked(plan['catalog'])
    rates=plan['workload_rps'];ids=[r['request_id'] for r in plan['trace']]
    if (not rates or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in rates.values())
            or len(set(ids))!=len(ids) or set(candidate['routes'])!=set(ids)):
        raise ValueError('Positive public rates and exact full routed trace required')
    # Original fixed family/task aggregate proxy, not input-conditioned fitting.
    result=arithmetic.aggregate_prediction(plan,candidate,document['loss_by_family_task'])
    return dict(schema=SCHEMA+'.prediction',status='ESTIMATED' if result['Q'] is not None else 'UNKNOWN',
        **result,candidate_id=candidate['id'],controller_sha256=identity,
        public_trace_sha256=digest(plan['trace']),route_sha256=digest(candidate['routes']),
        missing_support=[dict(family=f,task=t) for f,t in sorted({
            (plan['aliases'][candidate['routes'][r['request_id']]]['family'],r['task']) for r in plan['trace']})
            if document['loss_by_family_task'].get(f,{}).get(t) is None],
        model_sha256=document['model_sha256'],target_outcomes_used=False)


def training_cv(entries,runs):
    rows=[]
    for entry in entries:
        for run in runs:
            if run['seed']!=entry['omitted_seed']:continue
            cid=run['candidate_id'];model=read(checked(entry['route_quality'][cid]))
            candidate=next(c for c in run['plan']['candidates'] if c['id']==cid)
            prediction=predict(model,run['plan'],candidate)
            observed=arithmetic.observed_quality(run['plan'],run['quality_rows'])
            rows.append(dict(seed=run['seed'],candidate_id=cid,heldout_raw=run['raw_source'],
                training_seeds=model['training_seeds'],model=entry['route_quality'][cid],
                predicted=prediction,observed=observed,Q_absolute_error=
                abs(prediction['Q']-observed['Q']) if prediction['Q'] is not None and observed['Q'] is not None else None))
    return dict(schema=SCHEMA+'.training_cv',policy=POLICY,rows=rows,training_only=True,
        independent_validation_sources_used=[],model_selection=False,final_gain_eligible=False,GPU_started=False)


def fit(protocol_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    bundle=load_runs(protocol_path);runs=bundle['runs'];ref=receipt(protocol_path)
    documents={(omitted,cid):merge_route([r for r in runs if r['seed']!=omitted],cid,ref,omitted)
        for omitted in (None,*capacity.SEEDS) for cid in capacity.ROUTES}
    capacity.unique_refs(bundle['source_receipts']);output.mkdir(parents=True,exist_ok=False)
    entries=[]
    for omitted in (None,*capacity.SEEDS):
        refs={}
        for cid in capacity.ROUTES:
            path=output/((f'heldout-{omitted}' if omitted else 'full')+'-'+cid+'-quality.json')
            save(path,documents[(omitted,cid)]);refs[cid]=receipt(path)
        entries.append(dict(omitted_seed=omitted,training_seeds=[s for s in capacity.SEEDS if s!=omitted],route_quality=refs))
    save(output/'quality-training-cv.json',training_cv(entries[1:],runs))
    value=dict(schema=SCHEMA+'.manifest',created_ns=time.perf_counter_ns(),policy=POLICY,
        protocol=ref,full=entries[0],folds=entries[1:],quality_cv=receipt(output/'quality-training-cv.json'),
        source_receipts=bundle['source_receipts'],independent_validation_sources_used=[],
        no_validation_data=True,final_gain_eligible=False,GPU_started=False)
    value['manifest_sha256']=digest(value);save(output/'manifest.json',value);return value


def verify_manifest(path,*,recompute=True):
    value=read(path)
    if (value.get('schema')!=SCHEMA+'.manifest' or value.get('policy')!=POLICY
            or value.get('independent_validation_sources_used')!=[] or value.get('no_validation_data') is not True
            or value.get('final_gain_eligible') is not False or value.get('GPU_started') is not False
            or value.get('manifest_sha256')!=digest({k:v for k,v in value.items() if k!='manifest_sha256'})):
        raise ValueError('Changed admission quality manifest')
    verify_protocol(checked(value['protocol']));capacity.unique_refs(value['source_receipts'])
    entries=[value['full'],*value['folds']]
    if [e['omitted_seed'] for e in entries]!=[None,*capacity.SEEDS]:raise ValueError('Full and both whole-seed folds required')
    runs=load_runs(value['protocol']['path'])['runs'] if recompute else None
    for entry in entries:
        omitted=entry['omitted_seed'];seeds=[s for s in capacity.SEEDS if s!=omitted]
        if entry['training_seeds']!=seeds or set(entry['route_quality'])!=set(capacity.ROUTES):
            raise ValueError('Exact retained seed and quality route domain required')
        for cid,ref in entry['route_quality'].items():
            doc=read(checked(ref));capacity.unique_refs(doc['source_receipts'])
            if (doc.get('model_sha256')!=digest({k:v for k,v in doc.items() if k!='model_sha256'})
                    or doc.get('protocol')!=value['protocol'] or doc.get('policy')!=POLICY
                    or doc.get('candidate_id')!=cid or doc.get('omitted_seed')!=omitted
                    or doc.get('training_seeds')!=seeds or doc.get('independent_validation_sources_used')!=[]):
                raise ValueError('Quality route/controller/seed evidence changed')
            if runs is not None and doc!=merge_route([r for r in runs if r['seed']!=omitted],cid,value['protocol'],omitted):
                raise ValueError('Quality differs from recomputed native training outputs')
    cv=read(checked(value['quality_cv']))
    if runs is not None and cv!=training_cv(value['folds'],runs):raise ValueError('Quality CV differs from recomputed training')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True)
    a=sub.add_parser('prepare');a.add_argument('--campaign',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('fit');a.add_argument('--protocol',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('verify');a.add_argument('--manifest',required=True)
    args=p.parse_args()
    if args.operation=='prepare':value=prepare(args.campaign,args.output)
    elif args.operation=='fit':value=fit(args.protocol,args.output)
    else:value=verify_manifest(args.manifest)
    print(json.dumps(dict(schema=value['schema'],GPU_started=False)))


if __name__=='__main__':main()
