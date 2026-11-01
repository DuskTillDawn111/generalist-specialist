"""Training-only input-conditioned Q diagnostics; never changes live scoring.

Estimate expected native error counts by family/task/input, then sum them over
the exact public routed trace and divide by public reference units per task.
The fixed current family/task quality proxy remains unchanged.
"""
from __future__ import annotations

import argparse
import collections
import copy
import math
from pathlib import Path
import statistics
import time

from gsdeploy.quality import sample_totals
from gsdeploy.tasks import asr_normalize, page_ocr_text, score_result
from . import capacity_input_weighted_v1 as training
from .native_flow import checked, receipt, save
from .probe_validation import read
from .profiling import digest

SCHEMA='gsdeploy.input_conditioned_training_quality.v1'
POLICY=dict(stratum=['family','task','pool_id'],
    estimator='Arithmetic training mean native error numerator; deterministic reference denominator and observation count per input',
    task_loss='For each public target routed request sum E[numerator|family,task,input] / sum(public reference denominator), across all families in the task',
    Q='Original offered task-rate weighted mean of task total-error/total-reference ratios',
    missing='Any missing positive-weight routed input group makes its task and overall Q UNKNOWN; no renormalization or pooled fallback',
    pooling='Training outputs pooled across old profiles and retained continuous routes/seeds; family/task/input exchangeability is an empirical assumption, not concurrency invariance',
    reference='Frozen catalog and unchanged tasks.score_result/sample_totals normalization',
    cross_validation='Hold out the complete H90 training seed and all four routes; retain both old training collections and the other seed',
    scope='CPU training-CV diagnostic only; no parameter selection, current scorer/gates unchanged, no independent outcomes or GPU',
    repeated_inputs_are_independent=False,current_pipeline_quality_changed=False,final_gain_eligible=False)


def sources():
    import gsdeploy.quality,gsdeploy.tasks
    return [receipt(p) for p in (__file__,training.__file__,gsdeploy.quality.__file__,gsdeploy.tasks.__file__)]


def reference_denominator(task,reference):
    if reference.get('task')!=task:raise ValueError('Public reference task mismatch')
    if task in ('text','image'):return 1.
    if task=='ocr':value=len(page_ocr_text(reference['label']))
    elif task=='asr':value=len(asr_normalize(reference['label']).split())
    else:raise ValueError('Unsupported quality task')
    if value<=0:raise ValueError('Positive native reference denominator required')
    return float(value)


def score_rows(raw,catalog,aliases):
    """Native scorer on explicitly authorized training outputs only."""
    if (raw.get('kind'),raw.get('backend'),raw.get('status'))!=('measured','native','completed'):
        raise ValueError('Complete measured native training collection required')
    if any(raw.get(k) for k in ('errors','runtime_errors','telemetry_errors')):
        raise ValueError('Error-free training collection required')
    rows=raw['request_records'];seen=set();result=[]
    if not rows:raise ValueError('Nonempty training quality cohort required')
    for row in rows:
        rid=row['request_id'];task=row['task'];pool=row['pool_id']
        if row.get('split')!='training' or row.get('status')!='completed' or rid in seen:
            raise ValueError('Unique complete explicitly training quality rows required')
        seen.add(rid);family=aliases[row['model']]
        if isinstance(family,dict):family=family['family']
        response=copy.deepcopy(row['worker_response'])
        if response.get('task')!=task:raise ValueError('Training response task mismatch')
        if task=='ocr':response['prediction']=page_ocr_text(response['prediction'])
        reference=catalog['quality']['references'][task][pool]
        numerator,denominator=sample_totals(task,score_result(reference,response))
        if denominator!=reference_denominator(task,reference):raise ValueError('Native reference denominator mismatch')
        result.append(dict(request_id=rid,family=family,task=task,pool_id=pool,
            numerator=numerator,denominator=denominator))
    return result


def fit_observations(observations,*,execution_context_sha256=None,source_receipts=()):
    groups=collections.defaultdict(list);pooled=collections.defaultdict(list)
    for row in observations:
        n,d=row['numerator'],row['denominator']
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (n,d)) or n<0 or d<=0:
            raise ValueError('Finite native error numerator and positive reference units required')
        if row['task'] in ('text','image') and n>d:raise ValueError('Classification/EM error cannot exceed denominator')
        groups[(row['family'],row['task'],row['pool_id'])].append(row)
        pooled[(row['family'],row['task'])].append(row)
    if not groups:raise ValueError('Nonempty training quality observations required')
    conditional={};losses={}
    for (family,task,pool),rows in sorted(groups.items()):
        denominators={r['denominator'] for r in rows}
        if len(denominators)!=1:raise ValueError('One deterministic denominator per exact training input required')
        conditional.setdefault(family,{}).setdefault(task,{})[pool]=dict(
            expected_numerator=statistics.mean(r['numerator'] for r in rows),
            reference_denominator=next(iter(denominators)),observation_count=len(rows),
            numerator_sum=sum(r['numerator'] for r in rows),reference_denominator_sum=sum(r['denominator'] for r in rows))
    for (family,task),rows in sorted(pooled.items()):
        losses.setdefault(family,{})[task]=sum(r['numerator'] for r in rows)/sum(r['denominator'] for r in rows)
    doc=dict(schema=SCHEMA,policy=POLICY,kind='training_input_conditional_proxy',split='training',
        execution_context_sha256=execution_context_sha256,conditional=conditional,
        pooled_loss_by_family_task=losses,observation_count=len(observations),
        source_receipts=list(source_receipts),target_outcomes_used=False,GPU_started=False)
    doc['model_sha256']=digest(doc);return doc


def predict(plan,candidate,model,catalog):
    """Accepts only public trace/routes/references; no target output argument."""
    if (model.get('schema')!=SCHEMA or model.get('policy')!=POLICY or model.get('split')!='training' or
            model.get('model_sha256')!=digest({k:v for k,v in model.items() if k!='model_sha256'})):
        raise ValueError('Unchanged training-only input quality model required')
    if model.get('execution_context_sha256') not in (None,candidate.get('execution_context_sha256')):
        raise ValueError('Quality execution context mismatch')
    rates=plan['workload_rps'];total_rate=sum(rates.values())
    if not rates or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in rates.values()):
        raise ValueError('Positive original offered task rates required')
    ids=[r['request_id'] for r in plan['trace']]
    if len(set(ids))!=len(ids) or set(candidate['routes'])!=set(ids):raise ValueError('Unique public full trace/routes required')
    totals={t:dict(expected_numerator=0.,reference_denominator=0.,request_count=0,missing=[]) for t in rates}
    strata=collections.Counter()
    for row in plan['trace']:
        task,pool=row['task'],row['pool_id'];alias=candidate['routes'][row['request_id']]
        family=plan['aliases'][alias]['family'];target=totals[task]
        den=reference_denominator(task,catalog['quality']['references'][task][pool])
        value=model['conditional'].get(family,{}).get(task,{}).get(pool)
        target['reference_denominator']+=den;target['request_count']+=1
        strata[(family,task,pool)]+=1
        if value is None:
            target['missing'].append(dict(family=family,pool_id=pool,request_id=row['request_id']))
        else:
            if value['reference_denominator']!=den:raise ValueError('Target/training reference denominator changed')
            target['expected_numerator']+=value['expected_numerator']
    for target in totals.values():
        known=target['request_count']>0 and not target['missing']
        target['status']='ESTIMATED' if known else 'UNKNOWN'
        target['loss']=target['expected_numerator']/target['reference_denominator'] if known else None
        if not known:target['expected_numerator']=None
    known=all(t['loss'] is not None for t in totals.values())
    return dict(schema='gsdeploy.public_trace_input_quality_prediction.v1',status='ESTIMATED' if known else 'UNKNOWN',
        Q=sum(rates[t]/total_rate*v['loss'] for t,v in totals.items()) if known else None,
        per_task=totals,public_target_strata=[dict(family=f,task=t,pool_id=p,count=n) for (f,t,p),n in sorted(strata.items())],
        target_trace_sha256=digest(plan['trace']),route_sha256=digest(candidate['routes']),target_outcomes_used=False)


def aggregate_prediction(plan,candidate,losses):
    """Reproduce current forecast metric's mean routed family/task proxy loss."""
    values=collections.defaultdict(list)
    for row in plan['trace']:
        family=plan['aliases'][candidate['routes'][row['request_id']]]['family']
        values[row['task']].append(losses.get(family,{}).get(row['task']))
    tasks={t:statistics.mean(values[t]) if values[t] and all(v is not None for v in values[t]) else None for t in plan['workload_rps']}
    total=sum(plan['workload_rps'].values())
    return dict(Q=sum(plan['workload_rps'][t]/total*q for t,q in tasks.items()) if all(v is not None for v in tasks.values()) else None,per_task=tasks)


def observed_quality(plan,observations):
    totals={t:dict(numerator=0.,denominator=0.,request_count=0) for t in plan['workload_rps']}
    for row in observations:
        t=totals[row['task']];t['numerator']+=row['numerator'];t['denominator']+=row['denominator'];t['request_count']+=1
    for value in totals.values():value['loss']=value['numerator']/value['denominator'] if value['denominator'] else None
    rate=sum(plan['workload_rps'].values())
    return dict(Q=sum(plan['workload_rps'][t]/rate*v['loss'] for t,v in totals.items()) if all(v['loss'] is not None for v in totals.values()) else None,per_task=totals)


def load_old_quality(path):
    doc=read(path)
    if (doc.get('schema')!='gsdeploy.training_family_task_quality.v1' or doc.get('split')!='training' or
            doc.get('heldout_observations_used') is not False or doc.get('training_documents')):
        raise ValueError('Single native old training quality document required, not merged or validation proxy')
    for ref in doc['source_receipts']:checked(ref)
    raw_ref,catalog_ref=doc['source_receipts'][:2];raw=read(raw_ref['path']);catalog=read(catalog_ref['path'])
    checked(raw['context']);context=read(raw['context']['path'])
    rows=score_rows(raw,catalog,context['residents'])
    actual={r['request_id']:{k:v for k,v in r.items() if k!='request_id'} for r in rows}
    if actual!=doc['totals_by_request'] or len(rows)!=doc['request_count']:
        raise ValueError('Old quality document differs from recomputed native training outputs')
    return dict(rows=rows,raw=raw_ref,catalog=catalog_ref,execution_context_sha256=doc['execution_context_sha256'],
        sources=[receipt(path),*doc['source_receipts']])


def crossvalidate(suite_path,old_quality_paths,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if len(old_quality_paths)!=2:raise ValueError('Exactly the two original v7/v8 training collections required')
    suite=read(suite_path)
    if suite.get('seeds')!=[627401,627402]:raise ValueError('This diagnostic version requires the original H90 v10 training seeds')
    bundle=training.load_training(suite_path);old=[load_old_quality(p) for p in old_quality_paths]
    if len({d['raw']['sha256'] for d in old})!=2:raise ValueError('Distinct old native raw sources required')
    runs=[];identities={d['execution_context_sha256'] for d in old}
    for run in bundle['runs']:
        plan=run['plan'];raw=read(run['raw_source']['path']);catalog=read(checked(plan['catalog']))
        if plan['window_seconds']!=90.:raise ValueError('H90 primary training only for this version')
        # training_evidence already verified the pre-startup frozen role and
        # full cohort. Materialize only that verified role, never relabel validation.
        derived=copy.deepcopy(raw)
        derived['request_records']=[{**r,'split':'training'} for r in raw['request_records']]
        rows=score_rows(derived,catalog,plan['aliases']);identities.add(raw['execution_context_sha256'])
        runs.append(dict(seed=run['seed'],flow_id=run['flow_id'],candidate_id=run['candidate_id'],plan=plan,
            raw=run['raw_source'],rows=rows,catalog=catalog,source_receipts=[run['raw_source'],run['plan_source'],plan['catalog']]))
    if len(identities)!=1:raise ValueError('Single training execution context required')
    context=next(iter(identities));old_rows=[r for d in old for r in d['rows']]
    old_refs=[s for d in old for s in d['sources']]
    fixed=fit_observations(old_rows,execution_context_sha256=context,source_receipts=old_refs)
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(schema='gsdeploy.input_quality_training_cv_protocol.v1',created_ns=time.perf_counter_ns(),
        policy=POLICY,suite=receipt(suite_path),old_training_quality=[receipt(p) for p in old_quality_paths],
        source_receipts=sources(),independent_outcomes_read=False,GPU_started=False)
    save(output/'protocol.json',protocol);save(output/'fixed-old-input-model.json',fixed)
    cases=[];folds=[]
    for seed in bundle['seeds']:
        retained=[r for r in runs if r['seed']!=seed]
        model=fit_observations(old_rows+[row for r in retained for row in r['rows']],execution_context_sha256=context,
            source_receipts=[*sources(),*old_refs,*[s for r in retained for s in r['source_receipts']]])
        model['training_seed_exclusion']=dict(heldout_seed=seed,retained_flow_ids=[r['flow_id'] for r in retained],
            retained_raws=[r['raw'] for r in retained],old_training_raws=[d['raw'] for d in old])
        model.pop('model_sha256');model['model_sha256']=digest(model)
        path=output/f'heldout-seed-{seed}-model.json';save(path,model)
        folds.append(dict(heldout_seed=seed,model=receipt(path)))
        for run in runs:
            if run['seed']!=seed:continue
            plan=run['plan'];candidate=next(c for c in plan['candidates'] if c['id']==run['candidate_id'])
            prediction=predict(plan,candidate,model,run['catalog'])
            fixed_prediction=aggregate_prediction(plan,candidate,fixed['pooled_loss_by_family_task'])
            pooled_prediction=aggregate_prediction(plan,candidate,model['pooled_loss_by_family_task'])
            observed=observed_quality(plan,run['rows'])
            methods=dict(current_fixed_old_proxy=fixed_prediction,fold_pooled_proxy=pooled_prediction,input_conditioned=prediction)
            cases.append(dict(seed=seed,candidate_id=candidate['id'],heldout_raw=run['raw'],model=receipt(path),
                predictions=methods,observed=observed,
                Q_absolute_errors={k:abs(v['Q']-observed['Q']) if v['Q'] is not None and observed['Q'] is not None else None for k,v in methods.items()}))
    full=fit_observations(old_rows+[row for r in runs for row in r['rows']],execution_context_sha256=context,
        source_receipts=[*sources(),*old_refs,*[s for r in runs for s in r['source_receipts']]])
    save(output/'all-training-model.json',full)
    methods=('current_fixed_old_proxy','fold_pooled_proxy','input_conditioned')
    summary={k:dict(known=sum(c['Q_absolute_errors'][k] is not None for c in cases),total=len(cases),
        mean_Q_absolute_error=statistics.mean(c['Q_absolute_errors'][k] for c in cases if c['Q_absolute_errors'][k] is not None) if any(c['Q_absolute_errors'][k] is not None for c in cases) else None) for k in methods}
    report=dict(schema='gsdeploy.input_quality_training_cv_report.v1',policy=POLICY,protocol=receipt(output/'protocol.json'),
        cases=cases,summary=summary,folds=folds,full_model=receipt(output/'all-training-model.json'),
        source_receipts=[*sources(),*bundle['source_receipts'],*old_refs],
        independent_validation_sources_used=[],training_cv_only=True,current_pipeline_quality_changed=False,
        parameter_selection_performed=False,GPU_started=False,final_gain_eligible=False)
    save(output/'report.json',report)
    lines=['Input-conditioned quality training-CV diagnostic','',
        'The current pipeline remains on its fixed old quality proxy. These results change no scorer, gate, selected model or independent protocol.',
        '', 'Current training_quality correctly pools native errors/reference units within family/task. The forecast metric then averages these family/task losses over routed requests; score_flow instead pools native errors/reference units across families within each task. Public input mix and unequal ASR/OCR reference lengths can therefore change predicted Q.', '',
        'Whole H90 seed held out across all four routes; both old training collections retained. Missing positive-weight input stays UNKNOWN. Repeated requests are observations, not independent quality examples.','',
        '| Method | Known/total Q | Mean absolute Q error |','|---|---:|---:|']
    for key,value in summary.items():lines.append(f"| {key} | {value['known']}/{value['total']} | {value['mean_Q_absolute_error']} |")
    lines+=['','Per-task errors, target reference totals, missing groups, and source receipts are retained in report.json. The new estimator is diagnostic only; prospective adoption requires a new declared protocol.']
    (output/'results.md').write_text('\n'.join(lines)+'\n')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',required=True)
    p.add_argument('--old-quality',action='append',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(argv);crossvalidate(args.suite,args.old_quality,args.output);return 0


if __name__=='__main__':raise SystemExit(main())
