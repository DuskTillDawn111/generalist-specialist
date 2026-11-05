"""Fresh paired observer checks after the complete runtime-scope diagnostic.

Copies the earlier fixed observer cohort and thresholds, but captures all data
again in the selected runtime. No runtime-comparison capture becomes training.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import digest, save
from experiments.fit_activation_native import load_bundle


def prepare(output, diagnostic, cohort_base, scope, seed):
    report=json.loads((diagnostic/'description.json').read_text())
    if report.get('output_repeat_diagnostic_passed') is not True or report.get('issues'):
        raise ValueError('Complete clean runtime contrast required before candidate observer checks')
    if scope not in ('all_strict','g2_strict') or any(s in (6110491,6110492) for s in (seed,seed+1)):
        raise ValueError('Explicit runtime and nonconfirmation seeds required')
    prefix='strict' if scope=='all_strict' else 'scoped'
    source=diagnostic/(prefix+'-round1')
    bundle=load_bundle(source)
    if report['sources'][prefix+'-round1']!=bundle['source']:
        raise ValueError('Runtime selection receipt changed')
    context=bundle['context']
    if context['runtime_policy']['scope']!=scope:
        raise ValueError('Selected runtime context differs')
    cohort=json.loads((cohort_base/'semantic-round1/plan.json').read_text())
    criteria=json.loads((cohort_base/'observer-criteria.json').read_text())
    if (cohort['split']!='development_observer_diagnostic'
            or cohort['aliases']!=context['resident_aliases']
            or criteria['independent_paired_rounds']!=2
            or criteria['maximum_mean_absolute_e2e_deviation']!=.1
            or criteria['maximum_p95_absolute_e2e_deviation']!=.25):
        raise ValueError('Original full five-model cohort and unchanged observer criteria required')
    receipts=json.loads((source/'sources.json').read_text())
    for row in receipts:
        if any(hashlib.sha256(Path(row[k]).read_bytes()).hexdigest()!=row['sha256'] for k in ('path','snapshot')):
            raise ValueError('Frozen runtime source drift: '+row['path'])
    output.mkdir(parents=True,exist_ok=False)
    paths={row['path'] for row in receipts}
    for path in (Path(__file__).resolve(),ROOT/'experiments/analyze_activation_scoped_observer.py',
                 ROOT/'experiments/activation_completion_bands.py'):
        if str(path) not in paths:
            receipts.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    frozen=[]
    for i,row in enumerate(sorted(receipts,key=lambda r:r['path'])):
        path=Path(row['path']);snapshot=output/'source-snapshot'/str(i)/path.name
        snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(path.read_bytes())
        frozen.append(dict(path=str(path),sha256=row['sha256'],snapshot=str(snapshot)))
    orders={}
    for round_no in (1,2):
        jobs=copy.deepcopy(cohort['jobs']);random.Random(seed+round_no-1).shuffle(jobs)
        orders[round_no]=jobs
    captures=[]
    for name,mode,round_no in [('control-round1','complete',1),('semantic-round1','semantic',1),
                                ('semantic-round2','semantic',2),('control-round2','complete',2)]:
        directory=output/name;directory.mkdir()
        ctx=copy.deepcopy(context)
        for binding in ctx['worker_bindings'].values():
            binding['env']['GS_SCOPED_OBSERVER']=mode
        ctx.update(observer_diagnostic='five_model_scoped_'+mode,
            source_content=[{k:r[k] for k in ('path','sha256')} for r in frozen])
        plan=dict(schema='gsdeploy.native_activity_training_plan.v1',mode='overlap',
            split='development_observer_diagnostic',not_for_training=True,
            execution_context_sha256=digest(ctx),run_id=name,seed=seed+round_no-1,
            aliases=ctx['resident_aliases'],jobs=orders[round_no],budget_seconds=1800,
            priming='largest_three_per_supported_task',runtime_scope=scope,
            observer_diagnostic=ctx['observer_diagnostic'],matched_job_content_sha256=digest(orders[round_no]),
            expected_semantic_components=cohort['expected_semantic_components'],
            confirmation_seeds_reserved=[6110491,6110492],formal_utility_qualified=False)
        save(directory/'context.json',ctx);save(directory/'sources.json',frozen);save(directory/'plan.json',plan)
        captures.append(str(directory))
    save(output/'observer-criteria.json',criteria)
    save(output/'design.json',dict(schema='gsdeploy.scoped_observer_plan.v1',scope=scope,
        source_diagnostic=str(diagnostic/'description.json'),source_diagnostic_sha256=digest(report),
        cohort_source=str(cohort_base/'semantic-round1/plan.json'),cohort_sha256=digest(cohort),
        original_criteria_sha256=digest(criteria),seeds=[seed,seed+1],source_receipts=frozen,
        requests_per_capture=sum(len(j['requests']) for j in cohort['jobs']),
        inference='paired natural-run observer overhead plus runtime variability; no physical qualification'))
    save(output/'queue.json',dict(schema='gsdeploy.native_capture_queue.v1',state=str(output/'continuation.json'),captures=captures))
    print(json.dumps(dict(output=str(output),scope=scope,captures=4)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','diagnostic','cohort-base'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scope',choices=('all_strict','g2_strict'),required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();prepare(a.output.resolve(),a.diagnostic.resolve(),a.cohort_base.resolve(),a.scope,a.seed)
