"""Full-pool ABBA host diagnostic, replaying the two original baseline orders."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.collect_activation_training import save
from gsdeploy.forecasting.native_training import validate_capture,digest


def normalized_context(context):
    value=copy.deepcopy(context)
    for key in ('source_content','observer_diagnostic','passive_gc_policy'):value.pop(key,None)
    for binding in value['worker_bindings'].values():
        binding['env'].pop('GS_PASSIVE_GC_TRACE',None)
        binding['command'][1]='MATCHED_OBSERVER_ENTRY'
    return value


def prepare(gc_base,solo_base,output):
    report=json.loads((gc_base/'description.json').read_text())
    if report['schema']!='gsdeploy.passive_gc_diagnostic.v1' or not report['observer_repeat_check_passed'] or report['issues']:
        raise ValueError('Passed matched passive-GC diagnostic required before adding host brackets')
    reference=load_bundle(gc_base/'trace-round1');validate_capture(reference,allowed_splits=('development_gc_diagnostic',))
    if reference['source']!=report['sources']['trace-round1']:raise ValueError('GC report source mismatch')
    solos={i:load_bundle(solo_base/f'solo-round{i}') for i in (1,2)}
    for bundle in solos.values():
        validate_capture(bundle)
        if (len(bundle['plan']['jobs'])!=320 or bundle['plan']['mode']!='solo'
                or normalized_context(bundle['context'])!=normalized_context(reference['context'])):
            raise ValueError('Original matched full-pool source required')
    paths={}
    for receipt_path in [gc_base/'trace-round1/sources.json',gc_base/'analysis-dependency-freeze.json']:
        receipts=json.loads(receipt_path.read_text())
        if isinstance(receipts,dict):receipts=receipts['source_receipts']
        for r in receipts:
            if any(hashlib.sha256(Path(r[k]).read_bytes()).hexdigest()!=r['sha256'] for k in ('path','snapshot')):
                raise ValueError('Upstream source drift: '+r['path'])
            paths[r['path']]=r['sha256']
    additions=[Path(__file__).resolve(),ROOT/'tools/predictor_validation_20261027/host_trace_scoped_worker.py',
        ROOT/'experiments/analyze_activation_host_diagnostic.py',ROOT/'experiments/continue_activation_host_diagnostic.py',
        ROOT/'tests/test_host_diagnostic.py',gc_base/'description.json']
    for i in (1,2):additions.extend(solo_base/f'solo-round{i}'/name for name in ('context.json','plan.json','hardware.json'))
    for path in additions:paths[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    output.mkdir(parents=True,exist_ok=False);frozen=[]
    for i,(path,sha) in enumerate(sorted(paths.items())):
        snapshot=output/'source-snapshot'/str(i)/Path(path).name;snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(Path(path).read_bytes());frozen.append(dict(path=path,sha256=sha,snapshot=str(snapshot)))
    captures=[]
    for name,mode,index in [('control-round1','off',1),('trace-round1','on',1),('trace-round2','on',2),('control-round2','off',2)]:
        directory=output/name;directory.mkdir();ctx=copy.deepcopy(reference['context'])
        for binding in ctx['worker_bindings'].values():
            binding['command'][1]=str(ROOT/'tools/predictor_validation_20261027/host_trace_scoped_worker.py')
            binding['env']['GS_PASSIVE_HOST_TRACE']=mode
        ctx.update(observer_diagnostic='five_model_passive_host_'+mode,passive_host_policy=dict(mode=mode,extra_cuda_sync=False),
            source_content=[{k:r[k] for k in ('path','sha256')} for r in frozen])
        source=solos[index];jobs=copy.deepcopy(source['plan']['jobs'])
        plan=dict(schema='gsdeploy.native_activity_training_plan.v1',mode='overlap',split='development_host_diagnostic',
            not_for_training=True,execution_context_sha256=digest(ctx),run_id=name,seed=source['plan']['seed'],
            aliases=ctx['resident_aliases'],jobs=jobs,budget_seconds=2400,priming='largest_three_per_supported_task',
            gc_trace_mode='on',host_trace_mode=mode,matched_job_content_sha256=digest(jobs),
            history_replay_source=source['source'],seed_reuse='intentional exact old development order replay, not independent qualification',
            purpose='full-pool serial host/GC diagnosis under exact earlier request histories; no runtime policy intervention',
            confirmation_seeds_reserved=[6110491,6110492],formal_utility_qualified=False)
        save(directory/'context.json',ctx);save(directory/'sources.json',frozen);save(directory/'plan.json',plan);captures.append(str(directory))
    save(output/'design.json',dict(schema='gsdeploy.passive_host_diagnostic_plan.v1',source_receipts=frozen,
        full_pool_sources={str(i):b['source'] for i,b in solos.items()},gc_report=str(gc_base/'description.json'),
        jobs_per_capture=320,requests_per_capture=320,order_policy='exact two earlier scoped baseline orders; ABBA',
        criteria=dict(maximum_mean_absolute_e2e_deviation=.1,maximum_p95_absolute_e2e_deviation=.25,
                      all_outputs_unchanged=True,complete_gc_trace=True),
        gc_policy_changes=False,physical_predictor_qualified=False,formal_utility_qualified=False))
    save(output/'queue.json',dict(schema='gsdeploy.native_capture_queue.v1',state=str(output/'continuation.json'),captures=captures))
    print(json.dumps(dict(output=str(output),captures=4,requests_per_capture=320)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('gc-base','solo-base','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();prepare(a.gc_base.resolve(),a.solo_base.resolve(),a.output.resolve())
