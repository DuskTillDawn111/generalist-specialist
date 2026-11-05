"""Matched passive-GC diagnostic after observed full-pool baseline failures."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import POOL,digest,save
from experiments.fit_activation_native import load_bundle


def prepare(source,output,seed):
    if any(s in (6110491,6110492) for s in (seed,seed+1)):raise ValueError('Reserved confirmation seeds')
    bundle=load_bundle(source);context=bundle['context']
    if context.get('observer_diagnostic')!='five_model_scoped_semantic' or context['runtime_policy']['scope']!='g2_strict':
        raise ValueError('Exact completed scoped semantic context required')
    if bundle['raw']['status']!='completed':raise ValueError('Completed source capture required')
    content=POOL.read_bytes()
    if hashlib.sha256(content).hexdigest()!=context['input_pool_sha256']:raise ValueError('Frozen input pool changed')
    by_task={}
    for row in map(json.loads,content.decode().splitlines()):by_task.setdefault(row['task_type'],[]).append(row)
    jobs=[]
    def add(alias,task,pool,ident):
        jobs.append(dict(id=ident,requests=[dict(alias=alias,task=task,pool_id=pool,offset_ms=0.)]))
    for alias,task in [('generalist_2#0','image'),('asr#0','asr')]:
        for row in by_task[task]:add(alias,task,row['pool_id'],f'full:{alias}:{row["pool_id"]}')
    targets={'generalist_2#0':('image',[
        'imagenetv2-51d0b62725be6615ce76870359add350c4557916','imagenetv2-9d2ce8dbda34bdce46c2d2ef9ca79317a6e4fad5']),
        'asr#0':('asr',['librispeech-clean-251-118436-0011','librispeech-clean-6319-64726-0003',
                       'librispeech-clean-8224-274381-0008','librispeech-clean-908-157963-0028'])}
    for alias,(task,pools) in targets.items():
        if not set(pools)<={r['pool_id'] for r in by_task[task]}:raise ValueError('Diagnostic target missing')
        for pool in pools:
            for i in range(8):add(alias,task,pool,f'repeat:{alias}:{pool}:{i}')
    for alias,family in sorted(context['resident_aliases'].items()):
        for task in by_task if family=='generalist_2' else [family]:
            if (alias,task) in [('generalist_2#0','image'),('asr#0','asr')]:continue
            values=by_task[task]
            for i in (0,len(values)//4,len(values)//2,3*len(values)//4,len(values)-1):
                add(alias,task,values[i]['pool_id'],f'anchor:{alias}:{task}:{i}')
    receipts=json.loads((source/'sources.json').read_text())
    for r in receipts:
        if any(hashlib.sha256(Path(r[k]).read_bytes()).hexdigest()!=r['sha256'] for k in ('path','snapshot')):
            raise ValueError('Frozen scoped runtime source drift')
    output.mkdir(parents=True,exist_ok=False)
    worker=ROOT/'tools/predictor_validation_20261027/gc_trace_scoped_worker.py'
    paths={r['path']:r['sha256'] for r in receipts}
    for path in [worker,Path(__file__).resolve(),ROOT/'experiments/analyze_activation_gc_diagnostic.py',
                 ROOT/'experiments/continue_activation_gc_diagnostic.py',ROOT/'experiments/run_activation_queue.py',
                 ROOT/'tests/test_passive_gc_trace.py',ROOT/'tests/test_gc_diagnostic_analysis.py']:
        paths[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    frozen=[]
    for i,(path,sha) in enumerate(sorted(paths.items())):
        snapshot=output/'source-snapshot'/str(i)/Path(path).name;snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(Path(path).read_bytes());frozen.append(dict(path=path,sha256=sha,snapshot=str(snapshot)))
    orders={}
    for i in (1,2):
        order=copy.deepcopy(jobs);random.Random(seed+i-1).shuffle(order);orders[i]=order
    captures=[]
    for name,mode,round_no in [('control-round1','off',1),('trace-round1','on',1),('trace-round2','on',2),('control-round2','off',2)]:
        directory=output/name;directory.mkdir();ctx=copy.deepcopy(context)
        for binding in ctx['worker_bindings'].values():
            binding['command'][1]=str(worker);binding['env']['GS_PASSIVE_GC_TRACE']=mode
        ctx.update(observer_diagnostic='five_model_passive_gc_'+mode,
            passive_gc_policy=dict(mode=mode,forced_collection=False,threshold_changes=False,gc_disabled=False),
            source_content=[{k:r[k] for k in ('path','sha256')} for r in frozen])
        plan=dict(schema='gsdeploy.native_activity_training_plan.v1',mode='overlap',
            split='development_gc_diagnostic',not_for_training=True,execution_context_sha256=digest(ctx),
            run_id=name,seed=seed+round_no-1,aliases=ctx['resident_aliases'],jobs=orders[round_no],budget_seconds=1800,
            priming='largest_three_per_supported_task',gc_trace_mode=mode,
            matched_job_content_sha256=digest(orders[round_no]),
            purpose='serial mixed input repetition; passive GC/host-gap diagnosis; no claimed GPU kernel-only timing',
            confirmation_seeds_reserved=[6110491,6110492],formal_utility_qualified=False)
        save(directory/'context.json',ctx);save(directory/'sources.json',frozen);save(directory/'plan.json',plan)
        captures.append(str(directory))
    save(output/'design.json',dict(schema='gsdeploy.passive_gc_diagnostic_plan.v1',source=bundle['source'],
        target_selection='four ASR and two G2 image profiles failing full-pool constant-band feasibility; development only',
        jobs_per_capture=len(jobs),requests_per_capture=len(jobs),seeds=[seed,seed+1],source_receipts=frozen,
        criteria=dict(maximum_mean_absolute_e2e_deviation=.1,maximum_p95_absolute_e2e_deviation=.25,
            all_outputs_unchanged=True,complete_gc_trace=True),
        gc_policy_changes=False,physical_predictor_qualified=False,formal_utility_qualified=False))
    save(output/'queue.json',dict(schema='gsdeploy.native_capture_queue.v1',state=str(output/'continuation.json'),captures=captures))
    print(json.dumps(dict(output=str(output),captures=4,requests_per_capture=len(jobs))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seed',required=True,type=int);a=p.parse_args();prepare(a.source.resolve(),a.output.resolve(),a.seed)
