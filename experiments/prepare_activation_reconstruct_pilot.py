"""Five-model observer pilot for an explicitly changed ExLlama reduction path."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import POOL, digest, save
from experiments.prepare_activation_resident import prepare_resident


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--kernel-evidence',required=True,type=Path);parser.add_argument('--cohort-source',required=True,type=Path)
    parser.add_argument('--seed',required=True,type=int);parser.add_argument('--semantic',action='store_true')
    args=parser.parse_args();output=args.output.resolve()
    prepare_resident(output,args.reference.resolve(),dict(generalist_2=1,text=1,image=1,ocr=1,asr=1),mode='overlap',seed=args.seed)
    context=json.loads((output/'context.json').read_text());plan=json.loads((output/'plan.json').read_text())
    sources=json.loads((output/'sources.json').read_text());existing={row['path'] for row in sources}
    worker=ROOT/'tools/predictor_validation_20261027/reconstruct_observed_worker.py'
    for binding in context['worker_bindings'].values():
        binding['command'][1]=str(worker)
        binding['env'].update(CUBLAS_WORKSPACE_CONFIG=':4096:8',GS_RECONSTRUCT_OBSERVER='semantic' if args.semantic else 'complete')
    evidence=json.loads(args.kernel_evidence.read_text())
    extra=[worker,Path(__file__).resolve(),args.kernel_evidence.resolve(),
        ROOT/'tools/predictor_validation_20261027/deterministic_observed_worker.py',
        ROOT/'tools/predictor_validation_20261027/semantic_observed_worker.py',
        ROOT/'tools/predictor_validation_20261027/base_forward_observed_worker.py']
    for row in evidence['sources']:
        path=Path(row['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Pinned installed kernel or explanatory source changed')
        extra.append(path)
    for path in extra:
        if str(path) in existing:continue
        content=path.read_bytes();snapshot=output/'reconstruct-source-snapshot'/str(len(sources))/path.name
        snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(content)
        sources.append(dict(path=str(path),snapshot=str(snapshot),sha256=hashlib.sha256(content).hexdigest()));existing.add(str(path))
    sources.sort(key=lambda row:row['path'])
    context.update(source_content=[{key:row[key] for key in ('path','sha256')} for row in sources],
        observer_diagnostic='five_model_reconstruct_semantic' if args.semantic else 'five_model_reconstruct_complete',
        deterministic_runtime=dict(strict_algorithms=True,warn_only=False,CUBLAS_WORKSPACE_CONFIG=':4096:8',
            cudnn_benchmark=False,cudnn_deterministic=True,tf32_policy='unchanged original profiles'),
        exllama_runtime=dict(matmul_recons_thd=1,matmul_fused_remap=False,matmul_no_half2=False,
            configured_after_model_load=True,weights_and_generation_policy_unchanged=True,
            old_runtime_equivalence_not_assumed=True,kernel_evidence_sha256=hashlib.sha256(args.kernel_evidence.read_bytes()).hexdigest()))
    old=json.loads(args.cohort_source.read_text())
    if old['split']!='development_observer_diagnostic':raise ValueError('Original nontraining G2 observer cohort required')
    jobs=copy.deepcopy(old['jobs'])
    by_task={}
    for line in POOL.read_text().splitlines():
        item=json.loads(line);by_task.setdefault(item['task_type'],[]).append(item)
    def request(task,index,offset=0.):
        return dict(alias=task+'#0',task=task,pool_id=by_task[task][index%len(by_task[task])]['pool_id'],offset_ms=offset)
    # Preserve every original G2 diagnostic input and add actual specialist
    # pressure, so the tested roster includes the eventual continuous domain.
    for job in jobs:
        if job['id'].startswith(('corun:','sustained:')):
            for task in ('text','asr'):
                job['requests'].append(request(task,7,125.))
    for task in ('text','asr'):
        for index in (0,15):jobs.append(dict(id=f'solo:{task}:{index}',requests=[request(task,index)]))
    random.Random(args.seed).shuffle(jobs)
    expected={family+'/'+task:('language' if family=='generalist_2' else
        'image_features' if family=='image' else 'ocr_detection' if family=='ocr' else 'model_forward')
        for family in ('generalist_2','text','image','ocr','asr') for task in (by_task if family=='generalist_2' else (family,))}
    plan.update(jobs=jobs,split='development_observer_diagnostic',not_for_training=True,
        execution_context_sha256=digest(context),observer_diagnostic=context['observer_diagnostic'],
        matched_job_content_sha256=digest(jobs),expected_semantic_components=expected,
        exllama_runtime=context['exllama_runtime'],changed_runtime_requires_new_templates_and_episode_references=True,
        old_g2_determinism_failure_not_overridden=True,performance_qualified=False)
    save(output/'context.json',context);save(output/'sources.json',sources);save(output/'plan.json',plan)
    print(json.dumps(dict(output=str(output),context=plan['execution_context_sha256'],jobs=len(jobs),
        requests=sum(len(job['requests']) for job in jobs),not_for_training=True)))


if __name__=='__main__':main()
