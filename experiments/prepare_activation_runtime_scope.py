"""Predeclare a balanced runtime-scope diagnostic; never train from this pilot."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import POOL, digest, save


def prepare(base, source, seed, analyzer):
    base.mkdir(parents=True, exist_ok=False)
    original = json.loads((source/'context.json').read_text())
    if original.get('observer_diagnostic') != 'five_model_reconstruct_semantic':
        raise ValueError('Completed full semantic five-model context required')
    if json.loads((source/'raw.json').read_text())['status'] != 'completed':
        raise ValueError('Source capture must be complete')
    if seed in (6110491, 6110492) or seed+1 in (6110491, 6110492):
        raise ValueError('Reserved confirmation seeds')
    inputs = [json.loads(line) for line in POOL.read_text().splitlines()]
    if hashlib.sha256(POOL.read_bytes()).hexdigest() != original['input_pool_sha256']:
        raise ValueError('Input identity changed')
    by_task = {t: [r for r in inputs if r['task_type'] == t] for t in ('text','image','ocr','asr')}
    jobs = []
    # Uniform fixed indices, not selected for observed timing or quality.
    for alias, family in sorted(original['resident_aliases'].items()):
        for task in by_task if family == 'generalist_2' else (family,):
            pool = by_task[task]
            for idx in (0, len(pool)//4, len(pool)//2, 3*len(pool)//4, len(pool)-1):
                for repeat in range(2):
                    jobs.append(dict(id=f'solo:{alias}:{pool[idx]["pool_id"]}:{repeat}', requests=[
                        dict(alias=alias, task=task, pool_id=pool[idx]['pool_id'], offset_ms=0.)]))
    # A few actual co-runs expose possible output sensitivity to concurrency.
    for task in by_task:
        for idx in (0, 15):
            requests = []
            for alias, family in sorted(original['resident_aliases'].items()):
                active_task = task if family == 'generalist_2' else family
                sample = by_task[active_task][idx % len(by_task[active_task])]
                requests.append(dict(alias=alias, task=active_task, pool_id=sample['pool_id'], offset_ms=0.))
            jobs.append(dict(id=f'corun:{task}:{idx}', requests=requests))
    orders = {}
    for round_id in (1,2):
        order = copy.deepcopy(jobs); random.Random(seed+round_id-1).shuffle(order)
        orders[round_id] = order
    worker = ROOT/'tools/predictor_validation_20261027/scoped_reconstruct_observed_worker.py'
    source_receipts = json.loads((source/'sources.json').read_text())
    for receipt in source_receipts:
        if any(hashlib.sha256(Path(receipt[k]).read_bytes()).hexdigest() != receipt['sha256'] for k in ('path','snapshot')):
            raise ValueError('Frozen source drift: '+receipt['path'])
    extra = [worker, Path(__file__).resolve(), analyzer.resolve()]
    receipts = copy.deepcopy(source_receipts)
    existing = {r['path'] for r in receipts}
    for path in extra:
        if str(path) not in existing:
            receipts.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    frozen = []
    for i, receipt in enumerate(sorted(receipts,key=lambda r:r['path'])):
        path = Path(receipt['path']); snapshot = base/'source-snapshot'/str(i)/path.name
        snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(path.read_bytes())
        frozen.append(dict(path=str(path),snapshot=str(snapshot),sha256=receipt['sha256']))
    captures = []
    # ABBA reduces monotonic run-order drift; two rounds are diagnostic only.
    for name, scope, round_id in [('strict-round1','all_strict',1),('scoped-round1','g2_strict',1),
                                  ('scoped-round2','g2_strict',2),('strict-round2','all_strict',2)]:
        output = base/name; output.mkdir()
        context = copy.deepcopy(original)
        for binding in context['worker_bindings'].values():
            binding['command'][1] = str(worker)
            binding['env'].pop('GS_RECONSTRUCT_OBSERVER',None)
            binding['env'].pop('CUBLAS_WORKSPACE_CONFIG',None)
            binding['env'].update(GS_RUNTIME_SCOPE=scope,GS_SCOPED_OBSERVER='semantic')
        context.pop('deterministic_runtime',None)
        context.update(observer_diagnostic='five_model_runtime_scope_diagnostic',
            runtime_policy=dict(scope=scope,G2='strict plus ExLlama reconstruction threshold1',
                specialists='strict' if scope=='all_strict' else 'original fresh-process defaults',
                actual_settings_receipt='READY.runtime_scope'),
            source_content=[{k:r[k] for k in ('path','sha256')} for r in frozen])
        plan = dict(schema='gsdeploy.native_activity_training_plan.v1', mode='overlap',
            split='development_runtime_scope_diagnostic', not_for_training=True,
            execution_context_sha256=digest(context),run_id=name,seed=seed+round_id-1,
            aliases=context['resident_aliases'],jobs=orders[round_id],budget_seconds=1800,
            priming='largest_three_per_supported_task',runtime_scope=scope,
            matched_job_content_sha256=digest(orders[round_id]),
            confirmation_seeds_reserved=[6110491,6110492],formal_utility_qualified=False,
            purpose='matched runtime scope comparison; no causal isolation of individual torch flags')
        save(output/'context.json',context);save(output/'sources.json',frozen);save(output/'plan.json',plan)
        captures.append(str(output))
    save(base/'design.json',dict(schema='gsdeploy.runtime_scope_diagnostic_plan.v1',
        source_context=str(source/'context.json'),source_context_sha256=hashlib.sha256((source/'context.json').read_bytes()).hexdigest(),
        order=[Path(c).name for c in captures],seeds=[seed,seed+1],jobs_per_capture=len(jobs),
        requests_per_capture=sum(len(j['requests']) for j in jobs),
        criteria=dict(required_cross_scope_output_changes=0,required_within_scope_output_changes=0,
            required_missing_runtime_receipts=0,required_failed_requests=0),
        inference='report timing ratios and output changes; no predictor/observer accuracy qualification',
        acceptance_thresholds_unchanged=True,formal_utility_qualified=False,source_receipts=frozen))
    save(base/'queue.json',dict(schema='gsdeploy.native_capture_queue.v1',state=str(base/'continuation.json'),
        captures=captures,after=[dict(path=str(source.parent/'continuation.json'),status='capture_plan_completed')]))
    print(json.dumps(dict(base=str(base),captures=4,requests_per_capture=sum(len(j['requests']) for j in jobs))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--source',required=True,type=Path)
    p.add_argument('--seed',required=True,type=int)
    a=p.parse_args();prepare(a.output.resolve(),a.source.resolve(),a.seed,ROOT/'experiments/analyze_activation_runtime_scope.py')
