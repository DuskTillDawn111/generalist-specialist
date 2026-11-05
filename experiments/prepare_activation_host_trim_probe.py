"""Small development feasibility capture for measured post-prime host heap trim.

All five models load and complete the original full priming sequence. Eight
fixed input sentinels check output preservation. This cannot establish timing
repeatability or replace full-pool matched tests and new predictor training.
"""
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


def prepare(source,report_path,output):
    bundle=load_bundle(source);validate_capture(bundle,allowed_splits=('development_host_diagnostic',))
    report=json.loads(report_path.read_text())
    if (report.get('schema')!='gsdeploy.passive_host_diagnostic.v2' or not report['observer_repeat_check_passed']
            or report['issues'] or report['sources'].get(source.name)!=bundle['source']
            or bundle['plan']['host_trace_mode']!='on'):
        raise ValueError('Exact passed host diagnostic source required')
    groups={}
    for row in bundle['raw']['request_records']:
        groups.setdefault((row['model'],row['task']),[]).append(row['pool_id'])
    if len(groups)!=8:raise ValueError('All eight five-model paths required')
    jobs=[dict(id='sentinel:'+alias+':'+task,requests=[dict(alias=alias,task=task,pool_id=min(pools),offset_ms=0.)])
          for (alias,task),pools in sorted(groups.items())]
    paths={}
    for r in json.loads((source/'sources.json').read_text()):
        if any(hashlib.sha256(Path(r[k]).read_bytes()).hexdigest()!=r['sha256'] for k in ('path','snapshot')):
            raise ValueError('Frozen source drift: '+r['path'])
        paths[r['path']]=r['sha256']
    for path in [Path(__file__).resolve(),report_path,ROOT/'tools/predictor_validation_20261027/postprime_trim_scoped_worker.py',
                 ROOT/'experiments/analyze_activation_host_trim_probe.py',ROOT/'tests/test_postprime_trim_worker.py']:
        paths[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    output.mkdir(parents=True,exist_ok=False);frozen=[]
    for i,(path,sha) in enumerate(sorted(paths.items())):
        snap=output/'source-snapshot'/str(i)/Path(path).name;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(Path(path).read_bytes())
        frozen.append(dict(path=path,sha256=sha,snapshot=str(snap)))
    ctx=copy.deepcopy(bundle['context'])
    for binding in ctx['worker_bindings'].values():
        binding['command'][1]=str(ROOT/'tools/predictor_validation_20261027/postprime_trim_scoped_worker.py')
        binding['env']['GS_POSTPRIME_HOST_TRIM']='on'
    ctx.update(observer_diagnostic='five_model_postprime_host_trim_feasibility',
        postprime_host_trim=dict(mode='on',api='glibc malloc_trim(0)',timing='after full declared priming before worker READY',
            gc_policy_changes=False,extra_cuda_sync=False),source_content=[{k:r[k] for k in ('path','sha256')} for r in frozen])
    plan=dict(schema='gsdeploy.native_activity_training_plan.v1',mode='overlap',split='development_host_trim_feasibility',
        not_for_training=True,execution_context_sha256=digest(ctx),run_id='postprime-trim-feasibility',seed=None,
        aliases=ctx['resident_aliases'],jobs=jobs,budget_seconds=900,priming='largest_three_per_supported_task',
        original_source=bundle['source'],original_capture=str(source),sentinel_selection='lexicographically first pool_id per path, not timing selected',
        formal_latency_benefit_claimed=False,confirmation_seeds_reserved=[6110491,6110492],physical_predictor_qualified=False,formal_utility_qualified=False)
    save(output/'context.json',ctx);save(output/'sources.json',frozen);save(output/'plan.json',plan)
    save(output/'design.json',dict(schema='gsdeploy.postprime_heap_trim_feasibility.v1',source_receipts=frozen,
        purpose='Measure whether free CPU heap actually returns memory, retain full priming/lifecycle cost, check eight fixed outputs',
        required_followup='Full-pool matched repeated runtime comparison and fresh exact-context qualification before any adoption',
        original_source=bundle['source'],physical_predictor_qualified=False))
    print(json.dumps(dict(output=str(output),formal_requests=len(jobs),full_original_priming=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','observer-report','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();prepare(a.source.resolve(),a.observer_report.resolve(),a.output.resolve())
