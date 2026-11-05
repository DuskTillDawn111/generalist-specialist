"""Full-pool ABBA comparison of post-prime free-host-heap reclamation."""
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


def prepare(host_base,feasibility,output):
    gate_path=host_base/'description-v2.json';gate=json.loads(gate_path.read_text());probe=json.loads((feasibility/'description.json').read_text())
    if not gate['observer_repeat_check_passed'] or gate['issues'] or probe['output_issues'] or probe['latency_benefit_established']:
        raise ValueError('Passed observation and output-preserving feasibility required')
    bundles={i:load_bundle(host_base/f'trace-round{i}') for i in (1,2)}
    for i,b in bundles.items():
        validate_capture(b,allowed_splits=('development_host_diagnostic',))
        if b['source']!=gate['sources'][f'trace-round{i}'] or len(b['plan']['jobs'])!=320:raise ValueError('Exact full-pool source required')
    if probe['reference']!=bundles[1]['source'] or not any(r['rss_reduction_mib']>0 for r in probe['receipts']):
        raise ValueError('Measured free-heap reclamation receipt required')
    if bundles[1]['context']!=bundles[2]['context']:raise ValueError('Matched source contexts required')
    paths={}
    for r in json.loads((host_base/'trace-round1/sources.json').read_text()):
        if any(hashlib.sha256(Path(r[k]).read_bytes()).hexdigest()!=r['sha256'] for k in ('path','snapshot')):raise ValueError('Frozen source drift: '+r['path'])
        paths[r['path']]=r['sha256']
    extras=[Path(__file__).resolve(),gate_path,feasibility/'description.json',
        ROOT/'tools/predictor_validation_20261027/postprime_trim_scoped_worker.py',
        ROOT/'experiments/analyze_activation_host_trim_comparison.py',ROOT/'experiments/continue_activation_host_trim_comparison.py',
        ROOT/'experiments/analyze_activation_host_diagnostic_v2.py',ROOT/'tests/test_postprime_trim_worker.py']
    for path in extras:paths[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    output.mkdir(parents=True,exist_ok=False);frozen=[]
    for i,(path,sha) in enumerate(sorted(paths.items())):
        snap=output/'source-snapshot'/str(i)/Path(path).name;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(Path(path).read_bytes())
        frozen.append(dict(path=path,sha256=sha,snapshot=str(snap)))
    captures=[]
    for name,mode,index in [('control-round1','off',1),('trim-round1','on',1),('trim-round2','on',2),('control-round2','off',2)]:
        directory=output/name;directory.mkdir();ctx=copy.deepcopy(bundles[index]['context']);jobs=copy.deepcopy(bundles[index]['plan']['jobs'])
        for binding in ctx['worker_bindings'].values():
            binding['command'][1]=str(ROOT/'tools/predictor_validation_20261027/postprime_trim_scoped_worker.py')
            binding['env']['GS_POSTPRIME_HOST_TRIM']=mode
        ctx.update(observer_diagnostic='five_model_postprime_trim_'+mode,
            postprime_host_trim=dict(mode=mode,api='glibc malloc_trim(0)',timing='after full declared priming before worker READY',gc_policy_changes=False,extra_cuda_sync=False),
            source_content=[{k:r[k] for k in ('path','sha256')} for r in frozen])
        plan=dict(schema='gsdeploy.native_activity_training_plan.v1',mode='overlap',split='development_host_trim_comparison',
            not_for_training=True,execution_context_sha256=digest(ctx),run_id=name,seed=bundles[index]['plan']['seed'],
            aliases=ctx['resident_aliases'],jobs=jobs,budget_seconds=2400,priming='largest_three_per_supported_task',
            gc_trace_mode='on',host_trace_mode='on',host_trim_mode=mode,matched_job_content_sha256=digest(jobs),
            source=bundles[index]['source'],purpose='CPU heap runtime intervention, not observer overhead test; full source order replay',
            seed_reuse='intentional development history replay, not independent qualification',confirmation_seeds_reserved=[6110491,6110492],
            physical_predictor_qualified=False,formal_utility_qualified=False)
        save(directory/'context.json',ctx);save(directory/'sources.json',frozen);save(directory/'plan.json',plan);captures.append(str(directory))
    save(output/'design.json',dict(schema='gsdeploy.host_trim_runtime_comparison.v1',source_receipts=frozen,
        sources={str(i):b['source'] for i,b in bundles.items()},feasibility_source=str(feasibility/'description.json'),
        requests_per_capture=320,matched_order=True,runtime_change='malloc_trim after full priming, before worker READY',
        accuracy_gates_changed=False,qualification_scope='development only; new observer/baseline/training/independent qualification required before adoption',
        physical_predictor_qualified=False,formal_utility_qualified=False))
    save(output/'queue.json',dict(schema='gsdeploy.native_capture_queue.v1',state=str(output/'continuation.json'),captures=captures))
    print(json.dumps(dict(output=str(output),captures=4,requests_per_capture=320)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('host-base','feasibility','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();prepare(a.host_base.resolve(),a.feasibility.resolve(),a.output.resolve())
