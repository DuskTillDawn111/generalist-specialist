"""Trace CUDA APIs for one worker while replaying the matched five-model history.

Profiler data are development diagnosis only. The native, unprofiled host
capture remains the comparison; neither timing is an independent qualification.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.collect_activation_training import save
from gsdeploy.forecasting.native_training import validate_capture,digest


def prepare(source,report_path,output,alias):
    report=json.loads(report_path.read_text());bundle=load_bundle(source)
    validate_capture(bundle,allowed_splits=('development_host_diagnostic',))
    if (report.get('schema')!='gsdeploy.passive_host_diagnostic.v2' or not report['observer_repeat_check_passed']
            or report['issues'] or report['sources'].get(source.name)!=bundle['source']
            or bundle['plan']['host_trace_mode']!='on' or alias not in bundle['context']['resident_aliases']):
        raise ValueError('Passed complete matched host diagnostic and exact trace source required')
    nsys=shutil.which('nsys')
    if nsys is None:raise ValueError('Nsight Systems required for declared API diagnosis')
    version=subprocess.run([nsys,'--version'],check=True,capture_output=True,text=True).stdout.strip()
    paths={}
    for r in json.loads((source/'sources.json').read_text()):
        if any(hashlib.sha256(Path(r[k]).read_bytes()).hexdigest()!=r['sha256'] for k in ('path','snapshot')):
            raise ValueError('Source drift: '+r['path'])
        paths[r['path']]=r['sha256']
    additions=[Path(__file__).resolve(),report_path,source/'plan.json',source/'context.json',
        ROOT/'experiments/analyze_activation_cupti_host_probe.py',ROOT/'experiments/run_activation_cupti_host_probe.py',
        ROOT/'experiments/analyze_activation_host_diagnostic_v2.py',ROOT/'tests/test_cupti_host_probe.py']
    for path in additions:paths[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    output.mkdir(parents=True,exist_ok=False);frozen=[]
    for i,(path,sha) in enumerate(sorted(paths.items())):
        snap=output/'source-snapshot'/str(i)/Path(path).name;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(Path(path).read_bytes())
        frozen.append(dict(path=path,sha256=sha,snapshot=str(snap)))
    directory=output/'profiled-round1';directory.mkdir();ctx=copy.deepcopy(bundle['context'])
    trace_prefix=directory/(alias.replace('#','-')+'-trace')
    flags=['profile','--capture-range=none','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
        '--cuda-event-trace=false','--cuda-memory-usage=false','--cuda-trace-all-apis=true',
        '--osrt-threshold=1000000','--force-overwrite=false','--output='+str(trace_prefix)]
    original=ctx['worker_bindings'][alias]['command'];ctx['worker_bindings'][alias]['command']=[nsys,*flags,*original]
    ctx.update(observer_diagnostic='five_model_one_worker_cupti_host',
        profiler=dict(tool=nsys,version=version,executable_sha256=hashlib.sha256(Path(nsys).read_bytes()).hexdigest(),
            profiled_alias=alias,flags=flags,scope='one worker including startup/priming; remaining four resident workers unprofiled'),
        source_content=[{k:r[k] for k in ('path','sha256')} for r in frozen])
    jobs=copy.deepcopy(bundle['plan']['jobs'])
    plan=dict(schema='gsdeploy.native_activity_training_plan.v1',mode='overlap',split='development_cupti_host_diagnostic',
        not_for_training=True,execution_context_sha256=digest(ctx),run_id='profiled-round1',seed=bundle['plan']['seed'],
        aliases=ctx['resident_aliases'],jobs=jobs,budget_seconds=2400,priming='largest_three_per_supported_task',
        gc_trace_mode='on',host_trace_mode='on',profiled_alias=alias,matched_job_content_sha256=digest(jobs),
        unprofiled_source=bundle['source'],expected_nsys_report=str(trace_prefix)+'.nsys-rep',
        purpose='CUDA API attribution within matched NVTX requests; profiler perturbation retained; not predictor training',
        confirmation_seeds_reserved=[6110491,6110492],physical_predictor_qualified=False,formal_utility_qualified=False)
    save(directory/'context.json',ctx);save(directory/'sources.json',frozen);save(directory/'plan.json',plan)
    save(output/'design.json',dict(schema='gsdeploy.cupti_host_probe_plan.v1',source_receipts=frozen,
        capture=str(directory),unprofiled_capture=str(source),unprofiled_source=bundle['source'],profiled_alias=alias,
        nsys=ctx['profiler'],expected_nsys_report=plan['expected_nsys_report'],
        analysis='Match request/role/ordinal NVTX intervals on the same thread; retain all API overlaps and use interval unions, not sums of nested runtime/driver durations.',
        physical_predictor_qualified=False,formal_utility_qualified=False))
    print(json.dumps(dict(output=str(output),requests=len(jobs),profiled_alias=alias)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','observer-report','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--alias',default='asr#0');a=p.parse_args();prepare(a.source.resolve(),a.observer_report.resolve(),a.output.resolve(),a.alias)
