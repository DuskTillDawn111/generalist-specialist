"""Prepare/execute small training-only modality and shared SSG patches."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from .native_collect import ROOT,receipt,save,context_identity
from .operator_profile_campaign_v2 import prepare as source_packet,audit as source_audit
from .profiling import RESIDENTS,digest
from . import startup_collect_v4


def planned(selection):
    solo=[];shared=[]
    for task in ('image','asr'):
        for index in (0,3):
            j=dict(context='G2-solo-'+task,inputs={'generalist_2#0':selection['training'][task][index]['pool_id']},
                tasks={'generalist_2#0':task},start_offsets_ms={'generalist_2#0':0},input_stratum_index=index,
                split='training',validation_view='TRAINING_ONLY_NOT_VALIDATION',phase='modality-solo-training')
            j['id']=digest(j);solo.append(j)
    aliases=['text#0','text#1','generalist_2#0']
    for index in (0,3):
        for offset,order in [(0,aliases),(50,aliases),(50,list(reversed(aliases)))]:
            j=dict(context='SSG',inputs={a:selection['training']['text'][index]['pool_id'] for a in aliases},
                tasks=dict.fromkeys(aliases,'text'),start_offsets_ms={a:k*offset for k,a in enumerate(order)},
                input_stratum_index=index,split='training',validation_view='TRAINING_ONLY_NOT_VALIDATION',
                phase='shared-SSG-training')
            j['id']=digest(j);shared.append(j)
    return solo,shared


def prepare(output,smoke_report):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    source_packet(output/'_source-packet',smoke_report)
    common=json.loads((output/'_source-packet/structure-smoke/context.json').read_text())
    selection=json.loads((output/'_source-packet/input-selection.json').read_text())
    solo,shared=planned(selection)
    primers=[selection['training'][task][index] for task in ('text','image','asr','ocr') for index in (0,3)]
    pool=[json.loads(line) for line in (ROOT/'artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl').read_text().splitlines()]
    chosen={r['pool_id'] for r in primers}
    extra={Path(__file__).resolve(),Path(startup_collect_v4.__file__).resolve(),Path(__file__).with_name('nvtx_interval_index_v2.py')}
    extra.update(Path(r['sample']['input']) for r in pool if r['pool_id'] in chosen and 'input' in r['sample'])
    existing={s['original_path'] for s in common['execution_sources']}
    for index,path in enumerate(sorted(extra)):
        path=path.resolve()
        if str(path) in existing:continue
        frozen=output/'extra-source-snapshot'/str(index)/path.name;frozen.parent.mkdir(parents=True)
        shutil.copyfile(path,frozen);source=dict(original_path=str(path),frozen=receipt(frozen))
        common['execution_sources'].append(source);common['sources'].append(source['frozen'])
    common.update(training_only=True,collection_schema='gsdeploy.startup_native_phase_collection.v4',
        telemetry_scope='startup_before_runtime_construction_through_shutdown',task_domain=['text','image','asr','ocr'],
        collection_budget=dict(free_disk_reserve_bytes=12*2**30,total_timeout_seconds=2400),
        GPU_context_scheduling='UNKNOWN: no owned host-PID mapping in prior gpuctxsw capability probe')
    runs={}
    for shard,jobs,domains in [('modalities',solo,['metadata','light-kernel','natural']),
                               ('shared-SSG',shared,['light-kernel','natural'])]:
        for domain in domains:
            run=output/shard/domain;run.mkdir(parents=True)
            context=copy.deepcopy(common)
            context.update(worker_variant='operator_worker_v2.py' if domain=='metadata' else 'semantic_worker.py',
                affine_observation=domain=='metadata',measurement_domain=domain,
                profiler='none-natural-with-semantic-markers' if domain=='natural' else 'nsys-cuda-nvtx')
            sha=context_identity(context)
            plan=dict(schema='gsdeploy.small_startup_training_patch_plan.v4',execution_context_sha256=sha,
                resident_aliases=RESIDENTS,inputs=primers,jobs=jobs,split='training',training_only=True,domain=domain,
                shard=shard,paired_jobs_sha256=digest(jobs),input_selection=receipt(output/'_source-packet/input-selection.json'),
                inherited_runtime_smoke=common['inherited_runtime_smoke'],no_retries=True,GPU_started=False,
                claim='TRAINING_ONLY_STARTUP_AWARE_PATCH_NOT_VALIDATION_OR_DEPLOYMENT_GAIN')
            plan['plan_sha256']=digest(plan)
            save(run/'context.json',context);save(run/'plan.json',plan)
            runs[shard+'/'+domain]=dict(plan=receipt(run/'plan.json'),context=receipt(run/'context.json'),
                execution_context_sha256=sha,jobs=len(jobs),requests=sum(len(j['inputs']) for j in jobs))
    result=dict(schema='gsdeploy.small_startup_training_patch_campaign.v4',status='PREPARED_CPU_ONLY',runs=runs,
        no_gpu_started=True,validation_started=False,independent_capture_count=0,
        limitations=['No long decode coverage','Cross-process GPU context scheduling UNKNOWN',
                     'Per-worker profiler sessions do not establish bounded cross-process ordering',
                     'No heldout prediction or end-to-end utility claim'],
        execution_order=['modalities/metadata','modalities/light-kernel','modalities/natural',
                         'shared-SSG/light-kernel','shared-SSG/natural'],
        storage_scenario='12 modality requests across three domains plus 36 shared requests across light/natural; new image/ASR traces unknown size, review each shard before continuing.')
    save(output/'campaign.json',result)
    for name in runs:audit(output/name)
    return result


def audit(run):
    run=Path(run).resolve();result=source_audit(run)
    plan=json.loads((run/'plan.json').read_text())
    if plan.get('schema')!='gsdeploy.small_startup_training_patch_plan.v4' or plan.get('split')!='training':raise ValueError('Declared training patch required')
    count=4 if plan.get('shard')=='modalities' else 6 if plan.get('shard')=='shared-SSG' else None
    if count is None or len(plan['jobs'])!=count or any(j['split']!='training' for j in plan['jobs']):raise ValueError('Exact small patch required')
    return result


def execute(run):
    run=Path(run).resolve();audit(run)
    plan=json.loads((run/'plan.json').read_text());context=json.loads((run/'context.json').read_text())
    if shutil.disk_usage(run).free<context['collection_budget']['free_disk_reserve_bytes']:raise ValueError('Disk reserve insufficient')
    raw=startup_collect_v4.collect(run,trace=plan['domain']!='natural')
    return dict(status=raw['status'],raw=receipt(run/'raw.json'),requests=len(raw['request_records']))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True);p.add_argument('--smoke-report',required=True)
    p=sub.add_parser('audit');p.add_argument('--run',required=True)
    p=sub.add_parser('execute');p.add_argument('--run',required=True);p.add_argument('--execute',action='store_true',required=True)
    args=parser.parse_args();result=prepare(args.output,args.smoke_report) if args.op=='prepare' else audit(args.run) if args.op=='audit' else execute(args.run)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
