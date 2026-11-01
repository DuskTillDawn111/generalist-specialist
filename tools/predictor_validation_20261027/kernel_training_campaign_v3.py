"""Three-domain training bridge and predeclared follow-on design.

The bridge has 8 isolated requests/domain. The larger 62-job/domain design is
plan only; this runner cannot execute it. Validation is not implemented here.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import time

from . import background_collect as collector
from .native_collect import ROOT, context_identity, receipt, save
from .operator_profile_campaign_v2 import prepare as source_packet, audit as source_audit
from .profiling import RESIDENTS, digest

DOMAINS = ('metadata', 'light-kernel', 'natural')


def training_jobs(selection, *, bridge=False):
    inputs = selection['training']
    jobs = []
    for index in ((0, 3) if bridge else range(4)):
        for alias in RESIDENTS:
            task = 'ocr' if alias == 'ocr#0' else 'text'
            for repeat in range(1 if bridge else 2):
                job = dict(context='solo-'+alias, inputs={alias:inputs[task][index]['pool_id']},
                    tasks={alias:task}, start_offsets_ms={alias:0}, repeat=repeat, split='training',
                    validation_view='TRAINING_ONLY_NOT_VALIDATION', input_stratum_index=index,
                    phase='training-bridge' if bridge else 'proposed-full-training')
                job['id'] = digest(job); jobs.append(job)
    if not bridge:
        contexts = [('SS', ['text#0','text#1']), ('SG', ['text#0','generalist_2#0']),
                    ('SSG', ['text#0','text#1','generalist_2#0'])]
        for index in (0, 3):
            for name, aliases in contexts:
                for offset in (0, 50, 200):
                    for order in ([aliases] if offset == 0 else [aliases, list(reversed(aliases))]):
                        job = dict(context=name, inputs={a:inputs['text'][index]['pool_id'] for a in aliases},
                            tasks=dict.fromkeys(aliases,'text'), start_offsets_ms={a:k*offset for k,a in enumerate(order)},
                            repeat=0, split='training', validation_view='TRAINING_ONLY_NOT_VALIDATION',
                            input_stratum_index=index, phase='proposed-full-training')
                        job['id'] = digest(job); jobs.append(job)
    return jobs


def design(selection):
    bridge, full = training_jobs(selection, bridge=True), training_jobs(selection)
    def summary(jobs):
        counts = {alias:sum(alias in j['inputs'] for j in jobs) for alias in RESIDENTS}
        return dict(jobs_per_domain=len(jobs), measured_requests_per_domain=sum(counts.values()),
                    owner_requests_per_domain=counts, jobs_all_three_domains=3*len(jobs),
                    measured_requests_all_domains=3*sum(counts.values()))
    return dict(schema='gsdeploy.kernel_training_three_domain_design.v3', training_only=True,
        domains=list(DOMAINS), bridge=dict(summary(bridge), jobs=bridge),
        proposed_full_training=dict(summary(full),jobs=full, executable=False),
        partition='training public inputs only; smoke artifacts excluded from fitting',
        missing_domains=['G2 image and ASR', 'SSGO/background activity', 'independent validation',
                         'resource shares', 'joint batch/shared-weight execution'],
        inference_rule='No host-gap scaling and no kernel stage-ratio multiplication across profiling domains.',
        bridge_acceptance='Exact input/output/token/forward work plus ordered kernel signature correspondence; mismatches remain UNKNOWN.',
        storage_policy='12 GiB free reserve; one canonical parse per trace; no duplicate v2 and v3 parse export.',
        no_validation_execution=True)


def prepare(output, smoke_report):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    source_packet(output/'_source-packet',smoke_report)
    common=json.loads((output/'_source-packet/structure-smoke/context.json').read_text())
    selection=json.loads((output/'_source-packet/input-selection.json').read_text())
    planned=design(selection)
    save(output/'training-design.json',planned)
    bridge=planned['bridge']['jobs']
    primers=[selection['training'][task][index] for task in ('text','ocr') for index in (0,3)]
    pool=[json.loads(line) for line in (ROOT/'artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl').read_text().splitlines()]
    extra={Path(__file__).resolve(),Path(__file__).with_name('nvtx_interval_index_v2.py')}
    chosen={p['pool_id'] for p in primers}
    extra.update(Path(row['sample']['input']) for row in pool if row['pool_id'] in chosen and 'input' in row['sample'])
    bound={s['original_path'] for s in common['execution_sources']}
    for index,path in enumerate(sorted(extra)):
        path=path.resolve()
        if str(path) in bound:continue
        frozen=output/'extra-source-snapshot'/str(index)/path.name;frozen.parent.mkdir(parents=True)
        shutil.copyfile(path,frozen)
        source=dict(original_path=str(path),frozen=receipt(frozen))
        common['execution_sources'].append(source);common['sources'].append(source['frozen'])
    common['training_only']=True
    common['collection_budget']=dict(free_disk_reserve_bytes=12*2**30,loaded_phase_timeout_seconds=1800,
                                     over_budget='fail_preserve_raw_no_retry')
    runs={}
    for domain in DOMAINS:
        run=output/domain;run.mkdir()
        context=copy.deepcopy(common)
        context.update(worker_variant='operator_worker_v2.py' if domain=='metadata' else 'semantic_worker.py',
            affine_observation=domain=='metadata',profiler='none-natural-with-semantic-markers' if domain=='natural' else 'nsys-cuda-nvtx',
            measurement_domain=domain)
        sha=context_identity(context)
        plan=dict(schema='gsdeploy.kernel_training_bridge_plan.v3',execution_context_sha256=sha,
            resident_aliases=RESIDENTS,inputs=primers,jobs=bridge,no_retries=True,GPU_started=False,
            split='training',training_only=True,domain=domain,paired_jobs_sha256=digest(bridge),
            input_selection=receipt(output/'_source-packet/input-selection.json'),
            inherited_runtime_smoke=common['inherited_runtime_smoke'],
            claim='NEW_TRAINING_BRIDGE_NOT_SMOKE_FITTING_NOT_VALIDATION_NOT_DEPLOYMENT_GAIN')
        plan['plan_sha256']=digest(plan)
        save(run/'context.json',context);save(run/'plan.json',plan)
        runs[domain]=dict(plan=receipt(run/'plan.json'),context=receipt(run/'context.json'),
                          execution_context_sha256=sha,jobs=len(bridge))
    result=dict(schema='gsdeploy.kernel_training_bridge_campaign.v3',status='PREPARED_CPU_ONLY',runs=runs,
                design=receipt(output/'training-design.json'),validation_started=False,full_training_executable=False)
    save(output/'campaign.json',result)
    return result


def audit(run):
    result=source_audit(run)
    plan=json.loads((Path(run)/'plan.json').read_text())
    if (plan.get('split')!='training' or plan.get('training_only') is not True or len(plan['jobs'])!=8
            or any(j['split']!='training' or j['phase']!='training-bridge' for j in plan['jobs'])):
        raise ValueError('Only the declared 8-job training bridge is executable')
    return result


def execute(run):
    run=Path(run).resolve();audit(run)
    plan=json.loads((run/'plan.json').read_text());context=json.loads((run/'context.json').read_text())
    reserve=context['collection_budget']['free_disk_reserve_bytes']
    if shutil.disk_usage(run).free<reserve:raise RuntimeError('Insufficient free disk reserve before GPU load')
    native_monitor=collector.BoardMonitor
    class BudgetMonitor(native_monitor):
        def __init__(self,*args,**kwargs):
            self.budget_start=time.monotonic();self.last_budget_check=0
            super().__init__(*args,**kwargs)
        def sample(self):
            row=super().sample();now=time.monotonic()
            if now-self.last_budget_check>=1:
                self.last_budget_check=now
                if shutil.disk_usage(run).free<reserve:self.errors.append('TRAINING_DISK_RESERVE_EXCEEDED')
                if now-self.budget_start>context['collection_budget']['loaded_phase_timeout_seconds']:
                    self.errors.append('TRAINING_LOADED_PHASE_TIME_BUDGET_EXCEEDED')
            return row
    collector.BoardMonitor=BudgetMonitor
    try:raw=collector.collect(run,trace=plan['domain']!='natural')
    finally:collector.BoardMonitor=native_monitor
    return dict(status=raw['status'],raw=receipt(run/'raw.json'),requests=len(raw['request_records']),
                domain=plan['domain'],training_only=True,validation_started=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('design');p.add_argument('--selection',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True);p.add_argument('--smoke-report',required=True)
    p=sub.add_parser('audit');p.add_argument('--run',required=True)
    p=sub.add_parser('execute');p.add_argument('--run',required=True);p.add_argument('--execute',action='store_true',required=True)
    args=parser.parse_args()
    if args.op=='design':
        result=design(json.loads(Path(args.selection).read_text()));save(args.output,result)
    elif args.op=='prepare':result=prepare(args.output,args.smoke_report)
    elif args.op=='audit':result=audit(args.run)
    else:result=execute(args.run)
    if args.op=='design':print(json.dumps({name:{k:v for k,v in result[name].items() if k!='jobs'} for name in ('bridge','proposed_full_training')}))
    else:print(json.dumps(result,indent=2))
    return 1 if result.get('status')=='failed' else 0


if __name__=='__main__':raise SystemExit(main())
