"""Explicitly paired natural and kernel timing profiles, original four owners.

New isolated plans bind their mode and identical offered jobs. Historical v2
profiler-only plans cannot enter either execution branch. CPU preparation only;
actual launch is exclusively through profile_training_supervisor_v2.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from . import kernel_profile_campaign_v2 as old
from .background_collect import context_identity, receipt, save
from .profiling import RESIDENTS, digest
from .implementation_source_closure_v27 import implementation_sources
from .trace_export import export_sqlite, parse_sqlite

SCHEMA = 'gsdeploy.paired_profile_campaign.v5'
SMOKE_SCHEMA = 'gsdeploy.paired_profile_smoke.v5'
MODES = ('natural', 'kernel')


def required_sources(context):
    variant=context.get('worker_variant')
    if variant not in ('semantic_worker.py','semantic_progress_worker_v1.py'):
        raise ValueError('Explicit audited semantic worker variant required')
    worker=Path(__file__).with_name(variant)
    paths={r['path'] for r in implementation_sources(__file__,worker)}
    root=Path(__file__).resolve().parents[2]
    base=root/'artifacts/static-study-v6-20261021/inputs'
    paths.update(str(base/n) for n in ('worker-registry.json','runtime-inputs.jsonl'))
    registry=read(base/'worker-registry.json')
    if context['worker_bindings'] != {f:registry[f] for f in sorted(set(RESIDENTS.values()))}:
        raise ValueError('Original exact worker registry bindings required')
    for entry in context['worker_bindings'].values():
        spec_path=Path(entry['spec']);paths.add(str(spec_path));spec=read(spec_path)
        command=entry['command']
        frozen=Path(command[command.index('--source-root')+1])
        paths.update(str(p) for p in frozen.rglob('*.py'))
        if spec.get('adapter_source'):paths.add(spec['adapter_source'])
    pair_input_ids={x['pool_id'] for x in context.get('declared_inputs',[])}
    rows=[json.loads(line) for line in (base/'runtime-inputs.jsonl').read_text().splitlines()]
    paths.update(r['sample']['input'] for r in rows if r['pool_id'] in pair_input_ids and 'input' in r['sample'])
    return [receipt(path) for path in sorted(paths)]


def read(path):
    return json.loads(Path(path).read_text())


def checked(ref):
    if set(ref) != {'path', 'sha256'} or receipt(ref['path']) != ref:
        raise ValueError('Paired campaign immutable reference changed')
    return read(ref['path'])


def pilot_jobs(selection):
    """18 matched length/context cells plus8 shifted SG cells; training only."""
    jobs = old.make_jobs(selection, 'training', 20261028041)
    chosen = []
    for index in (0, 3):
        text_id = selection['training']['text'][index]['pool_id']
        ocr_id = selection['training']['ocr'][index]['pool_id']
        for context in (*old.CONTEXTS, 'S1G'):
            match = [j for j in jobs if j['context'] == context and j['repeat'] == 0
                and all(v == 0 for v in j['start_offsets_ms'].values())
                and all(v == (ocr_id if a == 'ocr#0' else text_id) for a,v in j['inputs'].items())]
            if len(match) != 1:
                raise ValueError('Unique declared pilot context/length cell required')
            chosen.extend(match)
        chosen.extend(j for j in jobs if j['context'] == 'SG' and j['repeat'] == 0
            and j['inputs']['text#0'] == text_id and max(j['start_offsets_ms'].values()) in (50,200))
    if len(chosen) != 26 or len({j['id'] for j in chosen}) != len(chosen):
        raise ValueError('Exactly26 distinct pilot jobs required')
    return chosen


def prepare(output, *, progress_observer=False):
    if type(progress_observer) is not bool:raise ValueError('Explicit boolean observer mode required')
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    old.prepare(output/'source-freeze')
    base = read(output/'source-freeze/training/context.json')
    base['worker_variant']='semantic_progress_worker_v1.py' if progress_observer else 'semantic_worker.py'
    base['device_progress_observer']=dict(enabled=progress_observer,
        scope='current_stream_cuda_event_completion_only_not_all_stream_frontier_or_progress_fraction',
        overhead='extra_event_record_and_nonblocking_query' if progress_observer else 'none')
    capability = Path(__file__).resolve().parents[2]/'artifacts/predictor-gpu-metrics-capability-20261028-readonly01/report.json'
    base.update(counters='UNAVAILABLE_CURRENT_ERR_NVGPUCTRPERM', counter_capability=receipt(capability))
    selection = read(output/'source-freeze/input-selection.json')
    extra = []
    base['declared_inputs']=read(output/'source-freeze/training/plan.json')['inputs']
    for index, ref in enumerate(required_sources(base)):
        path = Path(ref['path'])
        frozen = output/'new-source-freeze'/str(index)/path.name
        frozen.parent.mkdir(parents=True)
        shutil.copyfile(path, frozen)
        extra.append(dict(original_path=str(path), frozen=receipt(frozen)))
    existing = {x['original_path'] for x in base['execution_sources']}
    base['execution_sources'].extend(x for x in extra if x['original_path'] not in existing)
    base['sources'] = [x['frozen'] for x in base['execution_sources']]
    jobs = pilot_jobs(selection)
    smoke = read(output/'source-freeze/smoke/plan.json')
    runs = {}
    for split in ('smoke','training'):
        offered = smoke['jobs'] if split == 'smoke' else jobs
        inputs = smoke['inputs'] if split == 'smoke' else read(output/'source-freeze/training/plan.json')['inputs']
        pair = dict(schema=SCHEMA+'.pair', split=split, jobs=offered, inputs=inputs,
            progress_observer=progress_observer,
            residents=RESIDENTS, no_retries=True,
            scope='training-only pilot; finite job boundaries; not steady-state P95 or DeltaU')
        save(output/(split+'-pair.json'), pair)
        for mode in MODES:
            run = output/split/mode
            run.mkdir(parents=True)
            context = copy.deepcopy(base)
            context.update(protocol='paired_profile_v5_'+mode, paired_profile_mode=mode,
                profiler='none-natural-timing' if mode == 'natural' else 'nsys-cuda-nvtx-profiler-perturbed',
                supervisor='profile_training_supervisor_v2_outer_lease_only')
            plan = dict(schema=SCHEMA, mode=mode, split=split, pair=receipt(output/(split+'-pair.json')),
                progress_observer=progress_observer,
                resident_aliases=RESIDENTS, jobs=offered, inputs=inputs,
                execution_context_sha256=context_identity(context),
                mode_switch_at_execute=False, GPU_started=False, no_retries=True,
                fit_eligible=False, final_gain_eligible=False)
            plan['plan_sha256'] = digest(plan)
            save(run/'context.json', context); save(run/'plan.json', plan)
            runs[split+'/'+mode] = dict(plan=receipt(run/'plan.json'), context=receipt(run/'context.json'))
    result = dict(schema=SCHEMA+'.index', status='CPU_PREPARED_TRAINING_PILOT_ONLY',
        runs=runs, modes=list(MODES), training_jobs_per_mode=len(jobs),
        validation='NOT_PREPARED_REQUIRES_FROZEN_PREDICTOR_AND_SEPARATE_GATE',
        GPU_started=False, fit_eligible=False, final_gain_eligible=False)
    save(output/'campaign.json', result)
    for name in runs:
        audit(output/name)
    return result


def audit(run):
    run = Path(run).resolve()
    plan = read(run/'plan.json'); context = read(run/'context.json')
    if (plan.get('schema') != SCHEMA or plan.get('mode') not in MODES
            or plan.get('split') not in ('smoke','training')
            or plan.get('mode_switch_at_execute') is not False
            or plan.get('no_retries') is not True):
        raise ValueError('Explicit v5 paired mode/split plan required; old profiler plans refused')
    mode = plan['mode']
    progress=plan.get('progress_observer')
    expected_variant='semantic_progress_worker_v1.py' if progress else 'semantic_worker.py'
    if (type(progress) is not bool or context.get('worker_variant')!=expected_variant
            or context.get('device_progress_observer')!=dict(enabled=progress,
                scope='current_stream_cuda_event_completion_only_not_all_stream_frontier_or_progress_fraction',
                overhead='extra_event_record_and_nonblocking_query' if progress else 'none')):
        raise ValueError('Exact opt-in device progress observer identity required')
    if (context.get('paired_profile_mode') != mode or context.get('protocol') != 'paired_profile_v5_'+mode
            or context.get('profiler') != ('none-natural-timing' if mode == 'natural' else 'nsys-cuda-nvtx-profiler-perturbed')
            or context.get('supervisor') != 'profile_training_supervisor_v2_outer_lease_only'
            or plan['resident_aliases'] != RESIDENTS or context['residents'] != RESIDENTS
            or plan['execution_context_sha256'] != context_identity(context)
            or plan.get('plan_sha256') != digest({k:v for k,v in plan.items() if k != 'plan_sha256'})):
        raise ValueError('Paired profile mode/deployment/context identity mismatch')
    pair = checked(plan['pair'])
    if (pair.get('schema') != SCHEMA+'.pair' or pair['split'] != plan['split']
            or pair['jobs'] != plan['jobs'] or pair['inputs'] != plan['inputs'] or pair['residents'] != RESIDENTS
            or pair.get('progress_observer') != progress):
        raise ValueError('Natural/kernel offered job pairing mismatch')
    if not {x['pool_id'] for x in plan['inputs']} <= {x['pool_id'] for x in context.get('declared_inputs',[])}:
        raise ValueError('Complete declared input closure required')
    for src in context['execution_sources']:
        if receipt(src['frozen']['path']) != src['frozen'] or receipt(src['original_path'])['sha256'] != src['frozen']['sha256']:
            raise ValueError('Paired profile source/input drift: '+src['original_path'])
    if context['sources'] != [s['frozen'] for s in context['execution_sources']]:
        raise ValueError('Exact executable source closure required')
    checked(context['counter_capability'])
    required = required_sources(context)
    frozen = {(s['original_path'],s['frozen']['sha256']) for s in context['execution_sources']}
    if any((s['path'],s['sha256']) not in frozen for s in required):
        raise ValueError('New paired implementation closure missing')
    return plan


def execution_gate(run):
    run = Path(run).resolve(); plan = audit(run)
    if any((run/name).exists() for name in ('attempt.json','raw.json')) or list(run.glob('trace-*.nsys-rep')):
        raise FileExistsError('Paired run already attempted; no retry')
    if plan['split'] == 'training':
        for mode in MODES:
            smoke_run = run.parents[1]/'smoke'/mode
            smoke = audit(smoke_run)
            report = read(smoke_run/'smoke-report.json')
            if (report.get('schema') != SMOKE_SCHEMA or report.get('status') != 'PASS'
                    or report.get('mode') != mode or report.get('plan') != receipt(smoke_run/'plan.json')
                    or report.get('raw') != receipt(smoke_run/'raw.json') or report.get('issues') != []
                    or report.get('execution') is None):
                raise ValueError('Both exact independently reviewed paired smoke modes required')
            from .profile_training_supervisor_v2 import completed_execution
            completed_execution(report['execution'], smoke_run)
    return plan


def collect(run, *, lease_fd):
    from .profile_owned_collect_v2 import collect as owned_collect
    from .profile_training_supervisor_v2 import validate_lease
    validate_lease(lease_fd)
    run = Path(run).resolve(); plan = execution_gate(run)
    save(run/'attempt.json', dict(schema=SCHEMA+'.attempt', plan=receipt(run/'plan.json'),
        mode=plan['mode'], GPU_started=False, no_retries=True))
    raw = owned_collect(run, trace=(plan['mode'] == 'kernel'), lease_fd=lease_fd)
    audit(run)
    return raw


def verify_smoke(run, execution_path):
    """CPU-only review; natural mode does not invent CUDA-kernel observations."""
    run = Path(run).resolve(); plan = audit(run)
    if plan['split'] != 'smoke':
        raise ValueError('Only paired smoke can be verified')
    from .profile_training_supervisor_v2 import completed_execution
    execution = completed_execution(receipt(execution_path), run); raw = read(run/'raw.json')
    issues = []; outputs = {}
    if (execution.get('status') != 'CHILD_EXITED_REQUIRES_PROFILE_REVIEW'
            or execution.get('live_group_pids') != [] or execution.get('child_reaped') is not True):
        issues.append('SUPERVISOR_EXIT_FAILED')
    expected_scope = 'profiler_perturbed' if plan['mode'] == 'kernel' else 'unprofiled_with_phase_markers'
    if (raw.get('status') != 'completed' or raw.get('runtime_errors') or raw.get('telemetry_errors')
            or raw.get('errors') or raw.get('performance_scope') != expected_scope
            or raw.get('plan') != receipt(run/'plan.json') or raw.get('context') != receipt(run/'context.json')):
        issues.append('RAW_SCOPE_OR_SOURCE_OR_RUNTIME_FAILED')
    ready = raw.get('ready', {})
    if (set(ready) != set(RESIDENTS) or len({r.get('pid') for r in ready.values()}) != 4
            or any(type(r.get('pid')) is not int or r['pid'] <= 0 for r in ready.values())):
        issues.append('FOUR_DISTINCT_READY_PIDS_REQUIRED')
    rows = raw.get('request_records', [])
    if len(rows) != 4 or {r['model'] for r in rows} != set(RESIDENTS):
        issues.append('COMPLETE_FOUR_OWNER_COHORT_REQUIRED')
    executed = checked(raw['executed_jobs'])
    if executed != dict(split=None,contexts=None,limit=None,trace=(plan['mode']=='kernel'),jobs=plan['jobs']):
        issues.append('EXACT_FROZEN_JOB_EXECUTION_REQUIRED')
    batches = raw.get('batches',[])
    job = plan['jobs'][0]
    if (len(plan['jobs']) != 1 or len(batches) != 1 or batches[0].get('job_id') != job['id']
            or type(batches[0].get('start_ns')) is not int):
        issues.append('SMOKE_JOB_BOUNDARY_REQUIRED')
    else:
        for row in rows:
            alias = row['model']
            if (row.get('profile_job_id') != job['id'] or row.get('pool_id') != job['inputs'].get(alias)
                    or row.get('task') != job['tasks'].get(alias)
                    or row.get('scheduled_time_ns') != batches[0]['start_ns']+round(job['start_offsets_ms'][alias]*1e6)):
                issues.append('EXACT_SMOKE_TASK_INPUT_ARRIVAL_REQUIRED:'+row['request_id'])
    expected_priming=[]
    for alias,family in RESIDENTS.items():
        tasks=['text','image','asr'] if family=='generalist_2' else ['ocr' if family=='ocr' else 'text']
        for task in tasks:
            expected_priming.extend((alias,task,item['pool_id']) for item in sorted(
                (x for x in plan['inputs'] if x['task']==task),key=lambda x:x['input_size'],reverse=True)[:3])
    priming=raw.get('priming_request_records',[])
    if ([(r.get('model'),r.get('task'),r.get('pool_id')) for r in priming] != expected_priming
            or any(r.get('status')!='completed' for r in priming)):
        issues.append('EXACT_SUCCESSFUL_PRIMING_COHORT_REQUIRED')
    shutdown=raw.get('shutdown',{})
    if (set(shutdown)!=set(RESIDENTS) or any(r.get('exit_code')!=0 or r.get('forced') is not False
            or r.get('bye_received') is not True for r in shutdown.values())):
        issues.append('CLEAN_FOUR_OWNER_SHUTDOWN_REQUIRED')
    samples = raw.get('board_samples', [])
    if any(s.get('foreign_pids') for s in samples) or not any(set(s.get('owned_aliases', [])) == set(RESIDENTS) for s in samples):
        issues.append('ALL_OWNED_MEMORY_OR_FOREIGN_PROCESS_FAILED')
    for alias in RESIDENTS:
        selected = [r for r in rows if r['model'] == alias]
        for row in selected:
            response = row.get('worker_response', {}); nxt = row.get('previous_handler_next_request_ready', {})
            start,end = response.get('start_perf_ns'),response.get('end_perf_ns')
            receipt_ns = row.get('response_reader_event_enqueue_time_ns')
            if (row.get('status') != 'completed' or any(type(v) is not int for v in (start,end,receipt_ns,nxt.get('perf_ns'),nxt.get('received_time_ns')))
                    or not start <= end <= receipt_ns or not end <= nxt['perf_ns'] <= nxt['received_time_ns']):
                issues.append('HANDLER_RECEIPT_NEXT_READY_ORDER_FAILED:'+row['request_id'])
            starts=[e for e in row.get('execution_events',[]) if 'event' not in e]
            if (len(starts)!=1 or starts[0].get('request_id')!=row['request_id']
                    or starts[0].get('perf_ns')!=start or starts[0].get('pid')!=ready.get(alias,{}).get('pid')):
                issues.append('ACTUAL_START_EVENT_REQUIRED:'+row['request_id'])
            try:
                anchors=old.anchors_from_records([row],ready.get(alias,{}).get('pid'))
                if not anchors or any(not start<=a['start_ns']<=a['end_ns']<=end for a in anchors):
                    issues.append('MATCHED_HANDLER_PHASE_ANCHORS_REQUIRED:'+row['request_id'])
            except (ValueError,TypeError):
                issues.append('MATCHED_HANDLER_PHASE_ANCHORS_REQUIRED:'+row['request_id'])
        if plan['mode'] == 'natural':
            continue
        if alias not in ready:
            issues.append('MISSING_READY:'+alias); continue
        report = run/('trace-'+alias.replace('#','-')+'.nsys-rep')
        sql = report.with_suffix('.sqlite'); export = export_sqlite(report,sql)
        parsed = parse_sqlite(sql,old.anchors_from_records(selected,ready[alias]['pid']))
        path = run/('kernels-'+alias.replace('#','-')+'.json'); save(path,parsed)
        outputs[alias] = dict(export=export,parsed=receipt(path))
        if parsed['unmatched_anchor_count'] or not parsed['clock_mappings'] or any(not m['valid'] for m in parsed['clock_mappings'].values()):
            issues.append('CLOCK_ANCHOR_MAPPING_FAILED:'+alias)
        for row in selected:
            if not any(p['request_id'] == row['request_id'] and p['kernel_count'] > 0 for p in parsed['phases']):
                issues.append('NO_CORRELATED_REQUEST_KERNEL:'+row['request_id'])
    result = dict(schema=SMOKE_SCHEMA, status='FAIL' if issues else 'PASS', issues=issues,
        mode=plan['mode'], plan=receipt(run/'plan.json'), raw=receipt(run/'raw.json'),
        execution=receipt(execution_path), traces=outputs, hardware_counters='UNAVAILABLE_CURRENT_ERR_NVGPUCTRPERM',
        natural_kernel_activity='UNMEASURED' if plan['mode'] == 'natural' else 'CUPTI_CORRELATED',
        claim='CORRECTNESS_AND_CLOCK_ONLY_NOT_TRAINING_OR_PREDICTOR_QUALIFICATION')
    save(run/'smoke-report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='op',required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output',required=True)
    p.add_argument('--progress-observer',action='store_true')
    p = sub.add_parser('audit'); p.add_argument('--run',required=True)
    p = sub.add_parser('verify-smoke'); p.add_argument('--run',required=True); p.add_argument('--execution',required=True)
    a = parser.parse_args()
    if a.op == 'prepare': result = prepare(a.output,progress_observer=a.progress_observer)
    elif a.op == 'audit': audit(a.run); result = dict(status='CPU_AUDIT_PASS')
    else: result = verify_smoke(a.run,a.execution)
    print(json.dumps(dict(status=result['status'])))


if __name__ == '__main__':
    main()
