#!/usr/bin/env python3
"""Read-only terminal-development service/timeline diagnostics; no fit or GPU.

--check-only reads packet/baseline/source receipts but never progress or run raw.
Analysis reads only explicitly registered terminal run.json files. By default all
packet jobs must be terminal; --partial permits the registered terminal subset.
Failed/ineligible runs retain UNKNOWN requests and never acquire a utility score.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'artifacts/static-study-gpu-20261022/execution-source-v5'
sys.path.insert(0, str(FROZEN))
from gsdeploy.quality import quality_digest
from gsdeploy.static_study import validate_packet

DEFAULT_PACKET = ROOT / 'artifacts/static-spillover-study-20261022/text26-development-v1/experiment-packet.json'
DEFAULT_BASELINE = ROOT / 'artifacts/static-study-gpu-20261022/search-v2-v5-K8/system-config.json'
SEMANTIC_FILES = ['gsdeploy/runtime/protocol.py', 'gsdeploy/runtime/deployment.py',
    'gsdeploy/runtime/runner.py', 'gsdeploy/runtime/transport.py', 'gsdeploy/io.py',
    'gsdeploy/backends/compact_native.py', 'scripts/static_study_worker.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


TERMINAL = {'completed', 'completed_with_failures', 'failed'}


def load_final_json(path, expected_sha256=None):
    """Stable published JSON only; never inspect similarly named partial files."""
    path = Path(path).resolve()
    if path.name != 'run.json':
        raise ValueError('Expected exact final run.json path')
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    fields = ('st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, f) != getattr(after, f) for f in fields) or not payload.endswith(b'\n'):
        raise ValueError('Final run changed during read or lacks complete publication')
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Registered raw SHA256 differs: '+str(path))
    return json.loads(payload), dict(path=str(path), sha256=digest, bytes=len(payload),
        registered_sha256=expected_sha256)


def stats(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not values:
        return dict(count=0, mean=None, min=None, p50=None, p95=None, max=None)
    qs = np.percentile(values, [0, 50, 95, 100])
    return dict(count=len(values), mean=math.fsum(values)/len(values),
                min=float(qs[0]), p50=float(qs[1]), p95=float(qs[2]), max=float(qs[3]))


def corr(a, b):
    if len(a) < 3 or len(a) != len(b) or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def serial_summary(rows):
    rows = sorted(rows, key=lambda r: r['worker_start_unix_ns'])
    ratio = [r['service_over_baseline'] for r in rows]
    by_sample = defaultdict(list)
    for r in rows:
        by_sample[r['sample_id']].append(r['service_over_baseline'])
    medians = {i:float(np.median(v)) for i,v in by_sample.items()}
    residual = [r['service_over_baseline']-medians[r['sample_id']] for r in rows]
    block_count = min(4, len(rows))
    return dict(order='worker_start per path or alias; mixed path summary may interleave concurrent aliases',
        autocorrelation_ratio={str(lag):corr(ratio[:-lag],ratio[lag:]) if len(ratio)>lag else None for lag in [1,2,4,8,32]},
        autocorrelation_same_input_median_residual={str(lag):corr(residual[:-lag],residual[lag:]) if len(residual)>lag else None for lag in [1,2,4,8,32]},
        time_blocks=[dict(index=k, count=len(block), service_ratio=stats([r['service_over_baseline'] for r in block]))
                     for k,block in enumerate(np.array_split(np.array(rows,dtype=object),block_count))] if block_count else [],
        interpretation='Descriptive within-run correlation only. Same-input centering uses this completed raw; not a predictive feature or significance test.')


def inspect_metadata(packet_path, baseline_path):
    packet = json.loads(packet_path.read_text())
    checked = validate_packet(packet)
    if not checked['valid']:
        raise ValueError('Packet frozen bindings changed: '+str(checked['changed']))
    if not packet['jobs'] or any(j['phase']!='calibration' or j['seed'] not in range(51,60) for j in packet['jobs']):
        raise ValueError('Only development calibration seeds51..59 accepted; no holdout or confirmation raw')
    config = json.loads(baseline_path.read_text())
    data = config['catalog']
    if packet['config']['catalog']['quality'] != data['quality']:
        raise ValueError('Baseline and development quality/input contract differ')
    baseline = data['calibrated_replay_v2']['input_baselines']
    missing=[]
    for job in packet['jobs']:
        x=job['plan']['deployment']
        for t,row in job['plan']['routing'].items():
            for m,q in row.items():
                if q<=0:continue
                for sample in data['quality']['input_order'][t]:
                    try:
                        value=baseline[m][t][str(x[m])]['samples'][sample]['median_service_ms']
                        if not math.isfinite(value) or value<=0:raise ValueError('Invalid baseline service')
                    except KeyError:
                        missing.append([m,t,x[m],sample])
    if missing:raise ValueError('Missing input baselines: '+str(missing))
    worker_sources=set()
    for m in {m for job in packet['jobs'] for m in job['plan']['deployment']}:
        worker=packet['workers'][m];command=worker['command']
        script=Path(command[1]).resolve()
        worker_sources.update([script,script.parents[1]/'gsdeploy/runtime/protocol.py'])
        spec=json.loads(Path(worker['spec']).read_text())
        if spec.get('adapter_source'):
            worker_sources.add(Path(spec['adapter_source']).resolve())
        elif '--source-root' in command:
            source=Path(command[command.index('--source-root')+1]).resolve()
            worker_sources.add(source/'gsdeploy/backends/compact_native.py')
    metadata = dict(schema='gsdeploy.completed_service_diagnostic_readiness.v1',
        status='METADATA_READY_RAW_NOT_READ', packet_validation=checked,
        packet=receipt(packet_path), baseline=receipt(baseline_path),
        semantic_source_receipts=[receipt(FROZEN/f) for f in SEMANTIC_FILES],
        actual_worker_semantic_sources=[receipt(p) for p in sorted(worker_sources)],
        tool=receipt(Path(__file__)), jobs=[dict(id=j['id'],labels=j['labels'],seed=j['seed']) for j in packet['jobs']],
        raw_files_read=0, progress_read=False, gpu_started=False,
        note='No claim runs are complete; check-only never opens progress, run.json or partial files.')
    return packet,data,metadata


def completed_paths(packet, progress_paths):
    """Merge explicit registrations, never directory-scan or choose an attempt."""
    if isinstance(progress_paths, (str, Path)):
        progress_paths = [progress_paths]
    declared = {job['id'] for job in packet['jobs']}
    by_id = {}
    for progress_path in progress_paths:
        progress = json.loads(Path(progress_path).read_text())
        for row in progress.get('jobs', []) + progress.get('preserved_attempts', []):
            job_id = row.get('job_id')
            if job_id not in declared:
                raise ValueError('Undeclared progress job: '+str(job_id))
            if row.get('attempt', 1) != 1:
                raise ValueError('Only the predeclared attempt1 is accepted')
            item = dict(row)
            registered_raw = row.get('raw') or {}
            wrapper_error = row.get('status') == 'EXECUTION_ERROR'
            if wrapper_error:
                # The original driver records a process failure and an output
                # directory, not the terminal status of the published raw.
                item['wrapper_status'] = item.pop('status')
            item['_requires_raw_receipt'] = wrapper_error
            item['_has_terminal_receipt'] = bool(registered_raw.get('path')
                and registered_raw.get('sha256') and row.get('status') in TERMINAL)
            raw_path = registered_raw.get('path', row.get('output'))
            if raw_path:
                path = Path(raw_path)
                if not path.is_absolute(): path = ROOT/path
                if wrapper_error and not registered_raw.get('path'):
                    path = path/'run.json'
                item['_path'] = str(path.resolve())
            item['_sha256'] = registered_raw.get('sha256', row.get('sha256'))
            previous = by_id.get(job_id)
            if previous:
                for key in ['_path', '_sha256', 'status', 'eligible', 'exit_code']:
                    if (previous.get(key) is not None and item.get(key) is not None
                            and previous[key] != item[key]):
                        raise ValueError('Conflicting registration; no attempt or U selection: '+job_id)
                for key in ['_requires_raw_receipt', '_has_terminal_receipt']:
                    item[key] = item[key] or previous.get(key, False)
                item = {**previous, **{k:v for k,v in item.items() if v is not None}}
            by_id[job_id] = item
    waiting = []; paths = []
    for job in packet['jobs']:
        row = by_id.get(job['id'])
        if (not row or row.get('status') not in TERMINAL or
                (row.get('_requires_raw_receipt') and not row.get('_has_terminal_receipt'))):
            waiting.append(dict(job_id=job['id'], labels=job['labels'],
                metadata_status=(row.get('status',row.get('wrapper_status')) if row else 'NOT_LISTED'),
                reason='Terminal raw receipt required after wrapper failure' if row and row.get('_requires_raw_receipt') else None))
            continue
        path = Path(row.get('_path', ''))
        if path.name != 'run.json' or not path.is_file():
            waiting.append(dict(job_id=job['id'], metadata_status='FINAL_RUN_NOT_PUBLISHED'))
        else:
            paths.append((job, path, row))
    return waiting, paths


def analyze_run(job, raw, raw_receipt, data, gpu_memory_mib=None):
    if (raw.get('status') not in TERMINAL or raw.get('experiment_phase')!='calibration'
        or raw.get('experiment_job_id')!=job['id'] or raw.get('plan')!=job['plan']):
        raise ValueError('Raw is not the terminal declared development job')
    if raw.get('transport_config',{}).get('random_seed')!=job['seed']:
        raise ValueError('Raw seed differs')
    actual=raw.get('executed_measurement_identity',{})
    if not actual or raw.get('executed_measurement_id')!=quality_digest(actual):
        raise ValueError('Invalid executed measurement identity')
    for key,value in job['identity'].items():
        if key not in {'workers','hardware_contract'} and actual.get(key)!=value:
            raise ValueError('Raw identity differs: '+key)
    records=raw.get('request_records',[])
    if not isinstance(records, list):
        raise ValueError('request_records is not a list')
    window=raw.get('measurement_window') or {}
    utility=(raw.get('evaluation') or {}).get('utility') or {}
    memory=raw.get('memory_evidence') or {}
    stability=(raw.get('evaluation') or {}).get('stability') or {}
    shutdown=raw.get('shutdown') or {}
    cohort_complete=(type(window.get('offered_requests')) is int
                     and len(records)==window['offered_requests'])
    run_limitations=[]
    if not cohort_complete:run_limitations.append('Offered cohort missing or record count differs; unrecorded requests have UNKNOWN service.')
    if 'drain_end_time_ns' not in window:run_limitations.append('Final drain marker absent; observation window is censored.')
    if window.get('aborted'):run_limitations.append('Measurement window aborted; drain timestamp can be an abort marker, not successful drain.')
    if not shutdown or any(z.get('forced') or z.get('exit_code')!=0 or not z.get('bye_received') for z in shutdown.values()):
        run_limitations.append('Native shutdown incomplete or abnormal.')
    if raw.get('errors'):run_limitations.append('Run errors retained; completed-request distributions can have survival bias.')
    if memory.get('resource_valid') is False:
        run_limitations.append('Raw memory resource_valid=false; completed service remains diagnostic evidence, not an eligible deployment candidate.')
    baseline=data['calibrated_replay_v2']['input_baselines']
    x=job['plan']['deployment'];aliases=raw.get('replica_to_family') or {}
    prior_end={};unknown_prelude_aliases=set();unknown_prelude_global=False
    for rec in (raw.get('warmup_request_records') or [])+(raw.get('priming_request_records') or []):
        if not isinstance(rec,dict):
            unknown_prelude_global=True;continue
        alias=rec.get('model');end=(rec.get('worker_response') or {}).get('end_ns')
        if type(end) is int and alias is not None:
            prior_end[alias]=max(prior_end.get(alias,end),end)
        else:
            if alias is None:unknown_prelude_global=True
            else:unknown_prelude_aliases.add(alias)
    if unknown_prelude_global or unknown_prelude_aliases:
        run_limitations.append('Priming/warmup handler end missing; affected FIFO initial state is UNKNOWN.')
    rows=[];unknown=[]
    for record_index,rec in enumerate(records):
        try:
            if rec.get('status')!='completed':raise ValueError('Request status is not completed')
            w=rec['worker_response'];m=aliases[rec['model']];t=rec['task'];sample=rec['pool_id']
            b=baseline[m][t][str(x[m])]['samples'][sample]['median_service_ms']
            required=[rec[k] for k in ['scheduled_time_ns','enqueued_time_ns','actual_submit_time_ns','actual_submit_unix_ns','submit_write_completed_time_ns','completion_time_ns','completion_unix_ns']]
            if not all(type(v) is int for v in required):raise ValueError('Missing timestamp')
            if w.get('request_id')!=rec['request_id'] or w.get('pool_id')!=sample or w.get('submit_ns')!=rec['actual_submit_unix_ns']:
                raise ValueError('Worker request mapping differs')
            service=w.get('service_ms');handler=rec.get('worker_latency_ms');queue=rec.get('worker_queue_wait_ms')
            if not all(type(v) in (int,float) and math.isfinite(v) for v in [service,handler,queue]) or service<=0 or handler<=0:
                raise ValueError('Missing valid service/handler/wait duration')
            scheduled,enqueued,submit,submitunix,write,complete,completeunix=required
            start,end=w['start_ns'],w['end_ns']
            if type(start) is not int or type(end) is not int or end<=start:
                raise ValueError('Missing or invalid worker timestamps')
            if type(rec['arrival_index']) is not int:raise ValueError('Invalid arrival_index')
            host_dispatch=(enqueued-scheduled)/1e6;host_send=(submit-enqueued)/1e6
            schedule_delay=(submit-scheduled)/1e6;host_write=(write-submit)/1e6
            worker_interval=(end-start)/1e6;worker_wait=(start-submitunix)/1e6
            response_delay=(completeunix-end)/1e6;observed_e2e=(complete-scheduled)/1e6
            baseline_entry=baseline[m][t][str(x[m])]['samples'][sample]
            decomposition=schedule_delay+worker_wait+worker_interval+response_delay
            rows.append(dict(record_index=record_index,diagnostic_status='ANALYZED_COMPLETED',request_id=rec['request_id'],arrival_index=rec['arrival_index'],task=t,family=m,alias=rec['model'],replicas=x[m],sample_id=sample,
                request_sha256=w.get('request_sha256'),protocol_sha256=w.get('protocol_sha256'),raw_baseline_ms=b,
                baseline_input_identity={k:v for k,v in baseline_entry.items() if k!='median_service_ms'},
                baseline_equivalence_scope='Historical matched pool_id/n-state; current request/protocol equivalence is not asserted by these descriptive ratios.',
                service_ms=service,worker_handler_ms=handler,service_over_baseline=service/b,handler_over_baseline=handler/b,
                handler_minus_service_ms=handler-service,input_tokens=w.get('input_tokens'),generated_tokens=w.get('generated_tokens'),
                timestamps=dict(scheduled_mono_ns=scheduled,enqueued_mono_ns=enqueued,submit_mono_ns=submit,
                    submit_unix_ns=submitunix,write_done_mono_ns=write,complete_mono_ns=complete,complete_unix_ns=completeunix),
                worker_start_unix_ns=start,worker_end_unix_ns=end,
                ledger=dict(host_schedule_to_enqueue_ms=host_dispatch,host_enqueue_to_submit_ms=host_send,
                    host_submit_write_ms=host_write,schedule_to_submit_ms=schedule_delay,
                    submit_to_worker_start_ms=worker_wait,worker_handler_interval_ms=worker_interval,
                    worker_response_to_host_receive_ms=response_delay,observed_e2e_ms=observed_e2e,
                    observed_runtime_ms=(complete-submit)/1e6,
                    handler_field_residual_ms=handler-worker_interval,
                    queue_field_residual_ms=queue-worker_wait,
                    duration_decomposition_residual_ms=observed_e2e-decomposition,
                    reported_e2e_residual_ms=observed_e2e-rec.get('e2e_latency_ms',observed_e2e),
                    reported_runtime_residual_ms=(complete-submit)/1e6-rec.get('runtime_latency_ms',(complete-submit)/1e6)),
                original_record_sha256=quality_digest(rec),original_record=rec))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            unknown.append(dict(record_index=record_index,diagnostic_status='UNKNOWN', reason=str(exc),
                request_id=rec.get('request_id') if isinstance(rec,dict) else None,
                alias=rec.get('model') if isinstance(rec,dict) else None,
                original_record_sha256=quality_digest(rec), original_record=rec,
                service_ms=None, ledger=None, fifo_diagnostic=None))
    # Unknown work on an alias invalidates a complete FIFO recurrence. Keep its
    # valid completed ledgers but do not silently bridge the unobserved service.
    global_censor=(not cohort_complete or 'drain_end_time_ns' not in window or window.get('aborted')
                   or unknown_prelude_global or any(r['alias'] is None for r in unknown))
    censored_aliases={r['alias'] for r in unknown}|unknown_prelude_aliases
    if unknown:run_limitations.append('Service statistics use analyzable completed requests only; UNKNOWN requests are retained, not treated as zero.')
    by_alias=defaultdict(list)
    for row in rows:by_alias[row['alias']].append(row)
    replay_summary={}
    for alias,seq in by_alias.items():
        seq.sort(key=lambda r:(r['timestamps']['submit_mono_ns'],r['arrival_index']))
        if global_censor or alias in censored_aliases:
            for row in seq:
                row['fifo_diagnostic']=dict(status='UNKNOWN_CENSORED',
                    reason='Missing/unknown alias work or incomplete observation window; no gap or full-trace residual attribution.')
            replay_summary[alias]=dict(status='UNKNOWN_CENSORED', analyzed_completed_requests=len(seq),
                unknown_requests=sum(r['alias']==alias for r in unknown),
                full_trace_valid=False, one_step_unobserved_gap_ms=stats([]),
                handler_fifo_end_residual_ms=stats([]),service_fifo_end_residual_ms=stats([]))
            continue
        # Unix arrivals are actual host pre-write submission, not receive times.
        # Zero-gap recurrences intentionally omit parse/emit/transport gaps.
        origin=min(r['timestamps']['submit_unix_ns'] for r in seq)
        prev=prior_end.get(alias,origin);handler_clock=(prev-origin)/1e6;service_clock=handler_clock
        starts=[];one_step=[];prev_arrival=-1;order_failures=[]
        for row in seq:
            a=(row['timestamps']['submit_unix_ns']-origin)/1e6
            observed_start=(row['worker_start_unix_ns']-origin)/1e6
            observed_end=(row['worker_end_unix_ns']-origin)/1e6
            one_step_start=max(row['timestamps']['submit_unix_ns'],prev)
            gap=(row['worker_start_unix_ns']-one_step_start)/1e6
            handler_clock=max(a,handler_clock)+row['worker_handler_ms']
            service_clock=max(a,service_clock)+row['service_ms']
            if row['arrival_index']<=prev_arrival:order_failures.append(row['request_id'])
            if row['worker_start_unix_ns']<prev:order_failures.append(row['request_id']+':HANDLER_OVERLAP_OR_WALL_CLOCK_CHANGE')
            prev_arrival=row['arrival_index'];prev=row['worker_end_unix_ns']
            row['fifo_diagnostic']=dict(observed_start_minus_max_submit_previous_observed_end_ms=gap,
                actual_submit_handler_fifo_end_ms_relative_origin=handler_clock,
                actual_submit_service_fifo_end_ms_relative_origin=service_clock,
                observed_worker_end_ms_relative_origin=observed_end,
                handler_fifo_observed_minus_replay_end_ms=observed_end-handler_clock,
                service_fifo_observed_minus_replay_end_ms=observed_end-service_clock,
                interpretation='Oracle completed-service diagnostics only; pre-write submit includes unknown transport. One-step gap includes previous emit, input read/parse, OS scheduling and IPC; exact arrival at handler is unobserved.')
            starts.append(gap);one_step.append(observed_start)
        replay_summary[alias]=dict(status='COMPLETED_TRACE_DIAGNOSTIC',full_trace_valid=True,initial_previous_handler_end_unix_ns=prior_end.get(alias),
            initial_policy='last completed priming/warmup handler end; otherwise first host submit',
            oracle_fifo_origin_unix_ns=origin,
            start_order_failures=order_failures,
            one_step_unobserved_gap_ms=stats(starts),
            handler_fifo_end_residual_ms=stats([r['fifo_diagnostic']['handler_fifo_observed_minus_replay_end_ms'] for r in seq]),
            service_fifo_end_residual_ms=stats([r['fifo_diagnostic']['service_fifo_observed_minus_replay_end_ms'] for r in seq]),
            service_ms=stats([r['service_ms'] for r in seq]),handler_ms=stats([r['worker_handler_ms'] for r in seq]),
            service_over_baseline=stats([r['service_over_baseline'] for r in seq]),handler_over_baseline=stats([r['handler_over_baseline'] for r in seq]),
            per_task_counts=dict(Counter(r['task'] for r in seq)),
            mixed_task_warning='Alias-wide ratio can mix tasks; use per-path/input summaries for interpretation.',
            service_serial_correlation=serial_summary(seq))
    for alias in censored_aliases-set(replay_summary):
        if alias is not None:
            replay_summary[alias]=dict(status='UNKNOWN_CENSORED',analyzed_completed_requests=0,
                unknown_requests=sum(r['alias']==alias for r in unknown),full_trace_valid=False,
                one_step_unobserved_gap_ms=stats([]),handler_fifo_end_residual_ms=stats([]),
                service_fifo_end_residual_ms=stats([]))
    groups=defaultdict(list);inputgroups=defaultdict(list)
    for r in rows:
        groups[(r['family'],r['task'],r['replicas'])].append(r)
        inputgroups[(r['family'],r['task'],r['replicas'],r['alias'],r['sample_id'])].append(r)
    path_rows=[];input_rows=[]
    for (m,t,n),seq in sorted(groups.items()):
        samples=Counter(r['sample_id'] for r in seq);expected=set(data['quality']['input_order'][t])
        indexseq=sorted(r['arrival_index'] for r in seq)
        token_corr={}
        for field in ['input_tokens','generated_tokens']:
            selected=[r for r in seq if type(r[field]) in (int,float)]
            token_corr[field]=dict(count=len(selected),pearson_with_service=corr([r[field] for r in selected],[r['service_ms'] for r in selected]))
        path_rows.append(dict(family=m,task=t,replicas=n,requests=len(seq),aliases=sorted({r['alias'] for r in seq}),
            unique_inputs=len(samples),expected_pool_size=len(expected),missing_inputs=sorted(expected-set(samples)),sample_counts=dict(samples),
            service_ms=stats([r['service_ms'] for r in seq]),handler_ms=stats([r['worker_handler_ms'] for r in seq]),
            baseline_service_ms_for_routed_requests=stats([r['raw_baseline_ms'] for r in seq]),
            service_over_baseline=stats([r['service_over_baseline'] for r in seq]),handler_over_baseline=stats([r['handler_over_baseline'] for r in seq]),
            ratio_of_service_sums=sum(r['service_ms'] for r in seq)/sum(r['raw_baseline_ms'] for r in seq),
            ratio_of_handler_sums=sum(r['worker_handler_ms'] for r in seq)/sum(r['raw_baseline_ms'] for r in seq),
            handler_minus_service_ms=stats([r['handler_minus_service_ms'] for r in seq]),token_correlations=token_corr,
            correlation=serial_summary(seq),periodic_coverage=dict(global_arrival_index_gaps=stats(np.diff(indexseq).tolist()),
                note='Global arrival gaps include other tasks. Use task-sequence alias/sample residue table below for periodic routing, not these gaps as route period.')))
    # Task sequence positions are independent of other task arrivals.
    task_sequences={}
    sequence_records=[r for r in records if isinstance(r,dict) and
        all(type(r.get(k)) is str for k in ['task','model','pool_id']) and type(r.get('arrival_index')) is int]
    for t in sorted({r['task'] for r in sequence_records}):
        if t not in data['quality']['input_order']:continue
        seq=[dict(alias=r['model'],sample_id=r['pool_id'],request_status=r.get('status'))
            for r in sorted(sequence_records,key=lambda r:r['arrival_index']) if r['task']==t]
        pool=len(data['quality']['input_order'][t])
        table={alias:defaultdict(set) for alias in sorted({r['alias'] for r in seq})}
        for k,r in enumerate(seq):
            table[r['alias']][k%pool].add(r['sample_id'])
        task_sequences[t]=dict(pool_size=pool,task_requests=len(seq),
            includes_unknown_requests=True, sequence_record_coverage_complete=len(sequence_records)==len(records) and cohort_complete,
            request_status_sequence=[r['request_status'] for r in seq],
            alias_sequence=[r['alias'] for r in seq],sample_sequence=[r['sample_id'] for r in seq],
            alias_task_position_residues_mod_pool={a:{str(k):sorted(v) for k,v in tab.items()} for a,tab in table.items()},
            caveat='Fixed pool shuffle repeats each pool cycle; deterministic alias routing can repeatedly send only a subset to a family. Coverage is evidence, not a quality-adjustment permission.')
    for (m,t,n,alias,sample),seq in sorted(inputgroups.items()):
        input_rows.append(dict(family=m,task=t,replicas=n,alias=alias,sample_id=sample,observations=len(seq),raw_baseline_ms=seq[0]['raw_baseline_ms'],
            service_ms=stats([r['service_ms'] for r in seq]),handler_ms=stats([r['worker_handler_ms'] for r in seq]),
            service_ratio=stats([r['service_over_baseline'] for r in seq]),handler_ratio=stats([r['handler_over_baseline'] for r in seq])))
    ledger_keys=rows[0]['ledger'].keys() if rows else []
    ledger_summary={key:stats([r['ledger'][key] for r in rows]) for key in ledger_keys}
    negative_flags={key:[r['request_id'] for r in rows if r['ledger'][key]<-1e-6] for key in ['host_schedule_to_enqueue_ms','host_enqueue_to_submit_ms','submit_to_worker_start_ms','worker_response_to_host_receive_ms']}
    return dict(job_id=job['id'],labels=job['labels'],raw=raw_receipt,seed=job['seed'],plan=job['plan'],
        terminal_status=raw['status'],original_eligible=utility.get('eligible'),
        eligibility_source='raw.evaluation.utility.eligible; never inferred from terminal status',
        resource_evidence=dict(source='raw.memory_evidence and raw.evaluation.utility.M_norm; not rescored',
            resource_valid=memory.get('resource_valid'),safe_memory_mib=memory.get('safe_memory_mib'),
            gpu_memory_mib=gpu_memory_mib,original_M_norm=utility.get('M_norm'),scope=memory.get('scope')),
        original_stability_state=stability.get('stability_state_v2'),
        diagnostic_U=None,ranking_performed=False,errors=raw.get('errors',[]),shutdown=shutdown,
        measurement_window=window,cohort=dict(offered_requests=window.get('offered_requests'),
            recorded_requests=len(records),analyzed_completed_requests=len(rows),unknown_requests=len(unknown),
            unrecorded_requests=(max(0,window['offered_requests']-len(records)) if type(window.get('offered_requests')) is int else None),
            record_count_matches_offered=cohort_complete),censoring_limitations=run_limitations,
        requests=sorted(rows+unknown,key=lambda r:r['record_index']),paths=path_rows,same_input_alias_groups=input_rows,task_sequences=task_sequences,
        fifo_by_alias=replay_summary,ledger_summary=ledger_summary,negative_time_flags=negative_flags,
        status='TERMINAL_DEVELOPMENT_DIAGNOSTIC_NO_MODEL_FIT',production_U_recomputed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,default=DEFAULT_PACKET)
    parser.add_argument('--baseline-config',type=Path,default=DEFAULT_BASELINE)
    parser.add_argument('--queue-progress',type=Path,nargs='+',action='append',
        help='Explicit receipt(s); may repeat. No directory scanning or latest-attempt choice.')
    parser.add_argument('--partial',action='store_true',help='Analyze only registered terminal published jobs; retain waiting roster.')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    packet,data,metadata=inspect_metadata(args.packet.resolve(),args.baseline_config.resolve())
    if args.check_only:
        print(json.dumps(metadata,ensure_ascii=False));return
    if args.queue_progress is None or args.output is None:
        parser.error('Analysis requires --queue-progress and new --output; check-only requires neither')
    progress_paths=[p.resolve() for group in args.queue_progress for p in group]
    metadata['progress_receipts']=[receipt(p) for p in progress_paths]
    waiting,paths=completed_paths(packet,progress_paths)
    if (waiting and not args.partial) or not paths:
        print(json.dumps(dict(status='WAITING_REGISTERED_TERMINAL_JOBS',waiting=waiting,
            raw_files_read=0,gpu_started=False),ensure_ascii=False));raise SystemExit(3)
    if args.output.exists():raise FileExistsError('Output already exists')
    metadata.update(partial_requested=args.partial,waiting=waiting)
    # Only now may final published files be opened.
    reports=[]
    for job,path,registration in paths:
        raw,source=load_final_json(path,registration.get('_sha256'))
        if raw.get('status')!=registration['status']:
            raise ValueError('Raw terminal status differs from registration')
        eligible=((raw.get('evaluation') or {}).get('utility') or {}).get('eligible')
        if registration.get('eligible') is not None and registration['eligible']!=eligible:
            raise ValueError('Raw eligibility differs from registration')
        result=analyze_run(job,raw,source,data,packet['config']['gpu_memory_mib'])
        result['registration']={k:v for k,v in registration.items() if not k.startswith('_') and k!='U'}
        reports.append(result)
    for item in [metadata['packet'],metadata['baseline'],*metadata['progress_receipts'],metadata['tool'],
                 *metadata['semantic_source_receipts'],*metadata['actual_worker_semantic_sources'],
                 *[r['raw'] for r in reports]]:
        if sha(item['path'])!=item['sha256']:
            raise ValueError('Diagnostic source changed during analysis: '+item['path'])
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (args.output/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    metadata.update(status='COMPLETED_DIAGNOSTIC_ONLY',raw_files_read=len(reports),progress_read=True)
    save('protocol.json',dict(metadata=metadata,
        semantics=dict(service_ms='Native ProfileAdapter inner perf_counter interval: infer, post-infer CUDA sync/cache reclaim; excludes initial pre-start CUDA sync. Retained PP-OCR adapter has a separate interval around input hash/image load and det/rec pipeline; no generic cache-reclaim assumption.',
            worker_latency_ms='Wall-clock interval around complete handler; includes initial sync and wrapper overhead.',
            worker_queue_wait_ms='Worker start wallclock minus host pre-write submit wallclock; includes IPC/read/parse, previous-handler waiting and previous emit gap, not a pure queue measurement.',
            host_submit='Sender timestamp before stdin JSON write/flush; write completion is separate and may occur after worker starts.',
            completion='Host reader timestamp after stdout line parsing when event enters queue; includes emit/JSON/stdout/host parsing and reader scheduling.',
            clocks='Host scheduled/submit/completion are monotonic; worker/submit_unix/completion_unix are wallclock. Within-request span mismatch recorded; no absolute mixing of clock domains.',
            fifo='Actual host submit with observed service or handler duration, per alias, excludes unknown emit/read/parse gaps; residual is reported, never forced to0.',
            actual_overlap='Not used or fitted; future observed data never become predictor features.'),
        no_model_fit=True,no_scoring_or_packet_changes=True,no_gpu=True,
        limitations=['Oracle completed-service FIFO is an accounting diagnostic, not a deployable predictor.',
                    'Failed/incomplete requests retain UNKNOWN service; completed-only distributions are censored and can have survival bias.',
                    'FIFO is suppressed for aliases with unknown work or missing cohort/drain evidence; no unobserved service is assigned to IPC.',
                    'Exact worker line-receive/parse timestamps and handler service-start timestamp are absent.',
                    'Wallclock changes/cross-process offsets can affect worker waits and ledger; inspect residuals/negative flags.',
                    'Temporal/correlation statistics are descriptive, clustered within one run; no causal interference claim.']))
    overview=[]
    for k,report in enumerate(reports):
        name=f'job-{k+1}-{report["job_id"][:12]}.json';save(name,report)
        overview.append({key:report[key] for key in ['job_id','labels','raw','paths','fifo_by_alias','ledger_summary','negative_time_flags','status','terminal_status','original_eligible','eligibility_source','resource_evidence','original_stability_state','diagnostic_U','ranking_performed','cohort','censoring_limitations']})
    save('report.json',dict(schema='gsdeploy.terminal_service_diagnostics.v2',jobs=overview,
        raw_files_read=len(reports),waiting=waiting,partial_requested=args.partial,diagnostic_U=None,ranking_performed=False,fitted_parameters=False,gpu_started=False))
    save('manifest.json',dict(files={p.name:sha(p) for p in args.output.iterdir() if p.is_file()},
        inputs=[metadata['packet'],metadata['baseline'],*metadata['progress_receipts'],metadata['tool'],*[r['raw'] for r in reports]],
        source_receipts=metadata['semantic_source_receipts']+metadata['actual_worker_semantic_sources'],tool=receipt(Path(__file__)),gpu_started=False))
    print(json.dumps(dict(status='COMPLETED_DIAGNOSTIC_ONLY',jobs=len(reports),output=str(args.output.resolve()),gpu_started=False)))


if __name__=='__main__':
    main()
