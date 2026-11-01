"""Independent failed sustained diagnostic audit; verified failure evidence is not cohort PASS."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import importlib.util
import json
import math
import statistics
from pathlib import Path
import sys
import types
import traceback

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def runtime_protocol_digest(spec):
    """Native CompactAdapter hashes the current merged JSON object canonically.

    Historical quality paths separately retain their original task protocol
    identity. Byte hashes and those historical hashes are different contracts.
    """
    path=spec['resources'].get('protocol')
    return digest(json.loads(Path(path).read_text())) if path else None


def check_response_protocol(response,spec,request_id):
    if 'protocol_sha256' not in response:return
    expected=runtime_protocol_digest(spec)
    if expected is None or response['protocol_sha256']!=expected:
        raise AssertionError('runtime_worker_protocol_canonical_sha256: '+request_id)


def integrate(samples, begin, end):
    points = sorted((s['time_ns'], s['watts']) for s in samples)
    assert points[0][0] <= begin < end <= points[-1][0]
    total = 0.0
    for (a, p), (b, q) in zip(points, points[1:]):
        left, right = max(begin, a), min(end, b)
        if right <= left:
            continue
        assert 0 < b-a <= 2_000_000_000
        assert math.isfinite(p) and math.isfinite(q) and min(p, q) >= 0
        lp = p+(q-p)*(left-a)/(b-a)
        rp = p+(q-p)*(right-a)/(b-a)
        total += (lp+rp)/2*(right-left)/1e9
    return total


def classify_measurement(job, measured, window):
    """Final records are not retroactive proof of completion at the deadline."""
    aliases=job['active_aliases']; primary=[r for r in measured if r['role']=='primary']; support=[r for r in measured if r['role']=='support']
    assert len(primary)==96*len(aliases)
    assert window['primary_completion_time_ns'] is None and window['support_admission_stop_time_ns'] is None
    assert window['duration_seconds']>=600
    assert math.isclose(window['duration_seconds'],(window['end_time_ns']-window['start_time_ns'])/1e9,abs_tol=1e-9)
    planned={}; classes={};counts=defaultdict(Counter)
    offset=window['start_unix_ns']-window['start_time_ns']
    deadline=window['start_time_ns']+600_000_000_000
    for alias in aliases:
        rows=[r for r in primary if r['model']==alias];extras=[r for r in support if r['model']==alias]
        templates=job['planned_requests'][alias]; candidates=job['support_requests'][alias]
        assert len(rows)==len(templates)==96 and len(candidates)==1024
        assert len(extras)<=1024
        for row,template in zip(rows,templates):
            assert all(row[k]==v for k,v in template.items());planned[row['request_id']]=template
        for row,template in zip(extras,candidates):
            assert all(row[k]==v for k,v in template.items());planned[row['request_id']]=template
            assert row['scheduled_time_ns']>=max(r['completion_time_ns'] for r in rows if r['status']=='completed')
        released=[r for r in rows if r['scheduled_time_ns'] is not None]
        assert [r['sequence'] for r in released]==list(range(len(released)))
        assert window['primary_released_by_alias'][alias]==len(released)
        assert window['support_released_by_alias'][alias]==len(extras)
        assert window['released_by_alias'][alias]==len(released)+len(extras)
        sequence=released+extras
        for before,after in zip(sequence,sequence[1:]):
            assert before['status']=='completed' and before['completion_time_ns']<=after['scheduled_time_ns']
        for row in rows+extras:
            if row['scheduled_time_ns'] is None:
                assert row['role']=='primary' and row['status']=='incomplete' and row.get('release_state')=='not_released'
                assert row['completion_time_ns'] is None and not row.get('worker_response')
                label='UNRELEASED_UNKNOWN'
            else:
                assert row['scheduled_time_ns']<window['end_time_ns'] and row['status']=='completed'
                host=row['completion_time_ns'];native=row['worker_response']['end_ns']-offset
                # Keep native and host clocks separately; their offset estimate is not an exact clock transform.
                label='WITHIN_DECLARED_600S' if max(host,native)<=deadline else 'AFTER_DECLARED_600S'
            classes[row['request_id']]=label;counts[(alias,row['role'])][label]+=1
        in_primary=[r for r in rows if classes[r['request_id']]=='WITHIN_DECLARED_600S']
        observed=window['primary_completed_by_alias'][alias]
        assert observed==len(in_primary)
        if observed==96:assert window['alias_completion_time_ns'][alias]==max(r['completion_time_ns'] for r in rows)
        else:assert alias not in window['alias_completion_time_ns']
    assert len(planned)==len(measured)==len({r['request_id'] for r in measured})
    return primary,support,planned,classes,[dict(alias=a,role=role,counts=dict(c)) for (a,role),c in counts.items()]


def audit_job(packet,packet_path,registration,job,queue_bytes,queue):
    raw_path=Path(registration['raw']['path'])
    assert sha(raw_path)==registration['raw']['sha256']
    raw = json.loads(raw_path.read_text())
    assert raw['schema'] == 'gsdeploy.sustained_joint_capacity.raw.v1'
    assert raw['status'] == raw['status_before_shutdown'] == 'failed'
    assert raw['errors']==['TimeoutError: measurement wall timeout, including support drain']
    assert raw['packet_sha256'] == sha(packet_path)
    assert raw['job'] == job
    assert digest({k:v for k,v in job.items() if k!='job_identity_sha256'}) == job['job_identity_sha256'] == raw['job_identity_sha256']
    assert digest(packet['bindings']) == packet['bindings_sha256']
    for path, expected in packet['bindings'].items():
        assert sha(path) == expected, path
    config = json.loads(Path(packet['system_config']).read_text())
    workers = json.loads(Path(packet['workers']).read_text())
    admission = json.loads(Path(packet['admission']).read_text())
    assert raw['config'] == config
    assert raw['replica_to_family'] == packet['aliases']
    assert raw['plan']['deployment'] == packet['resident_deployment']
    assert raw['preflight_plan'] == packet['preflight_plan']
    expected_transport=dict(deployment={a:1 for a in packet['aliases']},workload_rps={'text':1},
        routing_rps={'text':{a:1/len(job['active_aliases']) for a in job['active_aliases']}},
        warmup_seconds=30,measurement_seconds=600,arrival_process='deterministic_periodic',
        random_seed=job['seed'],routing_source=packet['protocol'],startup_timeout_seconds=300,
        drain_timeout_seconds=300,shutdown_timeout_seconds=10,max_pending_per_model=1,idle_residency=False)
    assert raw['transport_config'] == expected_transport
    pre = raw['preflight']
    assert pre['ready'] is True and pre['gpu_checked'] is True
    assert pre['gpu']['uuid'] == packet['gpu_uuid']
    assert pre['sample_pool_sha256'] == sha(packet['samples'])
    for boundary in ['gpu_boundary_preflight', 'gpu_before_worker_start']:
        assert raw[boundary]['uuid'] == packet['gpu_uuid'] and raw[boundary]['compute_pids'] == []
    for evidence in [pre['admission_evidence'], raw['priming_path_identity']]:
        assert evidence['verified'] is True
        assert evidence['manifest_sha256'] == sha(packet['admission'])
        assert evidence['quality_sha256'] == admission['quality_sha256'] == digest(config['catalog']['quality'])
        for family in packet['resident_deployment']:
            assert evidence['checkpoints'][family] == admission['models'][family]['checkpoint']['sha256']
    expected_pairs = {('text','text'),('generalist_2','text'),('generalist_2','image'),('generalist_2','asr'),('generalist_2','ocr'),('ocr','ocr')}
    assert {(p['model'],p['task']) for p in raw['priming_path_identity']['pairs']} == expected_pairs
    for pair in raw['priming_path_identity']['pairs']:
        assert pair['measurement'] == config['catalog']['quality']['paths'][pair['model']][pair['task']]
        assert pair['runtime_checkpoint_sha256'] == admission['models'][pair['model']]['checkpoint']['sha256']
    runtime_specs={}
    for alias, family in packet['aliases'].items():
        entry = workers[family]
        spec = json.loads(Path(entry['spec']).read_text())
        runtime_specs[family]=spec
        ready = raw['worker_ready'][alias]
        assert ready['worker'] == family == spec['candidate']['slot']
        assert ready['model'] == spec['candidate']['model']
        assert ready['spec_sha256'] == sha(entry['spec']) == entry['spec_sha256']
        assert set(ready['tasks']) == set(config['catalog']['models'][family]['supported_tasks'])
        assert type(ready['pid']) is int and ready['pid'] > 0
        observed = raw['worker_registry'][alias]
        assert observed['command'] == entry['command']
        assert observed['cwd'] == entry['cwd']
        assert {k:v for k,v in observed.items() if k!='env'} == {k:v for k,v in entry.items() if k!='env'}
        assert {k:v for k,v in observed['env'].items() if k!='PYTHONPATH'} == {k:v for k,v in entry['env'].items() if k!='PYTHONPATH'}
        assert observed['env']['PYTHONPATH'].split(':')[0] == packet['frozen_source']
        assert all(observed['env'].get(key,'1') == '1' for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
    assert len({r['pid'] for r in raw['worker_ready'].values()}) == 4
    assert set(raw['cpu_launcher_checks']) == set(packet['resident_deployment'])
    pool_rows = [json.loads(line) for line in Path(packet['samples']).read_text().splitlines() if line.strip()]
    pool = {r['pool_id']:r for r in pool_rows}
    for family, result in raw['cpu_launcher_checks'].items():
        command = ['--check' if arg=='--execute' else arg for arg in workers[family]['command']] + ['--pool',packet['samples']]
        assert result['command'] == command and result['exit_code'] == 0
        response = json.loads(result['stdout'])
        tasks = config['catalog']['models'][family]['supported_tasks']
        assert set(response['tasks']) == set(tasks)
        assert response['gpu_started'] is False and response['status'] == 'CPU_READY_GPU_UNVERIFIED'
        assert response['inputs_checked'] == sum(r['task_type'] in tasks for r in pool_rows)
    priming = raw['priming_request_records']
    warmup = raw['warmup_request_records']
    measured_all = raw['request_records']
    measured = [r for r in measured_all if r['status']=='completed']
    assert [{k:r[k] for k in ['model','task','pool_id']} for r in priming] == packet['priming_requests']
    expected_count = 96*len(job['active_aliases'])
    assert len(priming) == 21
    primary, support, planned, cutoff_classes, cutoff_counts = classify_measurement(job, measured_all, raw['measurement_window'])
    assert len(primary) == expected_count
    ids = [r['request_id'] for r in priming+warmup+measured]
    assert len(set(ids)) == len(ids)
    phase_counts = {}
    max_handler_ledger_residual = 0.0
    for phase, rows in [('priming',priming),('warmup',warmup),('measurement',measured)]:
        phase_counts[phase] = dict(Counter(r['model'] for r in rows))
        by_alias = defaultdict(list)
        for row in rows:
            assert row['phase'] == phase and row['status'] == 'completed'
            response = row['worker_response']
            assert response['success'] is True
            for key in ['request_id','pool_id','arrival_index']:
                assert response[key] == row[key]
            assert response['task_type'] == response['task'] == row['task']
            assert response['worker'] == packet['aliases'][row['model']]
            family=packet['aliases'][row['model']]
            check_response_protocol(response,runtime_specs[family],row['request_id'])
            assert pool[row['pool_id']]['task_type'] == row['task']
            assert response['submit_ns'] == row['actual_submit_unix_ns']
            assert row['scheduled_time_ns'] <= row['enqueued_time_ns'] <= row['actual_submit_time_ns'] <= row['completion_time_ns']
            assert row['completion_time_ns']-row['scheduled_time_ns'] < 300_000_000_000
            assert row['actual_submit_time_ns'] <= row['submit_write_completed_time_ns'] <= row['completion_time_ns']
            assert response['submit_ns'] <= response['start_ns'] < response['end_ns'] <= row['completion_unix_ns']
            handler = (response['end_ns']-response['start_ns'])/1e6
            max_handler_ledger_residual = max(max_handler_ledger_residual,abs(handler-response['worker_latency_ms']))
            assert math.isclose(handler, response['worker_latency_ms'],abs_tol=1e-9)
            assert 0 < response['service_ms'] <= handler
            assert math.isclose((response['start_ns']-response['submit_ns'])/1e6,response['queue_wait_ms'],abs_tol=1e-9)
            by_alias[row['model']].append(row)
            if phase == 'measurement':
                assert all(row[key] == value for key,value in planned[row['request_id']].items())
        for alias, seq in by_alias.items():
            for before, after in zip(seq,seq[1:]):
                assert after['scheduled_time_ns'] >= before['completion_time_ns']
                assert after['worker_response']['start_ns'] >= before['worker_response']['end_ns']
    for before, after in zip(priming,priming[1:]):
        assert after['scheduled_time_ns'] >= before['completion_time_ns']
    window = raw['measurement_window']; warm_window = raw['warmup_window']
    assert warm_window['duration_seconds'] >= 30
    assert max(r['completion_time_ns'] for r in priming) <= warm_window['start_time_ns']
    assert max(r['completion_time_ns'] for r in warmup) <= warm_window['end_time_ns'] <= window['start_time_ns']
    assert set(r['model'] for r in warmup) == set(job['active_aliases'])
    assert min(r['scheduled_time_ns'] for r in measured) == window['synchronous_release_time_ns'] == window['start_time_ns']
    assert window['max_outstanding_per_alias']==1
    for alias in job['active_aliases']:
        rows=[r for r in primary if r['model']==alias]
        assert [r['pool_id'] for r in rows]==job['input_pool_ids']
    assert set(raw['shutdown']) == set(packet['aliases'])
    assert all(r['exit_code']==0 and not r['forced'] and r['bye_received'] for r in raw['shutdown'].values())
    memory = raw['memory_evidence']; memory_samples = memory['samples']
    assert memory['measured'] is True and memory['resource_valid'] is True
    assert memory_samples and all(not row['foreign_pids'] for row in memory_samples)
    safe_memory = config['catalog']['safe_factor']*max(r['owned_used_mib'] for r in memory_samples)
    assert math.isclose(safe_memory,memory['safe_memory_mib'],abs_tol=1e-7)
    assert safe_memory <= memory['budget_mib'] == config['gpu_memory_mib'] == packet['gpu_memory_mib']
    offset = window['start_unix_ns']-window['start_time_ns']
    assert min(r['unix_ns'] for r in memory_samples) <= window['start_unix_ns']
    memory_end_gap=max(0,window['end_time_ns']+offset-max(r['unix_ns'] for r in memory_samples))/1e9
    memory_window_coverage=memory_end_gap==0
    energy = raw['energy_evidence']
    assert energy['scope']=='whole_gpu_board' and energy['coverage_valid'] is True and not energy['sampling_errors']
    joules = integrate(energy['samples'],window['start_time_ns'],window['end_time_ns'])
    startup_joules = integrate(energy['samples'],energy['startup_begin_time_ns'],window['start_time_ns'])
    assert math.isclose(joules,energy['run_joules'],rel_tol=1e-12)
    assert math.isclose(startup_joules,energy['startup_joules'],rel_tol=1e-12)
    package = types.ModuleType('_capacity_sustained_independent_frozen')
    package.__path__ = [str(Path(packet['frozen_source'])/'gsdeploy')]
    sys.modules[package.__name__] = package
    tasks = importlib.import_module(package.__name__+'.tasks')
    quality = importlib.import_module(package.__name__+'.quality')
    reference = config['catalog']['quality']['references']
    scores = []; totals = defaultdict(lambda:[0,0]); support_totals = defaultdict(lambda:[0,0]); tail_totals = defaultdict(lambda:[0,0])
    for row in measured:
        result = dict(row['worker_response'])
        score = tasks.score_result(reference['text'][row['pool_id']],result)
        a,b = quality.sample_totals('text',score)
        role_totals = tail_totals if cutoff_classes[row['request_id']]=='AFTER_DECLARED_600S' else (totals if row['role']=='primary' else support_totals)
        role_totals[row['model']][0] += a;role_totals[row['model']][1] += b
        scores.append(dict(request_id=row['request_id'],alias=row['model'],pool_id=row['pool_id'],repetition=row['repetition'],role=row['role'],cutoff_class=cutoff_classes[row['request_id']],
            host_completion_time_ns=row['completion_time_ns'],native_end_unix_ns=result['end_ns'],native_end_perf_estimate_ns=result['end_ns']-offset,
            within_actual_loop_window_host=row['completion_time_ns']<=window['end_time_ns'],within_actual_loop_window_native=result['end_ns']-offset<=window['end_time_ns'],
            score=score,error_numerator=a,reference_denominator=b))
    utility = raw['evaluation']['utility']
    assert utility['U'] is None and utility['eligible'] is False
    assert raw['formal_stability'] is None and raw['rho'] is None
    assert sha(raw_path) == registration['raw']['sha256']
    receipt = dict(schema='gsdeploy.sustained_capacity_independent_audit.job.v1',status='EXPECTED_WALL_TIMEOUT_EVIDENCE_VERIFIED',job_id=job['job_id'],
        packet=dict(path=str(packet_path),sha256=sha(packet_path)),raw=dict(path=str(raw_path),sha256=sha(raw_path)),
        queue_snapshot_sha256=hashlib.sha256(queue_bytes).hexdigest(),queue_status_at_read=queue['status'],
        running_or_live_raw_read=False,terminal_raw_files_read=1,GPU_started=False,
        checks=dict(packet_and_all_bound_sources=True,queue_registered_terminal_SHA=True,worker_READY_and_CPU_launchers=True,
                    full_path_checkpoint_quality_receipts=True,priming21=True,warmup30_and_drained=True,cohort_complete=False,deadline_failure_preserved=True,
                    same32_inputs_three_repeats_per_alias=True,support_exact_prefix_and_stop_boundary=True,one_outstanding_per_alias=True,timestamp_ledger=True,
                    every_phase_native_success=True,safe_memory_independently_recomputed=True,memory_full_window_bracket=memory_window_coverage,
                    whole_board_power_coverage_and_joules_independently_recomputed=True,clean_shutdown=True,
                    native_reference_quality_rescored=True),
        phase_counts=phase_counts,cutoff_counts=cutoff_counts,cutoff_classes=cutoff_classes,cutoff_policy='Both host completion perf-clock and native end converted using start clock offset must be within declared600s; actual loop end overshoot also retained',
        window_end_overshoot_seconds=window['duration_seconds']-600,clock_offset_unix_minus_perf_ns=offset,
        warmup_seconds=warm_window['duration_seconds'],measurement_seconds=window['duration_seconds'],
        memory=dict(safe_memory_mib=safe_memory,budget_mib=memory['budget_mib'],samples=len(memory_samples),
                    window_bracketed=memory_window_coverage,measurement_end_unsampled_seconds=memory_end_gap,unsampled_tail_status='UNKNOWN' if memory_end_gap else 'BRACKETED',
                    scope='Sampled owned-process memory with frozen safety factor; no unsampled peak guarantee'),
        energy=dict(run_joules=joules,startup_joules=startup_joules,samples=len(energy['samples']),scope='Whole GPU board; CPU excluded'),
        diagnostic_primary_quality={alias:dict(requests=b,errors=a,EM=1-a/b) for alias,(a,b) in totals.items()},
        diagnostic_support_quality={alias:dict(requests=b,errors=a,EM=1-a/b) for alias,(a,b) in support_totals.items()},
        diagnostic_after_deadline_quality={str(key):dict(requests=b,errors=a,EM=1-a/b) for key,(a,b) in tail_totals.items()},
        primary_requests=len(primary),completed_primary_requests=sum(r['status']=='completed' for r in primary),support_requests=len(support),support_not_independent_primary_samples=True,
        handler_ledger_max_residual_ms=max_handler_ledger_residual,
        U=None,formal_utility_eligible=False,load_recommendation=None,new_GPU_experiment=None,
        limitations=['This is a finite-cohort diagnostic audit, not formal utility validation.',
                     'Complete closed-loop cohort is not open-loop stable capacity or a formal optimum.',
                     'Thread limits are enforced by the frozen queue main and inherited; actual worker inherited environments are not independently recorded.',
                     'Runtime checkpoint preflight and bound manifests are checked; this audit does not reread full model weights.'],
        audit_source_sha256=sha(__file__),frozen_scorers={str(Path(m.__file__)):sha(m.__file__) for m in [tasks,quality]})
    receipt['planned_measured_requests'] = expected_count
    return receipt,scores

def write_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def paired_baseline(packet, packet_path, queue_bytes, queue, failure_report):
    """Audit genuine completed B; never adapt failed C to a success schema."""
    source=Path(__file__).with_name('audit_sustained.py').resolve()
    spec=importlib.util.spec_from_file_location('_failure_baseline_sustained_audit',source)
    independent=importlib.util.module_from_spec(spec);spec.loader.exec_module(independent)
    baseline_report,_=independent.audit_job(packet,packet_path,queue['jobs'][0],packet['jobs'][0],queue_bytes,queue)
    baseline_path=Path(queue['jobs'][0]['raw']['path']);c_path=Path(queue['jobs'][1]['raw']['path'])
    baseline=json.loads(baseline_path.read_text());failed=json.loads(c_path.read_text())
    lookup={(r['model'],r['pool_id'],r['repetition']):r for r in baseline['request_records']}
    groups=defaultdict(list)
    for row in failed['request_records']:
        if row['model']!='generalist_2#0' or row['role']!='primary' or row['status']!='completed':continue
        ref=lookup[(row['model'],row['pool_id'],row['repetition'])]
        assert ref['sequence']==row['sequence']
        value=row['worker_response'];prior=ref['worker_response']
        groups[failure_report['cutoff_classes'][row['request_id']]].append(dict(request_id=row['request_id'],
            pool_id=row['pool_id'],sequence=row['sequence'],repetition=row['repetition'],service_ms=value['service_ms'],
            baseline_service_ms=prior['service_ms'],ratio=value['service_ms']/prior['service_ms'],
            prediction_matches=value.get('prediction')==prior.get('prediction'),
            generated_tokens_match=value.get('generated_tokens')==prior.get('generated_tokens')))
    summary={}
    for key,rows in groups.items():
        summary[key]=dict(requests=len(rows),unique_inputs=len({r['pool_id'] for r in rows}),
            same_input_repeat_ratio_median=statistics.median(r['ratio'] for r in rows),
            observed_service_mean_ms=statistics.mean(r['service_ms'] for r in rows),
            observed_service_median_ms=statistics.median(r['service_ms'] for r in rows),rows=rows)
    return dict(baseline_audit=baseline_report,comparisons=summary,
        censoring_note='Window results are observed completed requests only; incomplete third repetition is not full-pool calibration or capacity.',
        provenance=[dict(path=str(p),sha256=sha(p)) for p in [source,baseline_path]])


def audit(packet_path,queue_path,output):
    if not __debug__:raise RuntimeError('Run without Python -O')
    packet_path=Path(packet_path).resolve();queue_path=Path(queue_path).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Never overwrite failure audit evidence')
    packet=json.loads(packet_path.read_text());queue_bytes=queue_path.read_bytes();queue=json.loads(queue_bytes)
    assert queue['schema']=='gsdeploy.sustained_capacity_queue.v1'
    assert queue['status']=='PAUSED_ON_FAILED_ATTEMPT_NO_RETRY'
    assert queue['packet_sha256']==sha(packet_path)
    assert [(j['job_id'],j['status']) for j in queue['jobs']]==[(packet['jobs'][0]['job_id'],'completed'),(packet['jobs'][1]['job_id'],'failed')]
    assert packet['protocol']=='sustained_joint_primary_support_v1' and packet['diagnostic_only'] is True and packet['U'] is None
    job=packet['jobs'][1];registration=queue['jobs'][1]
    expected_path=queue_path.parent/(job['job_id']+'-attempt1')/'run.json'
    assert Path(registration['raw']['path'])==expected_path and expected_path.is_absolute()
    provenance=[dict(path=str(p),sha256=sha(p)) for p in [packet_path,queue_path,expected_path,Path(__file__).resolve()]]
    report,scores=audit_job(packet,packet_path,registration,job,queue_bytes,queue)
    paired=paired_baseline(packet,packet_path,queue_bytes,queue,report)
    provenance.extend(paired['provenance'])
    for item in provenance:assert sha(item['path'])==item['sha256']
    for path,expected in packet['bindings'].items():assert sha(path)==expected
    result=dict(schema='gsdeploy.sustained_failed_attempt_audit.v1',status=report['status'],
        research_gate='FAILED_INCOMPLETE_PRIMARY_COHORT',cohort_pass=False,formal_utility_eligible=False,U=None,
        second_block_started=False,queue_registered_jobs=2,terminal_raw_files_read=2,running_or_live_raw_read=False,
        GPU_started=False,provenance=provenance,job=report,baseline_comparison=paired,
        memory_full_window_coverage=report['memory']['window_bracketed'],
        followup_runner_issue='A future separately versioned runner should sample memory in failure-finally before close; current frozen source/raw remain unchanged.')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'audit.json',result);write_json(output/'actual-quality-scores.json',scores);write_json(output/'queue-snapshot.json',queue)
    (output/'README.zh.md').write_text('失败attempt独立证据审计：'+report['status']+'。这不是cohort通过或研究门槛通过。600秒硬界触发，未发布primary保留UNKNOWN；测量内完成与停止后的返回分列，support不替代primary。原生身份、时间账本、已采样显存和全测量窗整卡能量独立核验；显存末'+str(report['memory']['measurement_end_unsampled_seconds'])+'秒无采样，保留UNKNOWN，不宣称全窗显存覆盖。U=null，第二block未启动。\n')
    write_json(output/'manifest.json',dict(provenance=provenance,files={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--queue',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);result=audit(args.packet,args.queue,args.output)
    print(json.dumps(dict(status=result['status'],research_gate=result['research_gate'],U=None)))
    return 0


if __name__=='__main__':raise SystemExit(main())
