"""Independent sustained primary/support terminal CPU audit; no live raw or GPU execution."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import json
import math
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


def audit_measurement_cohorts(job, measured, window):
    """Check the native primary/support contract directly, without raw adaptation.

    Support is a released prefix of an independently frozen per-alias sequence.
    It maintains activity and is never a replacement for missing primary rows.
    """
    aliases = job['active_aliases']
    assert set(job['planned_requests']) == set(job['support_requests']) == set(aliases)
    assert all(r['role'] in {'primary', 'support'} for r in measured)
    primary = [r for r in measured if r['role'] == 'primary']
    support = [r for r in measured if r['role'] == 'support']
    assert len({r['request_id'] for r in measured}) == len(measured)
    planned = {}
    last_primary = {}
    for alias in aliases:
        templates = job['planned_requests'][alias]
        candidates = job['support_requests'][alias]
        assert len(templates) == 96 and len(candidates) == 1024
        rows = [r for r in primary if r['model'] == alias]
        extras = [r for r in support if r['model'] == alias]
        assert len(rows) == 96
        assert all(r['status'] == 'completed' and r['phase'] == 'measurement' for r in rows + extras)
        assert [r['request_id'] for r in rows] == [r['request_id'] for r in templates]
        assert [r['request_id'] for r in extras] == [r['request_id'] for r in candidates[:len(extras)]]
        for i, (row, template) in enumerate(zip(rows, templates)):
            assert row['role'] == 'primary' and row['sequence'] == row['arrival_index'] == i
            assert row['repetition'] == i // 32
            assert all(row[key] == value for key, value in template.items())
            planned[row['request_id']] = template
        for i, (row, template) in enumerate(zip(extras, candidates)):
            assert row['role'] == 'support' and row['support_sequence'] == i
            assert row['sequence'] == row['arrival_index'] == 96+i and row['repetition'] == i//32
            assert all(row[key] == value for key, value in template.items())
            planned[row['request_id']] = template
        if job.get('state') == 'B':
            assert not extras, 'Single-active baseline must stop at its last primary'
        last_primary[alias] = max(r['completion_time_ns'] for r in rows)
        assert window['alias_completion_time_ns'][alias] == last_primary[alias]
        assert window['primary_completed_by_alias'][alias] == 96
        assert window['support_released_by_alias'][alias] == len(extras)
        assert window['released_by_alias'][alias] == 96+len(extras)
        for row in extras:
            assert last_primary[alias] <= row['scheduled_time_ns'] < window['support_admission_stop_time_ns']
        # Complete native timeline order across the phase role boundary too.
        sequence = rows + extras
        for before, after in zip(sequence, sequence[1:]):
            assert before['completion_time_ns'] <= after['scheduled_time_ns']
    assert len(planned) == len(measured)
    assert set(window['alias_completion_time_ns']) == set(aliases)
    assert set(window['primary_completed_by_alias']) == set(window['support_released_by_alias']) == set(window['released_by_alias']) == set(aliases)
    primary_end = max(last_primary.values())
    assert window['primary_completion_time_ns'] == primary_end
    assert primary_end <= window['support_admission_stop_time_ns'] <= window['end_time_ns']
    assert max(r['completion_time_ns'] for r in measured) <= window['end_time_ns']
    assert 0 < window['end_time_ns']-window['start_time_ns'] <= 600_000_000_000
    assert math.isclose(window['duration_seconds'],(window['end_time_ns']-window['start_time_ns'])/1e9,abs_tol=1e-9)
    for row in measured:
        assert row['scheduled_time_ns'] <= row['completion_time_ns']
        assert row['completion_time_ns']-row['scheduled_time_ns'] <= 300_000_000_000
    return primary, support, planned


def audit_job(packet,packet_path,registration,job,queue_bytes,queue):
    raw_path=Path(registration['raw']['path'])
    assert sha(raw_path)==registration['raw']['sha256']
    raw = json.loads(raw_path.read_text())
    assert raw['schema'] == 'gsdeploy.sustained_joint_capacity.raw.v1'
    assert raw['status'] == 'completed' and not raw['errors']
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
    measured = raw['request_records']
    assert [{k:r[k] for k in ['model','task','pool_id']} for r in priming] == packet['priming_requests']
    expected_count = 96*len(job['active_aliases'])
    assert len(priming) == 21
    primary, support, planned = audit_measurement_cohorts(job, measured, raw['measurement_window'])
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
    assert max(r['completion_time_ns'] for r in measured) <= window['end_time_ns']
    assert window['primary_completed_by_alias'] == {alias:96 for alias in job['active_aliases']}
    assert window['max_outstanding_per_alias'] == 1
    for alias in job['active_aliases']:
        rows = [r for r in primary if r['model']==alias]
        assert Counter(r['pool_id'] for r in rows) == Counter({p:3 for p in job['input_pool_ids']})
        assert [r['pool_id'] for r in rows] == job['input_pool_ids']
        assert window['alias_completion_time_ns'][alias] == max(r['completion_time_ns'] for r in rows)
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
    assert max(r['unix_ns'] for r in memory_samples) >= window['end_time_ns']+offset
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
    scores = []; totals = defaultdict(lambda:[0,0]); support_totals = defaultdict(lambda:[0,0])
    for row in measured:
        result = dict(row['worker_response'])
        score = tasks.score_result(reference['text'][row['pool_id']],result)
        a,b = quality.sample_totals('text',score)
        role_totals = totals if row['role']=='primary' else support_totals
        role_totals[row['model']][0] += a;role_totals[row['model']][1] += b
        scores.append(dict(request_id=row['request_id'],alias=row['model'],pool_id=row['pool_id'],repetition=row['repetition'],role=row['role'],score=score,error_numerator=a,reference_denominator=b))
    utility = raw['evaluation']['utility']
    assert utility['U'] is None and utility['eligible'] is False
    assert raw['formal_stability'] is None and raw['rho'] is None
    assert sha(raw_path) == registration['raw']['sha256']
    receipt = dict(schema='gsdeploy.sustained_capacity_independent_audit.job.v1',status='PASS',job_id=job['job_id'],
        packet=dict(path=str(packet_path),sha256=sha(packet_path)),raw=dict(path=str(raw_path),sha256=sha(raw_path)),
        queue_snapshot_sha256=hashlib.sha256(queue_bytes).hexdigest(),queue_status_at_read=queue['status'],
        running_or_live_raw_read=False,terminal_raw_files_read=1,GPU_started=False,
        checks=dict(packet_and_all_bound_sources=True,queue_registered_terminal_SHA=True,worker_READY_and_CPU_launchers=True,
                    full_path_checkpoint_quality_receipts=True,priming21=True,warmup30_and_drained=True,cohort_complete=True,
                    same32_inputs_three_repeats_per_alias=True,support_exact_prefix_and_stop_boundary=True,one_outstanding_per_alias=True,timestamp_ledger=True,
                    every_phase_native_success=True,safe_memory_independently_recomputed=True,
                    whole_board_power_coverage_and_joules_independently_recomputed=True,clean_shutdown=True,
                    native_reference_quality_rescored=True),
        phase_counts=phase_counts,warmup_seconds=warm_window['duration_seconds'],measurement_seconds=window['duration_seconds'],
        memory=dict(safe_memory_mib=safe_memory,budget_mib=memory['budget_mib'],samples=len(memory_samples),
                    scope='Sampled owned-process memory with frozen safety factor; no unsampled peak guarantee'),
        energy=dict(run_joules=joules,startup_joules=startup_joules,samples=len(energy['samples']),scope='Whole GPU board; CPU excluded'),
        diagnostic_primary_quality={alias:dict(requests=b,errors=a,EM=1-a/b) for alias,(a,b) in totals.items()},
        diagnostic_support_quality={alias:dict(requests=b,errors=a,EM=1-a/b) for alias,(a,b) in support_totals.items()},
        primary_requests=len(primary),support_requests=len(support),support_not_independent_primary_samples=True,
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


def audit(packet_path,queue_path,output):
    if not __debug__:raise RuntimeError('Run audit without Python -O: assertions are required')
    packet_path=Path(packet_path).resolve();queue_path=Path(queue_path).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Never overwrite audit evidence')
    packet=json.loads(packet_path.read_text());queue_bytes=queue_path.read_bytes();queue=json.loads(queue_bytes)
    if queue.get('status')!='ALL_FOUR_SUSTAINED_DIAGNOSTIC_ATTEMPTS_COMPLETE':
        raise ValueError('All four sustained queue publication required before reading any terminal raw')
    if queue.get('packet_sha256')!=sha(packet_path):raise ValueError('Queue packet hash differs')
    if packet.get('protocol')!='sustained_joint_primary_support_v1' or packet.get('U') is not None or packet.get('diagnostic_only') is not True:
        raise ValueError('Explicit diagnostic packet required')
    if queue.get('schema')!='gsdeploy.sustained_capacity_queue.v1':raise ValueError('Unexpected sustained queue schema')
    expected=packet['jobs'];registered=queue['jobs']
    if [(j['block'],j['state'],j['seed']) for j in expected] != [(1,'B',103),(1,'C',103),(2,'C',104),(2,'B',104)]:
        raise ValueError('Exact sustained BC/CB block order required')
    if len(expected)!=4 or [r['job_id'] for r in registered]!=[j['job_id'] for j in expected]:
        raise ValueError('Exact four-job queue order required')
    if len({j['job_id'] for j in expected})!=4:raise ValueError('Duplicate job identity')
    raw_receipts=[]
    for registration,job in zip(registered,expected):
        if registration['status']!='completed':raise ValueError('Uncompleted registered condition')
        path=Path(registration['raw']['path'])
        expected_path=queue_path.parent/(job['job_id']+'-attempt1')/'run.json'
        if not path.is_absolute() or path!=expected_path:raise ValueError('Unexpected terminal path')
        if sha(path)!=registration['raw']['sha256']:raise ValueError('Terminal SHA differs')
        raw_receipts.append(dict(path=str(path),sha256=sha(path)))
    provenance=[dict(path=str(packet_path),sha256=sha(packet_path)),
        dict(path=str(queue_path),sha256=hashlib.sha256(queue_bytes).hexdigest()),
        dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))]+raw_receipts
    reports=[];all_scores=[]
    for registration,job in zip(registered,expected):
        try:
            report,scores=audit_job(packet,packet_path,registration,job,queue_bytes,queue)
            reports.append(report);all_scores.extend(dict(job_id=job['job_id'],**s) for s in scores)
        except Exception as exc:
            frames=traceback.extract_tb(exc.__traceback__)
            last=frames[-1] if frames else None
            reports.append(dict(job_id=job['job_id'],status='AUDIT_FAILED',error_type=type(exc).__name__,
                                error=str(exc) or (last.line if last else 'Unspecified audit failure'),
                                failed_check=last.line if last else None,
                                error_location=dict(file=last.filename,line=last.lineno,function=last.name) if last else None,
                                traceback=traceback.format_exc(),resources_and_energy_verified_independently=False,
                                U=None,formal_utility_eligible=False))
    # Detect any input mutation across the entire audit before publishing evidence.
    for item in provenance:
        if sha(item['path'])!=item['sha256']:raise ValueError('Audit input changed: '+item['path'])
    for path, expected_hash in packet['bindings'].items():
        if sha(path)!=expected_hash:raise ValueError('Bound input changed during audit: '+path)
    status='PASS' if all(r['status']=='PASS' for r in reports) else 'AUDIT_FAILED'
    result=dict(schema='gsdeploy.sustained_capacity_independent_audit.v1',status=status,jobs=reports,
                planned_measured_requests=sum(96*len(j['active_aliases']) for j in expected),
                released_support_requests=sum(r.get('support_requests',0) for r in reports),
                terminal_raw_files_read=4,running_or_live_raw_read=False,GPU_started=False,
                raw_status_is_not_audit_pass=True,
                resources_and_energy_verified_independently=all(r['status']=='PASS' for r in reports),
                resource_independently_verified=all(r.get('checks',{}).get('safe_memory_independently_recomputed') is True for r in reports),
                energy_independently_verified=all(r.get('checks',{}).get('whole_board_power_coverage_and_joules_independently_recomputed') is True for r in reports),
                U=None,formal_utility_eligible=False,provenance=provenance)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'audit.json',result);write_json(output/'actual-quality-scores.json',all_scores)
    write_json(output/'queue-snapshot.json',queue)
    for report in reports:write_json(output/(report['job_id']+'-audit.json'),report)
    with (output/'README.zh.md').open('x') as stream:
        stream.write('四条件持续共同活动独立终态审计：'+status+'。仅在原队列全部完成后读取已登记SHA的run.json。\n\n逐运行核对身份、预热、warmup、primary与support独立前缀、停止和排空边界、单在途、时间账本、实际输出质量、显存与整卡积分能量；资源有效性与能量覆盖分别核验。原运行status不替代审计结果。\n\nU=null；不读取live，不进行GPU实验，不证明开放负载稳定容量或正式最优。\n')
    manifest=dict(provenance=provenance,files={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    write_json(output/'manifest.json',manifest)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);result=audit(args.packet,args.queue,args.output)
    print(json.dumps(dict(status=result['status'],jobs=len(result['jobs']),U=None,GPU_started=False)))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
