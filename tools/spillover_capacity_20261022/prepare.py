"""Freeze six balanced closed-loop capacity diagnostics, without GPU loading."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/static-study-gpu-20261022/execution-source-v5'
BASE = ROOT / 'artifacts/static-spillover-study-20261022/text-capacity-diagnostic-v1'
sys.path.insert(0, str(SOURCE))
from gsdeploy.config import SystemConfig
from gsdeploy.quality import quality_digest
from gsdeploy.static_study import _bindings, bind_native_job
from gsdeploy.runtime.workload import read_jsonl, SamplePool, largest_inputs


def save(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    if not Path(sys.modules['gsdeploy.config'].__file__).resolve().is_relative_to(SOURCE):
        raise ValueError('Expected frozen v5 imports')
    runner = Path(__file__).with_name('run.py')
    if not runner.is_file():
        raise ValueError('Finish and test diagnostic runner before freezing packet')
    BASE.mkdir(parents=True, exist_ok=False)
    original_path = ROOT / 'artifacts/static-study-gpu-20261022/execution-packet-v2.json'
    original = json.loads(original_path.read_text())
    samples = Path(original['inputs']) / 'runtime-inputs.jsonl'
    items = read_jsonl(samples)
    text_ids = [item['pool_id'] for item in items if item['task_type'] == 'text']
    if len(text_ids) != 32 or len(set(text_ids)) != 32:
        raise ValueError('Expected original frozen 32-input text pool')
    aliases = {'text#0': 'text', 'text#1': 'text',
               'generalist_2#0': 'generalist_2', 'ocr#0': 'ocr'}
    deployment = {'text': 2, 'generalist_2': 1, 'ocr': 1}
    config = copy.deepcopy(original['config'])
    config.update(random_seed=101, horizon_seconds=600)
    config['catalog']['performance_mode'] = 'task_replica_proxy'
    config['catalog']['memory_mode'] = 'standalone_proxy'
    config['catalog']['measurement_context'] = {
        **config['catalog'].get('measurement_context', {}),
        'gpu_uuid': 'GPU-8a262fe0-598c-46b7-c953-41d9da84a47e'}
    q = config['workload']
    preflight_plan = dict(deployment=deployment, routing={
        'text': {'text': q['text']/2, 'generalist_2': q['text']/2},
        'image': {'generalist_2': q['image']},
        'asr': {'generalist_2': q['asr']}, 'ocr': {'ocr': q['ocr']}})
    constraints = SystemConfig(**config).validate_plan(**preflight_plan)
    if not constraints['constraints_valid']:
        raise ValueError('Preparation proxy checks failed: ' + str(constraints))
    binding_packet = {**original, 'config': config}
    fake_job = dict(plan=preflight_plan, seed=101, identity={'horizon_seconds': 600})
    native_config, native_admission, workers = bind_native_job(
        binding_packet, fake_job, BASE / 'native-binding')
    save(BASE / 'system-config.json', native_config)
    save(BASE / 'workers.json', workers)
    save(BASE / 'admission.json', json.loads(native_admission.read_text()))
    # Bind exact priming IDs, including lazy non-text components and idle OCR.
    pool = SamplePool(samples, ['text', 'image', 'ocr', 'asr'], 101)
    priming = []
    for alias in sorted(aliases):
        family = aliases[alias]
        for task in native_config['catalog']['models'][family]['supported_tasks']:
            for item in largest_inputs(pool, task, 3):
                priming.append(dict(model=alias, task=task, pool_id=item['pool_id']))
    if len(priming) != 21:
        raise ValueError('Expected full-resident largest-three 21-request priming')
    states = {'A': ['text#0', 'text#1'], 'B': ['generalist_2#0'],
              'C': ['text#0', 'text#1', 'generalist_2#0']}
    jobs = []
    for block, seed, order in [(1, 101, 'ABC'), (2, 102, 'CBA')]:
        ids = []
        for repetition in range(3):
            sequence = list(text_ids)
            random.Random(f'{seed}:balanced-text:{repetition}').shuffle(sequence)
            ids.extend(sequence)
        for state in order:
            job_id = f'block{block}_{state}_seed{seed}'
            planned = {}
            for alias in states[state]:
                planned[alias] = [dict(
                    request_id=f'measurement:{alias}:{i}', task='text', model=alias,
                    pool_id=pid, repetition=i//32, sequence=i, arrival_index=i)
                    for i, pid in enumerate(ids)]
            job = dict(job_id=job_id, state=state, block=block, seed=seed,
                       active_aliases=states[state], input_pool_ids=ids,
                       planned_requests=planned)
            job['job_identity_sha256'] = quality_digest(job)
            jobs.append(job)
    protocol = dict(
        schema='gsdeploy.balanced_capacity_declaration.v1',
        scope='Six finite-cohort development diagnostics; no formal utility, open-loop stability, model validation or optimum claim',
        purpose='Identify same-input resident/active-context service and reproducibility before choosing target load',
        states=states, block_order=['A', 'B', 'C', 'C', 'B', 'A'],
        seeds={'development': [101, 102], 'independent_validation_reserved': [81, 82],
               'final_confirmation_reserved': [71, 72, 73]},
        seed_usage_check='No101/102 found in existing spillover protocol/packet/declaration/scheduling metadata before preparation; diagnostic shuffles are separately namespaced',
        measured_text_inputs_per_alias=32, repetitions_per_input=3,
        measurement_cohort='96 per active alias, same block sequence across every alias/state',
        activity='One outstanding request per alias; release measurement together; actual common-activity interval and coverage audited separately from post-completion tail',
        priming='21 largest-three requests across all resident supported paths, sequential, responses retained; full multimodal lazy residency established before warmup',
        warmup='30seconds balanced closed-loop only active text aliases; drain fully before measurement',
        timeouts=dict(request_seconds=300, measurement_seconds=600,
                      startup_seconds=300, shutdown_seconds=10),
        preflight_plan_scope='Original low-rate workload proxy for worker/checkpoint identity and resource preparation only; never a claimed rate of this closed-loop cohort',
        resource_policy='Original safe_factor and49140MiB; remeasure peaks; no runtime/rho/stability weakening',
        telemetry='Frozen process-owned memory/power monitors plus10second GPU-state queries; query overhead retained as diagnostic context',
        preserved=['seven-model catalog', 'quality references and frozen outputs',
                   'six-term weights and references', 'rho<=0.8 formal constraint',
                   'formal runtime/SWRR/input rules and stability.py', 'failed transfer activation'],
        no_formal_U=True, no_auto_retry=True, no_following_batch_auto_enqueue=True,
        missing_coverage=['G2/image full-input service', 'G2/asr full-input service',
                          'OCR/ocr full-input service', 'formal background jointly active response',
                          'other deployments/families/nstates', 'open-loop load/route response'],
        decision_gate='Only after auditing same-input repeats/common activity and background evidence, choose2target rates and0/1/8/1/4 routes; no blind2.6rps continuation',
        source_plan=str(ROOT/'docs/spillover_recovery_experiment_plan_20261022.md'))
    save(BASE/'discovery-protocol.json', protocol)
    bindings = {b['path']: b['sha256'] for b in _bindings(Path(original['inputs']), SOURCE)}
    for path in [original_path, Path(__file__).resolve(), runner.resolve(),
                 ROOT/'docs/spillover_recovery_experiment_plan_20261022.md',
                 BASE/'system-config.json', BASE/'workers.json', BASE/'admission.json',
                 BASE/'discovery-protocol.json']:
        bindings[str(path.resolve())] = sha(path)
    packet = dict(schema_version=1, protocol='balanced_closed_loop_capacity_v1',
        diagnostic_only=True, U=None, frozen_source=str(SOURCE),
        system_config=str(BASE/'system-config.json'), samples=str(samples),
        workers=str(BASE/'workers.json'), admission=str(BASE/'admission.json'),
        gpu_uuid='GPU-8a262fe0-598c-46b7-c953-41d9da84a47e', device='0',
        gpu_memory_mib=49140, safe_factor=native_config['catalog']['safe_factor'],
        aliases=aliases, resident_deployment=deployment, preflight_plan=preflight_plan,
        priming_policy='largest_three_all_resident_supported_tasks',
        priming_requests=priming, warmup_seconds=30,
        measurement_timeout_seconds=600, request_timeout_seconds=300,
        startup_timeout_seconds=300, shutdown_timeout_seconds=10,
        telemetry_interval_seconds=10, jobs=jobs, bindings=bindings,
        bindings_sha256=quality_digest(bindings), gpu_started=False)
    save(BASE/'experiment-packet.json', packet)
    save(BASE/'preparation-receipt.json', dict(
        packet_sha256=sha(BASE/'experiment-packet.json'), jobs=len(jobs),
        priming_requests_per_job=len(priming), planned_measured_requests=sum(
            len(rows) for j in jobs for rows in j['planned_requests'].values()),
        checkpoint_identity=native_config['catalog']['measurement_context']['runtime_checkpoint_sha256'],
        proxy_constraints=constraints, GPU_started=False))
    print(json.dumps(dict(packet=str(BASE/'experiment-packet.json'), jobs=len(jobs),
                         measured_requests=1152, gpu_started=False)))


if __name__ == '__main__':
    main()
