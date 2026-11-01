"""Prepare five ASR1.9 development repetitions/expansion controls; no GPU work."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'artifacts/static-study-gpu-20261022'
BASE = ROOT / 'artifacts/static-spillover-study-20261022'
SOURCE = OLD / 'execution-source-v5'
sys.path.insert(0, str(SOURCE))
from gsdeploy import static_study
from gsdeploy.config import SystemConfig
from gsdeploy.static_study import _bindings, _file, _job, validate_packet
from gsdeploy.quality import quality_digest


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    if not Path(static_study.__file__).resolve().is_relative_to(SOURCE):
        raise ValueError('Expected frozen execution-source-v5 implementation')
    destination = BASE / 'development-asr19-replication-v1'
    destination.mkdir(parents=True, exist_ok=False)
    original_path = OLD / 'execution-packet-v2.json'
    original = json.loads(original_path.read_text())
    # Strict development-only allowlist. No validation/final results are read.
    history_path = BASE / 'development-small-spillover-v2/runs-asr1p9/queue-progress.json'
    history = json.loads(history_path.read_text())
    if history['status'] != 'CALIBRATION_COMPLETE' or len(history['jobs']) != 2:
        raise ValueError('ASR1.9 seed51 pair must be complete')
    expected = {'asr1p9_specialist_share_1p0': -0.6496031296448835,
                'asr1p9_specialist_share_0p875': -0.6494732101265066}
    observations = []
    for job in history['jobs']:
        if (job.get('status') != 'completed' or not job.get('eligible')
                or len(job['labels']) != 1 or expected.get(job['labels'][0]) != job.get('U')):
            raise ValueError('Unexpected or incomplete ASR1.9 development observation')
        observations.append(dict(labels=job['labels'], job_id=job['job_id'], U=job['U'],
            eligible=job['eligible'], raw_output=job['output']))
    if {j['labels'][0] for j in observations} != set(expected):
        raise ValueError('Missing paired development observation')
    delta = expected['asr1p9_specialist_share_0p875'] - expected['asr1p9_specialist_share_1p0']
    raw = copy.deepcopy(original['config'])
    raw.update(workload={'text': .05, 'image': .05, 'ocr': .05, 'asr': 1.9},
               horizon_seconds=600, random_seed=51, route_grid=8)
    raw['catalog']['performance_mode'] = 'task_replica_proxy'
    raw['catalog']['memory_mode'] = 'standalone_proxy'
    raw['catalog']['measurement_context'] = {**raw['catalog'].get('measurement_context', {}),
        'gpu_uuid': 'GPU-8a262fe0-598c-46b7-c953-41d9da84a47e'}
    contract = json.loads((Path(original['inputs']) / 'study-contract.json').read_text())
    warmup = contract['warmup']['seconds']
    if warmup != 30:
        raise ValueError('Unexpected original warmup')
    for key in ('weights', 'latency_reference_ms', 'gpu_memory_mib', 'drain_seconds',
                'arrival_process', 'energy_reference_watts', 'startup_amortization_seconds', 'utility_schema'):
        if raw[key] != original['config'][key]:
            raise ValueError('Frozen utility contract changed: ' + key)
    for key in ('models', 'quality', 'safe_factor', 'capacity_factors', 'memory_table_mib', 'task_replica_profiles'):
        if raw['catalog'][key] != original['config']['catalog'][key]:
            raise ValueError('Frozen model pool/evidence changed: ' + key)
    settings = [(52, 2, 1.), (52, 2, .875), (53, 2, 1.), (53, 2, .875), (51, 3, 1.)]
    protocol = dict(schema='gsdeploy.asr19_replication_development.v1',
        status='PREPARED_ONLY_NOT_SCHEDULED',
        purpose='Development paired repetitions and stronger replica-expansion control; no optimum, independent validation or final confirmation claim',
        rationale='ASR1.9 seed51 12.5% spillover improved measured U by only 0.0001299195183769. This single tiny development difference does not establish a reliable gain; repeat across two new development seeds and compare against ASR n3 without spillover.',
        prior_development_source=_file(history_path), prior_development_observations=observations,
        prior_seed51_spillover_minus_no_split_U=delta,
        workload=raw['workload'], horizon_seconds=600, base_config_seed=51, warmup_seconds=30,
        settings=[dict(seed=seed, asr_replicas=n, generalist_2_replicas=1, ocr_replicas=1,
                       specialist_share=share) for seed, n, share in settings],
        background_routes={'text': 'generalist_2', 'image': 'generalist_2', 'ocr': 'ocr'},
        multi_seed_contract='Frozen v5 validate_packet substitutes identity seed into packet config; native binding and execute_job both use each job seed. Base config remains seed51.',
        memory_policy='Original task_replica_proxy and standalone_proxy only. Preserve raw additive residency constraints; do not exclude ASR n3 with an old learned memory multiplier.',
        unchanged=dict(six_weights=raw['weights'], references=raw['latency_reference_ms'],
            utility_schema=raw['utility_schema'], gpu_memory_mib=raw['gpu_memory_mib'],
            model_pool_sha256=quality_digest(raw['catalog']['models']),
            quality_document_sha256=quality_digest(raw['catalog']['quality'])),
        validation_results_read=False, final_confirmation_results_read=False, holdout_observations_used=False,
        predictors_changed=False, production_math_changed=False, all_attempts_preserved=True,
        text_backup_policy='Existing text-backup-development-v1 remains unchanged and unscheduled; this tool does not enqueue any job.',
        validation_caveat='Standalone rho and residency are preparation proxies; actual mixed interference, memory, quality, energy and total utility require native measurement.',
        runtime_source=str(SOURCE), preparer_source=_file(Path(__file__).resolve()),
        original_packet_source=_file(original_path), gpu_started=False)
    save(destination / 'system-config.json', raw)
    save(destination / 'discovery-protocol.json', protocol)
    bindings = _bindings(original['inputs'], SOURCE)
    bindings += [_file(p) for p in [original_path, history_path, Path(__file__).resolve(),
        destination / 'system-config.json', destination / 'discovery-protocol.json']]
    bindings = sorted({b['path']: b for b in bindings}.values(), key=lambda b: b['path'])
    jobs = []
    for seed, n, share in settings:
        config_raw = copy.deepcopy(raw)
        config_raw['random_seed'] = seed
        config = SystemConfig(**config_raw)
        asrrow = {'asr': 1.9 * share}
        if share < 1:
            asrrow['generalist_2'] = 1.9 * (1-share)
        plan = dict(deployment={'asr': n, 'generalist_2': 1, 'ocr': 1},
            routing={'text': {'generalist_2': .05}, 'image': {'generalist_2': .05},
                     'ocr': {'ocr': .05}, 'asr': asrrow})
        constraints = config.validate_plan(**plan)
        if not constraints['constraints_valid']:
            raise ValueError('Raw proxy constraints failed: ' + str(constraints))
        label = f'asr19_n{n}_specialist_share_{share}_seed{seed}'.replace('.', 'p')
        job = _job(config, original['workers'], plan, label, 'calibration', seed, bindings, warmup)
        job['purpose'] = ('development_replica_expansion_strong_control' if n == 3
                          else 'development_paired_spillover_replication')
        job['proxy_constraints'] = constraints
        jobs.append(job)
    packet = dict(schema='gsdeploy.static_experiments.v1',
        status='PREDECLARED_ASR19_REPLICATION_DEVELOPMENT_NOT_SCHEDULED',
        inputs=original['inputs'], config=raw, workers=original['workers'], bindings=bindings,
        frozen_files_sha256=quality_digest(bindings), jobs=jobs, gpu_started=False,
        discovery_protocol=str(destination / 'discovery-protocol.json'),
        execution=dict(policy='Prepared only. Native serial collection uses unchanged frozen v5 source; scheduling is a later parent decision.',
                       runtime_source=str(SOURCE), scheduled=False))
    validation = validate_packet(packet)
    if not validation['valid'] or validation['jobs'] != 5:
        raise ValueError('Frozen v5 native packet validation failed: ' + str(validation))
    save(destination / 'experiment-packet.json', packet)
    save(destination / 'packet-validation.json', dict(native_validation=validation,
        implementation=_file(Path(static_study.__file__)), packet=_file(destination / 'experiment-packet.json'),
        all_five_raw_proxy_constraints_pass=True, multi_seed_supported=True, scheduled=False, gpu_started=False))
    save(destination / 'manifest.json', dict(schema='gsdeploy.asr19_replication_preparation_manifest.v1',
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.is_file()},
        tool=_file(Path(__file__).resolve()), bindings_count=len(bindings), scheduled=False, gpu_started=False))
    print(json.dumps(dict(packet=str(destination / 'experiment-packet.json'), validation=validation,
        bindings=len(bindings), prior_seed51_delta=delta, scheduled=False, gpu_started=False)))


if __name__ == '__main__':
    main()
