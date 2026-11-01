"""Prepare text fallback development jobs; never executes or schedules GPU work.

Uses the frozen v5 native packet implementation and unchanged model/quality pool.
Only the explicitly completed ASR seed51 development progress is read as history.
"""
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


def development_history():
    """Narrow allowlist: no validation, final-confirmation or ASR1.9 results."""
    sources = [BASE / 'development-asr18-v1/runs/queue-progress.json',
               BASE / 'development-small-spillover-v2/runs-asr1p8/queue-progress.json']
    observations = []
    for path in sources:
        data = json.loads(path.read_text())
        if data['status'] != 'CALIBRATION_COMPLETE':
            raise ValueError('Expected completed seed51 development history: ' + str(path))
        for job in data['jobs']:
            if job.get('status') != 'completed' or not job.get('eligible') or job.get('U') is None:
                raise ValueError('Development observation is incomplete or ineligible')
            observations.append(dict(labels=job['labels'], job_id=job['job_id'],
                U=job['U'], eligible=job['eligible'], raw_output=job['output']))
    expected = {'asr18_n2_specialist_share_1p0': -0.5525885133597289,
                'asr18_n2_specialist_share_0p75': -0.7049277553855252,
                'asr1p8_specialist_share_0p875': -0.6173423595747094}
    observed = {label: job['U'] for job in observations for label in job['labels']}
    if observed != expected:
        raise ValueError('Unexpected completed ASR development history; review before redeclaring')
    return sources, observations


def main():
    if not Path(static_study.__file__).resolve().is_relative_to(SOURCE):
        raise ValueError('Packet preparer must import frozen execution-source-v5')
    destination = BASE / 'text-backup-development-v1'
    destination.mkdir(parents=True, exist_ok=False)
    original_path = OLD / 'execution-packet-v2.json'
    original = json.loads(original_path.read_text())
    history_paths, observations = development_history()
    inputs = Path(original['inputs'])
    contract = json.loads((inputs / 'study-contract.json').read_text())
    warmup = contract['warmup']['seconds']
    raw = copy.deepcopy(original['config'])
    raw.update(workload={'text': 1.8, 'image': .01, 'ocr': .01, 'asr': .01},
               horizon_seconds=600, random_seed=51, route_grid=8)
    raw['catalog']['performance_mode'] = 'task_replica_proxy'
    raw['catalog']['memory_mode'] = 'standalone_proxy'
    raw['catalog']['measurement_context'] = {**raw['catalog'].get('measurement_context', {}),
        'gpu_uuid': 'GPU-8a262fe0-598c-46b7-c953-41d9da84a47e'}
    for key in ('weights', 'latency_reference_ms', 'gpu_memory_mib', 'drain_seconds',
                'arrival_process', 'energy_reference_watts', 'startup_amortization_seconds', 'utility_schema'):
        if raw[key] != original['config'][key]:
            raise ValueError('Changed frozen utility contract: ' + key)
    for key in ('models', 'quality', 'safe_factor', 'capacity_factors', 'memory_table_mib', 'task_replica_profiles'):
        if raw['catalog'][key] != original['config']['catalog'][key]:
            raise ValueError('Changed frozen pool or evidence: ' + key)
    settings = [(2, 1.), (2, .875), (2, .75), (1, .5)]
    protocol = dict(schema='gsdeploy.text_backup_development.v1',
        status='PREPARED_ONLY_NOT_SCHEDULED',
        purpose='Exploratory actual development proposals; no optimum, validated predictor or independent confirmation claim',
        rationale='ASR1.8 seed51 development showed lower measured U with both 25% and 12.5% spillover than no split. Text standalone profiles offer a stronger G2n1 versus specialist-n2 service advantage; test this separately without changing the ASR plan.',
        prior_ASR_development_observations=observations,
        prior_ASR_development_sources=[_file(p) for p in history_paths],
        decision_policy='Prepare only. Parent workflow evaluates the ongoing ASR1.9 pair; only if that direction fails may it separately decide to enqueue text development.',
        ASR1p9_results_read=False, validation_results_read=False, final_confirmation_results_read=False,
        holdout_observations_used=False, predictors_changed=False, production_math_changed=False,
        workload=raw['workload'], horizon_seconds=600, seed=51, route_grid=8, warmup_seconds=warmup,
        settings=[dict(text_replicas=n, generalist_2_replicas=1, ocr_replicas=1,
                       text_specialist_share=share, text_to_G2_share=1-share) for n, share in settings],
        background_routes={'image': 'generalist_2', 'asr': 'generalist_2', 'ocr': 'ocr'},
        unchanged=dict(six_weights=raw['weights'], references=raw['latency_reference_ms'],
            utility_schema=raw['utility_schema'], model_pool_sha256=quality_digest(raw['catalog']['models']),
            quality_document_sha256=quality_digest(raw['catalog']['quality']),
            runtime_input_pool_sha256=raw['catalog']['quality']['identity']['input_pool_sha256']),
        quality_caveat='Historical32-input cycle and deterministic share routing can align. Rescore actual outputs and disclose path coverage; no zero-quality-penalty assumption.',
        validation_caveat='Standalone rho and residency are preparation proxies only; mixed interference, actual peak memory, energy and utility remain unmeasured.',
        all_attempts_preserved=True, no_confirmation_seeds_used=True,
        runtime_source=str(SOURCE), preparer_source=_file(Path(__file__).resolve()),
        original_packet_source=_file(original_path),
        CPU_feasibility_report=_file(ROOT / 'docs/text_spillover_backup_feasibility_20261022.md'),
        gpu_started=False)
    save(destination / 'system-config.json', raw)
    save(destination / 'discovery-protocol.json', protocol)
    bindings = _bindings(original['inputs'], SOURCE)
    bindings += [_file(p) for p in [original_path, Path(__file__).resolve(), *history_paths,
        ROOT / 'docs/text_spillover_backup_feasibility_20261022.md',
        destination / 'system-config.json', destination / 'discovery-protocol.json']]
    bindings = sorted({b['path']: b for b in bindings}.values(), key=lambda b: b['path'])
    config = SystemConfig(**raw)
    jobs = []
    for n, share in settings:
        textrow = {'text': 1.8 * share}
        if share < 1:
            textrow['generalist_2'] = 1.8 * (1-share)
        plan = dict(deployment={'text': n, 'generalist_2': 1, 'ocr': 1},
            routing={'text': textrow, 'image': {'generalist_2': .01},
                     'asr': {'generalist_2': .01}, 'ocr': {'ocr': .01}})
        constraints = config.validate_plan(**plan)
        if not constraints['constraints_valid']:
            raise ValueError('Preparation proxy constraints failed: ' + str(constraints))
        label = f'text18_n{n}_specialist_share_{share}'.replace('.', 'p')
        job = _job(config, original['workers'], plan, label, 'calibration', 51, bindings, warmup)
        job['purpose'] = 'text_backup_exploratory_development_not_optimum_or_final_confirmation'
        job['proxy_constraints'] = constraints
        jobs.append(job)
    packet = dict(schema='gsdeploy.static_experiments.v1',
        status='PREDECLARED_TEXT_BACKUP_DEVELOPMENT_NOT_SCHEDULED', inputs=original['inputs'],
        config=raw, workers=original['workers'], bindings=bindings,
        frozen_files_sha256=quality_digest(bindings), jobs=jobs, gpu_started=False,
        discovery_protocol=str(destination / 'discovery-protocol.json'),
        execution=dict(policy='Prepared only; native serial collection under unchanged frozen v5 source requires a later parent scheduling decision',
                       runtime_source=str(SOURCE), scheduled=False))
    validation = validate_packet(packet)
    if not validation['valid'] or validation['jobs'] != 4:
        raise ValueError('Frozen native validate_packet failed: ' + str(validation))
    save(destination / 'experiment-packet.json', packet)
    save(destination / 'packet-validation.json', dict(native_validation=validation,
        implementation=_file(Path(static_study.__file__)), packet=_file(destination / 'experiment-packet.json'),
        all_four_proxy_constraints_pass=True, scheduled=False, gpu_started=False))
    save(destination / 'manifest.json', dict(schema='gsdeploy.text_backup_preparation_manifest.v1',
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.is_file()},
        tool=_file(Path(__file__).resolve()), bindings_count=len(bindings), scheduled=False, gpu_started=False))
    print(json.dumps(dict(packet=str(destination / 'experiment-packet.json'), jobs=len(jobs),
        validation=validation, bindings=len(bindings), scheduled=False, gpu_started=False)))


if __name__ == '__main__':
    main()
