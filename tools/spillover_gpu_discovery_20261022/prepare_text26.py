"""Prepare four text2.6 native development proposals, never enqueue or run."""
import copy
from fractions import Fraction
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
        raise ValueError('Expected frozen native v5 implementation')
    destination = BASE / 'text26-development-v1'
    destination.mkdir(parents=True, exist_ok=False)
    original_path = OLD / 'execution-packet-v2.json'
    original = json.loads(original_path.read_text())
    # Preparation receipts and CPU outputs only; no run/holdout files are read.
    old_packet_path = BASE / 'text-backup-development-v1/experiment-packet.json'
    old_packet = json.loads(old_packet_path.read_text())
    if (not validate_packet(old_packet)['valid'] or old_packet['gpu_started']
            or old_packet['execution']['scheduled']):
        raise ValueError('Original text1.8 backup receipt changed or was scheduled')
    old_directory = old_packet_path.parent
    if list(old_directory.glob('runs*')) or list(old_directory.glob('queue*')):
        raise ValueError('Original text backup has execution artifacts; review before redeclaring')
    cpu_v1_path = BASE / 'cpu-discovery/text-load-screen-v1/report.json'
    cpu_v2_path = BASE / 'cpu-discovery/text-load-screen-v2/report.json'
    cpu_v1, cpu_v2 = [json.loads(p.read_text()) for p in (cpu_v1_path, cpu_v2_path)]
    summary = next(s for s in cpu_v2['summaries'] if s['text_rps'] == 2.6)
    n2, n3 = summary['best_by_text_replicas']['2'], summary['best_by_text_replicas']['3']
    if n2['specialist_share'] != '4/7' or n3['specialist_share'] != '4/7':
        raise ValueError('CPU best-route receipt changed; review before preparing')
    raw = copy.deepcopy(original['config'])
    raw.update(workload={'text': 2.6, 'image': .01, 'ocr': .01, 'asr': .01},
               horizon_seconds=600, random_seed=51, route_grid=8)
    raw['catalog']['performance_mode'] = 'task_replica_proxy'
    raw['catalog']['memory_mode'] = 'standalone_proxy'
    raw['catalog']['measurement_context'] = {**raw['catalog'].get('measurement_context', {}),
        'gpu_uuid': 'GPU-8a262fe0-598c-46b7-c953-41d9da84a47e'}
    for key in ('weights', 'latency_reference_ms', 'gpu_memory_mib', 'drain_seconds',
                'arrival_process', 'energy_reference_watts', 'startup_amortization_seconds', 'utility_schema'):
        if raw[key] != original['config'][key]:
            raise ValueError('Utility contract changed: ' + key)
    for key in ('models', 'quality', 'safe_factor', 'capacity_factors', 'memory_table_mib', 'task_replica_profiles'):
        if raw['catalog'][key] != original['config']['catalog'][key]:
            raise ValueError('Frozen model pool/evidence changed: ' + key)
    config = SystemConfig(**raw)
    no_split_capacity = []
    for n in (1, 2, 3):
        measured = raw['catalog']['task_replica_profiles']['text']['text'][str(n)]['mean_service_ms']
        rho = 2.6*measured/1000/raw['catalog']['capacity_factors']['text'][str(n)]
        if rho <= .8:
            raise ValueError('Unexpected feasible no-split control; review exclusion')
        no_split_capacity.append(dict(text_replicas=n, original_mean_proxy_text_rho=rho,
            rho_limit=.8, proxy_infeasible=True, not_scheduled=True))
    settings = [(2, Fraction(4,7), 'target_majority_specialist'),
                (2, Fraction(1,2), 'same_deployment_adjacent_route'),
                (1, Fraction(1,2), 'feasible_one_specialist_strong_control'),
                (3, Fraction(n3['specialist_share']), 'strong_replica_expansion_control_cpu_v2_best')]
    protocol = dict(schema='gsdeploy.text26_development.v1', status='PREPARED_ONLY_NOT_SCHEDULED',
        purpose='Exploratory native development: test whether n2 split beats feasible n1 split and strong n3 expansion; no full-pool optimum or independent validation claim',
        workload=raw['workload'], horizon_seconds=600, seed=51, warmup_seconds=30,
        plan_change_receipt=dict(previous_packet=_file(old_packet_path),
            previous_status='Unchanged, prepared and unscheduled text1.8 packet',
            reason='Additional finite CPU screening found n1 strongest at text1.8, while n2 majority-share split leads at text2.6. Change to this new condition instead of assuming the original text1.8 target structure wins.',
            cpu_v1_source=_file(cpu_v1_path), cpu_v2_source=_file(cpu_v2_path),
            cpu_v2_text26_best_n2=n2, cpu_v2_text26_best_n3=n3,
            cpu_v2_n2_minus_feasible_n1_half_diagnostic=summary['n2_minus_previous_n1_half_external_score'],
            old_plan_overwritten=False, old_plan_enqueued=False),
        n3_route_choice='Use actual v2 text2.6 n3 best specialist share4/7 instead of proposed3/4; v1 has no text2.6 results. Limited candidate best only.',
        settings=[dict(text_replicas=n, generalist_2_replicas=1, ocr_replicas=1,
            specialist_share=str(share), purpose=purpose) for n, share, purpose in settings],
        background_routes={'image':'generalist_2', 'asr':'generalist_2', 'ocr':'ocr'},
        no_split_proxy_exclusions=no_split_capacity,
        no_split_comparison_scope='All n1/n2/n3 no-split proxy text rho exceeds0.8. Mixed versus those no-split structures is capacity rescue only; no infeasible no-split GPU run is forced.',
        target_comparison_scope='n2 split against feasible n1 split and n3 strong expansion; full physical mixed constraints and total utility still require actual evidence.',
        route_domain='Concrete development share4/7 is accepted by native validate_plan. route_grid8 remains a config search setting, not a claim this route belongs to the historical K8 optimum domain.',
        unchanged=dict(weights=raw['weights'], references=raw['latency_reference_ms'],
            gpu_memory_mib=raw['gpu_memory_mib'], utility_schema=raw['utility_schema'],
            model_pool_sha256=quality_digest(raw['catalog']['models']),
            quality_document_sha256=quality_digest(raw['catalog']['quality'])),
        evidence_caveats=['CPU energy is external unvalidated old-power extrapolation, not measured utility',
            'Raw residency and rho are preparation proxies, not verified mixed feasibility',
            'Single development seed, actual outputs must be rescored, no quality changes'],
        memory_policy='Original standalone_proxy safe residency sum; no old learned multiplier exclusion of n3.',
        incomplete_runs_read=False, validation_results_read=False, final_confirmation_results_read=False,
        holdout_observations_used=False, predictors_changed=False, production_math_changed=False,
        all_attempts_preserved=True, runtime_source=str(SOURCE),
        tool_source=_file(Path(__file__).resolve()), original_packet_source=_file(original_path),
        scheduled=False, gpu_started=False)
    save(destination/'system-config.json',raw)
    save(destination/'discovery-protocol.json',protocol)
    bindings = _bindings(original['inputs'],SOURCE)
    bindings += [_file(p) for p in (original_path, old_packet_path, cpu_v1_path, cpu_v2_path,
        Path(__file__).resolve(), destination/'system-config.json', destination/'discovery-protocol.json')]
    bindings = sorted({b['path']:b for b in bindings}.values(),key=lambda b:b['path'])
    jobs=[]
    for n, share, purpose in settings:
        routing=dict(text={'text':2.6*float(share), 'generalist_2':2.6*(1-float(share))},
            image={'generalist_2':.01},asr={'generalist_2':.01},ocr={'ocr':.01})
        plan=dict(deployment={'text':n,'generalist_2':1,'ocr':1},routing=routing)
        constraints=config.validate_plan(**plan)
        if not constraints['constraints_valid']:
            raise ValueError('Plan preparation proxy constraints failed: '+str(constraints))
        label=f'text26_n{n}_specialist_share_{share.numerator}of{share.denominator}'
        job=_job(config,original['workers'],plan,label,'calibration',51,bindings,30)
        job['purpose']='text26_exploratory_development_'+purpose
        job['proxy_constraints']=constraints
        jobs.append(job)
    packet=dict(schema='gsdeploy.static_experiments.v1',
        status='PREDECLARED_TEXT26_DEVELOPMENT_NOT_SCHEDULED',inputs=original['inputs'],
        config=raw,workers=original['workers'],bindings=bindings,
        frozen_files_sha256=quality_digest(bindings),jobs=jobs,gpu_started=False,
        discovery_protocol=str(destination/'discovery-protocol.json'),
        execution=dict(policy='Prepared only, no automatic execution. Future serial native collection uses unchanged frozenv5 source.',
                       runtime_source=str(SOURCE),scheduled=False))
    validation=validate_packet(packet)
    if not validation['valid'] or validation['jobs']!=4:
        raise ValueError('Native frozenv5 packet validation failed: '+str(validation))
    save(destination/'experiment-packet.json',packet)
    save(destination/'packet-validation.json',dict(native_validation=validation,
        implementation=_file(Path(static_study.__file__)),packet=_file(destination/'experiment-packet.json'),
        four_raw_proxy_constraints_pass=True,scheduled=False,gpu_started=False))
    save(destination/'manifest.json',dict(schema='gsdeploy.text26_preparation_manifest.v1',
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.is_file()},
        tool=_file(Path(__file__).resolve()),bindings_count=len(bindings),scheduled=False,gpu_started=False))
    print(json.dumps(dict(packet=str(destination/'experiment-packet.json'),validation=validation,
        n3_specialist_share=str(settings[-1][1]),bindings=len(bindings),scheduled=False,gpu_started=False)))


if __name__=='__main__':
    main()
