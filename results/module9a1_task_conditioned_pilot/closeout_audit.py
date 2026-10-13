"""CPU-only closeout of completed, immutable pilot measurements."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/root/autodl-tmp/generalist-specialist')
OUT = ROOT / 'results/module9a1_task_conditioned_pilot'


def load(name):
    return json.loads((OUT / name).read_text())


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert load('pilot_state.json')['status'] == 'COMPLETE'
    frozen = load('execution_freeze.json')
    evidence = load('development_evidence_freeze.json')
    protocol = load('measurement_protocol_freeze.json')
    checks = []
    for group, refs in [
        ('8F2_artifacts', list(evidence['all_8F2_artifacts'].values())),
        ('prior_protected_inputs_and_sources', evidence['all_prior_protected_references']),
        ('pilot_core', frozen['core_files']),
        ('pilot_execution_sources', frozen['execution_sources']),
        ('selected_model_assets', load('pre_GPU_asset_checks.json')['checks']),
    ]:
        for ref in refs:
            expected = ref.get('sha256', ref.get('expected'))
            actual = sha(ref['path'])
            row = {'group': group, 'path': ref['path'], 'expected': expected,
                   'actual': actual, 'match': expected == actual}
            checks.append(row)
            assert row['match'], row
        print('HASH_GROUP_PASS', group, len(refs), flush=True)
    assert sha(protocol['input_pool']) == protocol['input_pool_sha256']
    assert not evidence['future_v3_heldout_validation_eligible']
    protocol_hash = sha(OUT / 'measurement_protocol_freeze.json')
    measurements = []
    for run in frozen['plan']:
        model, task, co, seed = (run[k] for k in ['model', 'task', 'co', 'seed'])
        directory = 'task_path_kernel_profiles/kernel' if run['profile'] else (
            'pair_measurements/pair' if co else 'pair_measurements/standalone')
        tag = f'{model}__{task}__{co or "none"}__{seed}'
        raw = load(directory + '/' + tag + '.json')
        assert raw['status'] == 'COMPLETE'
        assert raw['protocol_sha256'] == protocol_hash
        assert (raw['model'], raw['task'], raw['co_model'], raw['seed']) == (model, task, co, seed)
        expected_count = protocol['signature_profile_samples'] if run['profile'] else protocol['target_sample_count']
        targets = raw['target_records']
        assert len(targets) == expected_count
        expected_ids = [r['pool_id'] for r in protocol['input_subsets'][str(seed)][task]][:expected_count]
        assert [r['pool_id'] for r in targets] == expected_ids
        assert [r['sample_index'] for r in targets] == list(range(expected_count))
        assert all(math.isfinite(r['worker_latency_ms']) and r['worker_latency_ms'] > 0 for r in targets)
        assert raw['warmup_configured_seconds'] == protocol['warmup_seconds']
        assert not raw['profiler_timing_ground_truth']
        assert raw['gpu_release']['verified']
        assert all(r['exit_code'] == 0 and not r['forced'] for r in raw['shutdown'].values())
        assert raw['actual_peak_vram_mib'] <= 49140
        assert not raw['telemetry_errors']
        if not run['profile']:
            assert raw['measurement_seconds'] == protocol['measurement_seconds']
            assert raw['configured_window_completed']
            if co:
                assert not raw['overlap_filter_applied']
                assert raw['all_targets_meet_overlap_gate']
                assert all(r['active_overlap_ratio'] >= protocol['minimum_overlap_ratio'] for r in targets)
                assert {r['task_type'] for r in raw['co_records']} == set(protocol['co_tasks'][co])
        if run['profile']:
            for suffix in ['.sqlite', '.nsys-rep', '.kernels.json']:
                assert (OUT / directory / (tag + suffix)).stat().st_size > 0
        measurements.append({'model': model, 'task': task, 'co': co, 'seed': seed,
                             'profile': run['profile'], 'target_samples': len(targets),
                             'peak_vram_mib': raw['actual_peak_vram_mib'], 'status': 'PASS'})
    assert len(measurements) == 60
    assert sum(r['profile'] for r in measurements) == 15
    assert sum(not r['profile'] for r in measurements) == 45
    assert sum(r['target_samples'] for r in measurements if not r['profile']) == 1350
    required = ['prior_art_boundary.md', 'failure_map.csv', 'pilot_selection.json', 'noise_audit.json',
                'task_path_kernel_summary.csv', 'task_path_kernel_distance.csv', 'task_conditioned_slowdown.csv',
                'statistical_tests.json', 'mechanism_correlation.json', 'pilot_gate.json', 'module9a1_summary.md']
    assert all((OUT / n).stat().st_size > 0 for n in required)
    distances = list(csv.DictReader((OUT / 'task_path_kernel_distance.csv').open()))
    slowdowns = list(csv.DictReader((OUT / 'task_conditioned_slowdown.csv').open()))
    summaries = list(csv.DictReader((OUT / 'task_path_kernel_summary.csv').open()))
    assert (len(distances), len(slowdowns), len(summaries)) == (4, 10, 5)
    tests = load('statistical_tests.json')['tests']
    assert len(tests) == 8
    noise = load('noise_audit.json')
    floor = noise['historical_noise_floor']
    for t in tests:
        expected = t['CI_excludes_zero'] and abs(t['Delta_S']) > floor and t['valid_controlled_pair_overlap']
        assert t['TASK_CONDITIONED_INTERFERENCE_DETECTED'] == expected
    gate = load('pilot_gate.json')
    assert gate['status'] == 'GATE_PARTIAL'
    assert not gate['recommend_full_Module9A']
    maximum = max(tests, key=lambda t: t['abs_Delta_S'])
    floor_driver = max((r for r in noise['path_audits'] if r['included_in_gate_floor']),
                       key=lambda r: r['bootstrap_null_abs_deltaS_p95'])
    coverage = {model + '/' + task: sum(r['model'] == model and r['task'] == task
                                      for r in noise['path_audits'])
                for model, tasks in protocol['tasks'].items() for task in tasks}
    reasons = {
        'gate': gate['status'], 'stable_kernel_contrasts': sum(r['stable_structural_difference'] == 'True' for r in distances),
        'CI_excludes_zero_contrasts': sum(t['CI_excludes_zero'] for t in tests),
        'exceeds_frozen_historical_noise_floor_contrasts': sum(t['exceeds_historical_noise_floor'] for t in tests),
        'detected_contrasts': sum(t['TASK_CONDITIONED_INTERFERENCE_DETECTED'] for t in tests),
        'maximum_interference_contrast': maximum, 'historical_noise_floor': floor,
        'historical_floor_driving_group': floor_driver, 'historical_steady_repeat_coverage_by_pilot_path': coverage,
        'interpretation': 'Stable task-path kernel differences and several nonzero pair differences are observed. '
                          'Every absolute pair difference remains below the pre-GPU historical noise floor, '
                          'which conservatively transfers the worst repeated multi-model steady-state variability. '
                          'This gate does not establish absence of task conditioning, and kernel distance does not prove causation.',
        'limitations': ['Only three independent repetitions per condition',
                       'Historical multi-model noise floor is not a directly estimated controlled-pair noise threshold',
                       'All five pilot paths have historical steady repeat evidence; the frozen gate nevertheless uses one conservative global floor',
                       'Fixed saturated co-backlog does not guarantee identical achieved co-throughput',
                       'Handler overlap does not establish simultaneous SM execution; host/context effects remain possible',
                       'Two Generalist clusters are insufficient for inferential mechanism correlation'],
        'decision': 'STOP_AFTER_PILOT; no automatic full Module 9A or full v3',
        'threshold_or_selection_changed': False, 'additional_GPU_runs': 0,
    }
    save('partial_gate_reason_audit.json', reasons)
    distortion = load('profiler_timing_distortion_audit.json')
    distortion['arrival_cadence_confound'] = {
        'profiled_samples': 'Three sequential matched requests after warmup',
        'unprofiled_comparison_samples': 'First three matching requests, scheduled seven seconds apart',
        'consequence': 'Ratios combine tracing/instrumentation, cadence, cache/clock and run variability; '
                       'they cannot identify pure causal profiler overhead.',
        'ground_truth': 'All real standalone and pair latency denominators/numerators use only unprofiled runs; frozen s0 remains unchanged.',
    }
    save('profiler_timing_distortion_audit.json', distortion)
    groups = {g: sum(r['group'] == g for r in checks) for g in sorted({r['group'] for r in checks})}
    closeout = {'status': 'PASS', 'CPU_only': True, 'hash_groups': groups, 'hash_checks': checks,
                'all_measurement_protocol_checks_passed': True, 'measurement_checks': measurements,
                'profiled_runs': 15, 'unprofiled_runs': 45, 'unprofiled_target_requests': 1350,
                'minimum_pair_overlap': min(float(r['minimum_active_overlap_ratio']) for r in slowdowns),
                'maximum_peak_vram_mib': max(r['peak_vram_mib'] for r in measurements),
                'all_processes_normal_exit_and_GPU_release_verified': True,
                'required_artifacts_present': required,
                '8F2_future_heldout_validation_eligible': False, 'new_GPU_runs_during_closeout': 0}
    save('closeout_validation.json', closeout)
    validation = load('validation_results.json')
    validation['post_campaign_closeout'] = {'status': 'PASS', 'hash_groups': groups,
                                          'measurement_protocol_checks': 60, 'additional_GPU_runs': 0}
    save('validation_results.json', validation)
    peak = closeout['maximum_peak_vram_mib']
    lines = ['','## PARTIAL gate 原因审计与最终核验','',
             f'4/4 个 kernel contrasts 有稳定结构差异；7/8 个 Delta-S 的 95% CI 不跨 0；'
             f'0/8 个 absolute Delta-S 超过 frozen historical noise floor={floor:.12f}。'
             f'最大 absolute Delta-S={maximum["abs_Delta_S"]:.12f}，仍低于门槛；不改阈值，不追加 GPU。',
             f'历史门槛由 {floor_driver["model"]}/{floor_driver["task"]} 的同状态 steady 重复驱动；'
             f'CV={floor_driver["latency_cv"]:.6f}，run means={floor_driver["run_means_ms"]} ms。'
             '这是跨路径/多模型历史噪声的保守迁移，不是本 controlled pair 的局部噪声估计。'
             '五个 pilot paths 均有历史 steady 重复证据；判定仍按预先冻结的全局门槛执行。',
             'Profiler 计时有可见差异，但 profiling 的三次请求连续执行，unprofiled 对照每 7 秒到达；'
             '比值包含 profiler、到达节奏、缓存/时钟与 run variability，不能当作纯 profiler overhead。'
             'Profile timing 未用于 s0、S_real 或 gate。',
             f'CPU closeout PASS：hash groups={groups}；60/60 runs 的输入身份、样本数、warmup、计时窗口、'
             f'pair overlap、正常退出/GPU 释放等检查通过。最大 peak={peak} MiB；'
             f'minimum pair overlap={closeout["minimum_pair_overlap"]:.9f}。',
             '决定：停止 pilot，审计原因，不自动进入完整 Module 9A。8F-2 继续永久冻结并仅作 development/failure evidence。']
    with (OUT / 'module9a1_summary.md').open('a') as stream:
        stream.write('\n'.join(lines) + '\n')
    save('artifact_hashes.json', {str(x.relative_to(OUT)): {'path': str(x.resolve()), 'sha256': sha(x)}
                                 for x in sorted(OUT.rglob('*')) if x.is_file()
                                 and x.name not in ['artifact_hashes.json', 'execution.log']
                                 and '__pycache__' not in x.parts})
    print(json.dumps({'status': 'PASS', 'hash_groups': groups, 'raw_runs': len(measurements),
                      'gate': gate['status'], 'maximum_peak_vram_mib': peak}), flush=True)


if __name__ == '__main__':
    main()
