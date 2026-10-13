"""Offline report clarification; never launch a worker or change frozen decisions.

Run only after the frozen campaign has completed. Cold adapter smoke memory is
kept separate from post-warmup search/confirmation memory. Linear comparisons
reuse the visited history at each actual incumbent, so tabu rules match.
"""
from .prepare import OUT, read, verify_freeze
from deployment_search.runtime_cache import save_json, file_hash
from deployment_core.replicas import replica_key
from deployment_core import Workload
from candidate_pool_service_demand.dry_run import load_completed_catalog
from deployment_core.replica_calibration import load_replica_calibration
from deployment_search import IntegerGuidedDeploymentSearch, SearchConfig


def main():
    assert read(OUT / 'campaign_status.json')['status'] == 'complete'
    verify_freeze()
    memory = read(OUT / 'memory_validation.json')
    warm, cold = [], []
    for row in memory['runs']:
        raw = read(row['path'])
        row['warmup_seconds'] = raw['warmup_seconds']
        row['idle_measurement_phase'] = ('post_warmup' if raw['warmup_seconds'] > 0
                                         else 'cold_pre_inference_adapter_smoke')
        (warm if raw['warmup_seconds'] > 0 else cold).append(row)
    memory['phase_aware_validation'] = {
        'post_warmup_run_count': len(warm), 'cold_smoke_run_count': len(cold),
        'post_warmup_idle_within_prefrozen_tolerance': all(
            r['within_prefrozen_5percent_idle_diagnostic_tolerance'] is True for r in warm),
        'post_warmup_max_idle_relative_deviation': max(
            abs(r['idle_relative_difference']) for r in warm),
        'post_warmup_max_actual_idle_mib': max(r['actual_idle_mib'] for r in warm),
        'all_runs_max_peak_device_mib': max(r['peak_device_mib'] for r in memory['runs']),
        'note': ('Cold smoke has zero warmup and is not comparable to a post-warmup idle table. '
                 'All original measurements and the unstratified diagnostic are retained; '
                 'no memory table or hard budget has been changed.')}
    save_json(OUT / 'memory_validation.json', memory)

    trace = read(OUT / 'search_trace.json')
    protocol = read(OUT / 'pilot_protocol.json')
    catalog, hardware, roles, _, _ = load_completed_catalog()
    cal = load_replica_calibration(catalog)
    workload = Workload(protocol['workload']['arrival_rates_rps'])
    visited = {replica_key(trace['seed']['deployment'])}
    comparisons = []
    for t in trace['iterations']:
        linear = IntegerGuidedDeploymentSearch().search(
            workload, catalog, hardware, SearchConfig(dry_run=True),
            initial_deployment=t['deployment'], phase_visited_deployments=set(visited),
            roles=roles, capacity_mode='linear', memory_mode='empirical', calibration=cal)
        linear_top = [a.mutation.deployment for a in linear.iterations[0]['candidates']
                      if a.selected_top_b]
        empirical_top = [a['mutation']['deployment'] for a in t['candidates']
                         if a['selected_top_b']]
        comparisons.append({
            'iteration': t['iteration'], 'fixed_incumbent': t['deployment'],
            'prior_visited_states': [dict(k) for k in sorted(visited)],
            'linear_Top_B': linear_top, 'empirical_Top_B': empirical_top,
            'selection_changed': linear_top != empirical_top,
            'actually_evaluated': [a['mutation']['deployment'] for a in t['candidates']
                                  if a['runtime'] is not None],
            'scope': 'CPU-only planning counterfactual; matching prior visited history, '
                     'memory/evidence bounds and incumbent. No linear GPU experiment.'})
        visited.update(replica_key(a['mutation']['deployment']) for a in t['candidates']
                       if a['runtime'] is not None)
    save_json(OUT / 'linear_counterfactual_audit.json', {
        'comparisons': comparisons, 'source_trace_sha256': file_hash(OUT / 'search_trace.json'),
        'postprocessor_sha256': file_hash(__file__), 'GPU_runs_added': 0})
    decisions = read(OUT / 'integer_replica_decisions.json')
    decisions['linear_vs_empirical_with_matching_visited_history'] = comparisons
    save_json(OUT / 'integer_replica_decisions.json', decisions)

    path = OUT / 'module7c3_summary.md'
    lines = path.read_text().splitlines()
    phase = memory['phase_aware_validation']
    for index, line in enumerate(lines):
        if line.startswith('10. '):
            lines[index] = (f'10. Search/confirmation 的 {len(warm)} 次 post-warmup idle '
                f'在冻结 5% tolerance 内：{phase["post_warmup_idle_within_prefrozen_tolerance"]}；'
                f'最大相对偏差={phase["post_warmup_max_idle_relative_deviation"]:.6%}。'
                f'所有 runs 最大 sampled device peak={phase["all_runs_max_peak_device_mib"]} MiB；'
                f'peaks≤49140：{memory["all_peaks_within_budget"]}。'
                '3 次 correctness smoke 无 warmup，cold idle 不与 post-warmup 表混比；'
                '原始 cold 数值及全样本诊断保留在 memory_validation.json。')
        if line.startswith('8. '):
            lines[index] = ('8. Calibration 改变 Top-B：'
                f'{[c["selection_changed"] for c in comparisons]}（按轮）。'
                'CPU linear counterfactual 使用相同实际 incumbent、memory/evidence bounds '
                '和 prior visited history；见 linear_counterfactual_audit.json。'
                '没有 linear GPU campaign，不宣称其完整运行路径或性能。')
    lines += ['', 'Offline report clarification：memory 按 cold/post-warmup phase 分层；'
        'linear counterfactual 匹配 visited history。原 search decision、raw metrics、'
        'pilot protocol、calibration 和 source freeze 均未改动，没有新增 GPU run。']
    lines += ['', '## 已观察的 search runtime states', '',
        '| Deployment（未列 types=0） | Repetitions | Stability | Mean E2E ms | Weighted task P95 ms | Planning idle MiB | Actual idle range MiB | Sampled device peak MiB |',
        '|---|---:|---|---:|---:|---:|---|---:|']
    for entry in read(OUT / 'runtime_evaluations.json')['evaluations']:
        if entry['status'] != 'complete':
            continue
        rt = entry['runtime_result']
        label = ', '.join(f'{m}={n}' for m, n in entry['identity']['deployment'].items() if n)
        observations = rt['memory_observations']
        lines.append(f'| {label} | {len(rt["repetitions"])} | {rt["stability_state_v2"]} '
            f'| {rt["weighted_mean_e2e"]:.3f} | {rt["weighted_p95_e2e"]:.3f} '
            f'| {observations[0]["planning_idle_mib"]:.0f} '
            f'| {min(o["actual_idle_mib"] for o in observations):.0f}–'
            f'{max(o["actual_idle_mib"] for o in observations):.0f} '
            f'| {max(o["peak_device_mib"] for o in observations):.0f} |')
    lines += ['', '没有对被 Top-B 裁剪的 Qwen=2 state 做 W2 runtime measurement，'
        '因此本 pilot 不能给出复制 Qwen 与新增 Specialist 的 W2 因果性能比较。'
        '三次双副本 smoke 只验证 adapter correctness；最终选多个 model types '
        '仍是合法整数搜索结果，不视为 replica-count 支持失败。']
    path.write_text('\n'.join(lines) + '\n')
    manifest = read(OUT / 'artifact_manifest.json')
    manifest['offline_report_postprocessor'] = {'path': __file__, 'sha256': file_hash(__file__),
                                              'GPU_runs_added': 0}
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ('.json', '.md') and p.name != 'artifact_manifest.json':
            manifest['artifacts'][p.name] = {'path': str(p), 'sha256': file_hash(p)}
    save_json(OUT / 'artifact_manifest.json', manifest)


if __name__ == '__main__':
    main()
