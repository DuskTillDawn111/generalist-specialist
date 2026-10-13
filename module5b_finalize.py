"""Post-run derived-metric audit. Never modifies search decisions or raw runs."""
import json
from pathlib import Path

from deployment_search.dynamic_real_pilot import OUT, PHASES, read
from deployment_search.neighborhood import deployment_key
from deployment_search.runtime_cache import file_hash, save_json


def acceptance_evaluation_index(record):
    """Count all Top-B evaluations preceding the actual accept decision.

    A final state first profiled at index 2 can be accepted only after index 3
    when both selected candidates are measured before the batch decision.
    The old first-profile index is preserved separately in finalized metrics.
    """
    if not record['runtime_accesses']:
        return None
    final = deployment_key(record['final_deployment'])
    initial_evaluated = deployment_key(record['runtime_accesses'][0]['deployment'])
    accepted_index = 1 if final == initial_evaluated else None
    count = 1
    for iteration in record['trace']['iterations']:
        count += iteration['num_runtime_candidates']
        if tuple(iteration['accepted'] or ()) == final:
            accepted_index = count
    return accepted_index


def finalize():
    metrics_path = OUT / 'cold_vs_warm_metrics.json'
    metrics = read(metrics_path)
    manifest = read(OUT / 'search_cache/runtime_cache_manifest.json')
    observations = []
    extra_errors = []
    for row in metrics['transitions']:
        phase = row['transition']
        for strategy, value in row['strategies'].items():
            record = read(OUT / phase / strategy / 'result.json')
            search = value['search']
            search['evaluations_to_first_profile_of_final_state'] = next((r['index']
                for r in record['runtime_accesses'] if r['deployment']==record['final_deployment']), None)
            search['evaluations_to_final_accepted_state'] = acceptance_evaluation_index(record)
            search['evaluations_before_physical_final_activation'] = record['trace']['runtime_evaluation_count']
            search['final_acceptance_semantics'] = 'search accept decision after full Top-B batch; physical activation occurs after search freezes'
            search['physical_attempts'] = sum(a['algorithm_id']==phase+'__'+strategy
                for a in manifest['run_attempts'])
            search['fresh_valid_state_measurements_owned_by_strategy'] = sum(
                a['algorithm_id']==phase+'__'+strategy and a.get('run_status')=='completed'
                for a in manifest['run_attempts']) // 3
            value['transition_starting_state_steady'] = (
                row['strategies']['D0']['performance']['stability_state_v2']=='steady'
                if row['strategies']['D0']['performance'] else None)
            result_path = OUT / phase / strategy / 'search_result_frozen.json'
            frozen = read(result_path)
            for key in ('final_deployment','final_objective','trace','switches'):
                if record[key] != frozen[key]:
                    extra_errors.append(f'{phase}/{strategy}: frozen decision changed: {key}')
        warm = read(OUT / phase / 'D2/result.json')
        trace = warm['trace']
        observations.append(dict(transition=phase,
            initial_feasible=trace['initial_feasible'],
            first_selected_top_b=trace['iterations'][0]['selected_top_b'] if trace['iterations'] else [],
            first_accepted_operation=next((dict(operation=c['mutation']['operation'],model=c['mutation']['model'])
                for t in trace['iterations'] for c in t['candidates'] if c['status']=='ACCEPTED'),None),
            accepted_removes=[c['mutation']['model'] for t in trace['iterations']
                for c in t['candidates'] if c['status']=='ACCEPTED' and c['mutation']['operation']=='REMOVE'],
            evaluated_mutations=[dict(iteration=t['iteration'],operation=c['mutation']['operation'],
                model=c['mutation']['model'],decision=c['status'],reason=c['reason'])
                for t in trace['iterations'] for c in t['candidates'] if c['runtime'] is not None],
            accepted_repairs=[r.get('accepted') for r in trace['repair_trace'] if r.get('accepted')],
            fallback_to_milp=warm['fallback_to_milp']))
    metrics['warm_mutation_observations'] = observations
    metrics['derived_metric_audit'] = dict(passed=not extra_errors, errors=extra_errors,
        search_decisions_preserved=True,
        evaluations_to_final_accepted_state='counts full evaluated Top-B batch before accepting final state',
        postprocessor_sha256=file_hash(__file__))
    save_json(metrics_path, metrics)
    events = read(OUT / 'reconfiguration_events.json')
    for event in events['events']:
        event['models_added'] = len(event['models_to_load'])
        event['models_removed'] = len(event['models_to_unload'])
        event['VRAM_added_mib'] = event['cost']['added_vram_mib']
        event['VRAM_removed_mib'] = event['cost']['removed_vram_mib']
        event['VRAM_moved_mib'] = event['VRAM_added_mib']+event['VRAM_removed_mib']
    save_json(OUT / 'reconfiguration_events.json', events)
    audit = read(OUT / 'decision_audit.json')
    audit['errors'].extend(extra_errors)
    audit['passed'] = not audit['errors']
    audit['post_confirmation_search_freeze_verified'] = True
    save_json(OUT / 'decision_audit.json', audit)
    summary = (OUT / 'module5b_summary.md').read_text()
    summary_lines = summary.splitlines()
    for row in metrics['transitions']:
        for strategy, value in row['strategies'].items():
            prefix = f"| {row['transition']} / {strategy} |"
            for i, line in enumerate(summary_lines):
                if line.startswith(prefix):
                    cells = line.split('|')
                    cells[-2] = (f" {value['search']['evaluations_to_first_steady_state']} / "
                                 f"{value['search']['evaluations_to_final_accepted_state']} ")
                    summary_lines[i] = '|'.join(cells)
    summary = '\n'.join(summary_lines)+'\n'
    lines = ['', '## Warm mutation observations 与接受时点审计', '']
    for observation in observations:
        phase=observation['transition']
        row=next(r for r in metrics['transitions'] if r['transition']==phase)
        lines.append(f"- {phase}：actual starting deployment feasible={observation['initial_feasible']}；first accepted mutation={observation['first_accepted_operation']}；accepted REMOVE={observation['accepted_removes']}；cold/warm evaluations to final accepted state={row['strategies']['D1']['search']['evaluations_to_final_accepted_state']}/{row['strategies']['D2']['search']['evaluations_to_final_accepted_state']}。")
        lines.append('  Evaluated mutations: '+ '; '.join(
            f"{m['operation']} {m['model']}: {m['decision']} ({m['reason']})"
            for m in observation['evaluated_mutations'])+'。')
    lines += ['', 'Final accepted-state evaluation index 按整批 Top-B 测完后的实际 accept 时点计算；首次 profile final 的 index 单独保存，二者不混用。上表中的 first steady 不变。',
        '该 index 指 search accept decision；accepted-path physical activation 在 search 冻结后执行。其前已完成 evaluation 数另存 evaluations_before_physical_final_activation，不冒充在线 time-to-ready。',
        f"Derived-metric audit={'PASS' if not extra_errors else 'FAIL'}；confirmation 后 final deployment、objective、trace 和 switches 与 frozen decision 逐项相同。", '',
        'Endpoint sensitivity（仅 structural，未执行该 alternative path）：']
    for strategy,value in metrics['totals']['endpoint_structural_sensitivity'].items():
        lines.append(f"- {strategy}: switches={value['switches']}, load={value['models_loaded']}, unload={value['models_unloaded']}, VRAM moved={value['VRAM_moved_mib']:.1f} MiB；measured time=null。")
    lines += ['', '## D0 / D1 / D2 final performance', '',
              '| Transition | Strategy | Weighted mean ms | Weighted task P95 ms | Stability |',
              '|---|---|---:|---:|---|']
    for row in metrics['transitions']:
        for strategy,value in row['strategies'].items():
            p=value['performance']
            lines.append(f"| {row['transition']} | {strategy} | {p['weighted_mean_e2e']:.3f} | {p['weighted_p95_e2e']:.3f} | {p['stability_state_v2']} |"
                         if p else f"| {row['transition']} | {strategy} | null | null | INFEASIBLE |")
    lines += ['', '本轮观察的是从已有良好部署继续搜索的价值。D2 在 T1 增加 DistilBERT 后拥有全部五模型；后续若 NO_SWITCH 且 D0 与 D2 相同，说明保留该部署已经足够，不证明每个 workload phase 都需要不同 specialist 组合。',
              '性能 gap 与成本统计比较的是冻结小预算 cold search，不能解释为对全局最优的 gap，也不能将 accepted-path reset 成本解释成所有 cold 系统唯一的执行方案。']
    (OUT / 'module5b_summary.md').write_text(summary+'\n'.join(lines)+'\n')
    save_json(OUT / 'derived_metric_audit.json', metrics['derived_metric_audit'])
    current_sources = read(OUT / 'source_model_hashes.json')
    historical_sources = read(OUT.parent / 'deployment_search_multiworkload/source_model_hashes.json')
    save_json(OUT / 'cache_compatibility_audit.json', dict(
        exact_source_identity_compatible=current_sources==historical_sources,
        differing_source_identity_fields=[k for k in current_sources if current_sources[k]!=historical_sources.get(k)],
        current_hardware_identity=current_sources['gpu_identity'],
        historical_hardware_identity=historical_sources['gpu_identity'],
        historical_performance_values_not_consulted=True))
    if extra_errors:
        raise RuntimeError('Post-run decision freeze audit failed')
    print(json.dumps(dict(decision=metrics['decision'],derived_metric_audit='PASS')))


if __name__=='__main__':
    finalize()
