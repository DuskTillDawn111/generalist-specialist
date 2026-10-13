"""Module 3B bounded single-workload closed loop. Run only by explicit pilot request."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path

from deployment_core.models import load_hardware, load_model_catalog
from deployment_core.workload import Workload
from . import GuidedDeploymentSearch, SearchConfig
from .measurement_runtime import MeasurementRuntimeEvaluator, MeasurementRuntimeEvaluation
from .neighborhood import memory_mib
from .pilot_provenance import asset_hashes, source_hashes, verify_assets_unchanged
from .real_objective import NoiseAwareRuntimeObjective
from .result import jsonable
from .runtime_cache import save_json, file_hash, digest
from .tests.fixtures import forbid_enumeration

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/deployment_search_real_pilot'
SETTINGS = dict(warmup_seconds=10, measurement_seconds=210,
                arrival_process='deterministic_periodic', routing_source='module3a_canonical_minmax_lp',
                startup_timeout_seconds=300, drain_timeout_seconds=60,
                shutdown_timeout_seconds=10, max_pending_per_model=10000)
SEEDS = (7101, 7202, 7303)
CONFIG = SearchConfig(top_b=2, max_iterations=3, no_improvement_patience=1,
                      runtime_evaluation_budget=6, rho_max=.8, remove_pressure_threshold=.2,
                      mean_tolerance_ms=1.)
MIN_RELATIVE_IMPROVEMENT = .05


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != jsonable(value):
            raise RuntimeError('Frozen pilot input changed: ' + str(path))
    else:
        save_json(path, value)


def prepare(output=OUT):
    output.mkdir(parents=True, exist_ok=True)
    catalog = load_model_catalog()
    source = ROOT / 'results/performance_model_comparable_groups/comparison_groups.json'
    groups = json.loads(source.read_text())['groups']
    group = next(g for g in groups if g['workload_group'] == 'W1_balanced')
    w = Workload(group['workload'])
    historical = []
    for c in group['configurations']:
        for r in c['repetitions']:
            path = Path(r['source_result_path'])
            historical.append(dict(deployment=c['deployment'], stability=r['stability_state_v2'],
                                   completion_ratio=r['system_metrics']['completion_ratio'],
                                   raw_result_path=str(path), sha256=file_hash(path)))
    if len(historical) != 15 or any(r['stability'] != 'steady' for r in historical):
        raise RuntimeError('W1 historical steady selection evidence changed')
    workload_record = dict(
        name=group['workload_group'], task_arrival_rates=dict(w.arrival_rate),
        workload_source=dict(path=str(source), sha256=file_hash(source)),
        selection_reason='Existing W1_balanced: all 15 Module 2G runs (5 configurations x 3) steady, '
                         'completion ratio 1.0; text/image/OCR/ASR/LID can split generalist and specialists. '
                         'W2 and W3 have unstable/incomplete runs and are excluded. No load scale tuning.',
        known_historical_stability=dict(counts=dict(Counter(r['stability'] for r in historical)), runs=historical),
        standalone_service_demand_estimate={m: {k: w.arrival_rate[k] * p.service_time_ms[k] / 1000
                                                for k in p.supported_tasks} for m, p in catalog.items()},
        qwen_only_total_pressure=sum(w.arrival_rate[k] * catalog['qwen'].service_time_ms[k] / 1000
                                     for k in w.active_tasks),
        estimate_note='Offered rate times standalone service seconds by eligible path, not additive across alternative executors. '
                      'Qwen-only sum is a planning estimate, not proof of current runtime stability.',
        measurement_seconds=210, historical_cache_eligibility='Historical raw records lack model weight content hashes. '
            'Cannot authenticate exact source/model identity; not imported into exact cache. Routing source labels alone '
            'are not treated as evidence of a substantive measurement change.',
        real_exhaustive_optimum='unavailable', no_new_exhaustive_baseline=True)
    freeze(output / 'pilot_workload.json', workload_record)
    settings_record = dict(search=jsonable(CONFIG), max_unique_runtime_evaluations=6,
                          repetitions_per_evaluation=3, max_real_runs=18, predictor=None,
                          measurement_settings=SETTINGS, repetition_seeds=list(SEEDS),
                          min_relative_improvement=MIN_RELATIVE_IMPROVEMENT,
                          noise_policy='Stability improvement bypasses the mean gate. Same-steady latency-based acceptance '
                          'requires >=5% workload-weighted mean gain; P95 cannot bypass the gate. Memory-only ties '
                          'retain the Module 3A rule. Std is recorded, not used in a significance test.',
                          aggregation='Worst stability across all three runs; equal-weight repetition mean; sample std ddof=1. '
                          'Never discard failed repetitions. Missing aggregate mean/P95 if any repeat lacks it.',
                          launch_policy='Sequential; no retries or replacement candidates. Persistent reservations count toward budget.')
    freeze(output / 'search_config.json', settings_record)
    ap = output / 'asset_hashes.json'
    if not ap.exists():
        save_json(ap, asset_hashes())
    assets = json.loads(ap.read_text())
    verify_assets_unchanged(assets)
    sources = source_hashes(assets)
    freeze(output / 'source_model_hashes.json', sources)
    freeze(output / 'pilot_freeze.json', dict(
        files={name: file_hash(output / name) for name in ('pilot_workload.json', 'search_config.json',
                                                          'source_model_hashes.json', 'asset_hashes.json')},
        search_sources={str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / 'deployment_search').glob('*.py'))},
        frozen_before_first_run=True))
    return w, catalog, load_hardware(), sources


def decision_audit(search, manifest, catalog, objective):
    errors, decisions = [], []
    def require(condition, reason):
        if not condition:
            errors.append(reason)
    require(len(manifest['configurations']) <= 6, 'unique configuration budget')
    require(len(manifest['run_attempts']) <= 18, 'real run budget')
    attempts = manifest['run_attempts']
    pairs = [(r['cache_key'], r['repetition']) for r in attempts if not r.get('interrupted_external')]
    require(len(pairs) == len(set(pairs)), 'duplicate completed/active measurement attempt')
    require(search['milp_solve_count'] == 1, 'MILP seed count')
    require(search['seed']['deployment'] == search['iterations'][0]['deployment'], 'seed does not initialize search')
    for t in search['iterations']:
        require(t['num_candidates_generated'] <= len(catalog), 'nonlinear candidate count')
        current = MeasurementRuntimeEvaluation(**t['current_runtime'])
        current_mem = memory_mib(t['deployment'], catalog)
        ranked = sorted((c for c in t['candidates'] if c['rank'] is not None), key=lambda c: c['rank'])
        require([c['delta_z'] for c in ranked] == sorted((c['delta_z'] for c in ranked), reverse=True), 'LP primary ranking')
        selected = [c for c in ranked if c['selected_top_b']]
        require(selected == ranked[:2], 'Top-B selection changed')
        best = None
        for c in t['candidates']:
            require(sum(c['mutation']['deployment'][m] != t['deployment'][m] for m in catalog) == 1, 'non one-hop mutation')
            if c['canonical'] is not None:
                require(abs(c['delta_z'] - (t['canonical']['max_pressure'] - c['canonical']['max_pressure'])) < 1e-10,
                        'delta_z inconsistency')
            if c['runtime'] is not None:
                require(c['selected_top_b'], 'runtime outside Top-B')
        for c in selected:
            if c['runtime'] is None:
                require(c['status'] == 'NOT_EVALUATED_BUDGET', 'selected candidate lacks measurement without budget reason')
                continue
            runtime = MeasurementRuntimeEvaluation(**c['runtime'])
            better, reason = objective.compare(runtime, current, c['memory_mib'], current_mem)
            gain = ((current.weighted_mean_e2e - runtime.weighted_mean_e2e) / current.weighted_mean_e2e
                    if current.weighted_mean_e2e and runtime.weighted_mean_e2e is not None else None)
            decisions.append(dict(iteration=t['iteration'], deployment=c['mutation']['deployment'],
                LP_rank=c['rank'], delta_z=c['delta_z'], current_state=current.stability_state_v2,
                candidate_state=runtime.stability_state_v2, current_mean=current.weighted_mean_e2e,
                candidate_mean=runtime.weighted_mean_e2e, relative_mean_gain=gain,
                current_std=current.replicate_statistics['weighted_mean_e2e']['sample_std'],
                candidate_std=runtime.replicate_statistics['weighted_mean_e2e']['sample_std'],
                current_p95=current.weighted_p95_e2e, candidate_p95=runtime.weighted_p95_e2e,
                objective_better_than_incumbent=better, objective_reason=reason,
                actual_decision=c['status'], recorded_reason=c['reason'], raw_result_paths=runtime.raw_result_paths))
            if better and (best is None or objective.compare(runtime, MeasurementRuntimeEvaluation(**best['runtime']),
                                                              c['memory_mib'], best['memory_mib'])[0]):
                best = c
        require((best is None and t['accepted'] is None) or
                (best is not None and best['status'] == 'ACCEPTED' and
                 sorted(m for m, n in best['mutation']['deployment'].items() if n) == t['accepted']),
                f"accept/rollback replay mismatch iteration {t['iteration']}")
        for c in selected:
            if c['runtime'] is not None and c is not best:
                require(c['status'] == 'ROLLBACK', 'nonwinner measurement not retained as rollback')
    for key, entry in manifest['configurations'].items():
        require(entry['status'] == 'complete' and len(entry['repetitions']) == 3, 'incomplete configuration aggregation')
        require(digest(entry['identity']) == key, 'cache key mismatch')
        require(file_hash(entry['aggregate']) == entry['aggregate_sha256'], 'aggregate hash drift')
        for r in entry['repetitions']:
            require(file_hash(r['runtime']['raw_result_path']) == r['runtime']['raw_sha256'], 'raw hash drift')
    return dict(passed=not errors, errors=errors, decisions=decisions,
                duplicate_measurements=len(pairs) - len(set(pairs)), unique_runtime_evaluations=len(manifest['configurations']),
                actual_real_runs=len(attempts), externally_interrupted_runs=sum(bool(r.get('interrupted_external')) for r in attempts),
                valid_repetition_runs=len(pairs), enumeration_calls=0,
                proof='Runtime tripwires forbade enumeration during search; decisions replayed from saved LP ranks and runtime aggregates.')


def write_reports(output, result, enumeration_calls):
    search = result.to_dict()
    manifest = json.loads((output / 'runtime_cache_manifest.json').read_text())
    evaluations = [dict(cache_key=key, identity=entry['identity'],
                        result=json.loads(Path(entry['aggregate']).read_text()))
                   for key, entry in manifest['configurations'].items()]
    save_json(output / 'runtime_evaluations.json', dict(evaluations=evaluations,
              unique_runtime_evaluations=len(evaluations), actual_real_runs=len(manifest['run_attempts']),
              externally_interrupted_runs=sum(bool(r.get('interrupted_external')) for r in manifest['run_attempts'])))
    audit = decision_audit(search, manifest, load_model_catalog(), NoiseAwareRuntimeObjective(1., .05))
    audit['enumeration_calls'] = enumeration_calls
    audit['passed'] &= enumeration_calls == 0
    save_json(output / 'decision_audit.json', audit)
    save_json(output / 'search_trace.json', search)
    seed = search['seed_runtime']; final = search['runtime']
    def active(x):
        return '+'.join(m for m, n in x.items() if n)
    def fmt(v):
        return 'unavailable' if v is None else f'{v:.3f}'
    gain = ((seed['weighted_mean_e2e'] - final['weighted_mean_e2e']) / seed['weighted_mean_e2e']
            if seed['weighted_mean_e2e'] and final['weighted_mean_e2e'] is not None else None)
    trace = ['# Module 3B search trace', '', 'Predictor=None；全部候选来自 Module 3A one-hop + LP Top-B。', '']
    for t in search['iterations']:
        current = t['current_runtime']
        trace += [f"## Iteration {t['iteration']}", '', f"Current: {active(t['deployment'])}",
                  f"Runtime: {current['stability_state_v2']}, mean={fmt(current['weighted_mean_e2e'])} ms, "
                  f"P95={fmt(current['weighted_p95_e2e'])} ms", '',
                  f"LP z={t['canonical']['max_pressure']:.9f}; bottleneck={t['feedback']['bottleneck_models']}; pressure tasks={t['feedback']['pressure_tasks']}", '',
                  '| Mutation | z | delta_z | LP rank | Decision | Reason |', '|---|---:|---:|---:|---|---|']
        for c in t['candidates']:
            trace.append(f"| {c['mutation']['operation']} {c['mutation']['model']} | "
                         f"{fmt(c['canonical']['max_pressure']) if c['canonical'] else '—'} | {fmt(c['delta_z'])} | "
                         f"{c['rank']} | {c['status']} | {c['reason']} |")
        trace += ['', 'Top-B: ' + '; '.join('+'.join(x) for x in t['selected_top_b']), '',
                  'Accepted: ' + ('+'.join(t['accepted']) if t['accepted'] else 'none; retain current'), '']
        for d in (d for d in audit['decisions'] if d['iteration'] == t['iteration']):
            trace.append(f"- {active(d['deployment'])}: mean {fmt(d['candidate_mean'])} ± {fmt(d['candidate_std'])} ms; "
                         f"P95 {fmt(d['candidate_p95'])} ms; {d['candidate_state']}; {d['actual_decision']} ({d['recorded_reason']}).")
        trace += ['', '完整 routing、pressure、contribution、三次 raw paths 均保存在 search_trace.json。', '']
    (output / 'search_trace.md').write_text('\n'.join(trace))
    accepted = [d for d in audit['decisions'] if d['actual_decision'] == 'ACCEPTED']
    summary = ['# Module 3B：Real-runtime Closed-loop Search Pilot', '',
               f"状态：{search['status']}；decision audit={'PASS' if audit['passed'] else 'FAIL'}。单 workload：W1_balanced。", '',
               f"1. Seed deployment：{active(search['seed']['deployment'])}，由本次 MILP 得到并重解 canonical routing。",
               f"2. 初始 bottleneck：{search['iterations'][0]['feedback']['bottleneck_models']}；z={search['iterations'][0]['canonical']['max_pressure']:.9f}。",
               '3. 每轮生成候选数：' + ', '.join(f"iteration {t['iteration']} = {t['num_candidates_generated']}" for t in search['iterations']) + '。',
               '4. LP Top-B：' + '; '.join(f"iteration {t['iteration']}: " + ', '.join('+'.join(x) for x in t['selected_top_b']) for t in search['iterations']) + '。',
               '5. 真实运行 states：' + '; '.join(active(e['identity']['deployment']) for e in evaluations) + '；每个 state 三次有效重复。',
               '6. Accept/rollback：' + '; '.join(f"iteration {d['iteration']} {active(d['deployment'])}: {d['actual_decision']} ({d['recorded_reason']})" for d in audit['decisions']) + '。',
               f"7. 最终 deployment：{active(search['deployment'])}。",
               f"8. Weighted mean E2E：{fmt(seed['weighted_mean_e2e'])} → {fmt(final['weighted_mean_e2e'])} ms；改善 " + (f'{gain:.2%}。' if gain is not None else 'unavailable。'),
               f"9. Weighted task P95：{fmt(seed['weighted_p95_e2e'])} → {fmt(final['weighted_p95_e2e'])} ms（不是 pooled system P95）。",
               f"10. Stability：{seed['stability_state_v2']} → {final['stability_state_v2']}；采用三次重复中的最差状态。",
               f"11. LP solves：{search['lp_solve_count']}（含 seed canonical LP，另有 {search['milp_solve_count']} 次 MILP；审计未重解 LP）。",
               f"12. Unique runtime evaluations：{audit['unique_runtime_evaluations']} / 6。",
               f"13. 真实 run attempts：{audit['actual_real_runs']} / 18；其中外部关机中断 {audit['externally_interrupted_runs']} 次，不作 runtime 稳定性标签。",
               f"14. 重复的完整测量：{audit['duplicate_measurements']}；关机中断后补测 "
               f"{audit['externally_interrupted_runs']} 次；cache hits={manifest['cache_hits']}（包含最终状态的显式缓存复用检查）。",
               f"15. Enumeration calls：{enumeration_calls}；未运行真实 exhaustive baseline。",
               '16. 初步支持范围：' + ('闭环执行、预算、LP pruning、runtime 接受/回滚与可重建审计通过。' if audit['passed'] else '审计未通过，需按 decision_audit.json 解释，不能宣称正确性验证成功。')
               + ('本轮接受了真实改善。' if accepted else '本轮保留 seed，候选按 runtime objective 回滚；这是合法的闭环行为。'), '',
               '## 重复波动与解释边界', '',
               f"Seed mean sample std={fmt(seed['replicate_statistics']['weighted_mean_e2e']['sample_std'])} ms；"
               f"final mean sample std={fmt(final['replicate_statistics']['weighted_mean_e2e']['sample_std'])} ms。",
               'Seed 的第三次 mean 约 38.4 秒，显著抬高其三次平均值；99.35% 是这批原始测量的样本内改善，不能解读为稳定、可泛化的常态加速比。',
               'Latency 接受需至少 5% mean 改善；稳定性改善优先。未进行统计显著性检验；顺序运行和环境漂移仍可能影响结果。',
               '所有 repetitions、worker errors、incomplete、drain timeout 与不稳定结果均保留。外部关机中断单独列账，不作模型 unstable；有效重复补足后聚合。三态使用既有 stability_v2 阈值，无 predictor 或人工候选补选。',
               '历史 W1 的 15 次运行均 steady，是选 workload 的依据，不保证本次稳定。历史 raw 缺少模型权重内容哈希，无法认证 exact-cache identity；未冒充本轮缓存命中。',
               '**Real exhaustive optimum = unavailable**。历史数据只覆盖 seed 与少量一跳配置，且没有完整来源哈希；未为 ground truth 新跑 31 个 deployments。',
               '这是单 workload、小预算 pilot；不证明全局最优，也不构成 final evaluation。Module 2 predictor 继续为 optional，主实验 predictor=None。', '',
               '已停止于 Module 3B；未进入多 workload、routing optimization、multi-replica、forecasting、predictor retraining 或 paper final benchmark。', '']
    correction_path = output / 'recovery_correction.json'
    if correction_path.exists():
        correction = json.loads(correction_path.read_text())
        summary.extend(['## 关机中断修正', '',
                        correction['reason'],
                        f"原配置上限 {correction['initial_plan_configuration_limit']}，修正后可测上限 "
                        f"{correction['effective_configuration_limit']}；实际物理运行尝试 "
                        f"{audit['actual_real_runs']}/{correction['physical_run_attempt_limit']}。",
                        '关机中断的原日志、进程元数据、适配器失败占位与旧 trace 存于 interruption_archive/；'
                        '它没有进入三次有效重复的稳定性聚合。', ''])
    (output / 'module3b_summary.md').write_text('\n'.join(summary))
    return audit


def run(output=OUT):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.pilot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'decision_audit.json').exists():
            print('Pilot already finalized; no new measurements will run.', flush=True)
            return
        w, catalog, hardware, sources = prepare(output)
        evaluator = MeasurementRuntimeEvaluator(output, catalog, sources, SETTINGS, SEEDS, 6)
        last = None
        def checkpoint(result):
            nonlocal last
            save_json(output / 'search_trace.json', result.to_dict())
            info = dict(event='search_checkpoint', status=result.status,
                        runtime_evaluations=result.runtime_evaluation_count,
                        lp_solves=result.lp_solve_count, deployment=result.deployment,
                        iterations=result.search_iterations,
                        selected=result.iterations[-1]['selected_top_b'] if result.iterations else [])
            if info != last:
                print(json.dumps(info), flush=True)
                last = info
        with forbid_enumeration() as guards:
            result = GuidedDeploymentSearch(evaluator, predictor=None,
                runtime_objective=NoiseAwareRuntimeObjective(1., MIN_RELATIVE_IMPROVEMENT),
                trace_callback=checkpoint).search(w, catalog, hardware, CONFIG)
        # Exercise durable exact-cache reuse; this must cause zero new real runs.
        before = json.loads(evaluator.cache.path.read_text())
        evaluator.evaluate(result.deployment, w, result.canonical.routing)
        after = json.loads(evaluator.cache.path.read_text())
        if len(before['run_attempts']) != len(after['run_attempts']):
            raise RuntimeError('Cache replay unexpectedly launched a measurement')
        assets = json.loads((output / 'asset_hashes.json').read_text())
        verify_assets_unchanged(assets)
        audit = write_reports(output, result, sum(g.call_count for g in guards))
        print(json.dumps(dict(event='pilot_complete', passed=audit['passed'],
                              unique_runtime_evaluations=audit['unique_runtime_evaluations'],
                              real_runs=audit['actual_real_runs'], output=str(output))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
        print('Pilot workload, settings and provenance frozen; no GPU measurement launched.', flush=True)
    else:
        run()
