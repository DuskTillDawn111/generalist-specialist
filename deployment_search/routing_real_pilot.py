"""Module 4B: one frozen W1 deployment, bounded routing search, real GPU feedback.

python -m deployment_search.routing_real_pilot --prepare-only
python -m deployment_search.routing_real_pilot

Re-entry replays deterministic decisions through the exact durable cache. No
deployment search, predictor, grid or new serving worker is used.
"""
import argparse
from contextlib import ExitStack
import copy
import fcntl
import json
import math
from pathlib import Path
from unittest.mock import patch

from deployment_core.models import load_model_catalog
from deployment_core.workload import Workload
from joint_deploy.measurement.schema import MeasurementConfig

from .campaign_runtime import CampaignRuntimeEvaluator, execution_status
from .measurement_runtime import (MeasurementRuntimeEvaluation, aggregate_repetitions,
                                  measurement_subprocess, summarize_repetition)
from .multiworkload import gpu_meta
from .pilot_provenance import source_hashes, verify_assets_unchanged
from .real_objective import NoiseAwareRuntimeObjective
from .real_pilot import SETTINGS, SEEDS, freeze
from .result import jsonable
from .routing import solve_canonical_routing
from .routing_search import (GuidedRoutingSearch, RoutingSearchConfig, routing_pressure,
                             validate_routing_candidate)
from .scoring import RuntimeObjective
from .runtime_cache import digest, file_hash, save_json
from .tests.fixtures import forbid_enumeration

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/routing_search_real_pilot'
MODULE3 = ROOT / 'results/deployment_search_multiworkload'
CONFIG = RoutingSearchConfig(routing_step_fraction=.10, routing_top_b=2,
    max_iterations=3, no_improvement_patience=1, runtime_evaluation_budget=6,
    rho_max=.8, mean_tolerance_ms=1.)
OBJECTIVE = NoiseAwareRuntimeObjective(1., .05)
SEARCH_PHYSICAL_CAP = 18
CONFIRM_PHYSICAL_CAP = 8
CONFIRM_SEEDS = (84101, 84202)


def read(path):
    return json.loads(Path(path).read_text())


def routing_dict(value):
    return {tuple(key.split('::')): amount for key, amount in value.items()}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = load_model_catalog()
    workload_file = MODULE3 / 'benchmark_workloads.json'
    record = next(w for w in read(workload_file)['workloads'] if w['name'] == 'W1_balanced')
    workload = Workload(record['arrival_rates_rps'])
    run_path = MODULE3 / 'algorithm_runs/W1_balanced__B2/run_result.json'
    deployment = read(run_path)['final_deployment']
    assert sorted(m for m, n in deployment.items() if n) == ['paddleocr', 'qwen', 'resnet50', 'whisper']
    canonical = solve_canonical_routing(deployment, workload, catalog, CONFIG.rho_max)
    if canonical is None:
        raise RuntimeError('Frozen deployment lacks feasible canonical routing')
    assets_path = ROOT / 'results/deployment_search_real_pilot/asset_hashes.json'
    assets = read(assets_path)
    verify_assets_unchanged(assets)
    sources = source_hashes(assets)
    sources['campaign_runtime_sha256'] = file_hash(ROOT / 'deployment_search/campaign_runtime.py')
    freeze(OUT / 'source_model_hashes.json', sources)
    frozen_config = dict(search=jsonable(CONFIG), top_b=2, max_iterations=3,
        no_improvement_patience=1, routing_step_fraction=.10, predictor=None,
        max_valid_unique_runtime_evaluations=6, repetitions_per_state=3,
        max_valid_search_runs=18, search_physical_attempt_cap=SEARCH_PHYSICAL_CAP,
        confirmation_physical_attempt_cap=CONFIRM_PHYSICAL_CAP,
        confirmation_repetitions_per_changed_state=2, confirmation_seeds=CONFIRM_SEEDS,
        measurement_settings=SETTINGS, repetition_seeds=SEEDS,
        runtime_objective='Module 3C NoiseAwareRuntimeObjective(1 ms, 5%)',
        confirmation_influences_search=False, deployment_mutation_allowed=False,
        baseline='canonical routing only',
        serving_stability_priority=['steady', 'recoverable_congested', 'unstable'],
        run_status_policy='Data Quality Audit v1: only completed measurements aggregate; completed genuine unstable retained; interruptions count as physical attempts only',
        routing_source_annotation='Retained Module 3 measurement annotation for exact compatibility; actual explicit routing_rps and full identity determine cache eligibility.')
    freeze(OUT / 'routing_search_config.json', frozen_config)
    freedom = {k: sorted(m for m, p in catalog.items() if deployment[m] and k in p.supported_tasks)
               for k in workload.active_tasks}
    freeze(OUT / 'pilot_freeze.json', dict(module='4B', workload_name=record['name'],
        workload_source=dict(path=str(workload_file), sha256=file_hash(workload_file)),
        frozen_workload=record, fixed_deployment=deployment,
        deployment_source=dict(path=str(run_path), sha256=file_hash(run_path)),
        canonical_routing=canonical.routing, canonical_initialization='solve_canonical_routing only',
        model_pressure=routing_pressure(canonical.routing, deployment, workload, catalog),
        routing_freedom=freedom, tasks_with_routing_freedom=[k for k, ms in freedom.items() if len(ms) > 1],
        catalog={m: dict(service_time_ms=dict(p.service_time_ms), memory_mib=p.memory_mib,
                        sources=p.sources) for m, p in catalog.items()},
        hardware_identity=sources['gpu_identity'], source_model_hashes_sha256=digest(sources),
        asset_manifest_sha256=file_hash(assets_path),
        config_sha256=file_hash(OUT / 'routing_search_config.json'),
        implementation_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in
            [Path(__file__), ROOT / 'deployment_search/routing_search.py',
             ROOT / 'deployment_search/real_objective.py', ROOT / 'deployment_search/scoring.py',
             ROOT / 'deployment_search/runtime_cache.py']},
        frozen_before_first_gpu_run=True, deployment_changed=False,
        cache_reuse_policy='Only canonical exact full identity imported from Module 3C; no routing equivalence or historical latency selection'))
    evaluator = CampaignRuntimeEvaluator(OUT / 'search_cache', catalog, sources, SETTINGS,
        'Module4B_W1_fixed_deployment', SEEDS, max_unique=6, physical_attempt_cap=SEARCH_PHYSICAL_CAP)
    import_canonical_cache(evaluator, deployment, workload, canonical.routing)
    return deployment, workload, catalog, evaluator


def import_canonical_cache(evaluator, deployment, workload, routing):
    identity = evaluator.identity(deployment, workload, routing)
    key = digest(identity)
    old_path = MODULE3 / 'shared_cache/runtime_cache_manifest.json'
    prior = read(old_path)
    entry = prior['configurations'].get(key)
    eligible = bool(entry and entry['status'] == 'complete' and entry['identity'] == jsonable(identity))
    imported = False
    with evaluator.cache.locked() as cache:
        if eligible and key not in cache.data['configurations']:
            aggregate = read(entry['aggregate'])
            if file_hash(entry['aggregate']) != entry['aggregate_sha256']:
                raise RuntimeError('Module 3C canonical aggregate corrupted')
            if len(aggregate['repetitions']) != 3:
                raise RuntimeError('Canonical cache lacks 3 repetitions')
            for rep in aggregate['repetitions']:
                if rep['run_status'] != 'completed' or file_hash(rep['raw_result_path']) != rep['raw_sha256']:
                    raise RuntimeError('Canonical raw cache fails audit')
                raw = read(rep['raw_result_path'])
                cfg = MeasurementConfig(identity['deployment'], identity['workload'], identity['routing'],
                                        random_seed=rep['random_seed'], **SETTINGS).as_dict()
                if any(raw.get(k) != v for k, v in cfg.items() if k != 'arrival_process'):
                    raise RuntimeError('Canonical raw configuration differs from exact identity')
            imported_entry = copy.deepcopy(entry)
            imported_entry['attempts'] = []
            imported_entry['imported_from'] = dict(manifest=str(old_path), cache_key=key,
                aggregate_sha256=entry['aggregate_sha256'], historical_physical_attempts=len(entry.get('attempts', [])))
            cache.data['configurations'][key] = imported_entry
            imported = True
        cache.flush()
        reused = key in cache.data['configurations'] and 'imported_from' in cache.data['configurations'][key]
    freeze(OUT / 'canonical_cache_eligibility.json', dict(cache_key=key, exact_identity_match=eligible,
        canonical_exact_cache_reused=reused, source_manifest=str(old_path),
        full_identity=identity, same_deployment_different_routing_allowed=False))
    print(json.dumps(dict(event='prepared', canonical_cache_eligible=eligible,
                         canonical_cache_imported=imported, canonical_cache_key=key)), flush=True)


def enrich_trace(result, workload, catalog):
    trace = result.to_dict()
    trace['analytical_runtime_mismatch'] = False
    trace['runtime_feedback_correction'] = False
    for source, row in zip(result.iterations, trace['iterations']):
        for candidate, saved in zip(source['candidates'], row['candidates']):
            saved['candidate_pressure'] = routing_pressure(candidate.routing, result.deployment, workload, catalog)
        measured = [c for c in row['candidates'] if c['objective'] is not None]
        mismatches = []
        for a in measured:
            for b in measured:
                if a['delta_z'] <= b['delta_z']:
                    continue
                ao, bo = a['objective'], b['objective']
                # Descriptive J ordering uses the same stability priority, then
                # steady mean and close-mean P95, without the 5% acceptance gate.
                better, reason = RuntimeObjective(1.).compare(
                    MeasurementRuntimeEvaluation(**bo), MeasurementRuntimeEvaluation(**ao), 0, 0)
                if better:
                    correction = a['status'] == 'ROLLBACK' and b['status'] == 'ACCEPTED'
                    mismatches.append(dict(analytical_preferred=[a['task'], a['source'], a['destination']],
                        runtime_preferred=[b['task'], b['source'], b['destination']],
                        delta_z_A=a['delta_z'], delta_z_B=b['delta_z'],
                        runtime_A=ao, runtime_B=bo, runtime_order_reason=reason,
                        runtime_feedback_correction=correction))
        row['analytical_runtime_mismatch'] = bool(mismatches)
        row['mismatches'] = mismatches
        trace['analytical_runtime_mismatch'] |= bool(mismatches)
        trace['runtime_feedback_correction'] |= any(m['runtime_feedback_correction'] for m in mismatches)
    return trace


class RecordingEvaluator:
    def __init__(self, evaluator, deployment, workload):
        self.evaluator, self.deployment, self.workload = evaluator, deployment, workload
        self.records = []

    def evaluate(self, deployment, workload, routing):
        if deployment != self.deployment or workload.arrival_rate != self.workload.arrival_rate:
            raise RuntimeError('Pilot attempted deployment/workload mutation')
        value = self.evaluator.evaluate(deployment, workload, routing)
        self.records.append(dict(evaluation_number=len(self.records) + 1,
            state_kind='canonical' if not self.records else 'Top-B routing candidate',
            deployment=deployment, workload=dict(workload.arrival_rate), routing=routing,
            full_identity=self.evaluator.identity(deployment, workload, routing), runtime=value))
        save_json(OUT / 'runtime_evaluations.json', self.records)
        export_manifest(self.evaluator)
        return value


def export_manifest(evaluator):
    save_json(OUT / 'runtime_cache_manifest.json', read(evaluator.cache.path))


def decision_audit(trace, records, manifest, workload, catalog, enumeration_calls):
    errors, decisions = [], []
    x = trace['deployment']
    canonical = solve_canonical_routing(x, workload, catalog, CONFIG.rho_max)
    if routing_dict(trace['initial_routing']) != canonical.routing:
        errors.append('noncanonical start')
    if len(records) != trace['runtime_evaluation_count'] or len(records) > 6:
        errors.append('evaluation count/budget')
    if len({r['runtime']['cache_key'] for r in records}) != len(records):
        errors.append('repeated search state')
    if len(manifest['run_attempts']) > SEARCH_PHYSICAL_CAP:
        errors.append('physical search cap')
    if enumeration_calls:
        errors.append('enumeration invoked')
    incumbent = MeasurementRuntimeEvaluation(**trace['initial_objective'])
    expected_routing = trace['initial_routing']
    for row in trace['iterations']:
        if row['routing'] != expected_routing or row['current_objective'] != jsonable(incumbent):
            errors.append('incumbent trace mismatch')
        ranked = sorted((c for c in row['candidates'] if c['rank'] is not None), key=lambda c: c['rank'])
        selected = [c for c in ranked if c['selected_top_b']]
        if selected != ranked[:2] or [c['delta_z'] for c in ranked] != sorted((c['delta_z'] for c in ranked), reverse=True):
            errors.append('Top-B or delta_z ranking')
        best = None
        for c in row['candidates']:
            r = routing_dict(c['routing'])
            if not validate_routing_candidate(r, x, workload, catalog, .8)[0]:
                if c['status'] != 'REJECT_INFEASIBLE':
                    errors.append('evaluated infeasible candidate')
            before = routing_dict(row['routing'])
            changed = {p for p in before if not math.isclose(before[p], r[p], abs_tol=1e-12)}
            if changed != {(c['task'], c['source']), (c['task'], c['destination'])}:
                errors.append('non one-transfer candidate')
            if not math.isclose(c['delta_rps'], .10 * workload.arrival_rate[c['task']], abs_tol=1e-12):
                errors.append('step differs from 10% workload rate')
            pressure = routing_pressure(r, x, workload, catalog)
            if not math.isclose(c['delta_z'], row['z'] - max(pressure.values()), abs_tol=1e-12):
                errors.append('pressure/delta_z mismatch')
            if c['prediction'] is not None:
                errors.append('predictor used')
        for c in selected:
            if c['objective'] is None:
                if c['status'] != 'NOT_EVALUATED_BUDGET':
                    errors.append('Top-B lacks runtime without budget reason')
                continue
            runtime = MeasurementRuntimeEvaluation(**c['objective'])
            better, reason = OBJECTIVE.compare(runtime, incumbent, 0, 0)
            decisions.append(dict(iteration=row['iteration'], mutation=[c['task'], c['source'], c['destination']],
                LP_rank=c['rank'], delta_z=c['delta_z'], candidate=runtime,
                incumbent=incumbent, improves_incumbent=better, objective_reason=reason,
                decision=c['status'], reason=c['reason']))
            if better and (best is None or OBJECTIVE.compare(runtime, MeasurementRuntimeEvaluation(**best['objective']), 0, 0)[0]):
                best = c
        actual = next((c for c in selected if c['status'] == 'ACCEPTED'), None)
        if actual != best:
            errors.append('accept/rollback differs from frozen RuntimeObjective')
        if actual:
            incumbent = MeasurementRuntimeEvaluation(**actual['objective'])
            expected_routing = actual['routing']
    if trace['final_routing'] != expected_routing:
        errors.append('final routing does not follow accepted transfers')
    for record in records:
        if record['deployment'] != x:
            errors.append('deployment changed')
        runtime = record['runtime']
        if len(runtime['repetitions']) != 3 or any(r['run_status'] != 'completed' for r in runtime['repetitions']):
            errors.append('invalid repetition aggregation')
        for rep in runtime['repetitions']:
            if file_hash(rep['raw_result_path']) != rep['raw_sha256']:
                errors.append('raw hash mismatch')
        if digest(record['full_identity']) != runtime['cache_key']:
            errors.append('cache identity mismatch')
    return jsonable(dict(passed=not errors, errors=errors, decisions=decisions,
        deployment_changed=False, routing_grid_enumeration_calls=enumeration_calls,
        predictor=None, confirmation_influences_search=False))


def confirm_one(which, repetition, routing, deployment, workload, catalog, evaluator):
    directory = OUT / 'confirmation' / which
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'rep_{repetition}.json'
    if target.exists():
        rep = read(target)
        if rep['run_status'] != 'completed' or file_hash(rep['raw_result_path']) != rep['raw_sha256']:
            raise RuntimeError('Saved confirmation invalid')
        return rep
    identity = evaluator.identity(deployment, workload, routing)
    seed = CONFIRM_SEEDS[repetition - 1]
    cfg = MeasurementConfig(identity['deployment'], identity['workload'], identity['routing'],
                            random_seed=seed, **SETTINGS)
    cfg.validate({m: p.supported_tasks for m, p in catalog.items()})
    ledger_path = OUT / 'confirmation_attempts.json'
    attempts = read(ledger_path) if ledger_path.exists() else []
    for _ in range(3):
        if len(attempts) >= CONFIRM_PHYSICAL_CAP:
            raise RuntimeError('Frozen confirmation physical cap reached')
        attempt_id = len(attempts) + 1
        stem = directory / f'rep_{repetition}_attempt_{attempt_id:03d}'
        raw_path, log_path, config_path = (Path(str(stem) + s) for s in ('.raw.json', '.log', '.config.json'))
        record = dict(attempt_id=attempt_id, which=which, repetition=repetition, seed=seed,
            status='started', full_identity=identity, raw_result_path=str(raw_path),
            config_path=str(config_path), log_path=str(log_path), environment_start=gpu_meta())
        save_json(config_path, cfg.as_dict())
        attempts.append(record)
        save_json(ledger_path, attempts)
        print(json.dumps(dict(event='confirmation_start', which=which, repetition=repetition)), flush=True)
        outcome = measurement_subprocess(cfg.as_dict(), raw_path, log_path, config_path)
        raw = read(raw_path) if raw_path.exists() else None
        if raw is not None and any(raw.get(k) != v for k, v in cfg.as_dict().items() if k != 'arrival_process'):
            outcome['configuration_mismatch'] = True
        status, reason = execution_status(raw, outcome, log_path)
        record.update(status=status, run_status=status, reason=reason, process_outcome=outcome,
                      environment_end=gpu_meta(), valid_runtime_repetition=status == 'completed')
        save_json(ledger_path, attempts)
        if status != 'completed':
            continue
        rep = summarize_repetition(raw, workload, catalog, dict(returncode=0), raw_path)
        rep.update(run_status='completed', post_search_confirmation=True,
                   repetition=repetition, random_seed=seed, valid_runtime_repetition=True,
                   full_identity=identity, identity_sha256=digest(identity), attempt_id=attempt_id)
        save_json(target, rep)
        return rep
    raise RuntimeError('Three invalid confirmation attempts; preserve evidence and stop')


def confirmation(trace, deployment, workload, catalog, evaluator):
    if trace['final_routing'] == trace['initial_routing']:
        value = dict(required=False, reason='final == canonical; no extra runs',
                     repetitions={}, confirmation_direction_maintained=None)
    else:
        reps = {'canonical': [], 'final': []}
        for which, index in [('canonical', 1), ('final', 1), ('final', 2), ('canonical', 2)]:
            routing = routing_dict(trace['initial_routing' if which == 'canonical' else 'final_routing'])
            reps[which].append(confirm_one(which, index, routing, deployment, workload, catalog, evaluator))
            save_json(OUT / 'confirmation_results.json', dict(required=True, complete=False, repetitions=reps))
        aggs = {which: aggregate_repetitions(rs, 'independent_confirmation', OUT / f'{which}_confirmation_aggregate.json')
                for which, rs in reps.items()}
        better, reason = RuntimeObjective(1.).compare(aggs['final'], aggs['canonical'], 0, 0)
        value = dict(required=True, complete=True, repetitions=reps, aggregates=aggs,
            confirmation_direction_maintained=better,
            mean_direction_maintained=(aggs['final'].weighted_mean_e2e < aggs['canonical'].weighted_mean_e2e
                if aggs['final'].weighted_mean_e2e is not None and aggs['canonical'].weighted_mean_e2e is not None else None),
            direction_reason=reason, influences_search_decision=False)
    save_json(OUT / 'confirmation_results.json', value)
    return jsonable(value)


def finish(trace, audit, records, conf, workload, catalog, evaluator):
    manifest = read(evaluator.cache.path)
    combined = {}
    for which, key in [('canonical', 'initial_objective'), ('final', 'final_objective')]:
        reps = trace[key]['repetitions'] + conf['repetitions'].get(which, [])
        combined[which] = aggregate_repetitions(reps, trace[key]['cache_key'], OUT / f'{which}_comparison.json')
        save_json(OUT / f'{which}_comparison.json', combined[which])
    before, after = combined['canonical'], combined['final']
    gain = ((before.weighted_mean_e2e - after.weighted_mean_e2e) / before.weighted_mean_e2e
            if before.weighted_mean_e2e and after.weighted_mean_e2e is not None else None)
    decision = ('FAIL' if not audit['passed'] else 'NEUTRAL' if trace['final_routing'] == trace['initial_routing']
                else 'PASS' if conf['confirmation_direction_maintained'] else 'NEUTRAL')
    reason = ('Correctness audit failed.' if decision == 'FAIL' else
        'Changed routing; independent confirmation maintains improvement direction.' if decision == 'PASS' else
        'This pilot did not establish a confirmed routing gain; canonical routing may already be sufficient for this workload/deployment.')
    attempts = read(OUT / 'confirmation_attempts.json') if (OUT / 'confirmation_attempts.json').exists() else []
    stats = dict(pilot_decision=decision, reason=reason,
        canonical_weighted_mean_e2e_ms=before.weighted_mean_e2e, final_weighted_mean_e2e_ms=after.weighted_mean_e2e,
        canonical_weighted_task_p95_ms=before.weighted_p95_e2e, final_weighted_task_p95_ms=after.weighted_p95_e2e,
        improvement=gain, accepted_mutations=len(trace['accepted_mutations']), rollbacks=len(trace['rejected_mutations']),
        runtime_evaluations=len(records), new_unique_runtime_evaluations=sum('imported_from' not in e for e in manifest['configurations'].values()),
        valid_search_repetitions=3 * len(records), new_valid_search_runs=sum(a['status'] == 'completed' for a in manifest['run_attempts']),
        search_physical_attempts=len(manifest['run_attempts']), confirmation_physical_attempts=len(attempts),
        physical_attempts=len(manifest['run_attempts']) + len(attempts),
        analytical_runtime_mismatch=trace['analytical_runtime_mismatch'], runtime_feedback_correction=trace['runtime_feedback_correction'],
        deployment_changed=False, routing_grid_enumeration_calls=audit['routing_grid_enumeration_calls'],
        compatible_repetitions_per_compared_state=len(before.repetitions), termination_reason=trace['termination_reason'])
    save_json(OUT / 'pilot_metrics.json', stats)
    def fmt(v):
        return 'unavailable' if v is None else f'{v:.6f}'
    lines = ['# Module 4B：Real-runtime Routing Search Pilot', '', f'**判定：{decision}**。{reason}', '',
        '仅 W1_balanced，固定 Module 3C B2 deployment：paddleocr + qwen + resnet50 + whisper。predictor=None。',
        '到此停止；未进入 Module 4C 或任何 deployment-routing、replica、dynamic、forecasting、paper benchmark。', '',
        '## Runtime metrics', '', '| Metric | Canonical | Final |', '|---|---:|---:|',
        f'| Weighted mean E2E (ms) | {fmt(before.weighted_mean_e2e)} | {fmt(after.weighted_mean_e2e)} |',
        f'| Weighted task P95 (ms) | {fmt(before.weighted_p95_e2e)} | {fmt(after.weighted_p95_e2e)} |',
        f'| Stability | {before.stability_state_v2} | {after.stability_state_v2} |',
        f'| Compatible repetitions | {len(before.repetitions)} | {len(after.repetitions)} |', '',
        f'Improvement = (J(r0) - J(rfinal)) / J(r0) = {fmt(gain)}（相对 weighted mean E2E）；绝对差 {fmt(before.weighted_mean_e2e - after.weighted_mean_e2e) if gain is not None else "unavailable"} ms。',
        'Weighted task P95 是 workload arrival-rate 加权各 task P95，再取 repetition 均值；不是 pooled global P95。', '',
        '## 必答 14 项', '',
        '1. Canonical routing：text、visual_qa 全部 qwen；image 全部 resnet50；OCR 全部 paddleocr；ASR、LID 全部 whisper。完整 rps 及 fraction 见下表。',
        '2. Routing freedom：image（qwen/resnet50）、ocr（paddleocr/qwen）、asr 和 lid（qwen/whisper）。text、visual_qa 无 routing freedom。',
        f'3. 初始 model pressure：`{json.dumps(trace["initial_pressure"], sort_keys=True)}`；z={max(trace["initial_pressure"].values()):.9f}。',
        '4. 每轮 Top-B（按照 delta_z 从大到小）：']
    for row in trace['iterations']:
        selected = sorted((c for c in row['candidates'] if c['selected_top_b']), key=lambda c: c['rank'])
        lines.append('   - 第 ' + str(row['iteration'] + 1) + ' 轮：' + '; '.join(
            f'{c["task"]}: {c["source"]} → {c["destination"]}, delta={c["delta_rps"]:.6f} rps, delta_z={c["delta_z"]:.9f}, {c["status"]} ({c["reason"]})' for c in selected))
    lines += [f'5. 被真实测量的 states：{len(records)} 个（含 canonical；exact cache reuse 是已完成真实测量，非新增物理运行），逐项 routing/identity/raw paths 见 runtime_evaluations.json。',
        f'6. analytical/runtime mismatch：{str(trace["analytical_runtime_mismatch"]).lower()}；runtime feedback correction：{str(trace["runtime_feedback_correction"]).lower()}。只基于同轮已测 Top-B 记录。',
        f'7. 真实 runtime rollback：{len(trace["rejected_mutations"])}；原因逐项见 decision_audit.json 和下方 trace。',
        '8. 最终 routing：见 fraction 表和 routing_search_trace.json 的 final_routing。',
        f'9. Weighted mean E2E：{fmt(before.weighted_mean_e2e)} → {fmt(after.weighted_mean_e2e)} ms；改善方向={gain > 0 if gain is not None else "unavailable"}。',
        f'10. Weighted task P95：{fmt(before.weighted_p95_e2e)} → {fmt(after.weighted_p95_e2e)} ms；改善方向={after.weighted_p95_e2e < before.weighted_p95_e2e if before.weighted_p95_e2e is not None and after.weighted_p95_e2e is not None else "unavailable"}。']
    changes = {}
    r0, rf = routing_dict(trace['initial_routing']), routing_dict(trace['final_routing'])
    for k in workload.active_tasks:
        changes[k] = max(abs(r0.get((k, m), 0.) - rf.get((k, m), 0.)) / workload.arrival_rate[k] for m in catalog)
    largest = max(changes.values())
    lines += [f'11. 最大 routing fraction 改变：{", ".join(k for k, v in changes.items() if v == largest)}，{largest * 100:.2f} percentage points。' if largest else '11. Routing fraction 均未改变；最大变化 0 percentage points。',
        '12. 是否改变 deployment：**no**。', f'13. Routing grid enumeration calls：**{audit["routing_grid_enumeration_calls"]}**。',
        f'14. Module 4B 判定：**{decision}**。', '', '## Per-task routing and latency', '',
        '| Task | Before fraction | After fraction | Mean before/after ms | P95 before/after ms |', '|---|---|---|---:|---:|']
    for k in workload.active_tasks:
        def fraction(r):
            return ', '.join(f'{m}={r.get((k, m), 0.) / workload.arrival_rate[k]:.2%} ({r.get((k, m), 0.):.6f} rps)' for m in catalog if deployment_active(trace, m) and k in catalog[m].supported_tasks)
        lines.append(f'| {k} | {fraction(r0)} | {fraction(rf)} | {fmt(before.per_task_mean[k]["mean"])} / {fmt(after.per_task_mean[k]["mean"])} | {fmt(before.per_task_p95[k]["mean"])} / {fmt(after.per_task_p95[k]["mean"])} |')
    lines += ['', '## Model pressure', '', '| Model | Before | After |', '|---|---:|---:|']
    for m in trace['initial_pressure']:
        lines.append(f'| {m} | {trace["initial_pressure"][m]:.9f} | {trace["final_pressure"][m]:.9f} |')
    lines += ['', '## Budget, cache and validity', '',
        f'Search runtime evaluations={len(records)}/6（canonical 包含在内），每 state 3 valid repetitions；新 physical search attempts={stats["search_physical_attempts"]}/18；新 valid search runs={stats["new_valid_search_runs"]}。',
        f'Confirmation physical attempts={len(attempts)}/{CONFIRM_PHYSICAL_CAP}；新增总 physical attempts={stats["physical_attempts"]}。历史 cache 运行不重复计为本 pilot physical attempts。',
        f'Accepted mutations={stats["accepted_mutations"]}，rollbacks={stats["rollbacks"]}；termination={trace["termination_reason"]}；decision audit passed={audit["passed"]}。',
        'Cache full identity 包括 deployment、冻结 workload、完整 explicit routing、measurement settings/seeds、model/source/input/interpreter hashes、hardware UUID/driver 和审计策略。相同 deployment 不同 routing 是不同 key。',
        'Execution status 与 serving stability 分开：只有 completed repetition 才计 latency，completed genuine unstable 保留；external/uncertain/harness/model interruption 不作为 serving label，仅计 physical attempts。',
        'Worst repetition stability 决定 aggregate class；同 steady 至少 5% weighted mean improvement 才能因 latency 接受，stability improvement 优先。',
        'Confirmation 与 search decision 隔离；若 final==canonical，无额外 confirmation。PASS 表示 pilot 真实收益证据，不表示统计显著性或多 workload 结论。']
    if conf['required']:
        lines += ['', '## Independent confirmation', '',
            f'独立 confirmation maintains direction={conf["confirmation_direction_maintained"]}，mean direction={conf["mean_direction_maintained"]}。',
            f'Canonical/final confirmation mean ms：{fmt(conf["aggregates"]["canonical"]["weighted_mean_e2e"])} / {fmt(conf["aggregates"]["final"]["weighted_mean_e2e"])}。']
    (OUT / 'module4b_summary.md').write_text('\n'.join(lines) + '\n')
    trace_lines = ['# Module 4B Routing Search Trace', '', f'Termination: {trace["termination_reason"]}', '']
    for row in trace['iterations']:
        trace_lines += [f'## Iteration {row["iteration"] + 1}', '',
            f'Current routing: `{json.dumps(row["routing"], sort_keys=True)}`',
            f'Pressures: `{json.dumps(row["pressure"], sort_keys=True)}`; z={row["z"]:.9f}', '',
            '| LP rank | Task | Source → destination | Delta rps | Candidate z | Delta z | Top-B | Runtime mean / P95 ms | Decision / reason |',
            '|---:|---|---|---:|---:|---:|---|---|---|']
        for c in sorted(row['candidates'], key=lambda c: c['rank'] if c['rank'] is not None else 100000):
            rt = c['objective']
            trace_lines.append(f'| {c["rank"]} | {c["task"]} | {c["source"]} → {c["destination"]} | {c["delta_rps"]:.6f} | {c["max_pressure_after"]:.9f} | {c["delta_z"]:.9f} | {c["selected_top_b"]} | {fmt(rt["weighted_mean_e2e"]) + " / " + fmt(rt["weighted_p95_e2e"]) if rt else "not evaluated"} | {c["status"]}: {c["reason"]} |')
        trace_lines += ['', f'Accepted: {row["accepted"]}; analytical_runtime_mismatch={row["analytical_runtime_mismatch"]}', '']
        for c in row['candidates']:
            trace_lines += [f'- {c["task"]} {c["source"]} → {c["destination"]} candidate pressure: `{json.dumps(c["candidate_pressure"], sort_keys=True)}`']
    (OUT / 'routing_search_trace.md').write_text('\n'.join(trace_lines) + '\n')
    export_manifest(evaluator)
    print(json.dumps(dict(event='pilot_complete', **stats)), flush=True)


def deployment_active(trace, model):
    return bool(trace['deployment'].get(model))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.pilot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deployment, workload, catalog, evaluator = prepare()
        if args.prepare_only:
            export_manifest(evaluator)
            return
        recorder = RecordingEvaluator(evaluator, deployment, workload)
        def checkpoint(result):
            save_json(OUT / 'routing_search_trace.json', enrich_trace(result, workload, catalog))
        with ExitStack() as stack:
            guards = stack.enter_context(forbid_enumeration())
            grid = stack.enter_context(patch('deployment_search.routing_foundation.small_grid_oracle',
                side_effect=AssertionError('Forbidden routing grid enumeration')))
            result = GuidedRoutingSearch(recorder, predictor=None, runtime_objective=OBJECTIVE,
                trace_callback=checkpoint).search(deployment, workload, catalog, CONFIG)
            enum_calls = sum(g.call_count for g in guards) + grid.call_count
        trace = enrich_trace(result, workload, catalog)
        records = jsonable(recorder.records)
        audit = decision_audit(trace, records, read(evaluator.cache.path), workload, catalog, enum_calls)
        save_json(OUT / 'decision_audit.json', audit)
        save_json(OUT / 'search_decision_frozen.json', dict(final_routing=trace['final_routing'],
            final_objective=trace['final_objective'], trace_sha256=digest(trace), audit_sha256=digest(audit),
            frozen_before_confirmation=True))
        if not audit['passed']:
            raise RuntimeError('Decision audit failed: ' + str(audit['errors']))
        conf = confirmation(trace, deployment, workload, catalog, evaluator)
        finish(trace, audit, records, conf, workload, catalog, evaluator)


if __name__ == '__main__':
    main()
