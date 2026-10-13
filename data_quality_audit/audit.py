"""Read-only provenance audit for historical GPU run artifacts.

Outputs are correction-layer records. This module never launches a worker or
rewrites source results. Run with the gsdeploy Python environment.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from deployment_core.models import load_model_catalog
from performance_model.local_calibration.stability_v2 import DEFAULT_THRESHOLDS, classify, reconstruct

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
OUT = RESULTS / 'data_quality_audit'
EXCLUDED_OUTPUTS = {'data_quality_audit', 'performance_model_generalization_corrected',
                    'performance_model_local_calibration_corrected', 'performance_model_ranking_corrected',
                    'performance_model_comparable_groups_corrected'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def jsonl(path, values):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False, allow_nan=False) + '\n' for x in values))


def original_files():
    return sorted(p for p in RESULTS.rglob('*') if p.is_file() and p.relative_to(RESULTS).parts[0] not in EXCLUDED_OUTPUTS)


def run_paths():
    paths = []
    paths += [RESULTS / 'stage2a/smoke_text.json', RESULTS / 'stage2a/smoke_multitask.json']
    paths += sorted((RESULTS / 'stage2b/routing_characterization/raw').glob('*.json'))
    paths += [RESULTS / 'stage2b/routing_characterization/full_deployment_smoke.json']
    paths += sorted((RESULTS / 'performance_model_profiling/qwen').glob('*/raw/*.json'))
    paths += sorted((RESULTS / 'performance_model_active_sampling/active_sampling_runs/raw').glob('*.json'))
    paths += sorted((RESULTS / 'performance_model_comparable_groups/raw_runs').glob('*.json'))
    manifest = json.loads((RESULTS / 'deployment_search_real_pilot/runtime_cache_manifest.json').read_text())
    for entry in manifest['configurations'].values():
        paths += [Path(r['raw_result_path']) for r in entry['repetitions']]
        paths += [Path(r['archived_artifacts']['failure_envelope']['path'])
                  for r in entry.get('interrupted_attempts', [])]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate physical run path in inventory')
    return sorted(paths), manifest


def phase(path):
    p = path.relative_to(RESULTS).parts
    return {'stage2a':'Stage 2A', 'stage2b':'Stage 2B',
            'performance_model_profiling':'Module 2B/2C profiling',
            'performance_model_active_sampling':'Module 2D/2E/2F source',
            'performance_model_comparable_groups':'Module 2G',
            'deployment_search_real_pilot':'Module 3B'}[p[0]]


def old_labels():
    labels = {}
    for folder in ('performance_model_local_calibration', 'performance_model_comparable_groups'):
        p = RESULTS / folder / 'stability_v2_labels.jsonl'
        for line in p.read_text().splitlines():
            x = json.loads(line)
            labels[x['source_result_path']] = x['stability_state_v2']
    m = json.loads((RESULTS/'deployment_search_real_pilot/runtime_cache_manifest.json').read_text())
    for entry in m['configurations'].values():
        for rep in entry['repetitions']:
            if rep.get('runtime'):
                labels[rep['raw_result_path']] = rep['runtime']['stability_state_v2']
        for rep in entry.get('interrupted_attempts', []):
            if rep.get('runtime'):
                labels[rep['archived_artifacts']['failure_envelope']['path']] = rep['runtime']['stability_state_v2']
    return labels


def classify_run(path, raw, catalog, legacy, known_external):
    window = raw.get('measurement_window') or {}
    system = raw.get('system_metrics') or {}
    duration = raw.get('measurement_seconds')
    full_window = bool(window and system and duration and window.get('duration_seconds') == duration
                       and window.get('arrival_loop_end_time_ns', 0) >= window.get('end_time_ns', 1)
                       and window.get('drain_end_time_ns', 0) >= window.get('arrival_loop_end_time_ns', 1))
    log = path.with_suffix('.log')
    log_present = log.exists()
    log_text = log.read_text(errors='replace') if log_present else ''
    footer = ('"output"' in log_text[-1500:] and str(path) in log_text[-1500:]) if log_present else None
    evidence = dict(raw_status=raw.get('status'), errors=raw.get('errors', []),
                    measurement_window_complete=full_window,
                    planned_duration_seconds=duration, recorded_duration_seconds=window.get('duration_seconds'),
                    arrival_loop_finished=bool(window.get('arrival_loop_end_time_ns')),
                    drain_finished=bool(window.get('drain_end_time_ns')),
                    offered_requests=system.get('offered_requests'),
                    completed_requests=system.get('completed_requests'),
                    incomplete_requests=system.get('incomplete_requests'),
                    completion_ratio=system.get('completion_ratio'),
                    log_path=str(log) if log_present else None,
                    log_sha256=sha(log) if log_present else None,
                    harness_footer_present=footer,
                    shutdown=raw.get('shutdown'))
    if str(path) in known_external:
        status = 'external_interruption'
        reason = 'Module 3B host-power-off correction: adapter failure envelope, no harness raw result; interruption archive and correction ledger.'
        evidence['external_correction'] = known_external[str(path)]
    elif full_window and (not log_present or footer or phase(path) == 'Module 2B/2C profiling'):
        # A full timed window, completed drain, request/system accounting and
        # serialized result establish a completed observation. A forced worker
        # shutdown after a 60 s drain is a consequence of backlog, not power-off.
        status = 'completed'
        reason = 'Full timed measurement, arrival loop and drain completed; system/request result serialized.'
        if raw.get('status') != 'completed':
            reason += ' Harness failed status reflects incomplete cohort or post-drain shutdown; serving stability is assessed separately.'
    elif raw.get('adapter_failure') or not window:
        status = 'uncertain_interruption'
        reason = 'No full harness result and no independent external-cause evidence.'
    elif any('worker' in str(e).lower() or 'model' in str(e).lower() for e in raw.get('errors', [])):
        status = 'model_failure'
        reason = 'Worker/model error prevents a trustworthy completed observation.'
    else:
        status = 'harness_failure'
        reason = 'Incomplete measurement protocol or missing completion evidence.'
    label = None
    if status == 'completed':
        ev = reconstruct(raw, catalog, DEFAULT_THRESHOLDS)
        label = classify(ev, DEFAULT_THRESHOLDS)['stability_state_v2']
    task_metrics = raw.get('per_task_metrics') or {}
    active = [k for k, v in (raw.get('workload_rps') or {}).items() if v > 0]
    latency_ok = status == 'completed' and bool(active) and all(
        task_metrics.get(k, {}).get('mean_e2e_latency_ms') is not None and
        task_metrics.get(k, {}).get('p95_e2e_latency_ms') is not None for k in active)
    valid = dict(valid_for_latency=latency_ok, valid_for_throughput=status == 'completed' and bool(system),
                 valid_for_stability=status == 'completed',
                 valid_for_ranking=latency_ok and label in ('steady','recoverable_congested'),
                 valid_for_attempt_accounting=True)
    return status, label, reason, evidence, valid


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths, manifest = run_paths()
    previous = OUT/'original_files_sha256.json'
    if previous.exists():
        inventory = json.loads(previous.read_text())
        actual = {str(p):sha(p) for p in original_files()}
        if inventory != actual:
            raise ValueError('Original file content changed since initial audit snapshot')
    else:
        inventory = {str(p):sha(p) for p in original_files()}
        save(previous, inventory)
    correction = json.loads((RESULTS/'deployment_search_real_pilot/recovery_correction.json').read_text())
    external = {}
    for entry in manifest['configurations'].values():
        for rep in entry.get('interrupted_attempts', []):
            if rep.get('external_interruption'):
                external[rep['archived_artifacts']['failure_envelope']['path']] = dict(correction_path=str(RESULTS/'deployment_search_real_pilot/recovery_correction.json'),
                    correction_sha256=sha(RESULTS/'deployment_search_real_pilot/recovery_correction.json'),
                    archived_artifacts=rep.get('archived_artifacts'), cause=correction['reason'])
    catalog, legacy = load_model_catalog(), old_labels()
    rows, labels, affected = [], [], []
    for p in paths:
        raw = json.loads(p.read_text())
        status, label, reason, evidence, valid = classify_run(p, raw, catalog, legacy, external)
        original = legacy.get(str(p))
        record = dict(run_id=p.stem, phase=phase(p), source_result_path=str(p), source_sha256=sha(p),
                      run_status=status, run_status_reason=reason,
                      raw_harness_status=raw.get('status'), original_stability_state_v2=original,
                      stability_state_v2_corrected=label, **valid, evidence=evidence)
        rows.append(record)
        labels.append({k:record[k] for k in ('run_id','phase','source_result_path','source_sha256',
            'run_status','original_stability_state_v2','stability_state_v2_corrected',
            'valid_for_latency','valid_for_throughput','valid_for_stability','valid_for_ranking','valid_for_attempt_accounting')})
        if original != label and original is not None:
            affected.append(dict(source_result_path=str(p), original=original, corrected=label,
                                 run_status=status, reason=reason))
    jsonl(OUT/'run_provenance_audit.jsonl', rows)
    jsonl(OUT/'corrected_stability_labels.jsonl', labels)
    save(OUT/'external_interruptions.json', [r for r in rows if r['run_status']=='external_interruption'])
    save(OUT/'uncertain_interruptions.json', [r for r in rows if r['run_status']=='uncertain_interruption'])
    save(OUT/'affected_runs.json', affected)
    by_phase = {p:dict(status=Counter(r['run_status'] for r in rows if r['phase']==p),
                        completed_stability=Counter(r['stability_state_v2_corrected'] for r in rows
                            if r['phase']==p and r['run_status']=='completed')) for p in sorted({r['phase'] for r in rows})}
    summary = dict(total_runs=len(rows), run_status_counts={s:sum(r['run_status']==s for r in rows)
            for s in ('completed','external_interruption','harness_failure','model_failure','uncertain_interruption')},
                   completed_stability_counts={s:sum(r['run_status']=='completed' and r['stability_state_v2_corrected']==s for r in rows)
                       for s in ('steady','recoverable_congested','unstable')},
                   legacy_unstable_reclassified_external=sum(r['original_stability_state_v2']=='unstable' and r['run_status']=='external_interruption' for r in rows),
                   legacy_unstable_reclassified_uncertain=sum(r['original_stability_state_v2']=='unstable' and r['run_status']=='uncertain_interruption' for r in rows),
                   by_phase=by_phase, original_file_count=len(inventory), original_files_unchanged=True)
    save(OUT/'audit_counts.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
