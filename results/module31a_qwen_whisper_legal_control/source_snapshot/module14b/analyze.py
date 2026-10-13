"""Final OCR three-way eligibility, preserve three14A choices, final pool/SLO."""
import statistics, json
from module14b.common import *
from module13b.analyze import stats


def aggregate(e):
    stem = 'ocr__' + e['model_id'].replace('/', '--')
    qp = RAW / (stem + '__quality__r0.summary.json')
    qd = read(qp) if qp.exists() else {}
    qr = [json.loads(l) for l in Path(qd['raw_path']).read_text().splitlines()] if qd.get('success') else []
    q = sum(bool(r['correct']) for r in qr) / 200 if len(qr) == 200 else None
    reps, latencies, idle, peak, host = [], [], [], [], []
    errors = []
    for rep in [1, 2, 3]:
        p = RAW / (stem + f'__latency__r{rep}.summary.json')
        d = read(p) if p.exists() else {}
        if not d.get('success'):
            errors.append({'phase': 'latency', 'repetition': rep, 'reason': d.get('error', 'NOT_MEASURED')})
            continue
        rows = [json.loads(l) for l in Path(d['raw_path']).read_text().splitlines()]
        assert len(rows) == 30
        values = [r['worker_latency_ms'] for r in rows]
        latencies.extend(values); idle.append(d['idle_model_vram_mib'])
        peak.append(d['peak_inference_vram_mib']); host.append(d['host_rss_peak_mib'])
        reps.append({'repetition': rep, 'latency_statistics': stats(values),
                     'idle_memory_mib': idle[-1], 'peak_memory_mib': peak[-1],
                     'raw': d['raw_path']})
    if qd.get('peak_inference_vram_mib') is not None: peak.append(qd['peak_inference_vram_mib'])
    if qd.get('host_rss_peak_mib') is not None: host.append(qd['host_rss_peak_mib'])
    complete = len(latencies) == 90 and len(idle) == 3
    mean_idle = statistics.mean(idle) if complete else None
    pre = next((p for p in read(OUT / 'module14b_cpu_preflight.json')['entries'] if p['model_id'] == e['model_id']), {})
    return {'task': 'ocr', 'model_id': e['model_id'], 'revision': e.get('revision'),
            'implementation_commit': e.get('implementation_commit'), 'checkpoint': e.get('weight_path'),
            'origin': 'MODULE14B_REAL_LOCAL_OCR_MEASUREMENT', 'quality': q,
            'quality_metric': 'exact_match', 'quality_n': len(qr),
            's0_ms': statistics.mean(latencies) if complete else None,
            'idle_memory_mib': mean_idle,
            'safe_memory_mib': mean_idle * read(OUT / 'baseline_references.json')['safe_factor'] if complete else None,
            'peak_inference_memory_mib': max(peak) if peak else None,
            'host_RSS_peak_mib': max(host) if host else None,
            'latency_statistics': stats(latencies) if latencies else None,
            'independent_repetitions': len(reps), 'repetition_statistics': reps,
            'local_run_success': bool(qd.get('success')),
            'CPU_preflight_success': pre.get('success'),
            'measurement_complete': q is not None and complete,
            'quality_raw': qd.get('raw_path'), 'quality_error': qd.get('error'),
            'performance_errors': errors,
            'deployment_status': 'LOCAL_DEPLOYMENT_SUCCESS' if qd.get('success') and complete else 'LOCAL_DEPLOYMENT_FAILED'}


def main():
    assert not (OUT / 'ocr_gapfill_selection.json').exists(), 'Final OCR selection/SLO write exactly once'
    proto = read(OUT / 'ocr_gapfill_shortlist.json')
    points = [eligibility(aggregate(e), 'ocr') for e in proto['entries']]
    csvsave('ocr_gapfill_measurements.csv', points)
    control = aggregate({'task': 'ocr', 'model_id': 'paddleocr', 'revision': None})
    control_valid = control['s0_ms'] is not None and control['safe_memory_mib'] is not None
    save('module14b_baseline_control.json', {'control': control, 'control_valid': control_valid,
        's0_ratio_vs_frozen': control['s0_ms'] / proto['baseline']['s0_ms'] if control_valid else None,
        'safe_memory_ratio_vs_frozen': control['safe_memory_mib'] / proto['baseline']['safe_memory_mib'] if control_valid else None,
        'historical_denominators_redefined': False, 'no_outlier_filter_or_best_run_selection': True})
    winner = select(points, 'ocr') if control_valid else None
    slo = (proto['baseline']['quality'] + winner['quality']) / 2 if winner else None
    if winner: assert proto['baseline']['quality'] < slo <= winner['quality']
    save('ocr_gapfill_selection.json', {'at_utc': now(), 'frozen': True,
        'status': 'SELECTED_HEAVY_OCR' if winner else 'OCR_HEAVY_SPECIALIST_NOT_FOUND',
        'selected': winner, 'candidates': points, 'control_valid': control_valid,
        'selection_rule': proto['selection_rule'], 'new_OCR_HQ_SLO': {
             'metric': 'exact_match', 'operator': '>=', 'value': slo,
             'construction': 'Midpoint of old/new measured quality, one-time freeze'} if winner else None,
        'deployment_or_V3_outcome_used': 'NO', 'final_OCR_round_exhausted': True,
        'third_round_forbidden': True})
    oldselection = read(OUT / 'hq_heavy_specialist_selection.json')
    oldslo = read(OUT / 'hq_quality_slo.json')
    taskmap, slomap = {}, {}
    for task in ['asr', 'lid', 'text']:
        w = oldselection['tasks'][task]['selected']
        assert w is not None
        point = eligibility(w, task)
        assert point['HEAVY_ELIGIBLE']
        taskmap[task] = {'selected': point, 'status': 'MODULE14A_FROZEN_RETAINED',
                         'source': ref(OUT / 'hq_heavy_specialist_selection.json')}
        slomap[task] = oldslo['tasks'][task]['new_HQ_SLO']
    image = read(OUT / 'image_heavy_selection.json')['selected']
    taskmap['image'] = {'selected': image, 'status': 'SELECTED_CEILING_AWARE_HEAVY'}
    taskmap['ocr'] = {'selected': winner, 'status': 'SELECTED_HEAVY_OCR' if winner else 'OCR_HEAVY_SPECIALIST_NOT_FOUND'}
    slomap['image'] = {'metric': 'accuracy', 'operator': '>=', 'value': 1.0,
                       'ceiling_exception': True, 'fixed_heavy_model_identity_required': image['model_id']}
    slomap['ocr'] = {'metric': 'exact_match', 'operator': '>=', 'value': slo,
                     'semantic_admission_retained': True} if winner else None
    ready = all(t['selected'] is not None and t['selected']['HEAVY_ELIGIBLE'] for t in taskmap.values())
    gate = 'HEAVY_SPECIALIST_POOL_READY' if ready else 'HEAVY_SPECIALIST_POOL_PARTIAL'
    groups = {}
    oldpool = read(OUT / 'high_quality_heavy_pool.json')['models']
    imageentry = next(e for e in read(OUT / 'measurement_protocol_freeze.json')['entries'] if e['model_id'] == image['model_id'])
    for task, row in taskmap.items():
        w = row['selected']
        if not w: continue
        name = w['model_id'] + '@' + w['revision']
        group = groups.setdefault(name, {'model_id': w['model_id'], 'revision': w['revision'],
            'tasks': {}, 'safe_memory_residency_envelope_mib': w['safe_memory_mib'],
            'local_artifact': oldpool[name]['local_artifact'] if name in oldpool else imageentry['directory'] if task == 'image' else w['checkpoint'],
            'new_g_gt1_measurement_in_Module14B': False})
        group['tasks'][task] = {k: w[k] for k in ['quality', 's0_ms', 'safe_memory_mib', 'quality_delta', 'latency_ratio', 'memory_ratio']}
        group['safe_memory_residency_envelope_mib'] = max(group['safe_memory_residency_envelope_mib'], w['safe_memory_mib'])
    save('quality_nondegraded_heavy_slo.json', {'at_utc': now(), 'regime': REGIME,
        'frozen_once': True, 'ASR_LID_Text_SLOs_exactly_preserved_from_Module14A': True,
        'tasks': slomap, 'full_five_task_SLO': ready, 'old_efficiency_quality_policy_changed': False,
        'deployment_or_V3_outcome_used': 'NO'})
    save('quality_nondegraded_heavy_pool.json', {'at_utc': now(), 'regime': REGIME,
        'complete_five_task_pool': ready, 'tasks': taskmap, 'models': groups,
        'quality_SLOs': slomap, 'gate': gate, 'old_regime': 'EFFICIENCY_OPTIMAL_SPECIALIST_REGIME',
        'old_regime_unmodified': True, 'VQA_specialist': None,
        'no_baseline_fallback_for_missing_task': True,
        'shared_checkpoint_memory': 'Shared resident envelope=max across independently measured task paths, not sum',
        'deployment_or_V3_outcome_used': 'NO', 'g_gt1_profiling_started': False})
    save('module14b_gate.json', {'at_utc': now(), 'gate': gate,
        'selected_task_count': sum(t['selected'] is not None for t in taskmap.values()),
        'missing_tasks': [t for t, row in taskmap.items() if row['selected'] is None],
        'OCR_status': 'SELECTED_HEAVY_OCR' if winner else 'OCR_HEAVY_SPECIALIST_NOT_FOUND',
        'OCR_final_gap_round_completed': True, 'third_OCR_round_allowed': False,
        'Image_ceiling_exception_authorized': True,
        'other_four_tasks_quality_rule_relaxed': False,
        'deployment_or_V3_outcome_used': 'NO', 'stop_after_Module14B': True})
    print('FINAL_POOL', gate, 'OCR', winner['model_id'] if winner else 'NOT_FOUND', 'SLO', slo, flush=True)


if __name__ == '__main__': main()
