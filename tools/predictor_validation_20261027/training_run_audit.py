"""CPU-only postcollection audit of prospective training runs.

Native observations and their scored annotation are immutable. Recompute quality
from the frozen catalog and remap all actual semantic events without fitting a
model or starting workers. Each report is created exclusively, never replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from pathlib import Path
import time

from gsdeploy.quality import sample_totals
from gsdeploy.tasks import page_ocr_text, score_result
from .flow_crossfit_report import scored_heldout
from .flow_profiles import map_flow_collection
from .native_flow import checked, receipt, save
from .probe_validation import read
from .profiling import digest


def quality_totals(records, catalog):
    """Reproduce per-request observed Q; reject incomplete or duplicate rows."""
    totals = {}
    for row in records:
        rid = row['request_id']
        if rid in totals or row.get('status') != 'completed':
            raise ValueError('Unique completed full-cohort rows required')
        task = row['task']
        result = copy.deepcopy(row['worker_response'])
        if task == 'ocr':
            result['prediction'] = page_ocr_text(result['prediction'])
        reference = catalog['quality']['references'][task][row['pool_id']]
        numerator, denominator = sample_totals(task, score_result(reference, result))
        totals[rid] = dict(task=task, numerator=numerator, denominator=denominator)
    return totals


def verify_quality_annotation(raw, plan, catalog):
    evidence = raw['quality_evidence']
    if evidence.get('kind') != 'measured_per_request' or evidence.get('errors'):
        raise ValueError('Clean measured per-request quality evidence required')
    totals = quality_totals(raw['request_records'], catalog)
    if totals != evidence.get('totals_by_request'):
        raise ValueError('Measured Q annotation differs from independently recomputed totals')
    expected = {r['request_id'] for r in plan['trace']}
    if set(totals) != expected or len(totals) != len(plan['trace']):
        raise ValueError('Quality must cover every offered request exactly once')
    if plan['catalog'] not in evidence.get('source_receipts', []):
        raise ValueError('Measured Q must bind the exact frozen plan catalog')
    return {'count': len(totals), 'task_counts': dict(Counter(t['task'] for t in totals.values())),
        'totals_sha256': digest(totals), 'independently_recomputed': True,
        'complete_offered_cohort': True, 'source_receipts': evidence['source_receipts']}


def audit_run(raw_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    raw_ref = receipt(Path(raw_path).resolve())
    raw, scored_ref = scored_heldout(raw_ref)
    plan = read(checked(raw['plan']))
    if plan.get('phase') != 'predictor_training' or plan.get('split') != 'training':
        raise ValueError('Only prospective training runs may enter this audit')
    # Strict remapping independently verifies sources, runtime cohort, phases,
    # telemetry, FIFO, bounded drain and clean shutdown on the original raw.
    mapped = map_flow_collection(raw_ref['path'])
    quality = verify_quality_annotation(raw, plan, read(checked(plan['catalog'])))
    trace = mapped['traces'][0]
    resources = mapped['resource_evidence']
    result = {'schema': 'gsdeploy.training_run_postcollection_audit.v1',
        'status': 'PASS', 'created_ns': time.perf_counter_ns(), 'source': receipt(__file__),
        'raw': raw_ref, 'scored_raw': scored_ref, 'plan': raw['plan'],
        'candidate_id': raw['candidate_id'], 'seed': plan['seed'],
        'window_seconds': plan['window_seconds'], 'drain_seconds': plan['drain_seconds'],
        'measurement_role': plan.get('measurement_role'),
        'run_label': plan.get('training_run_label'),
        'request_count': len(trace['requests']), 'quality': quality,
        'semantic_trace_sha256': digest(trace),
        'semantic_segment_count': sum(len(r['segments']) for r in trace['requests']),
        'mapping_report': mapped['mapping_report'],
        'resources': {k: resources[k] for k in ('startup_joules','measurement_joules','drain_joules',
            'raw_all_sampled_owned_peak_mib','raw_prephase_sampled_owned_peak_mib',
            'raw_measurement_and_drain_sampled_owned_peak_mib','safe_factor_applied','capacity_bound')},
        'board_sample_count': len(raw['board_samples']), 'native_observations_modified': False,
        'GPU_started': False, 'fitting_performed': False, 'independent_validation': False,
        'final_gain_eligible': False}
    save(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = audit_run(args.raw, args.output)
    print('PASS', result['candidate_id'], result['request_count'], 'requests; GPU_started=False')


if __name__ == '__main__':
    main()
