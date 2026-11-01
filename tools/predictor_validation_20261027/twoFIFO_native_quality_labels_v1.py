"""Actual output quality labels; never a forecasting provider.

Frozen references must match the exact prepared input payload. Missing
outputs remain unknown, including requests not completed by the fixed drain.
Original task scoring, error/reference aggregation and offered rates are used.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from gsdeploy.prediction import components
from gsdeploy.quality import quality_digest, sample_totals
from gsdeploy.tasks import page_ocr_text, score_result
from . import twoFIFO_source_rebound_observed_score_v1 as labels
from .stage_response_features_v1 import checked, receipt

SCHEMA = 'gsdeploy.twoFIFO_actual_native_quality_label.v1'
require = labels.require


def prepared_references(plan, catalog_ref):
    """Bind labels to exact text or media bytes, preserving reference metadata."""
    catalog = checked(catalog_ref)
    quality = catalog['quality']
    require(quality.get('schema_version') == 1
        and quality_digest(quality['references']) == quality['identity']['reference_sha256'],
        'Frozen native reference fingerprint required')
    pool_ref = plan['prepared_pool']
    require(receipt(pool_ref['path']) == pool_ref, 'Prepared input pool changed')
    entries, references = {}, {}
    with Path(pool_ref['path']).open() as stream:
        for line in stream:
            entry = json.loads(line)
            key = (entry['task_type'], entry['pool_id'])
            require(key not in entries, 'Duplicate prepared task/input identity')
            sample = entry['sample']
            reference = quality['references'].get(key[0], {}).get(key[1])
            fields = ('id', 'task', 'context', 'question') if key[0] == 'text' else ('id', 'task', 'input', 'input_sha256')
            require(type(reference) is dict and sample.get('task') == key[0]
                and sample.get('id') == reference.get('id')
                and all(k in sample and k in reference and sample[k] == reference[k] for k in fields)
                and all(reference[k] == v for k, v in sample.items() if k in reference),
                'Native reference must match the complete exact prepared sample')
            if key[0] != 'text':
                require(receipt(sample['input'])['sha256'] == sample['input_sha256'],
                        'Native media payload bytes changed')
            entries[key] = sample
            references[key] = reference
    require(references and receipt(pool_ref['path']) == pool_ref
        and receipt(catalog_ref['path']) == catalog_ref, 'Native input/reference evidence changed')
    return references


def measure_case(raw, plan, case, epoch, scoring_plan, catalog_ref):
    """Use only target native outcomes as labels, with no training fallback."""
    require(raw.get('schema') == labels.executor.RAW_SCHEMA and raw.get('backend') == 'native'
        and raw.get('status') == 'completed' and raw.get('split') == 'training',
        'Complete explicitly training native raw required for this adapter')
    require(all(raw.get(k) == plan[k] for k in ('capture_id', 'attempt_id', 'execution_id')),
            'Exact native plan identity required')
    require([c for c in plan['cases'] if c['case_id'] == case['case_id']] == [case]
        and [c['observation_start_ns'] for c in raw['cases'] if c['case_id'] == case['case_id']] == [epoch],
        'Exact native case and measurement epoch required')
    references = prepared_references(plan, catalog_ref)
    actual = {r['request_id']: r for r in raw['request_records']}
    require(len(actual) == len(raw['request_records']), 'Duplicate native request identity')
    rates = scoring_plan['workload_rps']
    require(rates and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in rates.values()),
            'Original positive offered rates required')
    totals = {t: dict(numerator=0., denominator=0., offered=0, scored=0, missing=[]) for t in rates}
    rows = []
    deadline = epoch + labels.generated.MEASUREMENT_NS + labels.generated.DRAIN_NS
    for request in labels.generated.public_case(case, epoch):
        task = request['task']
        require(task in rates, 'Native task outside original scoring plan')
        target = totals[task]
        target['offered'] += 1
        row = actual.get(request['request_id'])
        require(row is not None and (row['model'], row['task'], row['pool_id'], row['scheduled_time_ns'])
            == (request['owner'], task, request['pool_id'], request['arrival_ns']),
            'Exact complete native output cohort and route required')
        require((task, row['pool_id']) in references, 'Native output has no exact prepared reference')
        if row.get('status') != 'completed' or row.get('completion_time_ns') is None or row['completion_time_ns'] > deadline:
            target['missing'].append(row['request_id'])
            rows.append(dict(request_id=row['request_id'], task=task, status='UNKNOWN_NATIVE_OUTCOME'))
            continue
        require(type(row['completion_time_ns']) is int and request['arrival_ns'] <= row['completion_time_ns'],
                'Native output receipt precedes arrival')
        response = copy.deepcopy(row['worker_response'])
        require(response.get('task') == task and response.get('request_id') == row['request_id'],
                'Native output task/request binding changed')
        require(type(response.get('success')) is bool, 'Native success must be an explicit boolean')
        if response['success'] is not True:
            target['missing'].append(row['request_id'])
            rows.append(dict(request_id=row['request_id'], task=task, status='UNVERIFIED_QUALITY_OUTPUT'))
            continue
        reference = references[task, row['pool_id']]
        require(row.get('sample_id') == reference['id'], 'Native output sample identity changed')
        if task == 'ocr':
            response['prediction'] = page_ocr_text(response['prediction'])
        numerator, denominator = sample_totals(task, score_result(reference, response))
        target['numerator'] += numerator
        target['denominator'] += denominator
        target['scored'] += 1
        rows.append(dict(request_id=row['request_id'], task=task, pool_id=row['pool_id'],
            owner=row['model'], status='MEASURED_NATIVE_OUTCOME', numerator=numerator, denominator=denominator))
    reasons = []
    for task, target in totals.items():
        if target['offered'] == 0:
            reasons.append('NO_COHORT_OBSERVATION_FOR_TASK:' + task)
        if target['missing']:
            reasons.append('MISSING_NATIVE_OUTCOME:' + task)
        known = target['offered'] > 0 and not target['missing'] and target['denominator'] > 0
        target['loss'] = target['numerator'] / target['denominator'] if known else None
    Q = (math.fsum(rates[t] / math.fsum(rates.values()) * v['loss'] for t, v in totals.items())
         if all(v['loss'] is not None for v in totals.values()) else None)
    require(receipt(catalog_ref['path']) == catalog_ref and receipt(plan['prepared_pool']['path']) == plan['prepared_pool'],
            'Native quality sources changed while evaluating labels')
    return dict(schema=SCHEMA, status='ACTUAL_TRAINING_OUTCOME_LABEL_NOT_PREDICTION',
        attempt_id=plan['attempt_id'], case_id=case['case_id'], measured_Q=Q, per_task=totals,
        requests=rows, offered_requests=len(rows), unknown_reasons=reasons,
        scoring_contract=copy.deepcopy(scoring_plan),
        source_evidence=dict(catalog=catalog_ref, prepared_pool=plan['prepared_pool']),
        quality_from_frozen_training_expectations_not_native_outcome=False,
        numerator_denominator_summed_before_division=True, training_quality_fallback_used=False,
        future_outcomes_used_for_evaluation_only=True, validation_labels_read=False,
        native_evidence_authenticated_by_this_pure_helper=False, final_gain_eligible=False, GPU_started=False)


def with_measured_quality(observed_label, measured_quality, scoring_plan):
    """Keep T/L/P/M/E and original weights; substitute actual measured Q only."""
    require(measured_quality.get('schema') == SCHEMA, 'Explicit measured native quality label required')
    require(observed_label.get('schema') == labels.SCHEMA
        and all(observed_label[k] == measured_quality[k] for k in ('attempt_id', 'case_id')),
        'Matching measured native score label required')
    require(scoring_plan['weights'] == {k: 1/6 for k in 'QTLPME'}, 'Original equal six weights required')
    require(measured_quality.get('scoring_contract') == scoring_plan,
            'Actual native quality must use the same complete scoring contract')
    rates, refs = scoring_plan['workload_rps'], scoring_plan['latency_reference_ms']
    require(rates and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in rates.values())
        and all(t in refs and type(refs[t]) in (int, float) and math.isfinite(refs[t]) and refs[t] > 0 for t in rates),
        'Finite positive original rates and latency references required')
    observed_score = observed_label['measured_original_score']
    resources = observed_label['measured_resources']
    absolute_energy_known = resources['startup_energy_joules'] is not None and resources['measurement_board_energy_joules'] is not None
    score = copy.deepcopy(observed_score)
    require(set(score['terms']) == set('QTLPME'), 'Exactly the six original terms required')
    score['terms']['Q'] = measured_quality['measured_Q']
    score['unknown_reasons'] = [r for r in score['unknown_reasons'] if not r.startswith('UNKNOWN_AGGREGATE_QUALITY:')]
    score['unknown_reasons'] = sorted(set(score['unknown_reasons'] + measured_quality['unknown_reasons']))
    terms = score['terms']
    require(all(v is None or type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in terms.values()),
            'Every known score term must be finite and nonnegative')
    complete = all(v is not None for v in terms.values())
    metric = (components(score['per_task'], scoring_plan['workload_rps'], scoring_plan['latency_reference_ms'],
        terms['T'], terms['M'], scoring_plan['weights'], E=terms['E'], Q=terms['Q'])
        if terms['L'] is not None and terms['P'] is not None else None)
    if metric is not None:
        require(all(metric[k + '_norm'] is None if terms[k] is None else
                    metric[k + '_norm'] is not None and math.isclose(metric[k + '_norm'], terms[k], rel_tol=1e-12, abs_tol=1e-12)
                    for k in 'TLPME'), 'Scoring contract changes a preserved non-quality term')
        require(metric['U'] is None or math.isfinite(metric['U']), 'Finite six-term utility required')
    score.update(metric_components=metric, six_term_comparison_complete=complete,
        absolute_utility_complete=complete and absolute_energy_known,
        cost=-metric['U'] if complete and metric is not None else None,
        full_U=metric['U'] if complete and absolute_energy_known and metric is not None else None,
        quality_scope='Actual native outputs; task summed errors/reference units; original offered task weights',
        quality_totals_by_task=copy.deepcopy(measured_quality['per_task']))
    return score
