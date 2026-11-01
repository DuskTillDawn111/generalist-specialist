"""Fit natural handoff and original quality units from startup-aware training.

This is a new adapter; the older v2 inputs and receipts are left intact. Each
input profile requires two distinct raw captures. Repeated trials inside one
capture never satisfy that requirement, and validation is never accepted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path

from gsdeploy.forecasting.lifecycle import export_training_observations, fit_lifecycle
from gsdeploy.quality import sample_totals
from gsdeploy.tasks import page_ocr_text, score_result
from .native_collect import receipt, save
from .natural_lifecycle_fit_v2 import CLOCK_SCHEMA
from .natural_quality_fit_v2 import expected_totals
from .profiling import RESIDENTS
from .small_training_domain_bridge_review_v4 import load_report
from .smoke_verifier_v3 import check_receipt
from .startup_prediction_fit_v4 import targets


def load_natural(path):
    report, plan, raw = load_report(path, 'natural')
    context = json.loads(Path(report['context']['path']).read_text())
    ctx = plan['execution_context_sha256']
    if (raw.get('execution_context_sha256') != ctx
            or context['execution'].get('admission') != 'default_shared_no_permit'
            or raw.get('not_for_training', False)
            or any(job.get('split') != 'training' for job in plan['jobs'])):
        raise ValueError('Same-context, exclusively training shared natural source required')
    targets(raw)  # Also enforces completed native, exact lifecycle, and telemetry.
    jobs = {job['id']: job for job in plan['jobs']}
    expected = {(job['id'], alias) for job in jobs.values() for alias in job['inputs']}
    actual = set()
    for row in raw['request_records']:
        key = row.get('profile_job_id'), row.get('model')
        if (key not in expected or key in actual or row.get('split') != 'training'
                or row.get('status') != 'completed'
                or row.get('validation_view') != 'TRAINING_ONLY_NOT_VALIDATION'):
            raise ValueError('Exact completed training job cohort required')
        job = jobs[key[0]]
        if row['pool_id'] != job['inputs'][key[1]] or row['task'] != job['tasks'][key[1]]:
            raise ValueError('Training input/task differs from declared job')
        actual.add(key)
    if actual != expected:
        raise ValueError('Incomplete training cohort')
    return report, plan, raw


def fit(reviews, clock_evidence_path, catalog_path, output):
    clock_path = Path(clock_evidence_path).resolve()
    clock = json.loads(clock_path.read_text())
    if (clock.get('schema') != CLOCK_SCHEMA or not clock.get('sources')
            or not all(check_receipt(ref) for ref in clock['sources'])):
        raise ValueError('Explicit unchanged host clock evidence required')
    contract = copy.deepcopy(clock['clock_contract'])
    contract['source'] = 'sha256:' + receipt(clock_path)['sha256']
    catalog_path = Path(catalog_path).resolve()
    catalog = json.loads(catalog_path.read_text())
    aliases = {a: dict(family=f) for a, f in RESIDENTS.items()}
    observations, totals, captures, sources = [], [], [], []
    seen, contexts = set(), set()
    for path in reviews:
        report, plan, raw = load_natural(path)
        sha = report['raw']['sha256']
        if sha in seen:
            raise ValueError('Repeated raw is not an independent capture')
        seen.add(sha)
        contexts.add(plan['execution_context_sha256'])
        if len(contexts) != 1:
            raise ValueError('Natural training execution contexts differ')
        if report['raw'] not in clock['sources'] or report['context'] not in clock['sources']:
            raise ValueError('Clock evidence must bind every actual training raw and context')
        run = 'raw-sha256:' + sha
        observations.extend(export_training_observations(
            {**raw, 'split': 'training', 'admission_enabled': False}, aliases,
            execution_context_sha256=plan['execution_context_sha256'],
            clock_contract=contract, source=run))
        for row in raw['request_records']:
            response = copy.deepcopy(row['worker_response'])
            task, pool = row['task'], row['pool_id']
            if task == 'ocr':
                response['prediction'] = page_ocr_text(response['prediction'])
            reference = catalog['quality']['references'][task][pool]
            numerator, denominator = sample_totals(task, score_result(reference, response))
            totals.append(dict(run_id=run, request_id=row['request_id'],
                               family=RESIDENTS[row['model']], task=task, pool_id=pool,
                               numerator=numerator, denominator=denominator))
        captures.append(dict(review=receipt(path), raw=report['raw'], run_id=run))
        sources.extend([receipt(path), report['raw'], report['plan'], report['context'],
                        *report['evaluator_sources']])
    if not captures:
        raise ValueError('Nonempty natural training captures required')
    grouped = defaultdict(list)
    for row in observations:
        grouped[row['family'], row['task'], row['pool_id']].append(row)
    support, eligible = [], set()
    for key, rows in sorted(grouped.items()):
        count = len({row['source'] for row in rows})
        known = count >= 2
        if known:
            eligible.add(key)
        support.append(dict(family=key[0], task=key[1], pool_id=key[2],
                            observations=len(rows), independent_capture_count=count,
                            status='SUPPORTED_TRAINING_INPUT' if known else 'UNKNOWN',
                            reason=None if known else 'TWO_INDEPENDENT_CAPTURES_PER_INPUT_REQUIRED'))
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'lifecycle-observations.json', dict(observations=observations))
    save(out / 'training-totals.json', dict(records=totals))
    save(out / 'training-index.json', dict(
        schema='gsdeploy.independent_prediction_gate_training_index.v2', split='training',
        records=[{k: row[k] for k in ('run_id', 'request_id', 'pool_id', 'task')} for row in totals]))
    refs = dict(observations=receipt(out / 'lifecycle-observations.json'),
                training_totals=receipt(out / 'training-totals.json'),
                training_index=receipt(out / 'training-index.json'))
    eligible_observations = [row for row in observations
                             if (row['family'], row['task'], row['pool_id']) in eligible]
    incomplete = False
    if eligible_observations:
        model = fit_lifecycle(eligible_observations, next(iter(contexts)))
        incomplete = any(p['scenarios']['mean'][field] is None for p in model['profiles']
                         for field in ('parent_receipt_ms', 'worker_next_ready_ms'))
        save(out / 'lifecycle.json', model)
        refs['model'] = receipt(out / 'lifecycle.json')
    quality_groups = defaultdict(list)
    for row in totals:
        if (row['family'], row['task'], row['pool_id']) in eligible:
            quality_groups[row['family'], row['task']].append(row)
    quality = {}
    evidence = 'sha256:' + refs['training_totals']['sha256']
    for (family, task), rows in sorted(quality_groups.items()):
        pools = defaultdict(list)
        for row in rows:
            pools[row['pool_id']].append(row)
        quality.setdefault(family, {})[task] = dict(
            by_input={pool: expected_totals(members, evidence) for pool, members in sorted(pools.items())},
            fallback=expected_totals(rows, evidence))
    if quality:
        save(out / 'quality-loss.json', quality)
        refs['quality_loss'] = receipt(out / 'quality-loss.json')
    root = Path(__file__).resolve().parents[2]
    source_files = [Path(__file__), Path(__file__).with_name('small_training_domain_bridge_review_v4.py'),
                    Path(__file__).with_name('natural_quality_fit_v2.py'),
                    Path(__file__).with_name('startup_prediction_fit_v4.py'),
                    root / 'gsdeploy/forecasting/lifecycle.py', root / 'gsdeploy/tasks.py', root / 'gsdeploy/quality.py']
    result = dict(schema='gsdeploy.natural_training_handoff_quality_fit.v4',
                  status=('FITTED' if len(eligible) == len(grouped) and not incomplete else 'FITTED_PARTIAL')
                  if eligible else 'UNKNOWN',
                  execution_context_sha256=next(iter(contexts)), captures=captures,
                  independent_capture_count=len(captures), request_count=len(totals),
                  profile_support=support, incomplete_handoff_boundaries=incomplete,
                  clock_evidence=receipt(clock_path), catalog=receipt(catalog_path),
                  sources=sources + [receipt(p) for p in source_files], **refs,
                  fallback_assumption='Same family/task distribution as supported training inputs; not held-out output quality.',
                  limitation='Exact-input mean joint handoff and per-request quality totals; no unseen-input handoff or load-independent tail guarantee.',
                  GPU_started=False, validation_started=False, physical_accuracy_validated=False)
    save(out / 'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews', nargs='+', required=True)
    parser.add_argument('--clock-evidence', required=True)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = fit(args.reviews, args.clock_evidence, args.catalog, args.output)
    print(json.dumps(dict(status=result['status'], requests=result['request_count'], output=args.output)))


if __name__ == '__main__':
    main()
