"""Explicit recovery provenance with the unchanged phase12 numerical interface."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile

from . import twoFIFO_prelaunch_recovery_v1 as recovery
from . import twoFIFO_prelaunch_recovery_intake_v1 as intake
from . import twoFIFO_source_rebound_phase12_fit_v1 as numeric
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA = 'gsdeploy.twoFIFO_recovery_phase12.v1'
require, receipt, checked, verify = numeric.require, numeric.receipt, numeric.checked, numeric.verify
FEATURES, DESIGN = numeric.FEATURES, numeric.DESIGN
adapter, executor = numeric.adapter, numeric.executor
fit_bundle, _validate_bundle = numeric.fit_bundle, numeric._validate_bundle
INTAKE_STATUS = 'COMPLETE_SOURCE_REBOUND_EXACT12_TRAINING_INTAKE_NOT_FIT'


def _save(path, value):
    numeric._write(path, value)
    return receipt(path)


def _entries():
    return [receipt(m.__file__) for m in (recovery, intake, numeric, numeric.adapter)] + [receipt(__file__)]


def prepare(recovery_manifest_path, continuation_path, independent_review_paths, output):
    recovery_ref = receipt(recovery_manifest_path)
    recovered = recovery.audit(recovery_manifest_path)
    final = Path(continuation_path).resolve()
    require(final == Path(recovered['run_output']) / 'report.json' and not final.exists(),
            'Predeclare the exact new recovery final before it exists')
    reviews = [receipt(p) for p in independent_review_paths]
    numeric._review_bindings(reviews, _entries())
    sources = strict_references([implementation_sources(__file__), recovery_ref, recovered['sources'], reviews])
    for ref in sources:
        verify(ref)
    manifest = dict(schema=SCHEMA + '.manifest', status='CPU_PREDECLARED_RECOVERY_PHASE12_NO_FIT',
        recovery_manifest=recovery_ref, original_executor_manifest=recovered['proof']['original_manifest'],
        expected_continuation_report_path=str(final), independent_reviews=reviews,
        reviewed_entry_sources=_entries(), sources=sources, features=list(FEATURES), design=DESIGN,
        measured_requests=546, history_only_priming_requests=216, blocks=3, attempts=12,
        training_attempts_per_fold=8, heldout_attempts_per_fold=4, producer_schema=SCHEMA,
        upstream_intake_schema=intake.SCHEMA, numerical_row_interface=numeric.adapter.SCHEMA,
        numerical_math_unchanged=True, automatic_fit=False, numerical_fit_run=False,
        validation_labels_read=False, GPU_started=False)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    return _save(output / 'manifest.json', manifest)


def audit(path):
    manifest = checked(receipt(path))
    require(manifest.get('schema') == SCHEMA + '.manifest'
        and manifest.get('status') == 'CPU_PREDECLARED_RECOVERY_PHASE12_NO_FIT'
        and all(manifest.get(k) == v for k, v in dict(features=list(FEATURES), design=DESIGN,
            measured_requests=546, history_only_priming_requests=216, blocks=3, attempts=12,
            training_attempts_per_fold=8, heldout_attempts_per_fold=4, producer_schema=SCHEMA,
            upstream_intake_schema=intake.SCHEMA, numerical_row_interface=numeric.adapter.SCHEMA,
            numerical_math_unchanged=True, automatic_fit=False, numerical_fit_run=False,
            validation_labels_read=False, GPU_started=False).items()), 'Exact recovery phase12 contract required')
    verify(manifest['recovery_manifest'])
    recovered = recovery.audit(manifest['recovery_manifest']['path'])
    require(manifest['original_executor_manifest'] == recovered['proof']['original_manifest']
        and manifest['expected_continuation_report_path'] == str(Path(recovered['run_output']) / 'report.json')
        and manifest['reviewed_entry_sources'] == _entries(), 'Recovery producer/source/final binding changed')
    numeric._review_bindings(manifest['independent_reviews'], _entries())
    expected = strict_references([implementation_sources(__file__), manifest['recovery_manifest'],
        recovered['sources'], manifest['independent_reviews']])
    require(manifest['sources'] == expected, 'Recovery coordinator implementation closure changed')
    for ref in expected:
        verify(ref)
    return manifest


def complete_final(manifest, final):
    """Metadata role only; full recovery/native proof is intake's next gate."""
    recovered = recovery.audit(manifest['recovery_manifest']['path'])
    rows, tail = final.get('completed_attempts'), final.get('completed_tail')
    return (final.get('schema') == recovery.SCHEMA + '.run' and final.get('status') == recovery.FINAL
        and final.get('policy') == recovery.POLICY and final.get('recovery_manifest') == manifest['recovery_manifest']
        and final.get('original_manifest') == manifest['original_executor_manifest']
        and final.get('stopped_report') == recovered['proof']['stopped_report']
        and final.get('prefix_attempts') == recovered['proof']['prefix']
        and type(rows) is list and len(rows) == 12 and type(tail) is list and len(tail) == 8
        and rows[:4] == recovered['proof']['prefix'] and rows[4:] == tail
        and all(type(r) is dict for r in rows)
        and [r.get('attempt_id') for r in tail] == [r['attempt_id'] for r in recovered['proof']['tail']]
        and final.get('no_started_capture_retries') is True and final.get('fit_eligible') is False
        and final.get('numerical_fit_run') is False and final.get('automatic_fit') is False
        and final.get('validation_labels_read') is False)


def check_intake(document, manifest):
    recovered = recovery.audit(manifest['recovery_manifest']['path'])
    require(document.get('schema') == intake.SCHEMA and document.get('status') == INTAKE_STATUS
        and document.get('recovery_manifest') == manifest['recovery_manifest']
        and document.get('manifest') == manifest['original_executor_manifest']
        and document.get('continuation', {}).get('path') == manifest['expected_continuation_report_path']
        and document.get('recovery_policy') == recovery.POLICY
        and document.get('stopped_report') == recovered['proof']['stopped_report']
        and document.get('preserved_prefix_attempts') == 4 and document.get('new_policy_tail_attempts') == 8
        and document.get('attempt_count') == 12 and document.get('measured_requests') == 546
        and document.get('priming_requests') == 216 and document.get('numerical_fit_run') is False
        and document.get('validation_labels_read') is False,
        'Exact new recovery intake required; old intake/producer is rejected')
    require(document['continuation'] == receipt(manifest['expected_continuation_report_path']),
            'Recovery intake final receipt changed')


def row_bundle(document, workdir):
    """Normalize verified new provenance to a shared mathematical row type."""
    require(document.get('schema') == intake.SCHEMA and document.get('status') == INTAKE_STATUS,
            'Only new recovery intake can compile recovery rows')
    Path(workdir).mkdir(parents=True, exist_ok=False)
    plans, docs, refs = numeric.adapter._rederive(document, workdir=workdir)
    rows, denominator = numeric.adapter._rows(docs, plans)
    bundle = dict(schema=numeric.adapter.SCHEMA, producer_schema=SCHEMA + '.rows',
        upstream_intake_schema=intake.SCHEMA, numerical_row_interface_only=True,
        features=list(FEATURES), design=DESIGN, numerical_fit_run=False,
        validation_labels_read=False, GPU_started=False, plans=plans, documents=docs,
        review_receipts=refs, denominator=denominator, full=dict(rows=rows),
        observed_support=numeric.adapter.frozen.observed_support(docs, plans))
    numeric._validate_bundle(bundle)
    return bundle


def with_verified_bundle(manifest_path, continuation_path, callback):
    manifest_ref = receipt(manifest_path); manifest = audit(manifest_path)
    final_ref = receipt(continuation_path)
    require(final_ref['path'] == manifest['expected_continuation_report_path']
        and complete_final(manifest, checked(final_ref)), 'Exact complete new-policy recovery final required')
    with tempfile.TemporaryDirectory(prefix='recovery-phase12-') as work:
        intake_ref = intake.intake(manifest['original_executor_manifest']['path'], final_ref['path'],
            recovery_manifest_path=manifest['recovery_manifest']['path'], output=Path(work) / 'intake')
        document = checked(intake_ref); check_intake(document, manifest)
        for ref in document['sources']:
            verify(ref)
        try:
            bundle = row_bundle(document, Path(work) / 'rows')
            return callback(bundle, copy.deepcopy(document))
        finally:
            for ref in document['sources']:
                verify(ref)
            verify(final_ref); verify(manifest_ref); audit(manifest_path)


def run(manifest_path, output, *, fit_enabled=False):
    manifest_ref = receipt(manifest_path); manifest = audit(manifest_path)
    final = Path(manifest['expected_continuation_report_path'])
    result = dict(schema=SCHEMA, manifest=manifest_ref, status='WAITING_FOR_RECOVERY_EXACT12_FINAL',
        numerical_fit_run=False, validation_labels_read=False, GPU_started=False,
        main_queue_provider_eligible=False, final_gain_eligible=False,
        sources=strict_references([manifest_ref, manifest['sources']]))
    if final.is_file():
        final_ref = receipt(final); result['continuation'] = final_ref
        if not complete_final(manifest, checked(final_ref)):
            result['status'] = 'HOLD_INCOMPLETE_OR_FOREIGN_RECOVERY_FINAL'
        elif not fit_enabled:
            result['status'] = 'READY_EXPLICIT_RECOVERY_FIT_REQUIRED'
        else:
            _save(Path(manifest_path).resolve().parent / 'phase12-fit-claim.json',
                dict(manifest=manifest_ref, continuation=final_ref, no_retries=True))
            def fit(bundle, document):
                result['numerical_fit_run'] = True
                return numeric.fit_bundle(bundle), document
            try:
                fitted, document = with_verified_bundle(manifest_path, final, fit)
                result.update(status='COMPLETE_RECOVERY_SOURCE_BOUND_PHASE12_TRAINING_DIAGNOSTICS',
                    fit=fitted, intake_evidence=document)
                result['sources'] = strict_references([result['sources'], document['sources']])
            except Exception as exc:
                result.update(status='STOPPED_RECOVERY_FIT_NO_RETRY', error=type(exc).__name__ + ': ' + str(exc))
        result['sources'] = strict_references([result['sources'], final_ref])
    verify(manifest_ref); audit(manifest_path)
    for ref in result['sources']:
        verify(ref)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    return _save(output / 'report.json', result)


def audit_fit(value):
    ref = verify(value) if type(value) is dict else receipt(value)
    report = checked(ref)
    require(report.get('schema') == SCHEMA
        and report.get('status') == 'COMPLETE_RECOVERY_SOURCE_BOUND_PHASE12_TRAINING_DIAGNOSTICS'
        and report.get('numerical_fit_run') is True and report.get('validation_labels_read') is False
        and report.get('GPU_started') is False, 'Saved recovery fit required')
    verify(report['manifest']); manifest = audit(report['manifest']['path'])
    verify(report['continuation'])
    require(report['continuation']['path'] == manifest['expected_continuation_report_path']
        and complete_final(manifest, checked(report['continuation'])), 'Saved recovery final role changed')
    check_intake(report['intake_evidence'], manifest)
    expected = strict_references([report['manifest'], manifest['sources'], report['continuation'],
        report['intake_evidence']['sources']])
    require(report['sources'] == expected and report['fit'].get('features') == list(FEATURES)
        and report['fit'].get('design') == DESIGN and report['fit'].get('measured_requests') == 546
        and report['fit'].get('priming_history_requests') == 216
        and report['fit'].get('no_columns_deleted') is True and report['fit'].get('thresholds_changed') is False
        and len(report['fit'].get('whole_block_leaveout', [])) == 3, 'Saved recovery math/source contract changed')
    for source in expected:
        verify(source)
    return dict(report=report, receipt=ref, numeric_recomputed=False, source_contract_verified=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--recovery-manifest', required=True); a.add_argument('--continuation', required=True)
    a.add_argument('--independent-reviews', nargs='+', required=True); a.add_argument('--output', required=True)
    a = sub.add_parser('audit'); a.add_argument('--manifest', required=True)
    a = sub.add_parser('run'); a.add_argument('--manifest', required=True); a.add_argument('--output', required=True); a.add_argument('--fit', action='store_true')
    a = p.parse_args()
    print(json.dumps(prepare(a.recovery_manifest,a.continuation,a.independent_reviews,a.output)
        if a.command=='prepare' else audit(a.manifest) if a.command=='audit' else run(a.manifest,a.output,fit_enabled=a.fit)))
