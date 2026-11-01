"""Exact12 rebound resource model and whole-block donor provider.

Native inputs retain their new schema. Only the existing pure resource table,
whole-block averaging and timeline mathematics are reused. Preparation reads
source metadata only; complete exact12 intake precedes resource label access.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from . import twoFIFO_source_rebound_phase12_fit_v1 as response
from . import twoFIFO_resource_provider_v3 as resource
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA = 'gsdeploy.twoFIFO_source_rebound_resource_training.v1'
require, receipt, checked = response.require, response.receipt, response.checked
DESIGN = dict(whole_block_count=3, attempts_per_block=4, measured_requests=546,
    history_only_priming_requests=216, training_attempts_per_fold=8,
    heldout_attempts_per_fold=4, minimum_supported_whole_blocks=2,
    scenario_choices=['mean', 'max'], equal_block_aggregation=True,
    memory_key='EXACT_STATE_AND_ORDERED_PRIOR_STATE_PATH',
    memory_history_reset_between_cases=False,
    energy_scope='ORIGINAL_MEASUREMENT_ONLY_EXCLUDES_DRAIN',
    memory_scope='OWNED_RESIDENT_PEAK_STARTUP_THROUGH_CURRENT_DRAIN',
    startup_occurrences_per_attempt=1, startup_amortized_in_provider=False,
    memory_safety_factor_applied_in_provider=False,
    native_schema_relabeling=False, heldout_unit='whole_capture_block_all_four_shards')


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return receipt(path)


def prepare(fit_manifest_path, output):
    """Predeclare before final exact12 exists; do not inspect native labels."""
    manifest_ref = receipt(fit_manifest_path)
    manifest = response.audit(fit_manifest_path)
    require(not Path(manifest['expected_continuation_report_path']).exists(),
            'Resource design must be fixed before final exact12 continuation exists')
    require(not (Path(fit_manifest_path).resolve().parent / 'phase12-fit-claim.json').exists(),
            'Resource design must be fixed before native response fitting')
    sources = strict_references([manifest_ref, manifest['sources'], implementation_sources(__file__)])
    for ref in sources:
        response.verify(ref)
    document = dict(schema=SCHEMA + '.design', status='CPU_PREDECLARED_BEFORE_EXACT12_FINAL',
        fit_manifest=manifest_ref, expected_continuation_report_path=manifest['expected_continuation_report_path'],
        design=copy.deepcopy(DESIGN), sources=sources, resource_fit_run=False,
        native_training_resource_labels_read=False, numerical_fit_run=False,
        validation_labels_read=False, GPU_started=False)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    return save(output / 'design.json', document)


def audit(design_path):
    design = checked(receipt(design_path))
    require(design.get('schema') == SCHEMA + '.design'
        and design.get('status') == 'CPU_PREDECLARED_BEFORE_EXACT12_FINAL'
        and design.get('design') == DESIGN and design.get('resource_fit_run') is False
        and design.get('native_training_resource_labels_read') is False
        and design.get('numerical_fit_run') is False and design.get('validation_labels_read') is False
        and design.get('GPU_started') is False, 'Exact predeclared resource design required')
    response.verify(design['fit_manifest'])
    manifest = response.audit(design['fit_manifest']['path'])
    require(design['expected_continuation_report_path'] == manifest['expected_continuation_report_path'],
            'Expected exact12 continuation changed')
    expected = strict_references([design['fit_manifest'], manifest['sources'], implementation_sources(__file__)])
    require(design.get('sources') == expected, 'Resource design implementation closure changed')
    for ref in expected:
        response.verify(ref)
    return design


def _fits(tables):
    """Use unchanged mathematics, retaining the complete full/fold contract."""
    require(len(tables) == 12 and len({t['attempt_id'] for t in tables}) == 12,
            'Exactly twelve distinct resource attempts required')
    full = resource.fit_tables(tables)
    require(len(full['training_block_ids']) == 3 and len(full['training_attempt_ids']) == 12,
            'Exactly three whole resource blocks required')
    folds = []
    for block in full['training_block_ids']:
        donor_tables = [t for t in tables if t['capture_id'] != block]
        donor = resource.fit_tables(donor_tables)
        heldout = sorted(t['attempt_id'] for t in tables if t['capture_id'] == block)
        require(len(donor['training_block_ids']) == 2 and len(donor['training_attempt_ids']) == 8
            and len(heldout) == 4 and not set(heldout) & set(donor['training_attempt_ids']),
            'Every fold must exclude all four heldout attempts')
        folds.append(dict(heldout_block_id=block, training_model=donor,
            heldout_attempt_ids=heldout, independent_prediction_accuracy_validated=False))
    return full, folds


def build_from_bundle(bundle, intake_document):
    """Pure callback after with_verified_bundle; native new schema stays intact."""
    response._validate_bundle(bundle)
    require(intake_document.get('schema') == response.intake.SCHEMA
        and intake_document.get('status') == 'COMPLETE_SOURCE_REBOUND_EXACT12_TRAINING_INTAKE_NOT_FIT'
        and intake_document.get('attempt_count') == 12 and intake_document.get('measured_requests') == 546
        and intake_document.get('priming_requests') == 216
        and intake_document.get('validation_labels_read') is False,
        'Complete true exact12 intake required before resource labels')
    implementation_start = implementation_sources(__file__)
    start_sources = strict_references([intake_document['sources'], bundle['review_receipts'],
        implementation_start])
    for ref in start_sources:
        response.verify(ref)
    plans = {p['attempt_id']: p for p in bundle['plans']}
    documents = {d['attempt_id']: d for d in bundle['documents']}
    require(len(documents) == 12 and set(documents) == set(plans), 'Exact twelve macro identities required')
    reviews = {}
    for ref in bundle['review_receipts']:
        review = checked(ref)
        require(review.get('schema') == response.intake.native_review.SCHEMA
            and review.get('status') == 'PASS_COMPLETE_SOURCE_REBOUND_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION'
            and review.get('training_feature_extraction_eligible') is True
            and review.get('issues') == [] and review.get('validation_labels_read') is False
            and review.get('source_rebound_manifest') == intake_document['manifest'],
            'Only complete rebound native reviews accepted')
        plan = checked(review['inputs']['plan.json'])
        require(plan['attempt_id'] not in reviews and plans.get(plan['attempt_id']) == plan,
                'Reviewed resource plan identity differs')
        reviews[plan['attempt_id']] = (ref, review)
    require(set(reviews) == set(plans), 'Every resource attempt must have one reviewed native source')
    tables, sources = [], start_sources
    for aid, plan in plans.items():
        review_ref, review = reviews[aid]
        raw_ref = response.verify(review['raw'])
        raw = checked(raw_ref)
        require(raw.get('schema') == response.executor.RAW_SCHEMA and raw.get('backend') == 'native'
            and raw.get('status') == 'completed' and raw.get('split') == 'training'
            and raw.get('attempt_id') == aid and raw.get('capture_id') == plan['capture_id']
            and raw.get('execution_id') == plan['execution_id']
            and checked(raw['plan']) == plan and raw['context'] == plan['context'],
            'Exact source-rebound raw/context identity required; no old raw-schema relabeling')
        context = checked(raw['context'])
        doc = documents[aid]
        require(doc.get('schema') == response.adapter.MACRO_SCHEMA
            and doc.get('capture_id') == plan['capture_id'] and doc.get('execution_id') == plan['execution_id']
            and doc.get('source_rebound_manifest') == intake_document['manifest'],
            'Exact source-rebound macro identity required')
        table = resource.table_from_verified(raw, doc, plan, context)
        table.update(raw=raw_ref, context=raw['context'], plan=raw['plan'],
            training_review=review_ref, capture_sha256=raw_ref['sha256'])
        tables.append(table)
        sources = strict_references([sources, raw_ref, raw['context'], raw['plan'], review['sources'], doc['sources']])
    require(sum(t['measured_requests'] for t in tables) == 546
        and sum(t['priming_requests'] for t in tables) == 216,
        'Resource tables must retain 546 measured requests and 216 priming history requests')
    full, folds = _fits(tables)
    require(implementation_sources(__file__) == implementation_start, 'Resource implementation closure changed')
    for ref in sources:
        response.verify(ref)
    return dict(resource_model=full, whole_block_leaveout=folds, sources=sources,
        denominator=copy.deepcopy(bundle['denominator']), measured_requests=546,
        history_only_priming_requests=216, training_attempts=12, resource_fit_run=True,
        native_training_resource_labels_read=True, response_numerical_fit_run=False,
        validation_labels_read=False, GPU_started=False, resource_prediction_qualified=False,
        main_queue_provider_eligible=False, final_gain_eligible=False)


def run(design_path, output, *, fit_enabled=False):
    """Gate all native label access on complete exact12 and explicit opt-in."""
    design_ref = receipt(design_path)
    design = audit(design_path)
    manifest = response.audit(design['fit_manifest']['path'])
    final_path = Path(design['expected_continuation_report_path'])
    result = dict(schema=SCHEMA, design=design_ref, status='WAITING_FOR_EXACT12_CONTINUATION_FINAL',
        resource_fit_run=False, native_training_resource_labels_read=False,
        validation_labels_read=False, GPU_started=False, resource_prediction_qualified=False,
        main_queue_provider_eligible=False, final_gain_eligible=False,
        sources=strict_references([design_ref, design['sources']]))
    if final_path.is_file():
        final_ref = receipt(final_path)
        result['continuation'] = final_ref
        if not response._complete_final(manifest, checked(final_ref)):
            result['status'] = 'HOLD_INCOMPLETE_OR_FOREIGN_CONTINUATION_FINAL'
        elif not fit_enabled:
            result['status'] = 'READY_EXPLICIT_RESOURCE_FIT_REQUIRED'
        else:
            def callback(bundle, intake_doc):
                fitted = build_from_bundle(bundle, intake_doc)
                fitted['intake_evidence'] = copy.deepcopy(intake_doc)
                return fitted
            result.update(response.with_verified_bundle(design['fit_manifest']['path'], final_path, callback))
            result.update(status='COMPLETE_SOURCE_REBOUND_RESOURCE_TRAINING_NOT_QUALIFICATION')
        result['sources'] = strict_references([result['sources'], design_ref, design['sources'], final_ref])
    audit(design_path)
    response.verify(design_ref)
    for ref in result['sources']:
        response.verify(ref)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    return save(output / 'model.json', result)


class FoldResourceTrainingProvider(resource.ResourceTrainingProvider):
    """Reauthenticate exact12 and rederive full model plus all three folds."""
    def __init__(self, model_path, target_plan, target_context, *, heldout_block_id,
                 allow_training_resource_hypothesis=False, scenario='mean'):
        require(allow_training_resource_hypothesis is True, 'Explicit training resource hypothesis opt-in required')
        require(scenario in ('mean', 'max'), 'Explicit empirical resource scenario required')
        model_ref = receipt(model_path)
        doc = checked(model_ref)
        require(doc.get('schema') == SCHEMA
            and doc.get('status') == 'COMPLETE_SOURCE_REBOUND_RESOURCE_TRAINING_NOT_QUALIFICATION'
            and doc.get('resource_fit_run') is True and doc.get('validation_labels_read') is False
            and doc.get('GPU_started') is False, 'Complete new-schema resource model required')
        for ref in doc['sources']:
            response.verify(ref)
        design = audit(doc['design']['path'])
        response.verify(doc['design'])
        response.verify(doc['continuation'])
        require(doc['continuation']['path'] == design['expected_continuation_report_path'],
                'Resource model continuation differs from design')
        def rederive(bundle, intake_doc):
            require(intake_doc == doc['intake_evidence'], 'Resource native intake changed')
            return build_from_bundle(bundle, intake_doc)
        actual = response.with_verified_bundle(design['fit_manifest']['path'],
            doc['continuation']['path'], rederive)
        require(all(doc.get(k) == v for k, v in actual.items() if k != 'sources'),
                'Full resource law and all three folds must rederive from native polls')
        expected_sources = strict_references([actual['sources'], doc['design'], design['sources'], doc['continuation']])
        require(doc['sources'] == expected_sources, 'Resource model source closure differs')
        selected = next((f for f in actual['whole_block_leaveout'] if f['heldout_block_id'] == heldout_block_id), None)
        require(selected is not None, 'Exact whole heldout block required')
        require(target_plan.get('capture_id') == heldout_block_id
            and target_plan.get('attempt_id') in selected['heldout_attempt_ids'],
            'Target must belong to the selected four-attempt heldout block')
        identity = resource.physical_execution_identity(target_plan, target_context)
        require(identity == selected['training_model']['physical_execution_identity'],
                'Target physical worker/config/priming axes differ')
        self.document = copy.deepcopy(selected['training_model'])
        self.scenario = scenario
        self.target_plan, self.target_context = copy.deepcopy(target_plan), copy.deepcopy(target_context)
        self.physical_execution_identity = identity
        self.fold_binding = dict(schema=SCHEMA + '.fold_provider', full_native_model=model_ref,
            selected_fold_model_sha256=resource.digest(self.document), heldout_block_id=heldout_block_id,
            training_block_ids=copy.deepcopy(self.document['training_block_ids']),
            training_attempt_ids=copy.deepcopy(self.document['training_attempt_ids']),
            heldout_attempt_ids=copy.deepcopy(selected['heldout_attempt_ids']),
            sources=strict_references([doc['sources'], model_ref, implementation_sources(__file__)]),
            all_three_folds_rederived=True, complete_twelve_native_attempt_gate=True,
            startup_recomputed_from_only_eight_donor_attempts=True,
            resource_prediction_qualified=False, main_queue_provider_eligible=False,
            final_gain_eligible=False, validation_labels_read=False, GPU_started=False)
        for ref in self.fold_binding['sources']:
            response.verify(ref)


def predict_timeline(timeline, provider, *, target_context, execution_binding):
    require(isinstance(provider, FoldResourceTrainingProvider), 'Explicit rebound whole-block resource provider required')
    result = resource.predict_timeline(timeline, provider, target_context=target_context,
        execution_binding=execution_binding)
    result['resource_fold_binding'] = copy.deepcopy(provider.fold_binding)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--fit-manifest', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('audit'); p.add_argument('--design', required=True)
    p = sub.add_parser('run'); p.add_argument('--design', required=True); p.add_argument('--output', required=True)
    p.add_argument('--fit', action='store_true', default=False)
    args = parser.parse_args()
    value = prepare(args.fit_manifest, args.output) if args.command == 'prepare' else (
        audit(args.design) if args.command == 'audit' else run(args.design, args.output, fit_enabled=args.fit))
    print(json.dumps(value))
