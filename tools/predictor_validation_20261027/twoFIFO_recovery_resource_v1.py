"""Recovery-provenance resource entry; original table/fold/timeline math only."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from . import twoFIFO_recovery_phase12_v1 as response
from . import twoFIFO_source_rebound_resource_v1 as base
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA = 'gsdeploy.twoFIFO_recovery_resource_training.v1'
COMPLETE_STATUS = 'COMPLETE_RECOVERY_RESOURCE_TRAINING_NOT_QUALIFICATION'
require, receipt, checked, verify = response.require, response.receipt, response.checked, response.verify
resource, DESIGN, save = base.resource, base.DESIGN, base.save


def prepare(fit_manifest_path, output):
    ref = receipt(fit_manifest_path); manifest = response.audit(fit_manifest_path)
    require(not Path(manifest['expected_continuation_report_path']).exists()
        and not (Path(ref['path']).parent / 'phase12-fit-claim.json').exists(),
        'Recovery resource design must precede recovery final and numerical fit')
    sources = strict_references([ref, manifest['sources'], implementation_sources(__file__)])
    for source in sources: verify(source)
    doc = dict(schema=SCHEMA + '.design', status='CPU_PREDECLARED_RECOVERY_RESOURCE_NO_FIT',
        fit_manifest=ref, design=copy.deepcopy(DESIGN), sources=sources,
        resource_fit_run=False, native_training_resource_labels_read=False,
        validation_labels_read=False, GPU_started=False)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    return save(output / 'design.json', doc)


def audit(path):
    doc = checked(receipt(path))
    require(doc.get('schema') == SCHEMA + '.design'
        and doc.get('status') == 'CPU_PREDECLARED_RECOVERY_RESOURCE_NO_FIT'
        and doc.get('design') == DESIGN and doc.get('resource_fit_run') is False
        and doc.get('native_training_resource_labels_read') is False
        and doc.get('validation_labels_read') is False and doc.get('GPU_started') is False,
        'Exact explicit recovery resource design required')
    verify(doc['fit_manifest']); manifest = response.audit(doc['fit_manifest']['path'])
    expected = strict_references([doc['fit_manifest'], manifest['sources'], implementation_sources(__file__)])
    require(doc['sources'] == expected, 'Recovery resource source closure changed')
    for ref in expected: verify(ref)
    return doc


def build_from_bundle(bundle, document, manifest):
    """Authenticated recovery callback; new intake/raw identities are retained."""
    response.check_intake(document, manifest); response._validate_bundle(bundle)
    require(bundle.get('producer_schema') == response.SCHEMA + '.rows'
        and bundle.get('upstream_intake_schema') == response.intake.SCHEMA,
        'Explicit recovery row producer required')
    implementation_start = implementation_sources(__file__)
    sources = strict_references([document['sources'], bundle['review_receipts'], implementation_start])
    for ref in sources: verify(ref)
    plans = {p['attempt_id']: p for p in bundle['plans']}
    docs = {d['attempt_id']: d for d in bundle['documents']}
    require(len(plans) == len(docs) == 12 and set(plans) == set(docs), 'Exact twelve recovery resource attempts required')
    reviews = {}
    for ref in bundle['review_receipts']:
        review = checked(ref); plan = checked(review['inputs']['plan.json'])
        aid = plan['attempt_id']
        require(aid not in reviews and plans.get(aid) == plan
            and review.get('schema') == response.intake.native_review.SCHEMA
            and review.get('source_rebound_manifest') == document['manifest']
            and review.get('issues') == [] and review.get('training_feature_extraction_eligible') is True,
            'Recovery resource review/plan identity differs')
        reviews[aid] = (ref, review)
    require(set(reviews) == set(plans), 'Every resource attempt needs an authenticated review')
    tables = []
    for aid, plan in plans.items():
        ref, review = reviews[aid]; raw = checked(review['raw']); doc = docs[aid]
        require(raw.get('schema') == response.executor.RAW_SCHEMA and raw.get('backend') == 'native'
            and raw.get('status') == 'completed' and raw.get('split') == 'training'
            and raw.get('attempt_id') == aid and raw.get('execution_id') == plan['execution_id']
            and raw.get('capture_id') == plan['capture_id'] and checked(raw['plan']) == plan
            and raw['context'] == plan['context'] and doc.get('schema') == response.adapter.MACRO_SCHEMA
            and doc.get('source_rebound_manifest') == document['manifest']
            and doc.get('execution_id') == plan['execution_id'], 'Original measured raw/macro identity changed')
        table = resource.table_from_verified(raw, doc, plan, checked(raw['context']))
        table.update(raw=review['raw'], context=raw['context'], plan=raw['plan'],
            training_review=ref, capture_sha256=review['raw']['sha256'])
        tables.append(table)
        sources = strict_references([sources, review['raw'], raw['context'], raw['plan'], review['sources'], doc['sources']])
    require(sum(t['measured_requests'] for t in tables) == 546 and sum(t['priming_requests'] for t in tables) == 216,
            'All measured requests and history-only priming must remain')
    full, folds = base._fits(tables)
    require(implementation_sources(__file__) == implementation_start, 'Recovery resource implementation changed')
    for ref in sources: verify(ref)
    return dict(resource_model=full, whole_block_leaveout=folds, sources=sources,
        denominator=copy.deepcopy(bundle['denominator']), measured_requests=546, history_only_priming_requests=216,
        training_attempts=12, resource_fit_run=True, native_training_resource_labels_read=True,
        response_numerical_fit_run=False, validation_labels_read=False, GPU_started=False,
        resource_prediction_qualified=False, main_queue_provider_eligible=False, final_gain_eligible=False)


def run(design_path, output, *, fit_enabled=False):
    ref = receipt(design_path); design = audit(design_path)
    manifest = response.audit(design['fit_manifest']['path'])
    final = Path(manifest['expected_continuation_report_path'])
    result = dict(schema=SCHEMA, design=ref, status='WAITING_FOR_RECOVERY_EXACT12_FINAL',
        resource_fit_run=False, native_training_resource_labels_read=False,
        validation_labels_read=False, GPU_started=False, resource_prediction_qualified=False,
        main_queue_provider_eligible=False, final_gain_eligible=False,
        sources=strict_references([ref, design['sources']]))
    if final.is_file():
        final_ref = receipt(final); result['continuation'] = final_ref
        if not response.complete_final(manifest, checked(final_ref)):
            result['status'] = 'HOLD_INCOMPLETE_OR_FOREIGN_RECOVERY_FINAL'
        elif not fit_enabled:
            result['status'] = 'READY_EXPLICIT_RECOVERY_RESOURCE_FIT_REQUIRED'
        else:
            def callback(bundle, document):
                fitted = build_from_bundle(bundle, document, manifest)
                fitted['intake_evidence'] = document
                return fitted
            result.update(response.with_verified_bundle(design['fit_manifest']['path'], final, callback))
            result['status'] = COMPLETE_STATUS
        result['sources'] = strict_references([result['sources'], ref, design['sources'], final_ref])
    audit(design_path); verify(ref)
    for source in result['sources']: verify(source)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    return save(output / 'model.json', result)


class FoldResourceTrainingProvider(resource.ResourceTrainingProvider):
    def __init__(self, model_path, target_plan, target_context, *, heldout_block_id,
                 allow_training_resource_hypothesis=False, scenario='mean'):
        require(allow_training_resource_hypothesis is True and scenario in ('mean', 'max'),
                'Explicit training-only resource scenario required')
        model_ref = receipt(model_path); doc = checked(model_ref)
        require(doc.get('schema') == SCHEMA and doc.get('status') == COMPLETE_STATUS
            and doc.get('resource_fit_run') is True and doc.get('validation_labels_read') is False
            and doc.get('GPU_started') is False, 'Complete recovery resource model required')
        for ref in doc['sources']: verify(ref)
        verify(doc['design']); design = audit(doc['design']['path'])
        manifest = response.audit(design['fit_manifest']['path']); verify(doc['continuation'])
        require(doc['continuation']['path'] == manifest['expected_continuation_report_path'], 'Recovery final changed')
        def rederive(bundle, document):
            require(document == doc['intake_evidence'], 'Recovery resource intake changed')
            return build_from_bundle(bundle, document, manifest)
        actual = response.with_verified_bundle(design['fit_manifest']['path'], doc['continuation']['path'], rederive)
        require(all(doc.get(k) == v for k, v in actual.items() if k != 'sources'),
                'Recovery full resource model and every fold must rederive')
        expected = strict_references([actual['sources'], doc['design'], design['sources'], doc['continuation']])
        require(doc['sources'] == expected, 'Recovery resource model source closure differs')
        selected = next((f for f in actual['whole_block_leaveout'] if f['heldout_block_id'] == heldout_block_id), None)
        require(selected is not None and target_plan.get('capture_id') == heldout_block_id
            and target_plan.get('attempt_id') in selected['heldout_attempt_ids'], 'Exact heldout target required')
        identity = resource.physical_execution_identity(target_plan, target_context)
        require(identity == selected['training_model']['physical_execution_identity'], 'Target physical axes differ')
        self.document = copy.deepcopy(selected['training_model']); self.scenario = scenario
        self.target_plan, self.target_context = copy.deepcopy(target_plan), copy.deepcopy(target_context)
        self.physical_execution_identity = identity
        self.fold_binding = dict(schema=SCHEMA + '.fold_provider', full_native_model=model_ref,
            selected_fold_model_sha256=resource.digest(self.document), heldout_block_id=heldout_block_id,
            training_block_ids=self.document['training_block_ids'], training_attempt_ids=self.document['training_attempt_ids'],
            heldout_attempt_ids=selected['heldout_attempt_ids'], all_three_folds_rederived=True,
            complete_twelve_native_attempt_gate=True, startup_recomputed_from_only_eight_donor_attempts=True,
            sources=strict_references([doc['sources'], model_ref, implementation_sources(__file__)]),
            resource_prediction_qualified=False, main_queue_provider_eligible=False, final_gain_eligible=False,
            validation_labels_read=False, GPU_started=False)
        for ref in self.fold_binding['sources']: verify(ref)


def predict_timeline(timeline, provider, *, target_context, execution_binding):
    require(isinstance(provider, FoldResourceTrainingProvider), 'Explicit recovery fold provider required')
    value = resource.predict_timeline(timeline, provider, target_context=target_context, execution_binding=execution_binding)
    value['resource_fold_binding'] = copy.deepcopy(provider.fold_binding)
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--fit-manifest', required=True); a.add_argument('--output', required=True)
    a = sub.add_parser('audit'); a.add_argument('--design', required=True)
    a = sub.add_parser('run'); a.add_argument('--design', required=True); a.add_argument('--output', required=True); a.add_argument('--fit', action='store_true')
    a = p.parse_args()
    print(json.dumps(prepare(a.fit_manifest,a.output) if a.command=='prepare' else audit(a.design)
        if a.command=='audit' else run(a.design,a.output,fit_enabled=a.fit)))
