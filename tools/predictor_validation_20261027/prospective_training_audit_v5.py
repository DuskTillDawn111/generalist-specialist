"""Check that prospective exclusions cover every model's real training lineage.

Reads frozen training receipts and public input identities, never validation
outcomes. A draft remains a draft after this audit: no prediction seal or GPU
validation authorization is produced here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gsdeploy.forecasting.prediction_gate import DATA_SCHEMA
from .historical_evidence_v4 import HistoricalReceiptResolver, resolve_closure
from .native_collect import receipt, save
from .prospective_validation_prepare_v3 import declared_training_inputs

STRICT_SCHEMAS = {'gsdeploy.training_collection_strict_review.v3',
                  'gsdeploy.startup_training_collection_strict_review.v4'}
SCHEMA = 'gsdeploy.prospective_training_lineage_audit.v5'


def record_key(row):
    return tuple(row[k] for k in ('run_id', 'request_id', 'pool_id', 'task'))


def check_documents(index, public, reviews, model_indices):
    """Pure check over already source-verified training/public documents."""
    if index.get('schema') != DATA_SCHEMA or index.get('split') != 'training':
        raise ValueError('Explicit complete training index required')
    indexed = [record_key(r) for r in index['records']]
    if not indexed or len(indexed) != len(set(indexed)):
        raise ValueError('Nonempty duplicate-free training index required')
    indexed = set(indexed)
    validation_ids = {r['pool_id'] for r in public['inputs']}
    validation_hashes = {r['exact_input_sha256'] for r in public['inputs']}
    if len(validation_ids) != len(public['inputs']) or len(validation_hashes) != len(public['inputs']):
        raise ValueError('Unique prospective input identities and contents required')
    required, inputs, capture_ids = set(), {}, set()
    for report, plan, raw in reviews:
        if (report.get('schema') not in STRICT_SCHEMAS or report.get('status') != 'PASS'
                or report.get('issues') or not report.get('training_only')
                or plan.get('split') != 'training' or raw.get('status') != 'completed'):
            raise ValueError('Complete strict PASS training evidence required')
        run = 'raw-sha256:' + report['raw']['sha256']
        if report.get('capture_run_id') != run:
            raise ValueError('Actual raw training capture identity required')
        capture_ids.add(run)
        declared_ids = set()
        for item in plan['inputs']:
            if item['pool_id'] in declared_ids:
                raise ValueError('Duplicate declared training input identity')
            declared_ids.add(item['pool_id'])
        declared = declared_training_inputs(plan)
        for item in declared.values():
            old = inputs.setdefault(item['pool_id'], item)
            if old != item:
                raise ValueError('Conflicting declared training input content')
            if item['pool_id'] in validation_ids or item['exact_input_sha256'] in validation_hashes:
                raise ValueError('Prospective public input overlaps model training content')
        ids = set()
        for row in raw['request_records']:
            if (row.get('split') != 'training' or row.get('status') != 'completed'
                    or row['request_id'] in ids or row['pool_id'] not in declared
                    or row['task'] != declared[row['pool_id']]['task']):
                raise ValueError('Complete nonduplicate declared training request required')
            ids.add(row['request_id'])
            required.add(record_key(dict(row, run_id=run)))
        for row in raw.get('priming_request_records', []):
            if (row.get('status') != 'completed' or row['pool_id'] not in declared
                    or row['task'] != declared[row['pool_id']]['task']):
                raise ValueError('Actual priming must use completed declared training inputs')
    if not required or required != indexed:
        raise ValueError('Prospective index omits real model training captures/requests')
    for model_index in model_indices:
        if model_index.get('schema') != DATA_SCHEMA or model_index.get('split') != 'training':
            raise ValueError('Explicit nested model training index required')
        keys = [record_key(r) for r in model_index['records']]
        if len(keys) != len(set(keys)) or not set(keys) <= indexed or not set(keys) <= required:
            raise ValueError('Prospective index omits or duplicates nested model training requests')
    return dict(training_captures=len(capture_ids), required_unique_training_requests=len(required),
                indexed_training_requests=len(indexed), declared_training_inputs=len(inputs),
                prospective_public_inputs=len(public['inputs']), nested_training_indices=len(model_indices))


def audit(design_path, model_paths, output, *, historical_resolution_path=None):
    design_ref = receipt(design_path)
    design = json.loads(Path(design_path).read_text())
    if design.get('status') != 'DESIGN_DRAFT_NOT_FROZEN' or design.get('frozen_protocol_created') is not False:
        raise ValueError('This audit expects an explicitly unfrozen prospective draft')
    resolver = HistoricalReceiptResolver(historical_resolution_path)
    def read_ref(ref):
        return json.loads(Path(resolver.resolve(ref)['path']).read_text())
    # Caller-supplied extras can add a not-yet-wired model, never remove any
    # actual candidate bundle or its transitive model artifacts from the audit.
    model_roots = list(dict.fromkeys([*design['model_paths'].values(), *model_paths]))
    roots = model_roots + design['source_configuration']['training_reviews']
    closure = resolve_closure(roots, historical_resolution_path)
    reviews, indices, review_refs, nonpassing_refs, nonpassing_raws = [], [], [], [], []
    declared_reviews = {receipt(p)['sha256'] for p in design['source_configuration']['training_reviews']}
    for ref in closure['materialized_sources']:
        if Path(ref['path']).suffix != '.json':
            continue
        doc = read_ref(ref)
        if not isinstance(doc, dict):
            continue
        if doc.get('schema') in STRICT_SCHEMAS:
            if doc.get('status') != 'PASS' or doc.get('issues') or not doc.get('training_only'):
                if ref['sha256'] in declared_reviews:
                    raise ValueError('A declared training capture lacks strict PASS')
                # Corrected reviews can retain a historical failed review as
                # evidence. Keeping it is not using that failed review to fit.
                nonpassing_refs.append(ref)
                nonpassing_raws.append(doc.get('raw'))
                continue
            reviews.append((doc, read_ref(doc['plan']), read_ref(doc['raw'])))
            review_refs.append(ref)
        elif doc.get('schema') == DATA_SCHEMA:
            indices.append(doc)
    passing_raws = {report['raw']['sha256'] for report, _, _ in reviews}
    if any(not isinstance(ref, dict) or ref.get('sha256') not in passing_raws for ref in nonpassing_raws):
        raise ValueError('A retained failed training review requires a strict PASS correction of the identical raw')
    if len(design['training_data_paths']) != 1:
        raise ValueError('One explicit union training index required')
    index_ref = receipt(design['training_data_paths'][0])
    result = check_documents(read_ref(index_ref), read_ref(design['public_domain']), reviews, indices)
    result.update(schema=SCHEMA, status='TRAINING_LINEAGE_AND_PUBLIC_EXCLUSIONS_PASS',
                  design=design_ref, training_index=index_ref, public_domain=design['public_domain'],
                  model_roots=[receipt(p) for p in model_roots], training_reviews=review_refs,
                  retained_nonpassing_historical_reviews=nonpassing_refs,
                  evidence_closure=closure, evaluator=receipt(__file__),
                  validation_outcomes_read=False, predictions_sealed=False,
                  physical_accuracy_validated=False, GPU_started=False)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', required=True)
    parser.add_argument('--model', action='append', required=True)
    parser.add_argument('--historical-resolution')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.design, args.model, args.output, historical_resolution_path=args.historical_resolution)
    print(json.dumps({k: result[k] for k in ('status', 'training_captures', 'required_unique_training_requests',
                                          'indexed_training_requests', 'declared_training_inputs')}))


if __name__ == '__main__':
    main()
