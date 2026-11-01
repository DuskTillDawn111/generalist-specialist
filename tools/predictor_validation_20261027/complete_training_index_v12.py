"""Build a conservative raw-identity training index; never freeze a protocol.

Only explicitly supplied, receipt-bound strict training captures enter the index.
Completed explicit priming rows are indexed too, because startup/availability
models can consume them. They do not become independent captures or validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .known_input_gate_v5 import DATA_SCHEMA
from .kernel_profile_campaign_v2 import input_identity
from .generation_work_v4 import input_content_identity

FIELDS = {'run_id', 'request_id', 'pool_id', 'task'}
STRICT = {
    'gsdeploy.training_collection_strict_review.v3',
    'gsdeploy.startup_training_collection_strict_review.v4',
    'gsdeploy.unified_corun_strict_review.v8_1',
    'gsdeploy.native_queue_training_strict_review.v8',
}


def receipt(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def checked(ref):
    if set(ref) != {'path', 'sha256'} or receipt(ref['path']) != ref:
        raise ValueError('Changed or malformed receipt: ' + str(ref))
    return json.loads(Path(ref['path']).read_text())


def records_for_capture(raw, plan, raw_reference):
    if (raw.get('kind') != 'measured' or raw.get('status') != 'completed'
            or raw.get('not_for_training') or raw.get('split') not in (None, 'training')
            or plan.get('split') != 'training' or plan.get('training_only') is not True
            or raw.get('execution_context_sha256') != plan.get('execution_context_sha256')
            or any(raw.get(k) for k in ('errors', 'runtime_errors', 'telemetry_errors'))):
        raise ValueError('Completed explicit training capture and matching plan required')
    rows = raw.get('request_records', [])
    primers = raw.get('priming_request_records', [])
    if not rows:
        raise ValueError('Nonempty measured training requests required')
    for row in rows:
        if (row.get('split') != 'training' or row.get('status') != 'completed'
                or row.get('not_for_training')):
            raise ValueError('Every measured row must explicitly be completed training')
    for row in primers:
        if (row.get('phase') != 'priming' or row.get('status') != 'completed'
                or row.get('split') not in (None, 'training') or row.get('not_for_training')):
            raise ValueError('Only explicit completed priming permits an absent split')
    run = 'raw-sha256:' + raw_reference['sha256']
    result = [dict(run_id=run, **{k:r[k] for k in ('request_id', 'pool_id', 'task')})
              for r in rows + primers]
    validate_index(dict(schema=DATA_SCHEMA, split='training', records=result))
    return result, len(rows), len(primers)


def validate_index(index):
    if set(index) != {'schema', 'split', 'records'} or index['schema'] != DATA_SCHEMA or index['split'] != 'training':
        raise ValueError('Exact training index v2 schema required')
    seen = set()
    for row in index['records']:
        if set(row) != FIELDS or any(not isinstance(v, str) or not v for v in row.values()):
            raise ValueError('Exactly four nonempty identity strings required')
        if (row['run_id'], row['request_id']) in seen:
            raise ValueError('Repeated/conflicting raw request identity')
        seen.add((row['run_id'], row['request_id']))
    if not seen:
        raise ValueError('Nonempty training index required')


def review_capture(reference):
    review = checked(reference)
    if (review.get('schema') not in STRICT or review.get('status') != 'PASS'
            or review.get('issues') or review.get('training_only') is not True):
        raise ValueError('Explicit strict PASS training review required: ' + reference['path'])
    evidence = review.get('evidence', review)
    def ref(name):
        return evidence.get(name, evidence.get(name + '.json'))
    raw_ref, plan_ref, context_ref = ref('raw'), ref('plan'), ref('context')
    plan, context = checked(plan_ref), checked(context_ref)
    # Qualify the plan before ever opening its outcome payload.
    if plan.get('split') != 'training' or plan.get('training_only') is not True:
        raise ValueError('Nontraining raw refused before opening')
    raw = checked(raw_ref)
    records, measured, primers = records_for_capture(raw, plan, raw_ref)
    if context.get('training_only') is not True:
        raise ValueError('Explicit training context required')
    return records, dict(run_id=records[0]['run_id'], raw=raw_ref, plan=plan_ref,
        context=context_ref, reviews=[reference], original_run_label=raw.get('run_id'),
        domain=plan.get('domain', 'native_queue'), measured_requests=measured,
        priming_requests=primers), plan


def merge_capture(registry, records, capture):
    run = capture['run_id']
    if run in registry:
        old_records, old_capture = registry[run]
        if old_records != records:
            raise ValueError('Same original raw SHA has conflicting request index')
        for review in capture['reviews']:
            if review not in old_capture['reviews']:
                old_capture['reviews'].append(review)
        return False
    registry[run] = records, capture
    return True


def public_disjointness(index, jobs, known_catalog, declared_inputs):
    validate_index(index)
    request_ids = {r['request_id'] for r in index['records']}
    run_ids = {r['run_id'] for r in index['records']}
    pools = {r['pool_id']:r['task'] for r in index['records']}
    audit = []
    registry = None
    for job in jobs:
        if job['run_id'] in run_ids:
            raise ValueError('Fresh validation run overlaps training')
        public = checked(job['public_inputs'])['forecast_inputs']
        rows = [r for queue in public['queues'].values() for r in queue]
        rows += public['future_workload']['declared_arrivals']
        identities = [{k:r[k] for k in ('request_id', 'pool_id', 'task')} for r in rows]
        if len({r['request_id'] for r in identities}) != len(identities):
            raise ValueError('Duplicate fresh public request')
        if request_ids & {r['request_id'] for r in identities}:
            raise ValueError('Fresh request identity overlaps any training run')
        for row in identities:
            pool = known_catalog.get(row['pool_id'])
            declaration = declared_inputs.get(row['pool_id'])
            if (pools.get(row['pool_id']) != row['task'] or pool is None
                    or pool['task_type'] != row['task'] or declaration is None
                    or declaration['task'] != row['task']
                    or declaration['exact_input_sha256'] != input_identity(pool)):
                raise ValueError('Fresh input not exactly trained and public')
        signature = sorted(identities, key=lambda r:r['request_id'])
        if registry is None:
            registry = signature
        elif registry != signature:
            raise ValueError('Candidate public cohorts differ')
        audit.append(dict(run_id=job['run_id'], candidate_id=job['candidate_id'],
            public_inputs=job['public_inputs'], requests=len(identities),
            disjoint_run=True, disjoint_request_ids=True, trained_public_content=True))
    return audit


def references(value):
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256'}:
            yield value
        else:
            for item in value.values():
                yield from references(item)
    elif isinstance(value, list):
        for item in value:
            yield from references(item)


def located_references(value, location=()):
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256'}:
            yield location, value
        else:
            for key, item in value.items():
                yield from located_references(item, location + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from located_references(item, location + (str(index),))


def lineage_audit(roots, captures):
    """Audit bounded metadata graph; do not open any outcome raw through it.

    Large derived arrays and implementation snapshots are explicit boundaries,
    not evidence that a universe of historical training has been reconstructed.
    """
    docs, edges, boundaries, raw_refs = {}, [], [], {}
    def visit(reference):
        path = reference['path']
        if path in docs:
            if docs[path]['reference'] != reference:
                raise ValueError('Conflicting model lineage receipt')
            return
        value = checked(reference)
        docs[path] = dict(reference=reference, schema=value.get('schema') if isinstance(value, dict) else None)
        for location, child in located_references(value):
            p = Path(child['path'])
            edges.append(dict(source=path, field_path=list(location), target=child))
            if p.name == 'raw.json':
                raw_refs.setdefault(child['sha256'], dict(reference=child, parents=[]))['parents'].append(
                    dict(path=path, field_path=list(location), schema=docs[path]['schema']))
                continue
            if p.suffix != '.json':
                continue
            tokens = ('validation', 'sealed', 'observed', 'source-snapshot', '_source-packet',
                'parsed-', 'trace-', 'operator-', 'executed-jobs', 'execution.json',
                'context.json', 'plan.json', 'intent.json', 'selected-input', 'tokenizer',
                'source-closure', 'source-before', 'source-after')
            if any(t in str(p).lower() for t in tokens):
                boundaries.append(dict(reference=child, reason='NONMODEL_OR_OUTCOME_READ_GUARD'))
            elif p.stat().st_size > 30_000_000:
                boundaries.append(dict(reference=child, reason='LARGE_DERIVED_ARRAY_NOT_EXPANDED'))
            else:
                visit(child)
    for root in roots:
        visit(root)
    indexed = {c['raw']['sha256'] for c in captures}
    unresolved = []
    for sha, item in raw_refs.items():
        if sha in indexed:
            item['classification'] = 'INDEXED_ORIGINAL_TRAINING_RAW'
        elif ('/smoke/' in item['reference']['path'] and all(
                'materialized_sources' in parent['field_path']
                or parent['schema'] == 'gsdeploy.kernel_smoke_strict_review.v3'
                for parent in item['parents'])):
            item['classification'] = 'INHERITED_RUNTIME_SMOKE_PROVENANCE_ONLY_NOT_INDEXED'
        else:
            item['classification'] = 'UNRESOLVED_RAW_REFERENCE_NOT_OPENED'
            unresolved.append(item)
    return dict(roots=roots, documents=list(docs.values()), edges=edges,
        raw_references=list(raw_refs.values()), unresolved_raw_references=unresolved,
        traversal_boundaries=boundaries, exhaustive_historical_noncontamination_proved=False,
        scope='Receipt metadata graph and supplied strict captures; numerical fitting lineage is audited separately from inherited runtime smoke provenance.')


def build(configuration, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = checked(configuration['old_protocol'])
    previous = [checked(ref) for ref in protocol['training_data']]
    for index in previous:
        validate_index(index)
    registry, declared = {}, {}
    for ref in configuration['training_reviews']:
        rows, capture, plan = review_capture(ref)
        merge_capture(registry, rows, capture)
        from .prospective_validation_prepare_v3 import declared_training_inputs
        if plan.get('inputs'):
            for pool, value in declared_training_inputs(plan).items():
                if pool in declared and declared[pool] != value:
                    raise ValueError('Training pool changed task/content')
                declared[pool] = value
    records = [row for run in sorted(registry) for row in registry[run][0]]
    index = dict(schema=DATA_SCHEMA, split='training', records=records)
    validate_index(index)
    new_rows = {tuple(sorted(r.items())) for r in records}
    for old in previous:
        if any(tuple(sorted(r.items())) not in new_rows for r in old['records']):
            raise ValueError('Old sealed training index is not an exact subset')
    catalog_ref = protocol['public_input_catalog']
    if receipt(catalog_ref['path']) != catalog_ref:
        raise ValueError('Changed immutable public catalog')
    catalog = {r['pool_id']:r for r in map(json.loads, Path(catalog_ref['path']).read_text().splitlines())}
    prepared_jobs = checked(configuration['fresh_jobs'])
    jobs = [dict(run_id=j['run_id'], candidate_id=j['candidate_id'],
        public_inputs=receipt(j['public_inputs_path'])) for j in prepared_jobs]
    public = public_disjointness(index, jobs, catalog, declared)
    captures = [registry[run][1] for run in sorted(registry)]
    lineage = lineage_audit(configuration['model_lineage_roots'], captures)
    content = [dict(pool_id=pool, task=catalog[pool]['task_type'],
        input_content_sha256=input_content_identity(catalog[pool]))
        for pool in sorted({r['pool_id'] for r in records}) if pool in catalog]
    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    save('training-index.json', index)
    save('source-manifest.json', dict(configuration=configuration, generator=receipt(__file__), captures=captures))
    save('lineage-audit.json', lineage)
    save('public-disjointness.json', dict(candidates=public, trained_catalog_contents=content,
        validation_raw_or_labels_read=False, protocol_frozen=False, GPU_started=False))
    report = dict(schema='gsdeploy.complete_training_index_audit.v12',
        status='INDEX_BUILT_PUBLIC_DISJOINT_WITH_EXPLICIT_LINEAGE_BOUNDARIES',
        original_training_records=sum(len(i['records']) for i in previous),
        original_runs=len({r['run_id'] for i in previous for r in i['records']}),
        original_indices_preserved_as_exact_subset=True,
        independent_original_raw_runs=len(captures), records=len(records),
        measured_records=sum(c['measured_requests'] for c in captures),
        priming_records=sum(c['priming_requests'] for c in captures),
        repeated_review_references=len(configuration['training_reviews'])-len(captures),
        tasks=dict(Counter(r['task'] for r in records)),
        unresolved_raw_lineage_count=len(lineage['unresolved_raw_references']),
        lineage_boundaries=len(lineage['traversal_boundaries']),
        public_candidates=len(public), public_requests_per_candidate=[c['requests'] for c in public],
        artifacts={name:receipt(output/name) for name in ('training-index.json', 'source-manifest.json',
            'lineage-audit.json', 'public-disjointness.json')},
        priming_policy='All completed explicit priming rows of qualified training captures are conservatively indexed, including absent priming-only split; phase counts remain separate and primers do not create independent runs.',
        strict_review_recomputed=False, strict_raw_plan_context_receipts_reverified=True,
        historical_noncontamination_proved=False, validation_raw_or_labels_read=False,
        protocol_frozen=False, predictions_sealed=False, GPU_started=False)
    save('report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = build(json.loads(Path(args.configuration).read_text()), args.output)
    print(json.dumps({k:report[k] for k in ('status', 'independent_original_raw_runs', 'records',
        'measured_records', 'priming_records', 'unresolved_raw_lineage_count', 'public_requests_per_candidate')}))


if __name__ == '__main__':
    main()
