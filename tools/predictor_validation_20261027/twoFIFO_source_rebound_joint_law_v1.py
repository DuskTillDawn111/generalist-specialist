"""Compile authenticated rebound macros into the existing mathematical law.

New source documents retain their schema and identities. Only request-vector
and protocol algebra is reused; the old native intake is never impersonated.
This pure compiler requires its caller to authenticate the exact12 bundle.
"""
from __future__ import annotations

import hashlib
import json

from . import twoFIFO_joint_request_law_v4 as joint
from . import twoFIFO_source_rebound_phase12_adapter_v1 as adapter
from . import twoFIFO_source_rebound_intake_v1 as intake

require = adapter.require


def complete_documents(documents, plans, raws):
    require(len(documents) == len(plans) == len(raws) == 12,
            'Exact twelve rebound attempts required')
    intake._attempt_layout(plans)
    ids = set()
    raw_hashes = set()
    macro_count = 0
    for doc, plan, raw in zip(documents, plans, raws):
        intake._macro_identity(doc, plan, doc.get('source_rebound_manifest'))
        require(doc.get('source_rebound_manifest') == documents[0].get('source_rebound_manifest')
                and type(doc.get('source_rebound_manifest')) is dict
                and set(doc['source_rebound_manifest']) == {'path', 'sha256'},
                'One explicit rebound manifest binding required')
        require(all(doc.get(k) == plan[k] == raw.get(k)
                    for k in ('capture_id', 'attempt_id', 'execution_id')),
                'Raw, macro and plan attempt binding changed')
        require(raw.get('backend') == 'native' and raw.get('status') == 'completed'
                and len(raw.get('priming_request_records', [])) == 18,
                'Complete native raw with actual priming history required')
        arrivals = [a for c in plan['cases'] for a in c['arrivals']]
        require([r['request_id'] for r in raw['request_records']] ==
                [r['request_id'] for r in doc['requests']], 'Native complete cohort order changed')
        hashes = set()
        for request, arrival in zip(doc['requests'], arrivals):
            require(request['request_id'] not in ids, 'Duplicate offered request across attempts')
            ids.add(request['request_id'])
            require(request['owner'] == arrival['owner']
                    and all(request.get(k) == plan[k] for k in ('capture_id', 'attempt_id', 'execution_id')),
                    'Request source identity changed')
            require(request['macros'] and all(
                m['request_id'] == arrival['request_id'] and m['owner'] == arrival['owner']
                and m['pool_id'] == arrival['pool_id'] and m['task'] == arrival['task']
                and m['training_replication_unit'] == plan['capture_id']
                and m['attempt_id'] == plan['attempt_id'] for m in request['macros']),
                'Exact public input and macro identity required')
            hashes.update(m['capture_sha256'] for m in request['macros'])
            macro_count += len(request['macros'])
        require(len(hashes) == 1 and not hashes.intersection(raw_hashes),
                'Twelve distinct raw provenance receipts required')
        raw_hashes.update(hashes)
        require(doc.get('macros') == sum(len(r['macros']) for r in doc['requests']),
                'Every macro must remain in its document')
    # Compile/check complete new-schema paths, including endpoints, continuity
    # and all processed-prefix covariates, before request-vector construction.
    rows, denominator = adapter._rows(documents, plans)
    require(len(rows) == macro_count and len(ids) == 546,
            '546 measured requests required; macro count is data dependent')
    return denominator


def build_joint_law(documents, plans, raws, qualified_fit, *, heldout_block_id=None):
    denominator = complete_documents(documents, plans, raws)
    blocks_all = sorted({p['capture_id'] for p in plans})
    require(qualified_fit.get('response_law_compilation_eligible') is True
            and qualified_fit.get('features') == list(adapter.FEATURES)
            and qualified_fit.get('fitted_root_numerical_qualification') is True
            and qualified_fit.get('denominator') == denominator,
            'Qualified complete rebound phase12 response required')
    fit = qualified_fit['full_fit']
    require(fit.get('numerical_qualification') is True and set(fit['whole_blocks']) == set(blocks_all),
            'Full root must retain all three blocks')
    folds = qualified_fit['whole_block_leaveout']
    require(len(folds) == 3 and {f['heldout_block_id'] for f in folds} == set(blocks_all),
            'All three whole-block fits required')
    for fold in folds:
        heldout = fold['heldout_block_id']
        require(fold['training_fit'].get('numerical_qualification') is True
                and fold['evaluation']['summary']['unknown_rows'] == 0
                and set(fold['training_fit']['whole_blocks']) == set(blocks_all) - {heldout}
                and set(fold['training_attempt_ids']) == {p['attempt_id'] for p in plans if p['capture_id'] != heldout}
                and set(fold['heldout_attempt_ids']) == {p['attempt_id'] for p in plans if p['capture_id'] == heldout},
                'Fold root/support and complete 8/4 attempt separation required')
    require(heldout_block_id is None or heldout_block_id in blocks_all, 'Unplanned heldout block')
    blocks = [b for b in blocks_all if b != heldout_block_id]
    if heldout_block_id is not None:
        fit = next(f['training_fit'] for f in folds if f['heldout_block_id'] == heldout_block_id)
    vectors, provenance, decompositions = [], [], []
    for doc, plan, raw in zip(documents, plans, raws):
        if plan['capture_id'] == heldout_block_id:
            continue
        messages = joint.source_messages(raw)
        derived = joint.derive(messages)
        require(len(messages) == len(derived), 'Protocol message lost in algebra')
        indexed = {tuple(m['identity']): m['protocol_slacks'] for m in derived}
        require(len(indexed) == len(messages), 'Duplicate native protocol identity')
        prior, constraints = {}, {}
        priming_ids = {r['request_id'] for r in raw['priming_request_records']}
        for row in joint.public_source_order(raw, plan):
            if row['request_id'] not in priming_ids:
                constraints[row['request_id']] = joint.transport_constraints(row, prior.get(row['model']))
            prior[row['model']] = row
        by_id = {r['request_id']: r for r in raw['request_records']}
        for request in doc['requests']:
            row = by_id[request['request_id']]
            slacks = {identity[2]: values for identity, values in indexed.items()
                      if identity[:2] == (row['model'], row['request_id'])}
            vector = joint.request_vector(request, row, fit, slacks, constraints[row['request_id']])
            vector['training_block_id'] = plan['capture_id']
            vector['vector_id'] = hashlib.sha256(json.dumps(
                {k: v for k, v in vector.items() if k != 'vector_id'}, sort_keys=True).encode()).hexdigest()
            vectors.append(vector)
            provenance.append(dict(vector_id=vector['vector_id'], block_id=plan['capture_id'],
                attempt_id=plan['attempt_id'], execution_id=plan['execution_id'],
                request_id=row['request_id'], case_id=row['profile_job_id'],
                raw_sha256=request['macros'][0]['capture_sha256']))
        decompositions.append(dict(attempt_id=plan['attempt_id'], block_id=plan['capture_id'],
            messages=len(messages), priming_messages=sum(len(r['execution_events']) + 2 for r in raw['priming_request_records']),
            priming_history_requests=18, all_observed_clocks_reconstructed=True,
            stdout_producer_scope='one_persistent_producer_per_owner_per_attempt'))
    return dict(schema=joint.SCHEMA,
        status='COMPLETE_REQUEST_WORK_PROTOCOL_VECTORS_PAIRED_TRANSPORT_PARTIALLY_IDENTIFIED',
        source_macro_schema=adapter.MACRO_SCHEMA, source_rebound_manifest=documents[0]['source_rebound_manifest'],
        source_documents_relabelled=False, features=list(adapter.FEATURES), coefficients=fit['coefficients'],
        vectors=vectors, provenance_not_used_for_lookup=provenance, whole_attempt_protocol_decompositions=decompositions,
        blocks=blocks, block_weights=[1/len(blocks)]*len(blocks), heldout_block_id=heldout_block_id,
        measured_request_vectors=len(vectors), authenticated_campaign_startup_attempts=12,
        donor_startup_attempts=len(decompositions), training_replication_units=len(blocks),
        public_lookup_key='owner_task_exact_input_and_public_key_arrival_ordinal',
        common_block_latent_across_inputs=True, complete_request_vector_drawn_jointly=True,
        conditional_request_draws_independent_given_block_assumption=True,
        per_macro_or_lag_independent_sampling=False, source_raw_ids_are_provenance_only=True,
        protocol_slacks_are_measured_CPU_service=False, startup_control_unobserved_collector_work_remains_limitation=True,
        observed_submit_to_START_is_worker_demand=False,
        busy_transport_law='PAIRED_FEASIBLE_SET_NOT_LATENT_PROBABILITY_LAW',
        prospective_template_transfer_qualified=False, main_queue_provider_eligible=False,
        final_gain_eligible=False, validation_labels_read=False, GPU_started=False)
