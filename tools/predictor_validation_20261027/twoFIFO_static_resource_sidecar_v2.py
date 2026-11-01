"""Native v3 whole-attempt nominal structure sidecar, CPU-only diagnostics.

Reuse source-bound seven-projection features and the existing causal pair-state
helper. Whole-attempt priming informs visible history but never creates labels.
Attention/KV, measured DRAM/L2 and physical resource classification stay UNKNOWN.
"""
import argparse
import json
from pathlib import Path

from . import twoFIFO_composable_macros_v3 as macros_v3
from .twoFIFO_owned_training_review_v3 import attempt_events, SCHEMA as REVIEW_SCHEMA, RAW_SCHEMA, PLAN_SCHEMA
from .static_language_work_v1 import features_from_visible_begin
from .macro_projection_contrast_training_v1 import checked_static_inventory, descriptor, pair_state
from .stage_response_features_v1 import checked, receipt
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA = 'gsdeploy.twoFIFO_static_resource_structure_sidecar.v2'
MACRO_SCHEMA = 'gsdeploy.twoFIFO_composable_response_macro_training.v3'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unknown_resources():
    return dict(attention_kv_length=None, attention_matmul_flops=None, attention_kv_cache_bytes=None,
        measured_dram_bytes=None, measured_l2_bytes=None, hardware_cache_hit_rate=None,
        sm_service_ns=None, achieved_compute_throughput=None, achieved_dram_throughput=None,
        runtime_quantized_weight_storage_bytes=None, runtime_quantization_scales=None,
        additional_gpu_memory_bytes=None, physical_classification='unknown',
        cache_reuse_gain=None)


def enrich(macros, events, bindings, owner_families, *, capture_sha256):
    """Pure diagnostic transformation, not native source authentication.

    Caller supplies whole-attempt processed events. Features are retrieved only
    for the exact macro BEGIN and only at/after its processing clock. No macro
    forward_shape, future END, output length or completed exposure fills a hole.
    Binding/source/native verification belongs to extract(), not this helper.
    """
    require(macros.get('schema') == MACRO_SCHEMA and macros.get('mode') == 'natural'
        and macros.get('status') == 'COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER'
        and macros.get('validation_labels_read') is False
        and macros.get('whole_attempt_history_used_for_visible_prefix') is True
        and macros.get('case_history_reset') is False, 'Exact whole-attempt training macro schema required')
    require(len(macros['requests']) == macros['measured_requests'], 'Full measured request cohort required')
    begins = {}; unsupported = {}; available = {}; seen = set()
    for event in events:
        if event.get('event') != 'start' or event.get('semantic_phase') not in ('prefill', 'decode'):
            continue
        key = event['owner'], event['request_id'], event['segment_id']
        require(key not in seen, 'Distinct whole-attempt BEGIN identity required')
        seen.add(key)
        require(type(event.get('collector_processing_time_ns')) is int
            and type(event.get('perf_ns')) is int
            and event['perf_ns'] <= event['collector_processing_time_ns'],
            'Actual BEGIN occurrence and parent processing clocks required')
        available[key] = event['collector_processing_time_ns']
        family = owner_families.get(event['owner'])
        if family not in bindings:
            unsupported[key] = 'OWNER_STATIC_PROJECTION_BINDING_UNKNOWN'; continue
        # Missing shape stays UNKNOWN; malformed supplied dimensions still fail
        # in the existing feature API rather than being repaired or guessed.
        if event.get('input_batch_size') is None or event.get('input_sequence_length') is None:
            unsupported[key] = 'ACTUAL_BEGIN_SHAPE_MISSING'; continue
        feature = features_from_visible_begin(bindings[family], event,
            parent_now_ns=event['collector_processing_time_ns'])
        begin = descriptor(feature, event, capture_sha256)
        begin['nominal_projection_features'] = feature
        begins[key] = begin
    records = []; request_ids = set(); visible_ns = known_ns = unknown_ns = 0
    for request in macros['requests']:
        identity = request['owner'], request['request_id']
        require(identity not in request_ids, 'Distinct measured request identity required')
        request_ids.add(identity)
        for macro in request['macros']:
            require((macro['owner'], macro['request_id']) == identity
                and macro['capture_sha256'] == capture_sha256, 'Exact macro/request/capture identity required')
            path = macro['causal_covariate_path']
            require(path and path[0]['left_ns'] == macro['left_ns'] and path[-1]['right_ns'] == macro['right_ns']
                and all(a['right_ns'] == b['left_ns'] for a,b in zip(path,path[1:])),
                'Complete contiguous macro exposure required')
            enriched = []
            key = macro['owner'], macro['request_id'], macro['macro_id']
            for piece in path:
                duration = piece['right_ns']-piece['left_ns']
                require(type(duration) is int and duration > 0 and piece['duration_ns'] == duration
                    and piece['state']['cutoff_processing_ns'] == piece['left_ns']
                    and piece['each_state_uses_only_its_processed_prefix'] is True,
                    'Exact positive processed-prefix macro piece required')
                peers=piece['state']['peer_macros']
                require(len({peer['owner'] for peer in peers})==len(peers)
                    and all(peer['owner']!=macro['owner'] and type(peer.get('available_ns')) is int
                        and peer['available_ns']<=piece['left_ns'] for peer in peers),
                    'Every peer must be distinct and processed by the current cutoff')
                state = pair_state(macro, piece, begins)
                if state['own_supported']:
                    reasons = []
                elif macro['kind'] != 'language_forward_macro':
                    reasons = ['OUTSIDE_DECLARED_LANGUAGE_PROJECTIONS']
                elif key in available and available[key] > piece['left_ns']:
                    reasons = ['CURRENT_MACRO_BEGIN_NOT_YET_PROCESSED']
                elif key in unsupported:
                    reasons = [unsupported[key]]
                else:
                    reasons = ['CURRENT_MACRO_BEGIN_NOT_OBSERVED']
                state.update(own_unknown_reasons=reasons, unknown_physical_resources=unknown_resources(),
                    nominal_projection_bytes_are_not_measured_l2_or_dram_traffic=True,
                    phase_visible_proxy_is_not_actual_resource_activity=True)
                # Existing pair_state retains all N peers, including unsupported
                # OCR/entry/tails; K+U=N and all-peer aggregates are None if U>0.
                visible_ns += duration*state['peer_denominator_N']
                known_ns += duration*state['known_pair_denominator_K']
                unknown_ns += duration*state['unknown_pair_denominator_U']
                enriched.append(dict(left_ns=piece['left_ns'], right_ns=piece['right_ns'],
                    duration_ns=duration, cutoff_processing_ns=piece['left_ns'], projection_state=state))
            require(sum(p['duration_ns'] for p in enriched) == macro['response_wall_ns'],
                'No macro response interval may be dropped')
            records.append(dict(owner=macro['owner'], request_id=macro['request_id'],
                public_input=macro['pool_id'], task=macro['task'], macro_id=macro['macro_id'],
                kind=macro['kind'], semantic_phase=macro['semantic_phase'],
                capture_sha256=capture_sha256, response_wall_ns=macro['response_wall_ns'], path=enriched))
    require(len(records) == macros['macros'] and known_ns+unknown_ns == visible_ns,
        'Complete macro and peer denominators required')
    priming_history=[dict(event) for event in events if event.get('job_id')=='__priming__']
    return dict(schema=SCHEMA, status='WHOLE_ATTEMPT_NOMINAL_PROJECTION_SIDECAR_ONLY',
        capture_id=macros['capture_id'], attempt_id=macros['attempt_id'], execution_id=macros['execution_id'],
        leaveout_unit='whole_capture_block_all_shards', requests=macros['measured_requests'],
        macros=len(records), records=records, history_event_count=len(events),
        priming_rows_excluded_as_labels=macros['priming_rows_excluded_as_labels'],
        priming_history_included=True, case_history_reset=False,
        priming_processed_history=priming_history,priming_history_event_count=len(priming_history),
        actual_BEGIN_descriptor_catalog=list(begins.values()),
        unsupported_BEGIN_catalog=[dict(owner=key[0],request_id=key[1],phase_id=key[2],
            available_at_processing_ns=available[key],reason=reason) for key,reason in unsupported.items()],
        known_BEGIN_descriptors=len(begins), unsupported_BEGIN_descriptors=len(unsupported),
        visible_peer_exposure_ns=visible_ns, known_pair_exposure_ns=known_ns,
        unknown_pair_exposure_ns=unknown_ns,
        known_pair_exposure_fraction=known_ns/visible_ns if visible_ns else None,
        all_macro_rows_and_unknown_peers_retained=True, actual_BEGIN_processed_only=True,
        input_macro_schema=MACRO_SCHEMA,input_review_schema=REVIEW_SCHEMA,
        complementary_service_multiplier=None,complementarity_qualified=False,
        future_shape_backfilled=False, unknown_physical_resources=unknown_resources(),
        structural_basis='Nominal seven decoder projections from exact actual processed BEGIN shape; logical operand sizes, not measured traffic.',
        own_reference_wall_baseline_inferred=False, reference_rate_modified=False,
        native_current_visible_risk_features=None, native_source_authenticated=False,
        physical_compute_memory_classification_qualified=False, cache_gain_qualified=False,
        prospective_shape_template_qualified=False, main_queue_provider_eligible=False,
        independent_prediction_accuracy_validated=False, final_gain_eligible=False,
        fitted_model_qualified=False, validation_labels_read=False, GPU_started=False)


def extract(training_review, static_inventory):
    """Verify new native review/macros and existing static source/header binding.

    Uses actual v3 native extraction and review rederivation, preserving schema identities. This tool
    is downstream-only; the collector does not import it or change source identity.
    """
    macros = macros_v3.extract(training_review)
    review_ref = receipt(training_review); review = checked(review_ref)
    raw = checked(review['raw']); context = checked(raw['context'])
    plan = checked(raw['plan'])
    require(review.get('schema')==REVIEW_SCHEMA and raw.get('schema')==RAW_SCHEMA
        and plan.get('schema')==PLAN_SCHEMA and raw.get('backend')=='native'
        and all(macros.get(key)==plan[key]==raw[key] for key in ('attempt_id','capture_id','execution_id')),
        'Exact native v3 review/raw/plan/macro identities required')
    inventory, bindings, static_sources = checked_static_inventory(static_inventory)
    families = context['residents']
    for owner, family in families.items():
        if family in bindings:
            expected = bindings[family]['spec']['sha256']
            require(context['worker_bindings'][family]['spec_sha256'] == expected
                and raw['ready'][owner]['spec_sha256'] == expected,
                'Actual native worker and static topology spec differ')
    result = enrich(macros, attempt_events(raw), bindings, families,
        capture_sha256=review['raw']['sha256'])
    sources = strict_references([macros['sources'], static_sources,
        review_ref, raw['context'], implementation_sources(__file__)])
    require(all(receipt(ref['path']) == ref for ref in sources), 'Sidecar source closure changed')
    result.update(training_review=review_ref, static_inventory=receipt(static_inventory),
        input_macro_schema=MACRO_SCHEMA, sources=sources,
        source_closure_and_worker_spec_verified=True,
        native_training_review_rederived_by_v3_extractor=True,
        native_source_authenticated=False,
        source_verification_scope='Existing complete native training review and declared topology/header closure; no runtime loaded-operator or physical resource authentication added.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-review', required=True);parser.add_argument('--static-inventory', required=True)
    parser.add_argument('--output', required=True);args = parser.parse_args()
    result = extract(args.training_review, args.static_inventory)
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False);stream.write('\n')
    print(json.dumps(dict(schema=result['schema'], macros=result['macros'], output=args.output)))
