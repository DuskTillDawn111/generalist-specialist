"""CPU propagation of distinct, explicitly hypothetical native-busy worlds.

This is a necessary-boundary feasibility diagnostic, not native assimilation or
a posterior. START notice transport and semantic PHASE/QUERY mapping remain
unidentified. Every predefined slot keeps its weight, including mismatches.
"""
from copy import deepcopy
from fractions import Fraction

from gsdeploy.forecast_errors import UnknownEvidence
from . import actual_worker_twoFIFO_reference_v5_2 as engine
from . import twoFIFO_forecast_ensemble_v1 as ensemble
from . import twoFIFO_native_busy_state_v1 as native
from .twoFIFO_settled_history_initial_v1 import digest
from .actual_worker_twoFIFO_query_adapter_v5_2 import clock
from .actual_worker_twoFIFO_reference_v5_2 import decode
from .macro_response_queue_reference_v1 import encode

SCHEMA = 'gsdeploy.twoFIFO_hypothetical_busy_candidate_continuation.v1'
INITIAL = {'initial_sender_available_ns', 'initial_worker_read_release_ns', 'initial_protocol_state'}
RESOURCES = ensemble.SHARED_FIELDS - INITIAL - {'continuation_state', 'settled_initial_state'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _screen(state, prefix, inputs):
    """Check only modeled necessary facts; never infer a hidden START notice."""
    public = {r['request_id']: r for r in state['binding']['public_cohort']}
    events = state['collector_processed_prefix']
    if any(e['request_id'] not in public for e in events) or set(state['sender_prefix']) - set(public):
        raise UnknownEvidence('UNMODELED_NATIVE_PRIOR_HISTORY')
    if any(e['kind'] not in native.BOUNDARIES for e in events):
        raise UnknownEvidence('UNMAPPED_NATIVE_PHASE_OR_QUERY')
    for row in public.values():
        supplied = inputs['plans'][(row['owner'], row['task'], row['pool_id'])]
        plans = supplied if type(supplied) is tuple else (supplied,)
        if any(p.ready_producer_id != state['binding']['owner_producers'][row['owner']] for p in plans):
            raise UnknownEvidence('NATIVE_CANDIDATE_PRODUCER_BINDING_MISMATCH')
    sender_keys = {'enqueued_time_ns': 'enqueued_ns', 'actual_submit_time_ns': 'parent_submit_ns',
                   'submit_write_completed_time_ns': 'parent_write_record_ns'}
    for rid, sender in state['sender_prefix'].items():
        for key, generated_key in sender_keys.items():
            if key in sender and prefix['transport_schedule'][rid][generated_key] != sender[key]:
                raise UnknownEvidence('CANDIDATE_SENDER_BOUNDARY_MISMATCH:' + rid + ':' + key)
    observed = {(e['request_id'], e['kind']): e for e in events}
    generated = {}
    for event in prefix['trace']:
        if event['kind'] == 'processed_notification' and event['action'] in ('result', 'ready'):
            kind = 'RESULT' if event['action'] == 'result' else 'WORKER_READY'
            generated[(event['request_id'], kind)] = dict(perf_ns=event['occurrence_ns'],
                received_time_ns=event['reader_receipt_ns'],
                collector_dequeued_time_ns=event['collector_dequeued_ns'],
                collector_processing_time_ns=event['available_ns'])
    # Absence is collector-processing survival, not physical END survival.
    for key in set(generated) | {k for k in observed if k[1] != 'START'}:
        if key not in generated or key not in observed or any(
                observed[key][k] != value for k, value in generated[key].items()):
            raise UnknownEvidence('CANDIDATE_PROCESSED_BOUNDARY_PATTERN_OR_CLOCK_MISMATCH:' + str(key))
    for (rid, kind), event in observed.items():
        if kind == 'START' and prefix['starts'].get(rid) != event['perf_ns']:
            raise UnknownEvidence('CANDIDATE_OBSERVED_START_OCCURRENCE_MISMATCH:' + rid)


def evaluate(state, draws, *, scoring_plan, quality, window_start_ns, window_end_ns,
             drain_end_ns, **resources):
    """Generate each complete candidate to the cut, then resume its own state.

    A candidate has the usual draw fields. Its inputs are either explicit
    UnknownForecastInputs or the four joint-law fields plus ``initial_state``.
    The latter is a complete hypothetical initial FIFO/protocol state or a
    source-bound settled_initial_state. Arbitrary busy checkpoints are rejected.
    No finite catalog is claimed to cover all native feasible residual states.
    """
    need(type(state) is dict and state.get('schema') == native.SCHEMA and
         state.get('state_sha256') == digest({k: v for k, v in state.items() if k != 'state_sha256'}),
         'Unmodified native prefix state required')
    need(set(resources) <= RESOURCES, 'Only common resources and producer FIFO may be shared')
    a, b, d = map(clock, (window_start_ns, window_end_ns, drain_end_ns))
    cut = state['cutoff_ns']
    need(a <= cut <= d and state['forecast_origin_ns'] == cut,
         'Cut must remain inside the original calendar; native forecast origin must equal cut')
    requests = state['binding']['public_cohort']
    need(type(draws) is list and draws and all(type(r) is dict and set(r) == ensemble.DRAW_FIELDS for r in draws),
         'Closed predefined draw catalog required')
    for draw in draws:
        value = draw['inputs']
        if type(value) is ensemble.UnknownForecastInputs:
            continue
        need(type(value) is dict and set(value) == ensemble.INPUT_FIELDS | {'initial_state'},
             'Each candidate requires complete joint inputs and its own hypothetical initial state')
        initial = value['initial_state']
        need(type(initial) is dict and (set(initial) == INITIAL or set(initial) == {'settled_initial_state'}),
             'Explicit original initial stations or settled history required; no arbitrary checkpoint')
    # Validate score, full calendar, exact mass and catalog before any candidate.
    placeholders = [dict(r, inputs=ensemble.UnknownForecastInputs('candidate not yet propagated')) for r in draws]
    common = dict(scoring_plan=scoring_plan, quality=quality, window_start_ns=window_start_ns,
                  window_end_ns=window_end_ns, drain_end_ns=drain_end_ns, **resources)
    out = decode(ensemble.evaluate(requests, placeholders, **common))
    rows = []
    for draw in draws:
        row = dict(draw_id=draw['draw_id'], block_id=draw['block_id'], weight=Fraction(draw['weight']),
                   offered_requests=len(requests), status='UNKNOWN_WORLD', forecast=None,
                   all_public_request_ids_retained=[r['request_id'] for r in requests])
        value = draw['inputs']
        if type(value) is ensemble.UnknownForecastInputs:
            row.update(reason=value.reason, evidence=deepcopy(value.evidence))
            rows.append(row)
            continue
        inputs = {k: value[k] for k in ensemble.INPUT_FIELDS}
        initial = value['initial_state']
        try:
            prefix = decode(engine.simulate(requests, **inputs, **initial,
                window_start_ns=window_start_ns, window_end_ns=window_end_ns,
                drain_end_ns=drain_end_ns, cutoff_ns=cut, producer_fifo=resources.get('producer_fifo', ())))
            _screen(state, prefix, inputs)
            singleton = dict(draw, weight=Fraction(1), inputs=inputs)
            result = decode(ensemble.evaluate(requests, [singleton], **common,
                                             continuation_state=prefix['continuation_state']))['draws'][0]
            row.update({k: v for k, v in result.items() if k not in ('weight', 'draw_id', 'block_id')})
            row['candidate_checkpoint_at_cut'] = prefix['continuation_state']
            row['boundary_screen_status'] = 'MODELED_NECESSARY_CONSTRAINTS_COMPATIBLE'
        except UnknownEvidence as exc:
            row.update(reason=str(exc), evidence=deepcopy(getattr(exc, 'evidence', None)))
        rows.append(row)
    out.update(schema=SCHEMA, status='HYPOTHETICAL_BUSY_CANDIDATES_NOT_NATIVE_POSTERIOR', draws=rows,
        native_prefix_state_sha256=state['state_sha256'], cutoff_ns=cut,
        world_unknown_probability_mass=sum((r['weight'] for r in rows if r['status'] == 'UNKNOWN_WORLD'), Fraction(0)),
        six_terms={t: ensemble.distribution(rows, lambda f, t=t: f['scoring']['original_score']['terms'][t]) for t in 'QTLPME'},
        utility=ensemble.distribution(rows, lambda f: f['scoring']['original_score']['full_U']),
        expected_world_complete_cohort_P95_ns=ensemble.distribution(rows, lambda f:
            None if f['queue_diagnostics']['reader_E2E_P95_ns'] is None else clock(f['queue_diagnostics']['reader_E2E_P95_ns'])),
        fixed_drain_risk=[dict(request_id=r['request_id'], fixed_drain_incomplete_probability_bounds=[
            sum((w['weight'] for w in rows if w['status'] == 'GENERATED_WORLD' and r['request_id'] in w['forecast']['world']['unfinished_request_ids']), Fraction(0)),
            sum((w['weight'] for w in rows if w['status'] == 'UNKNOWN_WORLD' or r['request_id'] in w['forecast']['world']['unfinished_request_ids']), Fraction(0))]) for r in requests],
        start_notice_transport_modeled=False, native_full_prefix_likelihood_identified=False,
        native_source_authenticated=False, native_busy_forecast_eligible=False,
        finite_support_complete=False, calibrated_posterior=False,
        native_guard_unchanged=True, host_elapsed_converted_to_GPU_progress=False,
        mismatch_slots_retained_as_unknown=True)
    return encode(out)
