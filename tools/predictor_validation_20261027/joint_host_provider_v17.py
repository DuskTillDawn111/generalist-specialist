"""Source-bound, whole-job observed host checkpoint scenarios.

This provider executes a conditional observation contract. It does not identify
exogenous CUDA releases for another input, route, arrival process, or execution.
No natural/light pairing, marginal owner composition, or scenario probabilities
are manufactured. Every selected scenario retains its complete offered cohort.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence

BANK_SCHEMA = 'gsdeploy.joint_training_host_prefix_observations.v13'
SCHEMA = 'gsdeploy.source_bound_joint_host_checkpoint_provider.v17'
CHECKPOINTS = {
    'natural': ('submit_offset_ns', 'handler_start_offset_ns',
                'pre_sync_start_offset_ns', 'pre_sync_end_offset_ns',
                'first_semantic_start_offset_ns'),
    'unified-light-v8_1': ('first_kernel_launch_API_entry_ns',
                         'first_kernel_launch_API_return_ns',
                         'first_kernel_launch_activity_start_ns',
                         'first_kernel_activity_start_ns',
                         'first_kernel_activity_API_entry_ns',
                         'first_kernel_activity_API_return_ns'),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def receipt(path):
    path = Path(path).resolve()
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return dict(path=str(path), sha256=sha.hexdigest())


def _reference(ref):
    if (set(ref) != {'path', 'sha256'} or not Path(ref['path']).is_absolute()
            or len(ref['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in ref['sha256'])):
        raise ValueError('Explicit absolute-path SHA256 receipt required')
    return dict(ref)


def _workload(rows):
    if not rows or not isinstance(rows, list):
        raise ValueError('Complete nonempty workload required')
    for row in rows:
        if (set(row) != {'alias', 'task', 'pool_id', 'arrival_ns'}
                or any(not isinstance(row[k], str) or not row[k]
                       for k in ('alias', 'task', 'pool_id'))
                or type(row['arrival_ns']) is not int or row['arrival_ns'] < 0):
            raise ValueError('Exact input/route/task/integer-arrival identity required')
    if len({r['alias'] for r in rows}) != len(rows):
        raise UnknownEvidence('REPEATED_OWNER_COHORT_NOT_IDENTIFIED_BY_THIS_BANK')
    # Order is representational; owner/arrival identities are never changed.
    return sorted(copy.deepcopy(rows), key=lambda r: r['alias'])


def _identity(scenario):
    return digest(dict(domain=scenario['domain'], source_raw=scenario['source_raw'],
                       execution_context_sha256=scenario['execution_context_sha256'],
                       job_id=scenario['job_id']))


class JointHostScenarioProvider:
    """Only construction reads evidence; subsequent results are detached copies.

    All transitive source bytes are checked, without parsing/reading any new
    labels. Call verify_sources again around an enclosing prediction run to
    detect changes after construction. Frozen observations remain the sole
    source of scenario values.
    """

    def __init__(self, observations):
        self.observations = _reference(observations)
        if receipt(self.observations['path']) != self.observations:
            raise ValueError('Joint host observations receipt mismatch')
        bank = json.loads(Path(self.observations['path']).read_text())
        flags = {'training_only': True, 'whole_job_correlation_preserved': True,
                 'per_owner_mean_composition_used': False,
                 'cross_domain_pairing_identified': False,
                 'cross_domain_joint_probabilities_assigned': False,
                 'synchronization_wait_is_exogenous_supply': False,
                 'complete_host_supply_provider': False}
        if bank.get('schema') != BANK_SCHEMA or any(bank.get(k) is not v for k, v in flags.items()):
            raise ValueError('Original training observation bank and explicit limits required')
        sources = [_reference(r) for r in bank['sources']]
        if len({r['path'] for r in sources}) != len(sources):
            raise ValueError('Unique source paths required')
        self.sources = [self.observations, *sources]
        refs = {r['path']: r for r in sources}
        self._scenarios = {}
        for original in bank['scenarios']:
            s = copy.deepcopy(original)
            for key in ('source_raw', 'source_review'):
                if refs.get(s[key]['path']) != s[key]:
                    raise ValueError('Scenario source is absent from frozen training closure')
            workload = _workload(s['public_workload'])
            if workload != s['public_workload'] or digest(workload) != s['workload_sha256']:
                raise ValueError('Complete canonical public workload digest mismatch')
            domain = s['domain']
            if (domain not in CHECKPOINTS or s['counterfactual_host_supply_identified'] is not False
                    or s['empirical_scenario_probability'] is not None):
                raise ValueError('Unsupported domain or invented causal/probability claim')
            if (len(s['prefixes']) != len(workload)
                    or {r['alias'] for r in s['prefixes']} != {r['alias'] for r in workload}
                    or len({r['request_id'] for r in s['prefixes']}) != len(workload)):
                raise ValueError('Every whole-job member must have one original prefix')
            expected_origin = 'host_perf_counter' if domain == 'natural' else 'native_Nsight_session'
            if (s['prefix_clock_origin']['domain'] != expected_origin
                    or s['public_arrival_and_light_offsets_share_origin'] is not (domain == 'natural')):
                raise ValueError('Original domain-specific clock origin required')
            for row in s['prefixes']:
                for field in CHECKPOINTS[domain]:
                    if row.get(field) is not None and (type(row[field]) is not int or row[field] < 0):
                        raise ValueError('Observed checkpoint must be nonnegative integer or missing')
                if domain == 'natural':
                    if (row['synchronization_wait_removed'] is not False
                            or row['first_semantic_is_first_CUDA_release'] is not False):
                        raise ValueError('Natural semantic boundary cannot become CUDA readiness')
                    arrival = next(r['arrival_ns'] for r in workload if r['alias'] == row['alias'])
                    if row['scheduled_offset_ns'] != arrival:
                        raise ValueError('Original public arrival origin mismatch')
                    ordered_times = [row.get(f) for f in ('scheduled_offset_ns', *CHECKPOINTS[domain])]
                    known_times = [t for t in ordered_times if t is not None]
                    if known_times != sorted(known_times):
                        raise ValueError('Natural host boundaries are not causally ordered')
                elif (row['API_wall_contains_possible_GPU_wait'] is not True
                      or row['observed_prefix_is_counterfactual_release'] is not False):
                    raise ValueError('Observed API prefix cannot become exogenous host supply')
            field = s['joint_event_field']
            if field not in CHECKPOINTS[domain]:
                raise ValueError('Unknown joint event boundary')
            ordered = sorted((r[field], r['alias']) for r in s['prefixes'] if r[field] is not None)
            gaps = {alias: stamp-ordered[0][0] for stamp, alias in ordered}
            if ([alias for _, alias in ordered] != s['joint_event_order']
                    or gaps != s['joint_event_gap_from_first_ns']):
                raise ValueError('Joint order/gap differs from complete original vector')
            scenario_id = _identity(s)
            if scenario_id in self._scenarios:
                raise ValueError('Duplicate source capture/job cannot add probability mass')
            self._scenarios[scenario_id] = s
        self.verify_sources()

    def verify_sources(self):
        for ref in self.sources:
            if receipt(ref['path']) != ref:
                raise ValueError('Joint host source closure changed: '+ref['path'])
        return copy.deepcopy(self.sources)

    def select(self, *, domain, workload, execution_context_sha256,
               source_capture_sha256=None, purpose='observed_checkpoint'):
        if purpose != 'observed_checkpoint':
            raise UnknownEvidence('COUNTERFACTUAL_JOINT_HOST_SUPPLY_NOT_IDENTIFIED')
        if domain not in CHECKPOINTS:
            raise UnknownEvidence('JOINT_HOST_DOMAIN_NOT_IDENTIFIED')
        workload = _workload(workload)
        selected = []
        for sid, s in self._scenarios.items():
            if (s['domain'] == domain and s['public_workload'] == workload
                    and s['execution_context_sha256'] == execution_context_sha256
                    and (source_capture_sha256 is None
                         or s['source_raw']['sha256'] == source_capture_sha256)):
                selected.append(dict(scenario_id=sid, **copy.deepcopy(s)))
        if not selected:
            raise UnknownEvidence('JOINT_HOST_CONTEXT_INPUT_ROUTE_OR_ARRIVAL_UNSUPPORTED')
        return dict(schema=SCHEMA, purpose=purpose, scenarios=selected,
                    scenario_probabilities=None, cross_domain_pairing=None,
                    counterfactual_release_identified=False,
                    observed_checkpoints_only=True, complete_host_supply_provider=False,
                    independent_prediction_validation=False)

    def schedule(self, scenario_id, *, checkpoint, purpose='observed_checkpoint',
                 clock_origin='observed', require_complete=True):
        if purpose != 'observed_checkpoint':
            raise UnknownEvidence('COUNTERFACTUAL_JOINT_HOST_SUPPLY_NOT_IDENTIFIED')
        if scenario_id not in self._scenarios:
            raise UnknownEvidence('UNKNOWN_SOURCE_BOUND_JOINT_SCENARIO')
        s = self._scenarios[scenario_id]
        if checkpoint not in CHECKPOINTS[s['domain']]:
            raise UnknownEvidence('JOINT_CHECKPOINT_NOT_OBSERVED_IN_THIS_DOMAIN')
        if clock_origin not in ('observed', 'public_arrival'):
            raise ValueError('No implicit per-owner or cross-domain clock map')
        if clock_origin == 'public_arrival' and not s['public_arrival_and_light_offsets_share_origin']:
            raise UnknownEvidence('LIGHT_TO_PUBLIC_ARRIVAL_CLOCK_MAP_NOT_IDENTIFIED')
        events = [dict(alias=r['alias'], request_id=r['request_id'],
                       offset_ns=r.get(checkpoint), observed=r.get(checkpoint) is not None)
                  for r in s['prefixes']]
        missing = [e['request_id'] for e in events if not e['observed']]
        if missing and require_complete:
            raise UnknownEvidence('JOINT_CHECKPOINT_MISSING_MEMBERS:'+','.join(missing))
        # Sorting does not drop unknown members or transform either owner's time.
        events.sort(key=lambda e: (e['offset_ns'] is None,
                                  e['offset_ns'] if e['observed'] else 0, e['alias']))
        return dict(schema=SCHEMA, scenario_id=scenario_id, checkpoint=checkpoint,
                    clock_origin=copy.deepcopy(s['prefix_clock_origin']), events=events,
                    public_workload=copy.deepcopy(s['public_workload']),
                    source_raw=copy.deepcopy(s['source_raw']),
                    missing_request_ids=missing, cohort_size=len(events),
                    scenario_probability=None, causal_release=False,
                    arrival_clock_aligned=s['public_arrival_and_light_offsets_share_origin'])

    def preparation_vector(self, scenario_id, *, require_complete=True):
        """Materialize the proposed statistical candidate; do not fit or execute it.

        Keeping P=semantic-handler jointly is distinct from using the original
        semantic-scheduled offsets as per-request preparation. H contains host
        transport/scheduling and is not an identified exogenous delay. Neither
        H nor P has had pre-sync GPU waiting silently removed.
        """
        if scenario_id not in self._scenarios:
            raise UnknownEvidence('UNKNOWN_SOURCE_BOUND_JOINT_SCENARIO')
        s = self._scenarios[scenario_id]
        if s['domain'] != 'natural':
            raise UnknownEvidence('NATURAL_JOINT_PREPARATION_BOUNDARIES_NOT_OBSERVED')
        rows = []
        for p in s['prefixes']:
            handler = p['handler_start_offset_ns']
            semantic = p['first_semantic_start_offset_ns']
            rows.append(dict(alias=p['alias'], request_id=p['request_id'],
                scheduled_to_handler_ns=handler-p['scheduled_offset_ns'],
                handler_to_first_semantic_ns=None if semantic is None else semantic-handler,
                pre_sync_start_from_handler_ns=None if p['pre_sync_start_offset_ns'] is None
                    else p['pre_sync_start_offset_ns']-handler,
                pre_sync_end_from_handler_ns=None if p['pre_sync_end_offset_ns'] is None
                    else p['pre_sync_end_offset_ns']-handler))
        missing = [r['request_id'] for r in rows if r['handler_to_first_semantic_ns'] is None]
        if require_complete and missing:
            raise UnknownEvidence('JOINT_PREPARATION_MISSING_MEMBERS:'+','.join(missing))
        return dict(schema=SCHEMA, scenario_id=scenario_id, vectors=rows,
                    missing_request_ids=missing, source_raw=copy.deepcopy(s['source_raw']),
                    public_workload=copy.deepcopy(s['public_workload']),
                    candidate_kind='UNVALIDATED_JOINT_PREPARATION_REPLACEMENT',
                    pre_sync_wait_removed=False, scenario_probability=None,
                    actual_first_ready_order_preserved_by_P_alone=False,
                    counterfactual_release_identified=False)
