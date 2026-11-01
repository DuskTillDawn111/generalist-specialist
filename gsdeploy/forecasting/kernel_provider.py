"""FullHorizonModel adapter for an opt-in, unvalidated static packing model.

Kernel static constraints never become physical throughput evidence here.
Board power and owned memory require independent providers.
"""
from __future__ import annotations

import copy

from ..forecast_errors import UnknownEvidence
from .kernel_resources import allocate_block_progress, static_occupancy, _number


KERNEL_PROFILE_SCHEMA = 'gsdeploy.kernel_resource_profiles.v2'


class KernelResourceProvider:
    """Training-bound ordered kernel baselines with analytical progress only.

    Segment duration_ms maps named scenarios to isolated baseline durations.
    A GPU segment additionally declares kernel (static_occupancy input),
    solo_timing_basis and timing_evidence. Profiled-isolated timing requires a
    separate explicit opt-in and stays labelled in its profiler domain. Non-GPU segments
    progress independently at one; there is no implicit host contention model.
    """

    def __init__(self, document, execution_context_sha256, *, allow_cpu_fixture=False,
                 allow_analytical_hypothesis=False, allow_profiled_baselines=False,
                 power_provider=None, memory_provider=None):
        self.document = copy.deepcopy(document)
        if (document.get('schema') != KERNEL_PROFILE_SCHEMA or
                not isinstance(execution_context_sha256, str) or not execution_context_sha256 or
                document.get('execution_context_sha256') != execution_context_sha256):
            raise ValueError('Kernel resource execution context/schema mismatch')
        provenance = document.get('provenance', {})
        if (provenance.get('kind') not in ('measured', 'cpu_fixture') or
                provenance.get('split') != 'training' or
                not isinstance(provenance.get('sources'), list) or not provenance['sources'] or
                any(not isinstance(s, str) or not s for s in provenance['sources'])):
            raise ValueError('Explicit training-only kernel evidence provenance required')
        if provenance['kind'] == 'cpu_fixture' and not allow_cpu_fixture:
            raise ValueError('CPU fixture is not physical profiling evidence')
        if type(allow_analytical_hypothesis) is not bool:
            raise ValueError('Explicit boolean analytical hypothesis opt-in required')
        if type(allow_profiled_baselines) is not bool:
            raise ValueError('Explicit boolean profiled baseline opt-in required')
        self.execution_context_sha256 = execution_context_sha256
        self.physical_evidence = False
        self.physically_validated = False
        self.deployment_qualified = False
        self.allow_analytical_hypothesis = allow_analytical_hypothesis
        self.allow_profiled_baselines = allow_profiled_baselines
        self.synthetic_inputs = provenance['kind'] == 'cpu_fixture'
        self.hardware = self.document.get('hardware', {})
        self.policy = self.document.get('policy', 'equal_block_round_robin')
        if self.policy not in ('equal_block_round_robin', 'fcfs_blocks'):
            raise ValueError('Unsupported static block allocation policy')
        self.concurrency_evidence = self.document.get('concurrency_evidence')
        self.power_provider, self.memory_provider = power_provider, memory_provider
        self.profiles, self.lookup, self.segments = {}, {}, {}
        for profile in self.document.get('profiles', []):
            for key in ('id', 'family', 'task', 'pool_id'):
                if not isinstance(profile.get(key), str) or not profile[key]:
                    raise ValueError('Explicit kernel profile identities required')
            execution = profile.get('execution', {})
            if (type(execution.get('batch_size')) is not int or execution['batch_size'] < 1 or
                    not isinstance(execution.get('backend'), str) or not execution['backend'] or
                    type(execution.get('weight_reuse')) is not bool):
                raise ValueError('Explicit batch/backend/weight-reuse configuration required')
            key = profile['family'], profile['task'], profile['pool_id']
            if profile['id'] in self.profiles or key in self.lookup or not profile.get('segments'):
                raise ValueError('Duplicate or empty kernel profile')
            ids = set()
            for segment in profile['segments']:
                if (not isinstance(segment.get('id'), str) or not segment['id'] or segment['id'] in ids or
                        segment.get('kind') not in ('gpu', 'host', 'sync', 'cleanup')):
                    raise ValueError('Invalid kernel segment identity/kind')
                ids.add(segment['id'])
                durations = segment.get('duration_ms')
                if not isinstance(durations, dict) or not durations:
                    raise ValueError('Named isolated baseline duration scenarios required')
                for scenario, duration in durations.items():
                    if not isinstance(scenario, str) or not scenario:
                        raise ValueError('Nonempty duration scenario identity required')
                    _number(duration, 'isolated segment duration')
                if segment['kind'] == 'gpu':
                    if not isinstance(segment.get('kernel', {}), dict):
                        raise ValueError('Kernel static fields must be a mapping')
                    # Validate malformed attributes now; incomplete evidence stays UNKNOWN.
                    static_occupancy(segment.get('kernel', {}), self.hardware)
                self.segments[profile['id'], segment['id']] = segment
            self.profiles[profile['id']] = profile
            self.lookup[key] = profile
        if not self.profiles:
            raise ValueError('Nonempty training-bound kernel profiles required')

    def plan(self, row, aliases, scenario):
        alias = row['alias']
        profile = self.lookup.get((aliases[alias]['family'], row['task'], row['pool_id']))
        if profile is None:
            raise UnknownEvidence('UNKNOWN_INPUT_EXECUTION_PROFILE:' + alias + ':' + row['pool_id'])
        segments = []
        for source in profile['segments']:
            if scenario not in source['duration_ms']:
                raise UnknownEvidence('UNKNOWN_SEGMENT_DURATION_SCENARIO:' + scenario)
            segment = dict(id=source['id'], kind=source['kind'], duration_ms=source['duration_ms'][scenario],
                           demand={k: None if source['kind'] == 'gpu' else 0. for k in ('sm', 'dram', 'l2')})
            segments.append(segment)
        return dict(profile_id=profile['id'], execution=copy.deepcopy(profile['execution']), segments=segments)

    def _active_segments(self, active, aliases, scenario):
        result = {}
        for alias, row in active.items():
            profile = self.profiles.get(row.get('profile_id'))
            if profile is None or (profile['family'], profile['task'], profile['pool_id']) != (
                    aliases[alias]['family'], row['task'], row['pool_id']):
                raise ValueError('Active kernel profile does not match request identity')
            source = self.segments.get((profile['id'], row.get('segment_id')))
            if source is None or source['kind'] != row.get('segment_kind'):
                raise ValueError('Active kernel segment does not match its ordered plan')
            if scenario not in source['duration_ms']:
                raise UnknownEvidence('UNKNOWN_SEGMENT_DURATION_SCENARIO:' + scenario)
            if row.get('segment_duration_ms') != source['duration_ms'][scenario]:
                raise ValueError('Active baseline differs from training-bound plan')
            result[alias] = source
        return result

    @staticmethod
    def _unknown(reason, **extra):
        return dict(known=False, context_supported=False, reason=reason,
                    physically_validated=False, deployment_qualified=False, **extra)

    def service(self, row, active, aliases, scenario):
        sources = self._active_segments(active, aliases, scenario)
        alias = row['alias']
        if alias not in active or active[alias] != row:
            raise ValueError('Target row must belong to active kernel context')
        if sources[alias]['kind'] != 'gpu':
            return dict(known=True, context_supported=True, service_ms=row['segment_duration_ms'],
                        assumed_relative_progress=1., scope='independent_non_gpu_baseline',
                        physically_validated=False, deployment_qualified=False)
        if not self.allow_analytical_hypothesis:
            return self._unknown('ANALYTICAL_HYPOTHESIS_OPT_IN_REQUIRED')
        if not self.concurrency_evidence:
            return self._unknown('CROSS_CONTEXT_CONCURRENCY_EVIDENCE_REQUIRED')
        kernels = {}
        timing_domains = {}
        # Stable alias order is the declared allocation priority at every active
        # boundary; it is not a claim about CUDA scheduler priority or fairness.
        for name in sorted(sources):
            source = sources[name]
            if source['kind'] != 'gpu':
                continue
            basis = source.get('solo_timing_basis')
            allowed = basis == 'unprofiled_isolated' or basis == 'synthetic' and self.synthetic_inputs
            if basis == 'profiled_isolated':
                if not self.allow_profiled_baselines:
                    return self._unknown('PROFILED_BASELINE_OPT_IN_REQUIRED')
                if (not isinstance(source.get('profiler_id'), str) or not source['profiler_id']
                        or source.get('profiler_perturbed') is not True
                        or not isinstance(source.get('perturbation_flags'), list)
                        or not source['perturbation_flags']
                        or any(not isinstance(v, str) or not v for v in source['perturbation_flags'])):
                    return self._unknown('EXPLICIT_PROFILER_DOMAIN_REQUIRED')
                allowed = True
            if not source.get('timing_evidence') or not allowed:
                return self._unknown('ISOLATED_KERNEL_TIMING_REQUIRED')
            kernels[name] = source.get('kernel', {})
            timing_domains[name] = dict(basis=basis, profiler_id=source.get('profiler_id'),
                                        profiler_perturbed=source.get('profiler_perturbed'),
                                        perturbation_flags=source.get('perturbation_flags'))
        allocation = allocate_block_progress(kernels, self.hardware, policy=self.policy)
        if not allocation['known']:
            return self._unknown('STATIC_RESOURCE_CONTEXT_UNKNOWN', detail=allocation)
        speeds = allocation['assumed_relative_progress']
        return dict(known=True, context_supported=True,
                    service_ms=row['segment_duration_ms']/speeds[alias] if speeds[alias] > 0 else None,
                    stalled=speeds[alias] == 0,
                    assumed_relative_progress=speeds[alias], allocation=allocation,
                    status='ANALYTICAL_HYPOTHESIS', scope='static_block_occupancy_progress_hypothesis',
                    model_assumption='Uniform full-grid per-SM packing; progress=allocated blocks/solo static ceiling; no DRAM/cache slowdown.',
                    evidence=self.document['provenance']['sources'], synthetic_inputs=self.synthetic_inputs,
                    timing_domains=timing_domains, natural_timing_validated=False,
                    physically_validated=False, deployment_qualified=False,
                    independently_validated=False)

    def power(self, active, aliases, scenario):
        if self.power_provider is None:
            return dict(known=False, watts=None, reason='INDEPENDENT_BOARD_POWER_PROVIDER_REQUIRED')
        return self.power_provider.power(active, aliases, scenario)

    def memory(self, active, aliases, scenario):
        if self.memory_provider is None:
            return dict(known=False, peak_memory_mib=None, reason='INDEPENDENT_MEMORY_PROVIDER_REQUIRED')
        return self.memory_provider.memory(active, aliases, scenario)

    def power_preparing(self, preparing, active, aliases, scenario):
        method = getattr(self.power_provider, "power_preparing", None)
        return method(preparing, active, aliases, scenario) if method else dict(known=False, watts=None)

    def memory_preparing(self, preparing, active, aliases, scenario):
        method = getattr(self.memory_provider, "memory_preparing", None)
        return method(preparing, active, aliases, scenario) if method else dict(known=False, peak_memory_mib=None)

    def demand_summary(self, profile_id):
        profile = self.profiles[profile_id]
        gpu = any(s['kind'] == 'gpu' for s in profile['segments'])
        return dict(classification='unknown' if gpu else 'host',
                    **{k: None if gpu else 0. for k in ('sm', 'dram', 'l2')},
                    reason='STATIC_OCCUPANCY_IS_NOT_COMPUTE_OR_DRAM_THROUGHPUT')
