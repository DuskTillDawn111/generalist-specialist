"""Training-only ordered timeline conversion and opt-in profiler-domain replay.

This adapter is outside frozen runtime sources. Sequential-stream traces retain
each actual kernel and every positive non-kernel interval. A gap has no inferred
CPU/DRAM demand. Cross-request launch delays require a separate conditional
model; isolated profiled gaps never become natural timing or zero interference.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.kernel_provider import KernelResourceProvider, KERNEL_PROFILE_SCHEMA
from gsdeploy.resources import identity
from .kernel_sequence_audit_v2 import load_collection, IDENTITY_FIELDS
from .native_collect import receipt
from .trace_export import linux_pid

SCHEMA = 'gsdeploy.ordered_kernel_profiles.v2'


def convert_collection(raw_path, parsed_paths, *, hardware=None, occupancy_resolver=None):
    """Convert completed isolated light-kernel training, with full denominator.

    occupancy_resolver(kernel) may return an exact-source-bound patched kernel.
    Missing occupancy is retained as missing, rather than guessed. Rejecting an
    unsupported request preserves its identity/reason in rejected_requests.
    """
    raw, plan, context, requests, audit = load_collection(raw_path, parsed_paths)
    if (plan.get('split') != 'training' or plan.get('training_only') is not True
            or plan.get('domain') != 'light-kernel' or context.get('measurement_domain') != 'light-kernel'
            or any(j.get('split') != 'training' or len(j['inputs']) != 1 for j in plan['jobs'])
            or any(r['row'].get('split') != 'training' for r in requests.values())):
        raise ValueError('Only isolated light-kernel training can supply ordered profiles; smoke/validation rejected')
    if audit['summary']['unassigned_kernel_count']:
        raise ValueError('Canonical kernel ownership is incomplete')
    jobs = {j['id']: j for j in plan['jobs']}
    profiles, rejected, seen = [], [], set()
    for result in audit['requests']:
        rid = result['request_id']; request = requests[rid]; row = request['row']
        reason = None
        if not result['single_serial_stream_expressible']:
            reason = 'NON_SERIAL_STREAM_DAG_REQUIRED'
        boundary = result.get('host_boundaries', {})
        if any(type(boundary.get(k)) is not int or boundary[k] < 0
               for k in ('prefix_unclassified_ns', 'suffix_unclassified_ns')):
            reason = reason or 'HANDLER_BOUNDARY_CLOCK_MAPPING_REQUIRED'
        job = jobs.get(row.get('profile_job_id'))
        if job is None or job['inputs'] != {row['model']: row['pool_id']}:
            raise ValueError('Training request is not bound to its exact declared isolated job')
        key = row['model'], row['task'], row['pool_id']
        if key in seen:
            reason = reason or 'REPEATED_INPUT_REQUIRES_EXPLICIT_SCENARIO_AGGREGATION'
        if reason:
            rejected.append(dict(request_id=rid, alias=row['model'], task=row['task'], pool_id=row['pool_id'], reason=reason))
            continue
        seen.add(key)
        ordered = sorted(request['kernels'], key=lambda k: tuple(k.get(n, 0) or 0 for n in IDENTITY_FIELDS))
        segments = []
        def gap(duration, evidence):
            if duration:
                segments.append(dict(id='segment-'+str(len(segments)), kind='unclassified_gap',
                    duration_ms=dict(profiled=duration/1e6), evidence=evidence,
                    demand=None, classification='UNKNOWN_NOT_CPU_OR_FREE_GPU', natural_timing_validated=False))
        gap(boundary['prefix_unclassified_ns'], dict(boundary='handler_to_first_kernel', mapping=boundary['mapping']))
        previous = None
        for kernel in ordered:
            if previous is not None:
                gap(kernel['nsys_start_ns']-previous['nsys_end_ns'],
                    dict(previous=[previous[n] for n in IDENTITY_FIELDS], next=[kernel[n] for n in IDENTITY_FIELDS],
                         source=kernel['source_sqlite'], clock='same_trace_nsys'))
            patched = occupancy_resolver(kernel) if occupancy_resolver else None
            if patched is not None:
                evidence = patched.get('static_evidence', {})
                launch = evidence.get('launch', {})
                if (evidence.get('source') != kernel['source_sqlite']
                        or launch.get('start') != kernel['nsys_start_ns'] or launch.get('end') != kernel['nsys_end_ns']
                        or launch.get('correlation_id') != kernel['correlation_id']
                        or launch.get('context_id') != kernel['context_id']
                        or linux_pid(launch.get('global_pid', 0)) != kernel['pid']):
                    raise ValueError('Occupancy patch is not bound to this exact completed launch/source')
                for field in ('block', 'grid', 'registers_per_thread', 'static_shared_bytes', 'dynamic_shared_bytes', 'cluster'):
                    if patched.get(field) != kernel.get(field):
                        raise ValueError('Occupancy patch changes observed launch geometry/resource declaration')
            segments.append(dict(id='segment-'+str(len(segments)), kind='gpu',
                duration_ms=dict(profiled=(kernel['nsys_end_ns']-kernel['nsys_start_ns'])/1e6),
                kernel=copy.deepcopy(patched or kernel), solo_timing_basis='profiled_isolated',
                timing_evidence=dict(source=kernel['source_sqlite'], identity=[kernel[n] for n in IDENTITY_FIELDS]),
                profiler_id='nsys-light-kernel:'+raw['execution_context_sha256'], profiler_perturbed=True,
                perturbation_flags=['CUPTI/profiler overhead uncalibrated; not natural timing'],
                natural_timing_validated=False))
            previous = kernel
        gap(boundary['suffix_unclassified_ns'], dict(boundary='last_kernel_to_handler', mapping=boundary['mapping']))
        profiles.append(dict(id=rid, alias=row['model'], family=result['family'], task=row['task'], pool_id=row['pool_id'],
            execution=dict(batch_size=1, backend='native-profiled:'+raw['execution_context_sha256'], weight_reuse=False),
            boundary_mapping=boundary['mapping'], observed_host_anchors=boundary.get('anchors', []),
            segments=segments, scope='isolated_profiler_domain_timeline_with_uncalibrated_boundary_mapping'))
    document = dict(schema=SCHEMA, execution_context_sha256=raw['execution_context_sha256'],
        provenance=dict(kind='measured', split='training', sources=audit['source_receipts']+[receipt(__file__)]+
                        getattr(occupancy_resolver,'source_receipts',[])),
        hardware=hardware or {}, profiles=profiles, rejected_requests=rejected,
        denominator=dict(request_count=len(requests), converted_requests=len(profiles), rejected_requests=len(rejected)),
        sequence_audit=audit, concurrency_evidence=None, natural_timing_validated=False,
        physically_validated=False, deployment_qualified=False,
        cross_request_launch_delay_model='UNKNOWN_CONDITIONAL_MODEL_REQUIRED')
    document['profiles_sha256'] = identity(document)
    return document


class OrderedKernelProfileProvider:
    """Executable ordered plan; gaps cannot silently progress at speed one.

    allow_isolated_profiled_replay is an explicit diagnostic hypothesis. It is
    only usable while the request is the sole active owner, and does not certify
    handler-clock calibration. Cross-request GPU service uses the existing
    opt-in static packing hypothesis, with no claim of physical correctness.
    """
    def __init__(self, document, execution_context_sha256, *, allow_cpu_fixture=False,
                 allow_isolated_profiled_replay=False, allow_analytical_hypothesis=False,
                 gap_provider=None, power_provider=None, memory_provider=None):
        self.document = copy.deepcopy(document)
        if document.get('schema') != SCHEMA or document.get('execution_context_sha256') != execution_context_sha256:
            raise ValueError('Ordered profile context/schema mismatch')
        provenance = document.get('provenance', {})
        if (provenance.get('split') != 'training' or provenance.get('kind') not in ('measured','cpu_fixture')
                or not provenance.get('sources') or provenance.get('kind') == 'cpu_fixture' and not allow_cpu_fixture):
            raise ValueError('Explicit training provenance/fixture opt-in required')
        if type(allow_isolated_profiled_replay) is not bool:
            raise ValueError('Explicit boolean isolated replay opt-in required')
        self.execution_context_sha256 = execution_context_sha256
        self.allow_isolated_profiled_replay = allow_isolated_profiled_replay
        self.gap_provider = gap_provider
        self.physical_evidence = self.physically_validated = self.deployment_qualified = False
        self.profiles, self.lookup, self.segments = {}, {}, {}
        projected = []
        for profile in self.document['profiles']:
            key = profile['alias'], profile['task'], profile['pool_id']
            if key in self.lookup or profile['id'] in self.profiles:
                raise ValueError('Duplicate ordered profile identity')
            self.lookup[key] = profile; self.profiles[profile['id']] = profile
            ids = set()
            for segment in profile['segments']:
                if (segment['id'] in ids or segment['kind'] not in ('gpu','unclassified_gap')
                        or not segment.get('evidence' if segment['kind'] == 'unclassified_gap' else 'timing_evidence')):
                    raise ValueError('Explicit ordered segment kind/identity/evidence required')
                ids.add(segment['id']); self.segments[profile['id'], segment['id']] = segment
                for value in segment['duration_ms'].values():
                    if type(value) not in (int,float) or not 0 < value < float('inf'):
                        raise ValueError('Positive finite profiled duration required')
            projected.append(dict(profile, family=profile['alias'],
                                  segments=[s for s in profile['segments'] if s['kind']=='gpu']))
        internal = dict(schema=KERNEL_PROFILE_SCHEMA, execution_context_sha256=execution_context_sha256,
            provenance=dict(provenance, sources=[str(s) for s in provenance['sources']]),
            profiles=projected, hardware=document.get('hardware', {}),
            concurrency_evidence=document.get('concurrency_evidence'))
        self.internal = KernelResourceProvider(internal, execution_context_sha256, allow_cpu_fixture=allow_cpu_fixture,
            allow_analytical_hypothesis=allow_analytical_hypothesis, allow_profiled_baselines=True,
            power_provider=power_provider, memory_provider=memory_provider)

    def plan(self, row, aliases, scenario):
        profile = self.lookup.get((row['alias'],row['task'],row['pool_id']))
        if profile is None or aliases[row['alias']]['family'] != profile['family']:
            raise UnknownEvidence('UNKNOWN_INPUT_EXECUTION_PROFILE:'+row['alias'])
        if any(scenario not in s['duration_ms'] for s in profile['segments']):
            raise UnknownEvidence('UNKNOWN_ORDERED_DURATION_SCENARIO:'+scenario)
        return dict(profile_id=profile['id'], execution=copy.deepcopy(profile['execution']), segments=[
            dict(id=s['id'],kind=s['kind'],duration_ms=s['duration_ms'][scenario],
                 demand=dict(sm=None,dram=None,l2=None)) for s in profile['segments']])

    @staticmethod
    def _unknown(reason):
        return dict(known=False, context_supported=False, reason=reason,
                    physically_validated=False, deployment_qualified=False, natural_timing_validated=False)

    def _validated_sources(self, row, active, aliases, scenario):
        sources = {}
        for alias, candidate in active.items():
            profile = self.lookup.get((alias,candidate['task'],candidate['pool_id']))
            if (profile is None or candidate.get('alias') != alias or candidate.get('profile_id') != profile['id']
                    or aliases[alias]['family'] != profile['family']):
                raise ValueError('Active ordered profile identity mismatch')
            source = self.segments.get((profile['id'],candidate.get('segment_id')))
            if (source is None or source['kind'] != candidate.get('segment_kind')
                    or source['duration_ms'].get(scenario) != candidate.get('segment_duration_ms')):
                raise ValueError('Active ordered segment differs from source-bound timeline')
            sources[alias] = source
        alias = row['alias']
        if alias not in active or active[alias] != row:
            raise ValueError('Target must be an active ordered row')
        return sources

    def service(self, row, active, aliases, scenario):
        sources=self._validated_sources(row,active,aliases,scenario)
        alias=row['alias']
        if len(active) == 1 and self.allow_isolated_profiled_replay:
            return dict(known=True,context_supported=True,service_ms=row['segment_duration_ms'],
                assumed_relative_progress=1.,status='ISOLATED_PROFILED_REPLAY_HYPOTHESIS',
                natural_timing_validated=False, physically_validated=False,deployment_qualified=False)
        if sources[alias]['kind'] == 'unclassified_gap':
            if self.gap_provider is None:
                return self._unknown('CROSS_REQUEST_LAUNCH_DELAY_MODEL_REQUIRED')
            if getattr(self.gap_provider,'execution_context_sha256',None) != self.execution_context_sha256:
                raise ValueError('Conditional gap provider execution context mismatch')
            return self.gap_provider.service(row,active,aliases,scenario)
        if any(s['kind'] == 'unclassified_gap' for s in sources.values()):
            return self._unknown('UNCLASSIFIED_GAP_GPU_INTERFERENCE_MODEL_REQUIRED')
        private_aliases = {name:dict(aliases[name],family=name) for name in aliases}
        return self.internal.service(row,active,private_aliases,scenario)

    def power(self,*args): return self.internal.power(*args)
    def memory(self,*args): return self.internal.memory(*args)
    def power_preparing(self,*args): return self.internal.power_preparing(*args)
    def memory_preparing(self,*args): return self.internal.memory_preparing(*args)
    def demand_summary(self,profile_id): return self.internal.demand_summary(profile_id)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',required=True);parser.add_argument('--parsed',nargs='+',required=True)
    parser.add_argument('--hardware');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--occupancy-audit')
    args=parser.parse_args()
    hardware=json.loads(Path(args.hardware).read_text()) if args.hardware else None
    resolver=None
    if args.occupancy_audit:
        from .ordered_occupancy_resolver_v2 import AuditedOccupancyResolver
        resolver=AuditedOccupancyResolver(args.occupancy_audit)
        if hardware is not None and hardware!=resolver.hardware:raise ValueError('Hardware differs from occupancy audit')
        hardware=resolver.hardware
    document=convert_collection(args.raw,args.parsed,hardware=hardware,occupancy_resolver=resolver)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'profiles.json').open('x') as stream:json.dump(document,stream,indent=2,allow_nan=False)
    print(json.dumps(document['denominator']))


if __name__=='__main__':main()
