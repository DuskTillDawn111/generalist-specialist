"""Initialize checked availability providers once for a real queue-cutoff DAG.

Only public inputs and immutable training sources materialize work, host supply
and phases. Execution uses the complete original graph compiled to its necessary
node kinds/edges; source-profile digests preserve the original training binding.
Missing modalities or policy states remain unsupported, with no rate fallback.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.resources import identity
from .availability_forecast_v11 import AvailabilityForecast
from .background_collect import context_identity
from .dag_busy_support_v9 import JointLifecycleFromSupport,verify_support
from .dag_checkpoint_v7 import receipt
from .dag_frozen_model_v10 import FrozenDagModel,FrozenNodeTable,plain
from .dag_phase_engine_v10 import DagPhaseCheckpointModel
from .dag_scheduler_state_v10 import CheckpointTrainedScheduler
from .multi_native_composite_v5 import checked
from .native_queue_busy_replay_v9 import queue_context_bridge


def compile_original_graph(plans):
    """Drop unused metadata only after retaining every original node and edge."""
    execution={};manifest=[]
    for coordinate,profile in plans.items():
        nodes=[dict(id=n['id'],kind=n['kind']) for n in profile['nodes']]
        ids={n['id'] for n in nodes}
        if len(ids)!=len(nodes):raise ValueError('Distinct original graph node IDs required')
        edges=[dict(before=e['before'],after=e['after']) for e in profile['edges']]
        if any(e['before'] not in ids or e['after'] not in ids for e in edges):
            raise ValueError('Original graph dependency has a foreign endpoint')
        source_digest=hashlib.sha256(json.dumps(profile,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        execution[coordinate]=dict(nodes=nodes,edges=edges,cyclic=profile.get('cyclic',False),
            unresolved=copy.deepcopy(profile.get('unresolved',[])),
            default_stream_semantics=profile.get('default_stream_semantics',''),
            original_training_profile_sha256=source_digest)
        manifest.append(dict(alias=coordinate[0],task=coordinate[1],pool_id=coordinate[2],
            original_training_profile_sha256=source_digest,original_node_count=len(nodes),
            original_edge_count=len(edges),nodes_or_edges_dropped=False))
    return execution,manifest


def _references(value):
    result={}
    def visit(item):
        if isinstance(item,dict):
            if set(item)=={'path','sha256'}:
                result[item['path']]=copy.deepcopy(item)
            else:
                for child in item.values():visit(child)
        elif isinstance(item,(list,tuple)):
            for child in item:visit(child)
    visit(value)
    return [result[k] for k in sorted(result)]


class NativeQueuePhaseFactory:
    def __init__(self,configuration_ref,support,public_cohort,target_context_ref,*,maximum_events=5_000_000,
                 progress=None,availability_adapter=None):
        started=time.perf_counter()
        if type(maximum_events) is not int or maximum_events<=0:raise ValueError('Original full-cohort event budget required')
        verify_support(support)
        self.support=copy.deepcopy(support);self.cohort=copy.deepcopy(public_cohort)
        self.maximum_events=maximum_events
        configuration=checked(configuration_ref)
        target=checked(target_context_ref)
        if context_identity(target)!=support['execution_context_sha256']:
            raise ValueError('Joint reader support and target full context differ')
        if availability_adapter is None:
            adapter=AvailabilityForecast(configuration,allow_unvalidated_training_hypothesis=True)
        elif type(availability_adapter) is AvailabilityForecast and availability_adapter.configuration==configuration:
            adapter=availability_adapter
        else:
            raise ValueError('Reused availability adapter must bind the same checked configuration')
        self.availability_adapter=adapter
        initialization=time.perf_counter()-started
        if progress:progress(dict(stage='RECONSTRUCTED_ORIGINAL_AVAILABILITY_PROVIDERS',elapsed_seconds=initialization))
        bridge=queue_context_bridge(adapter.bundle,target_context_ref)
        generation=adapter.generation.check(list(public_cohort.values()))
        if progress:progress(dict(stage='VERIFIED_PUBLIC_INPUT_GENERATION_AND_QUEUE_CONTEXT',elapsed_seconds=time.perf_counter()-started))
        if any(r['alias'] not in adapter.aliases or r['task'] not in adapter.aliases[r['alias']]['supported_tasks']
                for r in public_cohort.values()):
            raise ValueError('Public cohort changed declared owner/task domain')
        source_evidence=_references([configuration_ref,configuration,self.support['sources'],
            adapter.work.provenance(),adapter.phases.provenance(),adapter.host,bridge,receipt(__file__)])
        compile_started=time.perf_counter()
        execution,manifest=compile_original_graph(adapter.plans)
        self.model=FrozenDagModel(adapter.aliases,execution)
        compile_seconds=time.perf_counter()-compile_started
        if progress:progress(dict(stage='COMPILED_ORIGINAL_GRAPH_WITHOUT_DROPPING_NODES_OR_EDGES',
            static_templates=len(execution),compile_seconds=compile_seconds,elapsed_seconds=time.perf_counter()-started))
        work={};release={};phases={};coverage=[]
        table_started=time.perf_counter()
        for coordinate,profile in adapter.plans.items():
            row=next((r for r in public_cohort.values() if (r['alias'],r['task'],r['pool_id'])==coordinate),None)
            if row is None:
                row=dict(request_id='public-training-template',alias=coordinate[0],task=coordinate[1],pool_id=coordinate[2],arrival_ns=0)
            kinds={};full_work=0.
            for node in profile['nodes']:
                key=(*coordinate,node['id']);release[key]=adapter.release(node,row)
                if node['kind'] in ('gpu','copy','memset'):
                    value=adapter.work(node,row)
                    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                        raise UnknownEvidence('POSITIVE_SOURCE_BOUND_REFERENCE_WORK_REQUIRED')
                    work[key]=value;full_work+=value;kinds[node['kind']]=kinds.get(node['kind'],0)+1
                if node['kind']=='gpu':phases[key]=adapter.phases(node,row)
            coverage.append(dict(alias=coordinate[0],task=coordinate[1],pool_id=coordinate[2],
                node_counts=kinds,reference_work_ns_sum=full_work,co_run_lifetimes_used_as_reference_work=False))
            if progress:progress(dict(stage='MATERIALIZED_SOURCE_BOUND_TEMPLATE',alias=coordinate[0],task=coordinate[1],
                pool_id=coordinate[2],node_counts=kinds,elapsed_seconds=time.perf_counter()-started))
        self.duration=FrozenNodeTable(work,sources=source_evidence,domain='TWO_SOLO_ORDINAL_REFERENCE_NS_WITH_EXPLICIT_COPY_HYPOTHESIS')
        self.release=FrozenNodeTable(release,sources=source_evidence,domain='SOURCE_REDERIVED_PREPARATION_HOST_SUPPLY_NS')
        self.phases=FrozenNodeTable(phases,sources=source_evidence,domain='EXACT_ORIGINAL_DAG_TRAINING_PHASE')
        self.policy=copy.deepcopy(adapter.model['policy'])
        self.evidence=source_evidence;self.quality=copy.deepcopy(adapter.lifecycle.quality)
        missing=sorted({(r['alias'],r['task'],r['pool_id']) for r in public_cohort.values()}-set(adapter.plans))
        self.diagnostic=dict(schema='gsdeploy.actual_native_queue_phase_provider_initialization.v10',
            configuration=configuration_ref,source_evidence=source_evidence,
            complete_public_cohort=len(public_cohort),missing_public_templates=[list(k) for k in missing],
            generation_support=generation,queue_context_bridge=bridge,original_graph_compilation=manifest,
            work_coverage=coverage,training_provider_initialization_seconds=initialization,
            immutable_graph_compile_seconds=compile_seconds,materialized_tables_seconds=time.perf_counter()-table_started,
            total_initialization_seconds=time.perf_counter()-started,
            resource_axes='UNKNOWN_UNTIL_SEPARATELY_BOUND_RESOURCE_PROVIDER',
            full_context_equality_claimed=False,collector_transfer_is_hypothesis=True,
            independent_prediction_validation=False,physical_accuracy_validated=False,GPU_started=False)

    def risk_reference_work(self,provenance_ref):
        """Reuse already verified component/generation work for the same risk fit.

        Other components still rederive their two-solo ordinal costs. Exact
        evidence and resulting feature identity must equal the frozen risk
        training provenance; no coordinate is fitted or read from target output.
        """
        from .native_queue_risk_rows_v9 import ReconstructedReferenceWork,public_trace
        from .solo_reference_work_v8 import SoloOrdinalReferenceWork
        expected=checked(provenance_ref);configuration=checked(expected['solo_reference_configuration'])
        adapter=self.availability_adapter
        if expected['bundle']!=adapter.configuration['component_bundle']:
            raise ValueError('Risk work and phase DAG must bind the same source component bundle')
        plan=checked(expected['plan']);bridge=queue_context_bridge(adapter.bundle,plan['context'])
        generation=adapter.generation.check(public_trace(plan,0))
        composite=checked(adapter.bundle['composite']);specs={c['spec']['id']:c['spec'] for c in composite['components']}
        if set(configuration['components'])!=set(specs):raise ValueError('Complete risk component solo references required')
        values={};diagnostic={}
        for cid,refs in configuration['components'].items():
            if refs['dag']!=specs[cid]['dag']:raise ValueError('Risk solo source differs from original component DAG')
            if refs['dag']==adapter.host['dag'] and refs['solo_reviews']==adapter.parameters['solo_light_reviews']:
                work=adapter.work
            else:
                work=SoloOrdinalReferenceWork(refs['dag'],refs['solo_reviews'])
            for row in work.groups:
                key=json.dumps([row['alias'],row['task'],row['pool_id']],separators=(',',':'))
                if key in values:raise ValueError('Duplicate risk public reference coordinate')
                values[key]=row['reference_ns_sum']/1e6
            diagnostic[cid]=work.provenance()
        evidence=dict(bundle=expected['bundle'],plan=expected['plan'],
            solo_reference_configuration=expected['solo_reference_configuration'],bridge=bridge,
            provider_diagnostic=diagnostic,generation_support=generation,
            purpose='FIXED_PUBLIC_TRAINING_REFERENCE_FEATURE_COORDINATES',
            coordinate='sum of ordinal median GPU kernel lifetimes from two real isolated captures; excludes host/copy/lifecycle',
            co_run_lifetimes_used=False,physical_capacity_transfer_claimed=False,active_gpu_residual_included=False)
        binding=identity(dict(values=values,evidence=evidence))
        if dict(model_sha256=binding,**evidence)!=expected:
            raise ValueError('Reused source work fails exact frozen risk training provenance identity')
        provider=ReconstructedReferenceWork.__new__(ReconstructedReferenceWork)
        provider.values=values;provider.evidence=evidence;provider.binding=binding
        provider._verify()
        return provider

    def __call__(self,*,scenario_id,now_ns,request_start_provider):
        life=JointLifecycleFromSupport(self.support,scenario_id,self.cohort)
        return DagPhaseCheckpointModel(self.model.aliases,self.model.plans,frozen_model=self.model,
            duration_provider=self.duration,release_provider=self.release,phase_provider=self.phases,
            scheduler=CheckpointTrainedScheduler(self.policy),lifecycle_provider=life,
            request_start_provider=request_start_provider,max_events=self.maximum_events,
            allow_independent_copy_hypothesis=True,allow_no_implicit_default_stream_hypothesis=True,
            checkpoint_binding=dict(execution_context_sha256=self.support['execution_context_sha256'],evidence=self.evidence))
