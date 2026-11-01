"""Node-keyed DAG kernel clocks, with no serial-profile admission dependency.

Only static launch fields and a separately fitted reference law enter numeric
propagation. The serial compiler remains unchanged; no OCR/native qualification
is inferred by accepting a multistream DAG here.
"""
from __future__ import annotations

import copy
import math
from scipy.special import ndtri

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.resources import identity
from . import kernel_competition_features_v2 as features
from .kernel_competition_response_provider_v1 import DOMAIN
from .dag_phase_scheduler_v8 import SchedulingDecision
from .stage_response_features_v1 import receipt
from .dag_frozen_model_v10 import freeze,plain,static_digest
from .dag_checkpoint_v7 import callable_identity

BUNDLE_SCHEMA='gsdeploy.kernel_dag_response_training.v1'
STATIC_FIELDS=('name','hardware_id','static_evidence','block','grid','registers_per_thread',
    'static_shared_bytes','dynamic_shared_bytes','allocated_registers_per_block',
    'allocated_shared_bytes','allocation_evidence','architecture_blocks_per_sm_ceiling',
    'cluster','cooperative_launch')


def need(value,message):
    if not value:raise ValueError(message)


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def training_roles(profiles_document,bundle):
    """Reject foreign/invalid roles before copying, hashing or reading labels."""
    need(type(bundle) is dict and bundle.get('split')=='training'
         and bundle.get('validation_labels_read') is False,'Explicit training response bundle role required')
    need(type(profiles_document) is dict and profiles_document.get('training_only') is True
         and profiles_document.get('validation_labels_read') is False,
         'Explicit training-only catalog role and unread validation labels required')
    profiles=profiles_document.get('profiles')
    need(type(profiles) is list and profiles,'Nonempty explicit training DAG profiles required')
    for profile in profiles:
        need(type(profile) is dict and profile.get('split')=='training',
             'Every DAG profile must have the training role')
        need(type(profile.get('unresolved')) is list and not profile['unresolved']
             and profile.get('cyclic') is False,'Resolved acyclic training DAG profile required')
        nodes=profile.get('nodes')
        need(type(nodes) is list and nodes,'Nonempty explicit DAG node list required')
        seen=set()
        for node in nodes:
            need(type(node) is dict and type(node.get('id')) is str and node['id']
                 and node['id'] not in seen
                 and node.get('kind') in ('gpu','copy','memset','event_record','event_wait','host_sync','handler_end'),
                 'Unique typed DAG node identity required')
            seen.add(node['id'])
            need(node['kind']!='gpu' or type(node.get('kernel')) is dict,
                 'Explicit static GPU node declaration required')


class KernelDagNodeClockAdapter:
    __slots__=('bundle_sha256','catalog_sha256','execution_context_sha256','profiler_id',
        'hardware','allow_response','fit','qualified','beta','baselines','sigma','scenarios',
        'scenario','non_gpu_duration_provider','lookup','static','_configuration','_sealed')

    def __init__(self,profiles_document,hardware,bundle,*,scenario='mean',scenarios=None,
            allow_unvalidated_profiled_response=False,allow_cpu_fixture=False,
            non_gpu_duration_provider=None):
        object.__setattr__(self,'_sealed',False)
        training_roles(profiles_document,bundle)
        profiles_document=copy.deepcopy(profiles_document)
        hardware=copy.deepcopy(hardware)
        need(type(allow_unvalidated_profiled_response) is bool and type(allow_cpu_fixture) is bool,
             'Explicit boolean model/fixture opt-ins required')
        doc=copy.deepcopy(bundle);seal=doc.pop('bundle_sha256',None)
        need(seal==identity(doc),'Node response bundle changed')
        need(doc.get('schema')==BUNDLE_SCHEMA and doc.get('feature_schema')==features.SCHEMA
             and doc.get('features')==list(features.FEATURES) and doc.get('response_domain')==DOMAIN
             and doc.get('validation_labels_read') is False,'Exact training node response bundle required')
        need(doc.get('kernel_catalog_sha256')==identity(profiles_document)
             and doc.get('execution_context_sha256')==profiles_document.get('execution_context_sha256')
             and doc.get('hardware_id')==hardware.get('hardware_id')
             and doc.get('hardware_sha256')==identity(hardware),'Exact DAG catalog/context/hardware binding required')
        need(doc.get('evidence_kind') in ('declared_measured','cpu_fixture')
             and (doc['evidence_kind']!='cpu_fixture' or allow_cpu_fixture),'Explicit CPU fixture opt-in required')
        need(type(doc.get('profiler_id')) is str and doc['profiler_id'],'Explicit fitted profiler domain required')
        need(type(doc.get('sources')) is list and doc['sources']
             and all(type(r) is dict and receipt(r['path'])==r for r in doc['sources']),
             'Unchanged declared response source receipts required')
        self.bundle_sha256=seal;self.catalog_sha256=doc['kernel_catalog_sha256']
        self.execution_context_sha256=doc['execution_context_sha256'];self.profiler_id=doc['profiler_id']
        self.hardware=copy.deepcopy(hardware);self.allow_response=allow_unvalidated_profiled_response
        self.fit=doc['response_fit'];self.qualified=self.fit.get('numerical_qualification') is True
        self.beta=None;self.baselines={};self.sigma=None
        if self.qualified:
            need(self.fit.get('features')==list(features.FEATURES),'Complete six-column fitted node law required')
            beta=self.fit.get('coefficients');sigma=self.fit.get('lognormal_reference_sigma')
            need(type(beta) is list and len(beta)==len(features.FEATURES) and all(finite(v) for v in beta)
                 and finite(sigma) and sigma>0,'Finite complete coefficients and positive reference sigma required')
            self.beta=beta;self.sigma=sigma
            for entry in self.fit.get('baselines',[]):
                key=tuple(entry['group'])
                need(len(key)==5 and all(type(k) is str and k for k in key)
                     and key[-2:]==(self.execution_context_sha256,self.profiler_id)
                     and key not in self.baselines and finite(entry['log_reference_ms']),
                     'Exact unique owner/profile/node/context/profiler reference required')
                self.baselines[key]=entry['log_reference_ms']
            need(self.baselines,'Fitted node reference baselines required')
        self.scenarios=copy.deepcopy(scenarios if scenarios is not None else {'mean':dict(reference_functional='mean')})
        need(type(self.scenarios) is dict and self.scenarios,'Predeclared node reference scenarios required')
        for name,spec in self.scenarios.items():
            need(type(name) is str and name and type(spec) is dict,'Named reference scenario required')
            need((spec==dict(reference_functional='mean')) or
                 (set(spec)=={'reference_functional','quantile'} and spec['reference_functional']=='quantile'
                  and finite(spec['quantile']) and 0<spec['quantile']<1),'Closed mean/quantile reference scenario required')
        need(scenario in self.scenarios,'Unknown node reference scenario')
        self.scenario=scenario;self.non_gpu_duration_provider=non_gpu_duration_provider
        self.lookup={};self.static={}
        for profile in profiles_document['profiles']:
            key=(profile['alias'],profile['task'],profile['pool_id'])
            need(key not in self.lookup,'One exact public request DAG template required; do not pool duplicate profiles')
            need(type(profile['request_id']) is str and profile['request_id'],'Explicit catalog profile ID required')
            self.lookup[key]=profile['request_id']
            for node in profile['nodes']:
                if node['kind']!='gpu':continue
                nodekey=(*key,node['id'])
                need(nodekey not in self.static,'Distinct owner/input/node binding required')
                # Training catalog observations may retain timestamps, but no
                # observed START/END/release/duration becomes a prospective law.
                self.static[nodekey]={k:copy.deepcopy(node['kernel'][k]) for k in STATIC_FIELDS if k in node['kernel']}
        configuration=dict(kind='kernel_dag_node_clock_v11',bundle_sha256=self.bundle_sha256,
            catalog_sha256=self.catalog_sha256,execution_context_sha256=self.execution_context_sha256,
            profiler_id=self.profiler_id,scenario=self.scenario,scenarios=self.scenarios,
            allow_unvalidated_profiled_response=self.allow_response,
            cpu_fixture_opt_in=allow_cpu_fixture,sources=doc['sources'],
            static_sha256=static_digest(self.static),lookup_sha256=static_digest(self.lookup),
            fit_sha256=static_digest(self.fit),hardware_sha256=static_digest(self.hardware))
        for name in ('hardware','fit','beta','baselines','scenarios','lookup','static'):
            setattr(self,name,freeze(getattr(self,name)))
        self._configuration=freeze(configuration)
        object.__setattr__(self,'_sealed',True)

    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False):raise AttributeError('Frozen kernel DAG node clock cannot change')
        object.__setattr__(self,name,value)

    def __delattr__(self,name):raise AttributeError('Frozen kernel DAG node clock cannot change')

    def _reference(self,owner,task,pool,node):
        if not self.allow_response:raise UnknownEvidence('UNVALIDATED_PROFILED_NODE_RESPONSE_OPT_IN_REQUIRED')
        if not self.qualified:raise UnknownEvidence('NUMERICALLY_QUALIFIED_KERNEL_DAG_FIT_REQUIRED')
        profile=self.lookup.get((owner,task,pool))
        key=(owner,profile,node,self.execution_context_sha256,self.profiler_id)
        if profile is None or key not in self.baselines:raise UnknownEvidence('UNKNOWN_EXACT_NODE_REFERENCE:'+repr(key))
        spec=self.scenarios[self.scenario]
        try:
            shift=self.sigma**2/2 if spec['reference_functional']=='mean' else self.sigma*float(ndtri(spec['quantile']))
            result=math.exp(self.baselines[key]+shift)*1e6
        except OverflowError as exc:raise UnknownEvidence('NONFINITE_KERNEL_NODE_REFERENCE') from exc
        if not math.isfinite(result) or result<=0:raise UnknownEvidence('NONFINITE_KERNEL_NODE_REFERENCE')
        return result

    def duration(self,node,row):
        if node['kind']!='gpu':
            if self.non_gpu_duration_provider is None:raise UnknownEvidence('EXPLICIT_NON_GPU_NODE_REFERENCE_DURATION_REQUIRED')
            return self.non_gpu_duration_provider(node,row)
        return self._reference(row['alias'],row['task'],row['pool_id'],node['id'])

    def phase(self,node,row):return node.get('semantic_phase',node['id'])

    def decide(self,view):
        active=list(view.active_gpu_nodes)
        need(len({n.key for n in active})==len(active),'Distinct generated node keys required')
        sources={}
        for node in active:
            self._reference(node.key[0],node.task,node.pool_id,node.key[1])
            key=(node.key[0],node.task,node.pool_id,node.key[1])
            if key not in self.static:raise UnknownEvidence('UNKNOWN_EXACT_KERNEL_DAG_NODE')
            sources[node.key]=self.static[key]
        rates={}
        for node in active:
            state=features.competition_features(sources[node.key],
                [sources[peer.key] for peer in active if peer.key!=node.key],self.hardware)
            if not state['known']:raise UnknownEvidence(state['reason'])
            linear=sum(state['vector'][name]*b for name,b in zip(features.FEATURES,self.beta))
            try:rate=math.exp(-linear)
            except OverflowError as exc:raise UnknownEvidence('NONFINITE_NODE_RESPONSE_RATE') from exc
            if not math.isfinite(rate):raise UnknownEvidence('NONFINITE_NODE_RESPONSE_RATE')
            rates[node.key]=rate
        return SchedulingDecision(rates)

    # All model data are recursively immutable; this provider has no dynamic
    # state. Optional external callbacks retain their own identity checks.
    def fresh(self):return self
    def saved_state(self):return {}
    def restore_state(self,state):need(state=={},'Deterministic kernel node clock has no dynamic state')

    def checkpoint_configuration(self):
        result=plain(self._configuration)
        result['sources']=[dict(ref) for ref in result['sources']]
        # Recheck only the optional callback, never the large frozen catalog.
        # Its code/source/configuration remain part of the checkpoint identity.
        result['non_gpu_duration_provider']=callable_identity(self.non_gpu_duration_provider)
        return result

    def provenance(self):
        return dict(kind='UNVALIDATED_DAG_KERNEL_RESPONSE_HYPOTHESIS',response_domain=DOMAIN,
            features=list(features.FEATURES),same_owner_peer_nodes_included=True,
            rate_equation='d_reference_ns_d_wall_ns=exp(-beta_times_six_covariates)',
            static_occupancy_used_as_throughput=False,serial_training_gate_changed=False,
            native_sources_authenticated=False,target_future_END_read=False,natural_timing_validated=False,
            physical_accuracy_validated=False,deployment_qualified=False)
