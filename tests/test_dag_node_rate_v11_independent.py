"""Independent synthetic v11 adversarial/event integrations; never GPU or fits."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.resources import identity
from tests.test_kernel_resources import hardware,kernel
from tools.predictor_validation_20261027 import kernel_node_clock_adapter_v11 as adapter
from tools.predictor_validation_20261027 import dag_node_rate_engine_v11 as engine
from tools.predictor_validation_20261027 import dag_node_rate_checkpoint_v11 as cp
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import ActiveGpuNode,SchedulingView,SchedulingDecision
from tools.predictor_validation_20261027.dag_node_rate_scheduler_v11 import CheckpointNodeScheduler,validate_decision

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/predictor-node-rate-dag-v11-20261029-cpu01-independent/source-before-role-fix/kernel_node_clock_adapter_v11.py'


def seal(value):
    out=copy.deepcopy(value);out.pop('bundle_sha256',None);out['bundle_sha256']=identity(out);return out


def row(rid,alias,arrival=0,**kw):return dict(request_id=rid,alias=alias,arrival_ns=arrival,task='task',pool_id='input',**kw)

def duration(node,row):return node['reference_ns']
def phase(node,row):return node['id']
def start(row):return row.get('host_start_ns',row['arrival_ns'])
def release(node,row):return node.get('supply',0)
def copy_rate(node,row,clock,active):return dict(rate=.25,next_transition_ns=None)


class Lifecycle:
    def delays(self,row,aliases,scenario):return dict(known=True,parent_receipt_ms=0.,worker_next_ready_ms=0.)


class Law:
    def __init__(self,mode='frontier'):self.mode=mode;self.history=[]
    def decide(self,view):
        self.history.append([view.clock_ns,[(n.key,n.remaining_reference_work_ns) for n in view.active_gpu_nodes]])
        rate=(.5 if len(view.active_gpu_nodes)>1 else 2.) if self.mode=='frontier' else 2.
        return SchedulingDecision({n.key:rate for n in view.active_gpu_nodes})
    def fresh(self):return Law(self.mode)
    def checkpoint_configuration(self):return dict(kind='INDEPENDENT_CPU_LAW',mode=self.mode)
    def saved_state(self):return dict(history=copy.deepcopy(self.history))
    def restore_state(self,state):self.history=copy.deepcopy(state['history'])
    def provenance(self):return dict(kind='INDEPENDENT_CPU_LAW',history=copy.deepcopy(self.history))


def plan(work,kind='gpu'):
    return dict(nodes=[dict(id='node',kind=kind,reference_ns=work)],edges=[],unresolved=[],cyclic=False)


class IndependentNodeRateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.evidence_path=Path(self.temp.name)/'evidence.json';self.evidence_path.write_text('{"independent_CPU_only":true}')
        self.evidence=cp.base.receipt(self.evidence_path)

    def engine(self,plans,mode='frontier',non_gpu=None):
        aliases={key[0]:dict(family=key[0],supported_tasks=['task']) for key in plans}
        return engine.DagNodeRateCheckpointModel(aliases,plans,scheduler=CheckpointNodeScheduler(Law(mode)),
            duration_provider=duration,phase_provider=phase,release_provider=release,request_start_provider=start,
            lifecycle_provider=Lifecycle(),non_gpu_rate_provider=non_gpu,
            allow_independent_copy_hypothesis=non_gpu is not None,
            checkpoint_binding=dict(execution_context_sha256='a'*64,evidence=[self.evidence]))

    def document_and_bundle(self):
        doc=dict(training_only=True,validation_labels_read=False,execution_context_sha256='a'*64,
            profiles=[dict(alias='owner',request_id='profile',task='task',pool_id='input',split='training',
                nodes=[dict(id=name,kind='gpu',kernel=dict(kernel(shared=1024),name=name),
                    observed_start_ns=99,observed_end_ns=100,duration_ns=1,release_ns=999) for name in ('n0','n1')],
                edges=[],unresolved=[],cyclic=False)])
        bundle=seal(dict(schema=adapter.BUNDLE_SCHEMA,split='training',feature_schema=adapter.features.SCHEMA,
            features=list(adapter.features.FEATURES),response_domain=adapter.DOMAIN,execution_context_sha256='a'*64,
            kernel_catalog_sha256=identity(doc),hardware_id=hardware()['hardware_id'],hardware_sha256=identity(hardware()),
            profiler_id='independent-CPU',evidence_kind='cpu_fixture',validation_labels_read=False,sources=[self.evidence],
            response_fit=dict(numerical_qualification=True,features=list(adapter.features.FEATURES),
                coefficients=[-math.log(2),0,0,0,0,0],lognormal_reference_sigma=.2,
                baselines=[dict(group=['owner','profile',name,'a'*64,'independent-CPU'],
                    log_reference_ms=math.log(w)-.02) for name,w in [('n0',1.),('n1',2.)]])))
        return doc,bundle

    def provider(self,doc=None,bundle=None,module=adapter):
        default_doc,default_bundle=self.document_and_bundle()
        return module.KernelDagNodeClockAdapter(default_doc if doc is None else doc,hardware(),
            default_bundle if bundle is None else bundle,allow_cpu_fixture=True,allow_unvalidated_profiled_response=True)

    def view(self):return SchedulingView(0,tuple(ActiveGpuNode(('owner',n),1.,n,'fresh','task','input') for n in ('n0','n1')))

    def test_actual_same_owner_two_streams_reprice_all_remaining_work(self):
        p=plan(8);p['nodes'].append(dict(id='other-stream',kind='gpu',reference_ns=12))
        result=self.engine({('a','task','input'):p}).forecast([row('r','a')],0,30,drain_deadline_ns=40)
        self.assertEqual(result['finishes']['r']['finish_ns'],18)
        self.assertEqual(result['max_simultaneous_nodes_per_owner']['a'],2)
        self.assertIn([16,[(('a','other-stream'),4.)]],result['scheduler_provenance']['provider']['history'])
        self.assertEqual(result['reference_work_accounting']['r','other-stream'],dict(reference_ns=12.,consumed_ns=12.,remaining_ns=0.))

    def test_actual_future_arrival_reprices_inflight_nodes_from_remaining_not_original_work(self):
        result=self.engine({('a','task','input'):plan(12),('b','task','input'):plan(4)}).forecast(
            [row('first','a'),row('later','b',4)],0,20,drain_deadline_ns=30)
        self.assertEqual({k:v['finish_ns'] for k,v in result['finishes'].items()},{'first':12,'later':12})
        self.assertIn([4,[(('a','node'),4.),(('b','node'),4.)]],result['scheduler_provenance']['provider']['history'])

    def test_HOL_future_host_start_holds_later_request_without_global_admission_cap(self):
        plans={('a','task','input'):plan(2),('b','task','input'):plan(40),('c','task','input'):plan(40)}
        rows=[row('a0','a'),row('blocked','a',1,host_start_ns=20),row('behind','a',2),row('b','b'),row('c','c')]
        result=self.engine(plans,mode='constant').forecast(rows,0,40,drain_deadline_ns=50)
        self.assertEqual(result['starts'],{'a0':0,'b':0,'c':0,'blocked':20,'behind':21})
        self.assertEqual(result['queue_wait_ns']['behind'],19)
        self.assertEqual(result['max_simultaneous_activity_nodes'],3)

    def test_checkpoint_every_prefix_keeps_future_arrival_and_provider_state(self):
        plans={('a','task','input'):plan(12),('b','task','input'):plan(4)}
        rows=[row('first','a'),row('later','b',4),row('queued','a',5)]
        full=self.engine(plans).forecast(rows,0,40,drain_deadline_ns=50)
        pauses=0
        for n in range(1,full['events']):
            paused=self.engine(plans).forecast(rows,0,40,drain_deadline_ns=50,pause_after_event_count=n)
            if 'checkpoint' not in paused:continue
            pauses+=1;saved=json.loads(json.dumps(paused['checkpoint']))
            restored=self.engine(plans).forecast(rows,0,40,drain_deadline_ns=50,checkpoint=saved)
            self.assertEqual(restored,full)
        self.assertGreater(pauses,1)

    def test_resigned_conserving_work_and_provider_history_forgery_fail_prefix_replay(self):
        plans={('a','task','input'):plan(12),('b','task','input'):plan(4)};rows=[row('r','a'),row('b','b',4)]
        saved=self.engine(plans).forecast(rows,0,30,drain_deadline_ns=40,pause_after_event_count=1)['checkpoint']
        for field in ('state','scheduler_state'):
            bad=copy.deepcopy(saved);state=cp.base.decode(bad[field])
            if field=='state':
                work=state['node_work']['r','node'];work['consumed_ns']-=1;work['remaining_ns']+=1
                state['active']['a','node']['remaining_ns']+=1
            else:state['provider_state']['history'].append([999,[]])
            bad[field]=cp.base.encode(state);bad['checkpoint_sha256']=cp.base.digest({k:v for k,v in bad.items() if k!='checkpoint_sha256'})
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'prefix replay'):
                self.engine(plans).forecast(rows,0,30,drain_deadline_ns=40,checkpoint=bad)

    def test_copy_enable_does_not_supply_duration_or_rate_numbers(self):
        provider=self.provider()
        with self.assertRaisesRegex(UnknownEvidence,'NON_GPU_NODE_REFERENCE'):
            provider.duration(dict(id='copy',kind='copy'),row('r','owner'))
        plans={('a','task','input'):plan(8,kind='copy')};consumer=self.engine(plans)
        consumer.allow_independent_copy_hypothesis=True
        with self.assertRaisesRegex(UnknownEvidence,'COPY_MEMSET_REFERENCE_RATE'):
            consumer.forecast([row('r','a')],0,50,drain_deadline_ns=60)
        known=self.engine(plans,non_gpu=copy_rate).forecast([row('r','a')],0,50,drain_deadline_ns=60)
        self.assertEqual(known['finishes']['r']['finish_ns'],32)

    def test_node_rate_gt_one_allowed_but_bool_nan_inf_missing_key_rejected(self):
        view=self.view();rates={n.key:2. for n in view.active_gpu_nodes}
        self.assertEqual(validate_decision(view,SchedulingDecision(rates))[0],rates)
        for value in (True,float('nan'),float('inf'),-1.):
            bad=copy.deepcopy(rates);bad[('owner','n0')]=value
            with self.subTest(value=value),self.assertRaises(UnknownEvidence):validate_decision(view,SchedulingDecision(bad))
        with self.assertRaises(UnknownEvidence):validate_decision(view,SchedulingDecision({('owner','n0'):1.}))

    def test_observed_future_end_duration_and_release_never_become_numeric_clock(self):
        doc,bundle=self.document_and_bundle();base=self.provider(doc,bundle)
        changed=copy.deepcopy(doc)
        for node in changed['profiles'][0]['nodes']:
            node.update(observed_start_ns=-10**100,observed_end_ns={'FORBIDDEN_FUTURE_END':True},duration_ns=10**100,release_ns=10**100)
        other=copy.deepcopy(bundle);other['kernel_catalog_sha256']=identity(changed)
        provider=self.provider(changed,seal(other))
        self.assertEqual(provider.decide(self.view()),base.decide(self.view()))
        self.assertEqual(provider.duration(changed['profiles'][0]['nodes'][0],row('r','owner')),1_000_000.)

    def test_actual_kernel_adapter_to_engine_multistream_and_pause_resume(self):
        doc,bundle=self.document_and_bundle();provider=self.provider(doc,bundle)
        plans={('owner','task','input'):doc['profiles'][0]}
        def make():
            return engine.DagNodeRateCheckpointModel(
                {'owner':dict(family='owner',supported_tasks=['task'])},plans,
                scheduler=CheckpointNodeScheduler(provider),duration_provider=provider.duration,
                phase_provider=provider.phase,release_provider=release,request_start_provider=start,
                lifecycle_provider=Lifecycle(),checkpoint_binding=dict(execution_context_sha256='a'*64,evidence=[self.evidence]))
        rows=[row('fresh','owner')]
        full=make().forecast(rows,0,3_000_000,drain_deadline_ns=4_000_000)
        self.assertEqual(full['finishes']['fresh']['finish_ns'],1_500_000)
        self.assertEqual(full['max_simultaneous_nodes_per_owner']['owner'],2)
        paused=make().forecast(rows,0,3_000_000,drain_deadline_ns=4_000_000,pause_after_event_count=1)
        restored=make().forecast(rows,0,3_000_000,drain_deadline_ns=4_000_000,
            checkpoint=json.loads(json.dumps(paused['checkpoint'])))
        self.assertEqual(restored,full)
        self.assertFalse(full['native_sources_authenticated']);self.assertFalse(full['model_fit'])

    def test_foreign_role_rejected_before_copy_hash_or_numeric_reads(self):
        class ForbiddenLabels:
            def __deepcopy__(self,memo):raise AssertionError('FOREIGN_LABEL_BODY_COPIED')
        for kind in ('bundle_split','catalog_validation','profile_split'):
            doc,bundle=self.mutate_role(kind);doc['DO_NOT_READ_LABELS']=ForbiddenLabels()
            with self.subTest(kind=kind),patch.object(adapter,'identity',side_effect=AssertionError('FOREIGN_LABEL_HASH')):
                with self.assertRaises(ValueError):self.provider(doc,bundle)

    def mutate_role(self,kind):
        doc,bundle=self.document_and_bundle()
        if kind=='bundle_split':bundle['split']='validation'
        elif kind=='catalog_training':doc['training_only']=False
        elif kind=='catalog_validation':doc['validation_labels_read']=True
        elif kind=='profile_split':doc['profiles'][0]['split']='validation'
        elif kind=='unresolved':doc['profiles'][0]['unresolved']=['unresolved edge']
        elif kind=='cyclic':doc['profiles'][0]['cyclic']=True
        else:raise ValueError(kind)
        bundle['kernel_catalog_sha256']=identity(doc)
        return doc,seal(bundle)

    def test_old_snapshot_reproduces_all_six_role_DAG_gaps_without_changing_frozen_source(self):
        module=types.ModuleType('tools.predictor_validation_20261027._independent_preserved_adapter')
        module.__package__='tools.predictor_validation_20261027';module.__file__=str(OLD)
        exec(compile(OLD.read_text(),str(OLD),'exec'),module.__dict__)
        for kind in ('bundle_split','catalog_training','catalog_validation','profile_split','unresolved','cyclic'):
            doc,bundle=self.mutate_role(kind)
            self.assertEqual(self.provider(doc,bundle,module).decide(self.view()).gpu_rates,{('owner','n0'):2.,('owner','n1'):2.})

    def test_current_consumer_rejects_all_six_role_DAG_gaps_before_numeric_or_hash(self):
        for kind in ('bundle_split','catalog_training','catalog_validation','profile_split','unresolved','cyclic'):
            doc,bundle=self.mutate_role(kind)
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.provider(doc,bundle)


if __name__=='__main__':unittest.main()
