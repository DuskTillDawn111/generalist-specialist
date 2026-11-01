"""Node DAG integration with explicit numerical CPU stubs, never fitted evidence."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.resources import identity
from tests.test_kernel_resources import hardware,kernel
from tests.test_dag_node_rate_engine_v11 import Lifecycle,release,start,row
from tools.predictor_validation_20261027 import kernel_node_clock_adapter_v11 as m
from tools.predictor_validation_20261027 import dag_node_rate_engine_v11 as engine
from tools.predictor_validation_20261027 import dag_node_rate_checkpoint_v11 as checkpoint
from tools.predictor_validation_20261027 import kernel_dag_response_training_v1 as training
from tools.predictor_validation_20261027.dag_node_rate_scheduler_v11 import CheckpointNodeScheduler
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import ActiveGpuNode,SchedulingView


def seal(bundle):
    result=copy.deepcopy(bundle);result.pop('bundle_sha256',None)
    result['bundle_sha256']=identity(result)
    return result


def nongpu_duration(node,row):return node['reference_ns']


class MutableCopyLaw:
    def __init__(self):self.scale=1
    def __call__(self,node,row):return node['reference_ns']*self.scale


class KernelNodeClockTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        path=Path(self.temp.name)/'explicit_CPU_stub.json'
        path.write_text('{"synthetic_numerical_law_stub":true,"fitted":false}')
        self.source=checkpoint.base.receipt(path)
        self.document=dict(training_only=True,validation_labels_read=False,execution_context_sha256='a'*64,profiles=[
            dict(alias='a',request_id='profile-a',task='text',pool_id='x',split='training',
                nodes=[dict(id=name,kind='gpu',kernel=dict(kernel(shared=1024),name=name),
                    observed_start_ns=999,observed_end_ns=1000,duration_ns=1,release_ns=999)
                    for name in ('g0','g1')],edges=[],unresolved=[],cyclic=False)])
        self.bundle=seal(dict(schema=m.BUNDLE_SCHEMA,split='training',feature_schema=m.features.SCHEMA,
            features=list(m.features.FEATURES),response_domain=m.DOMAIN,
            execution_context_sha256='a'*64,kernel_catalog_sha256=identity(self.document),
            hardware_id=hardware()['hardware_id'],hardware_sha256=identity(hardware()),
            profiler_id='synthetic-nsys',evidence_kind='cpu_fixture',
            validation_labels_read=False,sources=[self.source],response_fit=dict(
                numerical_qualification=True,features=list(m.features.FEATURES),
                coefficients=[-math.log(2),0,0,0,0,0],lognormal_reference_sigma=.2,
                baselines=[dict(group=['a','profile-a',name,'a'*64,'synthetic-nsys'],
                    log_reference_ms=math.log(work)-.2**2/2) for name,work in [('g0',1),('g1',2)]])))

    def adapter(self,document=None,bundle=None,**kwargs):
        return m.KernelDagNodeClockAdapter(self.document if document is None else document,
            hardware(),self.bundle if bundle is None else bundle,
            allow_unvalidated_profiled_response=True,allow_cpu_fixture=True,**kwargs)

    def view(self,names=('g0','g1')):
        return SchedulingView(0,tuple(ActiveGpuNode(('a',n),1.,n,'fresh-r','text','x') for n in names))

    def consumer(self,provider):
        return engine.DagNodeRateCheckpointModel(
            {'a':dict(family='a',supported_tasks=['text'])},
            {('a','text','x'):self.document['profiles'][0]},
            scheduler=CheckpointNodeScheduler(provider),duration_provider=provider.duration,
            phase_provider=provider.phase,release_provider=release,request_start_provider=start,
            lifecycle_provider=Lifecycle(),checkpoint_binding=dict(
                execution_context_sha256='a'*64,evidence=[self.source]))

    def test_direct_multistream_end_to_end_reprices_and_checkpoints(self):
        provider=self.adapter();rows=[row('fresh-r','a')]
        result=self.consumer(provider).forecast(rows,0,3_000_000,drain_deadline_ns=4_000_000)
        self.assertEqual(result['finishes']['fresh-r']['finish_ns'],1_500_000)
        self.assertEqual(result['max_simultaneous_nodes_per_owner']['a'],2)
        self.assertFalse(result['target_future_END_read'])
        self.assertFalse(result['scheduler_provenance']['provider']['serial_training_gate_changed'])
        for prefix in range(1,result['events']):
            paused=self.consumer(provider).forecast(rows,0,3_000_000,drain_deadline_ns=4_000_000,
                pause_after_event_count=prefix)
            if 'checkpoint' not in paused:continue
            saved=json.loads(json.dumps(paused['checkpoint']))
            restored=self.consumer(provider).forecast(rows,0,3_000_000,drain_deadline_ns=4_000_000,
                checkpoint=saved)
            self.assertEqual(restored,result)

    def test_same_owner_six_covariates_match_pure_training_mapper_and_rate_over_one(self):
        provider=self.adapter();view=self.view()
        original=m.features.competition_features
        with patch.object(m.features,'competition_features',wraps=original) as features:
            decision=provider.decide(view)
        self.assertEqual(features.call_count,2)
        for call in features.call_args_list:
            target,peers,limits=call.args
            self.assertEqual(len(peers),1)
            mapped=original(target,peers,limits)
            self.assertEqual(list(mapped['vector']),list(m.features.FEATURES))
            self.assertEqual(mapped['vector']['peer_kernel_count'],1)
        self.assertEqual(decision.gpu_rates,{('a','g0'):2.,('a','g1'):2.})
        self.assertEqual(provider.decide(self.view(('g1',))).gpu_rates,{('a','g1'):1.})

    def test_reference_mean_and_quantile_are_declared_not_catalog_durations(self):
        provider=self.adapter();node=self.document['profiles'][0]['nodes'][0]
        self.assertEqual(provider.duration(node,row('fresh','a')),1_000_000)
        quantile=self.adapter(scenario='median',scenarios={'median':dict(reference_functional='quantile',quantile=.5)})
        self.assertAlmostEqual(quantile.duration(node,row('fresh','a')),math.exp(-.02)*1e6)
        changed=copy.deepcopy(self.document)
        for node in changed['profiles'][0]['nodes']:
            node.update(observed_start_ns=1,observed_end_ns=10**18,duration_ns=10**18,release_ns=10**18)
        bundle=copy.deepcopy(self.bundle);bundle['kernel_catalog_sha256']=identity(changed)
        other=self.adapter(document=changed,bundle=seal(bundle))
        self.assertEqual(other.decide(self.view()),provider.decide(self.view()))
        self.assertEqual(other.duration(changed['profiles'][0]['nodes'][0],row('fresh','a')),1_000_000)

    def test_immutable_static_catalog_and_small_cached_configuration(self):
        provider=self.adapter()
        with self.assertRaises(AttributeError):provider.beta=(0,)*6
        with self.assertRaises(TypeError):provider.static['a','text','x','g0']['name']='forged'
        with self.assertRaises(TypeError):provider.hardware['registers_per_sm']=1
        with self.assertRaises(AttributeError):del provider.fit
        self.assertIs(provider.fresh(),provider)
        before=provider.checkpoint_configuration()
        with patch.object(m,'static_digest',side_effect=AssertionError('rehash frozen catalog')):
            for _ in range(10):
                self.assertEqual(before,provider.checkpoint_configuration())
                provider.decide(self.view())
        self.assertLess(len(json.dumps(before)),3000)
        self.document['profiles'][0]['nodes'][0]['kernel']['name']='caller-mutated'
        self.assertEqual(provider.static['a','text','x','g0']['name'],'g0')

    def test_non_gpu_callback_bound_and_mutation_detected(self):
        callback=MutableCopyLaw();provider=self.adapter(non_gpu_duration_provider=callback)
        wrapper=CheckpointNodeScheduler(provider);bound=provider.checkpoint_configuration()
        callback.scale=2
        self.assertNotEqual(bound,provider.checkpoint_configuration())
        with self.assertRaisesRegex(ValueError,'configuration changed'):wrapper.decide(self.view())
        explicit=self.adapter(non_gpu_duration_provider=nongpu_duration)
        self.assertEqual(explicit.duration(dict(id='copy',kind='copy',reference_ns=20),row('r','a')),20)
        with self.assertRaisesRegex(UnknownEvidence,'NON_GPU_NODE_REFERENCE'):
            self.adapter().duration(dict(id='copy',kind='copy'),row('r','a'))

    def test_missing_static_reference_and_failed_fit_remain_unknown(self):
        for mode in ('static','reference','failed'):
            doc=copy.deepcopy(self.document);bundle=copy.deepcopy(self.bundle)
            if mode=='static':doc['profiles'][0]['nodes'][0]['kernel'].pop('allocation_evidence')
            elif mode=='reference':bundle['response_fit']['baselines'].pop(0)
            else:bundle['response_fit']=dict(numerical_qualification=False,reason='STRUCTURAL_KERNEL_COVARIATE_RANK_FAILURE')
            bundle['kernel_catalog_sha256']=identity(doc)
            with self.subTest(mode=mode),self.assertRaises(UnknownEvidence):
                self.adapter(document=doc,bundle=seal(bundle)).decide(self.view())

    def test_compiler_actual_rank_failure_bundle_reaches_unknown_without_serial_gate(self):
        from tests.test_kernel_dag_response_training_v1 import fixture
        doc,cap=fixture()
        doc['validation_labels_read']=False
        bundle,prepared=training.training_bundle(doc,hardware(),[cap],allow_cpu_fixture=True)
        self.assertEqual(prepared['accepted_kernel_count'],3)
        self.assertFalse(bundle['response_fit']['numerical_qualification'])
        provider=self.adapter(document=doc,bundle=bundle)
        view=SchedulingView(0,(ActiveGpuNode(('a','x'),1.,'x','fresh','task','input'),))
        with self.assertRaisesRegex(UnknownEvidence,'NUMERICALLY_QUALIFIED'):provider.decide(view)

    def test_bundle_sources_optins_overflow_and_context_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'CPU fixture'):
            m.KernelDagNodeClockAdapter(self.document,hardware(),self.bundle)
        provider=m.KernelDagNodeClockAdapter(self.document,hardware(),self.bundle,allow_cpu_fixture=True)
        with self.assertRaisesRegex(UnknownEvidence,'OPT_IN'):provider.decide(self.view())
        changed=copy.deepcopy(self.bundle);changed['execution_context_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'binding'):self.adapter(bundle=seal(changed))
        changed_hardware=hardware();changed_hardware['max_warps_per_sm']*=2
        with self.assertRaisesRegex(ValueError,'hardware binding'):
            m.KernelDagNodeClockAdapter(self.document,changed_hardware,self.bundle,allow_cpu_fixture=True)
        overflow=copy.deepcopy(self.bundle);overflow['response_fit']['coefficients'][0]=-1000
        with self.assertRaisesRegex(UnknownEvidence,'NONFINITE_NODE_RESPONSE'):
            self.adapter(bundle=seal(overflow)).decide(self.view())
        Path(self.source['path']).write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'source receipts'):self.adapter()

    def test_role_rejection_precedes_deepcopy_or_label_hash(self):
        class ForbiddenBody:
            def __deepcopy__(self,memo):raise AssertionError('foreign role body was copied')
        for mode in ('bundle','catalog','profile','unresolved','cyclic','missing_flag'):
            doc=copy.deepcopy(self.document);bundle=copy.deepcopy(self.bundle)
            bundle['response_fit']=ForbiddenBody()
            if mode=='bundle':bundle['split']='validation'
            elif mode=='catalog':doc['training_only']=False
            elif mode=='profile':doc['profiles'][0]['split']='validation'
            elif mode=='unresolved':doc['profiles'][0]['unresolved']=['foreign']
            elif mode=='cyclic':doc['profiles'][0]['cyclic']=True
            else:doc.pop('validation_labels_read')
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                self.adapter(document=doc,bundle=bundle)

    def test_sealed_foreign_roles_and_invalid_non_gpu_nodes_rejected(self):
        for mode in ('validation_bundle','validation_catalog','validation_profile','unresolved','cyclic',
                     'duplicate_non_gpu','bad_id','bad_kind'):
            doc=copy.deepcopy(self.document);bundle=copy.deepcopy(self.bundle)
            if mode=='validation_bundle':bundle['split']='validation'
            elif mode=='validation_catalog':doc.update(training_only=False,validation_labels_read=True)
            elif mode=='validation_profile':doc['profiles'][0]['split']='validation'
            elif mode=='unresolved':doc['profiles'][0]['unresolved']=['foreign']
            elif mode=='cyclic':doc['profiles'][0]['cyclic']=True
            elif mode=='duplicate_non_gpu':doc['profiles'][0]['nodes']+=[dict(id='x',kind='host_sync')]*2
            elif mode=='bad_id':doc['profiles'][0]['nodes'].append(dict(id=1,kind='copy'))
            else:doc['profiles'][0]['nodes'].append(dict(id='x',kind='unmodeled'))
            bundle['kernel_catalog_sha256']=identity(doc)
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                self.adapter(document=doc,bundle=seal(bundle))


if __name__=='__main__':unittest.main()
