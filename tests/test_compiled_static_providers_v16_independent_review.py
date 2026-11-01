"""Independent branched-graph and exact host-release equivalence checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.compiled_static_providers_v16 import CompiledStaticProviders
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import DagPhaseFullHorizonModel,SchedulingDecision
from tools.predictor_validation_20261027.native_collect import receipt
from tools.predictor_validation_20261027 import learned_availability_forecast_v16 as adapter

METRICS = {}


class Scheduler:
    def decide(self,view):
        return SchedulingDecision({node.key:(.5 if node.phase=='prefill' else 1.) for node in view.active_gpu_nodes},view.clock_ns+7)
    def provenance(self):return dict(kind='independent_test_scheduler')


class Lifecycle:
    def delays(self,*args):return dict(known=True,parent_receipt_ms=.000003,worker_next_ready_ms=.000005)


class Resources:
    def power(self,handlers,*args):return dict(known=True,watts=9+len(handlers))
    def memory(self,handlers,*args):return dict(known=True,peak_memory_mib=20+len(handlers))


class IndependentCompiledReview(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.source=Path(self.temp.name)/'original.json';self.source.write_text('{"source":1}')
        self.aliases={a:dict(supported_tasks=['text'],family='text') for a in ('a','b')}
        self.plans={};self.calls=[]
        for owner in self.aliases:
            for pool in ('short','long'):
                nodes=[dict(id=n,kind=k,release_ns=987654,extra_source_metadata=['original',owner,pool,n])
                       for n,k in [('sync','host_sync'),('copy','copy'),('record','event_record'),
                                   ('g1','gpu'),('g2','gpu'),('memset','memset'),
                                   ('wait','event_wait'),('terminal','handler_end')]]
                edges=[('sync','copy'),('copy','record'),('record','g1'),('record','g2'),
                       ('copy','memset'),('g1','wait'),('g2','wait'),('memset','wait'),('wait','terminal')]
                self.plans[owner,'text',pool]=dict(nodes=nodes,edges=[dict(before=a,after=b,reason='original') for a,b in edges],
                    cyclic=False,unresolved=[],default_stream_semantics='UNMEASURED_EXPLICIT_HYPOTHESIS')

    def providers(self):
        def original(node,row):
            expected=next(n for n in self.plans[row['alias'],row['task'],row['pool_id']]['nodes'] if n['id']==node['id'])
            self.assertEqual(node,expected)
        def duration(node,row):
            original(node,row);self.calls.append(('duration',row['alias'],row['pool_id'],node['id']))
            return (5 if row['pool_id']=='short' else 9)+(2 if node['id']=='g2' else 0)
        def phase(node,row):
            original(node,row);self.calls.append(('phase',row['alias'],row['pool_id'],node['id']))
            return 'prefill' if node['id']=='g1' else 'decode'
        def release(node,row):
            original(node,row);self.calls.append(('release',row['alias'],row['pool_id'],node['id']))
            if node['id']=='terminal':return dict(not_before_offset_ns=0,after_predecessors_ns=13)
            if node['id']=='g2':return dict(not_before_offset_ns=6,after_predecessors_ns=5)
            if node['id']=='sync':return 2
            return dict(not_before_offset_ns=3,after_predecessors_ns=0)
        return duration,release,phase

    def compiled(self):
        d,r,p=self.providers()
        return CompiledStaticProviders(self.aliases,self.plans,duration=d,release=r,phase=p,sources=[receipt(self.source)])

    def engine(self,compiled=None,*,implicit=True):
        d,r,p=self.providers()
        return DagPhaseFullHorizonModel(self.aliases if compiled is None else compiled.graph.aliases,
            self.plans if compiled is None else compiled.graph.plans,
            duration_provider=d if compiled is None else compiled.duration,
            release_provider=r if compiled is None else compiled.release,
            phase_provider=p if compiled is None else compiled.phase,
            scheduler=Scheduler(),lifecycle_provider=Lifecycle(),power_provider=Resources(),memory_provider=Resources(),
            require_energy=True,require_memory=True,allow_independent_copy_hypothesis=True,
            allow_no_implicit_default_stream_hypothesis=implicit)

    def test_branched_parallel_all_node_kinds_and_integer_release_full_trajectories(self):
        original=copy.deepcopy(self.plans);compiled=self.compiled()
        self.assertEqual(len([c for c in self.calls if c[0]=='release']),32)
        self.assertEqual(len([c for c in self.calls if c[0]=='duration']),16)
        self.assertEqual(len([c for c in self.calls if c[0]=='phase']),8)
        rows=[dict(alias=a,task='text',pool_id='short' if i%2==0 else 'long',
                   request_id=a+str(i),arrival_ns=100+t,public_unused_value=i)
              for i,t in enumerate((0,0,8,19,48)) for a in self.aliases]
        compared=0;completed=None
        for drain in (105,120,145,200,1000):
            left=self.engine().forecast(rows,100,150,drain_deadline_ns=drain)
            right=self.engine(compiled).forecast(rows,100,150,drain_deadline_ns=drain)
            self.assertEqual(left,right);compared+=1
            self.assertEqual(len(left['cohort']),10)
            if drain==1000:completed=right
        self.assertEqual(self.plans,original)
        self.assertEqual(len(completed['finishes']),10)
        self.assertEqual(completed['completed_dag_nodes'],80)
        # Independent critical path for the first short request: sync2,
        # copy[3,8], record8, g1[8,18], g2[13,20], memset[8,13],
        # wait20, terminal33, parent receipt36 and next ready38.
        self.assertEqual(completed['finishes']['a0']['handler_end_ns'],133)
        self.assertEqual(completed['finishes']['a0']['finish_ns'],136)
        self.assertEqual(completed['finishes']['a0']['worker_next_ready_ns'],138)
        METRICS['branched_trajectories']=dict(exact_complete_forecasts=compared,requests_each=10,
            original_nodes_each_template=8,original_edges_each_template=9,templates=4,
            all_node_kinds=['host_sync','copy','event_record','gpu','memset','event_wait','handler_end'],
            completed_nodes_full=80,integer_and_dictionary_host_release_preserved=True,
            independent_first_handler_end_ns=133,independent_parent_receipt_ns=136,
            independent_worker_ready_ns=138,original_plans_unchanged=True)

    def test_original_profile_digest_and_nested_immutable_graph_binding(self):
        compiled=self.compiled()
        for coordinate,profile in self.plans.items():
            expected=hashlib.sha256(json.dumps(profile,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            frozen=compiled.graph.plans[coordinate]
            self.assertEqual(frozen['original_training_profile_sha256'],expected)
            self.assertEqual([(n['id'],n['kind']) for n in frozen['nodes']],[(n['id'],n['kind']) for n in profile['nodes']])
            self.assertEqual([(e['before'],e['after']) for e in frozen['edges']],[(e['before'],e['after']) for e in profile['edges']])
            with self.assertRaises(TypeError):frozen['edges'][0]['before']='different'
            with self.assertRaises(TypeError):frozen['nodes'][0]['kind']='gpu'
        row=dict(alias='a',task='text',pool_id='short',arrival_ns=999,request_id='any')
        release=compiled.release(dict(id='terminal'),row);release['after_predecessors_ns']=999
        self.assertEqual(compiled.release(dict(id='terminal'),row)['after_predecessors_ns'],13)
        self.source.write_text('{"source":2}')
        with self.assertRaisesRegex(ValueError,'source changed'):compiled.assert_sources_unchanged()
        METRICS['graph_binding']=dict(all_four_original_profile_hashes_independently_rebuilt=True,
            node_order_kind_and_all_edge_endpoints_exact=True,nested_mutation_rejected=True,
            returned_release_mutation_isolated=True,source_change_rejected=True)

    def test_default_stream_unresolved_and_cyclic_failures_unchanged(self):
        rows=[dict(alias='a',task='text',pool_id='short',arrival_ns=0,request_id='r')]
        for kind in ('default_stream','cyclic','unresolved'):
            with self.subTest(kind=kind):
                if kind=='cyclic':self.plans['a','text','short']['cyclic']=True
                if kind=='unresolved':
                    self.plans['a','text','short']['cyclic']=False
                    self.plans['a','text','short']['unresolved']=['unknown_dependency']
                compiled=self.compiled();messages=[]
                for representation in (None,compiled):
                    with self.assertRaises(UnknownEvidence) as caught:
                        self.engine(representation,implicit=kind!='default_stream').forecast(rows,0,100,drain_deadline_ns=200)
                    messages.append(str(caught.exception))
                self.assertEqual(*messages)
        METRICS['fail_closed_semantics']=dict(original_and_compiled_same_failure_for=['default_stream_opt_in','cyclic','unresolved'])

    def test_compilation_provenance_is_immutable_and_diagnostics_are_detached(self):
        compiled=self.compiled()
        with self.assertRaises(TypeError):compiled.sources[0]['path']='other'
        with self.assertRaises(TypeError):compiled.manifest[0]['original_node_count']=0
        before=compiled.diagnostic()
        changed=compiled.diagnostic()
        changed['original_graph_compilation'][0]['original_node_count']=0
        changed['duration']['sources'][0]['path']='other'
        self.assertEqual(compiled.diagnostic(),before)
        METRICS['immutable_compilation_provenance']=dict(
            nested_source_and_manifest_mutations_rejected=True,diagnostics_detached_from_compilation=True)

    def test_wrapper_rejects_request_dependent_types_and_binds_execution_dependencies(self):
        # Avoid rebuilding training providers: an empty exact-type fixture
        # exercises only adapter admission and dependency-receipt construction.
        source=receipt(self.source)
        def parent_init(obj,*args,**kw):
            obj.aliases={};obj.plans={};obj.host={}
            obj.work=object.__new__(adapter.SoloOrdinalReferenceWork)
            obj.work.sources=[source];obj.work.index={};obj.work.groups=[]
            obj.phases=object.__new__(adapter.StaticDagPhaseBinding)
            obj.phases.sources=[source];obj.phases.bindings={};obj.phases.node_count=0
            obj.release=object.__new__(adapter.DagPreparationHostReleaseProvider)
        with patch.object(adapter.V15Forecast,'__init__',parent_init):
            model=adapter.LearnedAvailabilityForecast({},allow_unvalidated_training_hypothesis=True)
        paths={str(Path(ref['path']).resolve()) for ref in model.compiled.sources}
        expected=['compiled_static_providers_v16','learned_availability_forecast_v16',
                  'dag_frozen_model_v10','native_queue_phase_busy_v10','dag_phase_scheduler_v8',
                  'solo_reference_work_v8','static_dag_phase_binding_v8','dag_preparation_host_release_v8']
        for module in expected:
            self.assertIn(str(Path(adapter.__file__).with_name(module+'.py').resolve()),paths)
        def request_dependent_parent(obj,*args,**kw):
            parent_init(obj,*args,**kw)
            obj.work=lambda node,row:row['arrival_ns']+1
        with patch.object(adapter.V15Forecast,'__init__',request_dependent_parent):
            with self.assertRaisesRegex(ValueError,'coordinate-only'):
                adapter.LearnedAvailabilityForecast({},allow_unvalidated_training_hypothesis=True)
        METRICS['wrapper_source_admission']=dict(execution_dependency_modules_explicitly_bound=expected,
            arbitrary_request_dependent_provider_type_rejected=True,
            fixture_scope='Only adapter admission/source collection; empty plans avoid any fit or source extraction.')


if __name__=='__main__':unittest.main()
