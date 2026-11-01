"""Strict graph/work/static equivalence checks with synthetic CPU declarations."""
import copy
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import kernel_dag_capture_alignment_v1 as m


class CaptureAlignmentTests(unittest.TestCase):
    def role_fixture(self):
        entry={key:{'path':key,'sha256':key} for key in ('profiles','review','mapping')}
        strict=dict(training_only=True,validation_started=False,raw={'path':'raw'},plan={'path':'plan'})
        plan=dict(training_only=True);raw=dict(request_records=[dict(split='training')])
        adapter=dict(schema=m.SCHEMA+'.catalog_role',training_only=True,validation_labels_read=False,
            catalog=entry['profiles'],strict_review=entry['review'],mapping=entry['mapping'],
            raw=strict['raw'],plan=strict['plan'],
            basis='EXPLICIT_TRAINING_NATIVE_REVIEW_PLAN_AND_COMPLETE_REQUEST_SPLITS')
        return entry,strict,plan,raw,adapter

    def test_missing_role_requires_adapter_and_adapter_does_not_mutate_catalog(self):
        entry,strict,plan,raw,adapter=self.role_fixture();catalog=dict(training_only=True)
        with self.assertRaisesRegex(ValueError,'requires source-bound adapter'):
            m.catalog_role(catalog,entry,strict,plan,raw)
        entry['catalog_role']={'path':'adapter'}
        with patch.object(m,'checked',return_value=adapter):
            result=m.catalog_role(catalog,entry,strict,plan,raw)
        self.assertIs(result['validation_labels_read'],False)
        self.assertNotIn('validation_labels_read',catalog)

    def test_adapter_never_overrides_read_labels_or_mismatched_sources(self):
        entry,strict,plan,raw,adapter=self.role_fixture();entry['catalog_role']={'path':'adapter'}
        for field in ('catalog','strict_review','mapping','raw','plan','validation_labels_read','training_only','basis'):
            broken=dict(adapter);broken[field]='incorrect'
            with self.subTest(field=field),patch.object(m,'checked',return_value=broken),self.assertRaises(ValueError):
                m.catalog_role(dict(training_only=True),entry,strict,plan,raw)
        for value in (True,None,0,'false'):
            with self.subTest(role=value),patch.object(m,'checked',side_effect=AssertionError('adapter must not override')):
                with self.assertRaises(ValueError):m.catalog_role(dict(training_only=True,validation_labels_read=value),entry,strict,plan,raw)

    def test_native_roles_must_be_explicit_before_catalog_adaptation(self):
        entry,strict,plan,raw,adapter=self.role_fixture()
        for mode in ('strict','validation','plan','split'):
            s,p,r=copy.deepcopy((strict,plan,raw))
            if mode=='strict':s.pop('training_only')
            if mode=='validation':s.pop('validation_started')
            if mode=='plan':p['training_only']=False
            if mode=='split':r['request_records'][0]['split']='validation'
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                m.catalog_role(dict(training_only=True,validation_labels_read=False),entry,s,p,r)

    def test_load_capture_rejects_missing_or_read_role_before_deepcopy(self):
        entry,strict,plan,raw,_=self.role_fixture()
        mapping=dict(schema='gsdeploy.kernel_dag_training_mapping_audit.v1',status='PASS_CPU_MAPPING_ONLY',
                     issues=[],sources=[entry['profiles']],raw=strict['raw'])
        for catalog in (dict(training_only=True),dict(training_only=True,validation_labels_read=True)):
            with self.subTest(catalog=catalog),patch.object(m,'load_report',return_value=(strict,plan,raw)),\
                 patch.object(m,'receipt',return_value=entry['review']),\
                 patch.object(m,'checked',side_effect=[mapping,catalog]),\
                 patch.object(m.copy,'deepcopy',side_effect=AssertionError('profile read before role gate')) as clone:
                with self.assertRaises(ValueError):m.load_capture(entry)
                clone.assert_not_called()

    def fixture(self):
        kernel=dict(name='same',hardware_id='gpu',block=[32,1,1],grid=[2,1,1],registers_per_thread=8,
            static_shared_bytes=0,dynamic_shared_bytes=0,allocated_registers_per_block=256,
            allocated_shared_bytes=1024,architecture_blocks_per_sm_ceiling=24,cluster=None,
            cooperative_launch=False,static_evidence={'native':'first'},allocation_evidence={'native':'first'},
            nsys_start_ns=20,nsys_end_ns=40,pid=1,stream=7,correlation_id=4,device=0,context_id=1)
        nodes=[];items=[]
        for index,kind in enumerate(('gpu','event_record','event_wait','gpu')):
            node=dict(id='native:'+str(index),kind=kind,issue_start_ns=10*index+1,
                issue_end_ns=10*index+2,issue_thread=1,stream_key=[1,0,1,7 if index<2 else 13])
            if kind=='gpu':
                k=dict(kernel,correlation_id=index,stream=node['stream_key'][-1],
                       nsys_start_ns=20+index*20,nsys_end_ns=30+index*20)
                node['kernel']=k
                items.append(dict(kernel=k,signature={'name':'same','stream_role':len(items)},
                                  group='__outside_language_forwards__',operator_id=None))
            nodes.append(node)
        nodes.append(dict(id='end',kind='handler_end'))
        edges=[dict(before=nodes[i]['id'],after=nodes[i+1]['id'],kind='dependency') for i in range(4)]
        profile=dict(nodes=nodes,edges=edges,unresolved=[],cyclic=False,
            pre_request_external_dependencies_assumed_completed=1,
            default_stream_semantics='UNMEASURED_NO_IMPLICIT_LEGACY_DEFAULT_STREAM_EDGES_ADDED')
        binding=dict(alias='ocr',task='ocr',pool_id='exact',exact_input_sha256='a'*64,
                     actual_work={'outputs':{'raw_output':'same actual OCR'},'forwards':[]})
        return profile,items,binding

    def test_native_ids_clocks_pids_stream_ids_and_evidence_are_separate_from_common_template(self):
        first=self.fixture();second=copy.deepcopy(first);profile,items,_=second
        renamed={n['id']:'new:'+n['id'] for n in profile['nodes']}
        for node in profile['nodes']:
            node['id']=renamed[node['id']]
            if 'issue_start_ns' in node:
                node['issue_start_ns']+=1000;node['issue_end_ns']+=1000;node['issue_thread']=999
                node['stream_key']=[999,0,55,node['stream_key'][-1]+100]
            if 'kernel' in node:
                node['kernel'].update(pid=999,context_id=55,stream=node['kernel']['stream']+100,
                    nsys_start_ns=node['kernel']['nsys_start_ns']+1000,nsys_end_ns=node['kernel']['nsys_end_ns']+1000,
                    static_evidence={'native':'second'},allocation_evidence={'native':'second'})
        for edge in profile['edges']:
            edge['before']=renamed[edge['before']];edge['after']=renamed[edge['after']]
        left,lmap,_=m.template(*first);right,rmap,_=m.template(*second)
        self.assertEqual(left,right);self.assertNotEqual(lmap,rmap)

    def test_same_names_never_override_static_configuration_or_dependencies(self):
        first=self.fixture();expected=m.template(*first)[0]
        for mode in ('grid','allocation','edge','stream','group','work'):
            profile,items,binding=copy.deepcopy(first)
            if mode=='grid':profile['nodes'][0]['kernel']['grid']=[3,1,1]
            if mode=='allocation':profile['nodes'][0]['kernel']['allocated_registers_per_block']=512
            if mode=='edge':profile['edges'][0]['kind']='different-dependency'
            if mode=='stream':profile['nodes'][2]['stream_key'][-1]=99
            if mode=='group':items[0]['group']='different-semantic-work'
            if mode=='work':binding['actual_work']['outputs']['raw_output']='different OCR work'
            with self.subTest(mode=mode):self.assertNotEqual(m.template(profile,items,binding)[0],expected)

    def test_missing_or_duplicate_native_kernels_and_unproven_allocations_rejected(self):
        for mode in ('missing','duplicate','static'):
            profile,items,binding=self.fixture()
            if mode=='missing':items.pop()
            if mode=='duplicate':items.append(items[0])
            if mode=='static':profile['nodes'][0]['kernel']['allocation_evidence']=None
            with self.subTest(mode=mode),self.assertRaises(ValueError):m.template(profile,items,binding)

    def test_tied_or_concurrent_host_order_and_incomplete_graph_rejected(self):
        for mode in ('tie','overlap','unresolved','cycle','foreign-edge'):
            profile,items,binding=self.fixture()
            if mode=='tie':profile['nodes'][1]['issue_start_ns']=1
            if mode=='overlap':profile['nodes'][0]['issue_end_ns']=20;profile['nodes'][1]['issue_thread']=2
            if mode=='unresolved':profile['unresolved']=[{}]
            if mode=='cycle':profile['cyclic']=True
            if mode=='foreign-edge':profile['edges'].append(dict(before='missing',after='end',kind='bad'))
            with self.subTest(mode=mode),self.assertRaises((ValueError,KeyError)):m.template(profile,items,binding)

    def test_same_owner_ocr_different_exact_input_or_output_does_not_share_work_identity(self):
        row=dict(model='ocr',task='ocr',pool_id='p',worker_response=dict(success=True,raw_output='x',prediction='x'))
        plan=dict(inputs=[dict(pool_id='p',task='ocr',exact_input_sha256='a'*64)])
        first=m.request_binding(row,plan)
        changed=copy.deepcopy(plan);changed['inputs'][0]['exact_input_sha256']='b'*64
        self.assertNotEqual(first,m.request_binding(row,changed))
        row['worker_response']['raw_output']='different'
        self.assertNotEqual(first,m.request_binding(row,plan))


if __name__=='__main__':unittest.main()
