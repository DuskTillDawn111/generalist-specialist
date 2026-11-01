"""CPU-only native matrix/lifecycle and complete-instance negative checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import kernel_dag_response_replication_producer_v1 as m
from tools.predictor_validation_20261027 import kernel_dag_repeated_instance_alignment_v1 as adapter
from tools.predictor_validation_20261027.native_collect import receipt
from tests.test_kernel_resources import hardware,kernel
from gsdeploy.resources import identity


class ReplicationProducerTests(unittest.TestCase):
    def fixture(self):
        bindings={owner:dict(pool_id='exact-'+owner,task='ocr' if owner=='ocr#0' else 'text',exact_input_sha256=str(i)*64)
                  for i,owner in enumerate(('text#0','text#1','generalist_2#0','ocr#0'))}
        jobs=m.jobs_for(bindings)
        plan=dict(training_only=True,split='training',domain=m.collector.LIGHT,no_retries=True,jobs=jobs,
                  inputs=copy.deepcopy(list(bindings.values())),execution_context_sha256='ctx')
        rows=[];batches=[]
        for job in jobs:
            batch=dict(job_id=job['id'],request_ids=[])
            for owner,pool in job['inputs'].items():
                rid='request:'+str(len(rows));batch['request_ids'].append(rid)
                rows.append(dict(request_id=rid,profile_job_id=job['id'],model=owner,pool_id=pool,
                    task=job['tasks'][owner],split='training',validation_view='TRAINING_ONLY_NOT_VALIDATION'))
            batches.append(batch)
        raw=dict(status='completed',execution_context_sha256='ctx',request_records=rows,batches=batches)
        return plan,raw,bindings

    def test_twenty_distinct_instances_include_actual_same_batch_all_four(self):
        plan,raw,bindings=self.fixture();instances=m.validate_records(plan,raw,bindings,'capture:one')
        self.assertEqual(len(instances),20);self.assertEqual(len({r['instance_id'] for r in instances}),20)
        next_capture=m.validate_records(plan,raw,bindings,'capture:two')
        self.assertFalse({r['instance_id'] for r in instances}&{r['instance_id'] for r in next_capture})
        self.assertEqual([len(b['request_ids']) for b in raw['batches']].count(4),1)

    def test_missing_duplicate_wrong_work_validation_or_split_all_four_rejected(self):
        for mode in ('missing','duplicate','job','owner','pool','task','validation','split','context','batch','split-all4'):
            plan,raw,bindings=self.fixture()
            if mode=='missing':raw['request_records'].pop()
            if mode=='duplicate':raw['request_records'][1]['request_id']=raw['request_records'][0]['request_id']
            if mode=='job':raw['request_records'][0]['profile_job_id']='invented'
            if mode=='owner':raw['request_records'][0]['model']='ocr#0'
            if mode=='pool':raw['request_records'][0]['pool_id']='different-exact-work'
            if mode=='task':raw['request_records'][0]['task']='ocr'
            if mode=='validation':raw['request_records'][0]['validation_view']='VALIDATION'
            if mode=='split':raw['request_records'][0]['split']='validation'
            if mode=='context':raw['execution_context_sha256']='new-nonce'
            if mode=='batch':raw['batches'][0]['request_ids'].append('unoffered')
            if mode=='split-all4':
                b=raw['batches'][-1];b['request_ids'].pop()
            with self.subTest(mode=mode),self.assertRaises(ValueError):m.validate_records(plan,raw,bindings,'capture')

    def test_old_split_ssg_ocr_jobs_or_new_input_hash_never_become_matrix(self):
        plan,_,bindings=self.fixture();plan['jobs'][-1]['inputs'].pop('ocr#0')
        with self.assertRaises(ValueError):m.validate_matrix(plan,bindings)
        plan,_,bindings=self.fixture();plan['inputs'][0]['exact_input_sha256']='bad'
        with self.assertRaises(ValueError):m.validate_matrix(plan,bindings)

    def test_matching_gate_requires_exact_manifest_status_and_complete_fifteen(self):
        campaign={'after_matching_manifest':{'path':'manifest','sha256':'p'}}
        report=dict(schema=m.MATCHING_SCHEMA,status=m.MATCHING_STATUS,manifest=campaign['after_matching_manifest'],
            completed_attempts=[dict(attempt_id=str(i)) for i in range(15)],numerical_fit_run=False)
        manifest=dict(attempts=[dict(attempt_id=str(i)) for i in range(15)])
        with patch.object(m,'read',side_effect=[report,manifest]),patch.object(m,'verify'):
            m.require_matching_completion(campaign,{'path':'final'})
        for mode in ('schema','status','manifest','count','fit','cohort'):
            broken=copy.deepcopy(report)
            if mode in ('schema','status','manifest'):broken[mode]='wrong'
            if mode=='count':broken['completed_attempts'].pop()
            if mode=='fit':broken['numerical_fit_run']=True
            if mode=='cohort':broken['completed_attempts'][0]['attempt_id']='foreign'
            with self.subTest(mode=mode),patch.object(m,'read',side_effect=[broken,manifest]),patch.object(m,'verify'),self.assertRaises(ValueError):
                m.require_matching_completion(campaign,{'path':'final'})

    def test_wrapper_prepares_once_then_reuses_complete_context_and_distinct_plans(self):
        plan,_,bindings=self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);matrix_path=base/'matrix.json';params=base/'parameters.json';params.write_text('{}')
            matching_path=base/'matching.json';matching_path.write_text(json.dumps(dict(schema=m.MATCHING_SCHEMA.removesuffix('.run'))))
            matrix=dict(schema='gsdeploy.kernel_dag_response_replication_plan.v1',validation_labels_read=False,
                captures=3,sources=[],exact_owner_inputs=bindings,jobs=plan['jobs'],parameters=receipt(params))
            matrix_path.write_text(json.dumps(matrix))
            def seed(output,source,parameters):
                run=output/'light';run.mkdir(parents=True)
                context=dict(execution_nonce='retain-full-nonce',execution_sources=[],sources=[],hardware_uuid='full')
                (run/'context.json').write_text(json.dumps(context));(run/'plan.json').write_text(json.dumps(dict(plan,plan_sha256='old')))
                return dict(runs={m.collector.LIGHT:dict(run=str(run))})
            with patch.object(m.collector,'prepare',side_effect=seed) as prepare,patch.object(m.collector,'audit'),\
                 patch.object(m,'implementation_sources',return_value=[]),patch.object(m,'fresh_source_packet',return_value='source'):
                campaign=m.prepare(matrix_path,'source',matching_path,base/'prepared')
                self.assertEqual(prepare.call_count,1)
                contexts=[json.loads(Path(e['context']['path']).read_text()) for e in campaign['runs']]
                self.assertTrue(all(c==contexts[0] for c in contexts));self.assertEqual(contexts[0]['execution_nonce'],'retain-full-nonce')
                self.assertEqual(len({e['plan']['sha256'] for e in campaign['runs']}),3)
                m.audit(base/'prepared/campaign.json')
                # read(entry refs) and collector.audit must refer to one directory.
                swapped=copy.deepcopy(campaign);swapped['runs'][0]['run']=swapped['runs'][1]['run']
                (base/'prepared/campaign.json').write_text(json.dumps(swapped))
                with self.assertRaisesRegex(ValueError,'Canonical lifecycle'):m.audit(base/'prepared/campaign.json')
                (base/'prepared/campaign.json').write_text(json.dumps(campaign))
                # A different nonce is rejected even if a caller updates its context receipt.
                entry=campaign['runs'][1];changed=dict(contexts[1],execution_nonce='new')
                Path(entry['context']['path']).write_text(json.dumps(changed));entry['context']=receipt(entry['context']['path'])
                (base/'prepared/campaign.json').write_text(json.dumps(campaign))
                with self.assertRaises(ValueError):m.audit(base/'prepared/campaign.json')

    def test_source_packet_preserves_old_planner_evidence_and_rejects_direct_drift(self):
        for drift in ('planner','direct','other'):
            with self.subTest(drift=drift),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);run=root/'historical';run.mkdir();(root/'gsdeploy').mkdir()
                direct=root/'native.py';direct.write_text('same direct source')
                planner=root/'gsdeploy/milp.py';planner.write_text('old planner')
                frozen_direct=run/'native.py';frozen_direct.write_bytes(direct.read_bytes())
                frozen_planner=run/'milp.py';frozen_planner.write_bytes(planner.read_bytes())
                context=dict(shared_native_execution_sources=[receipt(direct)],worker_bindings={},
                    execution_sources=[dict(original_path=str(direct),frozen=receipt(frozen_direct)),
                        dict(original_path=str(planner),frozen=receipt(frozen_planner))])
                (run/'context.json').write_text(json.dumps(context))
                (run/'plan.json').write_text(json.dumps(dict(execution_context_sha256=m.context_identity(context))))
                if drift=='planner':planner.write_text('new planner')
                if drift=='direct':direct.write_text('changed native execution')
                if drift=='other':
                    # An unclassified source cannot gain a fresh historical claim.
                    context['execution_sources'][1]['original_path']=str(root/'other.py')
                    (root/'other.py').write_text('different offline source')
                    (run/'context.json').write_text(json.dumps(context))
                    (run/'plan.json').write_text(json.dumps(dict(execution_context_sha256=m.context_identity(context))))
                with patch.object(m.collector,'ROOT',root),patch.object(m,'shared_execution_paths',return_value={direct}),\
                     patch.object(m,'source_audit'):
                    if drift!='planner':
                        with self.assertRaises(ValueError):m.fresh_source_packet(run,root/'new')
                    else:
                        new=m.fresh_source_packet(run,root/'new')
                        proof=json.loads((new/'source-rebind.json').read_text())
                        self.assertEqual(len(proof['planner_only_drift']),1)
                        self.assertEqual(proof['unchanged_shared_native_execution_sources'],[receipt(direct)])
                        self.assertFalse(proof['whole_domain_native_qualification_inherited'])
                        self.assertEqual(frozen_planner.read_text(),'old planner')

    def test_adapter_role_gate_precedes_profile_or_kernel_reads(self):
        bad=dict(schema=m.SCHEMA+'.review',status='PASS_CPU_MATRIX_NATIVE_BINDING_ONLY',training_only=True)
        with patch.object(adapter,'read',return_value=bad) as read:
            with self.assertRaises(ValueError):adapter.load_member(dict(producer_review={'path':'bad'}))
            self.assertEqual(read.call_count,1)

    def test_adapter_reauthenticates_exclusive_execution_chain_before_intake(self):
        forged=dict(schema=m.SCHEMA+'.review',status='PASS_CPU_MATRIX_NATIVE_BINDING_ONLY',training_only=True,
            validation_labels_read=False,campaign={'path':'campaign'},capture_id='capture',native_review={'path':'native'})
        with patch.object(adapter,'read',return_value=forged) as read,\
             patch.object(m,'authenticate_review',side_effect=ValueError('exclusive claim missing')) as auth:
            with self.assertRaisesRegex(ValueError,'exclusive claim missing'):
                adapter.load_member(dict(producer_review={'path':'forged'}))
            self.assertEqual(read.call_count,1);auth.assert_called_once_with('campaign','capture','native')

    def test_outer_guard_catches_change_and_restore_of_another_capture_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);entries=[]
            for index in range(3):
                run=root/str(index);run.mkdir();(run/'plan.json').write_text('{}');(run/'context.json').write_text('{}')
                entries.append(dict(capture_id=str(index),run=str(run),plan=receipt(run/'plan.json'),context=receipt(run/'context.json')))
            campaign=dict(runs=entries);path=root/'campaign.json';path.write_text(json.dumps(campaign))
            after=root/'matching.json';after.write_text('{}')
            def transient(run):
                other=Path(entries[1]['plan']['path']);original=other.read_bytes()
                other.write_text('changed while selected capture runs');other.write_bytes(original)
                return dict(status='COLLECTED',GPU_started=False)
            with patch.object(m,'audit',return_value=campaign),patch.object(m,'require_matching_completion'),\
                 patch.object(m.collector,'execute',side_effect=transient):
                with self.assertRaises(ValueError):m.execute(path,'0',receipt(after))
            self.assertTrue((root/'0/response-replication-claim.json').exists())
            self.assertFalse((root/'0/response-replication-result.json').exists())

    def test_repeated_adapter_rejects_raw_aliases_and_changed_full_context(self):
        design=dict(schema=adapter.SCHEMA+'.design',training_only=True,validation_labels_read=False,captures=[{}, {}, {}])
        def member(index):
            return dict(capture=dict(raw_sha256=str(index)),review=dict(capture_id=str(index),sqlite={'sha256':str(index)},campaign={'path':'same'}),
                context=dict(nonce='same',SM_count=16),hardware=dict(hardware_id='same',SM_count=16),observed=[])
        members=[member(i) for i in range(3)]
        for mode in ('alias','nonce','hardware'):
            changed=copy.deepcopy(members)
            if mode=='alias':changed[1]['capture']['raw_sha256']='0'
            if mode=='nonce':changed[1]['context']['nonce']='new'
            if mode=='hardware':changed[1]['hardware']['SM_count']=15
            with tempfile.TemporaryDirectory() as temp,patch.object(adapter,'receipt',return_value={'path':'design'}),\
                 patch.object(adapter,'read',return_value=design),patch.object(adapter,'load_member',side_effect=changed),\
                 patch.object(adapter,'implementation_sources',return_value=[]),self.assertRaises(ValueError):
                adapter.align('design',Path(temp)/'output')

    def repeated_fixture(self):
        plan,raw,bindings=self.fixture();members=[]
        for index in range(3):
            cap=dict(schema=adapter.compiler.CAPTURE_SCHEMA,split='training',validation_labels_read=False,
                evidence_kind='declared_measured',execution_context_sha256='ctx',hardware_id=hardware()['hardware_id'],
                profiler_id='synthetic-test-only',capture_id='c'+str(index),raw_sha256=str(index)*64,
                offered_request_ids=[],requests=[],kernels=[])
            member=dict(capture=cap,review=dict(capture_id='c'+str(index),sqlite={'sha256':str(index)*64},
                campaign={'path':'synthetic-test-only'}),context=dict(nonce='same'),hardware=hardware(),
                intake=dict(raw={'path':'synthetic-test-only','sha256':str(index)*64},
                    sqlite={'path':'synthetic-test-only','sha256':str(index)*64},artifacts={'capture.json':{'path':'synthetic-test-only'}}),
                observed=[],sources=[])
            for number,row in enumerate(raw['request_records']):
                owner=row['model'];rid=row['request_id'];bind=dict(alias=owner,task=row['task'],pool_id=row['pool_id'],
                    exact_input_sha256=bindings[owner]['exact_input_sha256'],actual_work=dict(output='same-'+owner))
                scalar=dict(kernel(shared=1024),name='same-'+owner,architecture_blocks_per_sm_ceiling=8)
                common=dict(work=bind,nodes=[dict(id='node:0',kind='gpu')],edges=[])
                member['observed'].append(dict(request_id=rid,instance=dict(job_id=row['profile_job_id'],owner=owner,
                    instance_id='c'+str(index)+':'+rid),binding=bind,work_sha256=identity(bind),common=common,
                    mapping={'native:0':'node:0'},static={'node:0':scalar},profile=dict(family='synthetic')))
                cap['offered_request_ids'].append(rid);cap['requests'].append(dict(request_id=rid,owner=owner,
                    profile_id=rid,task=row['task'],pool_id=row['pool_id']))
                native=dict(scalar,pid=index+1,device=0,context_id=1,stream=7,correlation_id=number,
                    nsys_start_ns=number*100+1,nsys_end_ns=number*100+99)
                cap['kernels'].append(dict(request_id=rid,owner=owner,node_id='native:0',kernel=native,
                    start_ns=native['nsys_start_ns'],end_ns=native['nsys_end_ns'],clock_domain='session:'+str(index)))
            members.append(member)
        return members

    def test_repeated_adapter_exports_sixty_distinct_responses_with_four_common_templates(self):
        design=dict(schema=adapter.SCHEMA+'.design',training_only=True,validation_labels_read=False,captures=[{},{},{}])
        members=self.repeated_fixture()
        with tempfile.TemporaryDirectory() as temp,patch.object(adapter,'read',return_value=design),\
             patch.object(adapter,'load_member',side_effect=members),patch.object(adapter,'implementation_sources',return_value=[]),\
             patch.object(adapter,'verify'),patch.object(adapter,'receipt',return_value={'path':'synthetic-design'}):
            result=adapter.align('design',Path(temp)/'output')
            self.assertEqual(result['offered_instances'],60);self.assertEqual(len(result['matched']),4)
            self.assertEqual(result['compiler_counts']['accepted_kernel_count'],60)
            caps=json.loads((Path(temp)/'output/captures.json').read_text())
            self.assertEqual([len(c['requests']) for c in caps],[20,20,20])
            self.assertEqual(len({r['profile_id'] for c in caps for r in c['requests']}),4)
            self.assertEqual(len({k['clock_domain'] for c in caps for k in c['kernels']}),3)

    def test_one_instance_work_or_dependency_change_retains_sixty_and_never_exports_partial(self):
        design=dict(schema=adapter.SCHEMA+'.design',training_only=True,validation_labels_read=False,captures=[{},{},{}])
        for mode in ('work','dependency'):
            members=self.repeated_fixture();observed=members[1]['observed'][0]
            if mode=='work':observed['work_sha256']='new-exact-work'
            else:observed['common']=dict(observed['common'],edges=[dict(kind='changed')])
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp,patch.object(adapter,'read',return_value=design),\
                 patch.object(adapter,'load_member',side_effect=members),patch.object(adapter,'implementation_sources',return_value=[]),\
                 patch.object(adapter,'verify'),patch.object(adapter,'receipt',return_value={'path':'synthetic-design'}):
                result=adapter.align('design',Path(temp)/'output')
                self.assertEqual(result['offered_instances'],60);self.assertTrue(result['rejected'])
                self.assertFalse(result['artifacts']);self.assertFalse((Path(temp)/'output/captures.json').exists())


if __name__=='__main__':unittest.main()
