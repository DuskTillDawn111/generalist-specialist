import copy
from fractions import Fraction as F
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_joint_request_law_v2 import source_fixture
from tools.predictor_validation_20261027 import twoFIFO_prospective_memory_v1 as m

PHYSICAL='0'*64
OWNERS=('text#0',)


def inputs():
    *_,vector=source_fixture()
    for node in vector['nodes']:
        if node['phase']=='prefill':node['forward_shape']=dict(input_sequence_length=16,input_batch_size=1,forward_index=0)
        elif node['phase']=='decode':node['forward_shape']=dict(input_sequence_length=1,input_batch_size=1,forward_index=1)
    sidecar=m.selected_shape_sidecar([dict(vector=vector,public_key_arrival_ordinal=0)],physical_execution_identity=PHYSICAL)
    request=dict(request_id='generated',owner='text#0',task=vector['public_key'][1],pool_id=vector['public_key'][2],arrival_ns=0)
    by_action={}
    for node in sidecar['slots'][0]['nodes']:
        for mark in node['marks']:
            if mark['action']=='phase':by_action[mark['phase']]=mark
            elif mark['action']=='clear':by_action['clear']=mark
    trace=[]
    for phase,at in [('entry',5),('prefill',10),('decode',20),('clear',30)]:
        mark=by_action[phase]
        body=dict(owner='text#0',request_id='generated',identity=mark['generated_mark_identity'],
            action=mark['action'],phase=mark['phase'],macro_id=mark['macro_id'],occurrence_ns=at-4)
        trace.extend([dict(kind='latent_occurrence',**body),dict(kind='processed_notification',available_ns=at,**body)])
    trace.append(dict(kind='processed_notification',owner='text#0',request_id='generated',identity='worker_READY',
        action='ready',phase=None,macro_id=None,occurrence_ns=31,available_ns=35))
    world=dict(schema=m.engine.SCHEMA,offered_requests=1,RESULT_receipts={'generated':32},unfinished_request_ids=[],
        original_window=[F(0),F(40)],cutoff_ns=F(40),trace=trace)
    initial=m.shapes.CausalState([],OWNERS).descriptor()
    return vector,sidecar,world,[request],initial


def absolute_fixture():
    vector,sidecar,world,requests,initial=inputs()
    generated=m.engine.decode(m.generated_shape_intervals(sidecar,world,requests,resident_owners=OWNERS,initial_descriptor=initial))
    polls=[]
    for row in generated['intervals']:
        at=(row['left_ns']+row['right_ns'])/2
        polls.append(dict(poll_begin_ns=at,poll_end_ns=at,sample_ns=at))
    intervals,_=m.attach_observation_support(generated['intervals'],polls)
    tables=[]
    for block in (1,2,3):
        for shard in (1,2,3,4):
            rows=copy.deepcopy(intervals)
            for row,value in zip(rows,[100.,120.,150.,140.,90.]):row['absolute_sampled_peak_target_mib']=value
            tables.append(dict(schema=m.TABLE_SCHEMA,capture_id='training-block-'+str(block),
                attempt_id='training-block-'+str(block)+'-shard-'+str(shard),physical_execution_identity=PHYSICAL,
                resident_owners=list(OWNERS),initial_shape_descriptor=initial,
                initial_after_priming_owned_sampled_peak_mib=80.,intervals=rows))
    return sidecar,world,requests,tables,dict(basis='caller_declared_generated_observation_schedule',polls=m.encode(polls))


class ProspectiveMemoryTests(unittest.TestCase):
    def test_complete_selected_nodes_and_marks_preserved_without_modifying_compiler(self):
        vector,sidecar,_,_,_=inputs();compiled=m.joint.compile_draw_v4(vector,0)
        self.assertEqual(len(sidecar['slots'][0]['nodes']),len(vector['nodes']))
        self.assertEqual(sum(len(n['marks']) for n in sidecar['slots'][0]['nodes']),sum(len(n['marks']) for n in vector['nodes']))
        self.assertFalse(hasattr(compiled['plan'].macros[1],'forward_shape'))
        self.assertEqual(sidecar['slots'][0]['nodes'][1]['marks'][0]['forward_shape']['input_sequence_length'],16)
        with self.assertRaisesRegex(ValueError,'Distinct complete public slots'):
            m.selected_shape_sidecar([dict(vector=vector,public_key_arrival_ordinal=0)]*2,physical_execution_identity=PHYSICAL)

    def test_shape_is_visible_only_at_processing_and_future_decode_cannot_change_past(self):
        vector,sidecar,world,requests,initial=inputs()
        before=m.engine.decode(m.generated_shape_intervals(sidecar,world,requests,resident_owners=OWNERS,initial_descriptor=initial))
        entry=next(r for r in before['intervals'] if r['left_ns']==5)
        prefill=next(r for r in before['intervals'] if r['left_ns']==10)
        self.assertIsNone(entry['state']['shape_state']['current_processed_owner_task_exact_input_phase_shape'][0]['actual_BEGIN_shape'])
        self.assertEqual(prefill['state']['shape_state']['current_processed_owner_task_exact_input_phase_shape'][0]['actual_BEGIN_shape']['input_sequence_length'],16)
        vector['nodes'][2]['forward_shape']['forward_index']=999
        changed=m.selected_shape_sidecar([dict(vector=vector,public_key_arrival_ordinal=0)],physical_execution_identity=PHYSICAL)
        after=m.engine.decode(m.generated_shape_intervals(changed,world,requests,resident_owners=OWNERS,initial_descriptor=initial))
        self.assertEqual([r for r in before['intervals'] if r['right_ns']<=20],[r for r in after['intervals'] if r['right_ns']<=20])
        self.assertNotEqual(before['final_shape_descriptor'],after['final_shape_descriptor'])

    def test_real_v4_compiled_engine_trace_binds_generated_shape_sidecar(self):
        plan,raw,_,_,vector=source_fixture();vector['nodes'][1]['forward_shape']=dict(input_sequence_length=8,input_batch_size=1,forward_index=0)
        compiled=m.joint.compile_draw_v4(vector,0);key=tuple(vector['public_key']);origin=raw['request_records'][0]['scheduled_time_ns']
        previous=raw['priming_request_records'][-1];requests=[dict(request_id='real-engine-generated',owner=key[0],task=key[1],pool_id=key[2],arrival_ns=origin)]
        world=m.engine.simulate(requests=requests,plans={key:compiled['plan']},rates={(key[0],n.phase,0,0):F(1) for n in compiled['plan'].macros},
            protocol_slacks={compiled['plan']:compiled['protocol_slacks']},transport_draws={(*key,0):compiled['transport_draw']},
            window_start_ns=origin,window_end_ns=origin+1,drain_end_ns=origin+10_000_000_000,
            initial_sender_available_ns={key[0]:previous['submit_write_completed_time_ns']},
            initial_worker_read_release_ns={key[0]:previous['previous_handler_next_request_ready']['perf_ns']},
            initial_protocol_state=dict(reader_free={},collector_free=origin,last_collector_arrival=origin,pending_messages=[]))
        sidecar=m.selected_shape_sidecar([dict(vector=vector,public_key_arrival_ordinal=0)],physical_execution_identity=PHYSICAL)
        generated=m.engine.decode(m.generated_shape_intervals(sidecar,world,requests,resident_owners=OWNERS))
        notice=next(r for r in generated['processed_shape_marks'] if r['state_after']['shape_state']['current_processed_owner_task_exact_input_phase_shape'] and
            r['state_after']['shape_state']['current_processed_owner_task_exact_input_phase_shape'][0]['phase']=='prefill')
        self.assertEqual(notice['state_after']['shape_state']['current_processed_owner_task_exact_input_phase_shape'][0]['actual_BEGIN_shape']['input_sequence_length'],8)
        self.assertFalse(generated['initial_shape_history_known'])

    def test_max_rollout_and_inserted_duplicate_queries_are_idempotent(self):
        sidecar,world,requests,tables,observation=absolute_fixture();model=m.fit_absolute_tables(tables[:8])
        kwargs=dict(resident_owners=OWNERS,physical_execution_identity=PHYSICAL,absolute_model=model,
            memory_origin=dict(basis='selected_training_donor_after_original_priming',donor_attempt_id='training-block-1-shard-1'),observation_plan=observation)
        baseline=m.rollout(sidecar,world,requests,query_times=[12],**kwargs)
        repeated=m.rollout(sidecar,world,requests,query_times=[1,12,12,12,22,38],**kwargs)
        self.assertEqual(baseline['peak_memory_mib'],150.)
        self.assertEqual(baseline['intervals'],repeated['intervals'])
        self.assertEqual(baseline['peak_memory_mib'],repeated['peak_memory_mib'])
        self.assertEqual(repeated['queries'][1]['prediction'],repeated['queries'][2]['prediction'])
        self.assertFalse(repeated['poll_increments_used']);self.assertFalse(repeated['capacity_bound'])

    def test_missing_origin_observation_support_or_model_stays_UNKNOWN(self):
        sidecar,world,requests,tables,observation=absolute_fixture();model=m.fit_absolute_tables(tables[:8])
        kwargs=dict(resident_owners=OWNERS,physical_execution_identity=PHYSICAL)
        self.assertIsNone(m.rollout(sidecar,world,requests,**kwargs)['peak_memory_mib'])
        self.assertIsNone(m.rollout(sidecar,world,requests,absolute_model=model,observation_plan=observation,**kwargs)['peak_memory_mib'])
        origin=dict(basis='selected_training_donor_after_original_priming',donor_attempt_id='training-block-1-shard-1')
        self.assertIsNone(m.rollout(sidecar,world,requests,absolute_model=model,memory_origin=origin,**kwargs)['peak_memory_mib'])
        changed=copy.deepcopy(observation);changed['polls'].append(changed['polls'][1])
        self.assertIsNone(m.rollout(sidecar,world,requests,absolute_model=model,memory_origin=origin,observation_plan=changed,**kwargs)['peak_memory_mib'])
        changed=copy.deepcopy(observation);changed['polls'][0]['sampled_owned_mib']=1e9
        with self.assertRaisesRegex(ValueError,'clocks only'):
            m.rollout(sidecar,world,requests,absolute_model=model,memory_origin=origin,observation_plan=changed,**kwargs)
        with self.assertRaisesRegex(ValueError,'settled seed is not memory'):
            m.rollout(sidecar,world,requests,absolute_model=model,memory_origin={'basis':'settled-seed'},**kwargs)

    def test_native_absolute_targets_keep_boundary_crossing_and_no_poll_UNKNOWN(self):
        def event(rid,at,segment,edge='start',phase=None,**shape):
            return dict(request_id=rid,perf_ns=at,received_time_ns=at,collector_dequeued_time_ns=at,
                collector_processing_time_ns=at,event=edge,segment_id=segment,semantic_phase=phase,**shape)
        priming=dict(request_id='prime',model='text#0',task='text',pool_id='prime',profile_job_id='priming',execution_events=[
            event('prime',101,'prefill',phase='prefill',input_sequence_length=8,input_batch_size=1,forward_index=0),
            event('prime',102,'native_inference',edge='end')])
        row=dict(request_id='r',model='text#0',task='text',pool_id='input',profile_job_id='case',execution_events=[
            event('r',112,'native_inference'),event('r',120,'prefill',phase='prefill',input_sequence_length=16,input_batch_size=1,forward_index=0),
            event('r',130,'decode',phase='decode',input_sequence_length=1,input_batch_size=1,forward_index=1),event('r',140,'native_inference',edge='end')])
        raw=dict(priming_request_records=[priming],request_records=[row],measurement_windows=[dict(case_id='case',start_ns=110,end_ns=120,fixed_drain_deadline_ns=150)],
            board_samples=[dict(time_ns=t,query_duration_ns=d,owned_used_mib=v) for t,d,v in [(105,0,80),(115,0,100),(125,0,120),(130,2,99999),(135,0,140)]])
        plan=dict(capture_id='training-block-1',attempt_id='training-block-1-shard-1',resident_owners=list(OWNERS))
        native=dict(premeasurement_owned_sampled_peak_mib=80,measured_case_resources=dict(cases=[dict(case_id='case')]))
        original=copy.deepcopy(raw)
        with patch.object(m.resource,'physical_execution_identity',return_value=PHYSICAL):table=m.absolute_table_from_verified(raw,plan,{},native)
        self.assertEqual(raw,original);self.assertEqual(len(table['poll_assignments']),5)
        self.assertFalse(table['compatibility_priming_job_annotation_used'])
        rows=m.engine.decode(table['intervals']);entry=next(r for r in rows if r['left_ns']==112)
        self.assertEqual(entry['absolute_sampled_peak_target_mib'],100)
        self.assertTrue(all(r['absolute_sampled_peak_target_mib'] is None for r in rows if r['left_ns'] in (120,130,140)))
        self.assertEqual(sum(bool(r['crossing_poll_ordinals']) for r in rows),2)

    def test_wholeblock_fold_poison_exclusion_and_partial_intake_never_fit(self):
        *_,tables,_=absolute_fixture();fold=m.fit_absolute_tables(tables[:8]);poisoned=copy.deepcopy(tables)
        for block in ('training-block-1','training-block-2','training-block-3'):
            fitted=m.fit_absolute_tables([t for t in tables if t['capture_id']!=block])
            self.assertEqual(len(fitted['training_attempt_ids']),8)
            self.assertEqual(len(fitted['training_block_ids']),2)
            self.assertNotIn(block,fitted['training_block_ids'])
        for table in poisoned[8:]:
            table['initial_after_priming_owned_sampled_peak_mib']=1e12
            for row in table['intervals']:row['absolute_sampled_peak_target_mib']=1e12
        self.assertEqual(fold,m.fit_absolute_tables(poisoned[:8]))
        with self.assertRaises(ValueError):m.fit_absolute_tables(tables[:7])
        campaign=Path('artifacts/predictor-twoFIFO-owned-training-design-v3-20261028-cpu01/campaign.json')
        metadata=m.checked(m.receipt(campaign))
        with tempfile.TemporaryDirectory() as directory,patch.object(m.resource.intake.campaign_v2,'audit',return_value=metadata),\
                patch.object(m.resource.macros_v2,'extract') as extract,patch.object(m,'absolute_table_from_verified') as adapter,patch.object(m,'fit_absolute_tables') as fit:
            with self.assertRaisesRegex(ValueError,'All twelve'):m.train_absolute_campaign(campaign,[],Path(directory)/'output')
        extract.assert_not_called();adapter.assert_not_called();fit.assert_not_called()

    def test_physical_mark_and_donor_binding_fail_closed(self):
        sidecar,world,requests,tables,observation=absolute_fixture();model=m.fit_absolute_tables(tables[:8])
        kwargs=dict(resident_owners=OWNERS,physical_execution_identity=PHYSICAL,absolute_model=model,
            memory_origin=dict(basis='selected_training_donor_after_original_priming',donor_attempt_id='training-block-1-shard-1'),observation_plan=observation)
        wrong=copy.deepcopy(world);wrong['trace'][1]['macro_id']='wrong-macro'
        with self.assertRaisesRegex(ValueError,'macro differs'):m.rollout(sidecar,wrong,requests,**kwargs)
        wrong=copy.deepcopy(sidecar);wrong['physical_execution_identity']='1'*64
        with self.assertRaisesRegex(ValueError,'physical'):m.rollout(wrong,world,requests,**kwargs)
        wrong=copy.deepcopy(sidecar);wrong['slots'][0]['training_block_id']='training-block-3'
        with self.assertRaisesRegex(ValueError,'outside absolute training donors'):m.rollout(wrong,world,requests,**kwargs)
        changed=copy.deepcopy(kwargs);changed['memory_origin']['donor_attempt_id']='training-block-2-shard-1'
        with self.assertRaisesRegex(ValueError,'share one whole-block latent'):m.rollout(sidecar,world,requests,**changed)


if __name__=='__main__':unittest.main()
