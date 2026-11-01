import copy
import json
import unittest
from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_actual_worker_twoFIFO_reference_v5 import inputs, decode
from tools.predictor_validation_20261027 import twoFIFO_settled_history_initial_v1 as history
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v5_1 as engine


def snapshot():
    rows=[];ready={}
    for owner,pid in [('owner',111),('idle',222)]:
        ready[owner]=dict(pid=pid,clock='time.perf_counter_ns',spec_sha256='a'*64)
        rows.append(dict(request_id='prior-'+owner,model=owner,status='completed',scheduled_time_ns=1,
            enqueued_time_ns=2,actual_submit_time_ns=3,submit_write_completed_time_ns=4,
            observed_worker_start_time_ns=5,worker_response=dict(success=True,start_perf_ns=5,end_perf_ns=10),
            completion_time_ns=11,response_reader_event_enqueue_time_ns=11,
            response_collector_dequeued_time_ns=12,response_collector_processing_time_ns=13,
            execution_events=[dict(pid=pid,perf_ns=5,received_time_ns=6,collector_dequeued_time_ns=7,collector_processing_time_ns=8)],
            previous_handler_next_request_ready=dict(request_id='prior-'+owner,pid=pid,perf_ns=14,received_time_ns=15,
                clock='host_perf_counter_ns',boundary='previous_handler_next_request_ready',scope='after_response_flush_before_next_read',
                collector_dequeued_time_ns=16,collector_processing_time_ns=17)))
    return dict(snapshot_time_ns=50,request_records=rows,ready=ready,
        pending_request_ids=[r['request_id'] for r in rows],queued_runtime_events=[],runtime_errors=[],empty_initial_state_assumed=False)


def seed():
    return history.from_snapshot(snapshot(),cutoff_ns=50,forecast_origin_ns=100,owner_producers={'owner':'owner','idle':'idle:stdout'})


def seeded_inputs():
    kw=inputs();kw['requests']=[dict(r,arrival_ns=r['arrival_ns']+100) for r in kw['requests']]
    for key in ('window_start_ns','window_end_ns','drain_end_ns','cutoff_ns'):kw[key]+=100
    for key in ('initial_sender_available_ns','initial_worker_read_release_ns','initial_protocol_state'):kw.pop(key)
    kw['settled_initial_state']=seed()
    return kw


class SettledHistoryTests(unittest.TestCase):
    def test_four_clock_past_seed_and_relative_scope(self):
        value=seed()
        self.assertEqual(value['sequence'],6)
        self.assertEqual(value['producer_next'],{'owner':3,'idle:stdout':3})
        self.assertEqual(value['initial_worker_read_release_ns']['owner'],14)
        self.assertEqual(value['initial_protocol_state']['last_collector_arrival'],15)
        self.assertEqual(value['initial_protocol_state']['collector_free'],17)
        self.assertFalse(value['native_source_authenticated'])
        self.assertIn('relative_offset',value['sequence_scope'])
        self.assertTrue(value['prior_pending_request_ids_mean_record_registry'])

    def test_future_unsettled_missing_lifetime_and_registry_rejected(self):
        mutations=[lambda s:s.update(queued_runtime_events=[{}]),lambda s:s.update(runtime_errors=['error']),
            lambda s:s['request_records'][0].update(status='enqueued'),
            lambda s:s['request_records'][0].update(scheduled_time_ns=51),
            lambda s:s['request_records'][0].update(submit_write_completed_time_ns=51),
            lambda s:s['request_records'][0]['previous_handler_next_request_ready'].update(collector_processing_time_ns=51),
            lambda s:s['request_records'][0]['previous_handler_next_request_ready'].update(pid=999),
            lambda s:s['request_records'][0].pop('previous_handler_next_request_ready'),
            lambda s:s.update(pending_request_ids=[])]
        for mutate in mutations:
            s=snapshot();mutate(s)
            with self.subTest(s=s),self.assertRaises(UnknownEvidence):
                history.from_snapshot(s,cutoff_ns=50,forecast_origin_ns=100,owner_producers={'owner':'owner','idle':'idle:stdout'})

    def test_conditioned_world_keeps_offsets_and_excludes_prior_labels(self):
        kw=seeded_inputs();w=decode(engine.simulate(**kw));p=w['continuation_state']['payload']
        self.assertEqual(w['starts'],{'r0':101,'r1':112})
        self.assertEqual(w['RESULT_receipts'],{'r0':119,'r1':130})
        self.assertEqual(w['offered_requests'],2)
        self.assertEqual(set(w['starts']),{'r0','r1'})
        issuance=[r for r in w['trace'] if r['kind']=='latent_occurrence']
        self.assertEqual(issuance[0]['producer_sequence'],3)
        self.assertEqual(issuance[0]['owner_issuance_ordinal'],3)
        self.assertEqual(p['producer_next'],{'owner':7,'idle:stdout':3})
        self.assertEqual(p['sequence'],10)
        self.assertEqual(w['original_window'],[100,102])
        self.assertEqual(w['original_drain_end_ns'],250)
        self.assertEqual(set(r['request_id'] for r in w['trace']),{'r0','r1'})
        self.assertFalse(w['native_assimilation_eligible'])

    def test_exact_JSON_continuation_seed_counters_unchanged_by_cut(self):
        kw=seeded_inputs();full=engine.simulate(**kw)
        for at in (100,101,105,111,119,130,219):
            with self.subTest(at=at):
                partial=engine.simulate(**dict(kw,cutoff_ns=at))
                saved=json.loads(json.dumps(partial['continuation_state']))
                resume=dict(kw,continuation_state=saved);resume.pop('settled_initial_state')
                self.assertEqual(engine.simulate(**resume),full)

    def test_binding_tamper_current_prior_label_and_missing_seed_rejected(self):
        kw=seeded_inputs();partial=engine.simulate(**dict(kw,cutoff_ns=105))
        for field in ('sequence','producer_next','initial_history'):
            state=copy.deepcopy(partial['continuation_state']);p=state['payload']
            if field=='sequence':p[field]+=1
            elif field=='producer_next':p[field]['owner']+=1
            else:p[field]['producer_next']['owner']+=1
            with self.subTest(field=field),self.assertRaises(UnknownEvidence):engine.simulate(**dict(kw,continuation_state=state))
        kw['requests'][0]['request_id']='prior-owner'
        with self.assertRaises(UnknownEvidence):engine.simulate(**kw)

    def test_seed_is_only_snapshot_not_final_raw(self):
        captured=snapshot();fake_final=dict(request_records=copy.deepcopy(captured['request_records']),case_histories=[captured])
        before=history.from_snapshot(captured,cutoff_ns=50,forecast_origin_ns=100,owner_producers={'owner':'owner','idle':'idle:stdout'})
        fake_final['request_records'][0]['previous_handler_next_request_ready']['perf_ns']=10**12
        self.assertEqual(before,history.from_snapshot(fake_final['case_histories'][0],cutoff_ns=50,forecast_origin_ns=100,owner_producers={'owner':'owner','idle':'idle:stdout'}))

    def test_old_or_unqualified_review_rejected_before_raw_access(self):
        import tempfile
        from pathlib import Path
        for schema in ('gsdeploy.original_twoFIFO_owned_training_review.v2','gsdeploy.original_twoFIFO_owned_training_review.v3'):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'review.json';p.write_text(json.dumps(dict(schema=schema,validation_labels_read=False,
                    status='NOT_ELIGIBLE',training_feature_extraction_eligible=False,issues=[],raw={'path':'/missing'})))
                with self.assertRaises(UnknownEvidence):history.extract(p,case_id='case',forecast_origin_ns=100,owner_producers={'owner':'owner'})
