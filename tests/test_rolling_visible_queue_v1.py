"""Actual generic-simulator rolling continuation; synthetic visible events only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.forecasting.observed_state import feedback_event
from gsdeploy.forecasting.simulator import FullHorizonModel
from tools.predictor_validation_20261027 import rolling_visible_queue_v1 as v
from tests import test_full_horizon_forecast as f

MS=f.MS


def rows():
    return [dict(request_id='a',alias='s',task='text',pool_id='x',arrival_ns=0),
            dict(request_id='b',alias='s',task='text',pool_id='x',arrival_ns=75*MS),
            dict(request_id='c',alias='g',task='text',pool_id='x',arrival_ns=125*MS)]


def start(sequence, rid, alias, at):
    return feedback_event(sequence,'START',dict(request_id=rid,alias=alias,perf_ns=at),at,'fixture')


def boundary(sequence, rid, alias, kind, at, received):
    payload=dict(request_id=rid,alias=alias,perf_ns=at)
    if kind=='WORKER_READY': payload.update(clock='host_perf_counter_ns',boundary='previous_handler_next_request_ready',scope='after_response_flush_before_next_read')
    return feedback_event(sequence,kind,payload,received,'fixture')


def cursor(root, rid, profile, fraction, at):
    path=root/(rid+'-'+str(at)+'.json')
    value=dict(schema='gsdeploy.visible_remaining_work_evidence.v1',kind='DIRECT_SEGMENT_PROGRESS',
        request_id=rid,execution_context_sha256='fixture',observed_at_ns=at,received_time_ns=at,
        provenance='cpu_fixture',profile_id=profile,segment_index=0,remaining_fraction=fraction)
    path.write_text(json.dumps(value));return v.receipt(path)


def score():
    return dict(rates={'text':3.},refs={'text':1000.},quality_loss={'text':{'text':.1},'generalist_2':{'text':.1}},
        memory_budget_mib=1000.,energy_reference_watts=100.,memory_safe_factor=1.1,weights={k:1/6 for k in 'QTLPME'})


def session():
    return v.new_session(rows(),f.aliases(),'fixture',0,1000*MS,2000*MS)


def model(): return FullHorizonModel(f.aliases(),f.provider())


class RollingTests(unittest.TestCase):
    def test_real_continuation_consumes_current_cursor_all_future_arrivals_queue_and_p95(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); prefix=v.advance(session(),50*MS,[start(1,'a','s',0)])
            ref=cursor(root,'a','s',.5,50*MS)
            result=v.forecast_suffix(prefix,model(),cursor_evidence={'a':ref},score_kwargs=score(),allow_cpu_fixture=True)
            self.assertEqual(result['status'],'SEGMENT_BACKEND_ROLLING_FORECAST')
            self.assertEqual(result['future_arrivals_repredicted'],2)
            self.assertTrue(result['all_original_request_ids_retained'])
            self.assertEqual({rid:r['finish_ns']/MS for rid,r in result['forecast']['finishes'].items()},dict(a=100,b=250,c=225))
            self.assertEqual(result['original_boundary_queue_wait_ms'],dict(a=0.,b=25.,c=0.))
            self.assertAlmostEqual(result['metrics']['per_task']['text']['p95_ms'],167.5)
            self.assertEqual(result['forecast']['finishes']['a']['start_ns'],0)
            self.assertEqual(result['remaining_state']['remaining_profile_work']['a']['solo_work_ms'],50.)
            self.assertIsNone(result['metrics']['terms']['E'])
            self.assertIsNone(result['metrics']['full_U'])
            self.assertFalse(result['simulated_checkpoint_used'])
            self.assertFalse(result['main_v54_rolling_complete'])
            changed=cursor(root,'a','s',.25,50*MS)
            corrected=v.forecast_suffix(prefix,model(),cursor_evidence={'a':changed},score_kwargs=score(),allow_cpu_fixture=True)
            self.assertEqual(corrected['forecast']['finishes']['a']['finish_ns'],75*MS)
            self.assertAlmostEqual(corrected['metrics']['per_task']['text']['p95_ms'],145.)

    def test_arrived_not_started_requires_current_waiting_proof_and_preserves_fifo(self):
        prefix=v.advance(session(),80*MS,[start(1,'a','s',0)])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);active=cursor(root,'a','s',.2,80*MS)
            without=v.forecast_suffix(prefix,model(),cursor_evidence={'a':active},allow_cpu_fixture=True)
            self.assertEqual(without['status'],'UNKNOWN')
            self.assertIn('VISIBLE_WAITING_NOT_STARTED_PROOF_REQUIRED:b',without['reason'])
            path=root/'waiting.json';path.write_text(json.dumps(dict(schema='gsdeploy.visible_remaining_work_evidence.v1',
                kind='WAITING_NOT_STARTED',request_id='b',execution_context_sha256='fixture',observed_at_ns=80*MS,
                received_time_ns=80*MS,provenance='cpu_fixture')))
            result=v.forecast_suffix(prefix,model(),cursor_evidence={'a':active},waiting_evidence={'b':v.receipt(path)},
                                     allow_cpu_fixture=True,score_kwargs=score())
            self.assertEqual(result['status'],'SEGMENT_BACKEND_ROLLING_FORECAST')
            self.assertEqual(result['original_boundary_queue_wait_ms']['b'],25.)
            self.assertEqual(result['current_waiting_depth'],1)

    def test_second_real_tick_conditions_new_prefix_and_recomputes_entire_remaining_work(self):
        first=v.advance(session(),50*MS,[start(1,'a','s',0)])
        feedback=[boundary(2,'a','s','HANDLER_END',100*MS,110*MS),boundary(3,'a','s','WORKER_READY',114*MS,115*MS),
                  start(4,'b','s',120*MS),start(5,'c','g',125*MS),
                  boundary(6,'b','s','HANDLER_END',200*MS,205*MS),boundary(7,'b','s','WORKER_READY',207*MS,208*MS)]
        second=v.advance(first,210*MS,feedback)
        with tempfile.TemporaryDirectory() as directory:
            ref=cursor(Path(directory),'c','g',.6,210*MS)
            result=v.forecast_suffix(second,model(),cursor_evidence={'c':ref},score_kwargs=score(),allow_cpu_fixture=True)
        self.assertEqual(result['status'],'SEGMENT_BACKEND_ROLLING_FORECAST')
        self.assertEqual(result['future_arrivals_repredicted'],0)
        self.assertEqual(result['forecast']['finishes']['c']['finish_ns'],240*MS)
        self.assertEqual({r['request_id'] for r in result['remaining_state']['completed_prefix']},{'a','b'})
        self.assertEqual(result['original_boundary_queue_wait_ms'],dict(a=0.,b=45.,c=0.))
        self.assertTrue(result['all_original_request_ids_retained'])
        self.assertEqual(first['observer']['observed_at_ns'],50*MS)
        self.assertEqual(first['observer']['last_feedback_sequence'],1)

    def test_no_future_worker_completion_body_is_accessed(self):
        class Bomb(dict):
            def get(self,*_): raise AssertionError('future worker response accessed')
        record=dict(request_id='a',model='s',execution_events=[dict(request_id='a',perf_ns=0,received_time_ns=0)],
                    response_reader_event_enqueue_time_ns=100*MS,worker_response=Bomb())
        observed=v.advance_runtime_records(session(),50*MS,[record])
        self.assertIsNone(observed['observer']['requests']['a']['handler_end_ns'])
        late=v.advance_runtime_records(observed,60*MS,[record])
        self.assertEqual(late['observer']['last_feedback_sequence'],1)
        tampered=copy.deepcopy(record);tampered['execution_events'][0]['perf_ns']=1
        with self.assertRaisesRegex(ValueError,'consumed event prefix'):
            v.advance_runtime_records(observed,60*MS,[tampered])

    def test_missing_busy_progress_unknown_and_age_never_becomes_fraction(self):
        prefix=v.advance(session(),50*MS,[start(1,'a','s',0)])
        result=v.forecast_suffix(prefix,model())
        self.assertEqual(result['status'],'UNKNOWN')
        self.assertIsNone(result['forecast'])
        self.assertIn('DIRECT_OBSERVED_REMAINING_SEGMENT_CURSOR_REQUIRED:a',result['reason'])
        self.assertFalse(result['remaining_state']['elapsed_fraction_used'])
        main=v.forecast_suffix(prefix,object())
        self.assertEqual(main['reason'],'MAIN_LIFECYCLE_CONTINUATION_NOT_IMPLEMENTED')

    def test_future_feedback_rewind_gap_and_stale_or_wrong_profile_cursor_rejected(self):
        original=session()
        with self.assertRaisesRegex(ValueError,'Future'):
            v.advance(original,50*MS,[start(1,'a','s',60*MS)])
        prefix=v.advance(original,50*MS,[start(1,'a','s',0)])
        with self.assertRaisesRegex(ValueError,'backward'): v.advance(prefix,40*MS)
        with self.assertRaisesRegex(ValueError,'sequence'): v.advance(prefix,60*MS,[start(3,'a','s',0)])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for ref in (cursor(root,'a','s',.5,40*MS),cursor(root,'a','g',.5,50*MS)):
                with self.assertRaises(ValueError):
                    v.forecast_suffix(prefix,model(),cursor_evidence={'a':ref},allow_cpu_fixture=True)
            ref=cursor(root,'a','s',.5,50*MS)
            with self.assertRaisesRegex(ValueError,'directly observed'):
                v.forecast_suffix(prefix,model(),cursor_evidence={'a':ref})
        frozen=v.new_session(rows(),f.aliases(),'fixture',0,1000*MS,2000*MS,mode='frozen',frozen_cutoff_ns=50*MS)
        frozen=v.advance(frozen,50*MS,[start(1,'a','s',0)])
        with self.assertRaisesRegex(ValueError,'Frozen'):
            v.advance(frozen,51*MS)


if __name__=='__main__':unittest.main()
