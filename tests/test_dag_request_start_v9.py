"""Host request-start supply delays execution without rewriting public arrivals."""
import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.cuda_dag_simulator_v9 import DagFullHorizonModel
from tools.predictor_validation_20261027.dag_checkpoint_v9 import receipt
from tools.predictor_validation_20261027.dag_checkpoint_v7 import digest
from tests import test_dag_checkpoint_v7 as checkpoint_tests


class StartSupply:
    def __init__(self,bounds):self.bounds=copy.deepcopy(bounds)
    def __call__(self,row):return self.bounds.get(row['request_id'],row['arrival_ns'])


class ReadyLagSupply:
    def __init__(self,lag):self.lag=lag
    def __call__(self,row):return row['arrival_ns']+self.lag
    def from_owner_ready(self,row,*,owner_ready_ns):return max(row['arrival_ns'],owner_ready_ns)+self.lag


class RequestStartTests(unittest.TestCase):
    def model(self,bounds=None,shift=0,diagnostic=True):
        old,rows=checkpoint_tests.AtomicCheckpointTests().engine(shift,diagnostic=diagnostic)
        model=DagFullHorizonModel(old.aliases,old.plans,
            allow_context_time_slice_hypothesis=old.allow_context_time_slice_hypothesis,
            allow_independent_copy_hypothesis=old.allow_independent_copy_hypothesis,
            lifecycle_provider=old.lifecycle_provider,duration_provider=old.duration_provider,
            release_provider=old.release_provider,power_provider=old.power_provider,memory_provider=old.memory_provider,
            require_energy=old.require_energy,require_memory=old.require_memory,queue_diagnostics=old.queue_diagnostics,
            checkpoint_binding=dict(execution_context_sha256='a'*64,evidence=[receipt(__file__)]),
            request_start_provider=StartSupply(bounds) if bounds is not None else None)
        return model,rows

    def test_default_none_full_result_is_field_for_field_equal_v7(self):
        for shift in (0,10**18):
            for diagnostic in (False,True):
                old,rows=checkpoint_tests.AtomicCheckpointTests().engine(shift,diagnostic=diagnostic)
                new,rows=self.model(shift=shift,diagnostic=diagnostic)
                self.assertEqual(old.forecast(rows,shift,shift+60,drain_deadline_ns=shift+200),
                    new.forecast(rows,shift,shift+60,drain_deadline_ns=shift+200))

    def test_late_dispatch_retains_original_arrival_wait_and_owner_fifo_head_blocks(self):
        model,rows=self.model({'a1':40,'a2':7})
        value=model.forecast(rows,0,60,drain_deadline_ns=200)
        self.assertEqual(value['starts']['a1'],40)
        self.assertEqual(value['queue_wait_ns']['a1'],40)
        self.assertEqual(value['cohort']['a1']['arrival_ns'],0)
        self.assertGreater(value['starts']['a2'],40)
        self.assertEqual(value['cohort']['a2']['arrival_ns'],7)
        self.assertEqual(value['queue_wait_ns']['a2'],value['starts']['a2']-7)
        self.assertLess(value['starts']['b1'],40)

    def test_every_actual_boundary_continues_with_new_host_supply_and_diagnostics(self):
        model,rows=self.model({'a1':40,'a2':7})
        expected=model.forecast(rows,0,60,drain_deadline_ns=200)
        for event in range(1,expected['events']):
            model,rows=self.model({'a1':40,'a2':7})
            paused=model.forecast(rows,0,60,drain_deadline_ns=200,pause_after_event_count=event)
            if 'checkpoint' not in paused:continue
            model,rows=self.model({'a1':40,'a2':7})
            actual=model.forecast(rows,0,60,drain_deadline_ns=200,checkpoint=paused['checkpoint'])
            self.assertEqual(actual,expected)

    def test_changed_start_provider_configuration_or_source_rejected(self):
        model,rows=self.model({'a1':40})
        cp=model.forecast(rows,0,60,drain_deadline_ns=200,pause_after_event_count=1)['checkpoint']
        self.assertEqual(cp['schema'],'gsdeploy.simulated_dag_atomic_checkpoint.v9')
        self.assertIn('request_start',cp['contract']['providers'])
        for kind in ('bound','source','context'):
            model,rows=self.model({'a1':41 if kind=='bound' else 40})
            bad=copy.deepcopy(cp)
            if kind=='source':
                bad['contract']['sources'][0]['sha256']='0'*64
                bad['checkpoint_sha256']=digest({k:v for k,v in bad.items() if k!='checkpoint_sha256'})
            if kind=='context':model.checkpoint_binding['execution_context_sha256']='b'*64
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                model.forecast(rows,0,60,drain_deadline_ns=200,checkpoint=bad)

    def test_unknown_start_supply_is_unknown_instead_of_zero_or_arrival(self):
        model,rows=self.model({'a1':None})
        with self.assertRaisesRegex(UnknownEvidence,'REQUEST_START_SUPPLY_REQUIRED'):
            model.forecast(rows,0,60,drain_deadline_ns=200)

    def test_busy_owner_ready_plus_lag_is_added_after_fifo_wait_and_resumes_exactly(self):
        model,rows=self.model()
        model.aliases={'a':model.aliases['a']}
        model.plans={k:v for k,v in model.plans.items() if k[0]=='a'}
        model.queue_diagnostics=None
        model.request_start_provider=ReadyLagSupply(5)
        rows=[dict(request_id='a1',alias='a',task='text',pool_id='input',arrival_ns=0),
              dict(request_id='a2',alias='a',task='text',pool_id='input',arrival_ns=0)]
        expected=model.forecast(rows,0,60,drain_deadline_ns=200)
        first=expected['finishes']['a1'];second=expected['finishes']['a2']
        self.assertEqual(first['start_ns'],5)
        self.assertEqual(second['start_ns'],first['worker_next_ready_ns']+5)
        self.assertEqual(expected['queue_wait_ns']['a2'],second['start_ns'])
        self.assertEqual(ReadyLagSupply(5).from_owner_ready(rows[1],owner_ready_ns=100),105)
        for event in range(1,expected['events']):
            model.queue_diagnostics=None
            paused=model.forecast(rows,0,60,drain_deadline_ns=200,pause_after_event_count=event)
            if 'checkpoint' not in paused:continue
            self.assertEqual(model.forecast(rows,0,60,drain_deadline_ns=200,checkpoint=paused['checkpoint']),expected)


if __name__=='__main__':unittest.main()
