"""Atomic continuation must conserve the full original numerical forecast."""
import copy
import functools
import json
from pathlib import Path
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.cuda_dag_simulator_v6 import DagFullHorizonModel as V6
from tools.predictor_validation_20261027.cuda_dag_simulator_v7 import DagFullHorizonModel
from tools.predictor_validation_20261027.dag_checkpoint_v7 import decode, digest, encode, receipt
from tools.predictor_validation_20261027.dag_queue_diagnostics_v6 import DagQueueDiagnostics


def duration(node, row):
    return node['duration_ns']


def scaled_duration(node, row, scale=1):
    return scale * node['duration_ns']


class ConfiguredDuration:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, node, row):
        return node['duration_ns'] * self.scale


def release(node, row):
    return dict(not_before_offset_ns=node.get('offset_ns', 0),
                after_predecessors_ns=node.get('delay_ns', 0))


class Lifecycle:
    def delays(self, *args):
        return dict(known=True, parent_receipt_ms=.000007, worker_next_ready_ms=.000003)


class Resources:
    def power(self, rows, *args):
        return dict(known=True, watts=20. + 2 * len(rows))

    def memory(self, rows, *args):
        return dict(known=True, peak_memory_mib=30. + 4 * len(rows))


class AtomicCheckpointTests(unittest.TestCase):
    def inputs(self, shift=0):
        aliases = {a:dict(family='text', supported_tasks=['text']) for a in ('a','b')}
        plan = dict(nodes=[dict(id='x',kind='gpu',duration_ns=10.25),
                           dict(id='y',kind='gpu',duration_ns=20.25,offset_ns=2),
                           dict(id='copy',kind='copy',duration_ns=2.1),
                           dict(id='end',kind='handler_end',delay_ns=2)],
                    edges=[dict(before='x',after='copy'),dict(before='copy',after='end'),
                           dict(before='y',after='end')],unresolved=[],cyclic=False)
        plans = {(a,'text','input'):copy.deepcopy(plan) for a in aliases}
        rows = [dict(request_id=rid,alias=a,task='text',pool_id='input',arrival_ns=shift+at)
                for rid,a,at in [('a1','a',0),('b1','b',5),('a2','a',7),('b2','b',26)]]
        return aliases,plans,rows

    def engine(self, shift=0, budget=100, diagnostic=True, old=False):
        aliases,plans,rows = self.inputs(shift)
        diag = DagQueueDiagnostics(aliases,plans,duration,now_ns=shift) if diagnostic else None
        options=dict(duration_provider=diag.duration if diag else duration,release_provider=release,
                     lifecycle_provider=Lifecycle(),power_provider=Resources(),memory_provider=Resources(),
                     require_energy=True,require_memory=True,queue_diagnostics=diag,
                     allow_context_time_slice_hypothesis=True,allow_independent_copy_hypothesis=True,
                     max_events=budget)
        if not old:
            options['checkpoint_binding']=dict(execution_context_sha256='a'*64,evidence=[receipt(__file__)])
        return (V6 if old else DagFullHorizonModel)(aliases,plans,**options),rows

    def test_every_natural_atomic_boundary_restores_exact_full_result_including_diagnostics(self):
        for shift in (0,30_000_000_000,10**18):
            engine,rows=self.engine(shift)
            expected=engine.forecast(rows,shift,shift+60,drain_deadline_ns=shift+200)
            old,rows=self.engine(shift,old=True)
            self.assertEqual(expected,old.forecast(rows,shift,shift+60,drain_deadline_ns=shift+200))
            for event in range(1,expected['events']):
                with self.subTest(shift=shift,event=event):
                    first,rows=self.engine(shift)
                    paused=first.forecast(rows,shift,shift+60,drain_deadline_ns=shift+200,
                                          pause_after_event_count=event)
                    if 'checkpoint' not in paused:
                        self.assertEqual(paused,expected)
                        continue
                    checkpoint=json.loads(json.dumps(paused['checkpoint'],allow_nan=False))
                    self.assertFalse(checkpoint['physical_progress_observed'])
                    resumed,rows=self.engine(shift)
                    actual=resumed.forecast(rows,shift,shift+60,drain_deadline_ns=shift+200,
                                            checkpoint=checkpoint)
                    self.assertEqual(actual,expected)

    def test_repeated_serialized_continuation_preserves_original_budget_and_cumulative_energy(self):
        model,rows=self.engine()
        expected=model.forecast(rows,0,60,drain_deadline_ns=200)
        checkpoint=None
        for event in range(1,expected['events']+1):
            model,rows=self.engine()
            result=model.forecast(rows,0,60,drain_deadline_ns=200,checkpoint=checkpoint,
                                  pause_after_event_count=event)
            if 'checkpoint' not in result:
                self.assertEqual(result,expected)
                break
            checkpoint=json.loads(json.dumps(result['checkpoint']))
            self.assertEqual(decode(checkpoint['state'])['event_count'],event)
        else:
            self.fail('Repeated continuation did not finish')

    def paused(self, event=3, budget=100, diagnostic=True):
        model,rows=self.engine(budget=budget,diagnostic=diagnostic)
        return model.forecast(rows,0,60,drain_deadline_ns=200,pause_after_event_count=event)['checkpoint']

    def test_checkpoint_requires_explicit_binding_and_does_not_invent_progress(self):
        model,rows=self.engine();model.checkpoint_binding=None
        with self.assertRaisesRegex(ValueError,'execution identity'):
            model.forecast(rows,0,60,pause_after_event_count=1)

    def test_changed_provider_evidence_window_plan_route_or_options_rejected(self):
        checkpoint=self.paused()
        for change in ('identity','deadline','drain','plan','route','option','evidence'):
            with self.subTest(change=change):
                model,rows=self.engine();deadline=60;drain=200
                if change=='identity':model.checkpoint_binding['execution_context_sha256']='b'*64
                if change=='deadline':deadline=61
                if change=='drain':drain=201
                if change=='plan':model.plans['a','text','input']['nodes'][0]['duration_ns']=99.
                if change=='route':rows[0]['alias']='b'
                if change=='option':model.allow_context_time_slice_hypothesis=False
                if change=='evidence':model.checkpoint_binding['evidence'][0]['sha256']='0'*64
                with self.assertRaises(ValueError):
                    model.forecast(rows,0,deadline,drain_deadline_ns=drain,checkpoint=checkpoint)

    def test_checksum_and_duplicate_lifecycle_state_rejected(self):
        checkpoint=self.paused()
        bad=copy.deepcopy(checkpoint);bad['physical_progress_observed']=True
        model,rows=self.engine()
        with self.assertRaisesRegex(ValueError,'Changed/foreign'):
            model.forecast(rows,0,60,drain_deadline_ns=200,checkpoint=bad)
        bad=copy.deepcopy(checkpoint);state=decode(bad['state'])
        running=next(iter(state['running'].values()))['row']
        state['pending'][running['alias']].append(copy.deepcopy(running))
        bad['state']=encode(state);bad['checkpoint_sha256']=digest({k:v for k,v in bad.items() if k!='checkpoint_sha256'})
        model,rows=self.engine()
        with self.assertRaisesRegex(ValueError,'loses, duplicates'):
            model.forecast(rows,0,60,drain_deadline_ns=200,checkpoint=bad)

    def test_budget_is_not_reset_by_continuation(self):
        checkpoint=self.paused(event=3,budget=3)
        model,rows=self.engine(budget=3)
        with self.assertRaisesRegex(UnknownEvidence,'EVENT_BUDGET'):
            model.forecast(rows,0,60,drain_deadline_ns=200,checkpoint=checkpoint)

    def test_no_diagnostic_path_and_bounded_drain_conserve_unfinished_requests(self):
        model,rows=self.engine(diagnostic=False)
        expected=model.forecast(rows,0,60,drain_deadline_ns=61)
        model,rows=self.engine(diagnostic=False)
        first=model.forecast(rows,0,60,drain_deadline_ns=61,pause_after_event_count=3)
        model,rows=self.engine(diagnostic=False)
        self.assertEqual(model.forecast(rows,0,60,drain_deadline_ns=61,checkpoint=first['checkpoint']),expected)
        self.assertLess(len(expected['finishes']),len(rows))

    def test_selected_callable_partial_and_object_configuration_bound_even_with_same_source_file(self):
        for old,new in ((duration,scaled_duration),
                        (functools.partial(scaled_duration,scale=1),functools.partial(scaled_duration,scale=2)),
                        (ConfiguredDuration(1),ConfiguredDuration(2))):
            for diagnostic in (False,True):
                with self.subTest(provider=type(old).__name__,diagnostic=diagnostic):
                    model,rows=self.engine(diagnostic=diagnostic)
                    if diagnostic:
                        model.queue_diagnostics.duration_provider=old
                    else:
                        model.duration_provider=old
                    paused=model.forecast(rows,0,60,drain_deadline_ns=200,pause_after_event_count=3)
                    model,rows=self.engine(diagnostic=diagnostic)
                    if diagnostic:
                        model.queue_diagnostics.duration_provider=new
                    else:
                        model.duration_provider=new
                    with self.assertRaisesRegex(ValueError,'Changed/foreign'):
                        model.forecast(rows,0,60,drain_deadline_ns=200,checkpoint=paused['checkpoint'])

    def test_self_inconsistent_arrival_and_duration_cache_rejected_after_rechecksum(self):
        for defect in ('arrival','cache','registered','remaining'):
            checkpoint=self.paused(event=1)
            if defect=='arrival':
                state=decode(checkpoint['state'])
                at,index,row=state['arrivals'][0]
                state['arrivals'][0]=(at+1,index,row)
                checkpoint['state']=encode(state)
            else:
                diag=decode(checkpoint['diagnostics'])
                if defect=='cache':
                    profile=next(iter(diag['cache'].values()))
                    profile['durations']['x']*=10
                if defect=='registered':diag['registered'].add('unoffered-request')
                if defect=='remaining':diag['remaining']['gpu']+=10
                checkpoint['diagnostics']=encode(diag)
            checkpoint['checkpoint_sha256']=digest({k:v for k,v in checkpoint.items() if k!='checkpoint_sha256'})
            model,rows=self.engine()
            with self.subTest(defect=defect),self.assertRaises(ValueError):
                model.forecast(rows,0,60,drain_deadline_ns=200,checkpoint=checkpoint)

    def test_resource_totals_flags_and_history_rejected_after_rechecksum(self):
        for defect in ('energy', 'peak', 'energy_known', 'memory_known', 'gap', 'watts', 'owners'):
            checkpoint = self.paused(event=3)
            state = decode(checkpoint['state'])
            if defect in ('energy', 'peak'):
                state[defect] += 123.
            elif defect in ('energy_known', 'memory_known'):
                state[defect] = False
            elif defect == 'gap':
                state['resource_transitions'][0]['time_ns'] += 1
            elif defect == 'owners':
                state['resource_transitions'][0]['handler_active_aliases'] = []
            else:
                # A consistent forged sum must still fail fresh provider review.
                row = state['resource_transitions'][0]
                row['board_watts'] += 10
                state['energy'] += 10 * (row['until_ns'] - row['time_ns']) / 1e9
            checkpoint['state'] = encode(state)
            checkpoint['checkpoint_sha256'] = digest({k:v for k,v in checkpoint.items() if k != 'checkpoint_sha256'})
            model, rows = self.engine()
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                model.forecast(rows,0,60,drain_deadline_ns=200,checkpoint=checkpoint)

    def test_unknown_resource_prefix_remains_unknown_after_continuation(self):
        def configure():
            model, rows = self.engine()
            model.require_energy = model.require_memory = False
            model.power_provider = model.memory_provider = None
            return model, rows
        model, rows = configure()
        expected = model.forecast(rows,0,60,drain_deadline_ns=200)
        model, rows = configure()
        checkpoint = model.forecast(rows,0,60,drain_deadline_ns=200,pause_after_event_count=3)['checkpoint']
        model, rows = configure()
        result = model.forecast(rows,0,60,drain_deadline_ns=200,checkpoint=checkpoint)
        self.assertEqual(result, expected)
        self.assertIsNone(result['board_energy_joules'])
        self.assertIsNone(result['peak_memory_mib'])

    def test_resource_interval_cannot_hide_later_handler_start(self):
        checkpoint = self.paused(event=3)
        state = decode(checkpoint['state'])
        # b1 starts at 5; a single [0,7) interval labelled a conceals it.
        self.assertEqual(state['starts']['b1'], 5)
        state['resource_transitions'] = [dict(time_ns=0, until_ns=7,
            handler_active_aliases=['a'], board_watts=22., peak_memory_mib=34.)]
        state['energy'], state['peak'] = 22 * 7 / 1e9, 34.
        checkpoint['state'] = encode(state)
        checkpoint['checkpoint_sha256'] = digest({k:v for k,v in checkpoint.items() if k != 'checkpoint_sha256'})
        model, rows = self.engine()
        with self.assertRaisesRegex(ValueError, 'crosses a handler boundary'):
            model.forecast(rows,0,60,drain_deadline_ns=200,checkpoint=checkpoint)


if __name__=='__main__':
    unittest.main()
