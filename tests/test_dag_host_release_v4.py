import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.dag_host_release_v4 import (
    SCHEMA,SCENARIOS,DagHostReleaseProvider,aggregate,fit_observation,structure)
from tools.predictor_validation_20261027.cuda_dag_simulator_v4 import DagFullHorizonModel


ALIASES={'a':dict(family='family',supported_tasks=['task'])}
ROW=dict(request_id='a',alias='a',task='task',pool_id='input',arrival_ns=0)


def profile():
    return dict(alias='a',family='family',task='task',pool_id='input',request_id='train',
        cyclic=False,unresolved=[],nodes=[
            dict(id='x',kind='gpu',duration_ns=10,release_ns=999,issue_start_ns=100,issue_end_ns=101,issue_thread=1),
            dict(id='y',kind='gpu',duration_ns=20,release_ns=999,issue_start_ns=102,issue_end_ns=103,issue_thread=1),
            dict(id='end',kind='handler_end',duration_ns=0,release_ns=999)],
        edges=[dict(before=n,after='end') for n in ('x','y')])


def natural(budget=100,tail=7):
    return dict(worker_response=dict(start_perf_ns=1000,end_perf_ns=1000+budget+tail,
        observed_stages=[dict(id='final_sync',end_ns=1000+budget)]))


def document(p=None):
    row=fit_observation(p or profile(),natural())
    rows=[dict(row,source='capture1',pool_id='input'),dict(row,source='capture2',pool_id='input2')]
    return dict(schema=SCHEMA,execution_context_sha256='natural',groups=aggregate(rows))


def provider(p=None,doc=None,scenario=SCENARIOS[0],**options):
    p=p or profile()
    return DagHostReleaseProvider(doc or document(p),'natural',{('a','task','input'):p},ALIASES,
        scenario=scenario,allow_identification_hypothesis=True,allow_cross_input_pooling_hypothesis=True,**options)


class HostSupplyTests(unittest.TestCase):
    def test_multistream_cp_not_duration_sum_and_shapes_reproduce_training_only(self):
        p=profile();observation=fit_observation(p,natural())
        self.assertEqual(observation['profiled_dag_critical_path_ns'],20)
        self.assertEqual(observation['activity_duration_sum_ns'],30)
        self.assertFalse(observation['training_endpoint_identity_is_prediction_accuracy'])
        releases=[]
        for scenario in SCENARIOS:
            release=provider(p,scenario=scenario)
            result=DagFullHorizonModel(ALIASES,{('a','task','input'):p},
                duration_provider=lambda n,r:n['duration_ns'],release_provider=release).forecast([ROW],0,1000)
            self.assertEqual(result['finishes']['a']['handler_end_ns'],107)
            releases.append(release(p['nodes'][0],ROW))
        self.assertNotEqual(*releases)

    def test_gpu_changes_propagate_without_endpoint_retuning_and_terminal_is_relative(self):
        p=profile();release=provider(p)
        class Lifecycle:
            def delays(self,*args):return dict(known=True,parent_receipt_ms=.000005,worker_next_ready_ms=.000003)
        rows=[ROW,dict(ROW,request_id='next',arrival_ns=1)]
        result=DagFullHorizonModel(ALIASES,{('a','task','input'):p},
            duration_provider=lambda n,r:n['duration_ns']*2,release_provider=release,
            lifecycle_provider=Lifecycle()).forecast(rows,0,1000)
        self.assertEqual(result['finishes']['a']['handler_end_ns'],127)
        self.assertEqual(result['finishes']['a']['finish_ns'],132)
        self.assertEqual(result['starts']['next'],130)
        self.assertEqual(result['queue_wait_ns']['next'],129)
        self.assertEqual(result['max_simultaneous_nodes_per_owner']['a'],2)

    def test_query_ignores_target_timings_and_absolute_issue_offsets(self):
        p=profile();doc=document(p);expected=provider(p,doc)
        poisoned=copy.deepcopy(p)
        for node in poisoned['nodes']:
            node['duration_ns']=None;node['release_ns']=-10**20
            node['observed_handler_end_ns']='forbidden'
            if 'issue_start_ns' in node:
                node['issue_start_ns']+=10**15;node['issue_end_ns']+=10**15
        actual=provider(poisoned,doc)
        for original,node in zip(p['nodes'],poisoned['nodes']):
            self.assertEqual(expected(original,ROW),actual(node,ROW))

    def test_overlap_retains_partial_order_and_no_arbitrary_serialization(self):
        p=profile();x,y=p['nodes'][:2]
        x.update(issue_start_ns=100,issue_end_ns=110)
        y.update(issue_start_ns=101,issue_end_ns=103)
        features=structure(p)
        self.assertEqual(features['issue_depth']['x'],1)
        self.assertEqual(features['issue_depth']['y'],1)
        p['nodes'].insert(2,dict(x,id='z',issue_start_ns=110,issue_end_ns=111))
        p['edges'].append(dict(before='z',after='end'))
        self.assertEqual(structure(p)['issue_depth']['z'],2)

    def test_missing_or_inconsistent_natural_bounds_unknown(self):
        row=natural();row['worker_response']['observed_stages']=[]
        with self.assertRaisesRegex(UnknownEvidence,'NATURAL_FINAL_SYNC'):fit_observation(profile(),row)
        with self.assertRaisesRegex(UnknownEvidence,'PRECEDES_PROFILED_DAG'):fit_observation(profile(),natural(budget=19))

    def test_pool_support_and_optins_required(self):
        p=profile();observation=fit_observation(p,natural())
        for rows in ([dict(observation,source='capture1')],
                     [dict(observation,source='capture1'),dict(observation,source='capture2')]):
            self.assertFalse(aggregate(rows)[0]['supported'])
        with self.assertRaisesRegex(ValueError,'opt-ins'):
            DagHostReleaseProvider(document(),'natural',{('a','task','input'):p},ALIASES,scenario=SCENARIOS[0])
        doc=document();doc['groups'][0]['supported']=False
        with self.assertRaisesRegex(UnknownEvidence,'NO_SUPPORTED_POOLED'):provider(p,doc)(p['nodes'][0],ROW)

    def test_invalid_explicit_release_is_unknown(self):
        for value in (dict(not_before_offset_ns=0),dict(not_before_offset_ns=0,after_predecessors_ns=-1),
                      dict(not_before_offset_ns=True,after_predecessors_ns=0)):
            with self.assertRaises(UnknownEvidence):
                DagFullHorizonModel(ALIASES,{('a','task','input'):profile()},
                    duration_provider=lambda n,r:10,release_provider=lambda n,r:value).forecast([ROW],0,1000)

    def test_resource_integral_uses_handler_state_full_window_and_memory_through_drain(self):
        class Resources:
            def power(self,active,*args):return dict(known=True,watts=100 if active else 10)
            def memory(self,active,*args):return dict(known=True,peak_memory_mib=200 if active else 100)
        p=profile();resource=Resources()
        result=DagFullHorizonModel(ALIASES,{('a','task','input'):p},
            duration_provider=lambda n,r:n['duration_ns'],release_provider=provider(p),
            power_provider=resource,memory_provider=resource).forecast([ROW],0,1000,drain_deadline_ns=2000)
        self.assertAlmostEqual(result['board_energy_joules'],(107*100+893*10)/1e9)
        self.assertEqual(result['peak_memory_mib'],200)
        self.assertEqual(result['resource_transitions'][0]['handler_active_aliases'],['a'])
        self.assertEqual(result['resource_transitions'][-1]['until_ns'],1000)
        self.assertFalse(result['memory_safe_factor_applied'])
        # A request may finish after the scoring window; memory still sees it.
        result=DagFullHorizonModel(ALIASES,{('a','task','input'):p},
            duration_provider=lambda n,r:n['duration_ns'],release_provider=provider(p),
            power_provider=resource,memory_provider=resource).forecast([ROW],0,50,drain_deadline_ns=2000)
        self.assertAlmostEqual(result['board_energy_joules'],50*100/1e9)
        self.assertEqual(result['memory_window_ns'],[0,107])
        self.assertIsNone(result['resource_transitions'][-1]['board_watts'])

    def test_unknown_resource_state_is_not_zero_and_strict_gate_raises(self):
        class Resources:
            def power(self,*args):return dict(known=False,watts=None)
            def memory(self,*args):return dict(known=False,peak_memory_mib=None)
        p=profile();resource=Resources()
        options=dict(duration_provider=lambda n,r:n['duration_ns'],release_provider=provider(p),
                     power_provider=resource,memory_provider=resource)
        model=DagFullHorizonModel(ALIASES,{('a','task','input'):p},**options)
        result=model.forecast([ROW],0,1000)
        self.assertIsNone(result['board_energy_joules']);self.assertIsNone(result['peak_memory_mib'])
        for strict in ('require_energy','require_memory'):
            model=DagFullHorizonModel(ALIASES,{('a','task','input'):p},**options,**{strict:True})
            with self.assertRaises(UnknownEvidence):model.forecast([ROW],0,1000)


if __name__=='__main__':unittest.main()
