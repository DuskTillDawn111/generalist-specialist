"""Explicit CPU clock hypotheses; no real fitting/native timing claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import dag_node_rate_engine_v11 as m
from tools.predictor_validation_20261027 import dag_node_rate_checkpoint_v11 as cp
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import SchedulingDecision
from tools.predictor_validation_20261027.dag_node_rate_scheduler_v11 import CheckpointNodeScheduler


class SyntheticLaw:
    def __init__(self,mode='peer'):
        self.mode=mode;self.history=[]
    def decide(self,view):
        self.history.append([view.clock_ns,[[list(n.key),n.remaining_reference_work_ns] for n in view.active_gpu_nodes]])
        rate=(0 if self.mode=='zero' else 2 if len(view.active_gpu_nodes)==1 else 1)
        if self.mode=='wait_peer':rate=2 if len(view.active_gpu_nodes)>1 else 0
        if self.mode=='nan':rate=float('nan')
        transition=3 if self.mode=='timer' and view.clock_ns<3 else None
        if transition is not None:rate=0
        return SchedulingDecision({n.key:rate for n in view.active_gpu_nodes},transition)
    def fresh(self):return SyntheticLaw(self.mode)
    def saved_state(self):return dict(history=copy.deepcopy(self.history))
    def restore_state(self,state):
        if type(state) is not dict or set(state)!={'history'}:raise ValueError('state')
        self.history=copy.deepcopy(state['history'])
    def checkpoint_configuration(self):return dict(kind='synthetic_CPU_fixture',mode=self.mode)
    def provenance(self):return dict(kind='synthetic_CPU_fixture',history=copy.deepcopy(self.history))


def duration(node,row):return node['reference_ns']
def release(node,row):return node.get('supply_ns',0)
def phase(node,row):return node['id']
def start(row):return row['arrival_ns']
def delayed_start(row):return row.get('host_start_ns',row['arrival_ns'])
def copy_rate(node,row,clock,active):return dict(rate=2,next_transition_ns=None)


class Lifecycle:
    def delays(self,row,aliases,scenario):
        return dict(known=True,parent_receipt_ms=0.,worker_next_ready_ms=0.)


def plan(nodes,edges=()):
    return dict(nodes=[dict(id=n,kind=k,reference_ns=w) for n,k,w in nodes],
        edges=[dict(before=a,after=b) for a,b in edges],unresolved=[],cyclic=False)


def row(rid,owner,arrival=0):return dict(request_id=rid,alias=owner,task='text',pool_id='x',arrival_ns=arrival)


class NodeRateEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        source=Path(self.temp.name)/'cpu.json';source.write_text('{"role":"synthetic"}')
        self.evidence=cp.base.receipt(source)
    def engine(self,plans,mode='peer',non_gpu=None,**kwargs):
        aliases={k[0]:dict(family=k[0],supported_tasks=['text']) for k in plans}
        start_provider=kwargs.pop('request_start',start)
        return m.DagNodeRateCheckpointModel(aliases,plans,
            scheduler=CheckpointNodeScheduler(SyntheticLaw(mode)),duration_provider=duration,
            phase_provider=phase,release_provider=release,request_start_provider=start_provider,lifecycle_provider=Lifecycle(),
            non_gpu_rate_provider=non_gpu,allow_independent_copy_hypothesis=non_gpu is not None,
            checkpoint_binding=dict(execution_context_sha256='a'*64,evidence=[self.evidence]),**kwargs)

    def test_same_owner_parallel_nodes_reprice_on_peer_completion_conserve_work(self):
        plans={('a','text','x'):plan([('g0','gpu',8),('g1','gpu',12)])}
        result=self.engine(plans).forecast([row('r','a')],0,20,drain_deadline_ns=30)
        self.assertEqual(result['finishes']['r']['finish_ns'],10)
        self.assertEqual(result['max_simultaneous_nodes_per_owner']['a'],2)
        history=result['scheduler_provenance']['provider']['history']
        self.assertIn([8,[[['a','g1'],4.]]],history)
        for work in result['reference_work_accounting'].values():
            self.assertEqual(work['remaining_ns'],0)
            self.assertEqual(work['reference_ns'],work['consumed_ns'])
        self.assertFalse(result['target_future_END_read'])

    def test_future_arrival_reprices_already_running_work(self):
        plans={('a','text','x'):plan([('g','gpu',10)]),('b','text','x'):plan([('g','gpu',4)])}
        result=self.engine(plans).forecast([row('a','a'),row('b','b',3)],0,20,drain_deadline_ns=30)
        self.assertEqual({k:r['finish_ns'] for k,r in result['finishes'].items()},{'a':7,'b':7})
        history=result['scheduler_provenance']['provider']['history']
        self.assertIn([3,[[['a','g'],4.],[['b','g'],4.]]],history)

    def test_dependency_release_and_explicit_copy_law(self):
        plans={('a','text','x'):plan([('g','gpu',8),('copy','copy',4),('next','gpu',4)],
            [('g','copy'),('copy','next')])}
        result=self.engine(plans,non_gpu=copy_rate).forecast([row('r','a')],0,20,drain_deadline_ns=30)
        self.assertEqual(result['finishes']['r']['finish_ns'],8)
        self.assertEqual(result['completed_dag_nodes'],3)
        broken=self.engine(plans);broken.allow_independent_copy_hypothesis=True
        with self.assertRaisesRegex(UnknownEvidence,'COPY_MEMSET_REFERENCE_RATE'):
            broken.forecast([row('r','a')],0,20,drain_deadline_ns=30)

    def test_timer_zero_rate_released_and_permanent_zero_unknown(self):
        plans={('a','text','x'):plan([('g','gpu',8)])}
        result=self.engine(plans,mode='timer').forecast([row('r','a')],0,20,drain_deadline_ns=30)
        self.assertEqual(result['finishes']['r']['finish_ns'],7)
        for mode in ('zero','nan'):
            with self.subTest(mode=mode),self.assertRaises(UnknownEvidence):
                self.engine(plans,mode=mode).forecast([row('r','a')],0,20,drain_deadline_ns=30)

    def test_zero_frontier_uses_future_arrival_or_pending_host_start(self):
        plans={('a','text','x'):plan([('g','gpu',8)]),('b','text','x'):plan([('g','gpu',8)])}
        for pending_host in (False,True):
            second=dict(row('b','b',0 if pending_host else 3),host_start_ns=3)
            with self.subTest(pending_host=pending_host):
                result=self.engine(plans,mode='wait_peer',request_start=delayed_start).forecast(
                    [row('a','a'),second],0,20,drain_deadline_ns=30)
                self.assertEqual({key:value['finish_ns'] for key,value in result['finishes'].items()},
                    {'a':7,'b':7})

    def test_checkpoint_all_atomic_pauses_preserve_full_future_and_provider_state(self):
        plans={('a','text','x'):plan([('g0','gpu',8),('g1','gpu',12)]),
            ('b','text','x'):plan([('g','gpu',4)])}
        rows=[row('a','a'),row('b','b',3),row('a2','a',4)]
        expected=self.engine(plans).forecast(rows,0,30,drain_deadline_ns=40)
        for consumed in range(1,expected['events']):
            paused=self.engine(plans).forecast(rows,0,30,drain_deadline_ns=40,pause_after_event_count=consumed)
            if 'checkpoint' not in paused:continue
            saved=json.loads(json.dumps(paused['checkpoint']))
            result=self.engine(plans).forecast(rows,0,30,drain_deadline_ns=40,checkpoint=saved)
            self.assertEqual(result,expected,consumed)

    def test_resigned_work_or_provider_state_tampering_cannot_resume(self):
        plans={('a','text','x'):plan([('g0','gpu',8),('g1','gpu',12)])}
        rows=[row('r','a')]
        saved=self.engine(plans).forecast(rows,0,20,drain_deadline_ns=30,pause_after_event_count=1)['checkpoint']
        for kind in ('state','scheduler_state'):
            bad=copy.deepcopy(saved);value=cp.base.decode(bad[kind])
            if kind=='state':value['node_work']['r','g1']['remaining_ns']-=1
            else:value['provider_state']['history']=[]
            bad[kind]=cp.base.encode(value);bad['checkpoint_sha256']=cp.base.digest({k:v for k,v in bad.items() if k!='checkpoint_sha256'})
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                self.engine(plans).forecast(rows,0,20,drain_deadline_ns=30,checkpoint=bad)

    def test_fixed_drain_keeps_unfinished_work_and_inclusive_completion(self):
        for work,complete in ((10,True),(20,False)):
            plans={('a','text','x'):plan([('g','gpu',work)])}
            result=self.engine(plans).forecast([row('r','a')],0,4,drain_deadline_ns=5)
            self.assertEqual('r' in result['finishes'],complete)
            ledger=result['reference_work_accounting']['r','g']
            self.assertEqual(ledger['consumed_ns']+ledger['remaining_ns'],work)
            self.assertEqual(ledger['consumed_ns'],10)
            self.assertEqual(len(result['cohort']),1)

    def test_changed_provider_configuration_or_source_cannot_resume(self):
        plans={('a','text','x'):plan([('g','gpu',10)])};rows=[row('r','a')]
        saved=self.engine(plans).forecast(rows,0,20,drain_deadline_ns=30,pause_after_event_count=1)['checkpoint']
        with self.assertRaisesRegex(ValueError,'immutable provider contract'):
            self.engine(plans,mode='timer').forecast(rows,0,20,drain_deadline_ns=30,checkpoint=saved)
        Path(self.evidence['path']).write_text('{"changed":true}')
        with self.assertRaises(ValueError):self.engine(plans).forecast(rows,0,20,drain_deadline_ns=30,checkpoint=saved)


if __name__=='__main__':unittest.main()
