"""CPU-only adapter contracts; virtual timings never enter campaign evidence."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deployment_core import Workload
from deployment_search.runtime import RuntimeEvaluation, RuntimeEvaluator
from deployment_search.runtime_cache import save_json
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from .common import F, OUT, inputs, read
from .campaign import Actor, request_for_slot, round_barrier, MeasurementBank
from .report import classify_revision, classify_gap, compare


class Virtual(RuntimeEvaluator):
    def __init__(self, decreasing=False): self.count=0; self.decreasing=decreasing
    def evaluate(self, deployment, workload, routing):
        index=self.count; self.count+=1
        mean=100*(.90**index) if self.decreasing else 100.
        return RuntimeEvaluation('steady', mean, mean*1.5, 1., 1.)


class Contracts(unittest.TestCase):
    def test_original_engine_trace_preserved_and_round_barrier(self):
        from method_v1 import create, config
        from method_v2 import MethodV2, frozen_config
        catalog, hardware, roles, cal=inputs()
        block=read(F/'execution_order.json')['workloads'][0]
        w=read(F/'heldout_workloads.json')['workloads'][0]
        workload=Workload(w['lambda_rps'])
        for decreasing in [False, True]:
            with tempfile.TemporaryDirectory() as tmp:
                actors={m:Actor(m,workload,catalog,hardware,roles,cal,Path(tmp)/m,persist=False) for m in ['M1','M2']}
                virtuals={m:Virtual(decreasing) for m in actors}
                for a in actors.values():a.thread.start();a.wait_pending()
                barriers=set()
                for slot in block['slots']:
                    if slot['phase']!='search' or slot['method']=='B2':continue
                    if slot['role']=='candidate' and slot['iteration'] not in barriers:
                        commitments=round_barrier(actors,slot['iteration']);barriers.add(slot['iteration'])
                        for m,a in actors.items():
                            if not a.done:self.assertTrue(commitments[m])
                    actor=actors[slot['method']]
                    req=request_for_slot(actor,slot)
                    if req is None:continue
                    value=virtuals[actor.method].evaluate(req.deployment,req.workload,req.routing)
                    actor.return_result(req,value);actor.wait_pending()
                for m,a in actors.items():
                    self.assertIsNone(a.wait_pending());self.assertTrue(a.done);self.assertIsNone(a.error)
                    self.assertLessEqual(a.evaluator.count,6)
                    source=create(Virtual(decreasing),runtime_objective=NoiseAwareRuntimeObjective(1.,.05)) if m=='M1' else MethodV2(Virtual(decreasing),predictor=None,runtime_objective=NoiseAwareRuntimeObjective(1.,.05))
                    direct=source.search(workload,catalog,hardware,config() if m=='M1' else frozen_config(),roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=cal).to_dict()
                    self.assertEqual(a.trace['deployment'],direct['deployment'])
                    self.assertEqual(a.trace['runtime_evaluation_count'],direct['runtime_evaluation_count'])
                    self.assertEqual(a.trace['status'],direct['status'])
                    self.assertEqual([tr['ranking'] for tr in a.trace['iterations']],[tr['ranking'] for tr in direct['iterations']])
                    self.assertEqual([tr['selected_top_b'] for tr in a.trace['iterations']],[tr['selected_top_b'] for tr in direct['iterations']])
                    self.assertEqual([[c['status'] for c in tr['candidates']] for tr in a.trace['iterations']],[[c['status'] for c in tr['candidates']] for tr in direct['iterations']])

    def test_sharing_exactness_and_confirmation_role_independence(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);context=tmp/'context.json';context.write_text('{}')
            config={'deployment':{'model':1},'routing_rps':{'text':{'model':1.}},'workload_rps':{'text':1.},'random_seed':7101}
            launches=[]
            def virtual_runner(cfg,raw,log,cp,ctx):
                launches.append(str(raw));save_json(raw,dict(cfg,valid_execution=True,measurement_window={'synthetic_fixture_only':True},execution_status='completed'))
                return {'returncode':0,'timed_out':False}
            bank=MeasurementBank(virtual_runner,persist=False);bank.catalog={};bank.workload=None
            active={'method':'M1','workload':'synthetic_fixture','workload_sha256':'fixture','logical_slot':0,'phase':'search','role':'seed','protocol_sha256':'fixture','sharing_policy_sha256':'fixture'}
            with patch('module8e3_heldout_validation.campaign.summarize',return_value={'valid_runtime_repetition':True,'stability_state_v2':'steady','weighted_mean_e2e':100.}):
                for index,(method,phase,role,seed) in enumerate([('M1','search','seed',7101),('M2','search','candidate',7101),('M1','confirmation','seed',8101),('M2','confirmation','seed',8101),('M1','confirmation','final',8101),('B2','confirmation','fixed',8101)]):
                    bank.active=dict(active,method=method,phase=phase,role=role,logical_slot=index)
                    cfg=dict(config,random_seed=seed);p=tmp/f'raw_{index}.json'
                    bank.raw_run(cfg,p,tmp/f'{index}.log',tmp/f'{index}.config.json',context)
                self.assertEqual(len(launches),4)
                self.assertEqual(sum(r['cache_hit'] for r in bank.references),2)
                self.assertEqual(len(bank.states),1)
                self.assertEqual(bank.references[0]['physical_raw_id'],bank.references[1]['physical_raw_id'])
                self.assertEqual(bank.references[2]['physical_raw_id'],bank.references[3]['physical_raw_id'])
                self.assertNotEqual(bank.references[2]['physical_raw_id'],bank.references[4]['physical_raw_id'])
                changed=dict(config,random_seed=8101,routing_rps={'text':{'model':.999999999999}})
                bank.active=dict(active,phase='confirmation',role='final')
                bank.raw_run(changed,tmp/'changed.json',tmp/'changed.log',tmp/'changed.config.json',context)
                self.assertEqual(len(launches),5)

    def test_no_retry_or_substitution_on_missing_raw(self):
        bank=MeasurementBank(lambda *a:{'returncode':-1},persist=False)
        with tempfile.TemporaryDirectory() as tmp:
            ctx=Path(tmp)/'context';ctx.write_text('{}')
            bank.active={'method':'M1','workload':'fixture','workload_sha256':'f','logical_slot':0,'phase':'search','role':'seed','protocol_sha256':'p','sharing_policy_sha256':'s'}
            cfg={'deployment':{'m':1},'workload_rps':{'text':1.},'routing_rps':{'text':{'m':1.}},'random_seed':7101}
            with self.assertRaises(RuntimeError):bank.raw_run(cfg,Path(tmp)/'raw',Path(tmp)/'log',Path(tmp)/'cfg',ctx)
            self.assertEqual(len(bank.runs),1);self.assertEqual(bank.first_failure['category'],'external_interruption');self.assertFalse(bank.donors)

    def test_frozen_success_rules(self):
        base={'W':3,'T':3,'L':0,'SR':0,'severe':False,'clear_regression':False,'activated':True,'helped':True}
        self.assertEqual(classify_revision(base,True)[0],'STRONG_SUPPORT')
        self.assertEqual(classify_revision(dict(base,W=1,T=5,helped=False),True)[0],'SUPPORT')
        self.assertEqual(classify_revision(dict(base,W=0,T=6,activated=False,helped=False),True)[0],'NEUTRAL')
        self.assertEqual(classify_revision(dict(base,W=0,T=3,L=3),True)[0],'NOT_SUPPORTED')
        self.assertEqual(classify_revision(dict(base,severe=True),True)[0],'NOT_SUPPORTED')
        self.assertIsNone(classify_revision(base,False)[0])

    def test_gap_rules(self):
        def rows(g1,g2,o1='LOSS',o2='LOSS'):
            return [{'M1_vs_B2':{'outcome':o1},'M2_vs_B2':{'outcome':o2},'gap_to_B2_v1':{'mean_gap':g1},'gap_to_B2_v2':{'mean_gap':g2}} for _ in range(6)]
        self.assertEqual(classify_gap(rows(.2,.1),True)['B2_GAP_STATUS'],'REDUCED')
        self.assertEqual(classify_gap(rows(.1,.2),True)['B2_GAP_STATUS'],'WORSENED')
        self.assertEqual(classify_gap(rows(.2,.201),True)['B2_GAP_STATUS'],'UNCHANGED')
        self.assertEqual(classify_gap(rows(.1,0,o2='TIE'),True)['B2_GAP_STATUS'],'CLOSED')
        self.assertEqual(classify_gap(rows(None,None),True)['B2_GAP_STATUS'],'UNCHANGED')
        self.assertIsNone(classify_gap(rows(.2,.1),False)['B2_GAP_STATUS'])

    def test_bidirectional_tie_boundary(self):
        def m(mean=100.,p95=140.,state='steady'):return {'weighted_mean_e2e':mean,'weighted_p95_e2e':p95,'stability_state_v2':state}
        self.assertEqual(compare(m(95),m(),100,100)['outcome'],'WIN')
        self.assertEqual(compare(m(96),m(),100,100)['outcome'],'TIE')
        self.assertEqual(compare(m(100,120),m(),100,100)['outcome'],'TIE')
        self.assertEqual(compare(m(state='unstable'),m(state='unstable'),100,100)['outcome'],'TIE')
        self.assertEqual(compare(m(),m(),90,100)['outcome'],'WIN')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'adapter_validation.log').open('w') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    record={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,
        'failures':[(str(t),e) for t,e in result.failures],'errors':[(str(t),e) for t,e in result.errors],
        'GPU_runs':0,'all_runtime_values_are_explicit_virtual_CPU_test_fixtures':True,
        'fixture_values_never_used_for_campaign_decisions_or_reports':True}
    save_json(OUT/'adapter_validation.json',record)
    print(record)
    raise SystemExit(0 if result.wasSuccessful() else 1)
