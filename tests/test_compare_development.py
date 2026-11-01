"""Development ledger respects cohort/contract identity and incomplete queues."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('compare_development_test',ROOT/'tools/spillover_context_20261022/compare_development.py')
ledger=importlib.util.module_from_spec(spec);spec.loader.exec_module(ledger)


class DevelopmentComparisonTests(unittest.TestCase):
    def fixture(self,directory,ids,registered,contract_change=False):
        config=dict(workload={'text':1.85},weights={t:1/6 for t in 'QTLPME'},latency_reference_ms={'text':1000},
            gpu_memory_mib=49140,horizon_seconds=600,drain_seconds=300,arrival_process='poisson',
            energy_reference_watts=450,startup_amortization_seconds=3600,utility_schema='six_term_quality_v1',
            catalog={'quality':{'identity':{'reference_sha256':'frozen'}}})
        if contract_change:config['energy_reference_watts']=300
        packet=dict(config=config,jobs=[dict(id=i,labels=[i],seed=107,plan={'deployment':{'text':2},'routing':{'text':{'text':1.85}}}) for i in ids])
        directory.mkdir();packetpath=directory/'packet.json';packetpath.write_text(json.dumps(packet));rows=[]
        for i in registered:
            run=directory/i/'run.json';run.parent.mkdir();run.write_text(json.dumps({'fake_terminal':i}))
            rows.append(dict(job_id=i,raw=ledger.audit.exposure.receipt(run)))
        queuepath=directory/'queue.json';queuepath.write_text(json.dumps(dict(packet=ledger.audit.exposure.receipt(packetpath),status='RUNNING',jobs=rows,
            current={'output':'NEVER_READ_LIVE_CURRENT'})))
        return packetpath,queuepath

    def mocked_audit(self,cohorts=None):
        def analyze(packet,job,raw,path,modules):
            i=job['id'];U=-1.0 if i=='a' else -2.0
            return dict(job_id=i,label=i,seed=107,plan=job['plan'],audit_status='VERIFIED_TERMINAL_EVIDENCE',
                cohort_sha256=(cohorts or {}).get(i,'same'),utility={'eligible':True,'U':U,**{t+'_norm':1. for t in 'QTLPME'}},
                memory_pass=True,energy_pass=True,shutdown_pass=True,stability={'stability_state_v2':'steady'})
        return analyze

    def test_running_queue_pending_never_ranks_and_snapshot_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);packet,queue=self.fixture(root/'first',['a','pending'],['a']);bytes_before=queue.read_bytes()
            modules={'static_study':SimpleNamespace(validate_packet=lambda _:dict(valid=True))}
            with patch.object(ledger.audit,'api',return_value=modules),patch.object(ledger.audit,'analyze_job',side_effect=self.mocked_audit()):
                report=ledger.compare([(packet,queue)],root/'out')
            compared=report['comparisons'][0]
            self.assertEqual(compared['declared_conditions'],2);self.assertEqual(compared['eligible_conditions'],1)
            self.assertEqual([r['job_id'] for r in compared['ranking_among_observed_eligible']],['a'])
            self.assertEqual(compared['pairwise_deltas'],[])
            self.assertEqual(compared['ineligible_or_pending'][0]['audit_status'],'DECLARED_NO_TERMINAL_RECEIPT')
            self.assertEqual((root/'out/queue-snapshot-1.json').read_bytes(),bytes_before)

    def test_same_seed_different_cohort_cannot_be_compared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);one=self.fixture(root/'first',['a'],['a']);two=self.fixture(root/'second',['b'],['b'])
            modules={'static_study':SimpleNamespace(validate_packet=lambda _:dict(valid=True))}
            with patch.object(ledger.audit,'api',return_value=modules),patch.object(ledger.audit,'analyze_job',side_effect=self.mocked_audit({'b':'different'})):
                with self.assertRaisesRegex(ValueError,'different task/input/arrival cohorts'):
                    ledger.compare([one,two],root/'out')
            self.assertFalse((root/'out').exists())

    def test_duplicate_physical_identity_and_scoring_contract_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);one=self.fixture(root/'first',['a'],['a']);same=self.fixture(root/'same',['a'],['a']);changed=self.fixture(root/'changed',['b'],['b'],True)
            modules={'static_study':SimpleNamespace(validate_packet=lambda _:dict(valid=True))}
            with patch.object(ledger.audit,'api',return_value=modules),patch.object(ledger.audit,'analyze_job',side_effect=self.mocked_audit()):
                with self.assertRaisesRegex(AssertionError,'identity duplicated'):
                    ledger.compare([one,same],root/'out1')
                with self.assertRaisesRegex(ValueError,'different workload/scoring/quality'):
                    ledger.compare([one,changed],root/'out2')


if __name__=='__main__':unittest.main()
