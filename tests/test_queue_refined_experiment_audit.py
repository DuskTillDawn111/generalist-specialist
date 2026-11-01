"""Independent packet corruption and seed/claim leakage checks; CPU only."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import itertools
import json
import statistics
import unittest
from unittest.mock import patch

from tools.queue_refined_20261023 import experiment_audit as a


class ExperimentAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.path = self.root/'packet.json'
        here = Path(a.__file__).parent
        names = ['protocol.py','native_runner.py','runtime_window.py','policy.py',
                 'guarded_policy.py','admission.py','evaluation.py']
        self.sources = {str((here/n).resolve()):a.receipt(here/n)['sha256'] for n in names}

    def packet(self):
        contract = dict(strategy='zero_probe', deployment={'text':2,'generalist_2':1,'ocr':1},
            service_replica_states={'text':2,'generalist_2':1,'ocr':1},
            source_fingerprints=self.sources,absolute_quality_gate=False)
        identity = dict(seed=201,phase='calibration',policy_contract_sha256=a.digest(contract),
            measurement_claim=a.CLAIMS['calibration'],horizon_seconds=600,warmup_seconds=30)
        return dict(inputs=[],sources={},seeds=[201],jobs=[dict(id=a.digest(identity),label='zero201',
            phase='calibration',seed=201,identity=identity,policy_contract=contract,
            nominal_plan=dict(deployment=contract['deployment']))])

    def audit(self, packet, **kwargs):
        a.write(self.path,packet)
        try:return a.audit_packet(self.path,inspect_inputs=False,**kwargs)
        finally:self.path.unlink()

    def rehash(self,job):
        job['identity']['policy_contract_sha256']=a.digest(job['policy_contract'])
        job['id']=a.digest(job['identity'])

    def test_seeds_claims_and_fingerprint_corruption_fail_independently(self):
        packet=self.packet();self.assertTrue(self.audit(packet)['valid'])
        packet['jobs'][0]['seed']=205
        result=self.audit(packet);self.assertFalse(result['valid'])
        self.assertTrue(any('UNDECLARED_PHASE_OR_SEED' in v for v in result['issues']))
        packet=self.packet();packet['jobs'][0]['identity']['measurement_claim']=a.CLAIMS['independent_validation']
        self.rehash(packet['jobs'][0]);self.assertFalse(self.audit(packet)['valid'])
        packet=self.packet();packet['jobs'][0]['policy_contract']['source_fingerprints']=dict(self.sources)
        packet['jobs'][0]['policy_contract']['source_fingerprints'][str(Path(a.__file__).with_name('policy.py').resolve())]='0'*64
        self.rehash(packet['jobs'][0]);self.assertFalse(self.audit(packet)['valid'])

    def test_duplicate_seed_register_and_untouched_old_raw_guard(self):
        design=a.draft_design();design['phase_seeds']=deepcopy(design['phase_seeds'])
        design['phase_seeds']['development']=[201,204]
        self.assertIn('PHASE_SEED_OVERLAP',self.audit(self.packet(),design=design)['issues'])
        raw=self.root/'old-run.json';raw.write_text('old terminal')
        ledger=dict(protected=[a.receipt(raw)])
        self.assertTrue(self.audit(self.packet(),old_ledger=ledger)['valid'])
        raw.write_text('tampered terminal')
        self.assertFalse(self.audit(self.packet(),old_ledger=ledger)['valid'])

    def test_alias_phase_and_replica_identity(self):
        packet=self.packet();job=packet['jobs'][0];job.update(phase='independent-validation',seed=205)
        job['identity'].update(phase='independent_validation',seed=205,
            measurement_claim=a.CLAIMS['independent_validation']);packet['seeds']=[205]
        self.rehash(job);self.assertTrue(self.audit(packet)['valid'])
        job['policy_contract']['service_replica_states']={'text':1};self.rehash(job)
        self.assertTrue(any('SERVICE_REPLICA_STATE_MISMATCH' in v for v in self.audit(packet)['issues']))

    def test_original_nontransitive_near_tie_search_and_analysis_all_orders(self):
        """Replay the old failing table through both real search entry points.

        Analysis receives synthetic observations so this checks ranking wiring,
        not the physical truth of the old synthetic table's eligibility flags.
        """
        from tools.queue_refined_20261023 import search, analysis
        source = a.ROOT/'artifacts/queue-aware-study-20261023/workflow-data-contract-readonly-audit-v1/synthetic-near-tie-table.json'
        table = a.read(source)
        values = {search.identity(c):statistics.mean(r['utility']['U'] for r in table['records']
                  if r['candidate_id']==search.identity(c)) for c in table['candidates']}
        maximum = max(values.values())
        ties = sorted(cid for cid, v in values.items() if maximum-v<=1e-12)
        self.assertEqual(len(ties), 2)
        lookup = {(r['candidate_id'],r['seed']):r for r in table['records']}
        with TemporaryDirectory() as temp:
            root=Path(temp)
            for index, ordered in enumerate(itertools.permutations(table['candidates'])):
                document=dict(table,candidates=list(ordered))
                results=[search.measured_table_search(document, mode) for mode in ('plain_dfs','bounded')]
                for result in results:
                    self.assertEqual(result['tie_ids'],ties)
                    self.assertEqual(result['winner_id'],ties[0])
                    self.assertEqual(result['objective'],values[ties[0]])
                    self.assertEqual(result['exact_max_objective'],maximum)
                observations={};paths=[]
                for candidate in ordered:
                    cid=search.identity(candidate)
                    for seed in table['seeds']:
                        path=root/f'{index}-{cid}-{seed}.json'
                        path.write_text(json.dumps({'config':{'weights':table['weights']}}))
                        utility=lookup[cid,seed]['utility']
                        observations[path]=dict(candidate_id=cid,candidate=candidate,seed=seed,
                            utility=utility,contributions={t:-table['weights'][t]*utility[t+'_norm'] for t in 'QTLPME'},
                            cohort_sha256=f'paired-{seed}',raw={'path':str(path),'sha256':'synthetic'})
                        paths.append(path)
                with patch.object(analysis,'read_run',side_effect=lambda p:observations[p]):
                    report=analysis.analyze(paths,table['seeds'],root/f'analysis-{index}')
                self.assertEqual(report['winner']['candidate_id'],ties[0])
                self.assertEqual(report['winner']['mean_U'],values[ties[0]])
                self.assertEqual(report['exact_max_objective'],maximum)

    def test_refined_native_adapter_both_modes_guard_replay_and_model_receipt(self):
        """Exercise real CPU worker transport with the main refined dispatcher.

        Mock resources remain unmeasured; this is a compatibility check only.
        """
        from tests.test_queue_refined_probe_native import NativeRunner
        from tools.queue_refined_20261023 import native_runner, protocol, evaluation, calibration
        fixture=NativeRunner();fixture.setUp();self.addCleanup(fixture.doCleanups)
        observations=[]
        for family,tasks in fixture.contract['service_ms'].items():
            for task,inputs in tasks.items():
                for pool in inputs:
                    observations.append(dict(family=family,task=task,pool_id=pool,
                        service_ms=.25 if family=='generalist_2' else 1.,seed=113,
                        context=dict(other_active_bin='0',g2_visible_outstanding=False),
                        deployment=fixture.contract['deployment'],raw_sha256='synthetic-training',request_id='training'))
        model_path=fixture.root/'service-model.json'
        calibration.save(model_path,calibration.fit(observations))
        for mode in ('conservative','global_incremental'):
            contract=deepcopy(fixture.contract)
            contract.update(strategy='queue_latency',mode='latency',guard=dict(
                schema='queue_aware_token_bucket_v1',task='text',family='generalist_2',rate_rps=10.,burst=1))
            contract['refinement']=dict(schema='queue_refined_policy_v1',model=protocol.receipt(model_path),
                mode=mode,margin_ms=0.,risk_quantile=.9,concurrency_context=True,allow_spill=True,
                horizon_seconds=.15,energy_proxy_joules_per_service_ms=None)
            def evaluate(system,raw):
                for phase in ('warmup','measurement'):
                    replay=evaluation.audit_phase(system,raw,phase)
                    self.assertTrue(replay['verified'],replay)
                    self.assertGreater(replay['guard_events_replayed'],0)
                return dict(utility=dict(U=None,eligible=False,reason='MOCK_BACKEND'),stability={})
            with patch('subprocess.run',side_effect=AssertionError('No GPU query in mock')):
                raw=native_runner.run(fixture.system,protocol.nominal_plan(contract),fixture.samples,contract,
                    backend='mock',duration=.15,warmup=.10,seed=113,arrival_process='deterministic_periodic',
                    startup_timeout=10.,drain_timeout=10.,evaluator=evaluate)
            self.assertEqual(raw['status'],'completed',raw['errors'])
            self.assertIsNone(raw['evaluation']['utility']['U'])
            for record in raw['request_records']:
                if record['task']=='text':
                    decision=record['routing_decision']['decision']
                    self.assertEqual(decision['mode'],mode)
                    self.assertIsNone(decision['full_U'])
        model_path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'model receipt changed'):
            protocol.validate_contract(contract)


if __name__=='__main__':unittest.main()
