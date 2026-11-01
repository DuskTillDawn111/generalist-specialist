"""Independent replay/scoring regressions using preserved native evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fine_analysis_test',ROOT/'tools/spillover_context_20261022/analyze_fine_development.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


class FineAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=json.loads((ROOT/'artifacts/static-spillover-study-20261022/text26-development-v1/runs/text26_n2_specialist_share_1of2-attempt1/run.json').read_text())
        cls.api=analysis.api();cls.config=cls.api['config'].SystemConfig(**cls.raw['config'])
        contract=json.loads((ROOT/'artifacts/static-study-v6-20261021/inputs/study-contract.json').read_text())
        cls.samples=contract['runtime_input_pool']

    def test_exact_native_arrival_input_alias_replay_detects_input_and_route_changes(self):
        report=analysis.exact_replay(self.config,self.raw,self.samples,self.api)
        self.assertTrue(report['route_and_input_replay_verified']);self.assertEqual(report['requests'],1631)
        for field,value in [('pool_id','wrong'),('model','text#1'),('scheduled_time_ns',0)]:
            invalid=copy.deepcopy(self.raw);invalid['request_records'][0][field]=value
            with self.assertRaises(AssertionError):analysis.exact_replay(self.config,invalid,self.samples,self.api)

    def test_native_quality_and_frozen_evaluator_match_preserved_run(self):
        quality=analysis.native_quality(self.config,self.raw,self.api)
        self.assertTrue(quality['complete'])
        self.assertEqual(quality['independently_rescored_Q_norm'],self.raw['evaluation']['utility']['Q_norm'])
        self.assertEqual(quality['per_task'],self.raw['evaluation']['utility']['quality_evidence']['per_task'])
        self.assertEqual(self.api['evaluation'].evaluate_run(self.config,copy.deepcopy(self.raw)),self.raw['evaluation'])

    def test_full_terminal_audit_preserves_native_binding_ocr_inventory_expansion(self):
        directory=ROOT/'artifacts/static-spillover-study-20261022/text26-development-v1'
        packet=json.loads((directory/'experiment-packet.json').read_text())
        job=next(j for j in packet['jobs'] if j['id']==self.raw['experiment_job_id'])
        path=directory/'runs/text26_n2_specialist_share_1of2-attempt1/run.json'
        report=analysis.analyze_job(packet,job,self.raw,path,self.api)
        self.assertEqual(report['audit_status'],'VERIFIED_TERMINAL_EVIDENCE')
        self.assertTrue(report['memory_pass']);self.assertTrue(report['energy_pass'])
        self.assertTrue(report['original_evaluate_run_verified'])

    def test_incomplete_native_output_cannot_become_quality_score(self):
        raw=copy.deepcopy(self.raw);raw['request_records'][0]['status']='incomplete'
        result=analysis.native_quality(self.config,raw,self.api)
        self.assertFalse(result['complete']);self.assertIsNone(result['independently_rescored_Q_norm'])
        self.assertEqual(len(result['unknown']),1)

    def test_alias_decomposition_retains_actual_input_mix_and_tail_queues(self):
        rows=analysis.alias_summary(self.raw)
        self.assertEqual(sum(sum(r['task_counts'].values()) for r in rows),1631)
        for row in rows:
            self.assertEqual(row['missing_service_requests'],0)
            self.assertAlmostEqual(row['observed_service_sum_over_arrival_horizon'],row['observed_service_sum_seconds']/600)
            self.assertIn('host_enqueue_to_submit_ms',row['timing_terms_ms'])
            self.assertIn('submit_to_worker_start_ms',row['arrival_tail_timing_terms_ms'])
        imbalance=analysis.specialist_imbalance({'per_alias':rows})
        self.assertIsNotNone(imbalance['observed_service_work_gap_seconds'])
        self.assertIn('remaps',imbalance['mechanism_limit'])

    def test_native_service_and_host_timestamp_identity_cannot_be_forged(self):
        row=copy.deepcopy(self.raw['request_records'][0])
        analysis.verify_response_timing(row)
        row['worker_response']['service_ms']=row['worker_response']['worker_latency_ms']+1
        with self.assertRaises(AssertionError):analysis.verify_response_timing(row)

    def test_unexecuted_or_failed_conditions_cannot_generate_pair_benefit(self):
        result=analysis.pairs([dict(seed=107,audit_status='DECLARED_NOT_EXECUTED')])
        self.assertEqual([r['seed'] for r in result],[107,108])
        self.assertTrue(all(r['delta_U'] is None for r in result))


if __name__=='__main__':unittest.main()
