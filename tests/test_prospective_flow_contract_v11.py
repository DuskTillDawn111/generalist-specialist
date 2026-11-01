"""Full suffix/FIFO/score boundary regressions for the fresh gate adapter."""
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.known_input_campaign_seal_v5 import public_inputs
from tools.predictor_validation_20261027.profiling import digest
from tools.predictor_validation_20261027.prospective_flow_contract_v11 import (
    bind_public, gate_cell, score_complete_forecast, original_flow_defaults)


class ProspectiveFlowContractTests(unittest.TestCase):
    def setUp(self):
        self.aliases = {'text#0':dict(family='text',supported_tasks=['text'])}
        trace = [dict(request_id='fresh:req'+str(i),task='text',pool_id='input'+str(i),offset_ns=t)
                 for i,t in enumerate([0,100_000_000,100_000_000])]
        candidate = dict(id='shared',routes={r['request_id']:'text#0' for r in trace},
                         execution_context_sha256='context')
        self.plan = dict(schema='gsdeploy.native_known_input_flow_plan.v5',
            validation_scope='FRESH_HELDOUT_CO_RUN_REQUESTS_OF_EXACT_TRAINED_PUBLIC_INPUTS',
            aliases=self.aliases,trace=trace,candidates=[candidate],baseline_candidate_id='shared',
            window_seconds=1.,drain_seconds=2.,workload_rps={'text':3.},
            latency_reference_ms={'text':1000.},memory_budget_mib=100.,memory_safe_factor=1.2,
            energy_reference_watts=50.,startup_amortization_seconds=3600.,warmup_seconds=0.,
            weights={k:1/6 for k in 'QTLPME'},queue_sample_offsets_ns=[0,100_000_000,110_000_000])
        self.plan['plan_sha256'] = digest(self.plan)
        document, self.routes = public_inputs(self.plan,candidate)
        self.public = document['forecast_inputs']
        self.defaults = {k:self.plan[k] for k in ('aliases','latency_reference_ms',
            'memory_budget_mib','memory_safe_factor','energy_reference_watts')}
        self.defaults['route_initial_with_continuation_policy'] = False
        self.rows = bind_public(self.plan,'shared',self.public,self.routes,self.defaults,self.plan['weights'])
        finishes = {}
        for i,row in enumerate(self.rows):
            start = row['arrival_ns']+i*10_000_000
            finishes[row['request_id']] = dict(row,start_ns=start,handler_end_ns=start+5_000_000,
                                                finish_ns=start+6_000_000)
        self.forecast = dict(cohort={r['request_id']:r for r in self.rows},finishes=finishes,
                             board_energy_joules=25.,peak_memory_mib=60.)
        self.startup = dict(energy_joules=360.,peak_memory_mib=70.,energy_includes_priming=True,
            safety_factor_applied=False,startup_amortization_seconds=3600.,warmup_seconds=0.)

    def bind(self):
        return bind_public(self.plan,'shared',self.public,self.routes,self.defaults,self.plan['weights'])

    def test_future_suffix_exact_and_preserves_fifo(self):
        self.assertEqual([r['request_id'] for r in self.rows],['fresh:req0','fresh:req1','fresh:req2'])
        self.public['future_workload']['declared_arrivals'].reverse()
        with self.assertRaisesRegex(ValueError,'suffix'):
            self.bind()

    def test_missing_future_or_foreign_route_rejected(self):
        self.public['future_workload']['declared_arrivals'].pop()
        with self.assertRaisesRegex(ValueError,'routes'):
            self.bind()
        self.setUp()
        self.routes['fresh:req1'] = 'other'
        with self.assertRaisesRegex(ValueError,'routes'):
            self.bind()

    def test_six_terms_and_startup_only_once(self):
        metrics = score_complete_forecast(self.forecast,self.rows,self.plan,{'text':{'text':.2}},
                                           self.startup,phase_start_ns=0)
        self.assertAlmostEqual(metrics['terms']['E'],25/(50*1)+360/(50*3600))
        self.assertAlmostEqual(metrics['terms']['M'],1.2*70/100)
        result = dict(forecast=self.forecast,metrics=metrics)
        cell = gate_cell(result,self.rows,self.plan['candidates'][0],'fresh-run',
                         self.plan['queue_sample_offsets_ns'],phase_start_ns=0)
        self.assertEqual(cell['terms'],metrics['terms'])
        self.assertAlmostEqual(metrics['full_U'],-sum(cell['terms'].values())/6)
        self.assertEqual([s['depth'] for s in cell['queue_depth_samples']],[0,2,1])
        self.assertEqual(cell['requests'][1]['service_ms'],5.)
        self.assertEqual(cell['requests'][1]['queue_ms'],10.)
        self.assertEqual(cell['requests'][1]['e2e_ms'],16.)

    def test_partial_forecast_never_scores_complete_subset(self):
        del self.forecast['finishes']['fresh:req2']
        with self.assertRaises(UnknownEvidence):
            score_complete_forecast(self.forecast,self.rows,self.plan,{'text':{'text':.2}},
                                    self.startup,phase_start_ns=0)
        cell = gate_cell(dict(forecast=self.forecast),self.rows,self.plan['candidates'][0],'fresh-run',
                         [0,100_000_000],phase_start_ns=0)
        self.assertEqual(len(cell['requests']),3)
        self.assertIsNone(cell['requests'][2]['e2e_ms'])
        self.assertEqual(cell['queue_depth_samples'][0]['depth'],0)
        self.assertIsNone(cell['queue_depth_samples'][1]['depth'])

    def test_scaled_startup_and_incomplete_resources_rejected(self):
        self.startup['safety_factor_applied'] = True
        with self.assertRaisesRegex(ValueError,'unscaled'):
            score_complete_forecast(self.forecast,self.rows,self.plan,{'text':{'text':.2}},
                                    self.startup,phase_start_ns=0)
        self.setUp()
        self.forecast['board_energy_joules'] = None
        with self.assertRaises(UnknownEvidence):
            score_complete_forecast(self.forecast,self.rows,self.plan,{'text':{'text':.2}},
                                    self.startup,phase_start_ns=0)

    def test_parent_receipt_chronology_and_unknown_adapter_rejected(self):
        self.forecast['finishes']['fresh:req1']['finish_ns'] = 0
        with self.assertRaisesRegex(ValueError,'chronology'):
            score_complete_forecast(self.forecast,self.rows,self.plan,{'text':{'text':.2}},
                                    self.startup,phase_start_ns=0)
        with self.assertRaisesRegex(ValueError,'single startup'):
            gate_cell(dict(metrics={'terms':{k:0. for k in 'QTLPME'}}),self.rows,
                      self.plan['candidates'][0],'fresh-run',[0],phase_start_ns=0)

    def test_drain_endpoint_is_inclusive_but_cannot_be_extended(self):
        self.forecast['finishes']['fresh:req2']['finish_ns'] = 3_000_000_000
        metrics = score_complete_forecast(self.forecast,self.rows,self.plan,{'text':{'text':.2}},
                                          self.startup,phase_start_ns=0)
        self.assertIsNotNone(metrics['full_U'])
        self.forecast['finishes']['fresh:req2']['finish_ns'] += 1
        with self.assertRaisesRegex(UnknownEvidence,'DRAIN_BOUNDARY'):
            score_complete_forecast(self.forecast,self.rows,self.plan,{'text':{'text':.2}},
                                    self.startup,phase_start_ns=0)

    def test_subset_tasks_keep_original_training_normalizers(self):
        trained = dict(self.defaults,latency_reference_ms={'text':1000.,'ocr':10_000.})
        self.assertEqual(original_flow_defaults(self.plan,trained),self.defaults)
        for field in ('memory_budget_mib','energy_reference_watts','memory_safe_factor'):
            original = self.plan[field];self.plan[field] = original*2
            with self.assertRaisesRegex(ValueError,'normalizer'):
                original_flow_defaults(self.plan,trained)
            self.plan[field] = original
        self.plan['latency_reference_ms']['text'] = 123.
        with self.assertRaisesRegex(ValueError,'normalizer'):
            original_flow_defaults(self.plan,trained)


if __name__ == '__main__':
    unittest.main()
