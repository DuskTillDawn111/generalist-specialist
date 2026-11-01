"""Flow/startup binding and once-only six-term diagnostic arithmetic."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import forecast_gate_multi_native_v6 as gate
from tools.predictor_validation_20261027.native_collect import receipt, save


class FlowBoundDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.aliases = {t:dict(family=t,supported_tasks=[t]) for t in ('text','image','asr','ocr')}
        self.rows = [dict(request_id=t,task=t,alias=t,pool_id='pool-'+t,arrival_ns=0) for t in self.aliases]
        self.defaults = dict(aliases=self.aliases,latency_reference_ms={t:1000 for t in self.aliases},
            memory_budget_mib=10000,energy_reference_watts=450.,memory_safe_factor=1.1,
            route_initial_with_continuation_policy=False)
        self.weights = {k:1/6 for k in 'QTLPME'}
        self.public = dict(phase_start_ns=0,now_ns=0,deadline_ns=10_000_000_000,drain_deadline_ns=130_000_000_000,
            future_workload=dict(process='declared_schedule',declared_arrivals=[],rates={t:1. for t in self.aliases},
                input_pools={t:['pool-'+t] for t in self.aliases}))
        self.plan = dict(**{k:v for k,v in self.defaults.items() if k!='route_initial_with_continuation_policy'},
            weights=self.weights,window_seconds=10.,drain_seconds=120.,warmup_seconds=0.,
            workload_rps=self.public['future_workload']['rates'],
            trace=[dict(request_id=r['request_id'],task=r['task'],pool_id=r['pool_id'],offset_ns=0) for r in self.rows],
            candidates=[dict(id='fixed',routes={r['request_id']:r['alias'] for r in self.rows},allow_spill=False)])
        self.startup = dict(energy_joules=5822.193265376832,peak_memory_mib=29766.,
            startup_amortization_seconds=3600.,execution_context_sha256='original-context',
            candidate_id='fixed',safety_factor_applied=False,warmup_seconds=0.,energy_includes_priming=True,
            target_contract=dict(priming_content_sha256='original-twelve-inputs'))

    def ref(self, name, value):
        path = self.root / name; save(path, value); return receipt(path)

    def test_complete_fixed_cohort_matches_and_changed_route_arrival_or_drain_fails(self):
        gate.bind_flow(self.plan,'fixed',self.public,self.defaults,self.weights,self.rows)
        for changed in ('route','arrival','missing','window','drain','rates','safe_factor','weights'):
            plan, public, defaults, rows, weights = map(copy.deepcopy,
                (self.plan,self.public,self.defaults,self.rows,self.weights))
            if changed=='route': plan['candidates'][0]['routes']['text']='image'
            elif changed=='arrival': plan['trace'][0]['offset_ns']=1
            elif changed=='missing': rows.pop()
            elif changed=='window': public['deadline_ns']+=1
            elif changed=='drain': public['drain_deadline_ns']=120_000_000_000
            elif changed=='rates': public['future_workload']['rates']['text']=2.
            elif changed=='safe_factor': defaults['memory_safe_factor']=1.2
            elif changed=='weights': weights['Q']=.5
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                gate.bind_flow(plan,'fixed',public,defaults,weights,rows)

    def startup_bundle(self):
        flow = self.ref('flow.json',self.plan); self.startup['flow_plan']=flow
        return dict(startup=dict(training_model=self.ref('training-model.json',{}),flow_plan=flow,
            candidate_id='fixed',prediction=self.ref('startup.json',self.startup)),
            forecast_defaults=self.defaults,weights=self.weights,allow_native_priming_transfer_hypothesis=True)

    def test_same_owner_same_arrival_queue_order_cannot_be_changed(self):
        rows = copy.deepcopy(self.rows)
        rows.append(dict(rows[0], request_id='second-text', pool_id='other-text'))
        plan = copy.deepcopy(self.plan)
        plan['trace'].append(dict(request_id='second-text', task='text', pool_id='other-text', offset_ns=0))
        plan['candidates'][0]['routes']['second-text'] = 'text'
        public = copy.deepcopy(self.public)
        public['future_workload']['input_pools']['text'] = ['other-text','pool-text']
        gate.bind_flow(plan,'fixed',public,self.defaults,self.weights,rows)
        rows[0], rows[-1] = rows[-1], rows[0]
        with self.assertRaisesRegex(ValueError,'Per-owner public queue order'):
            gate.bind_flow(plan,'fixed',public,self.defaults,self.weights,rows)

    def test_startup_is_recomputed_and_entire_prediction_including_priming_compared(self):
        bundle=self.startup_bundle(); adapter=SimpleNamespace(target_execution_context_sha256='original-context')
        with patch.object(gate,'startup_prediction',return_value=self.startup) as recompute:
            self.assertEqual(gate.verify_startup(bundle,self.public,adapter,self.rows),self.startup)
            recompute.assert_called_once_with(bundle['startup']['training_model']['path'],
                bundle['startup']['flow_plan']['path'],'fixed',allow_native_priming_transfer_hypothesis=True)
        for changed in ('priming','energy','candidate','context'):
            result=copy.deepcopy(self.startup)
            if changed=='priming': result['target_contract']['priming_content_sha256']='other-inputs'
            elif changed=='energy': result['energy_joules']+=1
            elif changed=='candidate': result['candidate_id']='other'
            else: result['execution_context_sha256']='other'
            with self.subTest(changed=changed), patch.object(gate,'startup_prediction',return_value=result), \
                    self.assertRaisesRegex(ValueError,'full source reverification'):
                gate.verify_startup(bundle,self.public,adapter,self.rows)

    def test_startup_energy_once_safety_once_and_drain_peak_included(self):
        # ASR finishes in drain: it contributes latency and peak, no extra board
        # energy is appended to the already measurement-only forecast integral.
        cohort={r['request_id']:r for r in self.rows}
        finishes={k:dict(r,finish_ns=12_000_000_000 if k=='asr' else 1_000_000_000) for k,r in cohort.items()}
        forecast=dict(cohort=cohort,finishes=finishes,peak_memory_mib=31000.,board_energy_joules=1000.)
        quality={t:{t:dict(numerator=0.,denominator=1.,basis='expected_per_request_totals',evidence='training')} for t in self.aliases}
        metrics=gate.diagnostic_score(forecast,SimpleNamespace(quality=quality),
            SimpleNamespace(rates=self.public['future_workload']['rates']),self.defaults,
            dict(weights=self.weights),self.public,self.startup)
        self.assertAlmostEqual(metrics['terms']['M'],1.1*31000/10000)
        self.assertAlmostEqual(metrics['terms']['E'],1000/4500+self.startup['energy_joules']/(450*3600))
        self.assertAlmostEqual(metrics['startup_energy_amortized_to_window_joules'],self.startup['energy_joules']/360)
        self.assertAlmostEqual(metrics['full_U'],-sum(metrics['terms'].values())/6)
        self.assertEqual(metrics['terms']['T'],.25)
        self.assertFalse(metrics['physical_six_term_improvement_certified'])

    def test_new_gate_requires_diagnostic_opt_in_before_models(self):
        with self.assertRaisesRegex(ValueError,'Explicit multi-native'):
            gate.execute_bundle(dict(schema=gate.BUNDLE_SCHEMA,kind='trained_hypothesis'),{})


if __name__ == '__main__':
    unittest.main()
