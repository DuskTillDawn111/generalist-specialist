import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.energy import energy_norm
from tools.predictor_validation_20261027 import h600_capacity_v1 as module
from tools.predictor_validation_20261027.flow_validation import _score
from tools.predictor_validation_20261027.native_flow import receipt
from tools.predictor_validation_20261027.profiling import digest


class H600CapacityTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.jobs=[]
        source=self.write('source.json',{});ref=receipt(source)
        for seed in (627601,627602):
            plan=dict(seed=seed,training_run_label='primary',measurement_role='H600_training',
                phase='predictor_training',kind='prospective_training',split='training',validation_tuning=False,
                window_seconds=600.,drain_seconds=300.,warmup_seconds=0.,training_campaign_protocol=ref,
                candidates=[{'id':cid} for cid in sorted(module.weighted.ROUTES)])
            plan['plan_sha256']=digest(plan);path=self.write(f'{seed}.json',plan)
            self.jobs.extend(dict(plan=receipt(path),candidate_id=cid,campaign_role='primary_training')
                for cid in sorted(module.weighted.ROUTES))
        self.suite=self.write('suite.json',dict(schema='gsdeploy.continuous_training_suite.v1',
            seeds=[627601,627602],jobs=self.jobs,source=ref,protocol=ref,
            no_validation_data=True,repeats_excluded=True))
        mocks=[patch.object(module,'condition',side_effect=lambda p,c:{'H':p['window_seconds'],'route':c['id']}),
               patch.object(module,'_generation',return_value={'law':'fixed'})]
        for mock in mocks:mock.start();self.addCleanup(mock.stop)

    def write(self,name,value):
        path=self.root/name;path.write_text(json.dumps(value));return path

    def mutate_plan(self,update):
        path=Path(self.jobs[0]['plan']['path']);value=json.loads(path.read_text());value.pop('plan_sha256')
        value.update(update);value['plan_sha256']=digest(value);path.write_text(json.dumps(value))
        suite=json.loads(self.suite.read_text())
        for job in suite['jobs']:
            if job['plan']['path']==str(path):job['plan']=receipt(path)
        self.suite.write_text(json.dumps(suite))

    def test_complete_primary_H600_public_suite_passes_without_raw(self):
        self.assertEqual(len(module.verify_suite(self.suite)['jobs']),8)

    def test_H90_cannot_enter_numerical_sources(self):
        self.mutate_plan({'window_seconds':90.})
        with self.assertRaisesRegex(ValueError,'H600'):
            module.verify_suite(self.suite)

    def test_repeat_preflight_or_validation_cannot_replace_primary(self):
        for change in ({'training_run_label':'repeat1'},{'measurement_role':'short_collection_preflight_excluded_from_H600_fit'},
                       {'split':'validation'},{'warmup_seconds':30.}):
            with self.subTest(change=change):
                self.mutate_plan(change)
                with self.assertRaises(ValueError):module.verify_suite(self.suite)
                self.mutate_plan({'training_run_label':'primary','measurement_role':'H600_training','split':'training','warmup_seconds':0.})

    def test_duplicate_or_missing_primary_route_rejected(self):
        value=json.loads(self.suite.read_text());value['jobs'][-1]=copy.deepcopy(value['jobs'][0])
        self.suite.write_text(json.dumps(value))
        with self.assertRaises(ValueError):module.verify_suite(self.suite)

    def test_qualification_keeps_original_predicted_denominator_and_missing_unknown(self):
        prediction={'estimates':[{'family':'g2','task':'asr','weighted_worker_service_ms':100.},
            {'family':'ocr','task':'ocr','weighted_worker_service_ms':None}]}
        observed=[{'family':'g2','task':'asr','worker_service_ms':125.},
                  {'family':'ocr','task':'ocr','worker_service_ms':500.}]
        checks=module.service_checks(prediction,observed)
        self.assertEqual(checks[0]['absolute_relative_mean_error'],.25)
        self.assertFalse(checks[0]['passed'])
        self.assertIsNone(checks[1]['passed'])
        self.assertEqual(module.POLICY['mean_service_relative_error_max'],.2)
        self.assertEqual(module.POLICY['rho_threshold'],.8)

    def test_startup_amortization_prediction_equals_actual_H600_arithmetic(self):
        plan={'window_seconds':600.,'startup_amortization_seconds':3600.,'aliases':{},'workload_rps':{},
            'latency_reference_ms':{},'memory_budget_mib':49140.,'energy_reference_watts':450.,
            'memory_safe_factor':1.,'weights':{}}
        with patch('tools.predictor_validation_20261027.flow_validation.score_forecast',return_value={}) as score:
            _score(plan,{}, {},startup_joules=18000.)
        amortized=score.call_args.kwargs['observed_board_energy_joules']
        self.assertEqual(amortized,3000.)
        actual=energy_norm(dict(scope='whole_gpu_board',kind='measured',coverage_valid=True,
            run_joules=180000.,startup_joules=18000.),600.,450.,3600.)
        self.assertAlmostEqual(actual,(180000.+amortized)/(600.*450.))


if __name__=='__main__':unittest.main()
