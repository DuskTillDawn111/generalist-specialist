"""CPU tests for run-level holdout isolation and unchanged M/E contracts."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('resource_candidates',ROOT/'tools/spillover_context_20261022/fit_resource_candidates.py')
fit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(fit)


class FitContracts(unittest.TestCase):
    def fixture(self):
        return [dict(run_id=f'r{i}',deployment_key='same' if i<2 else f'd{i}',deployment={'x':i+1},
            features={'intercept':1.,'x':float(i+1)},target=float(3+2*(i+1))) for i in range(6)]

    def test_held_run_target_cannot_affect_own_prediction(self):
        rows=self.fixture();groups=[r['run_id'] for r in rows]
        before=fit.split_diagnostics(rows,['intercept','x'],'target',groups)
        changed=copy.deepcopy(rows);changed[2]['target']=9000
        after=fit.split_diagnostics(changed,['intercept','x'],'target',groups)
        self.assertAlmostEqual(before['predictions'][2],after['predictions'][2])
        self.assertNotIn('r2',after['folds'][2]['training_run_ids'])

    def test_deployment_fold_removes_all_same_deployment_runs(self):
        rows=self.fixture();result=fit.split_diagnostics(rows,['intercept','x'],'target',[r['deployment_key'] for r in rows])
        fold=result['folds'][0]
        self.assertEqual(fold['held_run_ids'],['r0','r1'])
        self.assertNotIn('r0',fold['training_run_ids']);self.assertNotIn('r1',fold['training_run_ids'])

    def test_rank_deficiency_is_unknown_not_fitted(self):
        result=fit.fit([[1,1],[1,1]],[2,3])
        self.assertEqual(result['status'],'UNKNOWN_RANK_DEFICIENT');self.assertIsNone(result['coefficients'])

    def test_nnls_nonnegativity_is_not_upper_bound(self):
        result=fit.fit([[1],[1]],[1,9])
        self.assertFalse(result['safety_upper_bound'])
        self.assertTrue(result['nonnegative_function_on_nonnegative_features'])
        self.assertGreaterEqual(result['predictions'][1],0)
        self.assertLess(result['predictions'][1],9)

    def test_memory_safe_factor_applied_once_and_false_accept_exposed(self):
        rows=[{'run_id':'a','M_norm':1.1,'safe_factor':1.1},{'run_id':'b','M_norm':.8,'safe_factor':1.1}]
        result=fit.memory_constraint_diagnostics(rows,[.8,1.])
        self.assertEqual(result['counts'],{'false_accept':1,'false_reject':1})
        self.assertAlmostEqual(result['details'][0]['predicted_M_norm'],.88)
        self.assertEqual(result['false_accept_rate'],1.)

    def test_all_feasible_does_not_estimate_false_accept_rate(self):
        rows=[{'run_id':'a','M_norm':.8,'safe_factor':1.1}]
        result=fit.memory_constraint_diagnostics(rows,[.7])
        self.assertIsNone(result['false_accept_rate']);self.assertFalse(result['safety_guarantee'])

    def test_energy_startup_amortization_is_not_measurement_duration(self):
        evidence=dict(scope='whole_gpu_board',kind='measured',coverage_valid=True,run_joules=60000.,startup_joules=36000.)
        self.assertAlmostEqual(fit.energy_norm(evidence,600,450,3600),(100+10)/450)
        self.assertNotAlmostEqual(fit.energy_norm(evidence,600,450,3600),(60000+36000)/600/450)

    def test_missing_prediction_stays_unknown(self):
        result=fit.errors([1,2],[None,2.1])
        self.assertEqual(result['observations'],1)


class FrozenEvidenceContracts(unittest.TestCase):
    def test_exact_19_fields_and_original_components_reproduce_without_subprocess(self):
        with patch('subprocess.run',side_effect=AssertionError('No GPU query/child process allowed')):
            rows,audits,unknown,opened=fit.load_locked(fit.LOCK)
        self.assertEqual(len(rows),19);self.assertEqual(len(audits),19)
        self.assertTrue(all(r['seed'] not in fit.FORBIDDEN_SEEDS for r in rows))
        self.assertTrue(all(a['M_norm_reproduced'] and a['E_norm_reproduced'] and a['window_integral_reproduced'] and a['startup_integral_reproduced'] for a in audits))
        self.assertEqual(unknown,[])

    def test_manifest_and_all_report_folds_preserve_holdout_isolation(self):
        base=ROOT/'artifacts/static-spillover-study-20261022/context-resource-candidates-dev-v1/fit-v1'
        report=json.loads((base/'report.json').read_text());manifest=json.loads((base/'manifest.json').read_text())
        for path,expected in manifest['inputs'].items():self.assertEqual(fit.sha(path),expected)
        for name,expected in manifest['files'].items():self.assertEqual(fit.sha(base/name),expected)
        self.assertEqual(report['run_count'],19);self.assertFalse(report['predictor_activated']);self.assertIsNone(report['production_U'])
        for model in report['models'].values():
            for mode in ('leave_one_run_out','leave_one_deployment_out'):
                for fold in model[mode]['folds']:
                    self.assertFalse(set(fold['held_run_ids'])&set(fold['training_run_ids']))
                    self.assertEqual(len(fold['held_run_ids'])+len(fold['training_run_ids']),19)


if __name__=='__main__':unittest.main()
