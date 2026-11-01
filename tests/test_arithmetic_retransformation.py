"""Arithmetic target, weighting and fold-only residual checks, CPU only."""
import importlib.util
import math
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('arithmetic',ROOT/'tools/spillover_context_20261022/fit_arithmetic_retransformation.py')
f=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(f)


def row(rid,sid,ratio,family='text',task='text'):
    return dict(run_id=rid,family=family,task=task,n=1,sample_id=sid,features=[],baseline_ms=10.,
        service_ms=10*ratio,log_service_ratio=math.log(ratio))


class Arithmetic(unittest.TestCase):
    def test_repeated_response_exp_mean_not_exp_mean_log(self):
        s=f.smearing([row('r0','x',1),row('r0','x',9)],[],dict(coefficients=[]))
        self.assertAlmostEqual(s['value'],5.)
        self.assertNotAlmostEqual(s['value'],3.)

    def test_runs_paths_inputs_balanced_not_request_count(self):
        rr=[row('r0','x',1)]*20+[row('r0','y',3)]+[row('r1','z',10)]
        self.assertAlmostEqual(f.smearing(rr,[],dict(coefficients=[]))['value'],6.)
        rr+=[row('r1','o',2,family='ocr',task='ocr')]
        self.assertAlmostEqual(f.smearing(rr,[],dict(coefficients=[]))['value'],4.)

    def test_family_missing_and_training_run_threshold_unknown(self):
        rr=[row('r0','x',1),row('r1','x',2)]
        self.assertIsNone(f.smearing(rr,[],dict(coefficients=[]),'text')['value'])
        self.assertIsNone(f.smearing(rr,[],dict(coefficients=[]),'ocr')['value'])
        rr+=[row('r2','x',3)]
        self.assertAlmostEqual(f.smearing(rr,[],dict(coefficients=[]),'text')['value'],2.)

    def test_family_restricts_original_weights_without_rebalancing_runs(self):
        rr=[row('r0','x',1),row('r1','x',3),row('r2','x',5),row('r2','o',1,family='ocr',task='ocr')]
        # Original run/path weighting: text masses1/3,1/3,1/6,
        # so family average= (1+3+2.5)/2.5 =2.6, not3.
        self.assertAlmostEqual(f.smearing(rr,[],dict(coefficients=[]),'text')['value'],2.6)

    def test_heldout_actual_cannot_enter_smear(self):
        rr=[row('r0','x',1),row('r1','x',2),row('held','x',100)]
        result=f.fold(rr,[],{'held'},'test','held',{r['run_id']:{'text':1} for r in rr})
        self.assertEqual(result['factors']['global_duan']['training_run_ids'],['r0','r1'])
        self.assertAlmostEqual(result['factors']['global_duan']['value'],1.5)
        r=next(r for r in result['rows'] if r['candidate']=='global_duan')
        self.assertEqual(r['actual_arithmetic_total_ms'],1000.)
        self.assertEqual(r['predicted_arithmetic_total_ms'],15.)
        self.assertAlmostEqual(r['relative_error'],(r['predicted_arithmetic_mean_ms']-r['actual_arithmetic_mean_ms'])/r['actual_arithmetic_mean_ms'])


if __name__=='__main__':unittest.main()
