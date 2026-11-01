import math
import unittest

from scipy.stats import nbinom

from tools.predictor_validation_20261027 import projected_age_composition_v57_cpu as age
from tools.predictor_validation_20261027 import projected_state_adequacy_v57_cpu as adequacy


class DiagnosticMathTests(unittest.TestCase):
    def test_original_grid_exact_boundary_events_and_tail_exposure(self):
        parts=list(age.split(500,2500,True))
        self.assertEqual(parts,[(0,500,0),(1,1000,0),(2,500,1)])
        self.assertEqual(list(age.split(0,1000,True)),[(0,1000,1)])
        self.assertEqual(list(age.split(50_000_000,2_000_000_000,True)),[(15,1_950_000_000,1)])

    def test_fft_count_convolution_equals_closed_form_common_p(self):
        # Equal exposure ratios produce a closed NB sum even for different
        # shapes. This checks the convolution and integer quantile convention.
        cells=[age.prediction(3,100,50,1),age.prediction(12,300,150,2)]
        result=age.sum_bounds(cells);shape=3.5+12.5;p=2/3
        self.assertEqual(result['bounds95'],[int(nbinom.ppf(q,shape,p)) for q in (.025,.975)])
        self.assertLess(abs(result['convolved_mass']-1),1e-9)
        self.assertLess(result['analytic_omitted_tail_upper_bound'],2.1e-12)

    def test_marginal_predictive_log_and_no_borrowing(self):
        result=age.prediction(0,100,200,0)
        self.assertAlmostEqual(result['log_predictive_probability'],.5*math.log(1/3))
        self.assertEqual(result['expected'],1)
        self.assertEqual(age.prediction(0,0,200,3)['status'],'UNKNOWN_NO_TRAINING_EXPOSURE')

    def test_product_limit_tied_censor_and_terminal_events(self):
        result=adequacy.product_limit([(1.,True),(1.,False),(2.,True)],grid=(.5,1.,2.))
        values=result['product_limit_survival']
        self.assertEqual(values[0]['estimated_survival'],1.)
        self.assertAlmostEqual(values[1]['estimated_survival'],2/3)
        self.assertEqual(values[1]['at_risk_after_threshold'],1)
        self.assertEqual(values[2]['estimated_survival'],0.)
        self.assertEqual((result['events'],result['right_censored']),(2,1))


if __name__=='__main__':unittest.main()
