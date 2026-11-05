"""Guard error definitions used in the publication figures; no model runs."""
import unittest
from experiments.plot_poisson_predictor_errors_20261105 import ape, stats


class PoissonErrorReportingTests(unittest.TestCase):
    def test_over_and_underprediction_do_not_cancel(self):
        errors=[ape(50,100),ape(150,100)]
        self.assertEqual(stats(errors),dict(n=2,mean_pct=50.,min_pct=50.,max_pct=50.))
        self.assertEqual(ape((50+150)/2,100),0.)

    def test_request_weighted_mean_retains_extreme_and_zero_errors(self):
        values=[ape(100,100),ape(110,100),ape(190,100)]
        s=stats(values)
        self.assertEqual(s['min_pct'],0.)
        self.assertEqual(s['max_pct'],90.)
        self.assertAlmostEqual(s['mean_pct'],100/3)
        self.assertEqual(s['n'],3)

    def test_missing_invalid_or_zero_denominator_never_silently_dropped(self):
        for p,o in [(1,0),(1,-1),(float('nan'),1),(1,float('inf')),(-1,1)]:
            with self.assertRaises(ValueError):ape(p,o)
        for values in [[],[float('nan')],[-1],[float('inf')]]:
            with self.assertRaises(ValueError):stats(values)


if __name__=='__main__':unittest.main()
