"""Real frozen development objective integration; no GPU or synthetic scores."""
import unittest
from tools.spillover_context_20261022.development_oracle import DevelopmentOracle
from tools.spillover_context_20261022.development_bounds import DevelopmentBounds
from tools.spillover_context_20261022.development_affine_bounds import DevelopmentAffineBounds
from tools.spillover_context_20261022.search_development import domain,identity


class DevelopmentSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.o=DevelopmentOracle()

    def test_domain_has_all_role_endpoints_and_free_routes(self):
        d=domain(self.o.catalog)
        self.assertEqual(d['complete_plans'],7508)
        self.assertEqual(d['complete_deployments'],40)
        self.assertTrue(any(v['deployment']=={'generalist_2':1} for v in d['by_deployment']))
        self.assertTrue(any(not any(m.startswith('generalist') for m in v['deployment']) for v in d['by_deployment']))
        self.assertFalse(d['full_seven_pool_optimality'])

    def test_actual_envelopes_cover_six_term_cost_and_fast_mode(self):
        bounds=DevelopmentBounds(self.o,[107,108]);observed=[]
        for share in [0,1/32,.5,1.]:
            p=dict(deployment={'generalist_2':1,'text':2,'ocr':1},routing={
                'text':{m:q for m,q in [('generalist_2',1.85*share),('text',1.85*(1-share))] if q},
                'image':{'generalist_2':.01},'asr':{'generalist_2':.01},'ocr':{'ocr':.01}})
            scores=[]
            for seed in [107,108]:
                r=self.o.evaluate(p,seed,True,strict_context_checks=False)
                self.assertIsNone(r['U']);self.assertIsNone(r['supported_domain_development_arithmetic_U'])
                d=r['diagnostic_extrapolation_only'];scores.append(d['U'])
                self.assertTrue(all(v>=0 for v in d['terms'].values()))
            u=sum(scores)/2;self.assertGreaterEqual(bounds.upper(p['deployment'],p['routing'],0),u-1e-10)
            observed.append(identity(p))
        self.assertEqual(len(set(observed)),4)
        self.o.verify_inputs()

    def test_affine_resource_constant_once_and_rho_lower_envelope(self):
        b=DevelopmentAffineBounds(self.o,[107,108]);x={'generalist_2':1,'text':2,'ocr':1}
        route={'text':{'text':1.7921875,'generalist_2':.0578125},'image':{'generalist_2':.01},
               'asr':{'generalist_2':.01},'ocr':{'ocr':.01}}
        results=[self.o.evaluate(dict(deployment=x,routing=route),s,True,strict_context_checks=False)['diagnostic_extrapolation_only'] for s in [107,108]]
        self.assertGreaterEqual(b.upper(x,route,0),sum(r['U'] for r in results)/2-1e-10)
        # Parent floor includes only Q/T/L/P. The difference is the complete
        # affine M/E objective, including its intercept exactly once.
        parent=DevelopmentBounds(self.o,[107,108])
        resources=-b.upper(x,route,0)-sum(parent.task_floor(t,x,row) for t,row in route.items())
        expected=sum(self.o.config['weights'][t]*results[0]['terms'][t] for t in ['M','E'])
        self.assertAlmostEqual(resources,expected,places=11)
        for m,n in x.items():
            lower=sum(row.get(m,0)*b.pressure_service(m,t,n)/1000/self.o.catalog['capacity_factors'][m][str(n)] for t,row in route.items())
            self.assertLessEqual(lower,results[0]['rho_by_family'][m]+1e-11)
        self.o.verify_inputs()


if __name__=='__main__':unittest.main()
