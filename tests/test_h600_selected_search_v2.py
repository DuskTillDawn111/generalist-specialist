"""Same-domain search preserves public plans, all ties and unknown feasibility."""
import copy
import unittest
from unittest.mock import patch
from tests import test_h600_selected_validation_v2 as fixtures
from tools.predictor_validation_20261027 import h600_selected_search_v2 as module
from tools.predictor_validation_20261027.native_flow import receipt


class H600SearchTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.H600ValidationTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture;f=fixture.f;self.f=f
        self.ids=module.validation.CANDIDATES
        template=f.plan['candidates'][0];f.plan['candidates']=[]
        for index,cid in enumerate(self.ids):
            candidate=copy.deepcopy(template);candidate['id']=cid;f.plan['candidates'].append(candidate)
        f.plan.update(window_seconds=600.,drain_seconds=300.,baseline_candidate_id=self.ids[0])
        f.plan.pop('plan_sha256',None);f.plan.pop('startup_comparison_policy',None)
        f.write('plan.json',f.plan)
        self.public=copy.deepcopy(f.plan);self.before=f.plan_path.read_bytes()
        f.plan['catalog']=receipt(f.quality_path);f.write('plan.json',f.plan)
        self.public=copy.deepcopy(f.plan);self.before=f.plan_path.read_bytes()
        self.overrides={cid:dict(predicted_startup_joules=1000.+index,predicted_prephase_peak_mib=2000.+index)
            for index,cid in enumerate(self.ids)}
        protocol=dict(target_plan=receipt(f.plan_path),source_receipts=[],selection=receipt(f.model_path),estimator_selection={'selected':'WLS'})
        f.write(fixture.protocol.name,protocol)
        resources=dict(full=dict(resources=receipt(f.resources_path),candidate_overrides=self.overrides),source_receipts=[])
        patches=[patch.object(module.validation,'audit_protocol',return_value=protocol),
            patch.object(module.validation,'audit_plan',return_value=self.public),
            patch.object(module.validation,'audit_model',return_value=fixture.valid),
            patch.object(module.validation,'audit_resources',return_value=resources),
            patch.object(module,'predictor_sources',return_value=[])]
        for item in patches:item.start();self.addCleanup(item.stop)

    def run_check(self,evaluate):
        f=self.f;fixture=self.fixture
        with patch.object(module.core,'evaluate',side_effect=evaluate) as forecast:
            result=module.run(fixture.protocol,f.model_path,fixture.resource_manifest,f.quality_path,f.root/'search')
        return result,forecast

    def evaluation(self,candidate,*,value=-10.,eligible=True):
        return dict(comparison_U=value,model_eligible=eligible,
            original_rho={'status':'UNKNOWN','rho_admission_passed':None})

    def test_overlay_same_identity_two_fresh_searches_preserve_plan_and_all_ties(self):
        seen=[]
        def evaluate(plan,candidate,*args):
            seen.append(candidate['id']);self.assertEqual(candidate['routes'],self.public['candidates'][0]['routes'])
            self.assertEqual(candidate['predicted_startup_joules'],self.overrides[candidate['id']]['predicted_startup_joules'])
            self.assertEqual(candidate['predicted_prephase_peak_mib'],self.overrides[candidate['id']]['predicted_prephase_peak_mib'])
            value=-10. if candidate['id'] in self.ids[:2] else -11.
            return self.evaluation(candidate,value=value)
        result,forecast=self.run_check(evaluate)
        self.assertEqual(forecast.call_count,8);self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['all_tie_ids'],sorted(self.ids[:2]));self.assertEqual(self.before,self.f.plan_path.read_bytes())
        self.assertEqual(self.public,module.read(self.f.plan_path));self.assertFalse(result['search_acceleration_claim'])
        self.assertEqual(result['pruning_count'],0);self.assertFalse(result['original_gate_constrained_optimality_certified'])
        self.assertEqual(result['original_rho_admission_unknown_ids'],sorted(self.ids))

    def test_unknown_candidate_remains_in_full_domain_and_cannot_pass(self):
        def evaluate(plan,candidate,*args):
            if candidate['id']==self.ids[1]:return self.evaluation(candidate,value=None,eligible=None)
            return self.evaluation(candidate)
        result,forecast=self.run_check(evaluate)
        self.assertEqual(result['status'],'UNKNOWN');self.assertEqual(result['candidate_count'],4)
        self.assertEqual(result['plain']['unknown_ids'],[self.ids[1]])
        self.assertIsNone(result['same_optimum_and_all_ties']);self.assertEqual(forecast.call_count,8)

    def test_no_model_feasible_candidate_is_not_a_pass_or_an_optimum(self):
        result,forecast=self.run_check(lambda plan,candidate,*args:self.evaluation(candidate,eligible=False))
        self.assertEqual(result['status'],'NO_MODEL_FEASIBLE_CANDIDATE')
        self.assertTrue(result['no_model_feasible_candidate']);self.assertEqual(result['all_tie_ids'],[])
        self.assertIsNone(result['maximum_model_comparison_U']);self.assertIsNone(result['same_optimum_and_all_ties'])
        self.assertFalse(result['physical_optimality_certified']);self.assertEqual(forecast.call_count,8)

    def test_quality_context_drift_fails_before_either_search(self):
        f=self.f;quality=module.read(f.quality_path);quality.update(execution_context_sha256='other')
        f.write(f.quality_path.name,quality)
        with patch.object(module.core,'search') as search:
            with self.assertRaisesRegex(ValueError,'identities'):
                module.run(self.fixture.protocol,f.model_path,self.fixture.resource_manifest,f.quality_path,f.root/'search')
            search.assert_not_called()
        self.assertFalse((f.root/'search').exists())


if __name__=='__main__':unittest.main()
