"""Scientific checks: exact legacy replay, leakage, unsupported paths, censoring."""
import unittest
from .common import *
from .replay import replay
from .calibration import ResidualModel
from module20.optimizer import predict

class Checks(unittest.TestCase):
    def test_instrumented_replay_matches_original_for_mixed_and_pure_routes(self):
        d,refs,w=inputs()
        for rates in RATES.values():
            for fraction in [0.,.18,1.]:
                x={m:1 for m in MODELS};rr=route(d,x,rates,[0.,0.,fraction])
                a=predict(d,x,rr,rates,BUDGET,refs,w,phase=.37);b,_=replay(d,x,rr,rates,BUDGET,refs,w)
                self.assertEqual(a['eligible'],b['eligible'])
                self.assertEqual(a['end_backlog_requests'],b['end_backlog_requests'])
                for t in rates:
                    self.assertAlmostEqual(a['per_task'][t]['mean_ms'],b['per_task'][t]['mean_ms'],places=8)
                    self.assertAlmostEqual(a['per_task'][t]['p95_ms'],b['per_task'][t]['p95_ms'],places=8)
                if a['U'] is not None:self.assertAlmostEqual(a['U'],b['U'],places=11)
                else:self.assertIsNone(b['U'])

    def test_complete_combination_holdout_no_overlap(self):
        rows=read(OUT/'historical_observations.json')
        dev={r['combo'] for r in rows if r['split']=='DEVELOPMENT'}
        hold={r['combo'] for r in rows if r['split']=='RETROSPECTIVE_COMBINATION_HOLDOUT'}
        self.assertFalse(dev & hold)
        for m in [19,22]:self.assertTrue(all(r['split']=='DEVELOPMENT' for r in rows if r['module']==m))
        self.assertTrue(all(r['split']=='RETROSPECTIVE_COMBINATION_HOLDOUT' for r in rows if r['module'] in [16,17]))

    def test_unknown_path_retains_standalone_with_uncertainty(self):
        p=ResidualModel().parameter({G:1,S:1},S,1,'asr')
        self.assertEqual(p['ratio'],1.)
        self.assertEqual(p['scope'],'NO_MEASURED_PATH_SUPPORT')
        self.assertIsNone(p['ratio_range'])

    def test_fast_calibrated_objective_matches_detailed_replay(self):
        d,refs,w=inputs();rates=RATES['HIGH'];x={m:1 for m in MODELS};rr=route(d,x,rates,[0.,0.,.18])
        model=ResidualModel();frozen=read(OUT/'residual_model_history.json');model.exact=frozen['exact'];model.path=frozen['path']
        a,_=replay(d,x,rr,rates,BUDGET,refs,w,model);b,_=replay(d,x,rr,rates,BUDGET,refs,w,model,detail=False)
        self.assertEqual(a['eligible'],b['eligible']);self.assertEqual(a['per_task'],b['per_task'])
        self.assertEqual(a['U'],b['U']);self.assertEqual(a['rho'],b['rho'])

    def test_actual_unstable_or_censored_never_has_sortable_U(self):
        for r in read(OUT/'historical_observations.json'):
            if r['hard_unstable'] or r['censored']:
                self.assertIsNone(r['U']);self.assertFalse(r['eligible'])

    def test_holdout_is_excluded_from_fitted_groups(self):
        model=read(OUT/'residual_model_history.json');hold={r['group'] for r in read(OUT/'historical_observations.json') if r['split']=='RETROSPECTIVE_COMBINATION_HOLDOUT'}
        self.assertFalse(hold & set(model['training_groups']))

    def test_zero_holdout_pairs_does_not_pass_gate(self):
        off=read(OUT/'offline_validation.json')
        self.assertEqual(off['independent_ranking_contrast_groups'],0)
        self.assertFalse(off['independent_ranking_supported'])
        self.assertFalse(off['cross_combination_generalization_established'])

    def test_probe_native_inference_wrapper_only_changes_ready_metadata(self):
        import ast
        original=ast.parse((ROOT/'module13b/worker.py').read_text())
        edited=copy.deepcopy(original)
        changed=0
        for node in ast.walk(edited):
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='run_worker':
                node.value.args[2]=ast.parse("dict(metadata,tasks=['asr'])",mode='eval').body;changed+=1
        self.assertEqual(changed,1)
        before=[ast.dump(n) for n in original.body if isinstance(n,ast.FunctionDef)]
        after=[ast.dump(n) for n in edited.body if isinstance(n,ast.FunctionDef)]
        self.assertEqual(before,after)

if __name__=='__main__':unittest.main()
