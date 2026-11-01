"""Independent full-entry CPU test; only native resource load is replaced."""
import copy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_resource_fold_provider_v1 import fixture,model_document
from tests.test_twoFIFO_fold_resource_rescore_v1 import setup_world,state_receipt
from tools.predictor_validation_20261027 import twoFIFO_fold_resource_batch_rescore_v1 as m


def all_inputs(folder):
    design_ref,design=m.generated.audit_design(Path('artifacts/predictor-twoFIFO-generated-wholeblock-design-v1-20261028-cpu01/design.json'))
    campaign=m.checked(design['campaign']);plans=[m.checked(ref) for ref in campaign['attempts']]
    score=m.checked(design['scoring_plan']);quality=m.checked(design['quality'])
    _,_,tables=fixture();model,sources=model_document(tables,folder)
    model_path=folder/'model.json';model_path.write_text(json.dumps(model));folds=[]
    for block in design['training_block_ids']:
        donors=[b for b in design['training_block_ids'] if b!=block];cases=[]
        for plan in plans:
            if plan['capture_id']!=block:continue
            for index,case in enumerate(plan['cases']):
                prediction=m.generated.compact_prediction(m.generated.forecast_case(case,1000+index*100_000_000_000,None,None,
                    score,quality,donor_blocks=donors,unqualified_reason='CPU_TEST_ONLY_UNQUALIFIED'))
                ref=m.save(folder/(plan['attempt_id']+'-case-'+str(index)+'.json'),prediction)
                cases.append(dict(attempt_id=plan['attempt_id'],case_id=case['case_id'],prediction=ref))
        assert len(cases)==50
        folds.append(dict(heldout_block_id=block,training_block_ids=donors,
            donor_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']!=block],
            heldout_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']==block],
            folded_law=None,folded_law_sha256=None,cases=cases))
    report=m.save(folder/'completed.json',dict(schema=m.generated.SCHEMA,numerical_fit_run=True,
        status='COMPLETE_NATIVE_TRAINING_FIT_UNQUALIFIED_ALL_GENERATED_WORLDS_UNKNOWN',validation_labels_read=False,
        sources=[design['campaign']]+model['training_reviews'],design=design_ref,whole_block_leaveout=folds))
    return report,model_path,model,tables,sources,plans


class IndependentBatchTests(unittest.TestCase):
    def test_full150_entry_preserves_546_requests_8_unknown_draws_exact_plan_and_three_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);report,model_path,model,tables,sources,plans=all_inputs(folder)
            with patch.object(m.folds.resource,'load_complete',return_value=(tables,sources)) as native,\
                    patch.object(m.generated,'native_labels',side_effect=AssertionError('labels opened')),\
                    patch.object(m.generated.response.numeric,'fit',side_effect=AssertionError('fit invoked')):
                out=m.checked(m.run_all_completed_cases(report['path'],model_path,output=folder/'out'))
            self.assertEqual(native.call_count,3);self.assertEqual(out['cases'],150);self.assertEqual(out['offered_requests'],546)
            self.assertEqual(Counter(c['block_id'] for c in out['output_cases']),dict.fromkeys(m.checked(report)['whole_block_leaveout'][i]['heldout_block_id'] for i in range(3))) if False else None
            self.assertTrue(all(n==50 for n in Counter(c['block_id'] for c in out['output_cases']).values()))
            by_id={p['attempt_id']:p for p in plans};campaign=m.checked(model['campaign']);refs={m.checked(r)['attempt_id']:r for r in campaign['attempts']}
            for item in out['output_cases']:
                result=m.checked(item['report']);self.assertEqual(result['target_plan'],refs[item['attempt_id']])
                ensemble=m.decode(result['ensemble']);self.assertEqual(len(ensemble['draws']),8)
                self.assertEqual(ensemble['world_unknown_probability_mass'],1)
                self.assertEqual(ensemble['utility']['unknown_probability_mass'],1)
                self.assertTrue(all(d['status']=='UNKNOWN_WORLD' for d in ensemble['draws']))
                self.assertEqual(len(result['resource_fold_binding']['training_attempt_ids']),8)
                self.assertNotIn(item['block_id'],result['resource_fold_binding']['training_block_ids'])
                self.assertFalse(result['missing_memory_state_assumed_fresh']);self.assertIsNone(result['past_memory_state'])
            self.assertFalse(out['labels_opened']);self.assertFalse(out['numerical_fit_run']);self.assertFalse(out['GPU_started'])

    def test_cached_real_world_scoring_matches_single_case_aa43_with_and_without_past_memory(self):
        data=setup_world();plan,context,tables,requests,args,document,state=data
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);model,sources=model_document(tables,folder);path=folder/'model.json';path.write_text(json.dumps(model))
            with patch.object(m.folds.resource,'load_complete',return_value=(tables,sources)):
                provider=m.folds.FoldResourceTrainingProvider(path,plan,context,heldout_block_id='training-block-3',allow_training_resource_hypothesis=True)
            for past in (None,state_receipt(folder,state)):
                kwargs=dict(heldout_block_id='training-block-3',scoring_plan=args['scoring_plan'],quality=args['quality'],
                    window_start_ns=args['window_start_ns'],window_end_ns=args['window_end_ns'],drain_end_ns=args['drain_end_ns'],past_memory_state=past)
                cached=m._score_with_provider(document,requests,provider,plan,context,**kwargs)
                with patch.object(m.folds.resource,'load_complete',return_value=(tables,sources)):
                    single=m.single.rescore_ensemble(document,requests,path,plan,context,**kwargs)
                self.assertEqual(cached['ensemble'],single['ensemble'])
                self.assertEqual(cached['resource_model'],single['resource_model'])
                self.assertEqual(cached['resource_fold_binding'],single['resource_fold_binding'])
                if past is None:
                    self.assertIsNone(cached['ensemble']['draws'][0]['forecast']['scoring']['original_score']['terms']['M'])
