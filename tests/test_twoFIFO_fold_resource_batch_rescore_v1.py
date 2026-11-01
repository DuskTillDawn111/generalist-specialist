import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_fold_resource_rescore_v1 import setup_world, execute
from tests.test_twoFIFO_resource_fold_provider_v1 import fixture, model_document
from tools.predictor_validation_20261027 import twoFIFO_fold_resource_batch_rescore_v1 as m


class CacheTests(unittest.TestCase):
    def test_build_cache_constructs_one_provider_per_fold(self):
        plan, context, _ = fixture(); plans=[]; folds=[]
        for block in ('training-block-1','training-block-2','training-block-3'):
            for shard in range(1,5):
                p=copy.deepcopy(plan);p['capture_id']=block;p['attempt_id']=block+'-shard-'+str(shard)
                plans.append(p)
        for block in ('training-block-1','training-block-2','training-block-3'):
            folds.append(dict(heldout_block_id=block,
                donor_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']!=block],
                heldout_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']==block]))
        calls=[]
        class Fake:
            def __init__(self,*args,**kwargs):
                calls.append(1)
                self.target_plan=args[1];self.target_context=args[2]
                self.fold_binding=dict(heldout_block_id=kwargs['heldout_block_id'],sources=[])
        with patch.object(m.folds,'FoldResourceTrainingProvider',Fake) as ctor:
            cache=m.build_fold_provider_cache(Path('model.json'),plans,folds)
        self.assertEqual(len(calls),3);self.assertEqual(set(cache),set(('training-block-1','training-block-2','training-block-3')))

    def test_single_case_rescore_contract_stays_available_for_cached_provider_path(self):
        data=setup_world()
        with tempfile.TemporaryDirectory() as directory:
            out=execute(directory,data)
        self.assertEqual(out['ensemble']['offered_requests'],2)
        self.assertEqual(len(out['ensemble']['draws']),1)
        self.assertTrue(out['all_original_draws_and_probability_mass_retained'])

    def test_cache_rejects_incomplete_fold_before_constructing_extra_provider(self):
        plan,_,_=fixture();plans=[];folds=[]
        for block in ('training-block-1','training-block-2'):
            for shard in range(1,5):
                p=copy.deepcopy(plan);p['capture_id']=block;p['attempt_id']=block+'-shard-'+str(shard);plans.append(p)
            folds.append(dict(heldout_block_id=block,
                donor_attempt_ids=[],heldout_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']==block]))
        with patch.object(m.folds,'FoldResourceTrainingProvider') as ctor,self.assertRaises(ValueError):
            m.build_fold_provider_cache(Path('model.json'),plans,folds)
        self.assertEqual(ctor.call_count,0)

    def test_memory_receipts_require_full_case_key_when_case_ids_repeat(self):
        with self.assertRaises(ValueError):
            m.resolve_past_memory_ref({'same': 'wrong'}, block_id='training-block-1',
                attempt_id='a', case_id='same', case_id_count=2)
        self.assertEqual(m.resolve_past_memory_ref({'training-block-1|a|same':'ok'},
            block_id='training-block-1', attempt_id='a', case_id='same', case_id_count=2), 'ok')
        self.assertEqual(m.resolve_past_memory_ref({'same':'ok'}, block_id='training-block-1',
            attempt_id='a', case_id='same', case_id_count=1), 'ok')

    def test_full_actual_report_schema_all150_cases_write_exact_plans_and_three_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            design_ref,design=m.generated.audit_design(Path('artifacts/predictor-twoFIFO-generated-wholeblock-design-v1-20261028-cpu01/design.json'))
            campaign=m.checked(design['campaign']);plans=[m.checked(ref) for ref in campaign['attempts']]
            score=m.checked(design['scoring_plan']);quality=m.checked(design['quality'])
            _,_,tables=fixture();model,sources=model_document(tables,folder)
            model_path=folder/'model.json';m.save(model_path,model)
            folds=[];expected={}
            for block in design['training_block_ids']:
                cases=[];donors=[b for b in design['training_block_ids'] if b!=block]
                for plan_ref,plan in zip(campaign['attempts'],plans):
                    if plan['capture_id']!=block:continue
                    for index,case in enumerate(plan['cases']):
                        prediction=m.generated.forecast_case(case,1000+index*70_000_000_000,None,None,score,quality,
                            donor_blocks=donors,unqualified_reason='CPU_TEST_ONLY_UNKNOWN')
                        prediction_ref=m.save(folder/('prediction-'+str(len(expected))+'.json'),
                            m.generated.compact_prediction(prediction))
                        cases.append(dict(attempt_id=plan['attempt_id'],case_id=case['case_id'],prediction=prediction_ref))
                        expected[(block,plan['attempt_id'],case['case_id'])]=plan_ref
                folds.append(dict(heldout_block_id=block,training_block_ids=donors,
                    donor_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']!=block],
                    heldout_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']==block],
                    folded_law=None,cases=cases))
            report=m.save(folder/'completed.json',dict(schema=m.generated.SCHEMA,numerical_fit_run=True,
                status='COMPLETE_NATIVE_TRAINING_FIT_UNQUALIFIED_ALL_GENERATED_WORLDS_UNKNOWN',
                validation_labels_read=False,sources=[design['campaign']]+model['training_reviews'],
                design=design_ref,whole_block_leaveout=folds))
            with patch.object(m.folds.resource,'load_complete',return_value=(tables,sources)) as loader:
                result=m.checked(m.run_all_completed_cases(report['path'],model_path,output=folder/'out'))
            self.assertEqual(loader.call_count,3);self.assertEqual(result['cases'],150)
            self.assertEqual(result['offered_requests'],546);self.assertEqual(len(result['output_cases']),150)
            for row in result['output_cases']:
                case=m.checked(row['report']);key=(row['block_id'],row['attempt_id'],row['case_id'])
                self.assertEqual(case['target_plan'],expected[key])
                self.assertEqual(len(case['ensemble']['draws']),8)
                self.assertEqual(m.single.decode(case['ensemble'])['world_unknown_probability_mass'],1)


if __name__=='__main__':unittest.main()
