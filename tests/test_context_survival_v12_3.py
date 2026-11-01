"""Source-window truncation and leave-capture-out extension regressions."""
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.context_survival_v12_3 import (
    clip_episodes,ContextEnvelopeSurvival,SCHEMA)
from tools.predictor_validation_20261027.profiling import digest


class SourceBoundarySurvivalTests(unittest.TestCase):
    def test_clipped_request_never_becomes_restore_or_observed_save(self):
        result=clip_episodes(0,100,20,80,[])
        self.assertEqual(result,[dict(start_ns=20,end_ns=80,age_entry_ns=20,age_exit_ns=80,save_observed=False)])
        result=clip_episodes(0,100,20,100,[50])
        self.assertEqual([r['age_entry_ns'] for r in result],[20,50])
        self.assertEqual([r['save_observed'] for r in result],[False,True])

    def test_adjacent_request_windows_keep_original_age(self):
        first=clip_episodes(5,105,0,50,[])
        second=clip_episodes(5,105,50,120,[])
        self.assertEqual(first[0]['age_exit_ns'],45)
        self.assertEqual(second[0]['age_entry_ns'],45)
        self.assertFalse(first[0]['save_observed'])
        self.assertTrue(second[0]['save_observed'])
        self.assertEqual(clip_episodes(5,105,110,120,[]),[])

    def test_leave_training_capture_out_is_explicit_and_never_validation(self):
        frontier=[['ready','text','input',['prefill']]]
        rows=[dict(capture=c,envelope_id=c,held_owner='held',frontier=frontier,
                   age_entry_ns=0,age_exit_ns=t,save_observed=True) for c,t in [('a',10),('b',20)]]
        model=dict(schema=SCHEMA,training_only=True,episodes=rows);model['model_sha256']=digest(model)
        with self.assertRaisesRegex(ValueError,'leave-training'):
            ContextEnvelopeSurvival(model,allow_unvalidated_survival_hypothesis=True,
                allow_cpu_fixture=True,exclude_training_capture='a')
        provider=ContextEnvelopeSurvival(model,allow_unvalidated_survival_hypothesis=True,
            allow_cpu_fixture=True,allow_single_capture_training_diagnostic=True,exclude_training_capture='a')
        result=provider.query('held',frontier,0)
        self.assertEqual(result['median_residual_ns'],20)
        self.assertFalse(result['leave_one_capture_out_is_independent_validation'])
        self.assertEqual(result['leave_training_capture_out'],'a')


if __name__ == '__main__':
    unittest.main()
