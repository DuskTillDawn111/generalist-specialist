import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.predictor_validation_20261027 import context_next_owner_v14 as m
from tools.predictor_validation_20261027.native_collect import receipt
from tools.predictor_validation_20261027.profiling import digest


POOLS={'short':'text','ocr':'ocr','image':'image'}


def query(previous='text#1'):
    return dict(held_owner='text#0',context_age_ns=1_000_000,
        pending_frontier=[['text#0','text','short',['prefill']]],
        previous_held_owner=previous,public_workload=None,
        pending_age_ns_by_owner=None,pending_reference_work_ns_by_owner=None)


def transition(capture,a,b,next_restore,held,target,**extra):
    return dict(capture=capture,current_restore_start_ns=a,save_end_ns=b,
        next_restore_start_ns=next_restore,held_owner=held,next_owner=target,
        context_age_at_save_ns=b-a,job_id='j',pending_frontier=[],
        public_workload=[['text#0','text','short',0]],next_owner_is_pending=False,**extra)


def fixture():
    names=m.feature_names(POOLS);width=len(names);coef=np.zeros((width,4))
    coef[0,0]=3;coef[names.index('previous_held:text#1'),2]=5
    model=dict(schema=m.SCHEMA,training_only=True,CPU_fixture=True,parameters=copy.deepcopy(m.PARAMETERS),
        public_pool_tasks=POOLS,residents=list(m.RESIDENTS),feature_names=names,
        cutoff_state_signatures=[m.cutoff_signature(query())],
        fitted=dict(center=[0.]*width,scale=[1.]*width,coefficients=coef.tolist(),
            feature_minimum=[0.]*width,feature_maximum=[10.]*width,
            held_log_age_ranges={'text#0':[0.,10.]},label_counts={o:int(o=='text#0') for o in m.RESIDENTS}))
    model['model_sha256']=digest(model);return model


class CurrentFrontierMarkTests(unittest.TestCase):
    def test_same_capture_unique_past_link_missing_and_ambiguous_history_retained(self):
        a=transition('a',0,4,5,'generalist_2#0','text#0')
        b=transition('a',5,9,10,'text#0','text#1')
        foreign=transition('b',10,14,15,'text#1','text#0')
        c=transition('a',15,19,20,'text#0','text#1')
        history=m.previous_history([a,b,foreign,c])
        self.assertEqual(history[1]['previous_held_owner'],'generalist_2#0')
        self.assertIsNone(history[0]['previous_held_owner']);self.assertIsNone(history[3]['previous_held_owner'])
        duplicate=m.previous_history([a,copy.deepcopy(a),b])
        self.assertEqual(duplicate[-1]['status'],'AMBIGUOUS_PRIOR_HANDOFF');self.assertIsNone(duplicate[-1]['previous_held_owner'])
        self.assertEqual(len(duplicate),3)

    def test_future_57_cohort_counts_offsets_and_order_cannot_change_mark_or_OOD(self):
        fit=m.NextOwnerCandidate(fixture(),allow_cpu_fixture=True);base=query();reference=fit.predict_given_save_now(**base)
        future=[]
        for i in range(57):
            future.extend([['text#0','text','short',i*10**9],['text#1','text','short',i*10**10],
                ['ocr#0','ocr','ocr',i*10**11],['generalist_2#0','image','image',i*10**12]])
        for workload in (future,list(reversed(future)),[['text#0','text','short',0]]):
            changed=dict(base,public_workload=workload)
            self.assertEqual(m.features(base,POOLS),m.features(changed,POOLS))
            self.assertEqual(reference,fit.predict_given_save_now(**changed))
        self.assertFalse(any('workload' in n or 'offset' in n for n in m.feature_names(POOLS)))

    def test_history_is_a_feature_and_missing_history_is_explicit(self):
        fit=m.NextOwnerCandidate(fixture(),allow_cpu_fixture=True)
        with_history=fit.predict_given_save_now(**query());missing=fit.predict_given_save_now(**query(None))
        self.assertNotEqual(with_history['next_owner_probabilities'],missing['next_owner_probabilities'])
        self.assertTrue(missing['previous_held_owner_missing']);self.assertIsNotNone(missing['history_warning'])
        self.assertFalse(missing['complete_availability_oracle']);self.assertEqual(len(missing['next_owner_probabilities']),4)
        self.assertIn('ocr#0',missing['zero_observed_training_target_classes'])
        self.assertFalse(missing['save_time_predicted']);self.assertFalse(missing['service_ready_time_predicted'])

    def test_no_previous_history_ablation_omits_all_history_features(self):
        self.assertEqual(m.features(query(None),POOLS,include_previous_history=False),
            m.features(query('generalist_2#0'),POOLS,include_previous_history=False))
        self.assertFalse(any('previous' in n for n in m.feature_names(POOLS,include_previous_history=False)))

    def test_future_labels_work_and_unsupported_current_identity_are_rejected(self):
        for key in ('next_owner','next_restore_start_ns','save_end_ns'):
            with self.assertRaisesRegex(ValueError,'future label'):m.features(dict(query(),**{key:1}),POOLS)
        with self.assertRaises(ValueError):m.features(dict(query(),previous_held_owner='foreign'),POOLS)
        with self.assertRaises(ValueError):m.features(dict(query(),pending_age_ns_by_owner={'text#0':1}),POOLS)
        duplicate=query();duplicate['pending_frontier']*=2
        with self.assertRaisesRegex(ValueError,'FIFO'):m.features(duplicate,POOLS)
        bad=dict(query(),public_workload=[['text#0','text','foreign',0]])
        with self.assertRaises(ValueError):m.features(bad,POOLS)

    def test_OOD_age_and_unseen_current_history_combination_are_reported(self):
        fit=m.NextOwnerCandidate(fixture(),allow_cpu_fixture=True)
        result=fit.predict_given_save_now(**dict(query(),context_age_ns=10**14,previous_held_owner='ocr#0'))
        self.assertIn('HELD_OWNER_AGE_OUTSIDE_TRAINING_SUPPORT',result['ood_flags'])
        self.assertIn('HELD_FRONTIER_PREVIOUS_COMBINATION_UNSEEN',result['ood_flags'])

    def test_lookup_has_full_coverage_train_only_holder_and_global_fallbacks(self):
        train=[transition('a',0,4,5,'text#0','text#1')]
        test=[transition('b',0,4,5,'text#0','text#1'),transition('b',5,9,10,'ocr#0','text#1')]
        test[0]['pending_frontier']=[['text#0','text','short',['decode']]]
        history=[{'previous_held_owner':None}]
        predictions,fallbacks=m.baseline_predictions(train,test,history*2,history)
        for values in predictions.values():self.assertEqual(values.tolist(),[3,3])
        self.assertEqual(fallbacks['holder_frontier_lookup_with_fallback'],2)

    def test_fixture_optin_and_changed_digest_are_refused(self):
        with self.assertRaisesRegex(ValueError,'CPU fixture'):m.NextOwnerCandidate(fixture())
        changed=fixture();changed['fitted']['coefficients'][0][0]+=1
        with self.assertRaisesRegex(ValueError,'Source-bound'):m.NextOwnerCandidate(changed,allow_cpu_fixture=True)

    def test_train_complete_capture_CV_history_ablation_and_artifact_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audits=[];rows=[]
            for capture in ('capture-a','capture-b'):
                plan=root/(capture+'.json');plan.write_text(json.dumps(dict(inputs=[dict(pool_id='short',task='text')])))
                audits.append(dict(raw={'sha256':capture},plan=receipt(plan),report={'path':'unused','sha256':capture}))
                rows += [transition(capture,0,4,5,'text#0','text#1'),transition(capture,5,9,10,'text#1','text#0'),
                         transition(capture,10,14,15,'text#0','text#1')]
            sup=dict(schema=m.SUPERVISION_SCHEMA,training_only=True,validation_labels_used=False,
                training_audits=audits,sources=[],transition_observations=rows,initial_observations=[])
            sup['model_sha256']=digest(sup);source=root/'supervision.json';source.write_text(json.dumps(sup))
            with patch.object(m,'availability_training_sources',return_value=audits):
                report=m.train(source,root/'output')
            self.assertEqual(report['transitions'],6);self.assertEqual(report['all_target_not_pending_rows_retained'],6)
            self.assertEqual(report['previous_history_status_counts']['MISSING_PRIOR_HANDOFF'],2)
            self.assertEqual(len(report['folds']),2)
            for fold in report['folds']:
                self.assertNotIn(fold['excluded_capture'],fold['training_captures'])
                self.assertEqual(fold['candidate']['requested_transitions'],3)
                self.assertEqual(fold['holder_frontier_lookup_with_fallback']['unknown_transitions'],0)
                self.assertEqual(fold['no_previous_history_ablation']['requested_transitions'],3)
            fit=m.NextOwnerCandidate(json.loads((root/'output/model.json').read_text()))
            self.assertEqual(len((root/'output/capture-heldout-predictions.jsonl').read_text().splitlines()),6)
            self.assertEqual(len((root/'output/previous-holder-observations.jsonl').read_text().splitlines()),6)
            self.assertFalse(fit.predict_given_save_now(**dict(query(),public_workload=None))['future_public_workload_used_as_features'])
            history=root/'output/previous-holder-observations.jsonl';history.write_text('changed')
            with self.assertRaisesRegex(ValueError,'source-link'):m.NextOwnerCandidate(fit.model)


if __name__=='__main__':unittest.main()
