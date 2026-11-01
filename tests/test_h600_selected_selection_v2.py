"""The selected estimator uses equal, complete H600 comparison evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import h600_selected_validation_v2 as entry
from tools.predictor_validation_20261027.native_flow import receipt


class SelectionAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        evidence=self.write('evidence.json',{'training':True})
        self.rows=[];refs=[receipt(evidence)]
        for seed in (627601,627602):
            plan=self.write(str(seed)+'.json',dict(seed=seed,split='training',window_seconds=600.,
                trace=[{'request_id':str(i)} for i in range(1200)],
                baseline_candidate_id=entry.CANDIDATES[0],candidates=[{'id':c} for c in entry.CANDIDATES],
                workload_rps={t:1. for t in ('asr','image','ocr','text')}))
            refs.append(receipt(plan))
            for index,cid in enumerate(entry.CANDIDATES):
                self.rows.append(dict(seed=seed,candidate_id=cid,requested=1200,known=1200,
                    heldout_plan=receipt(plan),heldout_raw=receipt(evidence),heldout_scored=receipt(evidence),model=receipt(evidence),
                    predicted_comparison_U=-float(index+1),observed_U=-float(index+1),
                    observed_metrics={'measurement':'same for both estimators'},
                    timing_errors={k:{'mean_absolute_relative_error':.01} for k in ('service','e2e')},
                    queue_error={'mae':.01},per_task_p95={t:{'absolute_relative_error':.01} for t in ('asr','image','ocr','text')}))
        self.protocol=dict(training_protocol=receipt(evidence),resource_manifest=receipt(evidence),
            baseline_report=None,created_ns=20)
        self.baseline=dict(schema='gsdeploy.run_aware_H600_numerical_v2_training_report.v1',
            run_aware_protocol=receipt(evidence),resource_manifest=receipt(evidence),
            numerical_source=receipt(Path(entry.__file__).with_name('h600_event_simulator_v2.py')),
            entry_source=receipt(Path(entry.__file__).with_name('h600_numerical_report_v2.py')),created_ns=10)
        self.gamma=dict(schema='gsdeploy.Gamma_H600_training_CV.v1',policy=entry.gamma.POLICY,
            protocol=receipt(evidence),created_ns=30)
        for doc in (self.baseline,self.gamma):
            doc.update(candidates=copy.deepcopy(self.rows),source_receipts=refs,
                requested=9600,known=9600,complete_prediction_coverage=True,
                independent_validation=False,independent_results_read=False,thresholds_tuned=False,final_gain_eligible=False)
            self.assess(doc)
        p=patch.object(entry.gamma,'protocol',side_effect=lambda path:self.protocol)
        p.start();self.addCleanup(p.stop)

    def write(self,name,value):
        path=self.root/name;path.write_text(json.dumps(value));return path

    def assess(self,doc):
        contracts={seed:dict(baseline=entry.CANDIDATES[0],tie_tolerance=0.,
            tasks=['asr','image','ocr','text']) for seed in (627601,627602)}
        doc['original_threshold_assessment']=entry.evaluate(doc,
            dict(seeds=[627601,627602],routes=sorted(entry.CANDIDATES)),entry.LIMITS,contracts)

    @staticmethod
    def score(doc):
        a=doc['original_threshold_assessment']
        return [a['decision_errors'],a['temporal_failed_or_unknown'],
            a['normalized_temporal_error_mean'] if a['normalized_temporal_error_mean'] is not None else float('inf')]

    def audit(self):
        baseline=self.write('baseline.json',self.baseline)
        self.protocol['baseline_report']=receipt(baseline)
        self.write('report.json',self.gamma)
        selected='gamma' if self.score(self.gamma)<self.score(self.baseline) else 'WLS'
        result=self.gamma if selected=='gamma' else self.baseline
        path=self.write('selection.json',dict(baseline=receipt(baseline),gamma_score=self.score(self.gamma),
            baseline_score=self.score(self.baseline),selected=selected,independent_validation=False,
            original_thresholds_passed=result['original_threshold_assessment']['original_training_cv_thresholds_passed']))
        return entry.audit_selection(path)

    def test_exact_tie_keeps_wls_and_better_gamma_wins(self):
        self.assertEqual(self.audit()['selected'],'WLS')
        self.baseline['candidates'][0]['queue_error']['mae']=2.
        self.assess(self.baseline)
        self.assertEqual(self.audit()['selected'],'gamma')

    def test_wls_score_is_recomputed_even_with_self_consistent_selection(self):
        self.baseline['original_threshold_assessment']['decision_errors']=1
        with self.assertRaisesRegex(ValueError,'WLS original-threshold'):self.audit()

    def test_missing_seed_route_duplicate_or_false_total_is_not_a_smaller_domain(self):
        original=copy.deepcopy(self.gamma)
        for change in ('seed','route','duplicate','requested','known','coverage'):
            with self.subTest(change=change):
                self.gamma=copy.deepcopy(original)
                if change=='seed':self.gamma['candidates']=self.gamma['candidates'][:4]
                elif change=='route':self.gamma['candidates']=self.gamma['candidates'][:-1]
                elif change=='duplicate':self.gamma['candidates'][-1]=copy.deepcopy(self.gamma['candidates'][0])
                elif change=='coverage':self.gamma['complete_prediction_coverage']=False
                else:self.gamma[change]-=1
                with self.assertRaisesRegex(ValueError,'two H600 seeds|9600-request'):self.audit()

    def test_comparison_requires_identical_plan_raw_and_actuals(self):
        original=copy.deepcopy(self.gamma)
        alternative=self.write('alternative.json',{'changed':True})
        for field in ('heldout_plan','heldout_raw','heldout_scored','observed_U','observed_metrics'):
            with self.subTest(field=field):
                self.gamma=copy.deepcopy(original)
                if field=='heldout_plan':
                    plan=entry.read(self.rows[0]['heldout_plan']['path']);plan['same_seed_different_inputs']=True
                    self.write('alternative.json',plan)
                    self.gamma['candidates'][0][field]=receipt(alternative)
                    self.gamma['source_receipts'].append(receipt(alternative))
                elif field in ('heldout_raw','heldout_scored'):
                    self.gamma['candidates'][0][field]=receipt(alternative)
                    self.gamma['source_receipts'].append(receipt(alternative))
                elif field=='observed_U':self.gamma['candidates'][0][field]=-.9;self.assess(self.gamma)
                else:self.gamma['candidates'][0][field]={'changed':True}
                with self.assertRaisesRegex(ValueError,'identical heldout'):self.audit()

    def test_source_drift_or_omitted_source_and_nontraining_report_are_rejected(self):
        original=copy.deepcopy(self.gamma)
        self.gamma['source_receipts']=[]
        with self.assertRaisesRegex(ValueError,'source receipts'):self.audit()
        self.gamma=copy.deepcopy(original);self.gamma['source_receipts']=self.gamma['source_receipts'][:1]
        with self.assertRaisesRegex(ValueError,'omitted a case source'):self.audit()
        self.gamma=copy.deepcopy(original);self.gamma['independent_results_read']=True
        with self.assertRaisesRegex(ValueError,'Frozen complete Gamma'):self.audit()
        self.gamma=copy.deepcopy(original);self.write('evidence.json',{'changed':True})
        with self.assertRaises(ValueError):self.audit()

    def test_numerical_v2_and_prospective_selection_protocol_are_required(self):
        original=copy.deepcopy(self.baseline)
        for field in ('schema','run_aware_protocol','resource_manifest','numerical_source','entry_source','created_ns'):
            with self.subTest(field=field):
                self.baseline=copy.deepcopy(original)
                self.baseline[field]=21 if field=='created_ns' else 'wrong'
                with self.assertRaisesRegex(ValueError,'prospective numerical-v2'):self.audit()


if __name__=='__main__':unittest.main()
