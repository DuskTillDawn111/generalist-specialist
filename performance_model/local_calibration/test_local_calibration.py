"""Scientific accounting, time semantics and nested-group leakage tests."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from collections import defaultdict
import numpy as np

from performance_model.profiling.plan import ROOT, extractor
from performance_model.generalization.evaluate import metrics
from .stability_v2 import DEFAULT_THRESHOLDS, scaled_thresholds, reconstruct, classify, timeline
from .calibration import selection_plan, bias_from_probes, planning_descriptor


def record(i, submitted, completion, **overrides):
    return dict(request_id=str(i), task='text', model='m', scheduled_time_ns=0,
        actual_submit_time_ns=submitted, completion_time_ns=completion,
        status='completed' if completion is not None else 'incomplete', **overrides)


class SemanticsTests(unittest.TestCase):
    def setUp(self): self.catalog={'m':SimpleNamespace(service_time_ms={'text':100.})}

    def test_submitted_not_scheduled_and_end_not_final(self):
        rs=[record(0,2_000_000_000,4_000_000_000)]
        x=timeline(rs,0,3_000_000_000,5_000_000_000,self.catalog,3,'actual_submit_time_ns')
        self.assertEqual(x['peak_backlog_requests'],1)
        self.assertEqual(x['end_backlog_requests'],1)
        self.assertEqual(x['final_backlog_after_drain'],0)
        self.assertAlmostEqual(x['peak_service_demand_backlog'],.1)
        self.assertEqual(x['bins'][0]['mean_backlog_requests'],0)
        self.assertEqual(x['bins'][2]['mean_backlog_requests'],1)
        x=timeline(rs,0,3_000_000_000,5_000_000_000,self.catalog,3,'scheduled_time_ns')
        self.assertEqual(x['bins'][0]['mean_backlog_requests'],1)

    def test_simultaneous_submission_completion_does_not_inflate_peak(self):
        x=timeline([record(0,0,1_000_000_000),record(1,1_000_000_000,2_000_000_000)],
                   0,3_000_000_000,3_000_000_000,self.catalog,3,'actual_submit_time_ns')
        self.assertEqual(x['peak_backlog_requests'],1)
        self.assertAlmostEqual(x['peak_service_demand_backlog'],.1)

    def fixture(self,n=24):
        rs=[record(i,100_000_000,200_000_000) for i in range(n)]
        return dict(request_records=rs,measurement_window=dict(start_time_ns=0,end_time_ns=10_000_000_000,
            drain_end_time_ns=10_000_000_000,drain_seconds=0),status='completed',errors=[],shutdown={},drain_timeout_seconds=60)

    def test_warning_sensitivity_and_latency_is_not_a_label_threshold(self):
        data=self.fixture()
        for r in data['request_records']: r['e2e_latency_ms']=10000000
        e=reconstruct(data,self.catalog,DEFAULT_THRESHOLDS)
        self.assertEqual(classify(e,scaled_thresholds(.75))['stability_state_v2'],'recoverable_congested')
        self.assertEqual(classify(e,scaled_thresholds(1))['stability_state_v2'],'steady')
        self.assertEqual(classify(e,scaled_thresholds(1.25))['stability_state_v2'],'steady')

    def test_timeout_unstable_even_if_late_response_exists(self):
        data=self.fixture(1)
        data['request_records'][0].update(status='incomplete',completion_time_ns=None,error='drain timeout',
                                          late_response_time_ns=11_000_000_000)
        e=reconstruct(data,self.catalog,DEFAULT_THRESHOLDS)
        for factor in (.75,1,1.25): self.assertEqual(classify(e,scaled_thresholds(factor))['stability_state_v2'],'unstable')
        self.assertTrue(e['drain_timeout'])
        self.assertEqual(e['completion_ratio'],0.)

    def test_missing_s0_is_explicit_not_fabricated(self):
        x=timeline([record(0,0,1_000_000_000)],0,2_000_000_000,2_000_000_000,{},2,'actual_submit_time_ns')
        self.assertEqual(x['peak_backlog_requests'],1)
        self.assertIsNone(x['peak_service_demand_backlog'])
        self.assertEqual(x['missing_service_paths'],[['text','m']])

    def test_fewshot_selection_reserves_whole_configurations_and_test(self):
        desc={c:dict(vector={'u':i/10},planning_safe_candidate=True,risk=i) for i,c in enumerate('abcd')}
        plan=selection_plan(desc,{'train0':dict(vector={'u':0.}),'train1':dict(vector={'u':1.})})
        self.assertEqual(plan['max_supported_K'],3)
        self.assertEqual(len(plan['trials']),3)
        for trial in plan['trials']:
            self.assertEqual(len(trial['calibration_chain']),3)
            self.assertEqual(len(trial['test_configurations']),1)
            self.assertFalse(set(trial['calibration_chain']) & set(trial['test_configurations']))
        single=selection_plan({'a':desc['a']},{'t':desc['d']})
        self.assertEqual(single['max_supported_K'],0)
        self.assertEqual(single['trials'],[])

    def test_bias_cannot_see_test_labels_and_rejected_probe_consumes_budget(self):
        rs=[dict(_id='a1',configuration_id='a',run_id='a1',_y=20.,_s0=10.,measurement_seconds=10),
            dict(_id='a2',configuration_id='a',run_id='a2',_y=40.,_s0=10.,measurement_seconds=10),
            dict(_id='b1',configuration_id='b',run_id='b1',_y=1e7,_s0=10.,measurement_seconds=10)]
        labels={r['run_id']:dict(stability_state_v2='steady') for r in rs}
        yh={r['_id']:0. for r in rs}
        a=bias_from_probes(rs,['a'],yh,labels)
        self.assertAlmostEqual(a['bias_log_ratio'],np.median(np.log([2.,4.])))
        rs[-1]['_y']=.001
        labels['b1']['stability_state_v2']='unstable'
        self.assertEqual(a,bias_from_probes(rs,['a'],yh,labels))
        labels['a2']['stability_state_v2']='recoverable_congested'
        rejected=bias_from_probes(rs,['a'],yh,labels)
        self.assertEqual(rejected['effective_configuration_count'],0)
        self.assertEqual(rejected['bias_log_ratio'],0.)
        self.assertEqual(rejected['measurement_run_ids'],['a1','a2'])
        self.assertEqual(rejected['rejected_configurations'],['a'])

    def test_acquisition_ignores_heldout_latency_and_stability_fields(self):
        row=json.loads((ROOT/'results/performance_model_active_sampling/dataset_rows.jsonl').read_text().splitlines()[0])
        ex=extractor()
        keys=sorted({k for p in ex.resource_profiles.values() for k in p['values']})
        predict=lambda vectors:np.array([v['u'] for v in vectors])
        before=planning_descriptor(row,ex,keys,predict)
        changed=copy.deepcopy(row)
        for k in ('labels','stability','stable','stability_state_v2','deployment_id','configuration_id','run_id'):
            changed[k]='deliberately corrupted evaluation-only field'
        self.assertEqual(before,planning_descriptor(changed,ex,keys,predict))


OUT=ROOT/'results/performance_model_local_calibration'


@unittest.skipUnless((OUT/'calibration_selection_audit.json').exists(),'Run offline evaluation first')
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit=json.loads((OUT/'calibration_selection_audit.json').read_text())
        cls.scores=json.loads((OUT/'calibration_fold_metrics.json').read_text())
        cls.predictions=[json.loads(l) for l in (OUT/'calibration_predictions.jsonl').read_text().splitlines()]
        cls.oldrows=[json.loads(l) for l in Path(cls.audit['input_path']).read_text().splitlines()]
        cls.labels={r['run_id']:r for r in (json.loads(l) for l in (OUT/'stability_v2_labels.jsonl').read_text().splitlines())}

    def test_old_labels_and_source_rows_are_untouched(self):
        new=[json.loads(l) for l in (OUT/'dataset_rows_v2.jsonl').read_text().splitlines()]
        self.assertEqual(len(new),len(self.oldrows))
        for old,joined in zip(self.oldrows,new):
            self.assertEqual(old,{k:joined[k] for k in old})
            self.assertEqual(old['stability'],self.labels[old['run_id']]['original_stability'])

    def test_all_global_and_local_groups_are_isolated(self):
        lookup={r['run_id']+'::'+r['task']+'::'+r['target_model']:r for r in self.oldrows}
        for fold in self.audit['folds']:
            dep=fold['heldout_deployment']
            allheld={r['run_id'] for r in self.oldrows if r['deployment_id']==dep}
            self.assertFalse(allheld & set(fold['global_training_run_ids']))
            self.assertNotIn(dep,fold['global_training_deployments'])
            self.assertFalse(set(fold['heldout_configurations']) & set(fold['global_training_configurations']))
            self.assertFalse(set(fold['heldout_configurations']) & set(fold['selection_plan']['distance_training_configuration_ids']))
            train=[lookup[i] for i in fold['global_training_row_ids']]
            for r in train:
                self.assertEqual(self.labels[r['run_id']]['sensitivity_labels'][fold['scenario']]['stability_state_v2'],'steady')
            us=np.array([r['standalone_service_demand'] for r in train])
            j=fold['preprocessing']['transformed_feature_order'].index('u')
            self.assertAlmostEqual(fold['preprocessing']['training_means'][j],us.mean())
            self.assertAlmostEqual(fold['preprocessing']['training_scales'][j],us.std())
            self.assertAlmostEqual(fold['preprocessing']['training_medians']['u'],np.median(us))
            for trial in fold['trials']:
                self.assertFalse(set(trial['calibration_configurations']) & set(trial['test_configurations']))
                self.assertFalse(set(trial['selected_run_ids']) & set(trial['test_run_ids']))
                allselected={r['run_id'] for r in self.oldrows if r['configuration_id'] in trial['calibration_configurations']}
                self.assertEqual(allselected,set(trial['selected_run_ids']))
                b=trial['correction']
                self.assertAlmostEqual(b['bias_log_ratio'],float(np.median(b['residuals'])) if b['residuals'] else 0.)

    def test_all_evaluation_targets_exclude_unstable_and_metrics_reproduce(self):
        lookup=defaultdict(list)
        for p in self.predictions:
            self.assertNotEqual(p['stability_state_v2'],'unstable')
            self.assertAlmostEqual(p['predicted_latency_ms'],p['standalone_latency_ms']*math.exp(p['global_predicted_log_ratio']+p['bias_log_ratio']))
            key=(p['scenario'],p['heldout_deployment'],p['trial_id'],p['K'],p['method'])
            lookup[key].append(p)
        for scenario,result in self.scores['scenarios'].items():
            for trial in result['paired_trial_metrics']:
                for method in ('global','bias_only'):
                    ps=lookup[scenario,trial['heldout_deployment'],trial['trial_id'],trial['K'],method]
                    ps=[p for p in ps if trial['subset']=='extended' or p['stability_state_v2']=='steady']
                    got=metrics([p['true_latency_ms'] for p in ps],[p['predicted_latency_ms'] for p in ps])
                    for key in ('mae_ms','median_absolute_error_ms','smape_percent','p90_absolute_error_ms','spearman'):
                        if got[key] is None:self.assertIsNone(trial[method][key])
                        else:self.assertAlmostEqual(got[key],trial[method][key])


if __name__=='__main__': unittest.main()
