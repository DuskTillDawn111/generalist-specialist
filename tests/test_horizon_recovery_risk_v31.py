import copy
import unittest

import numpy as np

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting import recovery_risk as legacy
from gsdeploy.resources import identity
from tests.test_recovery_risk_v2 import CONTEXT,HORIZON,training
from tests.test_whole57_visible_risk_adapter_v30 import fixture
from tools.predictor_validation_20261027.whole57_visible_risk_adapter_v30 import snapshot_at
from tools.predictor_validation_20261027.horizon_recovery_domain_v31 import (
    actual_runtime_projection,execution_projection,leaf_differences)
from tools.predictor_validation_20261027.horizon_recovery_risk_v31 import (
    SCHEMA,binary_target,candidate_fit,capture_holdout,legacy_stumps_kernel,
    score_candidate,training_row,vector,whole_capture_partition)

H=1_000_000_000


def target_fixture(*,waiting=2,followup=H,event=False,elapsed=None):
    prefix=dict(cutoff_ns=10,waiting_count=waiting)
    labels=dict(window_start_ns=10,horizon_ns=H,observed_until_ns=10+followup,
        recovery=dict(event_observed=event,elapsed_or_censor_ns=followup if elapsed is None else elapsed))
    return prefix,labels


def domain_fixture():
    domain=dict(schema=SCHEMA+'.risk_domain',kind='cpu_fixture',actual_execution_qualification='CPU_ONLY',
        member_execution_context_sha256=['ctx'],prepared_execution_projection_exact_equal=True,
        common_execution_projection=dict(context={'residents':{'text#0':'text'}},original_input_registry=[{'task':'text'}]))
    domain['risk_domain_sha256']=identity(domain)
    return domain


def candidate_rows():
    rows=[]
    for run in range(4):
        for i,value in enumerate((1.,2.,3.,4.,6.,7.,8.,9.)):
            rows.append(dict(id=f'run{run}:row{i}',run_id=f'run{run}',group_id='shared-content-recipe',
                original_execution_context_sha256='ctx',horizon_ns=H,kind='cpu_fixture',split='training',
                risk_domain_sha256=domain_fixture()['risk_domain_sha256'],
                feature_names=['visible_fixture_coordinate'],feature_values=[value],label=int(value>5)))
    return rows


class HorizonRecoveryTests(unittest.TestCase):
    def test_exact_recovery_censored_but_complete_H_is_known_positive(self):
        prefix,labels=target_fixture()
        target=binary_target(prefix,labels)
        self.assertEqual(target['label'],1)
        self.assertFalse(labels['recovery']['event_observed'])
        self.assertEqual(labels['recovery']['elapsed_or_censor_ns'],H)

    def test_administrative_censor_before_H_remains_unknown(self):
        prefix,labels=target_fixture(followup=H-1)
        self.assertIsNone(binary_target(prefix,labels)['label'])

    def test_observed_clearance_before_H_is_known_even_if_capture_stops_early(self):
        prefix,labels=target_fixture(followup=H//2,event=True,elapsed=H//3)
        self.assertEqual(binary_target(prefix,labels)['label'],0)
        prefix,labels=target_fixture(event=True,elapsed=H)
        self.assertEqual(binary_target(prefix,labels)['label'],0)

    def test_empty_current_queue_is_outside_conditional_risk_population(self):
        prefix,labels=target_fixture(waiting=0)
        self.assertEqual(binary_target(prefix,labels)['status'],'NOT_AT_RISK_CURRENT_BACKLOG_EMPTY')
        self.assertIsNone(binary_target(prefix,labels)['label'])

    def test_numerical_stumps_loop_is_exactly_legacy_CPU_equivalent(self):
        rows=training()
        old=legacy.fit_recovery_risk(rows,CONTEXT,horizon_ns=HORIZON,recovery_threshold_ms=50.)
        X=[];y=[]
        for row in rows:
            _,values,label=legacy._label(row,CONTEXT,'training',HORIZON,50.)
            X.append(values);y.append(label)
        actual=legacy_stumps_kernel(X,y)
        self.assertEqual(actual['base_score'],old['base_score'])
        self.assertEqual(actual['trees'],old['trees'])

    def test_same_family_whole_capture_holdout_preserves_groups_and_original_context(self):
        domain=domain_fixture();rows=candidate_rows();train,test=whole_capture_partition(rows,'run3')
        self.assertEqual({r['run_id'] for r in train},{'run0','run1','run2'})
        self.assertEqual({r['run_id'] for r in test},{'run3'})
        self.assertEqual({r['group_id'] for r in train+test},{'shared-content-recipe'})
        model=candidate_fit(train,domain,horizon_ns=H,n_estimators=2,allow_cpu_fixture=True)
        result=capture_holdout(model,test,domain,allow_cpu_fixture=True)
        self.assertEqual(result['scored_known_n'],8)
        self.assertEqual(result['shared_recipe_content_groups'],['shared-content-recipe'])
        self.assertFalse(result['input_generalization_claimed'])
        self.assertFalse(result['probability_calibrated'])
        self.assertEqual({r['original_execution_context_sha256'] for r in train+test},{'ctx'})

    def test_whole_run_overlap_and_unknown_targets_cannot_enter_as_negative(self):
        domain=domain_fixture();rows=candidate_rows();train,test=whole_capture_partition(rows,'run3')
        model=candidate_fit(train,domain,horizon_ns=H,n_estimators=2,allow_cpu_fixture=True)
        changed=copy.deepcopy(test);changed[0]['run_id']='run0'
        with self.assertRaises(ValueError):capture_holdout(model,changed,domain,allow_cpu_fixture=True)
        changed=copy.deepcopy(test);changed[0]['label']=None
        result=capture_holdout(model,changed,domain,allow_cpu_fixture=True)
        self.assertEqual(result['scored_known_n'],7)
        self.assertEqual(result['unknown_counts']['UNKNOWN_OR_NOT_AT_RISK_TARGET'],1)

    def test_future_labels_do_not_affect_scoring_or_public_vector(self):
        domain=domain_fixture();train,test=whole_capture_partition(candidate_rows(),'run3')
        model=candidate_fit(train,domain,horizon_ns=H,n_estimators=2,allow_cpu_fixture=True)
        row=copy.deepcopy(test[0]);before=score_candidate(model,row,domain,allow_cpu_fixture=True)
        row['label']=1-row['label']
        self.assertEqual(score_candidate(model,row,domain,allow_cpu_fixture=True),before)
        plan,records=fixture();prefix=snapshot_at(plan,records,0,'ctx',15)
        names,values=vector(prefix,['text#0'],['text'])
        self.assertNotIn('GPUcursor',names)
        prefix['isolated_future_labels']={'completion':999999999}
        other_names,other_values=vector(prefix,['text#0'],['text'])
        self.assertEqual(names,other_names);np.testing.assert_array_equal(values,other_values)

    def test_unqualified_measured_domain_cannot_be_bypassed_by_CPU_flag(self):
        domain=domain_fixture();domain['kind']='measured';domain['risk_domain_sha256']=identity({k:v for k,v in domain.items() if k!='risk_domain_sha256'})
        with self.assertRaisesRegex(UnknownEvidence,'NOT_YET_QUALIFIED'):
            candidate_fit(candidate_rows(),domain,horizon_ns=H,allow_cpu_fixture=True)

    def test_nan_scoring_coordinate_is_rejected_and_unseen_support_is_unknown(self):
        domain=domain_fixture();train,test=whole_capture_partition(candidate_rows(),'run3')
        model=candidate_fit(train,domain,horizon_ns=H,n_estimators=2,allow_cpu_fixture=True)
        row=copy.deepcopy(test[0]);row['feature_values']=[float('nan')]
        with self.assertRaises(ValueError):score_candidate(model,row,domain,allow_cpu_fixture=True)
        row['feature_values']=[1000.]
        self.assertEqual(score_candidate(model,row,domain,allow_cpu_fixture=True)['status'],'UNKNOWN_OUTSIDE_TRAINING_SUPPORT')

    def test_only_nonce_route_metadata_may_differ_in_execution_projection(self):
        context={'execution_nonce':'nonce-a','whole57_training_route_id':'balanced','execution':{'batch_size':1},'device':'gpu'}
        plan={'resource_configuration':{'mode':'shared'},'inputs':[{'task':'text'}],'recipe':{'rps':1}}
        other=copy.deepcopy(context);other.update(execution_nonce='nonce-b',whole57_training_route_id='skewed')
        self.assertEqual(execution_projection(context,plan),execution_projection(other,plan))
        other['execution']['batch_size']=2
        differences=leaf_differences(execution_projection(context,plan),execution_projection(other,plan))
        self.assertEqual(differences[0]['path'],'/context/execution/batch_size')

    def test_actual_runtime_torch_spec_changes_are_not_erased_with_PID(self):
        raw=dict(final_source_audit='PASS',errors=[],runtime_errors=[],domain='natural',
            request_records=[{'status':'completed'}]*57,priming_request_records=[{'status':'completed'}]*12,
            ready={'a':{'pid':1,'torch':'x','spec_sha256':'s'}})
        other=copy.deepcopy(raw);other['ready']['a']['pid']=2
        self.assertEqual(actual_runtime_projection(raw),actual_runtime_projection(other))
        other['ready']['a']['torch']='different'
        self.assertNotEqual(actual_runtime_projection(raw),actual_runtime_projection(other))

    def test_adapter_retains_original_context_and_binary_censor_proof(self):
        plan,records=fixture();prefix=snapshot_at(plan,records,0,'ctx',15)
        _,labels=target_fixture();labels['window_start_ns']=15;labels['observed_until_ns']=15+H
        envelope=dict(id='row',run_id='run',group_id='family',kind='cpu_fixture',split='training',execution_context_sha256='ctx',decision_time_ns=15)
        row=training_row(dict(**envelope,prefix_features=prefix),dict(**envelope,offline_labels=labels),domain_fixture())
        self.assertEqual(row['original_execution_context_sha256'],'ctx')
        self.assertEqual(row['label'],1)


if __name__=='__main__':unittest.main()
