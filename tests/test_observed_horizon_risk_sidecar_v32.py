import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.resources import identity
from tests.test_horizon_recovery_risk_v31 import domain_fixture,H
from tests.test_whole57_visible_risk_adapter_v30 import fixture
from tools.predictor_validation_20261027.whole57_visible_risk_adapter_v30 import snapshot_at
from tools.predictor_validation_20261027.horizon_recovery_risk_v31 import candidate_fit,vector
from tools.predictor_validation_20261027.observed_horizon_risk_sidecar_v32 import ObservedHorizonRisk,evaluate_sealed,static_source_closure,SCHEMA
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import receipt


def risk_fixture():
    plan,records=fixture();domain=domain_fixture();rows=[]
    for run in range(3):
        for index in range(8):
            cutoff=11+index;prefix=snapshot_at(plan,records,0,'ctx',cutoff)
            names,values=vector(prefix,['text#0'],['text'])
            rows.append(dict(id=f'run{run}:{index}',run_id=f'run{run}',group_id='shared-recipe',kind='cpu_fixture',split='training',
                original_execution_context_sha256='ctx',risk_domain_sha256=domain['risk_domain_sha256'],
                horizon_ns=H,feature_names=names,feature_values=values.tolist(),label=int(index>=4)))
    model=candidate_fit(rows,domain,horizon_ns=H,n_estimators=2,allow_cpu_fixture=True)
    envelope=dict(id='held:15',run_id='held',group_id='shared-recipe',split='training',kind='cpu_fixture',
        domain='cpu_fixture',route_id='fixture',execution_context_sha256='ctx',decision_time_ns=15,
        prefix_features=snapshot_at(plan,records,0,'ctx',15))
    return model,domain,plan,records,envelope


class ObservedHorizonRiskSidecarTests(unittest.TestCase):
    def test_real_observer_interface_keeps_late_START_waiting_and_hides_future_results(self):
        model,domain,plan,records,envelope=risk_fixture();provider=ObservedHorizonRisk(model,domain,allow_cpu_fixture=True)
        before=provider.predict_runtime_records(plan,records,phase_start_ns=0,context_sha256='ctx',cutoff_ns=15,run_id='held',group_id='shared-recipe',kind='cpu_fixture')
        self.assertEqual(before['visible_waiting_count'],2)
        self.assertIsNotNone(before['risk_score'])
        changed=copy.deepcopy(records);changed[0]['worker_response']['end_perf_ns']=1000;changed[0]['response_reader_event_enqueue_time_ns']=1100
        after=provider.predict_runtime_records(plan,changed,phase_start_ns=0,context_sha256='ctx',cutoff_ns=15,run_id='held',group_id='shared-recipe',kind='cpu_fixture')
        self.assertEqual(before,after)

    def test_simulated_queue_sidecar_or_labels_are_rejected(self):
        model,domain,plan,records,envelope=risk_fixture();provider=ObservedHorizonRisk(model,domain,allow_cpu_fixture=True)
        with self.assertRaises(ValueError):provider.predict_prefix({'schema':'gsdeploy.full_public_queue_diagnostics_sidecar.v19','diagnostics':{'max_waiting':100}},cutoff_ns=15)
        changed=copy.deepcopy(envelope);changed['offline_labels']={'recovery':1}
        with self.assertRaises(ValueError):provider.predict_prefix(changed,cutoff_ns=15)

    def test_frozen_model_never_refits_and_score_does_not_change_rho_or_utility(self):
        model,domain,plan,records,envelope=risk_fixture();original=copy.deepcopy(model);provider=ObservedHorizonRisk(model,domain,allow_cpu_fixture=True)
        result=provider.predict_prefix(envelope,cutoff_ns=15)
        self.assertEqual(model,original);self.assertEqual(provider.model,original)
        self.assertFalse(result['six_term_utility_changed']);self.assertFalse(result['rho_changed']);self.assertFalse(result['rolling_forecast_implemented'])

    def test_unbridged_original_context_is_unknown_and_cutoff_mismatch_rejected(self):
        model,domain,plan,records,envelope=risk_fixture();provider=ObservedHorizonRisk(model,domain,allow_cpu_fixture=True)
        changed=copy.deepcopy(envelope);changed['execution_context_sha256']='foreign';changed['prefix_features']=snapshot_at(plan,records,0,'foreign',15)
        self.assertEqual(provider.predict_prefix(changed,cutoff_ns=15)['status'],'UNKNOWN_EXECUTION_CONTEXT_OUTSIDE_EXPLICIT_RISK_DOMAIN')
        with self.assertRaises(ValueError):provider.predict_prefix(envelope,cutoff_ns=16)

    def test_current_empty_queue_has_no_fabricated_low_probability(self):
        model,domain,plan,records,envelope=risk_fixture();provider=ObservedHorizonRisk(model,domain,allow_cpu_fixture=True)
        changed=copy.deepcopy(envelope);changed['prefix_features']=snapshot_at(plan,records,0,'ctx',80);changed['decision_time_ns']=80
        result=provider.predict_prefix(changed,cutoff_ns=80)
        self.assertEqual(result['status'],'NOT_AT_RISK_CURRENT_BACKLOG_EMPTY');self.assertIsNone(result['risk_score'])

    def test_model_mutation_and_CPU_evidence_cannot_be_silently_accepted(self):
        model,domain,plan,records,envelope=risk_fixture()
        with self.assertRaises(ValueError):ObservedHorizonRisk(model,domain)
        provider=ObservedHorizonRisk(model,domain,allow_cpu_fixture=True);provider.model['base_score']+=1
        with self.assertRaises(ValueError):provider.predict_prefix(envelope,cutoff_ns=15)

    def test_separate_label_join_cannot_refit_or_accept_training_capture(self):
        model,domain,plan,records,envelope=risk_fixture();provider=ObservedHorizonRisk(model,domain,allow_cpu_fixture=True)
        prediction=provider.predict_prefix(envelope,cutoff_ns=15)
        labels=dict(id=envelope['id'],run_id='held',execution_context_sha256='ctx',decision_time_ns=15,offline_labels=dict(
            window_start_ns=15,horizon_ns=H,observed_until_ns=15+H,recovery=dict(event_observed=False,elapsed_or_censor_ns=H)))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model_path=root/'model.json';model_path.write_text(json.dumps(model))
            domain_path=root/'domain.json';domain_path.write_text(json.dumps(domain))
            feature_path=root/'features.jsonl.gz'
            with gzip.open(feature_path,'wt') as stream:stream.write(json.dumps(envelope)+'\n')
            pred_path=root/'pred.jsonl.gz';label_path=root/'labels.jsonl.gz'
            with gzip.open(pred_path,'wt') as stream:stream.write(json.dumps(prediction)+'\n')
            with gzip.open(label_path,'wt') as stream:stream.write(json.dumps(labels)+'\n')
            pred_report=root/'prediction-report.json';label_report=root/'label-report.json'
            label_report.write_text(json.dumps(dict(labels=receipt(label_path),features=receipt(feature_path))))
            closure=static_source_closure()
            bindings=dict(model=receipt(model_path),risk_domain=receipt(domain_path),visible_report=receipt(label_report),features=receipt(feature_path))
            sources=[*bindings.values(),*closure]
            def write_pred_report():
                pred_report.write_text(json.dumps(dict(schema=SCHEMA,status='SEALED_OBSERVER_RISK_OUTPUT_NO_LABELS_READ',future_labels_read=False,model_refit=False,predictions=receipt(pred_path),frozen_bindings=bindings,source_receipts_before=sources,source_receipts_after=sources,complete_static_local_source_closure=closure)))
            write_pred_report()
            label_report.write_text(json.dumps(dict(labels=receipt(label_path),features=receipt(feature_path))))
            original=pred_path.read_bytes()
            result=evaluate_sealed(pred_report,label_report,model_path,root/'evaluation')
            self.assertEqual(result['scored_n'],1);self.assertFalse(result['model_refit']);self.assertFalse(result['probability_calibrated'])
            self.assertEqual(pred_path.read_bytes(),original)
            original_prediction=copy.deepcopy(prediction)
            for index,(key,value) in enumerate([('risk_score',float('nan')),('risk_score',1.01),('risk_score',.321),('status','UNKNOWN'),('status','NOT_AT_RISK_CURRENT_BACKLOG_EMPTY'),('visible_waiting_count',0),('visible_state_sha256','forged'),('risk_domain_sha256','swapped-domain'),('decision_time_ns',16)]):
                changed=copy.deepcopy(original_prediction);changed[key]=value
                with gzip.open(pred_path,'wt') as stream:stream.write(json.dumps(changed)+'\n')
                write_pred_report()
                with self.assertRaises(ValueError):
                    evaluate_sealed(pred_report,label_report,model_path,root/('tamper-'+str(index)))
            swapped=copy.deepcopy(model);swapped['base_score']+=1;swapped['model_sha256']=identity({k:v for k,v in swapped.items() if k!='model_sha256'})
            swapped_path=root/'swapped-model.json';swapped_path.write_text(json.dumps(swapped))
            with self.assertRaisesRegex(ValueError,'frozen model identity'):
                evaluate_sealed(pred_report,label_report,swapped_path,root/'swapped-model-evaluation')
            prediction['run_id']='run0'
            with gzip.open(pred_path,'wt') as stream:stream.write(json.dumps(prediction)+'\n')
            write_pred_report()
            with self.assertRaisesRegex(ValueError,'overlaps model training'):
                evaluate_sealed(pred_report,label_report,model_path,root/'rejected-evaluation')

    def test_forged_host_hash_or_derived_prefix_is_rejected(self):
        model,domain,plan,records,envelope=risk_fixture();provider=ObservedHorizonRisk(model,domain,allow_cpu_fixture=True)
        changed=copy.deepcopy(envelope);changed['prefix_features']['visible_state_sha256']='forged'
        changed['prefix_features']['prefix_content_sha256']=identity({k:v for k,v in changed['prefix_features'].items() if k!='prefix_content_sha256'})
        with self.assertRaises(ValueError):provider.predict_prefix(changed,cutoff_ns=15)
        changed=copy.deepcopy(envelope);changed['prefix_features']['oldest_waiting_age_ns']+=1
        changed['prefix_features']['prefix_content_sha256']=identity({k:v for k,v in changed['prefix_features'].items() if k!='prefix_content_sha256'})
        with self.assertRaisesRegex(ValueError,'Derived visible prefix'):
            provider.predict_prefix(changed,cutoff_ns=15)


if __name__=='__main__':unittest.main()
