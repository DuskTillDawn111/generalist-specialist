import copy
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gsdeploy.resources import identity
from tests.test_observed_horizon_risk_sidecar_v32 import risk_fixture
from tools.predictor_validation_20261027 import horizon_recovery_risk_v31 as risk
from tools.predictor_validation_20261027 import independent_risk_validation_v39 as validation
from tools.predictor_validation_20261027 import ready_risk_holdout_v38 as collector
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import receipt


def measured_interface_fixture():
    model,domain,plan,records,envelope=risk_fixture()
    domain=copy.deepcopy(domain);domain['kind']='measured';domain['actual_execution_qualification']='ALL_MEMBERS_PASS_EXACT_RUNTIME_PROJECTION'
    domain['risk_domain_sha256']=identity({k:v for k,v in domain.items() if k!='risk_domain_sha256'})
    model=copy.deepcopy(model);model['provenance']['kind']='measured';model['risk_domain_sha256']=domain['risk_domain_sha256']
    model['model_sha256']=identity({k:v for k,v in model.items() if k!='model_sha256'})
    envelope.update(kind='measured',split='independent_risk_validation')
    return model,domain,plan,records,envelope


class IndependentRiskValidationTests(unittest.TestCase):
    def test_metadata_rebind_preserves_every_numeric_and_provenance_field(self):
        model,domain,*_=measured_interface_fixture();expanded=copy.deepcopy(domain)
        expanded['member_execution_context_sha256'].append('predeclared-new-context')
        expanded['independent_target_extension']=dict(source_risk_domain_sha256=domain['risk_domain_sha256'])
        expanded['risk_domain_sha256']=identity({k:v for k,v in expanded.items() if k!='risk_domain_sha256'})
        before=copy.deepcopy(model);bound=collector.rebind_model(model,domain,expanded)
        self.assertEqual(model,before);self.assertEqual(collector.numeric_model(model),collector.numeric_model(bound))
        self.assertNotEqual(model['model_sha256'],bound['model_sha256'])
        self.assertEqual(bound['risk_domain_sha256'],expanded['risk_domain_sha256'])

    def test_actual_READY_hook_runs_after_native_start_and_before_control_returns(self):
        calls=[]
        class Runtime:
            def __init__(self):self.ready={'text#0':{'pid':7}};self.workers={'text#0':SimpleNamespace(process=SimpleNamespace(pid=7))}
            def start(self):calls.append('native-start');return 'started'
        with patch.object(collector.original,'DeploymentRuntime',Runtime),patch.object(collector.original.existing,'descendant_or_self',return_value=True),patch.object(collector,'qualify_ready',side_effect=lambda run,ready:calls.append('READY-only-qualification')):
            namespace,_=collector.execution_namespace('/declared/run');runtime=namespace['DeploymentRuntime']()
            self.assertEqual(runtime.start(),'started');calls.append('caller-can-prime-now')
        self.assertEqual(calls,['native-start','READY-only-qualification','caller-can-prime-now'])

    def test_READY_mismatch_fails_before_model_binding_or_any_service_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'domain.json';_,domain,*_=measured_interface_fixture()
            domain['actual_members']={'train':dict(projection=dict(ready={'text#0':dict(tasks=['text'],torch='exact')}))}
            domain['risk_domain_sha256']=identity({k:v for k,v in domain.items() if k!='risk_domain_sha256'});source.write_text(json.dumps(domain))
            plan=dict(execution_context_sha256='predeclared',source_risk_domain=receipt(source));(root/'plan.json').write_text(json.dumps(plan))
            with patch.object(collector,'audit',return_value={}):
                with self.assertRaisesRegex(ValueError,'no priming'):
                    collector.qualify_ready(root,{'text#0':dict(pid=7,tasks=['text'],torch='different')})
            report=json.loads((root/'readiness-only/qualification.json').read_text())
            self.assertEqual(report['status'],'UNKNOWN_ACTUAL_READY_DIFFERENCE_STOP_BEFORE_PRIMING')
            self.assertFalse((root/'readiness-only/expanded-risk-domain.json').exists())

    def test_independent_features_seal_then_later_labels_and_evaluation(self):
        model,domain,plan,records,envelope=measured_interface_fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);domain_path=root/'domain.json';domain_path.write_text(json.dumps(domain))
            bindings=[]
            for h in risk.HORIZONS_NS:
                candidate=copy.deepcopy(model);candidate['horizon_ns']=h;candidate['model_sha256']=identity({k:v for k,v in candidate.items() if k!='model_sha256'})
                path=root/('model-'+str(h)+'.json');path.write_text(json.dumps(candidate));bindings.append(dict(original=receipt(path),rebound=receipt(path)))
            q=root/'qualification.json';q.write_text(json.dumps(dict(status='ACTUAL_READY_EXACT_PASS_MODELS_BOUND_BEFORE_PRIMING',fit_called=False,numeric_model_parameters_changed=False,expanded_domain=receipt(domain_path),model_bindings=bindings)))
            feature_path=root/'prefix.jsonl.gz'
            with gzip.open(feature_path,'wt') as stream:stream.write(json.dumps(envelope)+'\n')
            raw_path=root/'raw.json';raw_path.write_text(json.dumps(dict(phase_start_ns=0,observation_end_ns=80,request_records=records)))
            plan_path=root/'plan.json';plan_path.write_text(json.dumps(plan))
            report=root/'features-report.json';report.write_text(json.dumps(dict(schema=validation.SCHEMA+'.features',status='INDEPENDENT_READER_PREFIXES_ONLY_NO_LABELS_EXPORTED',training_only=False,
                features=receipt(feature_path),readiness_qualification=receipt(q),raw_source=receipt(raw_path),plan=receipt(plan_path))))
            native_read=validation.read
            def forbidden_outcomes(path):
                if Path(path) in {raw_path,plan_path}:raise AssertionError('Scoring opened raw outcomes/plan')
                return native_read(path)
            with patch.object(validation,'read',side_effect=forbidden_outcomes):
                sealed=validation.seal(report,root/'sealed')
            self.assertEqual(sealed['n'],1);self.assertFalse(sealed['raw_service_queue_or_labels_opened'])
            self.assertFalse((root/'labels').exists())
            labels=validation.export_labels(root/'sealed/report.json',root/'labels')
            self.assertEqual(labels['n'],3)
            evaluation=validation.evaluate(root/'sealed/report.json',root/'labels/report.json',root/'evaluation')
            self.assertEqual(evaluation['n'],1);self.assertFalse(evaluation['model_refit'])
            self.assertFalse(evaluation['calibrated_probability']);self.assertFalse(evaluation['routing_gain_claimed'])

    def test_label_export_refuses_unsealed_or_training_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);report=root/'not-sealed.json';report.write_text(json.dumps(dict(schema='training-report')))
            with self.assertRaisesRegex(ValueError,'already be frozen'):
                validation.export_labels(report,root/'labels')
            self.assertFalse((root/'labels').exists())


if __name__=='__main__':unittest.main()
