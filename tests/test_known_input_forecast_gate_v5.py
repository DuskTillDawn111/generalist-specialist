import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_forecast_gate_v3 import fixtures
from tools.predictor_validation_20261027.known_input_gate_v5 import DATA_SCHEMA,freeze_gate_protocol
from tools.predictor_validation_20261027.known_input_forecast_gate_v5 import run_and_seal
from tools.predictor_validation_20261027.forecast_gate_v2 import INPUT_SCHEMA
from tools.predictor_validation_20261027.forecast_gate_v3 import evidence_closure


class KnownInputForecastProducerTests(unittest.TestCase):
    def test_new_scope_executes_and_seals_real_model_and_original_scorer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bundle,public,save,index=fixtures(root)
            known=save('known-training.json',dict(schema=DATA_SCHEMA,split='training',records=[
                dict(run_id='known-training',request_id='old-request',pool_id='input',task='text')]))
            bundle['training_indices'].append(known)
            catalog=root/'public-inputs.jsonl';catalog.write_text(json.dumps(dict(pool_id='input',task_type='text',
                sample=dict(context='public content',question='public question'))))
            model=save('bundle.json',bundle);public_ref=save('public.json',dict(schema=INPUT_SCHEMA,forecast_inputs=public))
            source=Path(__file__).resolve();closure=evidence_closure([model['path']])
            paths=list({r['path'] for r in closure}|{public_ref['path'],str(source),str(catalog),known['path']})
            protocol=freeze_gate_protocol(model_paths={'model':model['path']},source_paths=paths,
                training_data_paths=[index['path'],known['path']],public_input_catalog_path=catalog,
                candidates=[dict(id=c,model_id='model',execution_context_sha256='natural') for c in ('base','candidate')],
                baseline_candidate_id='base',validation_domain=[dict(run_id='new-co-run',requests=[
                    dict(request_id=a,pool_id='input',task='text') for a in ('a','b')])],
                thresholds=dict(service_mare_max=.2,p95_mare_max=.2,queue_depth_mae_max=1.),created_unix_ns=1,
                weights=bundle['weights'],queue_sample_offsets_ns=[0,50_000_000])
            protocol_ref=save('protocol.json',protocol)
            jobs=[dict(run_id='new-co-run',candidate_id=c,public_inputs_path=public_ref['path']) for c in ('base','candidate')]
            with patch('tools.predictor_validation_20261027.known_input_forecast_gate_v5.prediction_source_paths',return_value=[source]):
                result=run_and_seal(protocol_ref['path'],jobs,root/'output',allow_cpu_fixture=True)
            sealed=json.loads(Path(result['predictions']['path']).read_text())
            self.assertEqual(sealed['schema'],'gsdeploy.known_input_prediction_gate_seal.v5')
            self.assertEqual(len(sealed['cells']),2)
            self.assertTrue(all(len(c['requests'])==2 for c in sealed['cells']))
            self.assertFalse(result['unseen_input_generalization_claimed'])


if __name__=='__main__':unittest.main()
