import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_startup_prediction_fit_v4 import raw_fixture
from tools.predictor_validation_20261027 import startup_prediction_v5 as tool


class PremeasurementEnergyTests(unittest.TestCase):
    def test_energy_includes_priming_without_active_or_drain(self):
        result = tool.targets(raw_fixture())
        self.assertEqual(result['energy_joules'], 400.)
        self.assertEqual(result['loading_only_energy_joules'], 200.)
        self.assertEqual(result['peak_memory_mib'], 800.)
        self.assertTrue(result['energy_includes_priming'])
        self.assertFalse(result['safety_factor_applied'])

    def test_missing_start_bracket_and_foreign_process_are_rejected(self):
        for field in ('bracket', 'foreign'):
            raw = raw_fixture()
            if field == 'bracket':
                raw['board_samples'] = raw['board_samples'][2:]
            else:
                raw['board_samples'][4]['foreign_pids'] = [999]
            with self.assertRaises(ValueError):
                tool.targets(raw)

    def test_planned_priming_matches_longest_inputs_and_all_g2_tasks(self):
        inputs = [dict(task=task, input_size=size, pool_id=task+str(size))
                  for task in ('text','image','asr','ocr') for size in (1,3,2,4)]
        residents = dict(a='text', b='generalist_2', c='ocr')
        rows = tool.planned_priming(inputs, residents)
        self.assertEqual(len(rows), 15)
        self.assertEqual([r['pool_id'] for r in rows[:3]], ['text4','text3','text2'])
        self.assertEqual({r['task'] for r in rows if r['model']=='b'}, {'text','image','asr'})

    def test_fit_and_transfer_keep_same_native_priming_and_all_energy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def write(name, doc):
                path = root/name
                path.write_text(json.dumps(doc))
                return tool.receipt(path)
            context = dict(residents={'text#0':'text'})
            context_ref = write('context.json', context)
            inputs = [dict(task='text', pool_id='p', input_size=1)]
            plan = dict(split='training', inputs=inputs, jobs=[dict(split='training')])
            plan_ref = write('plan.json', plan)
            documents, paths = {}, []
            for i in range(2):
                raw = raw_fixture()
                raw.update(capture=i, execution_context_sha256='ctx',
                    request_records=[dict(split='training', status='completed')],
                    priming_request_records=[dict(model='text#0', task='text', pool_id='p', status='completed')])
                raw_ref = write(f'raw{i}.json', raw)
                report = dict(raw=raw_ref, plan=plan_ref, context=context_ref)
                path = write(f'review{i}.json', report)['path']
                documents[path] = report, plan, raw
                paths.append(path)
            native = dict(projection_sha256='native', original_full_context_sha256='ctx', measurement_domain='natural')
            with patch.object(tool, 'load_report', side_effect=lambda p,d:documents[p]), patch.object(
                    tool, 'native_execution_projection', return_value=native):
                model = tool.fit(paths, root/'fit')
                self.assertEqual(model['startup_prediction'], dict(energy_joules=400., peak_memory_mib=800.))
                result = tool.prediction(root/'fit/model.json', context_ref['path'], inputs,
                                         allow_native_projection_transfer=True)
                self.assertEqual(result['energy_joules'], 400.)
                with self.assertRaisesRegex(ValueError, 'target priming differs'):
                    tool.prediction(root/'fit/model.json', context_ref['path'],
                        [dict(inputs[0], pool_id='other')], allow_native_projection_transfer=True)
                with self.assertRaisesRegex(ValueError, 'Repeated raw'):
                    tool.fit([paths[0], paths[0]], root/'duplicate')
                with patch.object(tool, 'native_execution_projection', return_value=dict(native, projection_sha256='other')):
                    with self.assertRaisesRegex(ValueError, 'native execution differs'):
                        tool.prediction(root/'fit/model.json', context_ref['path'], inputs,
                                        allow_native_projection_transfer=True)
                Path(documents[paths[0]][0]['raw']['path']).write_text('{}')
                with self.assertRaisesRegex(ValueError, 'Changed receipt'):
                    tool.prediction(root/'fit/model.json', context_ref['path'], inputs,
                                    allow_native_projection_transfer=True)


if __name__ == '__main__':
    unittest.main()
