import unittest

from .freeze import evaluate


def fixture():
    models, measured, compatibility = [], {}, {}
    for index in range(10):
        name = f'm{index}'
        role = 'generalist' if index < 2 else 'specialist'
        tasks = ['text', 'image', 'ocr'] if role == 'generalist' else [['text'], ['image'], ['ocr']][(index - 2) % 3]
        models.append({'id': name, 'role': role, 'capabilities_verified': tasks, 'adapter_status': 'verified',
                       'measured_idle_vram_mib': 5000., 'weight_size_gb': .001})
        measured[name] = {'status': 'passed', 'gpu_memory_released': True}
        compatibility[name] = {task: {'status': 'verified'} for task in tasks}
    return models, measured, compatibility


class FreezeTests(unittest.TestCase):
    def test_strict_capacity_boundary_and_weights_never_used_as_vram(self):
        models, measured, compatibility = fixture()
        models[0]['measured_idle_vram_mib'] = 4140.
        models[0]['weight_size_gb'] = 10000.
        result = evaluate(models, measured, compatibility, True)
        self.assertEqual(result['pool_idle_vram_mib'], 49140.)
        self.assertFalse(result['frozen'])
        models[0]['measured_idle_vram_mib'] += 1
        self.assertTrue(evaluate(models, measured, compatibility, True)['frozen'])

    def test_missing_memory_and_provenance_prevent_freeze(self):
        models, measured, compatibility = fixture()
        models[0]['measured_idle_vram_mib'] = None
        result = evaluate(models, measured, compatibility, True)
        self.assertIsNone(result['pool_idle_vram_mib'])
        self.assertFalse(result['frozen'])
        models[0]['measured_idle_vram_mib'] = 5000.
        self.assertFalse(evaluate(models, measured, compatibility, False)['frozen'])

    def test_pending_and_unreleased_candidates_do_not_count(self):
        models, measured, compatibility = fixture()
        models[0]['adapter_status'] = 'pending'
        measured['m1']['gpu_memory_released'] = False
        result = evaluate(models, measured, compatibility, True)
        self.assertEqual(result['verified_generalists'], 0)
        self.assertEqual(result['total_model_types'], 8)
        self.assertFalse(result['frozen'])

    def test_matrix_must_agree_with_capability_flags(self):
        models, measured, compatibility = fixture()
        compatibility['m0']['text']['status'] = 'pending'
        self.assertFalse(evaluate(models, measured, compatibility, True)['frozen'])


if __name__ == '__main__':
    unittest.main()
