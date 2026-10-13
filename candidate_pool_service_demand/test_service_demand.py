"""CPU checks for audit gates; no profiling or synthetic search campaigns."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest

from deployment_core.candidate_pool_v2 import load_candidate_pool_v2
from .prepare import OUT, ROOT, REPETITIONS, read, write, targets
from .finalize import aggregate_repetitions, completeness, validate_repetition
from .dry_run import load_completed_catalog


class ServiceDemandTests(unittest.TestCase):
    def test_manifest_is_exact_verified_missing_set(self):
        records, _ = load_candidate_pool_v2()
        pending = read(OUT / 'pending_service_demand_paths.json')
        expected = {(m, k) for m, p in records.items() for k, value in p.service_time_ms.items() if value is None}
        paths = pending['paths']
        self.assertEqual(len(paths), 14)
        self.assertEqual({(p['model_id'], p['task']) for p in paths}, expected)
        self.assertTrue(all(p['capability_status'] == 'verified' and p['existing_value'] is None for p in paths))
        self.assertFalse(any(p['task'] == 'visual_qa' for p in paths))

    def test_historical_subset_includes_full_duration_audio(self):
        expected = targets('asr')
        self.assertGreater(max(r['sample']['duration_s'] for r in expected), 6)
        for rep in REPETITIONS:
            path = ROOT / f'benchmark/results/placement/{rep}/active_single_qwen_asr_n30.jsonl'
            import json
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r['pool_id'] for r in rows], [r['pool_id'] for r in expected])

    def measurements(self):
        return [validate_repetition(OUT / f'raw_measurements/r{rep}/active_single_convnext_tiny_image_n30.jsonl',
                                    'convnext_tiny', 'image', targets('image')) for rep in range(1, 4)]

    def test_real_aggregation_uses_three_means_and_90_raw_samples(self):
        repetitions = self.measurements()
        aggregate = aggregate_repetitions(repetitions)
        self.assertEqual(aggregate['measurement_count'], 90)
        self.assertAlmostEqual(aggregate['aggregate_value_ms'], sum(r['mean_worker_latency_ms'] * 30 for r in repetitions) / 90)
        self.assertGreaterEqual(aggregate['sample_std_ms'], 0)
        self.assertEqual(aggregate['unit'], 'milliseconds')
        with self.assertRaises(ValueError):
            aggregate_repetitions(repetitions[:2])

    def test_missing_or_failed_repetition_cannot_be_aggregated(self):
        repetitions = self.measurements()
        repetitions[0]['status'] = 'MEASUREMENT_FAILED'
        with self.assertRaises(ValueError):
            aggregate_repetitions(repetitions)

    def mutated_copy(self, directory):
        original = OUT / 'raw_measurements/r1/active_single_convnext_tiny_image_n30.jsonl'
        target = Path(directory) / original.name
        for suffix in ('', '.summary.json', '.validation.json'):
            shutil.copyfile(str(original) + suffix, str(target) + suffix)
        return target

    def test_changed_sample_payload_rejected_even_when_ids_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.mutated_copy(directory)
            audit = read(str(path) + '.validation.json')
            audit['requests'][1]['sample']['label_id'] = -1
            write(str(path) + '.validation.json', audit)
            with self.assertRaisesRegex(ValueError, 'sample'):
                validate_repetition(path, 'convnext_tiny', 'image', targets('image'))

    def test_foreign_gpu_or_incomplete_unload_is_rejected(self):
        for field, value in [('foreign_GPU_processes', [[{'pid': -1}]]), ('GPU_memory_released', False)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = self.mutated_copy(directory)
                audit = read(str(path) + '.validation.json')
                audit[field] = value
                write(str(path) + '.validation.json', audit)
                with self.assertRaises(ValueError):
                    validate_repetition(path, 'convnext_tiny', 'image', targets('image'))

    def test_queue_latency_substitution_is_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = self.mutated_copy(directory)
            audit = read(str(path) + '.validation.json')
            raw = [json.loads(line) for line in path.read_text().splitlines()]
            raw[0]['worker_latency_ms'] += raw[0]['queue_wait_ms'] + 10
            response = next(e['payload'] for e in audit['events'] if e['kind'] == 'RESULT' and e['payload']['request_id'] == 0)
            response['worker_latency_ms'] = raw[0]['worker_latency_ms']
            path.write_text(''.join(json.dumps(row) + '\n' for row in raw))
            write(str(path) + '.validation.json', audit)
            with self.assertRaisesRegex(ValueError, 'timer'):
                validate_repetition(path, 'convnext_tiny', 'image', targets('image'))

    def test_completeness_requires_finite_positive_real_values(self):
        records, _ = load_candidate_pool_v2()
        services = {m: dict(p.service_time_ms) for m, p in records.items()}
        gate = completeness(records, services)
        self.assertEqual((gate['required_paths'], gate['available_paths'], gate['still_pending_paths']), (25, 11, 14))
        for bad in (0, float('inf'), float('nan'), True, -1):
            with self.subTest(bad=bad):
                services['minicpm']['text'] = bad
                self.assertEqual(completeness(records, services)['still_pending_paths'], 14)

    def test_completed_loader_preserves_old_values_memory_and_capability(self):
        catalog, hardware, roles, original, data = load_completed_catalog()
        self.assertEqual(len(catalog), 12)
        self.assertEqual(sum(p.memory_mib for p in catalog.values()), 63140)
        self.assertEqual(hardware.gpu_memory_mib, 49140)
        self.assertEqual(sum(role == 'generalist' for role in roles.values()), 3)
        for m, p in original.items():
            for task, old_value in p.service_time_ms.items():
                if old_value is not None:
                    self.assertEqual(catalog[m].service_time_ms[task], old_value)
        for mutate in ('old_value', 'memory', 'pending', 'capability'):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                changed = copy.deepcopy(data)
                if mutate == 'old_value':
                    changed['models']['qwen']['service_time_ms']['text'] += 1
                elif mutate == 'memory':
                    changed['models']['qwen']['measured_idle_vram_mib'] += 1
                elif mutate == 'pending':
                    changed['models']['minicpm']['service_time_ms']['text'] = None
                else:
                    changed['models']['phi4mm']['capabilities_verified'].append('visual_qa')
                path = Path(directory) / 'catalog.json'
                write(path, changed)
                with self.assertRaises(ValueError):
                    load_completed_catalog(path)


if __name__ == '__main__':
    unittest.main()
