"""Independent CPU adversarial checks, explicitly synthetic capture intake."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_kernel_dag_capture_alignment_v1 as fixtures
from tools.predictor_validation_20261027 import kernel_dag_capture_alignment_v1 as m


class IndependentAlignmentTests(unittest.TestCase):
    def captures(self, directory):
        root = Path(directory)
        refs = []
        for index in range(2):
            path = root/f'synthetic-raw-{index}.json'
            path.write_text(json.dumps(dict(CPU_TEST_ONLY=True, capture=index)))
            refs.append(m.receipt(path))
        result = []
        for index, ref in enumerate(refs):
            profile, items, binding = fixtures.CaptureAlignmentTests().fixture()
            profile.update(request_id=f'capture-{index}:request', alias='ocr', family='ocr', task='ocr', pool_id='exact')
            for node in profile['nodes']:
                if node['kind'] == 'gpu':
                    node['kernel']['source_sqlite'] = ref
            result.append(dict(raw=ref, context='unchanged', profiler='same-profiler',
                hardware=dict(hardware_id='same-gpu', sm_count=80),
                requests={'work': dict(profile=profile, items=items, binding=binding)}, sources=[ref]))
        design = root/'design.json'
        design.write_text(json.dumps(dict(schema=m.SCHEMA+'.design', training_only=True,
            validation_labels_read=False, captures=[dict(CPU_TEST_ONLY=True, index=i) for i in range(2)])))
        return design, result

    def test_full_hardware_context_profiler_and_complete_cohort_are_required(self):
        for mode in ('hardware', 'context', 'profiler', 'cohort', 'raw-alias'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                design, captures = self.captures(directory)
                if mode == 'hardware':
                    captures[1]['hardware']['sm_count'] += 1
                elif mode in ('context', 'profiler'):
                    captures[1][mode] += '-different'
                elif mode == 'cohort':
                    captures[1]['requests'].clear()
                else:
                    captures[1]['raw'] = captures[0]['raw']
                output = Path(directory)/'output'
                with patch.object(m, 'load_capture', side_effect=captures), self.assertRaises(ValueError):
                    m.align(design, output)
                self.assertFalse(output.exists())

    def test_one_mismatch_preserves_denominator_and_suppresses_compiler_export(self):
        with tempfile.TemporaryDirectory() as directory:
            design, captures = self.captures(directory)
            for capture in captures:
                capture['requests']['work2'] = copy.deepcopy(capture['requests']['work'])
                capture['requests']['work2']['binding']['pool_id'] = 'second'
            captures[1]['requests']['work2']['profile']['nodes'][0]['kernel']['allocated_registers_per_block'] += 256
            with patch.object(m, 'load_capture', side_effect=captures):
                report = m.align(design, Path(directory)/'output')
            self.assertEqual((report['offered_profiles'], report['matched_profiles'], report['rejected_profiles']), (2, 1, 1))
            self.assertEqual(report['status'], 'PARTIAL_OR_UNKNOWN_FULL_COHORT_RETAINED')
            self.assertFalse(report['compiler_input_exported'])
            self.assertFalse(report['numerical_qualification'])
            self.assertFalse(report['fit_started'])

    def test_compiler_export_retains_per_capture_native_clock_and_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            design, captures = self.captures(directory)
            with patch.object(m, 'load_capture', side_effect=captures):
                report = m.align(design, Path(directory)/'output')
            exported = m.checked(report['compiler_inputs']['compiler-captures.json'])
            catalog = m.checked(report['compiler_inputs']['compiler-catalog.json'])
            self.assertIs(catalog['validation_labels_read'], False)
            self.assertEqual(len(exported), 2)
            clocks = []
            for original, capture in zip(captures, exported):
                self.assertEqual(capture['raw_sha256'], original['raw']['sha256'])
                self.assertEqual(len(capture['kernels']), 2)
                for kernel in capture['kernels']:
                    self.assertEqual(kernel['native_evidence']['raw'], original['raw'])
                    self.assertEqual(kernel['clock_domain'], 'native-session-sqlite:'+original['raw']['sha256'])
                clocks.append(capture['kernels'][0]['clock_domain'])
            self.assertNotEqual(*clocks)
            self.assertFalse(report['independent_prediction_accuracy_evaluated'])

    def test_copy_work_and_unproved_boundary_assumptions_affect_equivalence(self):
        profile, items, binding = fixtures.CaptureAlignmentTests().fixture()
        profile['nodes'][1].update(kind='copy', memory_metadata=dict(bytes=4096, copyKind=1))
        original = m.template(profile, items, binding)[0]
        for mode in ('bytes', 'direction', 'external', 'default-stream'):
            changed = copy.deepcopy(profile)
            if mode == 'bytes':
                changed['nodes'][1]['memory_metadata']['bytes'] *= 2
            elif mode == 'direction':
                changed['nodes'][1]['memory_metadata']['copyKind'] += 1
            elif mode == 'external':
                changed['pre_request_external_dependencies_assumed_completed'] += 1
            else:
                changed['default_stream_semantics'] = 'different-assumption'
            with self.subTest(mode=mode):
                self.assertNotEqual(m.template(changed, items, binding)[0], original)


if __name__ == '__main__':
    unittest.main()
