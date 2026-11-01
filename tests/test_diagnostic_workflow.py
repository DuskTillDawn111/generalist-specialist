"""CPU-only cohort integrity, single-factor checks and early quality barriers."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.backends.compact import CompactAdapter
from gsdeploy.profiling.compact_experiments import calibration_quality_screen, execute_job
from gsdeploy.profiling.diagnostics import validate_contrasts
from gsdeploy.profiling.prepare import digest
from gsdeploy.profiling.runner import save
from tests.test_compact_protocol import fixtures

EOS = {'finish_reason': 'eos', 'generated_tokens': 3, 'terminal_token_id': 2, 'eos_token_ids': [2]}


class DiagnosticWorkflowTests(unittest.TestCase):
    def fixture(self, root, bad_answer=False):
        adapter, study, samples = fixtures()
        candidate = study['specialists'][0]
        no_answer = {**samples['text'], 'id': 'unanswerable', 'answers': []}
        cohort = [samples['text'], no_answer]
        protocol, classes, spec_path = root / 'protocol.json', root / 'classes.json', root / 'worker.json'
        save(protocol, adapter.policy)
        save(classes, adapter.classmap)
        spec = {'candidate': candidate, 'execution': {'dtype': 'bfloat16'},
                'resources': {'protocol': str(protocol), 'classmap': str(classes)}}
        save(spec_path, spec)
        job = {'slot': 'text', 'samples': cohort}
        job_path = root / 'memory.json'
        save(job_path, job)
        manifest = {'workers': {'text': {'spec': str(spec_path), 'spec_sha256': digest(spec_path), 'tasks': ['text']}},
                    'jobs': [{'id': 'memory-text-n1', 'phase': 'memory', 'path': str(job_path), 'sha256': digest(job_path)}]}
        manifest_path = root / 'manifest.json'
        save(manifest_path, manifest)
        output = root / 'results/memory-text-n1'
        output.mkdir(parents=True)
        raw = output / 'text-round0.jsonl'
        rows = [adapter.finish(adapter.build(candidate, s), s,
                               'Paris' if s['answers'] or bad_answer else '<No Answer>', evaluate_quality=False, **EOS)
                for s in cohort]
        raw.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        save(output / 'profile.json', {'status': 'COMPLETE', 'model': spec,
                                     'rounds': [{'task': 'text', 'raw_records': str(raw)}]})
        save(output / 'receipt.json', {'status': 'COMPLETE', 'manifest_sha256': digest(manifest_path),
                                     'job_sha256': digest(job_path)})
        return manifest_path, manifest, raw

    def test_full_calibration_quality_reconstructed_from_raw_outputs(self):
        for bad in [False, True]:
            with self.subTest(bad_answer=bad), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path, manifest, raw = self.fixture(root, bad)
                result = calibration_quality_screen(path, manifest, root / 'results')
                self.assertEqual(result['passed'], not bad)
                self.assertFalse(result['pool_admitted'])
                self.assertEqual(result['gates']['text:text']['summary']['unanswerable']['EM'], 0 if bad else 1)

    def test_partial_wrong_receipt_or_changed_request_cannot_pass(self):
        for change in ['partial', 'receipt', 'request']:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path, manifest, raw = self.fixture(root)
                rows = [json.loads(l) for l in raw.read_text().splitlines()]
                if change == 'partial':
                    raw.write_text(json.dumps(rows[0]) + '\n')
                elif change == 'request':
                    rows[0]['request_sha256'] = 'wrong'
                    raw.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                else:
                    receipt = raw.with_name('receipt.json')
                    d = json.loads(receipt.read_text())
                    d['manifest_sha256'] = 'other'
                    save(receipt, d)
                    self.assertFalse(calibration_quality_screen(path, manifest, root / 'results')['passed'])
                    continue
                with self.assertRaises(ValueError):
                    calibration_quality_screen(path, manifest, root / 'results')

    def test_diagnostic_protocol_cannot_run_formal_job(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, manifest, raw = self.fixture(root)
            p = root / 'protocol.json'
            d = json.loads(p.read_text())
            d['diagnostic_only'] = True
            save(p, d)
            manifest.update(sources={}, code_hashes={})
            job_path = root / 'quality.json'
            save(job_path, {'phase': 'quality', 'requires': []})
            manifest['jobs'].append({'id': 'quality', 'phase': 'quality', 'path': str(job_path), 'sha256': digest(job_path)})
            save(path, manifest)
            with self.assertRaisesRegex(ValueError, 'Diagnostic protocols'):
                execute_job(path, 'quality', root / 'results/quality')
            self.assertFalse((root / 'results/quality').exists())

    def test_contrasts_reject_multiple_changes_different_inputs_and_broad_factors(self):
        adapter, study, samples = fixtures()
        policy = copy.deepcopy(adapter.policy)
        policy['diagnostic_only'] = True
        parent = {'id': 'native', 'parent': None, 'factor': None,
                  'candidate': study['generalists'][1], 'policy': policy,
                  'execution': {'dtype': 'float16', 'memory_optimizations': {}}, 'samples': [samples['asr']]}
        child = copy.deepcopy(parent)
        child.update(id='penalty', parent='native', factor=['policy', 'tasks', 'asr', 'generation_controls', 'repetition_penalty'])
        child['policy']['tasks']['asr']['generation_controls'] = {'repetition_penalty': 1.1}
        validate_contrasts([parent, child])
        for mutation in ['extra_setting', 'different_input', 'broad']:
            invalid = copy.deepcopy(child)
            if mutation == 'extra_setting':
                invalid['execution']['dtype'] = 'bfloat16'
            elif mutation == 'different_input':
                invalid['samples'][0]['id'] = 'another'
            else:
                invalid['factor'] = ['policy', 'tasks']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_contrasts([parent, invalid])


if __name__ == '__main__':
    unittest.main()
