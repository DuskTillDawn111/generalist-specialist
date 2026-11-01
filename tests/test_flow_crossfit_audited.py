import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.predictor_validation_20261027.flow_crossfit_audited import audit_manifest
from tools.predictor_validation_20261027.native_flow import receipt


class OmittedIdentityAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        plan = {'seed': 1, 'trace': [{'request_id': 'r'}],
                'candidates': [{'id': 'a'}, {'id': 'b'}], 'baseline_candidate_id': 'a'}
        self.plan = self.write('plan.json', plan)
        jobs, folds = [], []
        self.verifications = {}
        protocol_path = self.root / 'protocol.json'
        for cid in ('a', 'b'):
            fid = 'seed-1__' + cid
            raw = self.write(cid + '-raw.json', {'plan': receipt(self.plan), 'candidate_id': cid})
            jobs.append({'flow_id': fid, 'seed': 1, 'raw_path': str(raw),
                         'plan': receipt(self.plan), 'candidate_id': cid})
        self.write('protocol.json', {'expected_flow_count': 2, 'flow_jobs': jobs})
        for job in jobs:
            cid = job['candidate_id']
            model = self.write(cid + '-model.json', {'training_evidence_tree': {'protocol': receipt(protocol_path)}})
            raw_ref = receipt(job['raw_path'])
            folds.append({'fold_id': job['flow_id'], 'model': receipt(model), 'heldout_raw': raw_ref,
                          'heldout_plan': receipt(self.plan), 'candidate_id': cid,
                          'heldout_trace_id': 'flow:' + raw_ref['sha256']})
            self.verifications[str(model)] = {'training_raw_receipts': []}
        self.manifest = self.write('manifest.json', {'schema': 'gsdeploy.continuous_flow_crossfit_manifest.v1',
            'protocol': receipt(protocol_path), 'folds': folds})
        self.mock = patch('tools.predictor_validation_20261027.flow_crossfit_audited.verify_training_model',
                          side_effect=lambda p: self.verifications[p])
        self.mock.start(); self.addCleanup(self.mock.stop)

    def write(self, name, value):
        p = self.root / name; p.write_text(json.dumps(value)); return p

    def mutate_manifest(self, change):
        value = json.loads(self.manifest.read_text()); change(value)
        self.write('manifest.json', value)

    def test_complete_exact_heldout_sources_pass(self):
        result = audit_manifest(self.manifest)
        self.assertEqual(result['contracts'][1]['baseline'], 'a')

    def test_missing_or_duplicate_fold_rejected(self):
        self.mutate_manifest(lambda d: d['folds'].pop())
        with self.assertRaisesRegex(ValueError, 'omits'):
            audit_manifest(self.manifest)

    def test_trace_identity_cannot_mask_trained_source(self):
        self.mutate_manifest(lambda d: d['folds'][0].update(heldout_trace_id='fake'))
        with self.assertRaisesRegex(ValueError, 'identity'):
            audit_manifest(self.manifest)

    def test_actual_heldout_raw_must_be_absent_from_model(self):
        d = json.loads(self.manifest.read_text())
        fold = d['folds'][0]
        self.verifications[fold['model']['path']]['training_raw_receipts'] = [fold['heldout_raw']]
        with self.assertRaisesRegex(ValueError, 'identity'):
            audit_manifest(self.manifest)


if __name__ == '__main__':
    unittest.main()
