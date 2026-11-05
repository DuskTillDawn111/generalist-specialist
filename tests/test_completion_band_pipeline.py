"""CPU accounting tests; these cannot qualify the physical GPU predictor."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.activation_completion_provider import CompletionBandContextProvider
from experiments.fit_activation_completion_bands import fit_band_captures
from experiments.validate_activation_completion_bands import score_band_fit
from gsdeploy.forecasting.native_energy import SCHEMA as ENERGY_SCHEMA, IDLE_EDGES_SECONDS
from gsdeploy.forecasting.native_training import digest
from tests.test_semantic_band_validation import image_bundle


def ocr_bundle(run, directory, *, heldout=False):
    bundle=image_bundle(run,directory,heldout=heldout)
    # Rename the analytic singleton roster, including all lifecycle owners.
    bundle=json.loads(json.dumps(bundle).replace('image','ocr'))
    context=digest(bundle['context'])
    bundle['raw']['execution_context_sha256']=bundle['plan']['execution_context_sha256']=context
    for row in bundle['raw']['request_records']:
        span=row['worker_response']['observed_stages'][1]
        start=span['start_ns'];duration=span['end_ns']-start
        prototype=row['worker_response']['semantic_phase_events'][0]
        row['worker_response']['semantic_phase_events']=[
            dict(prototype,component='ocr_detection',role='ocr_detection',
                 host_start_ns=start+1,host_end_ns=start+duration//4),
            dict(prototype,component='ocr_recognition',role='ocr_recognition',
                 host_start_ns=start+duration*3//5,host_end_ns=span['end_ns']-1)]
    path=Path(directory)/(run+'.json');content=json.dumps(bundle['raw']).encode();path.write_bytes(content)
    bundle['source']=str(path)+':sha256:'+hashlib.sha256(content).hexdigest()
    return bundle


class CompletionPipelineTests(unittest.TestCase):
    def fitted(self, directory):
        training=[ocr_bundle('s1',directory),ocr_bundle('s2',directory)]
        original=copy.deepcopy(training)
        fit=fit_band_captures(training,[dict(task_type='ocr',pool_id='i')])
        self.assertEqual(training,original)
        self.assertFalse(fit['training_fit_gates_passed'])  # no identifiable idle design
        self.assertFalse(fit['independent_accuracy_qualified'])
        # Analytic constant-power fixture only, as in the original validator tests.
        fit['profiles']['native_idle_age_energy']=dict(schema=ENERGY_SCHEMA,status='FITTED_UNVALIDATED',
            admission_enabled=False,execution_context_sha256=fit['profiles']['execution_context_sha256'],
            categories=[['ocr','ocr']],active_terms=[[]],coefficients=[50.]*5,
            idle_edges_seconds=IDLE_EDGES_SECONDS,maximum_observed_idle_age_seconds=30.,
            independent_run_ids=['s1','s2'],support_vertices=[[0.],[1.]])
        return fit,training

    def test_completion_partition_replays_fifo_and_energy_without_reusing_heldout(self):
        with tempfile.TemporaryDirectory() as directory:
            fit,training=self.fitted(directory)
            segments=[s for s in fit['profiles']['profiles'][0]['segments'] if s['kind']=='gpu']
            self.assertEqual([s['duration_ms']['mean'] for s in segments],[6.,4.])
            independent=[ocr_bundle('v1',directory,heldout=True),ocr_bundle('v2',directory,heldout=True)]
            report=score_band_fit(fit,independent,training_replay=False)
            self.assertTrue(report['development_accuracy_passed'],report['reasons'])
            self.assertFalse(report['formal_utility_qualified'])
            with self.assertRaisesRegex(ValueError,'reuse training'):
                score_band_fit(fit,training,training_replay=False)
            with self.assertRaisesRegex(ValueError,'training-only'):
                fit_band_captures(independent,[dict(task_type='ocr',pool_id='i')])
            # Bad lifecycle must fail FIFO energy even when exposure energy is exact.
            bad=copy.deepcopy(fit)
            for p in bad['lifecycle']['profiles']:
                p['scenarios']['mean']['parent_receipt_ms']=1000.
                p['scenarios']['mean']['worker_next_ready_ms']=1000.
            failed=score_band_fit(bad,independent,training_replay=False)
            self.assertFalse(failed['development_accuracy_passed'])
            self.assertIn('FIFO_E2E_ERROR_OR_UNKNOWN',failed['reasons'])
            self.assertTrue(all(r['fifo_energy']['relative_error']>.15 for r in failed['runs']))

    def test_provider_rejects_wrong_policy_and_async_host_return_template(self):
        with tempfile.TemporaryDirectory() as directory:
            fit,_=self.fitted(directory);template=fit['profiles'];context=template['execution_context_sha256']
            CompletionBandContextProvider(template,context,allow_cpu_fixture=True)
            bad=copy.deepcopy(template);bad['derived_host_call_band_model'].pop('boundary_policy')
            with self.assertRaisesRegex(ValueError,'policy required'):
                CompletionBandContextProvider(bad,context,allow_cpu_fixture=True)
            bad=copy.deepcopy(template)
            next(s for s in bad['profiles'][0]['segments'] if s['kind']=='gpu')['native_band_role']='detection'
            with self.assertRaisesRegex(ValueError,'OCR template differs'):
                CompletionBandContextProvider(bad,context,allow_cpu_fixture=True)


if __name__=='__main__':unittest.main()
