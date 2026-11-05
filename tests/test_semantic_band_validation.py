"""Derived-band FIFO must retain energy timing and independent receipt gates."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.fit_activation_semantic_bands import fit_band_captures
from experiments.validate_activation_semantic_bands import score_band_fit
from gsdeploy.forecasting.native_training import digest
from gsdeploy.forecasting.native_energy import SCHEMA as ENERGY_SCHEMA, IDLE_EDGES_SECONDS
from tests.test_native_training import capture


def image_bundle(run,directory,*,heldout=False):
    bundle=capture(run,'solo')
    aliases={'image#0':'image'};bundle['context']['resident_aliases']=aliases
    identity=digest(bundle['context'])
    raw=bundle['raw'];plan=bundle['plan']
    raw['resident_aliases']=plan['aliases']=aliases
    raw['execution_context_sha256']=plan['execution_context_sha256']=identity
    rows=[row for row in raw['request_records'] if row['model']=='image#0']
    raw['request_records']=rows
    for key in ('ready','shutdown'):raw[key]={'image#0':raw[key]['image#0']}
    ids={row['request_id'] for row in rows}
    raw['batches']=[dict(batch,request_ids=[rid for rid in batch['request_ids'] if rid in ids])
                    for batch in raw['batches'] if any(rid in ids for rid in batch['request_ids'])]
    plan['jobs']=[dict(job,requests=[row for row in job['requests'] if row['alias']=='image#0'])
                  for job in plan['jobs'] if any(row['alias']=='image#0' for row in job['requests'])]
    for row in rows:
        span=row['worker_response']['observed_stages'][1]
        row['worker_response'].update(semantic_phase_schema='gsdeploy.semantic_native_observation.v1',
            semantic_phase_events=[dict(request_id=row['request_id'],component='image_features',role='image_features',
                host_start_ns=span['start_ns']+1,host_end_ns=span['end_ns']-1,
                device_event_complete=True,device_elapsed_ms=1.,stream_id=7,end_current_stream_id=7)])
    prime=copy.deepcopy(rows[0]);offset=prime['worker_response']['end_perf_ns']+1_000_000
    for event in prime['execution_events']:event['perf_ns']-=offset
    for span in prime['worker_response']['observed_stages']:
        span['start_ns']-=offset;span['end_ns']-=offset
    prime['worker_response']['start_perf_ns']-=offset;prime['worker_response']['end_perf_ns']-=offset
    raw['priming_request_records']=[prime]
    if heldout:raw['split']=plan['split']='development_validation'
    path=Path(directory)/(run+'.json');content=json.dumps(raw).encode();path.write_bytes(content)
    bundle['source']=str(path)+':sha256:'+hashlib.sha256(content).hexdigest()
    return bundle


class BandValidationTests(unittest.TestCase):
    def test_independent_fifo_passes_analytic_fixture_and_lifecycle_error_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            training=[image_bundle('s1',directory),image_bundle('s2',directory)]
            fit=fit_band_captures(training,[dict(task_type='image',pool_id='i')])
            # Explicit analytic constant-power fixture; the real fitter did
            # not qualify its insufficient idle data and remains rejected.
            self.assertFalse(fit['training_fit_gates_passed'])
            fit['profiles']['native_idle_age_energy']=dict(schema=ENERGY_SCHEMA,status='FITTED_UNVALIDATED',
                admission_enabled=False,execution_context_sha256=fit['profiles']['execution_context_sha256'],
                categories=[['image','image']],active_terms=[[]],coefficients=[50.]*5,
                idle_edges_seconds=IDLE_EDGES_SECONDS,maximum_observed_idle_age_seconds=30.,
                independent_run_ids=['s1','s2'],support_vertices=[[0.],[1.]])
            bundles=[image_bundle('v1',directory,heldout=True),image_bundle('v2',directory,heldout=True)]
            report=score_band_fit(fit,bundles,training_replay=False)
            self.assertTrue(report['development_accuracy_passed'],report['reasons'])
            self.assertFalse(report['formal_utility_qualified'])
            with self.assertRaisesRegex(ValueError,'reuse training'):
                score_band_fit(fit,training,training_replay=False)
            changed=copy.deepcopy(fit)
            for profile in changed['lifecycle']['profiles']:
                profile['scenarios']['mean']['parent_receipt_ms']=1000.
                profile['scenarios']['mean']['worker_next_ready_ms']=1000.
            failed=score_band_fit(changed,bundles,training_replay=False)
            self.assertFalse(failed['development_accuracy_passed'])
            self.assertIn('FIFO_E2E_ERROR_OR_UNKNOWN',failed['reasons'])
            for run in failed['runs']:
                self.assertAlmostEqual(run['energy']['relative_error'],0.,places=7)
                self.assertGreater(run['fifo_energy']['relative_error'],.15)


if __name__=='__main__':unittest.main()
