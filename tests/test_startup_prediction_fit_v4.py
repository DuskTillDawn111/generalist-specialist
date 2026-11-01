import copy
import json
from pathlib import Path
import tempfile
import unittest
from tools.predictor_validation_20261027.startup_prediction_fit_v4 import targets,fit,receipt


def raw_fixture():
    names=['startup_begin','startup_end','priming_begin','priming_end','active_begin','active_end','shutdown_begin','shutdown_end']
    times=[1,3,3,5,5,7,7,8]
    samples=[]
    for stamp in range(9):
        peak=900 if stamp==7 else 800 if stamp==4 else 500
        samples.append(dict(time_ns=stamp*1_000_000_000,watts=100.,owned_used_mib=peak,
            processes_mib={'123':peak},owned_pid_aliases={'123':'text#0'},owned_aliases=['text#0'],foreign_pids=[]))
    return dict(schema='gsdeploy.startup_native_phase_collection.v4',kind='measured',backend='native',status='completed',
        errors=[],runtime_errors=[],telemetry_errors=[],board_samples=samples,
        lifecycle_boundaries=[dict(boundary=name,time_ns=stamp*1_000_000_000,clock='host_perf_counter_ns') for name,stamp in zip(names,times)])


class StartupTargetsTests(unittest.TestCase):
    def test_energy_excludes_priming_peak_includes_priming_excludes_active(self):
        result=targets(raw_fixture())
        self.assertEqual(result['energy_joules'],200.)
        self.assertEqual(result['peak_memory_mib'],800.)
        self.assertFalse(result['safety_factor_applied'])

    def test_missing_start_bracket_cannot_be_filled_by_extrapolation(self):
        raw=raw_fixture();raw['board_samples']=raw['board_samples'][2:]
        with self.assertRaisesRegex(ValueError,'bracket'):targets(raw)

    def test_invalid_ownership_or_source_kind_refused(self):
        changes=[('foreign_pids',[999]),('owned_used_mib',0),('owned_pid_aliases',{})]
        for key,value in changes:
            raw=raw_fixture();raw['board_samples'][1][key]=value
            with self.assertRaises(ValueError):targets(raw)
        raw=raw_fixture();raw['schema']='gsdeploy.native_phase_collection.v1'
        with self.assertRaises(ValueError):targets(raw)

    def test_missing_startup_end_and_nonmonotonic_boundaries_refused(self):
        raw=raw_fixture();raw['lifecycle_boundaries'].pop(1)
        with self.assertRaises(ValueError):targets(raw)


class StartupFitIntegrationTests(unittest.TestCase):
    def test_independent_runs_required_with_exact_context_and_priming(self):
        # Synthetic local files exercise receipt arithmetic; no GPU is used.
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);reviews=[]
            def write(name,value):
                path=root/name;path.write_text(json.dumps(value));return path
            context=write('context.json',dict(cpu_fixture=True))
            plan=write('plan.json',dict(split='training',jobs=[{'split':'training'}],execution_context_sha256='ctx'))
            for index in range(2):
                raw=raw_fixture();raw.update(execution_context_sha256='ctx',fixture_capture=index,
                    priming_request_records=[dict(model='text#0',task='text',pool_id='input',status='completed')])
                source=write('raw'+str(index)+'.json',raw)
                reviews.append(write('review'+str(index)+'.json',dict(schema='gsdeploy.host_startup_collection_strict_review.v4',
                    status='PASS',raw=receipt(source),plan=receipt(plan),context=receipt(context))))
            single=fit(reviews[:1],root/'single')
            self.assertEqual(single['status'],'UNKNOWN');self.assertIsNone(single['startup_prediction'])
            result=fit(reviews,root/'two')
            self.assertEqual(result['status'],'FITTED')
            self.assertEqual(result['startup_prediction'],dict(energy_joules=200.,peak_memory_mib=800.))
            with self.assertRaisesRegex(ValueError,'Duplicate raw'):fit([reviews[0],reviews[0]],root/'duplicate')
            changed=json.loads((root/'raw1.json').read_text())
            changed['priming_request_records'][0]['pool_id']='different-priming-input'
            (root/'raw1.json').write_text(json.dumps(changed))
            review=json.loads(reviews[1].read_text());review['raw']=receipt(root/'raw1.json')
            reviews[1].write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError,'priming configuration'):fit(reviews,root/'different-priming')
        raw=raw_fixture();raw['lifecycle_boundaries'][1]['time_ns']=0
        with self.assertRaises(ValueError):targets(raw)


if __name__=='__main__':unittest.main()
