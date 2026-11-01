"""Trace conversion checks use synthetic CPU fixtures; no GPU is touched."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.forecasting.profiles import ProfileProvider, context_key, fit_progress
from tools.predictor_validation_20261027.profiling_analysis import (RESIDENTS, TRACE_SCHEMA,
    audit_trace, convert_training, main, make_plan, digest, map_native_collection, fit_native_collection)


class ProfilingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        source = Path(self.tmp.name) / 'synthetic-trace.txt'
        source.write_text('CPU test fixture, not physical evidence')
        self.receipt = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}

    def trace(self, index=0, *, pair=False):
        requests = []
        for alias, family, profile in [('text#0', 'text', 's')] + ([('generalist_2#0', 'generalist_2', 'g')] if pair else []):
            requests.append({'request_id': alias, 'alias': alias, 'family': family, 'task': 'text',
                'pool_id': 'x', 'profile_id': profile,
                'execution': {'batch_size': 1, 'backend': 'test-native-contract', 'weight_reuse': False},
                'handler_start_ns': 1_000_000, 'handler_end_ns': 5_000_000,
                'start_basis': 'observed_worker_START',
                'segments': [{'id': 'gpu', 'kind': 'gpu', 'start_ns': 1_000_000, 'end_ns': 5_000_000,
                    'observation_level': 'phase', 'observation_basis': 'test fixture markers',
                    'demand': {'sm': .6, 'dram': .2, 'l2': .1}, 'demand_evidence': 'test fixture normalization'}]})
        return {'schema': TRACE_SCHEMA, 'kind': 'measured', 'backend': 'native',
                'id': str(index), 'split': 'training', 'clock': 'host_monotonic_ns',
                'execution_context_sha256': 'fixture-only', 'sources': [self.receipt],
                'observation_window_ns': {'start': 0, 'end': 6_000_000},
                'requests': requests, 'admission_enabled': False,
                'power_samples': [{'time_ns': t, 'watts': 100} for t in range(0,7_000_000,1_000_000)],
                'memory_samples': [{'time_ns': t, 'owned_used_mib': 500,
                    'owned_aliases': list(RESIDENTS), 'foreign_pids': []} for t in range(0,7_000_000,1_000_000)]}

    def test_plan_covers_contexts_and_strict_input_holdout(self):
        inputs = [{'task': task, 'pool_id': task+str(i), 'input_size': i+1}
                  for task in ('text', 'ocr') for i in range(8)]
        plan = make_plan(inputs, 'physical-context')
        self.assertEqual(plan, make_plan(list(reversed(inputs)), 'physical-context'))
        self.assertEqual(set(j['context'] for j in plan['jobs']), {'S0','S1','G','O','SS','SG','SSG','SSGO'})
        heldout = {(r['task'], r['pool_id']) for r in plan['input_assignments'] if r['view'] == 'unseen_input'}
        for job in plan['jobs']:
            if job['split'] == 'training':
                self.assertTrue(all(('ocr' if a == 'ocr#0' else 'text', p) not in heldout for a,p in job['inputs'].items()))
        self.assertTrue(any(j['split'] == 'validation' and j['validation_view'] == 'independent_trial' for j in plan['jobs']))
        self.assertTrue(any(max(j['start_offsets_ms'].values()) == 600 for j in plan['jobs']))

    def test_answers_not_allowed_in_plan(self):
        with self.assertRaises(ValueError):
            make_plan([{'task':'text', 'pool_id':'x', 'input_size':3, 'answer':'secret'}], 'x')

    def test_raw_handler_is_unknown(self):
        trace = self.trace()
        trace['requests'][0].pop('segments')
        self.assertFalse(audit_trace(trace)['known'])
        with self.assertRaisesRegex(ValueError, 'UNKNOWN'):
            convert_training([trace])

    def test_submission_is_not_start(self):
        trace = self.trace(); trace['requests'][0]['start_basis'] = 'actual_submit_time_ns'
        self.assertFalse(audit_trace(trace)['known'])

    def test_segments_must_partition_without_overlap_or_gaps(self):
        for start in (0, 2_000_000):
            trace = self.trace(); trace['requests'][0]['segments'][0]['start_ns'] = start
            self.assertFalse(audit_trace(trace)['known'])

    def test_power_and_owned_residency_coverage(self):
        trace = self.trace(); trace['memory_samples'][1]['owned_aliases'].pop()
        self.assertFalse(audit_trace(trace)['known'])
        trace = self.trace(); trace['power_samples'] = trace['power_samples'][1:]
        self.assertFalse(audit_trace(trace)['known'])
        trace = self.trace(); trace['memory_samples'][1]['foreign_pids'] = [123]
        self.assertFalse(audit_trace(trace)['known'])

    def test_counter_and_receipt_required(self):
        trace = self.trace(); trace['requests'][0]['segments'][0].pop('demand_evidence')
        self.assertFalse(audit_trace(trace)['known'])
        trace = self.trace(); Path(self.receipt['path']).write_text('changed')
        self.assertFalse(audit_trace(trace)['known'])

    def test_admission_wait_required_when_enabled(self):
        trace = self.trace(); trace['admission_enabled'] = True
        self.assertFalse(audit_trace(trace)['known'])
        row = trace['requests'][0]; row.update(admission_wait_start_ns=0, admission_granted_ns=500_000)
        self.assertTrue(audit_trace(trace)['known'])

    def test_converter_fit_and_idle_context(self):
        result = convert_training([self.trace(0), self.trace(1)])
        self.assertEqual(len(result['observations']), 2)
        self.assertIn(context_key({}), result['document']['contexts'])
        fitted = fit_progress(result['document'], result['observations'])
        # Explicitly downgrade synthetic fixtures before provider usage.
        fitted['provenance']['kind'] = 'cpu_fixture'
        provider = ProfileProvider(fitted, 'fixture-only', allow_cpu_fixture=True)
        context = context_key({'text#0': {'profile_id':'s','segment_id':'gpu'}})
        self.assertAlmostEqual(provider.document['contexts'][context]['scenarios']['mean']['progress']['text#0'], 1)
        self.assertEqual(provider.document['contexts'][context]['scenarios']['mean']['board_watts'], 100)

    def test_validation_is_never_fitted(self):
        trace = self.trace(); trace['split'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'Training traces only'):
            convert_training([trace])

    def test_profile_identity_must_match_across_runs(self):
        one, two = self.trace(1), self.trace(2)
        two['requests'][0]['execution']['backend'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Profile identity changed'):
            convert_training([one, two])

    def test_solo_not_inferred_from_pair(self):
        with self.assertRaisesRegex(ValueError, 'UNKNOWN_SOLO_BASELINE'):
            convert_training([self.trace(0, pair=True), self.trace(1, pair=True)])

    def test_missing_interval_memory_is_unknown(self):
        a,b = self.trace(0), self.trace(1)
        for trace in (a,b):
            trace['memory_samples'] = [trace['memory_samples'][0], trace['memory_samples'][-1]]
        result = convert_training([a,b])
        key = context_key({'text#0': {'profile_id':'s','segment_id':'gpu'}})
        self.assertIsNone(result['document']['contexts'][key]['scenarios']['mean']['peak_memory_mib'])

    def test_counter_unavailable_can_fit_progress_but_not_classification(self):
        traces = [self.trace(0), self.trace(1)]
        for trace in traces:
            segment = trace['requests'][0]['segments'][0]
            segment['demand'] = {'sm': None, 'dram': None, 'l2': None}
            segment['demand_evidence'] = 'UNKNOWN:ERR_NVGPUCTRPERM'
            self.assertTrue(audit_trace(trace)['known'])
        converted = convert_training(traces)
        self.assertIsNone(converted['document']['profiles'][0]['segments'][0]['demand']['sm'])
        fitted = fit_progress(converted['document'], converted['observations'])
        fitted['provenance']['kind'] = 'cpu_fixture'
        provider = ProfileProvider(fitted, 'fixture-only', allow_cpu_fixture=True)
        self.assertEqual(provider.demand_summary('s')['classification'], 'unknown')

    def test_missing_counter_reason_is_not_implicit_zero(self):
        trace = self.trace(); trace['requests'][0]['segments'][0]['demand']['sm'] = None
        self.assertFalse(audit_trace(trace)['known'])

    def native_raw(self):
        root = Path(self.tmp.name)
        context = {'execution': {'batch_size':1, 'backend':'fixture-native-contract', 'weight_reuse':False}}
        ctx=root/'context.json'; ctx.write_text(json.dumps(context))
        plan=root/'plan.json'
        receipt=lambda p: {'path':str(p), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        rows,batches,samples=[],[],[]
        for index in range(3):
            offset=index*10_000_000; rid='r'+str(index)
            rows.append({'request_id':rid,'model':'text#0','task':'text','pool_id':'x','status':'completed',
                'split':'training' if index < 2 else 'diagnostic_validation',
                'observed_worker_start_time_ns':offset+1_000_000,
                'worker_response':{'start_perf_ns':offset+1_000_000,'end_perf_ns':offset+5_000_000,
                    'observed_stages':[{'id':'native_inference','kind':'gpu','start_ns':offset+2_000_000,'end_ns':offset+4_000_000}]}})
            batches.append({'job_id':'job'+str(index),'start_ns':offset,'end_ns':offset+6_000_000,'request_ids':[rid]})
            for t in range(0,7_000_000,1_000_000):
                samples.append({'time_ns':offset+t,'watts':100,'owned_used_mib':500,'owned_aliases':list(RESIDENTS),'foreign_pids':[]})
        plan.write_text(json.dumps({'execution_context_sha256':digest(context),'jobs':[{'id':b['job_id'],'inputs':{'text#0':'x'},'split':'training' if i<2 else 'validation'} for i,b in enumerate(batches)]}))
        raw={'schema':'gsdeploy.native_phase_collection.v1','status':'completed','backend':'native',
             'context':receipt(ctx),'plan':receipt(plan),'execution_context_sha256':digest(context),
             'request_records':rows,'batches':batches,'board_samples':samples,'performance_scope':'CPU-fixture'}
        path=root/'raw.json';path.write_text(json.dumps(raw));return path

    def test_native_mapping_is_complete_mixed_phase_and_freezes_validation_out(self):
        path=self.native_raw();mapped=map_native_collection(path)
        self.assertEqual(mapped['mapping_report']['training_traces'],2)
        self.assertEqual(mapped['mapping_report']['validation_traces'],1)
        self.assertFalse(mapped['mapping_report']['kernel_evidence_available'])
        for trace in mapped['traces']:
            self.assertTrue(audit_trace(trace)['known'])
            segments=trace['requests'][0]['segments']
            self.assertEqual([s['kind'] for s in segments],['host','gpu','host'])
            self.assertIsNone(segments[1]['demand']['sm'])
        fitted=fit_native_collection(path)
        self.assertTrue(fitted['known'])
        self.assertEqual(len(fitted['training_input']['observations']),6)
        self.assertTrue(all(row['split']=='training' for row in fitted['training_input']['observations']))

    def test_native_mapper_rejects_training_split_tampering(self):
        path=self.native_raw();raw=json.loads(path.read_text())
        raw['request_records'][-1]['split']='training';path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError,'split differs'):
            map_native_collection(path)

    def test_kernel_tail_is_reported_outside_mixed_phase(self):
        path=self.native_raw();export=Path(self.tmp.name)/'export.json'
        export.write_text(json.dumps({'schema':'gsdeploy.nsys_observed_phases.v1','unattributed_kernels':[],
            'phases':[{'request_id':'r0','segment_id':'native_inference','clock_mapping_valid':True,
                       'kernels':[{'end_ns':4_100_000}], 'kernel_union_busy_ns':50, 'kernel_duration_sum_ns':60}]}))
        report=map_native_collection(path,[export])['mapping_report']['kernel_phase_reports'][0]
        self.assertEqual(report['async_tail_kernel_count'],1)
        self.assertEqual(report['max_async_tail_ns'],100_000)

    def test_missing_solo_produces_diagnostic_unknown_without_fit(self):
        path=self.native_raw();raw=json.loads(path.read_text())
        raw['request_records'][0]['observed_worker_start_time_ns']=None
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError,'Actual START'):
            map_native_collection(path)

    def test_cli_no_overwrite(self):
        source = Path(self.tmp.name)/'in.json'; source.write_text(json.dumps(self.trace()))
        output = Path(self.tmp.name)/'out.json'
        self.assertEqual(main(['audit', '--input',str(source),'--output',str(output)]),0)
        with self.assertRaises(FileExistsError):
            main(['audit','--input',str(source),'--output',str(output)])


if __name__ == '__main__':
    unittest.main()
