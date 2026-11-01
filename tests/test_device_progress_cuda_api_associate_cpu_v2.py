"""Conditional operation association and conservative CPU counterexamples."""
import copy
from pathlib import Path
import tempfile
import unittest

from tests.test_device_progress_cuda_api_probe_cpu_v1 import sqlite_fixture,raw_fixture
from tools.predictor_validation_20261027 import device_progress_cuda_api_associate_cpu_v2 as v


def row_fixture():
    row=raw_fixture();row['request_id']='r';response=row['worker_response'];response.update(start_perf_ns=1080,end_perf_ns=1250)
    doc=response['semantic_forward_observations'];doc.update(request_id='r',records=[dict(id='prefill_0000',nvtx_name='GS_STAGE:r:gpu:prefill_0000')])
    p=doc['device_progress'];p['records']=[dict(phase_id='prefill_0000',event_id='r:prefill_0000:stream_end',begin_record_host_bracket_ns=[1109,1116],end_record_host_bracket_ns=[1179,1186])]
    p['queries']=[dict(request_id='r',phase_id='prefill_0000',observation_sequence=1,host_query_begin_ns=1209,host_query_end_ns=1216,end_event_complete=False),
        dict(request_id='r',phase_id='prefill_0000',observation_sequence=2,host_query_begin_ns=1219,host_query_end_ns=1226,end_event_complete=True)]
    return row


class ConditionalAPIAssociationTests(unittest.TestCase):
    def native(self,directory):
        path=Path(directory)/'synthetic.sqlite';sqlite_fixture(path);return v.read_sqlite(path)
    def test_complete_native_sequence_is_only_conditionally_associated(self):
        with tempfile.TemporaryDirectory() as d:
            doc=self.native(d);row=row_fixture();before=copy.deepcopy((row,doc));report=v.associate_request(row,100,doc)
            self.assertEqual((row,doc),before);self.assertTrue(report['conditional_query_elapsed_association_consistent'])
            self.assertEqual(report['consistent_API_layers'],['CUPTI_ACTIVITY_KIND_RUNTIME'])
            bindings=report['layers']['CUPTI_ACTIVITY_KIND_RUNTIME']['bindings']
            self.assertEqual([b['logical']['operation'] for b in bindings],['Record','Record','Query','Query','ElapsedTime'])
            self.assertFalse(report['query_and_elapsed_association_authenticated']);self.assertFalse(report['independent_clock_alignment_authenticated'])
            for key in ('event_handle_identity_authenticated','stream_handle_identity_authenticated','reported_elapsed_value_authenticated','all_request_GPU_work_complete','main_lifecycle_resume_eligible'):self.assertFalse(report[key])
            self.assertIsNone(report['kernel_cursor'])
    def test_elapsed_is_after_query_bracket_not_inside_it(self):
        operations=v.expected_operations(row_fixture());self.assertEqual(operations[-1]['host_bracket_ns'],[1226,1250])
        with tempfile.TemporaryDirectory() as d:
            doc=self.native(d);elapsed=next(a for a in doc['event_apis'] if a['operation']=='ElapsedTime');elapsed.update(native_start_ns=222,native_end_ns=224)
            self.assertFalse(v.associate_request(row_fixture(),100,doc)['conditional_query_elapsed_association_consistent'])
    def test_query_return_codes_cannot_be_inferred_from_plausible_JSON(self):
        for done_query,bad_value in [(False,0),(False,1),(True,600),(True,True)]:
            with tempfile.TemporaryDirectory() as d:
                doc=self.native(d);queries=[a for a in doc['event_apis'] if a['operation']=='Query'];queries[1 if done_query else 0]['return_value']=bad_value
                report=v.associate_request(row_fixture(),100,doc);self.assertFalse(report['conditional_query_elapsed_association_consistent'])
                self.assertTrue(any('RETURN_MISMATCH' in issue for issue in report['layers']['CUPTI_ACTIVITY_KIND_RUNTIME']['issues']))
    def test_missing_extra_ambiguous_failed_elapsed_and_wrong_thread_rejected(self):
        for mutation in ('missing_query','extra_query','ambiguous_query','failed_elapsed','wrong_thread'):
            with tempfile.TemporaryDirectory() as d:
                doc=self.native(d);query=next(a for a in doc['event_apis'] if a['operation']=='Query')
                if mutation=='missing_query':doc['event_apis'].remove(query)
                elif mutation=='extra_query':
                    extra=copy.deepcopy(query);extra.update(native_start_ns=235,native_end_ns=240);doc['event_apis'].append(extra)
                elif mutation=='ambiguous_query':doc['event_apis'].append(copy.deepcopy(query))
                elif mutation=='failed_elapsed':next(a for a in doc['event_apis'] if a['operation']=='ElapsedTime')['return_value']=600
                else:query['global_tid']+=1
                with self.subTest(mutation=mutation):self.assertFalse(v.associate_request(row_fixture(),100,doc)['conditional_query_elapsed_association_consistent'])
    def test_driver_and_runtime_partial_rows_cannot_be_combined(self):
        with tempfile.TemporaryDirectory() as d:
            doc=self.native(d)
            for call in doc['event_apis']:
                if call['operation'] in ('Query','ElapsedTime'):call['table']='CUPTI_ACTIVITY_KIND_DRIVER'
            report=v.associate_request(row_fixture(),100,doc);self.assertFalse(report['conditional_query_elapsed_association_consistent'])
    def test_native_record_presence_can_survive_empty_clock_but_query_association_cannot(self):
        with tempfile.TemporaryDirectory() as d:
            doc=self.native(d);row=row_fixture();row['worker_response']['semantic_forward_observations']['device_progress']['nvtx_host_brackets'][0].update(pop_before_ns=1300,pop_after_ns=1301)
            report=v.associate_request(row,100,doc);self.assertTrue(report['record_containment']['per_forward_record_calls_authenticated'])
            self.assertFalse(report['conditional_clock']['nonempty']);self.assertFalse(report['conditional_query_elapsed_association_consistent'])
    def test_API_constraints_must_share_one_offset(self):
        with tempfile.TemporaryDirectory() as d:
            doc=self.native(d);row=row_fixture();p=row['worker_response']['semantic_forward_observations']['device_progress']
            # Both fit the NVTX interval separately: begin needs delta=999,
            # end needs delta=1001. No common mapping can explain the pair.
            p['records'][0]['begin_record_host_bracket_ns']=[1109,1114]
            p['records'][0]['end_record_host_bracket_ns']=[1181,1186]
            self.assertFalse(v.associate_request(row,100,doc)['conditional_query_elapsed_association_consistent'])
    def test_perrequest_sequence_identity_and_source_order_invalid_raw_rejected(self):
        for mutation in ('sequence','request','record','overlap'):
            row=row_fixture();p=row['worker_response']['semantic_forward_observations']['device_progress']
            if mutation=='sequence':p['queries'][0]['observation_sequence']=2
            elif mutation=='request':p['queries'][0]['request_id']='other'
            elif mutation=='record':p['records'][0]['event_id']='other'
            else:p['queries'][0].update(host_query_begin_ns=1180,host_query_end_ns=1190)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):v.expected_operations(row)


if __name__=='__main__':unittest.main()
