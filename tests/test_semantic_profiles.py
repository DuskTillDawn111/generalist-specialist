"""Host semantic subdivision preserves coverage and rejects unsupported concurrency."""
import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.semantic_profiles import split_semantic_native
from tests.test_native_training import capture


def row_with_calls():
    row=copy.deepcopy(capture('one','solo')['raw']['request_records'][0])
    left=row['worker_response']['observed_stages'][1]['start_ns']
    def event(role,start,end):
        return dict(request_id=row['request_id'],role=role,component=role,
                    host_start_ns=left+start*1_000_000,host_end_ns=left+end*1_000_000,
                    device_event_complete=True,device_elapsed_ms=float(end-start),
                    stream_id=7,end_current_stream_id=7)
    row['worker_response'].update(semantic_phase_schema='gsdeploy.semantic_native_observation.v1',
        semantic_phase_events=[event('prepare',1,8),event('vision',2,4),event('language',8,9)])
    return row


class SemanticProfileTests(unittest.TestCase):
    def test_nested_calls_and_gaps_cover_the_original_mixed_handler(self):
        row=row_with_calls();original=copy.deepcopy(row);layout=split_semantic_native(row)
        self.assertEqual(row,original)
        self.assertEqual([s['semantic_role'] for s in layout],
                         ['setup','native_gap','prepare','vision','prepare','language','native_gap','tail'])
        self.assertEqual(sum(s['end_ns']-s['start_ns'] for s in layout),12_000_000)
        self.assertTrue(all(s['kind']=='gpu' and s['mixed_host_transfers_CUDA'] for s in layout[1:-1]))

    def test_asynchronous_device_elapsed_does_not_replace_host_boundaries(self):
        row=row_with_calls();row['worker_response']['semantic_phase_events'][1]['device_elapsed_ms']=1000.
        layout=split_semantic_native(row)
        vision=next(s for s in layout if s['semantic_role']=='vision')
        self.assertEqual(vision['end_ns']-vision['start_ns'],2_000_000)

    def test_crossing_calls_are_unknown_in_the_single_owner_segment_model(self):
        row=row_with_calls();events=row['worker_response']['semantic_phase_events']
        events[-1]['host_start_ns']=events[0]['host_end_ns']-1
        with self.assertRaisesRegex(UnknownEvidence,'CONCURRENT_NONNESTED'):
            split_semantic_native(row)

    def test_missing_incomplete_and_out_of_native_observations_are_rejected(self):
        row=row_with_calls();row['worker_response'].pop('semantic_phase_events')
        with self.assertRaisesRegex(UnknownEvidence,'UNKNOWN_SEMANTIC'):
            split_semantic_native(row)
        row=row_with_calls();row['worker_response']['semantic_phase_events'][0]['device_event_complete']=False
        with self.assertRaisesRegex(ValueError,'Complete causal'):
            split_semantic_native(row)
        row=row_with_calls();row['worker_response']['semantic_phase_events'][0]['host_start_ns']-=2_000_000
        with self.assertRaisesRegex(ValueError,'outside'):
            split_semantic_native(row)


if __name__=='__main__':unittest.main()
