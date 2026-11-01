import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from tests import test_twoFIFO_owned_training_review_v3 as attempt_fixture
from tests import test_static_language_work_v1 as static_fixture
from tools.predictor_validation_20261027 import twoFIFO_static_resource_sidecar_v2 as module
from tools.predictor_validation_20261027 import twoFIFO_composable_macros_v3 as macro_adapter
from tools.predictor_validation_20261027.implementation_source_closure_v27 import implementation_sources


def fixture():
    plan,raw=attempt_fixture.cohort_fixture()
    for row in raw['request_records']+raw['priming_request_records']:
        for event in row['execution_events']:
            if event.get('event')=='start' and event.get('semantic_phase') in ('prefill','decode'):
                event.update(input_batch_size=1,input_sequence_length=8 if event['semantic_phase']=='prefill' else 1)
    rows,events=macro_adapter.request_rows(raw,plan,'cpu-only')
    macros=dict(schema=module.MACRO_SCHEMA,status='COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER',mode='natural',requests=rows,measured_requests=len(rows),
        macros=sum(len(r['macros']) for r in rows),capture_id='block',attempt_id='shard',execution_id='execution',
        whole_attempt_history_used_for_visible_prefix=True,case_history_reset=False,
        priming_rows_excluded_as_labels=18,validation_labels_read=False)
    return macros,events,{'text':static_fixture.binding(True)},{'text#0':'text','text#1':'text'},raw


def run(args):return module.enrich(*args[:4],capture_sha256='cpu-only')


def find(out,rid,macro):return next(r for r in out['records'] if r['request_id']==rid and r['macro_id']==macro)


class TwoFIFOV3StaticResourceSidecarTests(unittest.TestCase):
    def test_existing_real_macro_adapter_all_rows_priming_history_and_explicit_UNKNOWN(self):
        args=fixture();before=copy.deepcopy(args);out=run(args)
        self.assertEqual(args,before)
        self.assertEqual(out['requests'],3);self.assertEqual(out['macros'],12)
        self.assertEqual(out['priming_rows_excluded_as_labels'],18)
        self.assertEqual(out['known_BEGIN_descriptors'],42)  # 18 priming+3 measured, 2 forward each
        self.assertEqual(out['input_macro_schema'],'gsdeploy.twoFIFO_composable_response_macro_training.v3')
        self.assertEqual(out['input_review_schema'],'gsdeploy.original_twoFIFO_owned_training_review.v3')
        self.assertEqual(out['priming_processed_history'],[e for e in args[1] if e.get('job_id')=='__priming__'])
        self.assertEqual(len({e['request_id'] for e in out['priming_processed_history']}),18)
        self.assertEqual(len(out['actual_BEGIN_descriptor_catalog']),42)
        self.assertFalse(any(r['request_id'].startswith('prime') for r in out['records']))
        self.assertEqual(out['known_pair_exposure_ns']+out['unknown_pair_exposure_ns'],out['visible_peer_exposure_ns'])
        observed=[p['projection_state']['own_BEGIN_descriptor'] for r in out['records'] for p in r['path']
                  if p['projection_state']['own_supported']]
        self.assertTrue(observed)
        full=observed[0]['nominal_projection_features']
        self.assertGreater(full['base_nominal_dense_gemm_flops'],0)
        self.assertIsNotNone(full['lora']);self.assertEqual(full['classification'],'unknown')
        for field in ('attention_kv_length','attention_matmul_flops','attention_kv_cache_bytes',
                      'measured_dram_bytes','hardware_cache_hit_rate'):
            self.assertIsNone(full[field])
        self.assertIsNone(out['unknown_physical_resources']['measured_l2_bytes'])
        self.assertFalse(out['reference_rate_modified']);self.assertFalse(out['cache_gain_qualified'])
        self.assertFalse(out['main_queue_provider_eligible'])
        self.assertIsNone(out['complementary_service_multiplier']);self.assertFalse(out['complementarity_qualified'])

    def test_unprocessed_own_BEGIN_remains_UNKNOWN_and_future_shape_change_cannot_backfill(self):
        args=fixture();before=run(args);row=find(before,'first','prefill_0000');early=row['path'][0]
        self.assertFalse(early['projection_state']['own_supported'])
        self.assertEqual(early['projection_state']['own_unknown_reasons'],['CURRENT_MACRO_BEGIN_NOT_YET_PROCESSED'])
        self.assertIsNone(early['projection_state']['own_BEGIN_descriptor'])
        changed=copy.deepcopy(args)
        for e in changed[1]:
            if e['request_id']=='first' and e.get('segment_id')=='prefill_0000' and e.get('event')=='start':
                e['input_sequence_length']=1234
        after=run(changed)
        self.assertEqual(early,find(after,'first','prefill_0000')['path'][0])
        later=[p for p in row['path'] if p['projection_state']['own_supported']]
        self.assertTrue(later)
        self.assertNotEqual(later[-1]['projection_state']['own_BEGIN_descriptor'],
            find(after,'first','prefill_0000')['path'][-1]['projection_state']['own_BEGIN_descriptor'])

    def test_missing_current_shape_is_not_filled_from_macro_shape_or_later_decode(self):
        args=fixture()
        for e in args[1]:
            if e['request_id']=='first' and e.get('segment_id')=='prefill_0000' and e.get('event')=='start':
                e.pop('input_sequence_length')
        out=run(args);row=find(out,'first','prefill_0000')
        self.assertTrue(all(not p['projection_state']['own_supported'] for p in row['path']))
        self.assertTrue(any(p['projection_state']['own_unknown_reasons']==['ACTUAL_BEGIN_SHAPE_MISSING'] for p in row['path']))
        self.assertTrue(any(p['projection_state']['own_supported'] for p in find(out,'first','decode_0001')['path']))
        self.assertFalse(out['future_shape_backfilled'])

    def test_unknown_peer_retained_with_K_plus_U_and_no_allpeer_imputation(self):
        args=fixture();args[3]['text#1']='ocr'  # Outside declared language binding despite same fake event shape.
        out=run(args);states=[p['projection_state'] for r in out['records'] if r['request_id']=='first' for p in r['path']]
        state=next(s for s in states if s['own_supported'] and s['peer_denominator_N'])
        self.assertEqual((state['peer_denominator_N'],state['known_pair_denominator_K'],state['unknown_pair_denominator_U']),(1,0,1))
        self.assertEqual(len(state['pairs']),1);self.assertIsNone(state['all_peer_contrast_sum'])
        self.assertIsNone(state['pairs'][0]['partial_decoder_projection_contrast'])
        self.assertEqual(state['known_partial_decoder_projection_contrast_sum'],0)
        for r in out['records']:
            if r['request_id']=='peer':self.assertTrue(all(not p['projection_state']['own_supported'] for p in r['path']))

    def test_future_peer_BEGIN_content_cannot_change_earlier_pair_state(self):
        args=fixture();before=run(args);changed=copy.deepcopy(args)
        peer_event=next(e for e in changed[1] if e['request_id']=='peer' and e.get('segment_id')=='prefill_0000' and e.get('event')=='start')
        cutoff=peer_event['collector_processing_time_ns'];peer_event['input_sequence_length']=999
        after=run(changed)
        before_states=[p for r in before['records'] for p in r['path'] if p['left_ns']<cutoff]
        after_states=[p for r in after['records'] for p in r['path'] if p['left_ns']<cutoff]
        self.assertEqual(before_states,after_states)

    def test_zero_peer_exposure_is_not_divided_or_called_a_known_resource(self):
        args=fixture();args[0]['requests']=args[0]['requests'][:2]
        args[0].update(measured_requests=2,macros=8)
        args[1][:]=[e for e in args[1] if e['owner']=='text#0']
        for request in args[0]['requests']:
            for macro in request['macros']:
                for piece in macro['causal_covariate_path']:
                    piece['state'].update(peer_macros=[],peer_visible_macro_owner_count=0,peer_same_visible_macro_phase_owner_count=0)
        out=run(args)
        self.assertEqual(out['visible_peer_exposure_ns'],0)
        self.assertIsNone(out['known_pair_exposure_fraction'])
        self.assertEqual(out['unknown_physical_resources']['physical_classification'],'unknown')
        self.assertTrue(any(not p['projection_state']['own_supported'] for r in out['records'] for p in r['path']))

    def test_source_schema_whole_attempt_no_reset_and_macro_denominator_cannot_be_forged(self):
        for mutate in (lambda d:d.update(schema='gsdeploy.composable_response_macro_training.v1'),
                       lambda d:d.update(schema='gsdeploy.twoFIFO_composable_response_macro_training.v2'),
                       lambda d:d.update(case_history_reset=True),lambda d:d.update(validation_labels_read=True),
                       lambda d:d.update(macros=11),lambda d:d.update(measured_requests=2)):
            args=fixture();mutate(args[0])
            with self.assertRaises(ValueError):run(args)

    def test_future_or_duplicate_visible_peer_is_rejected_without_imputation(self):
        for change in ('future','duplicate'):
            args=fixture()
            piece=next(p for r in args[0]['requests'] for macro in r['macros'] for p in macro['causal_covariate_path']
                if p['state']['peer_macros'])
            peers=piece['state']['peer_macros']
            if change=='future':peers[0]['available_ns']=piece['left_ns']+1
            else:peers.append(copy.deepcopy(peers[0]))
            with self.assertRaisesRegex(ValueError,'processed by the current cutoff'):run(args)

    def test_real_extract_entry_used_and_source_closure_is_downstream_only(self):
        # No native training evidence exists yet: a forged/absent review cannot
        # make the public entry silently use the pure CPU helper.
        with patch.object(module.macros_v3,'extract',side_effect=ValueError('native evidence required')) as actual:
            with self.assertRaisesRegex(ValueError,'native evidence required'):module.extract('review','inventory')
            actual.assert_called_once_with('review')
        root=Path(__file__).resolve().parents[1]
        collector=root/'tools/predictor_validation_20261027/twoFIFO_owned_collect_v3.py'
        closure={Path(r['path']).name for r in implementation_sources(collector)}
        self.assertNotIn('twoFIFO_static_resource_sidecar_v2.py',closure)
        self.assertFalse((root/'tools/predictor_validation_20261027/twoFIFO_static_resource_sidecar_v2.py').read_text().count('import torch'))


if __name__=='__main__':unittest.main()
