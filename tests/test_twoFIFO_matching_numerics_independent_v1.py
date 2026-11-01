"""Independent CPU counterexamples for the exact15 matching numerical chain."""
import copy
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_twoFIFO_matching_numerics_v1 as fixtures
from tests.test_twoFIFO_matching_numerics_v1 import (
    PRODUCER,matching_fixture,bundle_fixture,resource_tables,timeline)
from tools.predictor_validation_20261027 import twoFIFO_matching_phase12_v1 as phase
from tools.predictor_validation_20261027 import twoFIFO_matching_phase12_adapter_v1 as adapter
from tools.predictor_validation_20261027 import twoFIFO_matching_generated_v1 as generated
from tools.predictor_validation_20261027 import twoFIFO_matching_resource_v1 as resource

ROOT=Path(__file__).resolve().parents[1]
AUTHOR=ROOT/'artifacts/predictor-twoFIFO-matching-numerics-v1-20261029-cpu01-author/report.json'


class IndependentMatchingNumerics(unittest.TestCase):
    def test_real_prepared_chain_and_cpu02_producer_without_label_or_fit(self):
        author=phase.checked(phase.receipt(AUTHOR));refs=author['prepared']
        with patch.object(phase.original,'fitted_rows',side_effect=AssertionError('optimizer forbidden')),\
             patch.object(phase.intake,'intake',side_effect=AssertionError('native labels forbidden')):
            manifest=phase.audit(refs['phase_manifest']['path'])
            draw=generated.audit(refs['generated_design']['path'])
            power=resource.audit(refs['resource_design']['path'])
            with tempfile.TemporaryDirectory() as temp:
                waiting=phase.checked(phase.run(refs['phase_manifest']['path'],Path(temp)/'phase',fit_enabled=True))
                rwait=phase.checked(resource.run(refs['resource_design']['path'],Path(temp)/'resources',fit_enabled=True))
        self.assertEqual(manifest['executor_manifest'],phase.receipt(PRODUCER))
        self.assertEqual(draw['fit_manifest'],power['fit_manifest'])
        self.assertEqual(len(manifest['expected_attempt_ids']),15)
        self.assertEqual(waiting['status'],rwait['status'])
        self.assertEqual(waiting['status'],'WAITING_FOR_MATCHING_EXACT15_CONTINUATION_FINAL')

    def test_true_calendar_OCR_denominator_survives_arbitrary_case_names(self):
        plans,_,docs=matching_fixture()
        for p in plans:
            for i,c in enumerate(p['cases']):c['case_id']='arbitrary-'+str(i)
        actual=adapter.observed_support(docs,plans)
        pools={a['pool_id'] for p in plans for c in p['cases'] for a in c['arrivals'] if a['owner']=='ocr#0'}
        keys={(r['capture_id'],r['pool_id']) for r in actual['OCR_exact_input_by_whole_block']}
        self.assertEqual(keys,{(p['capture_id'],pool) for p in plans for pool in pools})
        for d in docs:
            for r in d['requests']:
                r['macros']=[m for m in r['macros'] if m['owner']!='ocr#0']
        missing=adapter.observed_support(docs,plans)
        self.assertFalse(missing['each_target_OCR_input_each_block_n0_and_npositive'])
        self.assertTrue(all(r['entry_macros']==0 for r in missing['OCR_exact_input_by_whole_block']))

    def test_full12_no_column_drop_and_whole_block_holdout_not_rawSHA(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle=bundle_fixture(temp);rows,blocks=phase._validate_bundle(bundle)
        self.assertEqual(len(rows),390);self.assertEqual(len(blocks),3)
        self.assertEqual(len({r['raw_sha256'] for r in rows}),15)
        self.assertTrue(all(r['matrix'].shape[1]==12 and r['ordered_matrix'].shape[1]==12 for r in rows))
        calls=[]
        def rejected(r):
            calls.append(set(x['attempt_id'] for x in r))
            return dict(numerical_qualification=False),None
        with patch.object(phase.original,'fitted_rows',side_effect=rejected):fitted=phase.fit_bundle(bundle)
        self.assertEqual([len(s) for s in calls],[15,10,10,10])
        self.assertEqual(sum(f['evaluation']['summary']['unknown_rows'] for f in fitted['whole_block_leaveout']),390)
        for fold in fitted['whole_block_leaveout']:
            self.assertFalse(set(fold['training_attempt_ids'])&set(fold['heldout_attempt_ids']))
            self.assertEqual(len(fold['heldout_attempt_ids']),5)

    def test_priming_semantics_not_request_ID_and_unknown_runtime_knobs_fail(self):
        plans,_,_=matching_fixture();p=plans[0];context=phase.checked(p['context'])
        original=resource.physical_execution_identity(p,context)
        relabel=copy.deepcopy(p)
        for i,r in enumerate(relabel['priming']):r['request_id']='fresh-prime-'+str(i)
        self.assertEqual(resource.physical_execution_identity(relabel,context),original)
        changed=copy.deepcopy(p);changed['priming'][0]['pool_id']='unknown-content'
        with self.assertRaises((ValueError,KeyError)):resource.physical_execution_identity(changed,context)
        changed=copy.deepcopy(p);changed['runtime_config']['unreviewed_knob']=1
        with self.assertRaisesRegex(ValueError,'runtime knob'):resource.physical_execution_identity(changed,context)

    def test_resource_state_missing_in_second_block_never_one_block_qualification(self):
        tables=resource_tables()
        for t in tables:
            if t['capture_id']!='block-1':t['slices'][0]['state_key']='other'
        fitted=resource.fit_tables(tables)
        self.assertFalse(fitted['power_states']['state']['known'])
        self.assertIsNone(fitted['power_states']['state']['mean'])
        self.assertTrue(fitted['power_states']['other']['known'])
        with self.assertRaisesRegex(ValueError,'five shards'):resource.fit_tables(tables[:-1])

    def test_window_or_total_exposure_drift_never_silent_resource_transfer(self):
        author_helper=fixtures.MatchingNumericalContracts();provider,context=author_helper.provider()
        for t in (timeline(9,60),timeline(10,61),timeline(90,120)):
            with patch.object(provider,'power',side_effect=AssertionError('unsupported lookup')),\
                 patch.object(provider,'memory',side_effect=AssertionError('unsupported lookup')):
                predicted=author_helper.predict(t,provider,context,dict(basis='fresh_after_original_priming'))
            self.assertIsNone(predicted['board_energy_joules'])
            self.assertEqual(predicted['memory_unknown_observation_duration_ns'],t['total_wall_ns'])
            self.assertEqual(predicted['power_unknown_measurement_duration_ns'],t['total_measurement_overlap_ns'])
        t=timeline();t['total_wall_ns']['numerator']+=1
        with self.assertRaisesRegex(ValueError,'denominator'):author_helper.predict(t,provider,context)


if __name__=='__main__':unittest.main()
