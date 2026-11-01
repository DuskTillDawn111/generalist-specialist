"""CPU timing-protocol tests; law/fit boundaries mocked, no capacity claim."""
import copy
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_matching_prospective_budget_v1 as m
from tools.predictor_validation_20261027.stage_response_features_v1 import receipt,checked
from tests.test_twoFIFO_fresh181_execute_v1 import PUBLIC


class BudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_twoFIFO_matching_prospective_bridge_v1 import BridgeTests
        BridgeTests.setUpClass();cls.fixture=BridgeTests
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.extracted=dict(snapshot=BridgeTests.prefix_doc['prefix_snapshot'],donor_plan=BridgeTests.plan,
            actual_priming_sha256=BridgeTests.prefix_doc['actual_priming_semantics_sha256'])
        cls.payload=dict(plans=BridgeTests.bundle['plans'],law=BridgeTests.law,unknown_reason=None,fit_report={})
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup();cls.fixture.tearDownClass()

    def test_exact_integer_margin_and_strict_predeclared_boundary(self):
        for duration in (1,2,3,180_000_000_000,10**30+1):
            self.assertEqual(m.required_budget_ns(duration),(3*duration+1)//2+30_000_000_000)
        for invalid in (0,-1,True,1.0):
            with self.assertRaises(ValueError):m.required_budget_ns(invalid)

    def test_real_io_all_four_calendars_both_passes_and_tamper_detection(self):
        fixture=self.fixture;calls=[]
        class Session:
            timings={'CPU_MOCK_PREAUTH_ONLY':True}
            def checked_payload(self):return copy.deepcopy(BudgetTests.payload)
        @contextmanager
        def opened(*args):yield Session()
        def replay(plans,law,reason,plan,prefix,gate):
            calls.append((plan['runtime_config'],gate))
            self.assertEqual(len(gate['public_requests']),181)
            self.assertEqual(gate['fixed_drain_deadline_ns']-gate['forecast_origin_ns'],210_000_000_000)
            return dict(source_supported_hypothesis_available=True,generated_worlds=12,
                forecast_sha256=m.core.fresh.digest(gate),world_unknown_probability_mass={'numerator':0,'denominator':1},
                forecast={'CPU_MOCK_NUMERICS_ONLY':True,'requests':gate['public_requests']})
        with patch.object(m.core,'open_law_session',side_effect=opened),patch.object(m,'_training_prefix',return_value=self.extracted),\
                patch.object(m.core,'replay_law',side_effect=replay),patch.object(m.core.response,'audit_fit',return_value={}):
            ref=m.benchmark(fixture.fit_ref['path'],PUBLIC,self.root/'measured')
            report=m.audit(ref['path'],fixture.fit_ref,lead_seconds=300)
            self.assertEqual(len(calls),8);self.assertEqual(len(report['cases']),4)
            for case in report['cases']:
                self.assertEqual([s['stage'] for s in case['stages']],list(m.STAGES))
                self.assertGreater(case['combined_live_workload_ns'],case['combined_replay_ns'])
                self.assertEqual(len(case['artifacts']),5)
            original=Path(ref['path']).read_bytes()
            try:
                changed=copy.deepcopy(report);changed['required_live_budget_ns']=1
                Path(ref['path']).write_text(m.core.json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'Timing budget'):m.audit(ref['path'],fixture.fit_ref,lead_seconds=300)
                Path(ref['path']).write_bytes(original)
                artifact=Path(report['cases'][0]['artifacts'][-1]['path']);data=artifact.read_bytes()
                artifact.write_bytes(data+b' ')
                try:
                    with self.assertRaises(ValueError):m.audit(ref['path'],fixture.fit_ref,lead_seconds=300)
                finally:artifact.write_bytes(data)
            finally:Path(ref['path']).write_bytes(original)
        self.assertFalse(report['GPU_started']);self.assertFalse(report['actual_target_prefix_overhead_verified'])

    def test_unqualified_real_fit_fails_without_output_or_replay(self):
        out=self.root/'unqualified'
        with patch.object(m.core,'replay_law') as replay:
            with self.assertRaises(ValueError):m.benchmark(self.fixture.fit_ref['path'],PUBLIC,out)
            replay.assert_not_called()
        self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
