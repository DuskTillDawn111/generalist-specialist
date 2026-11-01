"""Independent shared-dispatch deadline and failed-evidence persistence checks."""
from pathlib import Path
import unittest
from unittest.mock import patch

from tests import test_twoFIFO_fresh181_native_v4_collector as fixture
from tools.predictor_validation_20261027 import twoFIFO_fresh181_native_v4 as native


class IndependentNativeV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixture.NativeV4CollectorTests.setUpClass()

    @classmethod
    def tearDownClass(cls):fixture.NativeV4CollectorTests.tearDownClass()

    def test_failure_after_report_restoration_is_saved_after_worker_cleanup(self):
        helper=fixture.NativeV4CollectorTests()
        raw,runtime,_,_,_,plan=helper.collect('report_restore')
        persisted=native.checked(native.receipt(Path(plan['forecast_barrier']['prefix_path']).parent/'raw.json'))
        self.assertTrue(runtime.closed)
        self.assertEqual(persisted['status'],'failed')
        self.assertEqual(len(persisted['request_records']),181)
        self.assertTrue(all(row['status']=='not_started' for row in persisted['request_records']))
        self.assertTrue(any('ShutdownForecastSourceGuardError' in error for error in persisted['errors']))
        self.assertTrue(persisted['shutdown_failure_receipt_extra_write'])
        self.assertEqual(len(runtime.enqueued),18)

    def test_prefix_construction_crossing_origin_sends_no_target_or_producer(self):
        helper=fixture.NativeV4CollectorTests();clock=fixture.Clock()
        original=native.prefix.make_prefix
        def slow_prefix(*args,**kwargs):
            result=original(*args,**kwargs)
            clock.wait(args[4])
            return result
        with patch.object(fixture,'Clock',return_value=clock), \
                patch.object(native.prefix,'make_prefix',side_effect=slow_prefix):
            raw,runtime,_,calls,_,_=helper.collect()
        self.assertEqual(raw['status'],'failed')
        self.assertEqual(calls,[])
        self.assertEqual(len(runtime.enqueued),18)
        self.assertTrue(runtime.closed)
        self.assertTrue(all(row['status']=='not_started' for row in raw['request_records']))


if __name__=='__main__':unittest.main()
