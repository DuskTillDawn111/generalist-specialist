"""Independent CPU source-race and qualification review; no real GPU calls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import admission_search_check_v1 as m


class AdmissionSearchIndependentReview(unittest.TestCase):
    def files(self,folder):
        report=folder/'report.json';audit=folder/'validation-audit.json'
        m.save(report,dict(original='independent report'))
        m.save(folder/'seal-audit.json',dict(original='seal'))
        m.save(folder/'predictions.json',dict(original='predictions'))
        m.save(audit,dict(seal_audit=m.receipt(folder/'seal-audit.json')))
        return report,audit

    def test_post_verify_report_byte_drift_rejected_before_using_new_evidence(self):
        with tempfile.TemporaryDirectory()as tmp:
            folder=Path(tmp);report,audit=self.files(folder);old=m.receipt(report)
            def verify(path):
                report.write_text('{"unverified":"replacement"}')
                return dict(verified=True,report=old)
            with (patch.object(m.validation,'verify_report',side_effect=verify),
                  patch.object(m.validation,'check_sealed',side_effect=AssertionError('Do not consume unverified replacement'))):
                with self.assertRaises(ValueError):m.run(folder,report,folder/'out')
            self.assertFalse((folder/'out').exists())

    def test_post_verify_audit_byte_drift_rejected_even_when_same_seal_receipt(self):
        with tempfile.TemporaryDirectory()as tmp:
            folder=Path(tmp);report,audit=self.files(folder);old=m.receipt(report)
            def verify(path):
                value=m.read(audit);value['unverified_new_evidence']=True;audit.write_text(__import__('json').dumps(value))
                return dict(verified=True,report=old)
            with (patch.object(m.validation,'verify_report',side_effect=verify),
                  patch.object(m.validation,'check_sealed',side_effect=AssertionError('Audit changed after proof'))):
                with self.assertRaises(ValueError):m.run(folder,report,folder/'out')
            self.assertFalse((folder/'out').exists())

    def test_verification_packet_must_bind_exact_report_receipt(self):
        with tempfile.TemporaryDirectory()as tmp:
            folder=Path(tmp);report,_=self.files(folder)
            with (patch.object(m.validation,'verify_report',return_value=dict(verified=True,report=dict(path='foreign',sha256='0'*64))),
                  patch.object(m.validation,'check_sealed',side_effect=AssertionError('Foreign report proof'))):
                with self.assertRaises(ValueError):m.run(folder,report,folder/'out')

    def test_unknown_qualification_and_known_failure_never_gain_certificate(self):
        row=dict(complete_independent_evidence=True,statistical_service_applicability=True,
            original_rho_threshold_satisfied=True,original_stability=dict(original_stability_passed=True),
            global_permit_capacity_qualified=None,original_measured_profile_resolver_admission=None)
        self.assertIsNone(m.qualification(row,dict(rho_admission_passed=None))['passed'])
        row['original_stability']['original_stability_passed']=False
        self.assertIs(m.qualification(row,dict(rho_admission_passed=None))['passed'],False)
        for invalid in [1,0,'PASS']:
            with self.subTest(invalid=invalid):
                row['original_rho_threshold_satisfied']=invalid
                with self.assertRaises(ValueError):m.qualification(row,dict(rho_admission_passed=None))


if __name__=='__main__':unittest.main()
