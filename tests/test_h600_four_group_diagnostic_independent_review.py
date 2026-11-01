"""Independent CPU review regressions; no real prepare, worker or GPU.

These tests require immutable original objective/catalog/profile contracts and
source receipts that remain stable during the entire final actual audit.
"""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from tests import test_h600_four_group_diagnostic_v1 as fixtures
from tools.predictor_validation_20261027 import h600_four_group_diagnostic_v1 as m


class FinalIndependentReviewTests(unittest.TestCase):
    def setUp(self):
        fixtures.FinalDiagnosticTests.setUp(self)

    fixture=fixtures.FinalDiagnosticTests.fixture

    def rewrite_both_plans(self,mutate):
        original_path=self.folder/'native-prepared-plan.json'
        original=m.read(original_path);mutate(original)
        original['plan_sha256']=m.digest({k:v for k,v in original.items() if k!='plan_sha256'})
        fixtures.write(original_path,original)
        plan=m._project(original,self.plan['created_ns']);fixtures.write(self.plan_path,plan)

    def test_energy_reference_cannot_be_rehashed_away(self):
        self.rewrite_both_plans(lambda p:p.update(energy_reference_watts=4500.))
        with self.assertRaises(ValueError):m.audit_plan(self.plan_path)

    def test_latency_reference_cannot_be_rehashed_away(self):
        self.rewrite_both_plans(lambda p:p['latency_reference_ms'].update(text=100000.))
        with self.assertRaises(ValueError):m.audit_plan(self.plan_path)

    def test_foreign_catalog_path_cannot_be_rehashed_away(self):
        foreign=self.folder/'foreign-catalog.json'
        fixtures.write(foreign,m.read(self.plan['catalog']['path']))
        self.rewrite_both_plans(lambda p:p.update(catalog=m.receipt(foreign)))
        with self.assertRaises(ValueError):m.audit_plan(self.plan_path)

    def test_missing_native_quality_scorer_sources_rejected(self):
        raw,path=self.fixture()
        # Original native_flow always includes native_flow/tasks/quality too.
        scored_path=path.parent/'scored-raw.json';scored=m.read(scored_path)
        scored['quality_evidence']['source_receipts']=[self.plan['catalog']]
        fixtures.write(scored_path,scored)
        with self.assertRaises(ValueError):m.audit_actual(self.plan_path,path,seal_ns=self.plan['created_ns']+1)

    def test_raw_source_must_not_change_after_mechanism_audit(self):
        raw,path=self.fixture();scored_path=path.parent/'scored-raw.json'
        scored=m.read(scored_path)
        # Supply complete real scorer sources so this regression isolates the
        # source-changing race independently of missing-source rejection.
        import gsdeploy.tasks,gsdeploy.quality
        scored['quality_evidence']['source_receipts']=[self.plan['catalog'],m.receipt(m.native_flow.__file__),
            m.receipt(gsdeploy.tasks.__file__),m.receipt(gsdeploy.quality.__file__)]
        fixtures.write(scored_path,scored)
        original_audit=m.mechanism_audit.audit
        def mutate_after_contract(raw_path):
            report=original_audit(raw_path)
            changed=copy.deepcopy(raw)
            for sample in changed['board_samples']:sample['watts']*=2
            fixtures.write(path,changed)
            # Leave the scored native payload unchanged, but bind its pointer
            # to changed raw. The outer audit must catch this inconsistency.
            annotated=m.read(scored_path);annotated['native_raw']=m.receipt(path)
            fixtures.write(scored_path,annotated)
            return report
        with patch.object(m.mechanism_audit,'audit',side_effect=mutate_after_contract):
            with self.assertRaises(ValueError):m.audit_actual(self.plan_path,path,seal_ns=self.plan['created_ns']+1)


if __name__=='__main__':unittest.main()
