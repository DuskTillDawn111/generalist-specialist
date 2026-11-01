"""CPU timing-plumbing fixture; no native budget qualification is produced.

Saved-law and native-intake authentication are mocked explicitly. The complete
queue→original Q/M/E→six-marginal composition, dual-pass comparison, durable IO,
fixed margin, source receipts and tamper rejection execute normally.
"""
from contextlib import contextmanager,ExitStack
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_public181_full_score_budget_v1 as m
from tests import test_twoFIFO_public181_expected_score_v2 as fixtures


class FullScoreBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.ConditionalExpectedTests.setUpClass();cls.fixture=fixtures.ConditionalExpectedTests()

    @contextmanager
    def environment(self,root,*,missing_quality=False):
        f=self.fixture
        def write(name,document):
            path=root/name;path.write_text(json.dumps(document));return m.receipt(path)
        fit=write('fit.json',{'CPU_FIXTURE_NOT_REAL_FIT':True})
        reviews=[write('review'+str(i)+'.json',{'CPU_ONLY':i}) for i in range(8)]
        resource_manifest=write('resource.json',{'CPU_FIXTURE_NOT_NATIVE_MANIFEST':True})
        public_report=Path('artifacts/predictor-twoFIFO-prospective-public-contract-v1-20261029-cpu02-prepared/report.json').resolve()
        public_document=m.checked(m.receipt(public_report));contract=m.checked(public_document['artifacts']['public_contract'])
        intake=f.rows()
        if missing_quality:
            for row in intake['rows']:row['actual_native_quality']=None
        payload=dict(manifest_document=f.manifest,intake=dict(intake,unknown_resource_attempts=0),source_bindings=[f.plan['public_contract']])
        queue_draws=[dict(draw_id=str(i),block_id='b'+str(i//4),weight=F(1,12),status='GENERATED_WORLD',forecast={'T':1,'L':2,'P':3}) for i in range(12)]
        queue_distributions={t:m.expected.ensemble.distribution(queue_draws,lambda value,t=t:value[t]) for t in 'TLP'}
        queue_replay=dict(forecast=m.core.encode({'ensemble':{'draws':queue_draws,'six_terms':queue_distributions}}),source_supported_hypothesis_available=True)
        @contextmanager
        def resource_session(*args,**kwargs):yield SimpleNamespace(checked_payload=lambda:payload)
        @contextmanager
        def law_session(*args,**kwargs):yield SimpleNamespace(checked_payload=lambda:dict(law={'CPU_ONLY':True},plans=[],unknown_reason=None,fit_report={}))
        def proxy_inputs(resource,training,original,ref,candidate):
            donor=next(m.checked(item['plan']) for item in f.manifest['attempts'] if item['candidate_id']==candidate['id'])
            plan=copy.deepcopy(m.checked(donor['reference_validation_plan']))
            plan.update(shared_dispatch_protocol=m.expected.shared_dispatch_protocol(),evaluation_protocol=m.expected.EVALUATION_PROTOCOL)
            plan['forecast_barrier'].update(lead_seconds=300,prior_READY_settle_timeout_seconds=10)
            epoch=f.gate['forecast_origin_ns'];gate=dict(f.gate,
                public_requests=[dict(request_id=r['request_id'],owner=candidate['routes'][r['request_id']],task=r['task'],pool_id=r['pool_id'],arrival_ns=epoch+r['offset_ns']) for r in original['trace']])
            prefix=dict(f.prefix,seed={'CPU_ONLY':True});return plan,prefix,gate
        with ExitStack() as stack:
            stack.enter_context(patch.object(m.expected,'open_resource_session',side_effect=resource_session))
            stack.enter_context(patch.object(m.core,'open_law_session',side_effect=law_session))
            stack.enter_context(patch.object(m.core,'support',return_value={'complete_all_keys_in_every_donor_block':True}))
            stack.enter_context(patch.object(m.old,'_training_prefix',return_value={'snapshot':{'snapshot_time_ns':100}}))
            stack.enter_context(patch.object(m,'_proxy_inputs',side_effect=proxy_inputs))
            replay=stack.enter_context(patch.object(m.core,'replay_law',return_value=queue_replay))
            stack.enter_context(patch.object(m.core.response,'audit_fit',return_value={}))
            stack.enter_context(patch.object(m,'implementation_sources',return_value=[m.receipt(m.__file__)]))
            yield fit,public_report,resource_manifest,reviews,replay

    def test_full_score_timing_has_two_actual_composition_passes_per_candidate_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.environment(root) as (fit,public,resource,reviews,replay):
                report=m.benchmark(fit['path'],public,resource['path'],[r['path'] for r in reviews],root/'budget')
                doc=m.audit(report['path'],fit,resource_manifest_ref=resource,resource_review_refs=reviews)
                self.assertEqual(replay.call_count,8);self.assertTrue(doc['full_U_live_pipeline_included'])
                self.assertEqual(len(doc['cases']),4);self.assertFalse(doc['full_U_accuracy_qualified'])
                self.assertFalse(doc['actual_target_prefix_overhead_verified'])
                self.assertEqual(doc['required_live_budget_ns'],m.required_budget_ns(doc['maximum_measured_live_workload_ns']))
                for case in doc['cases']:
                    self.assertEqual(len(case['artifacts']),5);self.assertEqual([s['stage'] for s in case['stages']],list(m.STAGES))
                path=Path(report['path']);original=path.read_bytes()
                changes=[lambda d:d.update(full_U_live_pipeline_included=False),
                    lambda d:d.update(required_live_budget_ns=1),
                    lambda d:d['cases'][0]['stages'].pop(),
                    lambda d:d.update(evaluation_protocol={}),
                    lambda d:d.update(full_U_accuracy_qualified=True)]
                try:
                    for change in changes:
                        altered=copy.deepcopy(doc);change(altered);path.write_text(json.dumps(altered))
                        with self.assertRaises(ValueError):m.audit(path,fit,resource_manifest_ref=resource,resource_review_refs=reviews)
                finally:path.write_bytes(original)

    def test_unknown_quality_cannot_use_fastpath_to_claim_fullsixterm_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.environment(root,missing_quality=True) as (fit,public,resource,reviews,replay):
                with self.assertRaisesRegex(ValueError,'complete actual native Q/M/E support'):
                    m.benchmark(fit['path'],public,resource['path'],[r['path'] for r in reviews],root/'budget')
                self.assertEqual(replay.call_count,1);self.assertFalse((root/'budget/report.json').exists())

    def test_fixed_protocol_rejects_longer_lead_or_missing_native_review_denominator(self):
        with self.assertRaises(ValueError):m.benchmark('/missing','/missing','/missing',[],'/missing',lead_seconds=301)
        self.assertEqual(m.required_budget_ns(100),30_000_000_150)


if __name__=='__main__':unittest.main()
