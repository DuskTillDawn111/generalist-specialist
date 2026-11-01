import copy
import unittest

from tests import test_twoFIFO_owned_training_review_v2 as fixtures
from tools.predictor_validation_20261027 import twoFIFO_owned_training_review_v3 as review
from tools.predictor_validation_20261027.twoFIFO_owned_collect_v3 import ACTIVITY_SCHEMA,ACTIVITY_FILE,TIME_DOMAIN


def cohort_fixture():
    plan,raw=fixtures.fixture()
    plan['schema']=review.PLAN_SCHEMA;raw['schema']=review.RAW_SCHEMA
    return plan,raw


def fixture():
    plan,raw=fixtures.fixture();plan['schema']=review.PLAN_SCHEMA;raw['schema']=review.RAW_SCHEMA
    case=plan['cases'][0];epoch=raw['cases'][0]['epoch_ns'];deadline=raw['cases'][0]['fixed_drain_deadline_ns']
    spans=[]
    def add(kind,stage,at,end,cid=None):
        row=dict(span_id=len(spans),kind=kind,case_id=cid,phase='CPU_SYNTHETIC',stage=stage,
                 time_domain=TIME_DOMAIN,start_ns=at,end_ns=end)
        spans.append(row)
    def checkpoint(stage,at,cid=None):
        add('checkpoint',stage,at,at+6,cid)
        add('checkpoint_locked_snapshot',stage,at+1,at+2,cid)
        add('checkpoint_serialization_and_atomic_save',stage,at+3,at+5,cid)
    checkpoint('before_startup',10);checkpoint('postpriming',25)
    add('precase_history_snapshot',None,40,50,case['case_id'])
    checkpoint('precase',60,case['case_id']);checkpoint('postdrain',deadline+10,case['case_id'])
    checkpoint('shutdown',deadline+100)
    raw.update(no_mutating_checkpoint_during_case=True,checkpoint_spans_file=ACTIVITY_FILE,
        checkpoint_spans_schema=ACTIVITY_SCHEMA,checkpoint_spans_execution_id=plan['execution_id'],
        checkpoint_time_domain=TIME_DOMAIN,public_case_calendars=copy.deepcopy(plan['cases']),
        priming_public_calendar=copy.deepcopy(plan['priming']),
        measurement_windows=[dict(case_id=case['case_id'],start_ns=epoch,end_ns=epoch+review.MEASUREMENT_NS,
            fixed_drain_deadline_ns=deadline)],
        case_histories=[dict(case_id=case['case_id'],prior_history_snapshot=dict(snapshot_time_ns=45))],
        case_preparations=[dict(case_id=case['case_id'],public_arrivals_sha256=case['public_arrivals_sha256'],
            public_arrivals=copy.deepcopy(case['arrivals']),prior_history_index=0,prepared_before_epoch=True,
            durable_checkpoint_return_ns=67)])
    ledger=dict(schema=ACTIVITY_SCHEMA,execution_id=plan['execution_id'],time_domain=TIME_DOMAIN,
        checkpoint_spans=spans,all_spans_complete=True)
    return plan,raw,ledger


class CleanActivityReviewTests(unittest.TestCase):
    def test_complete_nonintersecting_operations_and_unaltered_sender_cohort(self):
        plan,raw,ledger=fixture();before=copy.deepcopy((plan,raw,ledger))
        out=review.inspect_checkpoint_activity(plan,raw,ledger)
        self.assertEqual((plan,raw,ledger),before)
        self.assertEqual(out['checkpoints'],5);self.assertEqual(out['critical_interval_overlaps'],0)
        self.assertEqual(review.inspect_cohort(plan,raw)['issues'],[])
        self.assertFalse(out['zero_observation_overhead_proven'])

    def test_operations_at_epoch_inside_measurement_or_drain_endpoint_rejected(self):
        for at in (100000,100001,100000+review.MEASUREMENT_NS,
                   100000+review.MEASUREMENT_NS+review.DRAIN_NS):
            plan,raw,ledger=fixture();s=ledger['checkpoint_spans'][1];s.update(start_ns=at,end_ns=at)
            with self.subTest(at=at),self.assertRaisesRegex(ValueError,'intersect'):
                review.inspect_checkpoint_activity(plan,raw,ledger)

    def test_ledger_omission_nested_span_forgery_and_sidecar_identity_fail(self):
        for mutation in ('missing','unfinished','childoutside','wrongexecution','wrongfile','falseclaim'):
            plan,raw,ledger=fixture()
            if mutation=='missing':ledger['checkpoint_spans'].pop(1)
            elif mutation=='unfinished':ledger['checkpoint_spans'][1]['end_ns']=None
            elif mutation=='childoutside':ledger['checkpoint_spans'][1]['end_ns']=24
            elif mutation=='wrongexecution':ledger['execution_id']='another'
            elif mutation=='wrongfile':raw['checkpoint_spans_file']='../other.json'
            else:raw['no_mutating_checkpoint_during_case']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                review.inspect_checkpoint_activity(plan,raw,ledger)

    def test_public_calendar_durable_preparation_and_snapshot_future_rejected(self):
        for mutation in ('calendar','prime','payload','future','notdurable','omittedpreparation'):
            plan,raw,ledger=fixture()
            if mutation=='calendar':raw['public_case_calendars'][0]['arrivals'].pop()
            elif mutation=='prime':raw['priming_public_calendar'].pop()
            elif mutation=='payload':raw['case_preparations'][0]['public_arrivals'].pop()
            elif mutation=='future':raw['case_histories'][0]['prior_history_snapshot']['snapshot_time_ns']=100001
            elif mutation=='notdurable':raw['case_preparations'][0]['durable_checkpoint_return_ns']=100000
            else:raw['case_preparations'].clear()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                review.inspect_checkpoint_activity(plan,raw,ledger)


if __name__=='__main__':unittest.main()
