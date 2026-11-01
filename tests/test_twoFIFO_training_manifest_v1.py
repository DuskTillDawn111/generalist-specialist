import copy
import unittest

from tools.predictor_validation_20261027 import twoFIFO_training_manifest_v1 as module


def fixture():
    inputs={task:{size:dict(pool_id=task+':'+size,task=task) for size in ('short','medium','long')} for task in ('text','ocr')}
    cases=module.cases(inputs)
    for case in cases:
        for i,row in enumerate(case['arrivals']):row.update(request_id=case['case_id']+':'+str(i),arrival_index=i)
        case.update(offered=len(case['arrivals']),public_arrivals_sha256=module.digest(case['arrivals']))
    manifest=dict(measurement_ns=module.MEASUREMENT_NS,drain_ns=module.DRAIN_NS,capture_blocks=[dict(cases=cases)])
    pool={row['pool_id']:dict(sample={'id':row['pool_id']}) for group in inputs.values() for row in group.values()}
    return manifest,cases,pool


class Clock:
    def __init__(self):self.now=100
    def clock(self):return self.now
    def wait(self,deadline):self.now=max(self.now,deadline)


class Runtime:
    def __init__(self,clock,*,fail=False,late=False):self.clock=clock;self.rows=[];self.timeout=None;self.fail=fail;self.late=late
    def enqueue(self,row,sample):
        self.rows.append(row)
        if self.fail and len(self.rows)==1:raise RuntimeError('injected enqueue error')
        row['status']='submitted'
        if self.late:self.clock.now=80_000_000_100
    def drain(self,rows,timeout):
        self.timeout=timeout
        for row in rows:
            if row['status']=='submitted':row.update(status='completed',completion_time_ns=self.clock.now)


class FIFOManifestTests(unittest.TestCase):
    def test_fixed_minimum_matrix_has_lengths_bursts_and_mixed_owner_counts(self):
        _,cases,_=fixture()
        self.assertEqual(len(cases),16)
        self.assertEqual(sum(case['offered'] for case in cases),100)
        for owner in module.OWNERS:
            own=[case for case in cases if case['case_id']=='HOL-'+owner.replace('#','-')][0]
            self.assertEqual(len(own['arrivals']),6)
            self.assertEqual([r['pool_id'].split(':')[-1] for r in own['arrivals'][:3]],['long','short','medium'])
        self.assertEqual(max(len({r['owner'] for r in case['arrivals']}) for case in cases),4)
        self.assertEqual(len(next(c for c in cases if c['case_id']=='sender-pressure-text-0')['arrivals']),18)

    def dispatch(self,*,fail=False,late=False):
        manifest,cases,pool=fixture();case=next(c for c in cases if c['case_id']=='HOL-text-0')
        clock=Clock();runtime=Runtime(clock,fail=fail,late=late);snapshots=[]
        result=module.dispatch_case(manifest,case,runtime,pool,epoch_ns=100,clock_ns=clock.clock,
            wait_until_ns=clock.wait,checkpoint=lambda raw:snapshots.append(copy.deepcopy(raw)))
        return result,runtime,snapshots,clock

    def test_all_offered_checkpointed_before_enqueue_no_response_gate_fixed_drain(self):
        result,runtime,snapshots,clock=self.dispatch()
        self.assertEqual(len(snapshots[0]['request_records']),6)
        self.assertTrue(all(r['status']=='planned' for r in snapshots[0]['request_records']))
        self.assertEqual(len(runtime.rows),6)
        self.assertEqual(runtime.timeout,60)
        self.assertEqual(clock.now,70_000_000_100)
        self.assertTrue(all(r['RESULT_reader_within_fixed_drain'] for r in result['request_records']))

    def test_enqueue_error_remains_offered_and_later_requests_continue(self):
        result,runtime,_,_=self.dispatch(fail=True)
        self.assertEqual(result['offered'],6)
        self.assertEqual(len(runtime.rows),6)
        self.assertEqual(result['request_records'][0]['status'],'error')
        self.assertEqual(result['request_records'][-1]['status'],'completed')

    def test_late_dispatch_does_not_extend_drain_or_delete_unsubmitted_offered(self):
        result,runtime,_,_=self.dispatch(late=True)
        self.assertEqual(len(runtime.rows),1)
        self.assertEqual(len(result['request_records']),6)
        self.assertEqual(runtime.timeout,0)
        self.assertEqual(result['fixed_drain_deadline_ns'],70_000_000_100)
        self.assertTrue(all(row['status']=='incomplete' for row in result['request_records'][1:]))
        self.assertFalse(result['request_records'][0]['RESULT_reader_within_fixed_drain'])

    def test_altered_case_or_arrival_fingerprint_rejected(self):
        manifest,cases,pool=fixture();case=copy.deepcopy(cases[0]);case['arrivals'][0]['arrival_ns']=1
        clock=Clock()
        with self.assertRaisesRegex(ValueError,'Exact complete prepared case'):
            module.dispatch_case(manifest,case,Runtime(clock),pool,epoch_ns=100,
                clock_ns=clock.clock,wait_until_ns=clock.wait,checkpoint=lambda raw:None)


if __name__=='__main__':unittest.main()
