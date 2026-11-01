"""Synthetic SQLite/clock fixtures are never actual GPU evidence."""
import copy
from pathlib import Path
import sqlite3
import tempfile
import unittest

from tools.predictor_validation_20261027 import device_progress_cuda_api_probe_cpu_v1 as v


def sqlite_fixture(path):
    c=sqlite3.connect(path);c.executescript('''
    create table StringIds(id integer,value text);
    create table NVTX_EVENTS(start integer,end integer,text text,textId integer,globalTid integer);
    create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,nameId integer,globalTid integer,returnValue integer,correlationId integer);
    create table CUPTI_ACTIVITY_KIND_DRIVER(start integer,end integer,nameId integer,globalTid integer,returnValue integer,correlationId integer);
    ''')
    tid=(100<<24)+7
    c.executemany('insert into StringIds values (?,?)',[(1,'cudaEventRecord_v3020'),(2,'cudaEventQuery_v3020'),(3,'cudaEventElapsedTime_v3020'),(4,'cuEventRecord'),(5,'cudaEventCreateWithFlags_v3020'),(6,'GS_STAGE:r:gpu:prefill_0000')])
    c.execute('insert into NVTX_EVENTS values (100,200,null,6,?)',(tid,))
    c.executemany('insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?,?,?)',[(110,115,1,tid,0,1),(180,185,1,tid,0,2),(210,215,2,tid,600,3),(220,225,2,tid,0,4),(226,230,3,tid,0,5),(90,95,5,tid,0,6)])
    c.execute('insert into CUPTI_ACTIVITY_KIND_DRIVER values (111,114,4,?,0,1)',(tid,));c.commit();c.close()


def raw_fixture():
    return dict(worker_response=dict(semantic_forward_observations=dict(device_progress=dict(nvtx_host_brackets=[dict(name='GS_STAGE:r:gpu:prefill_0000',push_before_ns=1099,push_after_ns=1101,pop_before_ns=1199,pop_after_ns=1201)]))))


class NativeEventAPITests(unittest.TestCase):
    def test_native_runtime_and_driver_rows_preserve_return_and_thread(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'synthetic.sqlite';sqlite_fixture(path);doc=v.read_sqlite(path)
            self.assertEqual(len(doc['event_apis']),6)
            self.assertEqual([a['return_value'] for a in doc['event_apis'] if a['operation']=='Query'],[600,0])
            self.assertTrue(all(a['pid']==100 for a in doc['event_apis']))
            self.assertEqual(doc['nvtx_ranges'][0]['label'],'GS_STAGE:r:gpu:prefill_0000')
            report=v.record_containment(raw_fixture(),100,doc)
            self.assertTrue(report['per_forward_record_calls_authenticated'])
            self.assertFalse(report['event_handle_identity_authenticated']);self.assertFalse(report['query_and_elapsed_association_authenticated'])
    def test_dual_sided_clock_interval_and_wrong_PID_rejection(self):
        native=[dict(label='GS_STAGE:r:gpu:prefill_0000',pid=100,global_tid=1,native_start_ns=100,native_end_ns=200)]
        offset=v.conditional_nvtx_offset(raw_fixture(),100,native);self.assertEqual((offset['lower_ns'],offset['upper_ns']),(999,1001));self.assertTrue(offset['nonempty'])
        with self.assertRaises(ValueError):v.conditional_nvtx_offset(raw_fixture(),101,native)
        with self.assertRaises(ValueError):v.conditional_nvtx_offset(raw_fixture(),100,native+native)
    def test_empty_offset_interval_is_retained_not_fitted_away(self):
        raw=raw_fixture();raw['worker_response']['semantic_forward_observations']['device_progress']['nvtx_host_brackets'][0].update(pop_before_ns=1300,pop_after_ns=1301)
        native=[dict(label='GS_STAGE:r:gpu:prefill_0000',pid=100,global_tid=1,native_start_ns=100,native_end_ns=200)]
        result=v.conditional_nvtx_offset(raw,100,native);self.assertFalse(result['nonempty']);self.assertFalse(result['independent_clock_drift_bound'])
    def test_wrong_thread_failed_record_and_extra_record_cannot_authenticate(self):
        for mutation in ('thread','failure','extra'):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'synthetic.sqlite';sqlite_fixture(p);doc=v.read_sqlite(p)
                target=next(a for a in doc['event_apis'] if a['table'].endswith('RUNTIME') and a['operation']=='Record')
                if mutation=='thread':target['global_tid']+=1
                elif mutation=='failure':target['return_value']=1
                else:doc['event_apis'].append(copy.deepcopy(target))
                self.assertFalse(v.record_containment(raw_fixture(),100,doc)['per_forward_record_calls_authenticated'])


if __name__=='__main__':unittest.main()
