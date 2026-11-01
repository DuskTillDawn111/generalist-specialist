"""Mock/tamper evidence fixtures only: no native GPU observations are invented."""
import copy
import unittest

from tests.test_queue_aware_admission import fixture
from tools.queue_aware_20261023.protocol import make_dispatcher
from tools.queue_aware_20261023.admission import policy_identity,measurement_identity,digest
from tools.queue_aware_20261023.evaluation import audit_phase,evaluate_run,decision_digest,_API

def bind(raw,c):
    raw['policy_identity']=policy_identity(raw['policy_contract'])
    raw['measurement_window']['dispatch_decisions_sha256']=decision_digest(raw['request_records'])
    identity=measurement_identity(c,raw);raw['measurement_identity']=identity
    raw['memory_evidence']['measurement_identity']=identity
    raw['energy_evidence'].update(measurement_identity=identity,profile_key=identity)

def measured_fixture():
    c,p,plan=fixture(text_only=True);samples=c.catalog['quality']['identity']['source']+'/runtime-inputs.jsonl'
    start=1_000_000_000;end=start+round(c.horizon_seconds*1e9);trace=_API['trace'].make_trace(c.workload,4,113,c.arrival_process)
    pool=_API['workload'].SamplePool(samples,list(c.workload),113);dispatcher=make_dispatcher(p);rows=[];observed=set();free={a:start for a in p['aliases']}
    for i,(offset,task) in enumerate(trace):
        item=pool.next(task);now=start+round(offset*1e9)+100;new=[];snapshot={a:dict(outstanding=[],last_completed_ns=None) for a in p['aliases']}
        for r in rows:
            a=r['model']
            if r['completion_time_ns']<=now:
                if r['request_id'] not in observed:new.append(dict(request_id=r['request_id'],model=a,completion_time_ns=r['completion_time_ns']));observed.add(r['request_id'])
                previous=snapshot[a]['last_completed_ns'];snapshot[a]['last_completed_ns']=max(previous or 0,r['completion_time_ns'])
            else:snapshot[a]['outstanding'].append(dict(request_id=r['request_id'],task=task,pool_id=r['pool_id'],enqueued_ns=r['enqueued_time_ns'],actual_submit_ns=r['actual_submit_time_ns']))
        for state in snapshot.values():state['outstanding'].sort(key=lambda r:(r['enqueued_ns'],r['request_id']))
        d=dispatcher.choose(task,item['pool_id'],now,snapshot);a=d['alias'];submit=now+20;finish=max(submit,free[a])+800_000_000;free[a]=finish
        rows.append(dict(request_id=f'measurement:{i}',phase='measurement',arrival_index=i,task=task,model=a,pool_id=item['pool_id'],
            sample_id=item['sample'].get('id'),scheduled_time_ns=start+round(offset*1e9),enqueued_time_ns=now+10,actual_submit_time_ns=submit,
            completion_time_ns=finish,status='completed',worker_latency_ms=800.,worker_queue_wait_ms=0.,
            worker_response=dict(task_type=task,success=True,prediction='deliberately wrong fixture answer'),
            routing_decision=dict(decision_time_ns=now,snapshot=copy.deepcopy(snapshot),decision=d,newly_observed_completions=new)))
    quality=c.catalog['quality'];pairs=[]
    for m in ['text','generalist_2']:
        pairs.append(dict(model=m,task='text',measurement=quality['paths'][m]['text'],runtime_checkpoint_sha256='fixture-runtime-'+m,offline_full_checkpoint_equivalence_verified=False))
    raw=dict(status='completed',backend='native',strategy=p['strategy'],policy_contract=p,plan=plan,replica_to_family={a:s['family'] for a,s in p['aliases'].items()},
        samples_path=samples,transport_config=dict(random_seed=113,arrival_process=c.arrival_process),request_records=rows,warmup_request_records=[],priming_request_records=[],
        preflight=dict(sample_pool_sha256=digest(samples)),admission_evidence=dict(verified=True,contract='quality_evidence_v1',manifest_sha256='fixture-only',reports=[{'fixture':True}],
            quality_sha256=_API['quality'].quality_digest(quality),checkpoints={m:'fixture-runtime-'+m for m in ['text','generalist_2']},pairs=pairs),
        measurement_window=dict(start_time_ns=start,end_time_ns=end,drain_end_time_ns=max(end,max(r['completion_time_ns'] for r in rows))+1,
            duration_seconds=4.,offered_requests=len(rows),trace_sha256=_API['trace'].trace_digest(trace)),
        memory_evidence=dict(measured=True,resource_valid=True,safe_memory_mib=c.gpu_memory_mib*.5,scope='synthetic test fixture only'),
        energy_evidence=dict(kind='measured',scope='whole_gpu_board',coverage_valid=True,run_joules=400.,startup_joules=100.))
    bind(raw,c);return c,raw

class DynamicEvaluationTests(unittest.TestCase):
    def test_complete_fixture_scores_original_six_terms_without_quality_threshold(self):
        c,r=measured_fixture();actual=evaluate_run(c,r);u=actual['utility']
        self.assertTrue(actual['dynamic_audit']['verified'],actual)
        self.assertTrue(u['eligible'],actual)
        self.assertEqual(u['Q_norm'],1.) # deliberately bad precision is scored, not rejected
        self.assertAlmostEqual(u['U'],-sum(c.weights[t]*u[t+'_norm'] for t in 'QTLPME'),places=12)
        self.assertEqual(actual['stability']['thresholds'],__import__('tools.queue_aware_20261023.evaluation',fromlist=['DEFAULT_THRESHOLDS']).DEFAULT_THRESHOLDS)

    def test_alias_or_snapshot_tamper_fails_even_after_rehash(self):
        for tamper in ['alias','snapshot','input','completion','bucket']:
            c,r=measured_fixture();row=r['request_records'][1]
            if tamper=='alias':row['model']='generalist_2#0' if row['model']!='generalist_2#0' else 'text#0'
            elif tamper=='snapshot':
                row['routing_decision']['snapshot']['generalist_2#0']['last_completed_ns']=1
            elif tamper=='input':row['pool_id']='forged-input'
            elif tamper=='completion':row['routing_decision']['newly_observed_completions']=[dict(request_id='measurement:999',model='text#0',completion_time_ns=1)]
            else:row['routing_decision']['decision']['rate_guard']['bucket_after_tokens']=1000.
            bind(r,c);actual=evaluate_run(c,r)
            self.assertFalse(actual['utility']['eligible'],(tamper,actual));self.assertIsNone(actual['utility']['U'])

    def test_missing_memory_energy_or_guard_remains_unknown(self):
        for missing in ['memory','energy','guard']:
            c,r=measured_fixture()
            if missing=='memory':r['memory_evidence']['measured']=False
            elif missing=='energy':r['energy_evidence']['coverage_valid']=False
            else:r['policy_contract']['guard']=None;bind(r,c)
            actual=evaluate_run(c,r);self.assertFalse(actual['utility']['eligible']);self.assertIsNone(actual['utility']['U'])
            if missing in ['memory','energy']:self.assertIsNone(actual['utility'][{'memory':'M_norm','energy':'E_norm'}[missing]])

    def test_resource_identity_cannot_reuse_static_or_other_policy_key(self):
        c,r=measured_fixture();r['energy_evidence']['profile_key']='static-key'
        actual=evaluate_run(c,r);self.assertFalse(actual['utility']['energy_verified']);self.assertIsNone(actual['utility']['U'])
        c,r=measured_fixture();r['policy_contract']['service_ms']['text']['text'][next(iter(r['policy_contract']['service_ms']['text']['text']))]*=2
        self.assertFalse(evaluate_run(c,r)['utility']['cohort_verified'])

    def test_mock_without_native_preflight_preserves_stability_but_not_U(self):
        c,r=measured_fixture();r['backend']='mock';r['preflight']=None
        result=evaluate_run(c,r)
        self.assertFalse(result['utility']['eligible']);self.assertIsNone(result['utility']['U'])
        self.assertIsNone(result['utility']['Q_norm'])
        self.assertNotEqual(result['stability']['stability_state_v2'],'invalid_observation')

if __name__=='__main__':unittest.main()
