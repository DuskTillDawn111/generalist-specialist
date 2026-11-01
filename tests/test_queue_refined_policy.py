"""Causal routing decisions and global pending-work externality diagnostics."""
import copy
import unittest
from tools.queue_refined_20261023.refined_policy import RefinedPolicy

class Model:
    def __init__(self,*,generalist=60.,interference=1.,residual=100.):
        self.generalist=generalist;self.interference=interference;self.tail=residual;self.calls=[]
    def predict(self,family,task,pool_id,context=None,quantile=.9):
        self.calls.append(('predict',family,context,quantile))
        if pool_id=='unknown':return {'conservative_ms':None,'evidence_status':'UNKNOWN'}
        value=self.generalist if family=='generalist_2' else 100.
        if family=='text' and context and context['g2_visible_outstanding']:value*=self.interference
        return dict(mean_ms=value,quantile_ms=value,conservative_ms=value,evidence_status='FIXTURE',support=1)
    def residual(self,family,task,pool_id,age_ms,context=None,quantile=.9):
        self.calls.append(('residual',family,age_ms,context,quantile))
        value=self.tail
        if family=='text' and context and context['g2_visible_outstanding']:value*=self.interference
        return dict(conditional_mean_ms=value,quantile_ms=value,conservative_ms=value,evidence_status='CONDITIONAL_FIXTURE')

class RefinedTests(unittest.TestCase):
    def aliases(self,n=1):
        return {**{f'text#{i}':dict(family='text',supported_tasks=['text']) for i in range(n)},
                'generalist_2#0':dict(family='generalist_2',supported_tasks=['text','image'])}
    def state(self,n=1):return {a:dict(outstanding=[],last_completed_ns=None) for a in self.aliases(n)}
    def row(self,rid='r',task='text',pool='x',enqueue=0,submit=0):
        return dict(request_id=rid,task=task,pool_id=pool,enqueued_ns=enqueue,actual_submit_ns=submit)
    def policy(self,*,mode='conservative',n=1,margin=0.,model=None,quality=(0.,0.),**kwargs):
        return RefinedPolicy(self.aliases(n),model or Model(),{'text':{'text':quality[0]},'generalist_2':{'text':quality[1],'image':0.}},
            dict(Q=1.,L=1.,P=1.,E=1.),{'text':100.,'image':100.},{'text':1.,'image':.01},mode=mode,margin_ms=margin,**kwargs)
    def test_margin_selects_specialist_and_never_forces_spill(self):
        state=self.state();self.assertEqual(self.policy().choose('text','x',0,state)['alias'],'generalist_2#0')
        self.assertEqual(self.policy(margin=250.).choose('text','x',0,state)['alias'],'text#0')
    def test_specialist_balancing_before_spill(self):
        state=self.state(2);state['text#0']['outstanding']=[self.row()]
        self.assertEqual(self.policy(n=2,model=Model(generalist=500.)).choose('text','x',0,state)['alias'],'text#1')
    def test_quality_is_cost_not_absolute_admission(self):
        result=self.policy(quality=(0.,1.)).choose('text','x',0,self.state())
        self.assertEqual(result['alias'],'text#0');self.assertFalse(result['absolute_quality_gate'])
    def test_conditional_head_residual_does_not_expire_to_five_ms(self):
        state=self.state();state['text#0']['outstanding']=[self.row()];m=Model(residual=600.)
        result=self.policy(model=m).choose('text','x',2_000_000_000,state)
        candidate=next(c for c in result['audit'] if c['alias']=='text#0')
        self.assertEqual(candidate['predicted_waiting_ms'],600.)
        self.assertTrue(any(c[0]=='residual' and c[2]==2000. for c in m.calls))
    def test_global_proxy_includes_existing_pending_interference(self):
        state=self.state();state['text#0']['outstanding']=[self.row('a'),self.row('b',submit=None)]
        local=self.policy(model=Model(interference=5.)).choose('text','x',0,state)
        global_=self.policy(mode='global_incremental',model=Model(interference=5.)).choose('text','x',0,state)
        self.assertEqual(local['alias'],'generalist_2#0');self.assertEqual(global_['alias'],'text#0')
        g=next(c for c in global_['audit'] if c['family']=='generalist_2')
        self.assertGreater(g['score_contributions']['L'],0.);self.assertIsNone(global_['full_U'])
        self.assertEqual(g['same_resident_memory_delta'],0.);self.assertFalse(g['energy_proxy_known'])
    def test_generalist_background_is_counted(self):
        state=self.state();state['generalist_2#0']['outstanding']=[self.row('a','image')]
        result=self.policy(model=Model(generalist=120.,residual=800.)).choose('text','x',0,state)
        self.assertEqual(result['alias'],'text#0')
        g=next(c for c in result['audit'] if c['family']=='generalist_2');self.assertEqual(g['predicted_waiting_ms'],800.)
    def test_energy_proxy_normalized_and_audited(self):
        result=self.policy(mode='global_incremental',energy_proxy_joules_per_service_ms={'text':{'text':.1},'generalist_2':{'text':.3,'image':.3}}).choose('text','x',0,self.state())
        self.assertTrue(all(c['energy_proxy_known'] for c in result['audit']))
        self.assertTrue(all(c['score_contributions']['E'] is not None for c in result['audit']))
    def test_future_service_answer_fields_have_no_effect_and_snapshot_unchanged(self):
        state=self.state();state['text#0']['outstanding']=[self.row()];before=copy.deepcopy(state)
        for mode in ('conservative','global_incremental'):
            expected=self.policy(mode=mode).choose('text','x',0,state);changed=copy.deepcopy(state)
            for v in changed.values():
                v.update(answer='SECRET',true_finish_ns=1)
                for r in v['outstanding']:r.update(actual_service_ms=1e9,reference='SECRET',future_finish_ns=1)
            changed['generalist_2#0']['outstanding'].append(self.row('future','unknown','unknown',100,100))
            self.assertEqual(expected,self.policy(mode=mode).choose('text','x',0,changed))
        self.assertEqual(state,before)
    def test_invalid_causal_timestamp_returns_unknown(self):
        state=self.state();state['text#0']['outstanding']=[self.row(submit=1)]
        result=self.policy().choose('text','x',0,state);self.assertIsNone(result['alias']);self.assertIn('INVALID_SUBMIT_NS',result['unknown_reasons'])
    def test_unknown_service_is_not_zero(self):
        result=self.policy().choose('text','unknown',0,self.state());self.assertIsNone(result['alias'])
    def test_risk_quantile_and_context_switch_forwarded(self):
        m=Model();self.policy(model=m,risk_quantile=.8,concurrency_context=False).choose('text','x',0,self.state())
        self.assertTrue(all(c[2] is None and c[3]==.8 for c in m.calls if c[0]=='predict'))
    def test_real_service_model_api_and_tail(self):
        from tools.queue_refined_20261023.calibration import fit,ServiceModel
        records=[]
        for family in ('text','generalist_2'):
            for i,value in enumerate([100.,200.,800.]):
                records.append(dict(family=family,task='text',pool_id='x',context=dict(other_active_bin='0',g2_visible_outstanding=False),service_ms=value,seed=113,raw_sha256='fixture',deployment={'text':1,'generalist_2':1}))
        model=ServiceModel(fit(records));state=self.state();state['text#0']['outstanding']=[self.row()]
        result=self.policy(model=model).choose('text','x',2_000_000_000,state)
        candidate=next(c for c in result['audit'] if c['family']=='text')
        self.assertGreater(candidate['predicted_waiting_ms'],5.)
        head=candidate['forecast']['aliases']['text#0']['requests'][0]
        self.assertEqual(head['service_evidence']['model']['evidence_status'],'TAIL_EXTRAPOLATION_UNVERIFIED')
    def test_new_fast_request_cannot_dilute_existing_cost(self):
        state=self.state();state['text#0']['outstanding']=[self.row('old')]
        policy=self.policy(mode='global_incremental',model=Model(generalist=1.,residual=1000.),quality=(1.,0.))
        result=policy.choose('text','x',0,state)
        g=next(c for c in result['audit'] if c['family']=='generalist_2')
        # Old pending mean would drop from1000 to500.5 and Q from1 to.5;
        # fixed expected-window sums preserve existing cost instead.
        self.assertGreater(g['score_contributions']['L'],0.)
        self.assertEqual(g['score_contributions']['Q'],0.)
        self.assertGreaterEqual(g['score_contributions']['P'],0.)
        self.assertGreater(g['score'],0.)
        self.assertEqual(g['forecast']['per_task']['text']['expected_window_requests'],600.)
        self.assertEqual(g['forecast']['per_task']['text']['fixed_tail_slots'],30)
    def test_fixed_tail_proxy_append_monotone_across_quantile_jump(self):
        state=self.state();state['text#0']['outstanding']=[self.row('old')]
        result=self.policy(mode='global_incremental',model=Model(generalist=1.,residual=1000.)).choose('text','x',0,state)
        for candidate in result['audit']:
            self.assertGreaterEqual(candidate['score_contributions']['P'],0.)
        self.assertIn('not whole-window P95',result['P_scope'])
    def test_global_margin_has_fixed_window_scale(self):
        state=self.state();base=self.policy(mode='global_incremental',margin=0.).choose('text','x',0,state)
        margin=self.policy(mode='global_incremental',margin=250.).choose('text','x',0,state)
        a=next(c for c in base['audit'] if c['family']=='generalist_2')
        b=next(c for c in margin['audit'] if c['family']=='generalist_2')
        self.assertAlmostEqual(b['score']-a['score'],(1/1.01)*250/100/600)
    def test_frozen_cache_warmup_needs_no_trace_or_future(self):
        model=Model();policy=self.policy(model=model)
        result=policy.warm_cache({'text':['x'],'image':['i']})
        self.assertEqual(result['frozen_path_inputs'],3)
        self.assertEqual(result['known_predictions'],18)
        self.assertFalse(result['future_arrivals_read'])
        self.assertTrue(all(c[0]=='predict' for c in model.calls))
        self.assertEqual(policy.choose('text','x',0,self.state())['alias'],'generalist_2#0')
    def test_specialist_control_retains_full_background_context(self):
        state=self.state(2);state['text#0']['outstanding']=[self.row('a')]
        state['generalist_2#0']['outstanding']=[self.row('bg','image')]
        model=Model(generalist=1.,interference=5.)
        result=self.policy(n=2,model=model,allow_spill=False).choose('text','x',0,state)
        self.assertEqual(result['alias'],'text#1')
        g=next(c for c in result['audit'] if c['family']=='generalist_2')
        self.assertEqual(g['status'],'SPILL_DISABLED_CONTROL')
        self.assertTrue(any(c[2] and c[2]['g2_visible_outstanding'] for c in model.calls if c[0]=='predict' and c[1]=='text'))
    def test_model_validates_full_resident_aliases(self):
        class ValidatingModel(Model):
            def validate_aliases(self,aliases):self.validated=sorted(aliases)
        model=ValidatingModel();self.policy(model=model,allow_spill=False)
        self.assertEqual(model.validated,['generalist_2#0','text#0'])
    def test_guard_wrapper_compatible(self):
        from tools.queue_refined_20261023.guarded_policy import RateLimitedPolicy
        base=self.policy();policy=RateLimitedPolicy(base,dict(schema='queue_aware_token_bucket_v1',task='text',family='generalist_2',rate_rps=.1,burst=1.))
        self.assertEqual(policy.choose('text','x',0,self.state())['alias'],'generalist_2#0')
        self.assertEqual(policy.choose('text','x',0,self.state())['alias'],'text#0')

if __name__=='__main__':unittest.main()
