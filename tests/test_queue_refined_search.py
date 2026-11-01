import unittest
import json
from pathlib import Path
import tempfile
import io
from unittest.mock import patch
from tools.queue_refined_20261023.refined_search import MeanEnvironment,finite_candidates,summarize,ScreeningStudy,run_study,_pick,structural_groups,quotient_bounded_search,digest

class MeanModel:
    def __init__(self):self.calls=[]
    def predict(self,family,task,pool_id,context=None,quantile=.5):
        self.calls.append(dict(family=family,context=context,quantile=quantile))
        return dict(mean_ms=100. if family=='text' else 300.,conservative_ms=9000.,quantile_ms=9000.,evidence_status='FIXTURE')

class RefinedSearchTests(unittest.TestCase):
    def aliases(self):return {'text#0':dict(family='text',supported_tasks=['text']),'generalist_2#0':dict(family='generalist_2',supported_tasks=['text','image'])}
    def row(self,rid,arrival=0,task='text'):return dict(request_id=rid,arrival_ns=arrival,task=task,pool_id='x')
    def test_environment_uses_mean_not_policy_risk_and_fifo(self):
        model=MeanModel();env=MeanEnvironment(self.aliases(),model)
        env.enqueue(self.row('a'),'text#0');env.enqueue(self.row('b'),'text#0')
        env.advance(100_000_000)
        self.assertEqual(env.completed,['a']);self.assertEqual(env.results['a']['finish_ns'],100_000_000)
        env.advance(200_000_000);self.assertEqual(env.completed,['a','b'])
        self.assertEqual(env.results['b']['finish_ns'],200_000_000)
        self.assertEqual(env.results['a']['environment_service_ms'],100.)
        self.assertTrue(all(c['quantile']==.5 for c in model.calls))
    def test_environment_context_is_visible_before_current_enqueue(self):
        model=MeanModel();env=MeanEnvironment(self.aliases(),model)
        env.enqueue(self.row('bg',task='image'),'generalist_2#0')
        self.assertFalse(env.results['bg']['environment_context']['g2_visible_outstanding'])
        env.enqueue(self.row('a'),'text#0')
        self.assertTrue(env.results['a']['environment_context']['g2_visible_outstanding'])
        self.assertEqual(env.results['a']['environment_context']['other_active_bin'],'1')
        env.advance(1_000_000_000);self.assertFalse(any(v['outstanding'] for v in env.snapshot().values()))
    def test_grid_is_51_with_matched_mode_and_risk_controls(self):
        entries=[dict(label=f'x{i}',deployment={'text':2},model_sha256=str(i)) for i in range(3)]
        candidates=list(finite_candidates(entries));self.assertEqual(len(candidates),51)
        keys={(c['label'],c['strategy'],c['mode'],c['margin_ms'],c['risk_quantile']) for c in candidates};self.assertEqual(len(keys),51)
        for e in entries:
            self.assertEqual(sum(c['label']==e['label'] and c['allow_spill'] and c['mode']!='legacy' for c in candidates),12)
            self.assertEqual({c['risk_quantile'] for c in candidates if c['label']==e['label'] and c['strategy']=='calibrated_specialist_only'},{.5,.9})
    def test_summary_risk_control_matches_selected_mode(self):
        entry=dict(label='primary',deployment={'text':2,'generalist_2':1},model_sha256='abc')
        candidates=list(finite_candidates([entry]));rows=[]
        for i,c in enumerate(candidates):
            objective=-.1 if c['strategy']=='calibrated_global_incremental' and c['risk_quantile']==.5 else -.2-i*.001
            rows.append(dict(candidate=c,candidate_id=str(i),status='COMPLETE',objective=objective,runs=[]))
        result=summarize(rows);control=result['risk_matched_specialist_controls']['global_incremental']
        self.assertEqual(control['candidate']['risk_quantile'],.5);self.assertEqual(control['candidate']['mode'],'global_incremental');self.assertFalse(control['candidate']['allow_spill'])
        self.assertIsNone(result['formal_U'])

    def test_exact_max_anchored_tie_matches_shared_search(self):
        from tools.queue_refined_20261023.search import search
        rows=[dict(candidate_id='b',objective=-1.),dict(candidate_id='c',objective=-1.-.9e-12),dict(candidate_id='a',objective=-1.-1.8e-12)]
        self.assertEqual(_pick(rows)['candidate_id'],'b')
        candidates=[{'id':r['candidate_id']} for r in rows]
        values={r['candidate_id']:r['objective'] for r in rows}
        # Shared search tie uses hashed candidate IDs, so verify its selected
        # row is the minimum hash within the exact-maximum anchored band.
        from tools.queue_refined_20261023.search import identity
        result=search(candidates,[213],lambda c,s:dict(U=values[c['id']],eligible=True),lambda c:0.)
        expected=min(identity(c) for c in candidates if c['id'] in ('b','c'))
        self.assertEqual(result['winner_id'],expected)
    def test_own_mean_bound_valid_for_fifo_latencies(self):
        study=ScreeningStudy.__new__(ScreeningStudy);model=MeanModel()
        entry=dict(deployment={'text':1,'generalist_2':1},reference_routing={'text':{'text':1.}})
        study.models={'primary':model};study.bylabel={'primary':entry}
        study.catalog=dict(models={'text':dict(supported_tasks=['text']),'generalist_2':dict(supported_tasks=['text'])})
        study.config=dict(workload={'text':1.},latency_reference_ms={'text':100.},weights=dict(Q=1.,T=1.,L=1.,P=1.))
        study.cohorts={213:[self.row('a'),self.row('b')],214:[self.row('a'),self.row('b')]};study._own_floor_cache={};study._upper_cache={}
        candidate=dict(label='primary',allow_spill=True)
        upper=study.upper(candidate)
        self.assertEqual(upper,-2.)
        # Two 100ms FIFO requests have mean150,p95=195, so U=-3.45.
        self.assertGreaterEqual(upper,-3.45)
        self.assertEqual(set(study.own_mean_upper_by_seed(candidate)),{213,214})
        self.assertEqual(len(model.calls),12)
    def test_own_mean_bound_uses_environment_nanosecond_rounding(self):
        class BoundaryModel:
            def predict(self,*args,**kwargs):
                return dict(mean_ms=1.0000004,evidence_status='FIXTURE')
        model=BoundaryModel();aliases={'text#0':dict(family='text',supported_tasks=['text'])}
        request=self.row('a');env=MeanEnvironment(aliases,model)
        env.enqueue(request,'text#0');env.advance(2_000_000)
        actual_latency_ms=(env.results['a']['finish_ns']-request['arrival_ns'])/1e6
        self.assertEqual(actual_latency_ms,1.)
        study=ScreeningStudy.__new__(ScreeningStudy)
        study.models={'primary':model}
        study.bylabel={'primary':dict(deployment={'text':1},reference_routing={'text':{'text':1.}})}
        study.catalog=dict(models={'text':dict(supported_tasks=['text'])})
        study.config=dict(workload={'text':1.},latency_reference_ms={'text':1.},weights=dict(Q=1.,T=1.,L=1.,P=1.))
        study.cohorts={213:[request],214:[request]};study._own_floor_cache={};study._upper_cache={}
        upper=study.upper(dict(label='primary',allow_spill=True))
        actual_partial_objective=-2*actual_latency_ms
        self.assertEqual(upper,-2.)
        self.assertGreaterEqual(upper,actual_partial_objective)
        # The raw floating mean would have produced an invalid tighter bound.
        self.assertLess(-2*1.0000004,actual_partial_objective)
    def test_proved_quotient_reduces51_to39_and_preserves_full_ties(self):
        entries=[dict(label=label,deployment=x,model_sha256=label) for label,x in [
            ('primary',{'text':2,'generalist_2':1,'ocr':1}),('pure-specialist',{'text':2,'image':1,'ocr':1,'asr':1}),('other-mixed',{'text':2,'generalist_2':1,'image':1,'ocr':1})]]
        catalog=dict(models={m:dict(supported_tasks=['text'] if m=='text' else (['text','image','ocr','asr'] if m=='generalist_2' else [m])) for m in ('text','generalist_2','ocr','image','asr')})
        items=list(finite_candidates(entries));groups=structural_groups(items,catalog)
        self.assertEqual(len(groups),39);self.assertEqual(sum(len(g['members'])-1 for g in groups),12)
        self.assertEqual(sorted(len(g['members']) for g in groups if len(g['members'])>1),[4,4,4,4])
        for group in groups:
            self.assertEqual(group['representative_id'],min(group['member_ids']))
            if len(group['members'])>1:self.assertFalse(group['proof']['objective_used_to_form_group'])
        from tools.queue_refined_20261023.search import search
        calls=[]
        def evaluate(c,seed):
            calls.append((c,seed));return dict(U=-.5,eligible=True,seed=seed,decision_sha256='equal-fixture')
        full=search(items,[213,214],evaluate,lambda c:0.)
        calls.clear();log=io.StringIO();quotient=quotient_bounded_search(items,[213,214],evaluate,lambda c:0.,catalog,log=log)
        self.assertEqual(quotient['oracle_calls'],78);self.assertEqual(len(calls),78)
        self.assertEqual(quotient['winner_id'],full['winner_id']);self.assertEqual(quotient['tie_ids'],full['tie_ids'])
        self.assertEqual(len(quotient['tie_ids']),51)
        expanded=[json.loads(l) for l in log.getvalue().splitlines()]
        self.assertEqual(len(expanded),51)
        self.assertEqual(sum(r['objective_inherited_from_proved_equivalent'] for r in expanded),12)
        self.assertEqual(quotient['objective_cache_hits'],0)
        self.assertGreaterEqual(quotient['elapsed_seconds'],quotient['group_construction_seconds'])

    def test_study_end_to_end_domain_shortlist_and_audit_wiring(self):
        class FakeStudy:
            def __init__(self,base,specs,rate,seeds):
                self.config=dict(workload={'text':rate})
                self.catalog=dict(models={m:dict(supported_tasks=['text'] if m=='text' else (['text','image','ocr','asr'] if m=='generalist_2' else [m])) for m in ('text','generalist_2','ocr','image','asr')})
                self.entries=[dict(label=label,deployment=x,model_sha256=str(i),model_path=specs[i]['model_path']) for i,(label,x) in enumerate([
                    ('primary',{'text':2,'generalist_2':1,'ocr':1}),('pure-specialist',{'text':2,'image':1,'ocr':1,'asr':1}),('other-mixed',{'text':2,'generalist_2':1,'image':1,'ocr':1})])]
            def evaluate(self,c,seed):
                effective_margin=c['margin_ms'] if 'generalist_2' in c['deployment'] else 0.
                value=-.1-effective_margin/10000-(.01 if c['mode']=='global_incremental' else .02)
                spill=c['allow_spill'] and 'generalist_2' in c['deployment']
                return dict(seed=seed,U=value,eligible=True,spilled_text_requests=int(spill),decision_sha256=digest([c['label'],c['mode'],c['risk_quantile'],effective_margin,spill]),formal_U=None)
            def upper(self,c):return 0.
            def own_mean_upper_by_seed(self,c):return {213:0.,214:0.}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'runtime-inputs.jsonl').write_text('{}\n')
            base=root/'base.json';base.write_text(json.dumps(dict(inputs=tmp)))
            models=[]
            for i in range(3):
                path=root/f'model{i}.json';path.write_text('{}');models.append(dict(label=str(i),model_path=str(path)))
            manifest=root/'models.json';manifest.write_text(json.dumps(dict(models=models)))
            with patch('tools.queue_refined_20261023.refined_search.ScreeningStudy',FakeStudy):run_study(root/'out',base,manifest,loads=[1.85])
            report=json.loads((root/'out/text1.85/report.json').read_text())
            self.assertEqual(report['independent_identity_count'],51)
            self.assertEqual(report['numeric_bound_leaf_checks'],51)
            self.assertTrue(report['same_winner']);self.assertEqual(report['bound_violations'],0)
            shortlist=report['comparison']['native_shortlist']
            self.assertTrue(all(key in shortlist for key in ('calibrated_spill','risk_matched_no_spill','legacy_latency','strong_other_deployment')))
            self.assertEqual(shortlist['calibrated_spill']['candidate']['mode'],shortlist['risk_matched_no_spill']['candidate']['mode'])
            self.assertEqual(shortlist['calibrated_spill']['candidate']['risk_quantile'],shortlist['risk_matched_no_spill']['candidate']['risk_quantile'])

if __name__=='__main__':unittest.main()
