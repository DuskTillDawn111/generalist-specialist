import json
import copy
from fractions import Fraction
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.queue_refined_weight_sensitivity_20261023 import (
    TERMS,ROLES,weights,utility,coefficients,delta_value,axis_interval,region,
    point_summary,rank_development,load_completed,analyze,receipt,identity)


def terms(**changes):return {t:changes.get(t,1.) for t in TERMS}

def pair(seed,coef):return dict(seed=seed,coefficients=coef)

class WeightSensitivityTests(unittest.TestCase):
    def test_explicit_normalization_and_delta_matches_two_utilities(self):
        a=terms(Q=.4,T=.1,L=3.,P=4.,M=.8,E=.9);b=terms(Q=.5,T=.2,L=1.,P=2.,M=.9,E=1.)
        c=coefficients(a,b)
        for l,p in [(0.,0.),(.25,.75),(1.,1.)]:
            w=weights(l,p);self.assertAlmostEqual(sum(w.values()),1.)
            self.assertEqual(w['Q'],w['T']);self.assertEqual(w['M'],w['E'])
            self.assertAlmostEqual(delta_value(c,l,p),utility(a,l,p)-utility(b,l,p))
        self.assertEqual(weights(0.,0.)['L'],0.)
        self.assertAlmostEqual(utility(a,1.,1.),-sum(a.values())/6)
        with self.assertRaises(ValueError):weights(-.1,1.)
    def test_break_even_strict_intervals_for_single_and_joint_axes(self):
        c=dict(A=1.,B=-2.,C=0.)
        interval=axis_interval([c],'L_only')
        self.assertEqual(interval['status'],'NONEMPTY');self.assertEqual(interval['lower'],0.)
        self.assertEqual(interval['upper'],.5);self.assertTrue(interval['lower_included']);self.assertFalse(interval['upper_included'])
        self.assertEqual(axis_interval([c],'P_only')['status'],'EMPTY')
        self.assertEqual(axis_interval([c],'both')['upper'],.5)
        c2=dict(A=-.25,B=1.,C=0.)
        interval=axis_interval([c,c2],'L_only')
        self.assertEqual((interval['lower'],interval['upper']),(.25,.5))
        self.assertFalse(interval['lower_included'])
    def test_analytic_halfplanes_find_region_and_detect_infeasibility(self):
        cs=[dict(A=1.,B=-1.,C=-1.),dict(A=-.2,B=1.,C=0.)]
        r=region(cs);self.assertEqual(r['status'],'NONEMPTY')
        l,p=r['strict_witness'];self.assertTrue(l>.2 and l+p<1.)
        impossible=region([dict(A=-.8,B=1.,C=0.),dict(A=.2,B=-1.,C=0.)])
        self.assertEqual(impossible['status'],'EMPTY')
        # Closed feasibility at L=0 does not satisfy the strict inequality -L>0.
        self.assertEqual(region([dict(A=0.,B=-1.,C=0.)])['status'],'EMPTY_STRICT_INTERIOR')
        self.assertEqual(region([dict(A=0.,B=0.,C=0.)])['status'],'EMPTY_STRICT_INTERIOR')
    def test_unknown_never_imputed_even_when_new_weight_zero(self):
        missing=terms(L=None)
        self.assertIsNone(utility(missing,0.,0.));self.assertIsNone(coefficients(missing,terms()))
        self.assertIsNone(utility(terms(),1.,1.,eligible=False))
        self.assertEqual(region([None])['status'],'UNKNOWN');self.assertEqual(axis_interval([None],'both')['status'],'UNKNOWN')
        pairs=[pair(s,dict(A=1.,B=0.,C=0.) if s!=209 else None) for s in range(205,210)]
        r=point_summary(pairs,0.,0.)
        self.assertIsNone(r['phases']['all5']['mean_delta_U']);self.assertIsNone(r['phases']['all5']['all_positive'])
        self.assertEqual(r['phases']['validation']['known_pairs'],2);self.assertTrue(r['phases']['validation']['all_positive'])
        self.assertEqual(r['phases']['confirmation']['known_pairs'],2)
    def native(self):
        rows=[]
        for load in [1.85,1.89]:
            for role in ROLES:
                for seed in [203,204]:
                    # Spill is better Q, worse L/P: changing preferences reorders.
                    t=terms(Q=0.,L=4.,P=4.) if role=='calibrated_spill' else terms(Q=2.,L=1.,P=1.)
                    if role=='legacy_latency':t=terms(Q=6.,L=6.,P=6.)
                    if role=='strong_other_deployment':t=terms(Q=3.,L=2.,P=2.)
                    rows.append(dict(stage='development-text'+str(load),load=load,role=role,contract=dict(deployment={role:1}),summary=dict(seed=seed,utility=dict(eligible=True,**{t0+'_norm':v for t0,v in t.items()}))))
        for seed in range(205,210):
            for role in ROLES[:2]:
                t=terms(Q=0.,L=4.,P=4.) if role=='calibrated_spill' else terms(Q=2.,L=1.,P=1.)
                rows.append(dict(stage='independent_validation' if seed<207 else 'final_confirmation',load=1.85,role=role,contract=dict(deployment={'text':2}),summary=dict(seed=seed,utility=dict(eligible=True,**{t0+'_norm':v for t0,v in t.items()}))))
        return rows
    def test_reranking_reports_only_actual_conditions_and_unknown_domain(self):
        native=self.native();old=rank_development(native,1.,1.);new=rank_development(native,0.,0.)
        self.assertEqual(len(old),2)
        for group in old:self.assertEqual(group['ranking'][0]['role'],'risk_matched_no_spill')
        for group in new:self.assertEqual(group['ranking'][0]['role'],'calibrated_spill')
        native[0]['summary']['utility']['eligible']=False
        group=rank_development(native,0.,0.)[0]
        self.assertFalse(group['whole_four_condition_domain_known'])
        spill=next(r for r in group['ranking'] if r['role']=='calibrated_spill')
        self.assertIsNone(spill['mean_U']);self.assertIsNone(spill['rank'])
    def test_gate_fails_before_reading_any_reserved_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'progress.json').write_text('{"status":"RUNNING"}')
            with self.assertRaisesRegex(ValueError,'Entire original execution'):
                load_completed(p,p/'nonexistent-data.json',p/'nonexistent-plan.json')
            (p/'progress.json').write_text('{"status":"COMPLETE_WITH_UNRESOLVED_EFFICACY"}')
            with self.assertRaisesRegex(ValueError,'Entire original execution'):
                load_completed(p,p/'nonexistent-data.json',p/'nonexistent-plan.json')
    def completed_fixture(self,root):
        execution=root/'execution';execution.mkdir();native=self.native();stages=[]
        base=root/'base.json';base.write_text('{}');base_bound=receipt(base)
        for label in ['development-text1.85','development-text1.89','independent_validation','final_confirmation']:
            rows=[r for r in native if r['stage']==label];jobs=[];registered=[]
            phase='development' if label.startswith('development') else label
            for row in rows:
                seed=row['summary']['seed'];role=row['role'];rate=row['load']
                spill=role=='calibrated_spill'
                contract=dict(deployment={'text':2,'generalist_2':1,'ocr':1},text_rate=rate,
                    strategy='queue_quality' if spill else 'queue_specialist_only',
                    guard=dict(task='text',family='generalist_2',rate_rps=rate/8,burst=1.) if spill else None,
                    refinement=dict(mode='conservative',risk_quantile=.5,margin_ms=0.,allow_spill=spill))
                if role=='legacy_latency':contract['strategy']='queue_latency'
                if role=='strong_other_deployment':contract['deployment']={'text':2,'generalist_2':1,'ocr':1,'image':1}
                workload=dict(text=rate,image=.01,ocr=.01,asr=.01)
                ident=dict(role=role,seed=seed,phase=phase,workload=workload,policy_contract_sha256=identity(contract))
                job=dict(id=identity(ident),seed=seed,role=role,phase=phase,identity=ident,policy_contract=contract,base_packet=base_bound)
                jobs.append(job)
                raw=dict(config=dict(weights={t:1/6 for t in TERMS},random_seed=seed,workload=workload),
                    transport_config=dict(random_seed=seed,workload_rps=workload),evaluation=dict(utility=row['summary']['utility']),
                    policy_contract=contract,experiment_job_id=job['id'],executed_measurement_identity=ident,
                    executed_measurement_id=identity(ident),experiment_phase=phase,source_packet=base_bound)
                path=root/f'raw{len(list(root.glob("raw*.json")))}.json';path.write_text(json.dumps(raw))
                row['contract']=contract;row['summary']['raw']=receipt(path)
                symbolic=dict(deployment=contract['deployment'],strategy=contract['strategy'],policy_contract_sha256=identity(contract))
                row['summary']['candidate']=symbolic;row['summary']['candidate_id']=identity(symbolic)
                registered.append(dict(job_id=job['id'],raw=receipt(path)))
            packet=root/(label+'-packet.json');packet.write_text(json.dumps(dict(phase=phase,jobs=jobs)))
            progress=root/(label+'-progress.json');progress.write_text(json.dumps(dict(status='COMPLETE',packet=receipt(packet),jobs=registered)))
            stages.append(dict(label=label,packet=receipt(packet),progress=str(progress),terminal_progress=receipt(progress)))
        source=root/'bound-source.py';source.write_text('print("A frozen source is not JSON")\n')
        (execution/'progress.json').write_text(json.dumps(dict(status='COMPLETE',stages=stages,bindings=[receipt(source)])))
        data=root/'data.json';data.write_text(json.dumps(dict(status='COMPLETE_AUDITED',issues=[],native=native)))
        plan=root/'plan.json';plan.write_text(json.dumps(dict(run_after_current_native_completion=True,current_experiment_weights_unchanged=True,axes=dict(L_scale=[0,.1,.25,.5,.75,1],P_scale=[0,.1,.25,.5,.75,1]))))
        return execution,data,plan,native
    def test_completed_synthetic_end_to_end_and_raw_sha_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);execution,data,plan,native=self.completed_fixture(root)
            with patch('tools.queue_refined_weight_sensitivity_20261023.plots'):
                report=analyze(execution,data,plan,root/'out')
            self.assertEqual(len(report['coarse']),36);self.assertEqual(len(report['fine_grid']['points']),10201)
            self.assertEqual(len(report['development_reranking']),72)
            self.assertEqual(report['strict_positive_all5_region']['status'],'NONEMPTY')
            self.assertLess(report['original']['phases']['all5']['mean_delta_U'],0.)
            self.assertTrue((root/'out'/'fine-grid.csv').exists());self.assertGreaterEqual(len(report['sources']),38)
            changed=Path(native[0]['summary']['raw']['path']);changed.write_text('{}')
            with self.assertRaisesRegex(ValueError,'SHA changed'):load_completed(execution,data,plan)
    def test_report_role_swap_seed_duplicate_and_missing_stage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            execution,data,plan,native=self.completed_fixture(Path(tmp))
            load_completed(execution,data,plan)
            original=json.loads(data.read_text())
            swapped=copy.deepcopy(original)
            for row in swapped['native']:
                if not row['stage'].startswith('development'):
                    row['role']='risk_matched_no_spill' if row['role']=='calibrated_spill' else 'calibrated_spill'
            data.write_text(json.dumps(swapped))
            with self.assertRaisesRegex(ValueError,'Report role/stage'):load_completed(execution,data,plan)
            swapped=copy.deepcopy(original);swapped['native'][-1]['summary']['seed']=205
            data.write_text(json.dumps(swapped))
            with self.assertRaisesRegex(ValueError,'Report role/stage'):load_completed(execution,data,plan)
            swapped=copy.deepcopy(original);swapped['native'][-1]=swapped['native'][-2]
            data.write_text(json.dumps(swapped))
            with self.assertRaisesRegex(ValueError,'Duplicate/unregistered'):load_completed(execution,data,plan)
            data.write_text(json.dumps(original));(execution/'progress.json').write_text('{"status":"COMPLETE","stages":[]}')
            with self.assertRaisesRegex(ValueError,'workflow stages'):load_completed(execution,data,plan)
    def test_exact_narrow_strict_region_does_not_round_witness_to_boundary(self):
        coefs=[dict(A=-.5,B=1.,C=0.),dict(A=.5000000000000001,B=-1.,C=0.)]
        interval=axis_interval(coefs,'L_only');r=region(coefs)
        self.assertEqual(interval['status'],'NONEMPTY');self.assertEqual(r['status'],'NONEMPTY')
        l,p=map(Fraction,r['strict_witness_exact'])
        self.assertGreater(l,Fraction.from_float(.5));self.assertLess(l,Fraction.from_float(.5000000000000001))
        self.assertTrue(all(Fraction(v)>0 for v in r['witness_numerator_margins_exact']))
        self.assertEqual(Fraction(interval['upper_exact'])-Fraction(interval['lower_exact']),Fraction(1,9007199254740992))
        # A rounded display witness can lie on a boundary; exact witness is authoritative.
        self.assertEqual(l,Fraction(9007199254740993,18014398509481984))

    def test_interrupted25_results_preserve_missing_control_and_energy_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);execution,data,plan,native=self.completed_fixture(root)
            state=json.loads((execution/'progress.json').read_text());state['status']='STOPPED_FOR_DIAGNOSIS_NO_RETRY'
            stage=state['stages'][-1];stage.pop('terminal_progress')
            progress=json.loads(Path(stage['progress']).read_text());progress['status']='STOPPED_FOR_DIAGNOSIS_NO_RETRY'
            packet=json.loads(Path(stage['packet']['path']).read_text())
            absent=next(j for j in packet['jobs'] if j['seed']==209 and j['role']=='risk_matched_no_spill')
            progress['jobs']=[r for r in progress['jobs'] if r['job_id']!=absent['id']]
            doc=json.loads(data.read_text());doc.update(status='COMPLETE_INTERRUPTED_AUDITED',interrupted=True,expected_formal_runs=26,
                unexecuted_jobs=[dict(stage='final_confirmation',role=absent['role'],seed=209,job_id=absent['id'],status='UNEXECUTED_NO_RETRY')])
            doc['native']=[r for r in doc['native'] if not(r['stage']=='final_confirmation' and r['summary']['seed']==209 and r['role']=='risk_matched_no_spill')]
            spill=next(r for r in doc['native'] if r['stage']=='final_confirmation' and r['summary']['seed']==209)
            raw_path=Path(spill['summary']['raw']['path']);raw=json.loads(raw_path.read_text())
            raw['evaluation']['utility'].update(E_norm=None,U=None,eligible=False)
            raw['energy_evidence']=dict(coverage_valid=False);raw_path.write_text(json.dumps(raw))
            spill['summary']['utility']=raw['evaluation']['utility'];spill['summary']['raw']=receipt(raw_path)
            for reg in progress['jobs']:
                if reg['job_id']==raw['experiment_job_id']:reg['raw']=receipt(raw_path)
            Path(stage['progress']).write_text(json.dumps(progress));(execution/'progress.json').write_text(json.dumps(state));data.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'Entire original execution'):load_completed(execution,data,plan)
            with patch('tools.queue_refined_weight_sensitivity_20261023.plots'):
                result=analyze(execution,data,plan,root/'interrupted-out',allow_interrupted=True)
            self.assertEqual(result['status'],'COMPLETE_INTERRUPTED_POSTHOC_SENSITIVITY')
            self.assertEqual(result['registered_native_runs'],25);self.assertEqual(result['evaluable_pairs_only']['seeds'],[205,206,207,208])
            last=result['pairs'][-1];self.assertEqual(last['pair_status'],'CONTROL_NOT_EXECUTED')
            self.assertIsNone(last['spill_terms']['E']);self.assertTrue(all(v is None for v in last['control_terms'].values()))
            self.assertIsNone(last['coefficients']);self.assertIsNone(last['control_raw'])
            for p in result['coarse']:
                self.assertIsNone(p['phases']['all5']['mean_delta_U']);self.assertIsNone(p['phases']['all5']['all_positive'])
                self.assertIsNone(p['phases']['confirmation']['mean_delta_U']);self.assertEqual(p['phases']['confirmation']['known_pairs'],2)
                self.assertEqual(p['phases']['evaluable_only']['known_pairs'],4)
            self.assertEqual(result['strict_positive_all5_region']['status'],'UNKNOWN')
            self.assertEqual(result['axis_critical_points']['both']['all5']['status'],'UNKNOWN')
            self.assertNotEqual(result['evaluable_pairs_only']['strict_positive_region']['status'],'UNKNOWN')
            rendered=(root/'interrupted-out'/'权重敏感性报告.md').read_text()
            for label in ['原权重','仅L减半','仅P减半','L/P均减半','仅L归零','仅P归零','L/P均归零']:
                self.assertIn('|'+label+'|',rendered)
            self.assertIn('该完整配对子集的均值解析式',rendered)
            self.assertIn('全部5对均值与正收益区域继续为UNKNOWN',rendered)
            doc['native'][-1]['role']='risk_matched_no_spill';data.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'Report role/stage'):load_completed(execution,data,plan,allow_interrupted=True)

if __name__=='__main__':unittest.main()
