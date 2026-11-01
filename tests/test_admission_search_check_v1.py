"""CPU search proofs, route-provider isolation and strict source boundaries."""
import copy
from contextlib import ExitStack
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import admission_search_check_v1 as m


class AdmissionSearchTests(unittest.TestCase):
    def setUp(self):
        self.ids = list(m.capacity.ROUTES)
        self.plan = dict(weights={k:1/6 for k in 'QTLPME'}, memory_safe_factor=1., memory_budget_mib=100.,
            workload_rps={'text':1.}, aliases={'s':{'family':'text'}},
            trace=[dict(request_id='r',task='text')],
            candidates=[dict(id=cid,routes={'r':'s'}) for cid in self.ids])
        self.model = {'frozen':'service'}
        self.rd = {cid:{'resource_route':cid} for cid in self.ids}
        self.qd = {cid:{'quality_route':cid,'loss_by_family_task':{'text':{'text':0.}}} for cid in self.ids}
        self.overrides = {cid:dict(predicted_startup_joules=0.,predicted_prephase_peak_mib=0.) for cid in self.ids}
        self.originals = {cid:{'rho_admission_passed':None,'rho_threshold_satisfied':True} for cid in self.ids}
        self.gates = {cid:{'passed':None,'checks':{'global_permit':None}} for cid in self.ids}

    def forecast(self, cid, value=-1., *, comparable=True, memory=True, incomplete=None):
        return dict(candidate_id=cid,comparison_U=value if comparable else None,
            metrics=dict(absolute_utility_complete=comparable,six_term_comparison_complete=comparable,
                         memory_budget_satisfied=memory,incomplete_ids=incomplete or [],
                         full_U=value if comparable else None,terms={k:1. for k in 'QTLPME'},unknown_reasons=[]),
            requests=[{'finish_ns':1}],quality_prediction={'candidate_id':cid},
            predicted_candidate_resource_estimates=self.overrides[cid],unknown_reason=None)

    def search(self, *, bounded=False):
        return m.search(self.plan,self.model,self.rd,self.qd,self.overrides,self.gates,self.originals,bounded=bounded)

    def test_two_fresh_searches_use_each_exact_route_model_and_preserve_all_ties(self):
        before=copy.deepcopy(self.plan);seen=[]
        def predict(plan,candidate,model,rd,qd,override):
            cid=candidate['id'];seen.append(cid)
            self.assertIs(model,self.model);self.assertIs(rd,self.rd[cid]);self.assertIs(qd,self.qd[cid])
            self.assertIs(override,self.overrides[cid]);return self.forecast(cid)
        with patch.object(m.validation,'predict',side_effect=predict) as forecast:
            plain=self.search();bounded=self.search(bounded=True)
        self.assertEqual(forecast.call_count,8);self.assertEqual(self.plan,before)
        self.assertEqual(plain['model']['certified_all_tie_ids'],sorted(self.ids))
        self.assertEqual(plain['model'],bounded['model']);self.assertEqual(bounded['bound_pruned'],[])
        self.assertEqual(plain['original_qualified']['status'],'UNKNOWN')
        self.assertIsNone(plain['original_qualified']['certified_maximum_U'])

    def test_true_qualified_candidate_is_not_pruned_by_unqualified_model_incumbent(self):
        first,second,*_=self.ids
        self.gates[first]['passed']=False;self.gates[second]['passed']=True
        self.overrides[second]['predicted_prephase_peak_mib']=30. # bound -.05 < model incumbent -.01
        def predict(plan,candidate,*args):
            return self.forecast(candidate['id'],-.01 if candidate['id']==first else -.5)
        with patch.object(m.validation,'predict',side_effect=predict):
            plain=self.search();bounded=self.search(bounded=True)
        self.assertEqual(plain['original_qualified']['all_known_tie_ids'],[second])
        self.assertEqual(plain['original_qualified']['maximum_known_U'],bounded['original_qualified']['maximum_known_U'])
        self.assertIn(second,[r['candidate_id'] for r in bounded['records']])

    def test_strict_bound_retains_tolerance_ties_and_prunes_only_lower_values(self):
        for cid in self.ids:self.gates[cid]['passed']=False
        a,b,c,d=self.ids
        # Candidate a is first under descending bounds; b lies exactly on the
        # tie-tolerance boundary, c and d have strictly lower upper bounds.
        self.overrides[a]['predicted_prephase_peak_mib']=0.
        self.overrides[b]['predicted_prephase_peak_mib']=6.*(1.+m.TOLERANCE)*100.
        self.overrides[c]['predicted_prephase_peak_mib']=1200.
        self.overrides[d]['predicted_prephase_peak_mib']=1800.
        values={a:-1.,b:-1.-m.TOLERANCE,c:-3.,d:-4.}
        with patch.object(m.validation,'predict',side_effect=lambda p,c,*args:self.forecast(c['id'],values[c['id']])):
            plain=self.search();bounded=self.search(bounded=True)
        self.assertEqual(plain['model']['all_known_tie_ids'],bounded['model']['all_known_tie_ids'])
        self.assertIn(b,[r['candidate_id'] for r in bounded['records']])
        self.assertEqual({r['candidate_id'] for r in bounded['bound_pruned']},{c,d})

    def test_unknown_keeps_full_domain_and_empty_domain_is_explicit(self):
        def predict(plan,candidate,*args):
            return self.forecast(candidate['id'],comparable=candidate['id']!=self.ids[0])
        with patch.object(m.validation,'predict',side_effect=predict):result=self.search()
        self.assertEqual(result['model']['status'],'UNKNOWN');self.assertEqual(result['candidate_count'],4)
        self.assertIsNone(result['model']['certified_maximum_U'])
        with patch.object(m.validation,'predict',side_effect=lambda p,c,*a:self.forecast(c['id'],memory=False)):
            result=self.search()
        for field in ('model','original_qualified'):
            self.assertEqual(result[field]['status'],'EMPTY_DOMAIN');self.assertTrue(result[field]['explicit_empty_domain'])
            self.assertIsNone(result[field]['certified_maximum_U']);self.assertEqual(result[field]['all_known_tie_ids'],[])

    def test_missing_route_models_never_fall_back_to_shared_resources(self):
        del self.rd[self.ids[1]]
        with patch.object(m.validation,'predict',side_effect=lambda p,c,*a:self.forecast(c['id'])):
            with self.assertRaises(KeyError):self.search()

    def test_provider_rejection_remains_unknown_and_drain_violation_excludes(self):
        with patch.object(m.validation,'predict',side_effect=UnknownEvidence('unseen route context')):
            result=self.search()
        self.assertEqual(result['model']['unknown_ids'],sorted(self.ids))
        with patch.object(m.validation,'predict',side_effect=lambda p,c,*a:self.forecast(c['id'],incomplete=['r'])):
            result=self.search()
        self.assertEqual(result['model']['status'],'EMPTY_DOMAIN')

    def test_unknown_term_with_definite_memory_violation_is_still_excluded(self):
        with patch.object(m.validation,'predict',side_effect=lambda p,c,*a:self.forecast(c['id'],comparable=False,memory=False)):
            result=self.search()
        self.assertEqual(result['model']['status'],'EMPTY_DOMAIN')

    def test_original_resolver_and_global_permit_are_not_upgraded_by_statistical_rho(self):
        row=dict(complete_independent_evidence=True,statistical_service_applicability=True,
            original_rho_threshold_satisfied=True,original_stability={'original_stability_passed':True},
            global_permit_capacity_qualified=None,original_measured_profile_resolver_admission=None)
        self.assertIsNone(m.qualification(row,{'rho_admission_passed':None})['passed'])
        row['statistical_service_applicability']=False
        self.assertIs(m.qualification(row,{'rho_admission_passed':None})['passed'],False)
        row.update(statistical_service_applicability=True,global_permit_capacity_qualified=True,
                   original_measured_profile_resolver_admission=True)
        self.assertIsNone(m.qualification(row,{'rho_admission_passed':None})['passed'])
        self.assertTrue(m.qualification(row,{'rho_admission_passed':True})['passed'])

    def test_invalid_bound_or_soundness_violation_fails(self):
        self.overrides[self.ids[0]]['predicted_prephase_peak_mib']=-1.
        with self.assertRaises(ValueError):self.search()
        self.overrides[self.ids[0]]['predicted_prephase_peak_mib']=0.
        with patch.object(m.validation,'predict',side_effect=lambda p,c,*a:self.forecast(c['id'],value=1.)):
            with self.assertRaisesRegex(AssertionError,'upper bound'):self.search()

    def test_report_must_bind_exact_seal_before_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'seal-audit.json').write_text('{}')
            (folder/'report.json').write_text('{}');(folder/'predictions.json').write_text('{}')
            m.save(folder/'validation-audit.json',{'seal_audit':{'path':'wrong','sha256':'wrong'}})
            with patch.object(m.validation,'verify_report',return_value={'verified':True,'report':m.receipt(folder/'report.json')}),patch.object(m,'search') as search:
                with self.assertRaisesRegex(ValueError,'exact search seal'):m.run(folder,folder/'report.json',folder/'out')
            search.assert_not_called();self.assertFalse((folder/'out').exists())

    def run_fixture(self, folder, *, drift=False):
        def write(name,value):
            path=folder/name;m.save(path,value);return m.receipt(path)
        # Deliberately invalid stored objectives prove that only independently
        # fresh forecasts may supply the search objective after audit.
        write('predictions.json',{'candidates':[{'comparison_U':999999.}]})
        catalog_ref=write('catalog.json',{})
        plan=copy.deepcopy(self.plan);plan['catalog']=catalog_ref
        protocol=dict(target_plan=write('plan.json',plan),model=write('model.json',self.model))
        rm=dict(full=dict(route_resources={cid:write(cid+'-r.json',self.rd[cid]) for cid in self.ids},
                          candidate_overrides=self.overrides))
        qm=dict(full=dict(route_quality={cid:write(cid+'-q.json',self.qd[cid]) for cid in self.ids}))
        audit=dict(protocol=write('protocol.json',protocol),resource_manifest=write('rm.json',rm),
            quality_manifest=write('qm.json',qm),capacity_manifest=write('cm.json',{}),capacity_seal=write('cs.json',{}))
        seal_ref=write('seal-audit.json',audit)
        rows=[dict(candidate_id=cid,complete_independent_evidence=True,statistical_service_applicability=True,
            original_rho_threshold_satisfied=True,original_stability={'original_stability_passed':True},
            global_permit_capacity_qualified=None,original_measured_profile_resolver_admission=None) for cid in self.ids]
        report=write('report.json',{})
        write('validation-audit.json',dict(seal_audit=seal_ref,capacity_qualification=write('qualification.json',{'rows':rows})))
        calls=[]
        def forecast(p,c,*args):
            calls.append(c['id'])
            if drift and len(calls)==8:(folder/'catalog.json').write_text('{"changed":true}')
            return self.forecast(c['id'])
        with ExitStack() as stack:
            for target,name,result in ((m.validation,'verify_report',{'verified':True,'report':report}),
                (m.validation,'check_sealed',audit),(m.validation,'audit_protocol',protocol),
                (m.resources,'verify_manifest',rm),(m.quality,'verify_manifest',qm),
                (m.capacity,'verify_manifest',{}),(m.capacity,'verify_seal',{})):
                stack.enter_context(patch.object(target,name,return_value=result))
            stack.enter_context(patch.object(m.core,'original_rho',new=lambda p,c,cat:{'rho_admission_passed':None}))
            stack.enter_context(patch.object(m.validation,'predict',side_effect=forecast))
            result=m.run(folder,report['path'],folder/'out')
        return result,calls

    def test_run_keeps_stored_objectives_out_and_source_binds_all_fresh_rollouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,calls=self.run_fixture(Path(tmp))
            self.assertEqual(len(calls),8);self.assertEqual(result['status'],'PASS')
            self.assertEqual(result['model']['certified_maximum_U'],-1.)
            self.assertEqual(result['model']['certified_all_tie_ids'],sorted(self.ids))
            self.assertEqual(result['original_qualified']['status'],'UNKNOWN')
            self.assertFalse(result['stored_prediction_used_as_objective']);self.assertFalse(result['F8_outcomes_read'])
            self.assertFalse(result['original_gate_constrained_optimality_certified'])
            self.assertTrue((Path(tmp)/'out/summary.json').exists())

    def test_source_drift_during_forecasts_prevents_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):self.run_fixture(Path(tmp),drift=True)
            self.assertFalse((Path(tmp)/'out').exists())


if __name__=='__main__':unittest.main()
