"""Source-bound training role, original g/n arithmetic and route identity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.routing import Router
from tools.predictor_validation_20261027.flow_capacity_catalog import build,training_evidence,verify_catalog
from tools.predictor_validation_20261027.flow_eligibility import original_rho
from tools.predictor_validation_20261027.probe_validation import receipt
from tools.predictor_validation_20261027.profiling import digest

NS=1_000_000_000

class CapacityCatalogTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.source=self.write('generator.py','training protocol source',text=True)
        self.context=self.write('context.json',{'fixture':'CPU'})
        self.catalog=self.write('base-catalog.json',{'capability_contract':'verified_executable_v1',
            'models':{'text':{'service_time_ms':{'text':123},'supported_tasks':['text']}},
            'capacity_factors':{'text':{'0':0,'1':1,'2':1.5}},'memory_table_mib':{'text':{'0':0,'1':100,'2':200}},'safe_factor':1.1})
        self.profile=self.write('profile.json',{'input_assignments':[{'view':'known_input','task':'text','pool_id':'p'}]})
        aliases={a:{'family':'text','supported_tasks':['text']} for a in ('s0','s1')};jobs=[];self.plans=[]
        for seed in (1,2):
            trace=[{'request_id':'r0','offset_ns':0,'task':'text','pool_id':'p'},
                   {'request_id':'r1','offset_ns':100_000_000,'task':'text','pool_id':'p'}]
            candidates=[]
            for label,weight in [('balanced',.5),('skewed',.25)]:
                weights={'text':{'s0':weight,'s1':1-weight}};router=Router(weights)
                candidates.append({'id':label,'execution_context_sha256':'cpu','routing_weights':weights,
                    'routes':{r['request_id']:router.choose('text') for r in trace},'resource_configuration':{'mode':'shared'}})
            plan={'schema':'gsdeploy.native_flow_plan.v1','phase':'predictor_training','kind':'prospective_training','split':'training',
                'validation_tuning':False,'created_ns':1,'seed':seed,'catalog':receipt(self.catalog),'context':receipt(self.context),
                'profile_plan':receipt(self.profile),'protocol_generator':receipt(self.source),'runner_source':receipt(self.source),
                'aliases':aliases,'trace':trace,'candidates':candidates,'workload_rps':{'text':2},'window_seconds':1,
                'drain_seconds':1,'warmup_seconds':0,'arrival_process':'declared_periodic'}
            plan['plan_sha256']=digest(plan);pp=self.write('plan'+str(seed)+'.json',plan);self.plans.append(pp)
            for c in candidates:
                folder=self.root/(str(seed)+'-'+c['id']);folder.mkdir();start=(seed+3)*NS
                duration=(200 if seed==1 else 400)*1_000_000
                records=[]
                for r in trace:
                    ws=start+r['offset_ns']+1_000_000
                    records.append({'request_id':r['request_id'],'task':'text','pool_id':'p','model':c['routes'][r['request_id']],
                        'scheduled_time_ns':start+r['offset_ns'],'status':'completed','observed_worker_start_time_ns':ws,
                        'worker_response':{'start_perf_ns':ws,'end_perf_ns':ws+duration,'success':True},'completion_time_ns':ws+duration+1})
                raw={'schema':'gsdeploy.native_flow_collection.v1','kind':'measured','backend':'native','status':'completed',
                    'collection_complete':True,'qualification_issues':[],'execution_context_sha256':'cpu','plan':receipt(pp),
                    'candidate_id':c['id'],'resource_configuration':c['resource_configuration'],'trace_sha256':digest(trace),
                    'startup_begin_ns':start-NS,'startup_end_ns':start-NS//2,'phase_start_ns':start,'deadline_ns':start+NS,
                    'observation_end_ns':start+NS+NS//2,'request_records':records,
                    'shutdown':{a:{'forced':False,'exit_code':0,'bye_received':True} for a in aliases},
                    'board_samples':[{'time_ns':start-NS+i*100_000_000,'owned_used_mib':500,'foreign_pids':[],
                        'owned_aliases':list(aliases) if i>=5 else []} for i in range(26)]}
                self.write(str(folder.relative_to(self.root))+'/raw.json',raw)
                jobs.append({'plan':receipt(pp),'candidate_id':c['id'],'output':str(folder)})
        self.suite=self.write('suite.json',{'schema':'gsdeploy.continuous_training_suite.v1','no_validation_data':True,
            'source':receipt(self.source),'seeds':[1,2],'jobs':jobs})

    def write(self,name,value,text=False):
        p=self.root/name;p.write_text(value if text else json.dumps(value));return p

    def test_two_catalogs_for_identical_family_routing_and_exact_original_normalization(self):
        result=build(self.suite,self.root/'out');self.assertEqual(len(result['conditions']),2)
        keys=[]
        for c in result['conditions']:
            catalog=json.loads(Path(c['catalog']['path']).read_text());audit=catalog['native_flow_capacity_audit']
            self.assertEqual(catalog['capacity_factors']['text']['2'],1.5)
            estimate=audit['service_estimates']['text']['text']
            self.assertEqual(estimate['worker_mean_service_ms'],300)
            self.assertEqual(estimate['family_normalized_service_ms'],225) # 300*g(2)/2
            self.assertTrue(verify_catalog(c['catalog']['path']))
            keys.extend(catalog['colocation_profiles'])
            plan=json.loads(self.plans[0].read_text());candidate=next(x for x in plan['candidates'] if x['id'] in c['candidate_ids'])
            rho=original_rho(plan,candidate,catalog)
            self.assertAlmostEqual(rho['rho_by_family']['text'],.3)
            self.assertTrue(rho['rho_admission_passed'])
        self.assertEqual(len(keys),len(set(keys))) # alias weights included in catalog context

    def test_derived_role_does_not_mutate_native_records(self):
        raw=self.root/'1-balanced/raw.json';before=raw.read_bytes()
        plan,c,original,derived,obs,peak=training_evidence(raw)
        self.assertNotIn('split',original['request_records'][0]);self.assertEqual(derived['request_records'][0]['split'],'training')
        self.assertFalse(derived['derivation']['original_observations_modified']);self.assertEqual(before,raw.read_bytes())

    def test_validation_plan_or_late_role_refused(self):
        p=json.loads(self.plans[0].read_text());p['phase']='validation';p.pop('plan_sha256');p['plan_sha256']=digest(p)
        self.write(self.plans[0].name,p)
        rawp=self.root/'1-balanced/raw.json';raw=json.loads(rawp.read_text());raw['plan']=receipt(self.plans[0]);self.write(str(rawp.relative_to(self.root)),raw)
        with self.assertRaisesRegex(ValueError,'predictor_training'):training_evidence(rawp)

    def test_missing_one_declared_training_run_and_original_factor_tamper_rejected(self):
        suite=json.loads(self.suite.read_text());suite['jobs'].pop();self.write(self.suite.name,suite)
        with self.assertRaisesRegex(ValueError,'Every condition'):build(self.suite,self.root/'out')

    def test_catalog_arithmetic_tamper_rejected(self):
        result=build(self.suite,self.root/'out');catalog=json.loads(Path(result['conditions'][0]['catalog']['path']).read_text())
        catalog['capacity_factors']['text']['2']=2
        with self.assertRaisesRegex(ValueError,'normalization|capacity'):verify_catalog(catalog)

if __name__=='__main__':unittest.main()
