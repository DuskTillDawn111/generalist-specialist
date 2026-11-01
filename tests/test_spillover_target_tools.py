"""CPU fixtures: target replacement and unchanged rejection gates; no real search."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
TOOLS=ROOT/'tools/spillover_discovery_20261022'


def load(name):
    spec=importlib.util.spec_from_file_location('_target_tools_'+name,TOOLS/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


roles=load('solve_roles');materialize=load('materialize_confirmation')
benchmark=load('benchmark_multireplica');full=load('verify_full_domain')
target=load('target_specialist')


def catalog():
    models={t:{'role':'specialist','supported_tasks':[t]} for t in target.TARGET_SPECIALISTS}
    models.update({f'generalist_{n}':{'role':'generalist','supported_tasks':list(target.TARGET_SPECIALISTS)} for n in range(1,4)})
    return {'models':models,'performance_mode':'replay_transfer_v1','memory_mode':'replay_transfer_v1',
        'replay_transfer_v1':{'status':'TRANSFER_VALIDATION_FAILED'}}


class TargetToolsTests(unittest.TestCase):
    def test_target_structure_and_strong_control_replace_target_only(self):
        for task in target.TARGET_SPECIALISTS:
            plan={'deployment':{task:2,'generalist_2':1},
                'routing':{task:{task:1.5,'generalist_2':1.1}}}
            self.assertTrue(target.target_structure(catalog(),plan,task)['achieved'])
            one=copy.deepcopy(plan);one['deployment'][task]=1
            self.assertFalse(target.target_structure(catalog(),one,task)['achieved'])
            target.assert_one_specialist_plan(one,task)
            with self.assertRaises(ValueError):target.assert_one_specialist_plan(plan,task)
            materialize.inspect_restrictions('best_one_specialist_max',{'replica_options':{task:[1]}},plan,task)
            if task!='asr':
                with self.assertRaises(ValueError):
                    materialize.inspect_restrictions('best_one_specialist_max',{'replica_options':{'asr':[1]}},plan,task)
        target.recorded_target({})
        with self.assertRaises(ValueError):target.recorded_target({},'text')
        with self.assertRaises(ValueError):target.recorded_target({'target_specialist':'text','target_task':'asr'},'text')

    def test_benchmark_defaults_and_child_target_propagation(self):
        old=benchmark.parse_options(None,list(benchmark.MODELS))
        new=benchmark.parse_options(None,list(benchmark.MODELS),'text')
        self.assertEqual(old['asr'],[1,2]);self.assertEqual(old['text'],[1])
        self.assertEqual(new['text'],[1,2]);self.assertEqual(new['asr'],[1])
        args=SimpleNamespace(source_root=ROOT,config=ROOT/'fixture.json',route_grid=4,time_limit=1,
            replica_options=new,models=list(benchmark.MODELS),target_specialist='text')
        command=benchmark.command(args);self.assertEqual(command[command.index('--target-specialist')+1],'text')
        args.seed=args.horizon=args.reference_result=None;args.replica_options=None
        command=full.command(args);self.assertEqual(command[command.index('--target-specialist')+1],'text')

    def test_exact_route_membership_preserves_current_union_domain(self):
        rates={'text':2.6}
        for share,denominator in ((4/7,7),(.5,2)):
            plan={'routing':{'text':{'text':2.6*share,'generalist_2':2.6*(1-share)}}}
            original=copy.deepcopy(plan)
            member=target.route_grid_membership(plan,rates,8)
            self.assertEqual(member['task_membership']['text'][0]['denominator'],denominator)
            self.assertEqual(plan,original)
        with self.assertRaises(ValueError):
            target.route_grid_membership({'routing':{'text':{'text':2.6*4/7,'generalist_2':2.6*3/7}}},rates,6)
        with self.assertRaises(ValueError):
            target.route_grid_membership({'routing':{'text':{'text':2.6*4/9,'generalist_2':2.6*5/9}}},rates,8)
        with self.assertRaises(ValueError):
            target.route_grid_membership({'routing':{'text':{'text':2.6*4/7+.00001,'generalist_2':2.6*3/7-.00001}}},rates,8)

    def test_solve_role_fixture_sends_text_n1_not_asr_restriction(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'gsdeploy').mkdir();wrapper=root/'gsdeploy/search_session.py';wrapper.write_text('# CPU mock only')
            config={'catalog':catalog(),'milp_max_iterations':2,'milp_max_columns':100}
            config['catalog']['replay_transfer_v1']['status']='VALIDATED_ON_NEW_WORKLOAD'
            calls=[]
            class Session:
                def __init__(self,source,raw):
                    self.raw=copy.deepcopy(raw)
                    self.config=SimpleNamespace(as_dict=lambda:self.raw,validate_plan=lambda *a,**kw:{'constraints_valid':True})
                def __enter__(self):return self
                def __exit__(self,*a):pass
                def module(self,name):return SimpleNamespace(check_scoring=lambda c:None)
                def optimizer(self,name):
                    def solve(**kw):
                        calls.append(kw)
                        return {'status':'CPU_FIXTURE_NO_REAL_SEARCH','winner':None,'metrics':{}}
                    return SimpleNamespace(solve=solve)
                def stats(self):return {'fixture':True}
            with patch.object(roles,'load_wrapper',return_value=(SimpleNamespace(SearchSession=Session),wrapper)):
                result=roles.solve_roles(config,root/'out',root,roles=['best_one_specialist_max'],target_specialist='text')
            self.assertEqual(calls,[{'K':8,'replica_options':{'text':[1]}}])
            self.assertEqual(result['target_specialist'],'text')
            self.assertNotIn('asr',result['candidates'][0]['search']['search_restrictions']['replica_options'])

    def test_failed_predictor_refuses_roles_and_materialization_before_wrapper(self):
        for task in ('asr','text','image','ocr'):
            with self.subTest(task=task),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'gsdeploy').mkdir()
                config={'catalog':catalog(),'milp_max_iterations':2,'milp_max_columns':100}
                with patch.object(roles,'load_wrapper',side_effect=AssertionError('No solver may start')):
                    result=roles.solve_roles(config,root/'roles',root,roles=['free'],target_specialist=task)
                self.assertEqual(result['status'],'PREFLIGHT_FAILED')
                self.assertIn('VALIDATED_ON_NEW_WORKLOAD',result['error'])
                config.update(horizon_seconds=1800,search_policy='nominal')
                cp=root/'config.json';cp.write_text(json.dumps(config))
                search={'schema':'gsdeploy.static_search_roles.v1','target_specialist':task,'target_task':task,
                    'config_sha256':materialize.fingerprint(config),'config_receipt':materialize.receipt(cp),
                    'engine':{'source_unchanged':True,'source_before':{},'source_after':{}}}
                sp=root/'search.json';sp.write_text(json.dumps(search))
                with patch.object(materialize,'wrapper',side_effect=AssertionError('No consumer may start')):
                    result=materialize.materialize(sp,cp,root,root/'materialized',study_id='CPU_FIXTURE',target_specialist=task)
                self.assertFalse(result['ready']);self.assertIn('validation required',result['error'])
                self.assertFalse((root/'materialized/contract.json').exists())

    def test_existing_confirmation_consumer_enforces_text_n1(self):
        from gsdeploy.spillover_confirmation import _roles,ROLE_NAMES
        from gsdeploy.quality import quality_digest
        cat=catalog();raw={'catalog':cat}
        config=SimpleNamespace(catalog=cat,as_dict=lambda:raw,validate_plan=lambda **kw:{'constraints_valid':True})
        winner={'deployment':{'text':2,'generalist_2':1},'routing':{'text':{'text':1.,'generalist_2':1.}}}
        one={'deployment':{'text':1,'generalist_2':1},'routing':winner['routing']}
        contract={'target_task':'text','target_specialist':'text'}
        candidates=[{'label':ROLE_NAMES[0],'plan':winner}]
        candidates += [{'label':label,'plan':None,'status':'CERTIFIED_INFEASIBLE'} for label in ROLE_NAMES[1:-1]]
        candidates += [{'label':ROLE_NAMES[-1],'plan':one}]
        bundle={'schema':'gsdeploy.spillover_confirmation_candidates.v1','config_sha256':quality_digest(raw),
            'study_contract_sha256':quality_digest(contract),'candidates':candidates}
        _roles(config,bundle,contract)
        candidates[-1]['plan']=winner
        with self.assertRaisesRegex(ValueError,'target-specialist'):_roles(config,bundle,contract)

    def test_text_materialization_emits_matching_contract_and_snapshot_consumer_roles(self):
        from gsdeploy.spillover_confirmation import _roles
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'gsdeploy').mkdir()
            config={'catalog':catalog(),'horizon_seconds':1800,'search_policy':'nominal','route_grid':8,
                'milp_absolute_gap':1e-9,'workload':{'text':2.6}}
            config['catalog']['replay_transfer_v1'].update(status='VALIDATED_ON_NEW_WORKLOAD',
                seed_roles={'development':[51],'validation':[61,62],'confirmation':[71,72,73]})
            cp=root/'config.json';cp.write_text(json.dumps(config));warm=root/'warmup.json';warm.write_text('{"warmup_seconds":30}')
            winner={'deployment':{'text':2,'generalist_2':1},'routing':{'text':{'text':2.6*4/7,'generalist_2':2.6*3/7}}}
            one=copy.deepcopy(winner);one['deployment']['text']=1
            alternative=copy.deepcopy(winner);alternative['deployment']['text']=3
            rows=[]
            for name in materialize.ROLE_MAP:
                plan=winner if name=='free' else one if name=='best_one_specialist_max' else alternative if name=='strongest_alternative_deployment' else None
                raw={'domain':{'K':8,'capacity_boundary_witnesses':False},'search_completed':True,'metrics':{'unknown_candidates':0},
                    'winner':dict(**plan,score=-1.,prediction={'eligible':True,'utility_complete':True}) if plan else None,
                    'status':'OPTIMAL_MILP_FINITE_DOMAIN' if plan else 'NO_FEASIBLE_GRID_PLAN',
                    'optimality_certified':bool(plan),'optimality_gap':0 if plan else None}
                rp=root/(name+'.raw.json');rp.write_text(json.dumps(raw))
                restrictions={'free':{},'best_global_no_split':{'no_split':True},
                    'best_same_deployment_no_split':{'deployment':winner['deployment'],'no_split':True},
                    'strongest_alternative_deployment':{'exclude_deployments':[winner['deployment']]},
                    'best_one_specialist_max':{'replica_options':{'text':[1]}}}[name]
                result={**raw,'eligible_incumbent':bool(plan),'constraints':{'constraints_valid':True} if plan else None,
                    'config_receipt':materialize.receipt(cp),'search_restrictions':restrictions,'target_specialist':'text'}
                rows.append({'label':name,'search':result,'raw_result_receipt':materialize.receipt(rp)})
            search={'schema':'gsdeploy.static_search_roles.v1','target_specialist':'text','target_task':'text',
                'config_sha256':materialize.fingerprint(config),'config_receipt':materialize.receipt(cp),
                'engine':{'source_unchanged':True,'source_before':{},'source_after':{}},'candidates':rows}
            sp=root/'search.json';sp.write_text(json.dumps(search))
            class Session:
                def __init__(self,source,raw):
                    self.config=SimpleNamespace(catalog=raw['catalog'],as_dict=lambda:raw,
                        validate_plan=lambda *a,**kw:{'constraints_valid':True})
                def __enter__(self):return self
                def __exit__(self,*a):pass
                def module(self,name):
                    if name=='replay_transfer':return SimpleNamespace(check_scoring=lambda c:None)
                    return SimpleNamespace(_gate=lambda c,k:target.recorded_target(k,'text'),_roles=_roles)
                def predict(self,*a):return {'U':-1.,'eligible':True,'utility_complete':True}
            with patch.object(materialize,'wrapper',return_value=SimpleNamespace(SearchSession=Session)):
                report=materialize.materialize(sp,cp,root,root/'out',study_id='CPU_FIXTURE_ONLY',warmup_source=warm,target_specialist='text')
            self.assertTrue(report['ready'],report.get('error'))
            contract=json.loads((root/'out/contract.json').read_text())
            self.assertEqual(contract['target_task'],'text');self.assertEqual(contract['target_specialist'],'text')
            self.assertEqual(contract['routing_domain']['kind'],'union_of_denominators_1_through_K')
            self.assertEqual(report['target_structure']['target_specialist_replicas'],2)

    def test_real_transfer_gate_still_rejects_for_target_aware_benchmark_and_full_domain(self):
        from tests.test_replay_transfer import ReplayTransferTests
        fixture=ReplayTransferTests(methodName='test_failed_validation_is_preserved_and_rejected')
        fixture.setUp()
        try:
            config,_,_=fixture.report(fail=True)
            path=fixture.root/'failed.json';path.write_text(json.dumps(config.as_dict()))
            args=SimpleNamespace(source_root=ROOT,config=path,models=['generalist','specialist'],target_specialist='text')
            with self.assertRaisesRegex(ValueError,'TRANSFER_NOT_VALIDATED'):benchmark.prepare(args)
            with patch.object(full,'MODEL_COUNT',2):
                with self.assertRaisesRegex(ValueError,'TRANSFER_NOT_VALIDATED'):full.prepare(args)
        finally:fixture.doCleanups()


if __name__=='__main__':unittest.main()
