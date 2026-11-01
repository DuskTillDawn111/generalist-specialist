import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027 import twoFIFO_prospective_contract_v1 as target


def fixture():
    aliases = {'text#0':dict(family='text',supported_tasks=['text']),
        'text#1':dict(family='text',supported_tasks=['text']),
        'generalist_2#0':dict(family='generalist_2',supported_tasks=['text','image','asr','ocr']),
        'ocr#0':dict(family='ocr',supported_tasks=['ocr'])}
    trace = [dict(request_id='old:'+task,task=task,pool_id=task+'-input',offset_ns=i*100)
        for i,task in enumerate(('text','image','asr','ocr'))]
    plan = dict(schema='gsdeploy.native_flow_plan.v1',aliases=aliases,
        workload_rps={task:rate for task,rate in zip(('text','image','asr','ocr'),(1.85,.05,.05,.05))},
        latency_reference_ms={task:1000 for task in ('text','image','asr','ocr')},
        weights={key:1/6 for key in 'QTLPME'},memory_budget_mib=49140,
        energy_reference_watts=450,memory_safe_factor=1.1,startup_amortization_seconds=3600.,
        window_seconds=90.,drain_seconds=120.,warmup_seconds=0.,baseline_candidate_id='balanced',
        queue_sample_offsets_ns=[0,90000000000],trace=trace,execution_context_sha256='old-context',
        validation_thresholds=dict(service_mare_max=.2,delta_sign_accuracy_min=1.,pair_rank_accuracy_min=1.),
        candidates=[dict(id=cid,execution_context_sha256='old-context',
            resource_configuration=dict(mode='shared'),
            routes={r['request_id']:(owner if r['task']=='text' else
                'ocr#0' if r['task']=='ocr' else 'generalist_2#0') for r in trace})
            for cid,owner in [('balanced','text#0'),('skewed','text#1'),('spill','generalist_2#0')]])
    objective=dict(schema=target.OBJECTIVE,weights=plan['weights'],stability_policy='existing_thresholds_unchanged')
    declarations={r['pool_id']:dict(pool_id=r['pool_id'],task=r['task'],exact_input_sha256=r['task']+'-sha')
        for r in trace}
    plans=[]
    for block in ('a','b','c'):
        for shard in range(4):
            rows=[dict(owner=owner,task='text',pool_id='text-input') for owner in ('text#0','text#1','generalist_2#0')]
            rows.append(dict(owner='ocr#0',task='ocr',pool_id='ocr-input'))
            plans.append(dict(attempt_id=block+str(shard),capture_id=block,split='training',cases=[dict(arrivals=rows)]))
    return plan,objective,plans,declarations


class PublicContractTests(unittest.TestCase):
    def test_entire_calendar_and_exact_routes_unchanged(self):
        plan,objective,_,_=fixture(); before=copy.deepcopy(plan)
        contract=target.public_contract(plan,objective)
        self.assertEqual(plan,before)
        self.assertEqual(contract['workload_rps'],plan['workload_rps'])
        self.assertEqual(contract['validation_thresholds'],plan['validation_thresholds'])
        for candidate in plan['candidates']:
            value=target.candidate_calendar(contract,candidate)
            self.assertEqual(value['original_window'],[0,90000000000])
            self.assertEqual(value['original_drain_end_ns'],210000000000)
            self.assertEqual(len(value['public_requests']),4)
            self.assertEqual([r['request_id'] for r in value['public_requests']],
                [r['request_id'] for r in plan['trace']])
            self.assertEqual([r['owner'] for r in value['public_requests']],
                [candidate['routes'][r['request_id']] for r in plan['trace']])

    def test_resident_capabilities_cannot_supply_image_asr_templates(self):
        plan,objective,plans,declarations=fixture()
        matrix=target.planned_support(target.public_contract(plan,objective),plans,declarations)
        missing={r['task'] for r in matrix['required_keys'] if r['status'].startswith('MISSING')}
        self.assertEqual(missing,{'image','asr'})
        self.assertEqual(matrix['required_key_count'],6)
        self.assertEqual(matrix['missing_key_count'],2)
        self.assertTrue(matrix['planned_coverage_is_not_fitted_support'])

    def test_missing_one_whole_block_is_not_pooled_away(self):
        plan,objective,plans,declarations=fixture()
        for row in plans:
            if row['capture_id']=='c':
                row['cases'][0]['arrivals']=[r for r in row['cases'][0]['arrivals'] if r['owner']!='text#1']
        matrix=target.planned_support(target.public_contract(plan,objective),plans,declarations)
        entry=next(r for r in matrix['required_keys'] if r['owner']=='text#1')
        self.assertEqual(entry['planned_measured_count_by_block'],dict(a=4,b=4,c=0))
        self.assertEqual(entry['status'],'MISSING_EXACT_NATIVE_TEMPLATE_SUPPORT')

    def test_task_subset_is_preserved_and_not_called_four_tasks(self):
        plan,objective,_,_=fixture()
        plan['trace']=[r for r in plan['trace'] if r['task'] in ('text','ocr')]
        for candidate in plan['candidates']:
            candidate['routes']={r['request_id']:candidate['routes'][r['request_id']] for r in plan['trace']}
        plan['workload_rps']={t:v for t,v in plan['workload_rps'].items() if t in ('text','ocr')}
        plan['latency_reference_ms']={t:v for t,v in plan['latency_reference_ms'].items() if t in ('text','ocr')}
        contract=target.public_contract(plan,objective)
        self.assertEqual(contract['task_counts'],dict(text=1,ocr=1))
        self.assertEqual(contract['resident_owner_count'],4)

    def test_nonoriginal_weights_hash_and_route_coverage_rejected(self):
        plan,objective,_,_=fixture()
        changed=copy.deepcopy(plan);changed['weights']={'Q':1.,'T':0.,'L':0.,'P':0.,'M':0.,'E':0.}
        with self.assertRaisesRegex(ValueError,'Original equal six-term'):
            target.public_contract(changed,objective)
        changed=copy.deepcopy(plan);changed['plan_sha256']='not-the-plan'
        with self.assertRaisesRegex(ValueError,'hash changed'):
            target.public_contract(changed,objective)
        changed=copy.deepcopy(plan);changed['candidates'][0]['routes'].pop('old:asr')
        with self.assertRaisesRegex(ValueError,'full route coverage'):
            target.public_contract(changed,objective)

    def test_end_to_end_preparation_is_hold_and_audit_rejects_calendar_tamper(self):
        plan,objective,plans,declarations=fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);catalog=[]
            for task in ('text','image','asr','ocr'):
                sample=dict(context='public context',question='public question') if task=='text' else dict(input=str(root/(task+'.data')))
                if task!='text':Path(sample['input']).write_bytes(task.encode())
                row=dict(pool_id=task+'-input',task_type=task,sample=sample);catalog.append(row)
                declarations[row['pool_id']]['exact_input_sha256']=target.input_identity(row)
            catalog_path=root/'inputs.jsonl';catalog_path.write_text('\n'.join(json.dumps(r) for r in catalog)+'\n')
            objective_path=root/'objective.json';target.save(objective_path,objective)
            public_path=root/'plan.json';target.save(public_path,plan)
            context_path=root/'context.json';context=target.save(context_path,dict(declared_inputs=list(declarations.values())))
            refs=[]
            for index,row in enumerate(plans):
                row['context']=context;refs.append(target.save(root/('attempt-'+str(index)+'.json'),row))
            campaign_path=root/'campaign.json';target.save(campaign_path,dict(schema=target.CAMPAIGN,context=context,attempts=refs))
            report=target.prepare(objective_path,public_path,campaign_path,catalog_path,root/'prepared')
            self.assertEqual(report['status'],'CPU_PUBLIC_CONTRACT_FROZEN_HOLD_NOT_PREDICTION_SEAL')
            self.assertEqual(report['missing_native_keys'],2)
            self.assertFalse(report['predictions_sealed']);self.assertFalse(report['GPU_started'])
            self.assertEqual(target.audit(root/'prepared/report.json'),report)
            report_path=root/'prepared/report.json'
            input_path=Path(report['artifacts']['public_inputs']['path'])
            original_inputs=input_path.read_text()
            for mutation in ('text_hash','missing_input'):
                changed=copy.deepcopy(report);manifest=json.loads(original_inputs)
                if mutation=='text_hash':
                    next(r for r in manifest if r['task']=='text')['exact_input_sha256']='f'*64
                else:manifest.pop()
                input_path.write_text(json.dumps(manifest))
                changed['artifacts']['public_inputs']=target.receipt(input_path)
                report_path.write_text(json.dumps(changed))
                with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,'input manifest changed'):
                    target.audit(report_path)
            input_path.write_text(original_inputs)
            for field in ('task_counts','resident_owner_count','required_native_keys','missing_native_keys'):
                changed=copy.deepcopy(report);changed[field]={} if field=='task_counts' else -1
                report_path.write_text(json.dumps(changed))
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,'summary changed'):
                    target.audit(report_path)
            report_path.write_text(json.dumps(report))
            path=Path(report['candidate_calendars']['spill']['path']);path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Changed referenced source'):
                target.audit(root/'prepared/report.json')


if __name__=='__main__':
    unittest.main()
