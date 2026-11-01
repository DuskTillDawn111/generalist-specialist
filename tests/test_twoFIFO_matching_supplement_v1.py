import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_matching_supplement_v1 as module


def groups():
    result={owner:[] for owner in module.OWNERS}
    for owner,count in zip(module.OWNERS,(3,3,8,2)):
        for index in range(count):
            task=('text' if owner.startswith('text') else 'ocr' if owner=='ocr#0'
                else 'text' if index<3 else 'image' if index<6 else 'asr')
            result[owner].append(dict(owner=owner,task=task,pool_id=owner+':'+str(index)))
    return result


class MatchingSupplementTests(unittest.TestCase):
    def test_merged_cases_cover_all_directed_modes_with_fewer_drains(self):
        target=groups();before=copy.deepcopy(target)
        cases=module.definitions(target);proof=module.coverage(target,cases)
        self.assertEqual(target,before)
        self.assertEqual(len(cases),57)
        self.assertEqual(sum(len(c['arrivals']) for c in cases),130)
        self.assertEqual(proof['required_keys'],16)
        self.assertEqual(proof['directed_pair_modes'],48)
        self.assertEqual(proof['target_mode_count'],80)
        self.assertFalse(proof['measured_overlap_inferred_from_calendar'])
        self.assertTrue(all(not c['actual_peer_overlap_known'] for c in cases))
        for row in proof['keys']:
            self.assertEqual(len(row['mode_case_ids']),5)
            self.assertTrue(all(row['mode_case_ids'].values()))
        self.assertEqual(cases,module.definitions(target))

    def test_dropping_just_one_directed_target_mode_is_rejected(self):
        target=groups();cases=module.definitions(target);key=target['text#0'][0]
        changed=[c for c in cases if not ({r['owner'] for r in c['arrivals']}=={'text#0','ocr#0'}
            and any(r['pool_id']==key['pool_id'] for r in c['arrivals']))]
        with self.assertRaisesRegex(ValueError,'directed target activity mode'):
            module.coverage(target,changed)

    def test_three_block_start_gaps_vary_without_changing_key_combinations_or_modes(self):
        target=groups();blocks=[module.definitions(target,i) for i in range(3)]
        keys=lambda c:sorted((r['owner'],r['task'],r['pool_id']) for r in c['arrivals'])
        self.assertEqual(module.PAIR_GAPS_NS,(1_000_000,20_000_000,300_000_000))
        self.assertEqual(module.ALL_FOUR_STEPS_NS,(1_000_000,20_000_000,50_000_000))
        for block_index,cases in enumerate(blocks):
            self.assertEqual(len(cases),57)
            self.assertEqual(sum(len(c['arrivals']) for c in cases),130)
            self.assertTrue(module.coverage(target,cases)['all_declared_modes_covered'])
            for baseline,case in zip(blocks[0],cases):
                self.assertEqual(baseline['case_id'],case['case_id'])
                self.assertEqual(baseline['mode'],case['mode'])
                self.assertEqual(keys(baseline),keys(case))
                times=[r['arrival_ns'] for r in case['arrivals']]
                if case['mode']=='pair_activity':self.assertEqual(times,[0,module.PAIR_GAPS_NS[block_index]])
                elif case['mode']=='all_four_activity':
                    self.assertEqual(times,[i*module.ALL_FOUR_STEPS_NS[block_index] for i in range(4)])
        for left,right in zip(blocks[0],blocks[1]):
            if left['mode']=='pair_activity':
                self.assertEqual([r['owner'] for r in left['arrivals']],
                    [r['owner'] for r in reversed(right['arrivals'])])
        orders=[[[r['owner'] for r in case['arrivals']] for case in cases if case['mode']=='all_four_activity']
            for cases in blocks]
        self.assertNotEqual(orders[0],orders[1]);self.assertNotEqual(orders[1],orders[2])
        for invalid in (-1,3,True):
            with self.assertRaises(ValueError):module.definitions(target,invalid)

    def test_solo_bursts_and_shared_cases_keep_every_arrival_without_caps(self):
        for case in module.definitions(groups()):
            arrivals=case['arrivals'];owners={r['owner'] for r in arrivals}
            self.assertEqual([r['arrival_ns'] for r in arrivals],sorted(r['arrival_ns'] for r in arrivals))
            self.assertTrue(all(0<=r['arrival_ns']<module.MEASUREMENT_NS for r in arrivals))
            if case['mode']=='solo_same_owner_burst':
                self.assertEqual(len(owners),1);self.assertEqual(len(arrivals),2)
                self.assertEqual([r['arrival_ns'] for r in arrivals],[0,1_000_000])
            elif case['mode']=='pair_activity':self.assertEqual(len(owners),2)
            else:self.assertEqual(owners,set(module.OWNERS))

    def test_union_keeps_old_exact_payload_and_adds_existing_public_input_id(self):
        old=dict(pool_id='old',task_type='text',sample=dict(task='text',id='old',context='c',question='q'))
        new=dict(pool_id='new',task_type='text',sample=dict(task='text',id='new',context='new c',question='q'))
        context=dict(declared_inputs=[dict(task='text',pool_id='old',input_size=2,
            exact_input_sha256=module.public.input_identity(old))])
        contract=dict(trace=[dict(pool_id='old',task='text'),dict(pool_id='new',task='text')])
        pool,declared,added=module.union_inputs(context,[old],[old,new],contract)
        self.assertEqual(added,['new']);self.assertEqual(len(pool),2)
        self.assertEqual(next(r for r in pool if r['pool_id']=='old'),old)
        self.assertEqual(declared[0],context['declared_inputs'][0])
        changed=copy.deepcopy(old);changed['sample']['context']='changed'
        with self.assertRaisesRegex(ValueError,'changed content'):
            module.union_inputs(context,[old],[changed,new],contract)

    def test_media_bytes_rechecked_and_missing_source_closure_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);media=root/'input';media.write_bytes(b'bytes')
            pool=root/'inputs.jsonl';pool.write_text(json.dumps(dict(pool_id='x',task_type='image',
                sample=dict(task='image',input=str(media),input_sha256=module.receipt(media)['sha256'])))+'\n')
            reference=module.receipt(pool);module.load_pool(reference)
            media.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'media bytes changed'):module.load_pool(reference)
        with self.assertRaisesRegex(ValueError,'Nonempty retained executable'):
            module.check_sources(dict(execution_sources=[]),dict(source_freeze=[]))

    def test_full_public_preparation_rebuilds_new_ids_blocks_pool_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'worker.py';source.write_text('pass\n')
            freeze=dict(original_path=str(source),frozen=module.receipt(source))
            rows=[];keys=[]
            for owner in module.OWNERS:
                task='ocr' if owner=='ocr#0' else 'text';pool=owner+'-pool'
                row=dict(pool_id=pool,task_type=task,sample=dict(task=task,id=pool,context='c',question='q'))
                if task=='ocr':
                    media=root/'ocr';media.write_bytes(b'ocr')
                    row['sample']=dict(task=task,id=pool,input=str(media),input_sha256=module.receipt(media)['sha256'])
                rows.append(row);keys.append(dict(owner=owner,task=task,pool_id=pool))
            pool_path=root/'pool.jsonl';pool_path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            pool_ref=module.receipt(pool_path)
            declared=[dict(task=r['task_type'],pool_id=r['pool_id'],input_size=2,
                exact_input_sha256=module.public.input_identity(r)) for r in rows]
            context_ref=module.save(root/'context.json',dict(declared_inputs=declared,execution_sources=[freeze],
                source_rebound_context_sha256='parent-digest'))
            old_plans=[]
            for index in range(12):
                old_plans.append(module.save(root/('old-'+str(index)+'.json'),dict(attempt_id='old-'+str(index),
                    capture_id='old-block-'+str(index//4),prepared_pool=pool_ref,context=context_ref,
                    cases=[dict(arrivals=[dict(request_id='old:'+str(index))])],priming=[dict(owner='text#0')]*18,
                    runtime_policy=dict(admission=None,batching=False,weight_reuse=False),runtime_config={'max_pending_per_model':64},
                    source_freeze=[freeze],collector_timing_policy={'no_mutating_checkpoint_during_case':True},
                    supervisor_budget={'total_timeout_seconds':2400},preflight_free_bytes_required=18*2**30)))
            trace=[dict(request_id='historic:'+str(i),task=k['task'],pool_id=k['pool_id'],offset_ns=i) for i,k in enumerate(keys)]
            contract=dict(aliases={o:dict(supported_tasks=['text','ocr']) for o in module.OWNERS},trace=trace,
                candidates=[dict(routes={r['request_id']:k['owner'] for r,k in zip(trace,keys)})])
            contract_ref=module.save(root/'contract.json',contract)
            parent=module.save(root/'prospective.json',dict(artifacts=dict(public_contract=contract_ref),
                training_context=context_ref,training_plans=old_plans,
                inputs=dict(public_catalog=pool_ref,public_plan=contract_ref,training_campaign=contract_ref)))
            with patch.object(module.public,'audit',side_effect=lambda p:module.checked(module.receipt(p))):
                out=root/'prepared';namespace='1234567890abcdef1234567890abcdef'
                manifest=module.prepare(parent['path'],namespace,out)
                self.assertEqual(module.audit(out/'manifest.json'),manifest)
                self.assertIsNone(manifest['execute_entrypoint']);self.assertFalse(manifest['resource_fit_eligible'])
                self.assertEqual(len(manifest['blocks']),3)
                plans=[module.checked(r) for b in manifest['blocks'] for r in b['attempts']]
                ids=[r['request_id'] for p in plans for c in p['cases'] for r in c['arrivals']]
                self.assertEqual(len(ids),len(set(ids)));self.assertFalse(set(ids)&{r['request_id'] for r in trace})
                self.assertEqual(len({p['execution_id'] for p in plans}),len(plans))
                self.assertTrue(all(p['schema']==module.SCHEMA+'.attempt' and p['split']=='training' for p in plans))
                self.assertTrue(all(len(p['cases'])<=13 for p in plans))
                context=module.checked(manifest['artifacts']['candidate_context'])
                self.assertNotIn('source_rebound_context_sha256',context)
                self.assertEqual(context['matching_macro_supplement']['inherited_source_rebound_digest_is_parent_only'],'parent-digest')
                path=out/'manifest.json';original=path.read_text()
                changed=copy.deepcopy(manifest);changed['summary']['measured_requests']=0;path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'summary changed'):module.audit(path)
                path.write_text(original)
                ref=manifest['blocks'][0]['attempts'][0];plan=module.checked(ref)
                plan['cases'][0]['arrivals'][0]['arrival_ns']+=1
                Path(ref['path']).write_text(json.dumps(plan));changed=copy.deepcopy(manifest)
                changed['blocks'][0]['attempts'][0]=module.receipt(ref['path']);path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'identity/calendar changed'):module.audit(path)


if __name__=='__main__':unittest.main()
