"""Matching 171case score integration; CPU fixtures are never native evidence."""
import copy
from contextlib import ExitStack
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_matching_numerics_v1 import matching_fixture,SCORE_DESIGN
from tests.test_twoFIFO_forecast_ensemble_v1 import arguments
from tests.test_twoFIFO_fold_resource_rescore_v1 import renamed,state_receipt
from tools.predictor_validation_20261027 import twoFIFO_matching_resource_rescore_v1 as m

SECOND=10**9


def put(path,doc):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(doc));return m.receipt(path)


def allcase_fixture(root):
    plans=matching_fixture()[0];score_design=m.checked(m.receipt(SCORE_DESIGN))
    score_ref,quality_ref=score_design['scoring_plan'],score_design['quality']
    score,quality=m.checked(score_ref),m.checked(quality_ref)
    plan_refs=[];reviews=[];archived=[];blocks=sorted({p['capture_id'] for p in plans})
    for index,plan in enumerate(plans):
        plan_refs.append(put(root/f'plan-{index}.json',plan))
        windows=[];cases=[];records=[]
        for number,case in enumerate(plan['cases']):
            epoch=(10+number*100)*SECOND
            windows.append(dict(case_id=case['case_id'],start_ns=epoch,end_ns=epoch+10*SECOND,
                fixed_drain_deadline_ns=epoch+70*SECOND))
            cases.append(dict(case_id=case['case_id'],observation_start_ns=epoch))
            records.extend(dict(request_id=a['request_id'],status='completed',
                completion_time_ns=epoch+a['arrival_ns']+10**6) for a in case['arrivals'])
            forecast=m.source.generated.forecast_case(case,epoch,None,None,score,quality,
                donor_blocks=[b for b in blocks if b!=plan['capture_id']],
                unqualified_reason='CPU_NUMERICAL_FIXTURE_UNQUALIFIED')
            archive=root/'archive'/str(index)/str(number)/'prediction.json'
            archive.parent.mkdir(parents=True,exist_ok=True)
            ref=m.save(archive,m.source.generated.compact_prediction(forecast))
            archived.append(dict(block=plan['capture_id'],attempt_id=plan['attempt_id'],
                case_id=case['case_id'],prediction=ref))
        last=windows[-1]['fixed_drain_deadline_ns']
        raw=dict(schema=m.source.response.executor.RAW_SCHEMA,backend='native',status='completed',split='training',
            CPU_FIXTURE_NOT_NATIVE=True,**{k:plan[k] for k in ('capture_id','attempt_id','execution_id')},
            lifecycle_boundaries=[dict(boundary=k,time_ns=v) for k,v in [('startup_begin',8*SECOND),
                ('priming_end',9*SECOND),('monitoring_end',last+SECOND),('shutdown_begin',last+2*SECOND)]],
            shutdown_memory_sampled=False,cases=cases,measurement_windows=windows,request_records=records,
            board_samples=[dict(time_ns=t,watts=20.,owned_used_mib=3000. if t==79*SECOND else 1000.)
                for t in range(8*SECOND,last+SECOND+1,SECOND)])
        raw_ref=put(root/f'raw-{index}.json',raw)
        review_ref=put(root/f'review-{index}.json',dict(raw=raw_ref,CPU_FIXTURE_NOT_NATIVE=True))
        reviews.append(dict(attempt_id=plan['attempt_id'],training_review=review_ref))
    executor=put(root/'executor.json',dict(attempts=[dict(plan=r) for r in plan_refs]))
    manifest=dict(executor_manifest=executor)
    manifest_ref=put(root/'manifest.json',manifest)
    final=put(root/'continuation.json',dict(CPU_FIXTURE_NOT_NATIVE=True))
    intake=dict(attempts=reviews,CPU_FIXTURE_NOT_NATIVE=True)
    design=dict(fit_manifest=manifest_ref,scoring_plan=score_ref,quality=quality_ref)
    design_ref=put(root/'design.json',design)
    resource_design=dict(fit_manifest=manifest_ref)
    resource_design_ref=put(root/'resource-design.json',resource_design)
    fit=dict(manifest=manifest_ref,continuation=final,intake_evidence=intake)
    fit_ref=put(root/'fit.json',fit)
    model=dict(schema=m.resource.SCHEMA,status='COMPLETE_MATCHING_RESOURCE_TRAINING_NOT_QUALIFICATION',
        resource_fit_run=True,validation_labels_read=False,design=resource_design_ref,
        continuation=final,intake_evidence=intake,sources=[])
    model_ref=put(root/'model.json',model)
    report=dict(schema=m.source.SCHEMA,status='COMPLETE_MATCHING_UNQUALIFIED_FIT_ALL_WORLDS_UNKNOWN',
        response_fit_reused=True,validation_labels_read=False,sources=[],design=design_ref,response_fit=fit_ref,
        whole_block_leaveout=[dict(heldout_block_id=b,training_block_ids=[x for x in blocks if x!=b],
            donor_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']!=b],
            heldout_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']==b],folded_law=None,
            cases=[{k:v for k,v in r.items() if k!='block'} for r in archived if r['block']==b]) for b in blocks])
    report_ref=put(root/'completed.json',report)
    return dict(plans=plans,report=report,report_ref=report_ref,model=model,model_ref=model_ref,
        fit=fit,design=design,resource_design=resource_design,manifest=manifest)


def bind_provider(provider,plan,context,model_ref,plans):
    block=plan['capture_id']
    provider.target_plan=copy.deepcopy(plan);provider.target_context=copy.deepcopy(context)
    provider.physical_execution_identity=m.resource.physical_execution_identity(plan,context)
    provider.document=dict(resource_domain=copy.deepcopy(m.resource.DOMAIN),priming_content_identity='fixture-prime')
    provider.fold_binding=dict(full_native_model=model_ref,selected_fold_model_sha256='a'*64,
        heldout_block_id=block,training_block_ids=sorted({p['capture_id'] for p in plans}-{block}),
        training_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']!=block],
        heldout_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']==block],sources=[])
    provider.power=lambda *a,**k:dict(known=True,watts=20.)
    provider.memory=lambda *a,**k:dict(known=True,peak_memory_mib=3000.)
    provider.startup=lambda:dict(joules=40.,premeasurement_owned_sampled_peak_mib=1000.)


class MatchingRescoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.data=allcase_fixture(cls.root/'data')
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def authentication(self,calls):
        data=self.data
        def init(provider,path,plan,context,**kwargs):
            calls.append((kwargs['heldout_block_id'],plan['attempt_id']))
            bind_provider(provider,plan,context,data['model_ref'],data['plans'])
        stack=ExitStack()
        for obj,name,value in [(m.source,'audit',data['design']),
            (m.source.response,'audit_fit',dict(report=data['fit'])),
            (m.source.response,'audit',data['manifest']),(m.resource,'audit',data['resource_design'])]:
            stack.enter_context(patch.object(obj,name,return_value=value))
        stack.enter_context(patch.object(m.resource.FoldResourceTrainingProvider,'__init__',new=init))
        stack.enter_context(patch.object(m.source.response.original,'fitted_rows',side_effect=AssertionError('NO_FIT')))
        return stack

    def test_all171_real_replay_score_and1368_unknown_worlds_persist_before_native_labels(self):
        out=self.root/'all-output';calls=[];labels=[];original=m.measured_score_case
        def measured(raw,plan,case,epoch,score,quality,*,source_evidence):
            ref=source_evidence['persisted_resource_prediction']
            self.assertEqual(m.receipt(ref['path']),ref)
            self.assertEqual(len(list(out.glob('*/*/case-*/prediction.json'))),len(labels)+1)
            labels.append((plan['attempt_id'],case['case_id']))
            return original(raw,plan,case,epoch,score,quality,source_evidence=source_evidence)
        with self.authentication(calls),patch.object(m,'measured_score_case',side_effect=measured):
            result=m.checked(m.run_all(self.data['report_ref']['path'],self.data['model_ref']['path'],out))
        self.assertEqual((result['cases'],result['offered_requests'],len(calls),len(labels)),(171,390,3,171))
        total=0
        for row in result['output_cases']:
            pred=m.checked(row['prediction']);value=m.batch.single.decode(pred['ensemble'])
            total+=len(value['draws']);self.assertEqual(len(value['draws']),8)
            self.assertEqual(value['utility']['unknown_probability_mass'],1)
            self.assertIsNone(value['utility']['expected_value'])
            self.assertIsNone(m.batch.single.decode(row['score_errors'])['U_error'])
            label=m.checked(row['measured_label'])
            self.assertEqual(label['measured_resources']['measurement_board_energy_joules'],200.)
            self.assertEqual(label['measured_resources']['startup_energy_joules'],40.)
            self.assertEqual(label['attempt_id'],row['attempt_id'])
        self.assertEqual(total,1368);self.assertFalse(result['native_label_values_used_as_predictions'])

    def generated(self,*,unknown=False,seconds=10,drain=60):
        requests,draws,args=renamed(arguments())
        plan=self.data['plans'][0];context=m.checked(plan['context'])
        args['scoring_plan']['aliases']={owner:dict(family=family,supported_tasks=['text'])
            for owner,family in context['residents'].items()}
        args['scoring_plan']['window_seconds']=seconds
        args.update(window_start_ns=0,window_end_ns=seconds*SECOND,drain_end_ns=(seconds+drain)*SECOND)
        donors=sorted({p['capture_id'] for p in self.data['plans']}-{plan['capture_id']})
        draws[0]['block_id']=donors[0]
        if unknown:
            draws[0]['weight']=F(1,3)
            draws.append(dict(draw_id='unknown',block_id=donors[1],weight=F(2,3),
                inputs=m.batch.ensemble.UnknownForecastInputs('SYNTHETIC_UNKNOWN')))
        document=m.batch.ensemble.evaluate(requests,draws,**args)
        provider=object.__new__(m.resource.FoldResourceTrainingProvider)
        bind_provider(provider,plan,context,self.data['model_ref'],self.data['plans'])
        return plan,context,requests,args,document,provider

    def score_generated(self,data,past=None):
        plan,context,requests,args,document,provider=data
        return m.score_with_provider(document,requests,provider,plan,context,heldout_block_id=plan['capture_id'],
            scoring_plan=args['scoring_plan'],quality=args['quality'],window_start_ns=args['window_start_ns'],
            window_end_ns=args['window_end_ns'],drain_end_ns=args['drain_end_ns'],past_memory_state=past)

    def test_actual_known_world_matching_provider_to_original_M_E_U_and_work_unchanged(self):
        data=self.generated();before=copy.deepcopy(data[4])
        state=dict(basis='explicit_prior_attempt_history',ordered_state_history_identity='0'*64,
            last_state_key='1'*64,prior_owned_sampled_peak_mib=900.)
        ref=state_receipt(self.root,state)
        with patch.object(m.resource.resource,'predict_timeline',side_effect=AssertionError('OLD_RESOURCE_ENTRY')):
            out=self.score_generated(data,ref)
        scored=out['ensemble']['draws'][0]['forecast']['scoring'];score=scored['original_score']
        self.assertEqual(scored['resource_prediction']['board_energy_joules'],200.)
        self.assertEqual(scored['resource_prediction']['startup_energy_joules'],40.)
        self.assertEqual(scored['resource_prediction']['peak_memory_mib'],3000.)
        spec=data[3]['scoring_plan']
        self.assertAlmostEqual(score['terms']['E'],(20+40/3600)/spec['energy_reference_watts'])
        self.assertAlmostEqual(score['terms']['M'],3000*spec['memory_safe_factor']/spec['memory_budget_mib'])
        self.assertIsNotNone(score['full_U'])
        self.assertEqual(data[4],before)
        self.assertEqual(out['ensemble']['draws'][0]['forecast']['world'],before['draws'][0]['forecast']['world'])
        self.assertEqual(len(out['resource_fold_binding']['training_attempt_ids']),10)

    def test_missing_memory_retains_E_and_QTLP_but_unknown_M_U_and_mass(self):
        data=self.generated(unknown=True);out=self.score_generated(data)
        decoded=m.batch.single.decode(out['ensemble'])
        score=decoded['draws'][0]['forecast']['scoring']['original_score']
        self.assertIsNotNone(score['terms']['E']);self.assertIsNone(score['terms']['M']);self.assertIsNone(score['full_U'])
        self.assertEqual(decoded['six_terms']['E']['known_probability_mass'],F(1,3))
        self.assertEqual(decoded['utility']['unknown_probability_mass'],1)
        self.assertIsNone(decoded['six_terms']['E']['expected_value'])
        self.assertEqual(decoded['draws'][1],m.batch.single.decode(data[4])['draws'][1])

    def test_90plus120_world_stays_unknown_for_M_E_before_old_or_matching_lookup(self):
        data=self.generated(seconds=90,drain=120)
        with patch.object(data[-1],'power',side_effect=AssertionError('UNSUPPORTED_LOOKUP')),\
             patch.object(data[-1],'memory',side_effect=AssertionError('UNSUPPORTED_LOOKUP')):
            out=self.score_generated(data)
        score=out['ensemble']['draws'][0]['forecast']['scoring']['original_score']
        self.assertIsNone(score['terms']['E']);self.assertIsNone(score['terms']['M']);self.assertIsNone(score['full_U'])

    def test_missing_case_donor_overlap_and_old_native_schema_rejected(self):
        for mode in ('case','donor'):
            report=copy.deepcopy(self.data['report']);fold=report['whole_block_leaveout'][0]
            if mode=='case':fold['cases'].pop()
            else:fold['donor_attempt_ids'][0]=fold['heldout_attempt_ids'][0]
            ref=put(self.root/f'bad-{mode}.json',report);calls=[];output=self.root/f'bad-{mode}'
            with self.authentication(calls),self.assertRaises(ValueError):
                m.run_all(ref['path'],self.data['model_ref']['path'],output)
            self.assertEqual(calls,[]);self.assertFalse(output.exists())
        plan=self.data['plans'][0];raw=dict(schema='gsdeploy.original_twoFIFO_owned_training.v3.raw',
            backend='native',status='completed',split='training')
        with self.assertRaisesRegex(ValueError,'matching native'):
            m.measured_score_case(raw,plan,plan['cases'][0],0,{}, {},source_evidence={'fixture':True})

    def test_skip_measured_labels_never_opens_native_raw_and_retains_all_cases(self):
        calls=[];original=m.checked
        def checked(ref):
            if type(ref) is dict and '/raw-' in ref.get('path',''):
                raise AssertionError('skip labels opened heldout native raw')
            return original(ref)
        with self.authentication(calls),patch.object(m,'checked',side_effect=checked),\
             patch.object(m,'measured_score_case',side_effect=AssertionError('labels forbidden')):
            out=original(m.run_all(self.data['report_ref']['path'],self.data['model_ref']['path'],
                self.root/'without-labels',include_measured_labels=False))
        self.assertEqual((out['cases'],out['offered_requests'],len(calls)),(171,390,3))
        self.assertTrue(all(row['measured_label'] is None and row['score_errors'] is None for row in out['output_cases']))

    def test_foreign_fold_context_future_memory_or_ambiguous_memory_mapping_rejected(self):
        data=self.generated();data[-1].fold_binding['training_attempt_ids'].pop()
        with self.assertRaisesRegex(ValueError,'fold provider'):self.score_generated(data)
        data=self.generated();data[1]['threads']['OMP_NUM_THREADS']='999'
        with self.assertRaisesRegex(ValueError,'fold provider'):self.score_generated(data)
        data=self.generated()
        future=put(self.root/'future-memory.json',dict(schema=m.batch.single.SCHEMA+'.causal_past_memory_state',
            cutoff_ns=1,forecast_origin_ns=0,state={},sources=[],future_labels_used=False,
            basis='immutable_pre_case_history_only'))
        with self.assertRaisesRegex(ValueError,'no later'):self.score_generated(data,future)
        counts={}
        for p in self.data['plans']:
            for c in p['cases']:counts[c['case_id']]=counts.get(c['case_id'],0)+1
        repeated=next(k for k,v in counts.items() if v>1);calls=[];out=self.root/'ambiguous-memory'
        with self.authentication(calls),self.assertRaisesRegex(ValueError,'Repeated case IDs'):
            m.run_all(self.data['report_ref']['path'],self.data['model_ref']['path'],out,
                past_memory_state_paths={repeated:'unbound'})
        self.assertEqual(calls,[]);self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
