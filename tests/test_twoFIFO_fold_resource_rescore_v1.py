"""CPU integration: only native load_complete is replaced; fold/scorer are real."""
import copy
from dataclasses import fields,is_dataclass,replace
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_forecast_ensemble_v1 import arguments
from tests.test_twoFIFO_resource_fold_provider_v1 import fixture,model_document
from tests import test_twoFIFO_generated_wholeblock_evaluation_v1 as archived_fixture
from tools.predictor_validation_20261027 import twoFIFO_fold_resource_rescore_v1 as m

resource=m.folds.resource


def renamed(value):
    if is_dataclass(value):return replace(value,**{f.name:renamed(getattr(value,f.name)) for f in fields(value)})
    if isinstance(value,dict):return {(k if k=='owner' and type(v) is str else renamed(k)):renamed(v)
        for k,v in value.items()}
    if isinstance(value,tuple):return tuple(map(renamed,value))
    if isinstance(value,list):return list(map(renamed,value))
    return 'text#0' if value=='owner' else value


def setup_world(*,unknown=False):
    plan,context,tables=fixture();plan=copy.deepcopy(plan)
    plan.update(capture_id='training-block-3',attempt_id='training-block-3-shard-1')
    requests,draws,args=renamed(arguments())
    args['scoring_plan']['aliases']={owner:dict(family=family,supported_tasks=['text'])
        for owner,family in context['residents'].items()}
    if unknown:
        draws[0]['weight']=F(1,3)
        draws.append(dict(draw_id='unknown',block_id='training-block-2',weight=F(2,3),
            inputs=m.ensemble.UnknownForecastInputs('CPU_TEST_ONLY_missing_transport')))
    document=m.ensemble.evaluate(requests,draws,**args)
    board=document['draws'][0]['forecast']['scoring']['resource_timeline']
    state=dict(basis='explicit_prior_attempt_history',ordered_state_history_identity='0'*64,
        last_state_key='1'*64,prior_owned_sampled_peak_mib=900.)
    for table in tables:
        history=state['ordered_state_history_identity'];previous=state['last_state_key'];slices=[]
        block=int(table['capture_id'][-1])
        for index,row in enumerate(board['slices']):
            key,_=resource.legacy.key(row['actual_START_to_READY_handlers'],row['resident_owners'])
            if key!=previous:history=resource.next_history(history,key);previous=key
            slices.append(dict(state_key=key,memory_state_key=resource.memory_key(key,history),
                power_training_joules=2.*block,power_training_duration_ns=1_000_000_000,
                sampled_owned_peak_mib=1000.*block+10*index))
        table['slices']=slices
    return plan,context,tables,requests,args,document,state


def state_receipt(directory,state,*,origin=0,cutoff=0):
    evidence=Path(directory)/'CPU_TEST_ONLY_past_evidence.json';evidence.write_text('{"past":true}')
    path=Path(directory)/'past-state.json'
    path.write_text(json.dumps(dict(schema=m.SCHEMA+'.causal_past_memory_state',cutoff_ns=cutoff,
        forecast_origin_ns=origin,basis='immutable_pre_case_history_only',future_labels_used=False,
        state=state,sources=[m.receipt(evidence)])))
    return m.receipt(path)


def execute(directory,data,*,past=True,tables=None,document=None,args=None):
    plan,context,original,requests,original_args,original_doc,state=data
    tables=original if tables is None else tables;args=original_args if args is None else args
    document=original_doc if document is None else document
    doc,sources=model_document(tables,directory);path=Path(directory)/'model.json';path.write_text(json.dumps(doc))
    ref=state_receipt(directory,state) if past else None
    with patch.object(resource,'load_complete',return_value=(tables,sources)) as native:
        result=m.rescore_ensemble(document,requests,path,plan,context,heldout_block_id='training-block-3',
            scoring_plan=args['scoring_plan'],quality=args['quality'],window_start_ns=args['window_start_ns'],
            window_end_ns=args['window_end_ns'],drain_end_ns=args['drain_end_ns'],past_memory_state=ref)
    native.assert_called_once()
    return result


class FoldResourceRescoreTests(unittest.TestCase):
    def test_real_fold_original_score_full_world_and_scope(self):
        data=setup_world();before=copy.deepcopy(data)
        with tempfile.TemporaryDirectory() as directory:out=execute(directory,data)
        self.assertEqual(data,before);row=out['ensemble']['draws'][0];scored=row['forecast']['scoring']
        prediction=scored['resource_prediction'];score=scored['original_score']
        self.assertEqual(len(out['resource_fold_binding']['training_attempt_ids']),8)
        self.assertEqual(out['resource_fold_binding']['training_block_ids'],['training-block-1','training-block-2'])
        self.assertEqual(prediction['startup_energy_joules'],15.)
        self.assertAlmostEqual(prediction['board_energy_joules'],6e-9)
        self.assertEqual(scored['resource_timeline']['total_measurement_overlap_ns'],m.encode(F(2)))
        n=len(scored['resource_timeline']['slices'])
        peak=1500.+10*(n-1)
        self.assertEqual(prediction['peak_memory_mib'],peak)
        self.assertAlmostEqual(score['terms']['M'],peak*1.2/1000)
        self.assertAlmostEqual(score['terms']['E'],(3+15/3600)/10)
        self.assertIsNotNone(score['full_U'])
        self.assertEqual(row['forecast']['world'],data[5]['draws'][0]['forecast']['world'])
        self.assertEqual(set(out['ensemble']['six_terms']),set('QTLPME'))
        self.assertFalse(out['full_U_resource_qualification'])

    def test_missing_past_never_fresh_keeps_E_but_unknown_M_and_U(self):
        with tempfile.TemporaryDirectory() as directory:out=execute(directory,setup_world(),past=False)
        score=out['ensemble']['draws'][0]['forecast']['scoring']
        self.assertIsNone(score['original_score']['terms']['M']);self.assertIsNone(score['original_score']['full_U'])
        self.assertIsNotNone(score['original_score']['terms']['E'])
        self.assertFalse(score['resource_prediction']['memory_initial_history_supplied'])
        self.assertEqual(out['ensemble']['utility']['unknown_probability_mass'],1)

    def test_heldout_future_poison_does_not_change_resource_values(self):
        data=setup_world();poison=copy.deepcopy(data[2])
        for table in poison:
            if table['capture_id']=='training-block-3':
                table.update(startup_joules=1e12,premeasurement_owned_sampled_peak_mib=1e12)
                for row in table['slices']:row.update(power_training_joules=1e12,sampled_owned_peak_mib=1e12)
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            baseline=execute(a,data);other=execute(b,data,tables=poison)
        for field in ('terms','full_U'):
            self.assertEqual(baseline['ensemble']['draws'][0]['forecast']['scoring']['original_score'][field],
                other['ensemble']['draws'][0]['forecast']['scoring']['original_score'][field])
        self.assertEqual(baseline['resource_fold_binding']['selected_fold_model_sha256'],
            other['resource_fold_binding']['selected_fold_model_sha256'])
        self.assertNotEqual(baseline['resource_model'],other['resource_model'])

    def test_partial_support_and_transport_unknown_preserve_mass_and_cohort(self):
        data=setup_world(unknown=True);tables=copy.deepcopy(data[2])
        for table in tables:
            if table['capture_id']=='training-block-2':
                for row in table['slices']:row.update(power_training_duration_ns=0,sampled_owned_peak_mib=None)
        with tempfile.TemporaryDirectory() as directory:out=execute(directory,data,tables=tables)
        value=m.decode(out['ensemble']);self.assertEqual(value['utility']['unknown_probability_mass'],1)
        self.assertEqual(value['six_terms']['E']['unknown_probability_mass'],1)
        self.assertEqual(value['six_terms']['P']['known_probability_mass'],F(1,3))
        self.assertEqual(value['draws'][1],m.decode(data[5])['draws'][1])
        self.assertEqual(value['draws'][1]['all_public_request_ids_retained'],['r0','r1'])
        self.assertIsNone(value['utility']['expected_value'])

    def test_domain_cohort_boundary_and_programming_errors_rejected(self):
        for mutation in ('fold','physical','cohort','drain','weight','score','compact'):
            data=list(setup_world())
            if mutation=='fold':data[0]['capture_id']='training-block-1'
            elif mutation=='physical':data[1]['threads']['OMP_NUM_THREADS']='999'
            elif mutation=='cohort':data[3][0]['arrival_ns']=1
            elif mutation=='drain':data[4]['drain_end_ns']+=1
            elif mutation=='weight':data[5]['draws'][0]['weight']=m.encode(F(1,2))
            elif mutation=='score':data[4]['scoring_plan']['weights']['P']=2
            else:data[5]['draws'][0]['forecast'].pop('world')
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory,self.assertRaises(ValueError):
                execute(directory,data)
        with tempfile.TemporaryDirectory() as directory,patch.object(m,'score_world',side_effect=ValueError('programming')):
            with self.assertRaisesRegex(ValueError,'programming'):execute(directory,setup_world())

    def test_past_contract_rejects_future_fresh_nonfinite_and_drift(self):
        state=setup_world()[-1]
        for mutation in ('future','fresh','hash','nan','drift'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                bad=copy.deepcopy(state)
                if mutation=='fresh':bad=dict(basis='fresh_after_original_priming')
                elif mutation=='hash':bad['last_state_key']='bad'
                elif mutation=='nan':bad['prior_owned_sampled_peak_mib']=float('nan')
                ref=state_receipt(directory,bad,cutoff=1 if mutation=='future' else 0)
                if mutation=='drift':(Path(directory)/'CPU_TEST_ONLY_past_evidence.json').write_text('{}')
                with self.assertRaises(ValueError):m.past_memory(ref,F(0))

    def test_incomplete_completed_report_stops_before_replay_or_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'incomplete.json';path.write_text(json.dumps(dict(schema=m.generated.SCHEMA,
                status='WAITING',numerical_fit_run=False)))
            with patch.object(m,'restore_full_ensemble') as replay,patch.object(resource,'fit_tables') as fit:
                with self.assertRaises(ValueError):m.run_completed_case(path,'unused',heldout_block_id='x',
                    attempt_id='x',case_id='x',output=Path(directory)/'out')
            replay.assert_not_called();fit.assert_not_called();self.assertFalse((Path(directory)/'out').exists())

    def test_compact_replay_recreates_original_full_world_and_hash(self):
        f=archived_fixture;law=f.law();score=f.scoring();quality=f.QUALITY
        prediction=m.generated.forecast_case(f.public_case(),1000,f.past(),law,score,quality)
        compact=m.encode(m.generated.compact_prediction(prediction))
        restored,requests=m.restore_full_ensemble(compact,law,score,quality,donor_blocks=law['blocks'])
        self.assertEqual(restored,prediction['ensemble'])
        self.assertEqual(requests,m.generated.public_case(f.public_case(),1000))
        bad=copy.deepcopy(compact);bad['ensemble']['draws'][0]['forecast_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'archived hash'):
            m.restore_full_ensemble(bad,law,score,quality,donor_blocks=law['blocks'])
        bad=copy.deepcopy(compact);bad['ensemble']['draws'].pop()
        with self.assertRaisesRegex(ValueError,'draw count'):
            m.restore_full_ensemble(bad,law,score,quality,donor_blocks=law['blocks'])

    def test_replay_preserves_all_unknown_seed_and_unqualified_fit(self):
        f=archived_fixture
        for missing_seed in (True,False):
            kwargs={} if missing_seed else dict(unqualified_reason='UNKNOWN_COMPLETE_RESPONSE_FIT_OR_OCR_SUPPORT_UNQUALIFIED')
            law=f.law() if missing_seed else None
            prediction=m.generated.forecast_case(f.public_case(),1000,None if missing_seed else f.past(),
                law,f.scoring(),f.QUALITY,donor_blocks=['training-block-1','training-block-2'],**kwargs)
            compact=m.encode(m.generated.compact_prediction(prediction))
            restored,_=m.restore_full_ensemble(compact,law,f.scoring(),f.QUALITY,
                donor_blocks=['training-block-1','training-block-2'])
            self.assertEqual(restored,prediction['ensemble'])
            self.assertEqual(m.decode(restored)['world_unknown_probability_mass'],1)

    def test_all_unknown_still_enforces_original_score_contract_and_past_state(self):
        data=list(setup_world());requests,args=data[3:5]
        draws=[dict(draw_id='unknown',block_id='training-block-1',weight=F(1),
            inputs=m.ensemble.UnknownForecastInputs('CPU_TEST_ONLY'))]
        data[5]=m.ensemble.evaluate(requests,draws,**args)
        with tempfile.TemporaryDirectory() as directory:out=execute(directory,data)
        self.assertEqual(out['ensemble']['draws'],data[5]['draws'])
        data[4]['scoring_plan']['startup_amortization_seconds']=60
        with tempfile.TemporaryDirectory() as directory,self.assertRaises(ValueError):execute(directory,data)

    def test_completed_case_exact_plan_and_fold_binding_without_labels_or_fit(self):
        design_path=Path('artifacts/predictor-twoFIFO-generated-wholeblock-design-v1-20261028-cpu01/design.json')
        design_ref,design=m.generated.audit_design(design_path)
        campaign=m.checked(design['campaign']);plans=[m.checked(ref) for ref in campaign['attempts']]
        plan=next(p for p in plans if p['capture_id']=='training-block-3');case=plan['cases'][0]
        score=m.checked(design['scoring_plan']);quality=m.checked(design['quality'])
        donors=[p['attempt_id'] for p in plans if p['capture_id']!='training-block-3']
        heldouts=[p['attempt_id'] for p in plans if p['capture_id']=='training-block-3']
        for mutation in (None,'donor','cohort'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);_,_,tables=fixture();model,sources=model_document(tables,folder)
                model_path=folder/'model.json';model_path.write_text(json.dumps(model))
                public=copy.deepcopy(case)
                if mutation=='cohort':public['arrivals'][0]['request_id']='wrong-public-case-request'
                compact=m.generated.compact_prediction(m.generated.forecast_case(public,1000,None,None,score,quality,
                    donor_blocks=['training-block-1','training-block-2'],unqualified_reason='UNQUALIFIED'))
                prediction=m.save(folder/'prediction.json',compact)
                fold=dict(heldout_block_id='training-block-3',training_block_ids=['training-block-1','training-block-2'],
                    donor_attempt_ids=donors,heldout_attempt_ids=heldouts,folded_law=None,
                    cases=[dict(attempt_id=plan['attempt_id'],case_id=case['case_id'],prediction=prediction)])
                if mutation=='donor':fold['donor_attempt_ids']=donors[:-1]+[heldouts[0]]
                report=m.save(folder/'completed.json',dict(schema=m.generated.SCHEMA,numerical_fit_run=True,
                    status='COMPLETE_NATIVE_TRAINING_FIT_UNQUALIFIED_ALL_GENERATED_WORLDS_UNKNOWN',
                    validation_labels_read=False,sources=[design['campaign']]+model['training_reviews'],
                    design=design_ref,whole_block_leaveout=[fold]))
                with patch.object(resource,'load_complete',return_value=(tables,sources)),\
                        patch.object(m.generated,'native_labels') as labels,patch.object(m.generated.response.numeric,'fit') as fit:
                    if mutation:
                        with self.assertRaisesRegex(ValueError,'split|planned public case'):
                            m.run_completed_case(report['path'],model_path,heldout_block_id='training-block-3',
                                attempt_id=plan['attempt_id'],case_id=case['case_id'],output=folder/'out')
                        self.assertFalse((folder/'out').exists())
                    else:
                        ref=m.run_completed_case(report['path'],model_path,heldout_block_id='training-block-3',
                            attempt_id=plan['attempt_id'],case_id=case['case_id'],output=folder/'out')
                        result=m.checked(ref);self.assertFalse(result['numerical_fit_run'])
                        self.assertFalse(result['original_labels_file_opened'])
                        self.assertEqual(m.decode(result['ensemble'])['world_unknown_probability_mass'],1)
                    labels.assert_not_called();fit.assert_not_called()


if __name__=='__main__':unittest.main()
