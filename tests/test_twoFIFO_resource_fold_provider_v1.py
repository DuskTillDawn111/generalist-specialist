import copy
import json
from pathlib import Path
import tempfile
import unittest
from fractions import Fraction
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_resource_fold_provider_v1 as m
from tools.predictor_validation_20261027 import macro_response_queue_reference_v1 as rational

resource=m.resource
CAMPAIGN=Path('artifacts/predictor-twoFIFO-owned-training-design-v3-20261028-cpu01/campaign.json')


def fixture():
    campaign=resource.checked(resource.receipt(CAMPAIGN));plan=resource.checked(campaign['attempts'][0])
    context=resource.checked(plan['context']);owners=plan['resident_owners']
    state,_=resource.legacy.key([],owners)
    history=resource.next_history(resource.history_start('CPU_TEST_ONLY_priming'),state)
    result=[]
    for block in (1,2,3):
        for shard in (1,2,3,4):
            result.append(dict(schema=resource.TABLE_SCHEMA,capture_id='training-block-'+str(block),
                attempt_id='training-block-'+str(block)+'-shard-'+str(shard),
                physical_execution_identity=resource.physical_execution_identity(plan,context),
                physical_execution_payload=resource.physical_execution_payload(plan,context),
                priming_content_identity='CPU_TEST_ONLY_priming',resident_owners=owners,
                startup_joules=10.*block,premeasurement_owned_sampled_peak_mib=100.*block,
                slices=[dict(state_key=state,memory_state_key=resource.memory_key(state,history),
                    power_training_joules=2.*block,power_training_duration_ns=1_000_000_000,
                    sampled_owned_peak_mib=1000.*block)]))
    return plan,context,result


def model_document(tables,directory):
    model=resource.fit_tables(tables);reviews=[]
    for index in range(12):
        path=Path(directory)/('CPU_TEST_ONLY_review-'+str(index)+'.json')
        path.write_text(json.dumps(dict(CPU_TEST_ONLY=True,ordinal=index)))
        reviews.append(resource.receipt(path))
    sources=[resource.receipt(CAMPAIGN)]
    model.update(campaign=resource.receipt(CAMPAIGN),training_reviews=reviews,sources=sources,
        whole_block_leaveout=[dict(heldout_block_id=b,
            training_model=resource.fit_tables([t for t in tables if t['capture_id']!=b]),
            heldout_attempt_ids=[t['attempt_id'] for t in tables if t['capture_id']==b],
            independent_prediction_accuracy_validated=False) for b in model['training_block_ids']])
    return model,sources


def construct(document,tables,sources,plan,context,path):
    path.write_text(json.dumps(document))
    # Only the native-loader boundary is replaced with CPU fixtures. Production
    # always invokes the original twelve-native-review rederivation.
    with patch.object(resource,'load_complete',return_value=(tables,sources)) as load:
        provider=m.FoldResourceTrainingProvider(path,plan,context,heldout_block_id='training-block-3',
            allow_training_resource_hypothesis=True)
    load.assert_called_once()
    return provider


def timeline_and_binding(provider,*,fresh=True):
    timeline=rational.encode(dict(schema='gsdeploy.macro_global_resource_timeline.v1',
        total_wall_ns=Fraction(1_000_000_000),total_measurement_overlap_ns=Fraction(1_000_000_000),
        slices=[dict(duration_ns=Fraction(1_000_000_000),measurement_overlap_ns=Fraction(1_000_000_000),
            resident_owners=provider.target_plan['resident_owners'],actual_START_to_READY_handlers=[])]))
    binding=dict(physical_execution_identity=provider.physical_execution_identity,
        resource_timeline_sha256=resource.digest(timeline),
        basis='caller_declared_explicit_reference_execution_binding')
    if fresh:binding['memory_initial_state']=dict(basis='fresh_after_original_priming')
    return timeline,binding


class ResourceFoldProviderTests(unittest.TestCase):
    def test_heldout_resource_poison_cannot_change_selected_model_or_predictions(self):
        plan,context,tables=fixture()
        with tempfile.TemporaryDirectory() as directory:
            doc,sources=model_document(tables,directory)
            original=construct(doc,tables,sources,plan,context,Path(directory)/'original.json')
            poisoned=copy.deepcopy(tables)
            for table in poisoned:
                if table['capture_id']=='training-block-3':
                    table['startup_joules']=1e12;table['premeasurement_owned_sampled_peak_mib']=1e12
                    table['slices'][0].update(power_training_joules=1e12,sampled_owned_peak_mib=1e12)
            poisoned_doc,sources=model_document(poisoned,directory)
            other=construct(poisoned_doc,poisoned,sources,plan,context,Path(directory)/'poisoned.json')
            self.assertEqual(original.document,other.document)
            self.assertEqual(len(original.document['training_attempt_ids']),8)
            self.assertNotIn('training-block-3',original.document['training_block_ids'])
            self.assertEqual(original.startup()['joules'],15.)
            self.assertEqual(original.startup()['premeasurement_owned_sampled_peak_mib'],150.)
            for provider in (original,other):
                timeline,binding=timeline_and_binding(provider)
                result=m.predict_timeline(timeline,provider,target_context=context,execution_binding=binding)
                self.assertEqual(result['board_energy_joules'],3.)
                self.assertEqual(result['peak_memory_mib'],1500.)
                self.assertEqual(result['startup_energy_joules'],15.)
                self.assertFalse(result['resource_prediction_qualified'])

    def test_unsupported_donor_state_or_history_does_not_borrow_heldout_support(self):
        plan,context,tables=fixture()
        for table in tables:
            if table['capture_id']=='training-block-2':
                table['slices'][0].update(power_training_duration_ns=0,sampled_owned_peak_mib=None)
        with tempfile.TemporaryDirectory() as directory:
            doc,sources=model_document(tables,directory)
            provider=construct(doc,tables,sources,plan,context,Path(directory)/'model.json')
            timeline,binding=timeline_and_binding(provider)
            result=m.predict_timeline(timeline,provider,target_context=context,execution_binding=binding)
            self.assertIsNone(result['board_energy_joules']);self.assertIsNone(result['peak_memory_mib'])
            self.assertFalse(provider.memory([],resident_owners=plan['resident_owners'],
                ordered_state_history_identity='f'*64)['known'])

    def test_seed_is_never_initial_history_and_unseen_prior_remains_UNKNOWN(self):
        plan,context,tables=fixture();plan['runtime_config']['random_seed']=987654
        with tempfile.TemporaryDirectory() as directory:
            doc,sources=model_document(tables,directory)
            provider=construct(doc,tables,sources,plan,context,Path(directory)/'model.json')
            timeline,binding=timeline_and_binding(provider,fresh=False)
            missing=m.predict_timeline(timeline,provider,target_context=context,execution_binding=binding)
            self.assertEqual(missing['board_energy_joules'],3.)
            self.assertIsNone(missing['peak_memory_mib']);self.assertFalse(missing['memory_initial_history_supplied'])
            binding['memory_initial_state']=dict(basis='explicit_prior_attempt_history',
                ordered_state_history_identity='0'*64,last_state_key='1'*64,prior_owned_sampled_peak_mib=9000.)
            unknown=m.predict_timeline(timeline,provider,target_context=context,execution_binding=binding)
            self.assertIsNone(unknown['peak_memory_mib'])
            self.assertEqual(unknown['premeasurement_owned_sampled_peak_mib'],9000.)

    def test_every_fold_full_model_and_physical_domain_are_reverified(self):
        plan,context,tables=fixture()
        with tempfile.TemporaryDirectory() as directory:
            doc,sources=model_document(tables,directory);path=Path(directory)/'model.json'
            for corrupt in ('other-fold','missing-fold','duplicate-fold','full-model'):
                changed=copy.deepcopy(doc)
                if corrupt=='other-fold':changed['whole_block_leaveout'][0]['training_model']['startup']['joules']+=1
                elif corrupt=='missing-fold':changed['whole_block_leaveout'].pop(0)
                elif corrupt=='duplicate-fold':changed['whole_block_leaveout'][0]=changed['whole_block_leaveout'][1]
                else:changed['startup']['joules']+=1
                with self.assertRaises(ValueError):construct(changed,tables,sources,plan,context,path)
            altered=copy.deepcopy(context);altered['threads']['OMP_NUM_THREADS']='999'
            with self.assertRaisesRegex(ValueError,'physical'):construct(doc,tables,sources,plan,altered,path)

    def test_missing_donor_startup_is_UNKNOWN_and_incomplete_native_gate_precedes_fitting(self):
        plan,context,tables=fixture();tables[0]['startup_joules']=None
        with tempfile.TemporaryDirectory() as directory:
            doc,sources=model_document(tables,directory);path=Path(directory)/'model.json'
            provider=construct(doc,tables,sources,plan,context,path)
            self.assertIsNone(provider.startup()['joules'])
            doc['training_reviews']=[];path.write_text(json.dumps(doc))
            campaign=resource.checked(resource.receipt(CAMPAIGN))
            with patch.object(resource.intake.campaign_v2,'audit',return_value=campaign),\
                    patch.object(resource.macros_v2,'extract') as extract,patch.object(resource,'fit_tables') as fit:
                with self.assertRaisesRegex(ValueError,'All twelve'):
                    m.FoldResourceTrainingProvider(path,plan,context,heldout_block_id='training-block-3',
                        allow_training_resource_hypothesis=True)
            extract.assert_not_called();fit.assert_not_called()


if __name__=='__main__':unittest.main()
