import copy
import json
from pathlib import Path
import tempfile
import unittest
from fractions import Fraction as F
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_resource_provider_v3 as m
from tools.predictor_validation_20261027 import macro_response_queue_reference_v1 as rational

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'artifacts/predictor-twoFIFO-owned-training-design-v2-20261028-cpu01/campaign.json'


def tables():
    result=[]
    for block in (1,2,3):
        for shard in (1,2,3,4):
            result.append(dict(schema=m.TABLE_SCHEMA,capture_id='training-block-'+str(block),
                attempt_id='training-block-'+str(block)+'-shard-'+str(shard),
                physical_execution_identity='physical',physical_execution_payload={'example':'CPU_TEST_ONLY'},
                priming_content_identity='prime',resident_owners=['text#0'],startup_joules=10.*block,
                premeasurement_owned_sampled_peak_mib=100.*block,
                slices=[dict(state_key='state',memory_state_key='history',power_training_joules=2.*block,
                    power_training_duration_ns=1_000_000_000,sampled_owned_peak_mib=1000.*block)]))
    return result


class ResourceProviderV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        campaign=m.checked(m.receipt(CAMPAIGN));cls.plan=m.checked(campaign['attempts'][0])
        cls.context=m.checked(cls.plan['context'])

    def test_physical_identity_separates_arrival_plan_from_workers_and_new_knobs(self):
        plan=copy.deepcopy(self.plan);context=copy.deepcopy(self.context)
        original=m.physical_execution_identity(plan,context)
        plan.update(execution_id='new-execution',execution_context_sha256='new-plan',cases=[])
        plan['runtime_config']['routing_rps']={};plan['runtime_config']['random_seed']=123
        self.assertEqual(m.physical_execution_identity(plan,context),original)
        context['threads']['OMP_NUM_THREADS']='99'
        self.assertNotEqual(m.physical_execution_identity(plan,context),original)
        plan['runtime_config']['new_physical_knob']=True
        with self.assertRaises(ValueError):m.physical_execution_identity(plan,context)

    def test_fit_uses_whole_blocks_equal_weight_and_every_four_shards(self):
        source=tables();model=m.fit_tables(source)
        self.assertEqual(model['power_states']['state']['mean'],4.)
        self.assertEqual(model['memory_history_states']['history']['mean'],2000.)
        self.assertEqual(model['startup']['joules'],20.)
        self.assertEqual(len(model['training_block_ids']),3)
        for block in model['training_block_ids']:
            fold=m.fit_tables([t for t in source if t['capture_id']!=block])
            self.assertEqual(len(fold['training_attempt_ids']),8)
        with self.assertRaises(ValueError):m.fit_tables(source[:-1])
        changed=copy.deepcopy(source);changed[0]['physical_execution_identity']='other'
        with self.assertRaises(ValueError):m.fit_tables(changed)

    def test_single_block_state_and_missing_startup_remain_UNKNOWN(self):
        source=tables()
        for table in source:
            if table['capture_id']!='training-block-1':
                table['slices'][0].update(power_training_duration_ns=0,sampled_owned_peak_mib=None)
        source[0]['startup_joules']=None
        model=m.fit_tables(source)
        self.assertFalse(model['power_states']['state']['known'])
        self.assertIsNone(model['power_states']['state']['mean'])
        self.assertFalse(model['memory_history_states']['history']['known'])
        self.assertIsNone(model['startup']['joules'])

    def test_partial_campaign_cannot_read_resources_or_extract(self):
        with patch.object(m.macros_v2,'extract') as extract:
            with self.assertRaisesRegex(ValueError,'HOLD_V2_IN_WINDOW'):m.load_complete(CAMPAIGN,[])
        extract.assert_not_called()

    def test_new_v3_partial_gate_waits_for_all_twelve_instead_of_legacy_HOLD(self):
        campaign=m.checked(m.receipt(CAMPAIGN));campaign['schema']=m.intake.campaign_v2.SCHEMA
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'campaign.json';path.write_text(json.dumps(campaign))
            with patch.object(m.intake.campaign_v2,'audit',return_value=campaign),patch.object(m.macros_v2,'extract') as extract:
                with self.assertRaisesRegex(ValueError,'All twelve'):m.load_complete(path,[])
            extract.assert_not_called()

    def test_prediction_global_measurement_energy_and_cumulative_memory_not_summed(self):
        plan=self.plan;context=self.context;physical=m.physical_execution_identity(plan,context)
        class Provider:
            target_plan=plan;physical_execution_identity=physical
            document={'priming_content_identity':'prime'}
            def power(self,active,**kw):return dict(known=True,watts=20.)
            def memory(self,active,**kw):return dict(known=True,peak_memory_mib=300.)
            def startup(self):return dict(joules=40.,premeasurement_owned_sampled_peak_mib=400.)
        timeline=rational.encode(dict(schema='gsdeploy.macro_global_resource_timeline.v1',
            total_wall_ns=F(4_000_000_000),total_measurement_overlap_ns=F(1_000_000_000),slices=[
            dict(duration_ns=F(2_000_000_000),measurement_overlap_ns=F(1_000_000_000),
                 resident_owners=list(m.intake.campaign_v2.OWNERS),actual_START_to_READY_handlers=[]),
            dict(duration_ns=F(2_000_000_000),measurement_overlap_ns=F(0),
                 resident_owners=list(m.intake.campaign_v2.OWNERS),actual_START_to_READY_handlers=[])]))
        binding=dict(physical_execution_identity=physical,resource_timeline_sha256=m.digest(timeline),
            basis='caller_declared_explicit_reference_execution_binding',
            memory_initial_state=dict(basis='fresh_after_original_priming'))
        result=m.predict_timeline(timeline,Provider(),target_context=context,execution_binding=binding)
        self.assertEqual(result['board_energy_joules'],20.)
        self.assertEqual(result['peak_memory_mib'],400.)
        self.assertEqual(result['startup_energy_joules'],40.)
        self.assertFalse(result['startup_energy_already_amortized'])
        self.assertFalse(result['safe_factor_already_applied'])
        missing=copy.deepcopy(binding);missing.pop('memory_initial_state')
        unknown=m.predict_timeline(timeline,Provider(),target_context=context,execution_binding=missing)
        self.assertIsNone(unknown['peak_memory_mib']);self.assertEqual(unknown['board_energy_joules'],20.)
        self.assertFalse(unknown['memory_initial_history_supplied'])
        prior=copy.deepcopy(binding);prior['memory_initial_state']=dict(basis='explicit_prior_attempt_history',
            ordered_state_history_identity='0'*64,last_state_key='1'*64,prior_owned_sampled_peak_mib=900.)
        cumulative=m.predict_timeline(timeline,Provider(),target_context=context,execution_binding=prior)
        self.assertEqual(cumulative['peak_memory_mib'],900.)
        binding['physical_execution_identity']='wrong'
        with self.assertRaises(ValueError):m.predict_timeline(timeline,Provider(),target_context=context,execution_binding=binding)

    def test_lookup_history_unknown_does_not_borrow_other_allocator_history(self):
        provider=object.__new__(m.ResourceTrainingProvider);provider.scenario='mean'
        owners=['text#0'];state,_=m.legacy.key([],owners)
        provider.document=dict(resident_owners=owners,memory_history_states={
            m.memory_key(state,'known'):dict(known=True,mean=1000.)})
        self.assertTrue(provider.memory([],resident_owners=owners,ordered_state_history_identity='known')['known'])
        result=provider.memory([],resident_owners=owners,ordered_state_history_identity='unseen')
        self.assertFalse(result['known']);self.assertIsNone(result['peak_memory_mib'])


if __name__=='__main__':unittest.main()
