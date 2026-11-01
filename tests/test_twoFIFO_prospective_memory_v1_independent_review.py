"""Independent generated-processing, observation and absolute-demand contracts."""
import copy
import unittest

from tests.test_twoFIFO_prospective_memory_v1 import inputs,absolute_fixture,OWNERS,PHYSICAL
from tools.predictor_validation_20261027 import twoFIFO_prospective_memory_v1 as m


def options(model,observation):
    return dict(resident_owners=OWNERS,physical_execution_identity=PHYSICAL,absolute_model=model,
        memory_origin=dict(basis='selected_training_donor_after_original_priming',donor_attempt_id='training-block-1-shard-1'),
        observation_plan=observation)


class ProspectiveMemoryIndependentReview(unittest.TestCase):
    def test_latent_reader_and_pending_prefill_shape_do_not_become_processed_state(self):
        _,sidecar,world,requests,initial=inputs()
        notice=next(b for b in world['trace'] if b.get('kind')=='processed_notification' and b.get('phase')=='prefill')
        world['trace'].remove(notice)
        world['trace'].append(dict(notice,kind='protocol_reader_available',reader_ns=notice['available_ns']))
        world['pending_notifications']=[copy.deepcopy(notice)]
        out=m.engine.decode(m.generated_shape_intervals(sidecar,world,requests,resident_owners=OWNERS,initial_descriptor=initial))
        for row in out['intervals']:
            if row['right_ns']<=20:
                active=row['state']['shape_state']['current_processed_owner_task_exact_input_phase_shape']
                self.assertTrue(all(a['phase']!='prefill' and a['actual_BEGIN_shape'] is None for a in active))
                self.assertEqual(row['state']['shape_state']['prior_processed_shape_highwater'],[])
        self.assertFalse(any(r['processing_ns']==10 for r in out['processed_shape_marks']))

    def test_processed_occurrence_and_reader_chronology_cannot_expose_future_shape(self):
        for mutation in ('future_occurrence','negative_occurrence','reader_before_occurrence','reader_after_processing'):
            _,sidecar,world,requests,initial=inputs()
            body=next(b for b in world['trace'] if b.get('kind')=='processed_notification' and b.get('phase')=='prefill')
            if mutation=='future_occurrence':body['occurrence_ns']=body['available_ns']+1
            elif mutation=='negative_occurrence':body['occurrence_ns']=-1
            elif mutation=='reader_before_occurrence':body['reader_receipt_ns']=body['occurrence_ns']-1
            else:body['reader_receipt_ns']=body['available_ns']+1
            with self.assertRaises(ValueError):
                m.generated_shape_intervals(sidecar,world,requests,resident_owners=OWNERS,initial_descriptor=initial)

    def test_two_same_key_public_ordinals_bind_distinct_shapes_independently_of_row_order(self):
        vector,_,base,requests,initial=inputs();other=copy.deepcopy(vector)
        next(n for n in other['nodes'] if n['phase']=='prefill')['forward_shape']['input_sequence_length']=64
        sidecar=m.selected_shape_sidecar([dict(vector=v,public_key_arrival_ordinal=i) for i,v in enumerate((vector,other))],
            physical_execution_identity=PHYSICAL)
        trace=[]
        for i,slot in enumerate(sidecar['slots']):
            marks={(mark['action'],mark['phase'],mark['macro_id']):mark for n in slot['nodes'] for mark in n['marks']}
            for body in base['trace']:
                if body['kind']!='processed_notification':continue
                body=copy.deepcopy(body);body.update(request_id='public-'+str(i),available_ns=body['available_ns']+40*i,
                    occurrence_ns=body['occurrence_ns']+40*i)
                if body['action']!='ready':body['identity']=marks[(body['action'],body['phase'],body['macro_id'])]['generated_mark_identity']
                trace.append(body)
        world=dict(base,trace=trace,offered_requests=2,RESULT_receipts={'public-0':32,'public-1':72},original_window=[0,80],cutoff_ns=80)
        public=[dict(requests[0],request_id='public-'+str(i),arrival_ns=i) for i in (1,0)]
        out=m.engine.decode(m.generated_shape_intervals(sidecar,world,public,resident_owners=OWNERS,initial_descriptor=initial))
        actual={}
        for row in out['processed_shape_marks']:
            active=row['state_after']['shape_state']['current_processed_owner_task_exact_input_phase_shape']
            if active and active[0]['phase']=='prefill':actual[row['request_id']]=active[0]['actual_BEGIN_shape']['input_sequence_length']
        self.assertEqual(actual,{'public-0':16,'public-1':64})
        corrupt=copy.deepcopy(world)
        first=next(b for b in trace if b['request_id']=='public-0' and b['phase']=='prefill')
        last=next(b for b in corrupt['trace'] if b['request_id']=='public-1' and b['phase']=='prefill')
        last['identity']=first['identity']
        with self.assertRaisesRegex(ValueError,'Unbound generated mark'):
            m.generated_shape_intervals(sidecar,corrupt,public,resident_owners=OWNERS,initial_descriptor=initial)

    def test_nonshape_processed_diagnostic_and_reporting_queries_do_not_add_demand(self):
        sidecar,world,requests,tables,observation=absolute_fixture();model=m.fit_absolute_tables(tables[:8])
        baseline=m.rollout(sidecar,world,requests,**options(model,observation))
        query=next(mark for slot in sidecar['slots'] for node in slot['nodes'] if node['phase']=='prefill'
            for mark in node['marks'] if mark['action']=='diagnostic')
        changed=copy.deepcopy(world)
        changed['trace'].append(dict(kind='processed_notification',owner='text#0',request_id='generated',
            identity=query['generated_mark_identity'],action=query['action'],phase=query['phase'],macro_id=query['macro_id'],
            occurrence_ns=11,reader_receipt_ns=11,available_ns=12))
        changed['trace'].sort(key=lambda b:b.get('available_ns',b.get('occurrence_ns',0)))
        after=m.rollout(sidecar,changed,requests,query_times=[12]*100,**options(model,observation))
        self.assertEqual(after['generated_shape_intervals']['intervals'],baseline['generated_shape_intervals']['intervals'])
        self.assertEqual(after['intervals'],baseline['intervals']);self.assertEqual(after['peak_memory_mib'],150)
        self.assertTrue(all(q['prediction']==after['queries'][0]['prediction'] for q in after['queries']))

    def test_boundary_touching_and_outside_polls_remain_in_denominator_without_imputed_support(self):
        rows=m._coalesce([dict(left_ns=0,right_ns=10,state={'state':'first'}),dict(left_ns=10,right_ns=20,state={'state':'second'})])
        polls=[dict(poll_begin_ns=t,poll_end_ns=t,sample_ns=t) for t in (5,10,15,21)]
        assigned,labels=m.attach_observation_support(rows,polls);assigned=m.engine.decode(assigned)
        self.assertEqual(len(labels),4)
        self.assertEqual([a['raw_poll_ordinal'] for a in labels],list(range(4)))
        self.assertEqual(labels[1]['interval_ids'],[0,1]);self.assertEqual(labels[3]['interval_ids'],[])
        self.assertEqual([r['crossing_poll_ordinals'] for r in assigned],[[1],[1]])
        self.assertTrue(all(r['observation_support_status'].startswith('UNKNOWN') for r in assigned))
        self.assertEqual([len(r['contained_polls']) for r in assigned],[1,1])
        clean,_=m.attach_observation_support(rows,polls[:1])
        self.assertTrue(clean[0]['observation_support_status'].startswith('SUPPORTED'))
        self.assertTrue(clean[1]['observation_support_status'].startswith('UNKNOWN'))

    def test_nonfinite_negative_or_boolean_absolute_targets_and_demands_are_rejected(self):
        for invalid in (-123,float('nan'),float('inf'),True):
            sidecar,world,requests,tables,observation=absolute_fixture()
            bad=copy.deepcopy(tables[:8]);bad[0]['intervals'][0]['absolute_sampled_peak_target_mib']=invalid
            with self.assertRaises(ValueError):m.fit_absolute_tables(bad)
            model=m.fit_absolute_tables(tables[:8])
            for state in model['states'].values():state['max']=invalid
            with self.assertRaises(ValueError):m.rollout(sidecar,world,requests,**options(model,observation))

    def test_support_body_key_and_actual_duration_must_agree_before_numeric_result(self):
        for self_consistent_hash in (False,True):
            sidecar,world,requests,tables,observation=absolute_fixture();bad=copy.deepcopy(tables[:8])
            row=bad[0]['intervals'][0];row['support']['duration_ns']=m.encode(m.rational(999))
            if self_consistent_hash:row['support_key']=m.digest(row['support'])
            with self.assertRaises(ValueError):m.fit_absolute_tables(bad)
        model=m.fit_absolute_tables(tables[:8])
        for state in model['states'].values():state['support']['duration_ns']=999
        with self.assertRaisesRegex(ValueError,'support body/key'):
            m.rollout(sidecar,world,requests,**options(model,observation))

    def test_unsupported_interval_keeps_full_peak_unknown_and_partial_max_explicit(self):
        sidecar,world,requests,tables,observation=absolute_fixture();model=m.fit_absolute_tables(tables[:8])
        value=m.engine.decode(m.rollout(sidecar,world,requests,**options(model,observation)))
        key=value['intervals'][2]['support_key'];del model['states'][key]
        result=m.engine.decode(m.rollout(sidecar,world,requests,query_times=[12,12,12],**options(model,observation)))
        self.assertEqual(len(result['intervals']),len(value['intervals']))
        self.assertIsNone(result['peak_memory_mib']);self.assertIsNotNone(result['partial_known_peak_mib'])
        self.assertEqual(sum(r['absolute_D_mib'] is None for r in result['intervals']),1)
        self.assertTrue(all(r['predicted_prefix_peak_mib'] is None for r in result['intervals'][2:]))
        for flag in ('provider_replaced','resource_prediction_qualified','main_queue_provider_eligible','final_gain_eligible',
                'capacity_bound','probability_law_qualified','GPU_started','validation_labels_read'):
            self.assertIs(result[flag],False)


if __name__=='__main__':unittest.main()
