import copy
import unittest

from tools.predictor_validation_20261027 import local_joint_macro_candidate_v1 as joint


def atom(name,response,end_notice,host_end=5,query_at=7,query_notice=9):
    def m(ident,at,available,**payload):
        return dict(mark_id=ident,occurrence_offset_ns=at,reader_delay_ns=available-at,
            dequeue_extra_ns=0,processing_extra_ns=0,payload=payload)
    return dict(atom_id=name,response_ns=response,marks=[m('BEGIN',0,2),
        m('HOST_END',host_end,6),m('QUERY:0',query_at,query_notice,end_event_complete=True),
        m('TERMINAL',response,end_notice)])


def features():
    return dict(owner='generalist_2#0',public_input_id='known-public-content-SHA',semantic_phase='decode',prompt_sequence_length=389,
        input_batch_size=1,forward_index=4,terminal_kind='native_END',
        peer_macro_count_at_actual_BEGIN=0,peer_same_phase_count_at_actual_BEGIN=0)


class LocalJointCandidateTests(unittest.TestCase):
    def test_missing_terminal_keeps_done_and_ongoing_even_after_true_query(self):
        atoms=[atom('done',10,50),atom('ongoing',40,60)]
        visible=[e for e in joint.realize(atoms[0],100)['notifications'] if e['processed_ns']<=120]
        result=joint.condition_atoms(atoms,visible,begin_ns=100,cutoff_ns=120,registry_complete_through_ns=120)
        self.assertEqual(result['completed_response_mass'],.5)
        self.assertEqual([w['remaining_response_ns'] for w in result['worlds']],[0,20])
        self.assertFalse(result['notification_absence_implies_unfinished'])
        self.assertIsNone(result['all_request_GPU_work_complete'])

    def test_joint_draw_does_not_pair_independent_milestones_or_delay(self):
        a=atom('short',10,50);b=atom('long',40,60)
        law=dict(schema=joint.SCHEMA,cells=[dict(features=features(),atoms=[a,b])])
        for quantile,expected in ((0,'short'),(.75,'long')):
            result=joint.draw(law,features(),begin_ns=100,quantile=quantile)
            self.assertEqual(result['atom_id'],expected)
            original=a if expected=='short' else b
            self.assertEqual(result['response_END_ns'],100+original['response_ns'])
            self.assertEqual(result['notifications'][-1]['processed_ns'],100+(50 if expected=='short' else 60))

    def test_response_conditional_sampling_requires_exact_supported_response(self):
        law=dict(schema=joint.SCHEMA,cells=[dict(features=features(),atoms=[atom('a',10,50),atom('b',40,60)])])
        result=joint.draw(law,features(),begin_ns=0,quantile=0,response_ns=40)
        self.assertEqual(result['atom_id'],'b')
        self.assertEqual(joint.draw(law,features(),begin_ns=0,quantile=0,response_ns=30)['status'],
            'UNKNOWN_NO_EMPIRICAL_LOCAL_SUPPORT')

    def test_cross_macro_notifications_survive_and_FIFO_conflict_is_not_repaired(self):
        law=dict(schema=joint.SCHEMA,cells=[dict(features=features(),atoms=[atom('a',10,50)])])
        result=joint.draw(law,features(),begin_ns=100,quantile=0)
        self.assertGreater(result['notifications'][-1]['processed_ns'],result['response_END_ns'])
        conflict=joint.draw(law,features(),begin_ns=111,quantile=0,
            prior_notification_watermarks=dict(reader_receipt_ns=150,dequeued_ns=150,processed_ns=150))
        self.assertEqual(conflict['status'],'UNKNOWN_CROSS_MACRO_FIFO_CONFLICT')

    def test_no_cohort_identity_lookup_and_detached_outputs(self):
        fs=features();bad=dict(fs,request_id='historical:0')
        with self.assertRaises(ValueError):joint.canonical(bad)
        law=dict(schema=joint.SCHEMA,cells=[dict(features=fs,atoms=[atom('a',10,50)])]);before=copy.deepcopy(law)
        result=joint.draw(law,fs,begin_ns=0,quantile=0);result['notifications'][0]['payload']['edited']=True
        self.assertEqual(law,before)
        other=dict(fs,prompt_sequence_length=4665)
        self.assertEqual(joint.draw(law,other,begin_ns=0,quantile=0)['status'],'UNKNOWN_NO_EMPIRICAL_LOCAL_SUPPORT')
        same_length_other_content=dict(fs,public_input_id='different-content-SHA')
        self.assertEqual(joint.draw(law,same_length_other_content,begin_ns=0,quantile=0)['status'],
            'UNKNOWN_NO_EMPIRICAL_LOCAL_SUPPORT')

    def test_local_FIFO_contradiction_rejected(self):
        a=atom('bad',10,50);a['marks'][2]['reader_delay_ns']=100
        with self.assertRaisesRegex(ValueError,'FIFO'):joint.realize(a,0)

    def test_incomplete_registry_and_no_exact_support_are_unknown(self):
        a=atom('a',10,50)
        self.assertEqual(joint.condition_atoms([a],[],begin_ns=0,cutoff_ns=10,registry_complete_through_ns=9)['status'],
            'UNKNOWN_INCOMPLETE_PROCESSING_REGISTRY')
        self.assertEqual(joint.condition_atoms([a],[],begin_ns=0,cutoff_ns=10,registry_complete_through_ns=10)['status'],
            'UNKNOWN_ZERO_EXACT_JOINT_SUPPORT')

    def test_future_outcomes_stay_pending_not_assimilated(self):
        a=atom('a',10,50);visible=joint.realize(a,0)['notifications'][:1]
        result=joint.condition_atoms([a],visible,begin_ns=0,cutoff_ns=3,registry_complete_through_ns=3)
        self.assertEqual(len(result['worlds'][0]['pending_notifications']),3)
        self.assertFalse(result['main_queue_provider_eligible'])
        with self.assertRaises(ValueError):joint.condition_atoms([a],joint.realize(a,0)['notifications'],
            begin_ns=0,cutoff_ns=3,registry_complete_through_ns=3)

    def test_final_query_occurs_in_joint_READY_tail_and_references_old_forward(self):
        a=atom('final',10,12,query_at=7,query_notice=9)
        a['marks'][2]['payload']['end_event_complete']=False
        a['following_ready_tail_ns']=20
        def mark(ident,at,delay,payload):
            return dict(mark_id=ident,occurrence_offset_ns=at,reader_delay_ns=delay,
                dequeue_extra_ns=0,processing_extra_ns=0,payload=payload)
        a['marks'].append(mark('QUERY:1',15,2,dict(end_event_complete=True,source_macro_id='decode_0004',
            occurrence_macro_id='native_end_to_worker_ready',policy_site='observer_context_exit_poll')))
        a['marks'].append(mark('RESULT',25,4,dict(semantic_kind='RESULT')))
        a['marks'].append(mark('READY',30,2,dict(semantic_kind='WORKER_READY')))
        world=joint.realize(a,100)
        true=next(n for n in world['notifications'] if n['mark_id']=='QUERY:1')
        self.assertGreater(true['occurrence_ns'],world['response_END_ns'])
        self.assertEqual(true['payload']['source_macro_id'],'decode_0004')
        self.assertEqual(true['payload']['occurrence_macro_id'],'native_end_to_worker_ready')
        self.assertEqual(world['ready_tail_END_ns'],130)
        self.assertFalse(world['query_true_ends_macro'])
        visible=[n for n in world['notifications'] if n['processed_ns']<=111]
        posterior=joint.condition_atoms([a],visible,begin_ns=100,cutoff_ns=111,registry_complete_through_ns=111)
        self.assertEqual(posterior['completed_response_mass'],1)
        self.assertEqual({n['mark_id'] for n in posterior['worlds'][0]['pending_notifications']},
            {'TERMINAL','QUERY:1','RESULT','READY'})

    def test_final_tail_is_sampled_with_forward_and_never_rematched_independently(self):
        a=atom('short_plus_long_tail',10,50);a['following_ready_tail_ns']=100
        b=atom('long_plus_short_tail',40,60);b['following_ready_tail_ns']=5
        law=dict(schema=joint.SCHEMA,cells=[dict(features=features(),atoms=[a,b])])
        first=joint.draw(law,features(),begin_ns=0,quantile=0)
        last=joint.draw(law,features(),begin_ns=0,quantile=.9)
        self.assertEqual((first['response_END_ns'],first['following_ready_tail_ns']),(10,100))
        self.assertEqual((last['response_END_ns'],last['following_ready_tail_ns']),(40,5))


if __name__=='__main__':unittest.main()
