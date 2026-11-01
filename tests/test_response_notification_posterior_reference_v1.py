import copy
import unittest
from tools.predictor_validation_20261027 import response_notification_posterior_reference_v1 as model


def world(name,response,end_delay,mass=.5,*,begin_delay=30,begin_processing=0,end_processing=0):
    delay=lambda reader,extra:dict(reader_delay_ns=reader,processing_extra_ns=extra)
    return dict(id=name,mass=mass,begin_ns=0,response_ns=response,shape_template_id='prior_shape',
        BEGIN=delay(begin_delay,begin_processing),END=delay(end_delay,end_processing) if end_delay is not None else None)

def begin(available=30):return dict(edge='BEGIN',occurrence_ns=0,available_ns=available,shape_template_id='prior_shape')

def posterior(worlds,events=None,cutoff=35,endpoint='parent_processing',closed=35):
    return model.condition(worlds,[begin()] if events is None else events,cutoff_ns=cutoff,
        visibility_endpoint=endpoint,registry_complete_through_ns=closed)


class JointPosteriorReferenceTests(unittest.TestCase):
    def test_delayed_BEGIN_and_missing_END_retain_already_done_and_ongoing_worlds(self):
        result=posterior([world('ended_pending_notice',10,40),world('ongoing',50,10)])
        self.assertEqual(result['completed_response_mass'],.5)
        self.assertEqual(result['ongoing_response_mass'],.5)
        self.assertEqual(result['mean_latent_remaining_host_response_ns'],7.5)
        self.assertFalse(result['notification_absence_implies_unfinished'])
        self.assertIsNone(result['kernel_cursor'])
        self.assertEqual([r['remaining_END_notice_ns'] for r in result['worlds']],[15,25])

    def test_independent_marginals_change_posterior_and_are_not_the_joint_contract(self):
        joint=posterior([world('a',10,40,begin_delay=5),world('b',50,10,begin_delay=5)],events=[begin(5)])
        independent=posterior([world('a1',10,40,.25,begin_delay=5),world('a2',10,10,.25,begin_delay=5),
            world('b1',50,40,.25,begin_delay=5),world('b2',50,10,.25,begin_delay=5)],events=[begin(5)])
        # Product-marginal shortcut assigns a counterfactual extra response-delay pairing.
        self.assertEqual(joint['completed_response_mass'],.5)
        self.assertAlmostEqual(independent['completed_response_mass'],1/3)

    def test_reader_and_processed_cutoffs_are_distinct(self):
        candidate=world('ended',10,5,1.,begin_delay=5,begin_processing=25,end_processing=30)
        reader=model.condition([candidate],[begin(5),dict(edge='END',occurrence_ns=10,available_ns=15,shape_template_id=None)],
            cutoff_ns=30,visibility_endpoint='reader_receipt',registry_complete_through_ns=30)
        processed=posterior([candidate],events=[begin()],cutoff=30,closed=30)
        self.assertEqual(reader['worlds'][0]['remaining_END_notice_ns'],0)
        self.assertEqual(processed['worlds'][0]['remaining_END_notice_ns'],15)
        self.assertEqual(processed['completed_response_mass'],1.)
        self.assertFalse(processed['response_completed_is_observed_fact'])

    def test_future_body_is_never_read_and_cannot_retroactively_change_prefix(self):
        class Forbidden(dict):
            def __getitem__(self,key):raise AssertionError('Future body read')
        envelopes=[dict(available_ns=30,body={k:v for k,v in begin().items() if k!='available_ns'}),
                   dict(available_ns=60,body=Forbidden())]
        result=model.visible_notifications(envelopes,35)
        self.assertEqual(result,[begin()])
        envelopes[1]['body']={'arbitrary_future_output':'must not enter features'}
        self.assertEqual(model.visible_notifications(envelopes,35),result)

    def test_loss_and_no_support_are_explicit_not_zero_or_survival_fallbacks(self):
        result=posterior([world('lost',10,None,1.)])
        self.assertEqual(result['END_lost_mass'],1.)
        self.assertIsNone(result['worlds'][0]['remaining_END_notice_ns'])
        result=posterior([world('already_visible_END',10,5,1.,begin_delay=5)])
        self.assertEqual(result['status'],'UNKNOWN_ZERO_JOINT_SUPPORT')
        result=posterior([world('a',50,10)],closed=34)
        self.assertEqual(result['status'],'UNKNOWN_INCOMPLETE_VISIBLE_REGISTRY')

    def test_observed_END_updates_physical_end_only_after_availability(self):
        events=[begin(),dict(edge='END',occurrence_ns=10,available_ns=50,shape_template_id=None)]
        with self.assertRaisesRegex(ValueError,'already-visible'):
            posterior([world('a',10,40,1.)],events=events,cutoff=49,closed=49)
        result=posterior([world('a',10,40,1.),world('b',50,10,1.)],events=events,cutoff=50,closed=50)
        self.assertEqual([w['id'] for w in result['worlds']],['a'])
        self.assertIsNone(result['all_request_GPU_work_complete'])

    def test_shape_only_enters_after_visible_BEGIN_and_inputs_detached(self):
        worlds=[world('a',50,10,1.,begin_delay=40)];before=copy.deepcopy(worlds)
        empty=posterior(worlds,events=[],cutoff=35,closed=35)
        self.assertEqual(empty['worlds'][0]['id'],'a')
        self.assertEqual(worlds,before)
        bad=[dict(begin(),shape_template_id='future_actual_shape')]
        self.assertEqual(posterior(worlds,events=bad)['status'],'UNKNOWN_ZERO_JOINT_SUPPORT')


if __name__=='__main__':unittest.main()
