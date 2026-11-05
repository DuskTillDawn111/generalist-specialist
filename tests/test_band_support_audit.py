"""Rare exposure cannot disappear, and one owner cannot occupy two phases."""
import copy
import unittest

from experiments.audit_activation_band_support import audit, possible_vertices
from tests.test_activation_semantic_bands import band_template


def document(run, overlap):
    bstart=1_999_999 if overlap else 12_000_000
    bend=2_000_000 if overlap else 22_000_000
    return dict(schema='gsdeploy.derived_host_call_band_observations.v1',split='training',
        execution_context_sha256='fixture',run_id=run,requests=[
            dict(request_id='a',alias='a#0',profile_id='a1',segments=[
                dict(segment_id='native.first',start_ns=0,end_ns=4_000_000),
                dict(segment_id='native.continuation',start_ns=4_000_000,end_ns=10_000_000),
                dict(segment_id='tail',start_ns=10_000_000,end_ns=11_000_000)]),
            dict(request_id='b',alias='b#0',profile_id='b1',segments=[
                dict(segment_id='native',start_ns=bstart,end_ns=bend),
                dict(segment_id='tail',start_ns=bend,end_ns=bend+1_000_000)])])


class SupportAuditTests(unittest.TestCase):
    def test_one_nanosecond_single_run_vertex_is_retained(self):
        template=band_template();docs=[document('one',True),document('two',False)]
        before=copy.deepcopy(docs)
        report=audit(template,docs,{'a#0':'a','b#0':'b'})
        self.assertEqual(docs,before)
        group=next(g for g in report['groups'] if g['family']=='a::first')
        self.assertFalse(group['necessary_repetition_checks_passed'])
        rare=group['insufficiently_repeated_vertices']
        self.assertEqual(len(rare),1)
        self.assertEqual(rare[0]['independent_run_ids'],['one'])
        self.assertAlmostEqual(rare[0]['total_exposure_ms'],.000001,places=12)
        self.assertFalse(report['physical_predictor_qualified'])
        docs[1]['run_id']='one'
        with self.assertRaisesRegex(ValueError,'Distinct'):
            audit(template,docs,{'a#0':'a','b#0':'b'})

    def test_g2_task_phase_alternatives_share_one_actual_owner(self):
        names=[('a::first','text'),('generalist_2::first','text'),
               ('generalist_2::continuation','ocr'),('b::image','image')]
        original={names[0]:'a',names[1]:'generalist_2',names[2]:'generalist_2',names[3]:'b'}
        vertices=possible_vertices(names,original,{'a#0':'a','g#0':'generalist_2','b#0':'b'},'a')
        self.assertEqual(len(vertices),6)
        self.assertTrue(all(v[0]==0 and v[1]+v[2]<=1 for v in vertices))
        self.assertNotIn((0.,1.,1.,0.),vertices)


if __name__=='__main__':unittest.main()
