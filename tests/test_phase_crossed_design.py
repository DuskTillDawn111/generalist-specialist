"""Training means may schedule attempts, but cannot certify actual overlap."""
import copy
import unittest

from experiments.activation_phase_crossed_design import align, phase_conditions


def template():
    profiles=[]
    for family in ('text','image','ocr','asr','generalist_2'):
        for task in ('text','image','ocr','asr') if family=='generalist_2' else (family,):
            roles=['whole'] if family=='image' else ['first','tail']
            profiles.append(dict(id=family+'/'+task,family=family,task=task,pool_id=task,
                segments=[dict(id=role,kind='gpu',native_band_role=role,native_root_id='native',
                               duration_ms={'mean':(i+1)*10.}) for i,role in enumerate(roles)]))
    return dict(execution_context_sha256='fixture',profiles=profiles)


class PhaseDesignTests(unittest.TestCase):
    def test_all_rosters_preserve_one_g2_task_per_actual_owner(self):
        t=template();aliases={f+'#0':f for f in ('text','image','ocr','asr','generalist_2')}
        counts=[]
        for n in (2,3,4,5):
            conditions=phase_conditions(t,aliases,[n]);counts.append(len(conditions))
            self.assertTrue(all(len({c['alias'] for c in condition})==n for condition in conditions))
            self.assertTrue(all(sum(c['family']=='generalist_2' for c in condition)<=1 for condition in conditions))
        self.assertEqual(counts,[74,164,168,64])

    def test_offsets_align_training_band_interiors_without_claiming_observed_overlap(self):
        t=template();before=copy.deepcopy(t)
        text=next(p for p in t['profiles'] if p['family']=='text')
        image=next(p for p in t['profiles'] if p['family']=='image')
        condition=[dict(alias='text#0',family='text',task='text',role='tail'),
                   dict(alias='image#0',family='image',task='image',role='whole')]
        result=align(condition,[text,image],t,{'text#0':4,'image#0':9})
        self.assertEqual([r['offset_ms'] for r in result['requests']],[0.,15.])
        self.assertEqual(result['intended_phase_alignment']['target_time_ms'],20.)
        self.assertFalse(result['intended_phase_alignment']['actual_overlap_guaranteed'])
        self.assertFalse(result['intended_phase_alignment']['future_observations_used'])
        self.assertEqual(t,before)
        with self.assertRaisesRegex(ValueError,'One request per actual owner'):
            align([condition[0],condition[0]],[text,text],t,{'text#0':4})


if __name__=='__main__':unittest.main()
