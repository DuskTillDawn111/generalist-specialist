import unittest

from tools.predictor_validation_20261027.physical_progress_evidence_v26 import (
    ContextIndex,ContextPrefix,KernelPrefix,evidence)


class PhysicalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.index=ContextIndex([(0,10,'a'),(12,30,'b'),(32,50,'a')])
        self.prefix=KernelPrefix({'a':[(16,18),(33,40)],'b':[(15,20)]})

    def test_nonheld_START_END_without_own_envelope_witness_is_distinct_from_service(self):
        row=evidence(dict(start=16,end=18),dict(start=13,end=14),self.index,self.prefix,'a')
        labels=row['observation_labels']
        self.assertEqual(labels['context_at_kernel_start']['owner'],'b')
        self.assertEqual(labels['context_at_kernel_end']['owner'],'b')
        self.assertEqual(labels['observed_lifetime_context_partition']['own_envelope_ns'],0)
        self.assertEqual(labels['observed_lifetime_context_partition']['other_envelope_ns'],2)
        self.assertNotIn('service_ns',labels)
        self.assertEqual(labels['other_owner_kernel_lifetime_union_overlap_ns'],{'b':2})

    def test_signed_API_return_START_is_not_clamped_to_fabricated_queue(self):
        row=evidence(dict(start=16,end=18),dict(start=13,end=19),self.index,self.prefix,'a')
        labels=row['observation_labels']
        self.assertEqual(labels['launch_API_return_to_kernel_start_signed_ns'],-3)
        self.assertIsNone(labels['nonnegative_return_to_start_context_partition'])
        self.assertTrue(row['prefix_features']['already_observed_kernel_start_at_cutoff'])

    def test_context_gap_boundary_and_ambiguous_envelopes_do_not_become_other_owner(self):
        self.assertIsNone(self.index.at(11)['owner'])
        self.assertTrue(self.index.at(12)['boundary'])
        dirty=ContextIndex([(0,30,'a')],[20])
        self.assertEqual(dirty.at(15)['status'],'UNPAIRED_EVENT_IN_ENVELOPE')
        with self.assertRaises(ValueError):ContextIndex([(0,10,'a'),(9,12,'b')])

    def test_lifetime_partition_and_union_overlap_conserve_elapsed_time(self):
        part=self.index.partition(5,40,'a')
        self.assertEqual(part['own_envelope_ns'],13)
        self.assertEqual(part['other_envelope_ns'],18)
        self.assertEqual(part['gap_ns'],4)
        self.assertEqual(sum(part[k] for k in ('own_envelope_ns','other_envelope_ns','gap_ns')),35)
        overlapping=KernelPrefix({'a':[(0,10),(2,5)],'b':[(0,4)]})
        self.assertEqual(overlapping.unions['a'].overlap(0,10),10)
        self.assertEqual(overlapping.outstanding_started_by_owner(3),{'a':2,'b':1})

    def test_future_context_SAVE_pairing_or_issue_cannot_change_causal_prefix_features(self):
        events=[(0,0,'a','RESTORE_START'),(10,1,'a','SAVE_END'),(12,2,'b','RESTORE_START')]
        first=ContextPrefix([*events,(30,3,'b','SAVE_END')])
        second=ContextPrefix([*events,(1000,3,'b','SAVE_END'),(1001,4,'a','RESTORE_START')])
        a=evidence(dict(start=16,end=18),dict(start=13,end=14),self.index,self.prefix,'a',first)
        dirty=ContextIndex([(0,10,'a'),(12,1000,'b')],[900])
        b=evidence(dict(start=160,end=180),dict(start=13,end=14),dirty,self.prefix,'a',second)
        self.assertEqual(a['prefix_features'],b['prefix_features'])
        self.assertNotEqual(a['observation_labels'],b['observation_labels'])
        self.assertNotIn('paired_context_at_launch_return_diagnostic',a['prefix_features'])

    def test_raw_overlapping_context_prefix_remains_ambiguous(self):
        events=ContextPrefix([(1,0,'a','RESTORE_START'),(2,1,'b','RESTORE_START'),(5,2,'a','SAVE_END')])
        self.assertEqual(events.at(3)['status'],'MULTIPLE_OPEN_CONTEXT_PREFIXES')
        self.assertIsNone(events.at(3)['owner'])
        self.assertEqual(events.at(5)['owner'],'b')


if __name__=='__main__':unittest.main()
