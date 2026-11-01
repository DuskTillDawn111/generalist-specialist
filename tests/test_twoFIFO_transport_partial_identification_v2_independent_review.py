"""Independent feasible-set checks; no native labels or random latent choice."""
import copy
from fractions import Fraction
import random
import unittest
from tools.predictor_validation_20261027 import twoFIFO_transport_partial_identification_v2 as m


class PartialIdentificationIndependentReview(unittest.TestCase):
    def test_generated_feasible_latent_truth_contained_and_translation_invariant(self):
        rng=random.Random(20261028)
        for _ in range(300):
            # Construct an independently feasible source by drawing a genuine
            # paired line/continuation, then discard that decomposition.
            W=Fraction(rng.randrange(20),3);u=W*Fraction(rng.randrange(11),10)
            x=Fraction(rng.randrange(20),7);q=Fraction(rng.randrange(-20,21),5)
            H=max(q,u)+x;s=max(100,-q)
            source=m.identify_source(s,s+W,s+H,s+q)
            qtarget=Fraction(rng.randrange(-20,21),5);target_submit=1000
            result=m.project_target(source,target_submit,target_submit+qtarget)
            true_START=target_submit+max(qtarget,u)+x
            lo,hi=result['START_ns_bounds']
            self.assertLessEqual(lo,true_START);self.assertLessEqual(true_START,hi)
            self.assertLessEqual(lo,hi)
            moved=m.project_target(source,target_submit+300,target_submit+300+qtarget)
            self.assertEqual(moved['START_ns_bounds'],[lo+300,hi+300])
            shifted_source=m.identify_source(s+500,s+500+W,s+500+H,s+500+q)
            self.assertEqual(shifted_source,source)
            self.assertEqual(m.project_target(source,s,s+q)['START_ns_bounds'],[s+H,s+H])
            self.assertFalse(result['latent_point_selected'])

    def test_continuation_image_extrema_are_attainable_paired_endpoints(self):
        # Both endpoints must be attainable by an admissible *paired* latent
        # vector, not by independently combining line and continuation bounds.
        for W,H,q in ((7,9,-3),(7,12,10),(10,12,4),(40,12,4),(0,8,-4)):
            source=m.identify_source(100,100+W,100+H,100+q)
            for target in (Fraction(-1),Fraction(1,3),Fraction(19,3),Fraction(20)):
                result=m.project_target(source,1000,1000+target)
                U=min(W,H);candidates=[Fraction(0),Fraction(U)]
                candidates.extend(t for t in (Fraction(q),target) if 0<=t<=U)
                attained=[]
                for u in candidates:
                    continuation=H-max(q,u)
                    self.assertGreaterEqual(continuation,0)
                    self.assertEqual(max(100+q,100+u)+continuation,100+H)
                    attained.append(max(1000+target,1000+u)+continuation)
                lo,hi=result['START_ns_bounds']
                self.assertIn(lo,attained);self.assertIn(hi,attained)

    def test_source_projection_schema_is_closed_exact_and_canonical(self):
        good=m.identify_source(100,110,120,115)
        malformed=[]
        value=copy.deepcopy(good);value['submit_to_write_record_ns']=-1;value['stdin_line_extra_ns_bounds']=[0,-1];malformed.append(value)
        for key in ('submit_to_write_record_ns','observed_submit_to_START_ns','previous_READY_minus_submit_ns'):
            value=copy.deepcopy(good);value[key]=float(value[key]);malformed.append(value)
        value=copy.deepcopy(good);value['stdin_line_extra_ns_bounds']=[0.,10.];malformed.append(value)
        value=copy.deepcopy(good);value['worker_continuation_extra_ns_bounds']=[0,999];malformed.append(value)
        value=copy.deepcopy(good);value['source_regime']='PREVIOUS_READY_PRECEDES_SUBMIT';malformed.append(value)
        value=copy.deepcopy(good);value['latent_probability_law_identified']=True;malformed.append(value)
        value=copy.deepcopy(good);value['unknown_extra']=123;malformed.append(value)
        for key in good:
            value=copy.deepcopy(good);value.pop(key);malformed.append(value)
        for value in malformed:
            with self.subTest(value=value),self.assertRaises(ValueError):m.project_target(value,1000,1020)

    def test_forensic_entry_role_gate_runs_before_request_labels(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        for raw_split,plan_split in (('validation','training'),('training','validation')):
            with tempfile.TemporaryDirectory() as tmp:
                tmp=Path(tmp);plan=tmp/'plan.json';raw=tmp/'raw.json'
                plan.write_text(json.dumps(dict(schema='gsdeploy.original_twoFIFO_owned_training_campaign.v3.attempt',split=plan_split)))
                raw.write_text(json.dumps(dict(schema='gsdeploy.original_twoFIFO_owned_training_capture.v3',split=raw_split,plan=m.receipt(plan))))
                with patch.object(m,'observed_dominance',side_effect=AssertionError('labels reached before role gate')):
                    with self.assertRaises(ValueError):m.run(raw,plan,tmp/'out')
