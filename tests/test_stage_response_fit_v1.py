import math
import unittest

import numpy as np

from tools.predictor_validation_20261027.stage_response_fit_v1 import fit, prepared, profile


def stage(duration, count, same=0, index=0, capture='capture'):
    start=1000000;end=start+round(duration*1e6)
    state=dict(cutoff_processing_ns=start,peer_visible_unconfirmed_owner_count=count,
        peer_same_semantic_phase_owner_count=same)
    return dict(stage_kind='language_forward',owner='text#0',pool_id='input',semantic_phase='prefill',
        forward_index=index,input_sequence_length=16,input_batch_size=1,capture_sha256=capture,
        request_id='r',phase_id='prefill',host_training_fit_diagnostic_eligible=True,
        stream_response_fit_eligible=False,host_channel_is_effective_SM_service=False,
        response_labels=dict(host_observed_interval_ns=[start,end],host_call_wall_ns=end-start),
        full_host_wall_covariate_path=[dict(left_ns=start,right_ns=end,duration_ns=end-start,
            state=state,each_state_uses_only_its_processed_prefix=True)])


class FitTests(unittest.TestCase):
    def data(self,beta):
        rows=[]
        # Balanced independent synthetic responses; negative beta must remain
        # recoverable, with baseline nuisance estimated rather than fixed.
        for count in range(3):
            for noise in (-.08,.08):
                rows.append(stage(10*math.exp(beta*count+noise),count))
        return rows

    def test_constant_covariate_positive_and_acceleration_recovery(self):
        for beta in (.3,-.3):
            result=fit(self.data(beta),'visible_count')
            self.assertAlmostEqual(result['coefficients'][0],beta,places=5)
            self.assertAlmostEqual(result['lognormal_reference_sigma'],.08,places=5)
            self.assertTrue(result['nonlinear_joint_response_jacobian']['full_column_rank'])
            self.assertFalse(result['prospective_predictor_qualified'])
            self.assertEqual(result['whole_capture_blocks'],1)

    def test_endpoint_density_jacobian_and_joint_nuisance(self):
        rows=prepared(self.data(.3),('peer_visible_unconfirmed_owner_count',))
        nll,mu,variance,_,_=profile(np.array([.3]),rows)
        expected=sum(math.log(x['durations'][0])+.5*math.log(2*math.pi*.08**2)+.5 for x in rows)
        self.assertAlmostEqual(nll,expected,places=5)
        self.assertAlmostEqual(next(iter(mu.values())),math.log(10),places=5)

    def test_dynamic_clock_integration_is_exact(self):
        row=stage(12,0)
        path=row['full_host_wall_covariate_path'];first=path[0];start=first['left_ns'];end=first['right_ns']
        first['right_ns']=start+4000000;first['duration_ns']=4000000
        path.append(dict(left_ns=first['right_ns'],right_ns=end,duration_ns=8000000,
            state=dict(first['state'],cutoff_processing_ns=first['right_ns'],
                peer_visible_unconfirmed_owner_count=1),each_state_uses_only_its_processed_prefix=True))
        other=stage(13,0)
        rows=prepared([row,other],('peer_visible_unconfirmed_owner_count',))
        beta=np.array([math.log(2)])
        _,mu,_,derivative,_=profile(beta,rows)
        self.assertAlmostEqual(next(iter(mu.values())),(math.log(8)+math.log(13))/2)
        self.assertAlmostEqual(derivative[0,0],-.5)

    def test_rank_includes_baseline_nuisance_and_detects_confounding(self):
        rows=self.data(.2)
        for row in rows:row['forward_index']=row['full_host_wall_covariate_path'][0]['state']['peer_visible_unconfirmed_owner_count']
        result=fit(rows,'visible_count')
        rank=result['nonlinear_joint_response_jacobian']
        self.assertFalse(rank['full_column_rank'])

    def test_incomplete_path_and_stream_channel_rejected(self):
        rows=self.data(.1);rows[0]['full_host_wall_covariate_path'][0]['left_ns']+=1
        with self.assertRaises(ValueError):fit(rows,'visible_count')
        rows=self.data(.1);rows[0]['stream_response_fit_eligible']=True
        with self.assertRaises(ValueError):fit(rows,'visible_count')


if __name__=='__main__':unittest.main()
