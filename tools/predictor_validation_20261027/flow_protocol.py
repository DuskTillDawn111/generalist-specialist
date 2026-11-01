"""Prospective known-input full-flow validation packet, without a GPU launch."""
import argparse
import json
from pathlib import Path

from .native_flow import prepare,receipt
from .profiling import digest


def prepare_validation(profile_dir,output,*,seed=627302,duration=90.):
    plan=prepare(profile_dir,output,seed=seed,duration=duration)
    plan_path=Path(output)/'plan.json'
    plan_path.rename(Path(output)/'prepared-plan.json')
    plan.update(protocol_generator=receipt(__file__),phase='predictor_validation',
        utility_tie_tolerance=0.,
        validation_thresholds={
            'complete_request_coverage':1.,'complete_candidate_coverage':1.,
            'service_mare_max':.2,'e2e_mare_max':.2,'each_task_p95_absolute_relative_error_max':.2,
            'queue_depth_mean_absolute_error_max_requests':1.,
            'paired_delta_sign_accuracy_min':1.,'candidate_pair_rank_accuracy_min':1.,
            'require_complete_paired_and_rank_coverage':True,
            'scope':'One prospective periodic known-input pilot; descriptive thresholds, no final-gain confidence claim',
            'post_observation_tuning':False},
        gate_order=['complete_prediction_domain','complete_native_cohort','service_and_parent_E2E',
                    'each_task_P95_and_queue','six_term_delta_sign_and_rank'],
        final_confirmation_enabled=False,
        scope='Closed known-input domain; validation probes of unknown inputs remain incomplete and are not relabelled')
    plan.pop('plan_sha256');plan['plan_sha256']=digest(plan)
    with plan_path.open('x') as f:
        json.dump(plan,f,indent=2,allow_nan=False);f.write('\n')
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile-dir',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=627302)
    p.add_argument('--duration',type=float,default=90.)
    args=p.parse_args();plan=prepare_validation(args.profile_dir,args.output,seed=args.seed,duration=args.duration)
    print(json.dumps(dict(status='PREPARED',requests=len(plan['trace']),candidates=len(plan['candidates']))))
