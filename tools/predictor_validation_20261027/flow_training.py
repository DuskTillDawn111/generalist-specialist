"""New continuous-arrival training runs, disjoint from all validation runs."""
import argparse
import json
from pathlib import Path
import random
import time

from .native_flow import prepare,execute,receipt,save
from .profiling import digest


def prepare_suite(profile_dir,output,*,seeds=(627401,627402),duration=90.):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for seed in seeds:
        folder=output/('seed-'+str(seed));plan=prepare(profile_dir,folder,seed=seed,duration=duration)
        (folder/'plan.json').rename(folder/'prepared-plan.json')
        plan.update(phase='predictor_training',kind='prospective_training',created_ns=time.perf_counter_ns(),
            protocol_generator=receipt(__file__),split='training',
            scope='New continuous-arrival training; prior v9 validation observations are not fitting data',
            validation_tuning=False)
        plan.pop('plan_sha256');plan['plan_sha256']=digest(plan)
        save(folder/'plan.json',plan)
        for candidate in plan['candidates']:
            jobs.append(dict(plan=receipt(folder/'plan.json'),candidate_id=candidate['id'],
                             output=str(folder/'runs'/candidate['id'])))
    random.Random('continuous-training-v10-predeclared-order').shuffle(jobs)
    packet=dict(schema='gsdeploy.continuous_training_suite.v1',created_ns=time.perf_counter_ns(),
        seeds=list(seeds),jobs=jobs,source=receipt(__file__),GPU_started=False,
        no_retries=True,no_validation_data=True)
    save(output/'suite.json',packet);return packet


def run_suite(path):
    from .native_flow import checked
    packet=json.loads(Path(path).read_text());checked(packet['source'])
    for job in packet['jobs']:
        checked(job['plan'])
        print(json.dumps(dict(event='starting_training_run',candidate=job['candidate_id'],plan=job['plan']['path'])),flush=True)
        raw=execute(job['plan']['path'],job['candidate_id'],job['output'])
        if raw['status']!='completed' or raw['qualification_issues']:
            raise RuntimeError('Training collection failed; no automatic seed replacement')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='op',required=True)
    a=s.add_parser('prepare');a.add_argument('--profile-dir',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('execute');a.add_argument('--suite',required=True)
    a=p.parse_args()
    if a.op=='prepare':
        packet=prepare_suite(a.profile_dir,a.output);print(json.dumps(dict(status='PREPARED',jobs=len(packet['jobs']))))
    else:run_suite(a.suite)
