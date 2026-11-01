"""Finite H600 training campaign with two new seeds and identical-trace repeats.

Prepare is CPU only. Explicit execute commands run the unchanged native
collector. Primary fit units and repeated-run diagnostics have separate suites
because the frozen fitter identifies a flow by seed and candidate.
"""
from __future__ import annotations

import argparse
import collections
import copy
from pathlib import Path
import random
import time

from .native_flow import prepare, execute, checked, receipt, save
from .probe_validation import read
from .profiling import digest

SCHEMA = 'gsdeploy.h600_training_campaign.v1'
SEEDS = (627601, 627602)
ROUTES = ('shared_balanced', 'shared_skewed', 'shared_spill_low', 'shared_spill_high')
POLICY = dict(measurement_seconds=600., drain_seconds=300., warmup_seconds=0.,
    workload_rps=dict(text=1.85, image=.05, asr=.05, ocr=.05),
    primary_seeds=list(SEEDS), primary_routes=list(ROUTES),
    repeats=[dict(seed=SEEDS[0], candidate_id='shared_balanced'),
             dict(seed=SEEDS[0], candidate_id='shared_spill_low')],
    repeat_interpretation='Two identical-trace pairs estimate descriptive run variation only for these routes; not calibrated population variance',
    primary_training_runs=8, identical_trace_repeat_runs=2,
    short_preflight=dict(seed=627600, duration_seconds=30., candidate_id='shared_spill_low',
        numerical_training=False, scope='Collection/readiness check only; never mixed with H600 calibration'),
    split='training', arrival_process='declared_periodic', resource_mode='shared',
    input_domain='Existing source-bound known-input domain; no claim of unseen-input generalization',
    predeclared_no_retries=True, predeclared_no_seed_replacement=True,
    evaluation='Preserve complete offered cohorts, original quality totals, parent E2E, owned sampled memory, board energy and native stability definitions',
    fitting='Original primary suite contains one run per seed/route; repeat runs stay separate until a run-identity-aware training protocol is explicitly frozen',
    selection='No hyperparameter sweep on previous eight flows; acquire matched long-horizon training statistics first')


def _freeze_plan(profile_dir, folder, *, seed, duration, protocol, run_label):
    plan = prepare(profile_dir, folder, seed=seed, duration=duration,
        drain=POLICY['drain_seconds'], warmup=0., text_rate=1.85, background_rate=.05)
    (folder / 'plan.json').rename(folder / 'prepared-plan.json')
    plan.update(phase='predictor_training', kind='prospective_training', split='training',
        validation_tuning=False, created_ns=time.perf_counter_ns(), protocol_generator=receipt(__file__),
        training_campaign_protocol=protocol, training_run_label=run_label,
        measurement_role='short_collection_preflight_excluded_from_H600_fit' if duration != 600. else 'H600_training',
        scope='Prospectively declared training campaign; no independent outcomes used in this plan')
    plan.pop('plan_sha256'); plan['plan_sha256'] = digest(plan); save(folder / 'plan.json', plan)
    return plan


def _job(plan_path, candidate_id, output, run_id, *, role):
    return dict(plan=receipt(plan_path), candidate_id=candidate_id, output=str(output.resolve()),
        run_id=run_id, campaign_role=role)


def prepare_campaign(profile_dir, output):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema='gsdeploy.h600_training_campaign_protocol.v1', created_ns=time.perf_counter_ns(),
        policy=copy.deepcopy(POLICY), source=receipt(__file__), profile_plan=receipt(Path(profile_dir)/'plan.json'),
        context=receipt(Path(profile_dir)/'context.json'), GPU_started=False, independent_results_read=False)
    protocol['protocol_sha256'] = digest(protocol); save(output / 'protocol.json', protocol)
    protocol_ref = receipt(output / 'protocol.json'); primary=[]; plans={}
    for seed in SEEDS:
        folder = output / ('seed-' + str(seed))
        plan = _freeze_plan(profile_dir, folder, seed=seed, duration=600., protocol=protocol_ref, run_label='primary')
        if tuple(c['id'] for c in plan['candidates']) != ROUTES:
            raise ValueError('Unchanged four static shared candidates required')
        plans[seed] = plan
        for cid in ROUTES:
            primary.append(_job(folder/'plan.json', cid, folder/'runs'/cid,
                'seed-'+str(seed)+'__'+cid+'__primary', role='primary_training'))
    primary_packet = dict(schema='gsdeploy.continuous_training_suite.v1', created_ns=time.perf_counter_ns(),
        seeds=list(SEEDS), jobs=primary, source=receipt(__file__), protocol=protocol_ref,
        GPU_started=False, no_retries=True, no_validation_data=True, repeats_excluded=True)
    save(output/'primary-training-suite.json', primary_packet)
    repeats=[]
    for item in POLICY['repeats']:
        seed, cid = item['seed'], item['candidate_id']; folder=output/'repeats'/('seed-'+str(seed)+'__'+cid)
        folder.parent.mkdir(parents=True, exist_ok=True)
        plan = _freeze_plan(profile_dir, folder, seed=seed, duration=600., protocol=protocol_ref, run_label='repeat1')
        if plan['trace'] != plans[seed]['trace']:
            raise ValueError('Repeat must preserve identical full input/arrival trace')
        original = next(c for c in plans[seed]['candidates'] if c['id']==cid)
        repeated = next(c for c in plan['candidates'] if c['id']==cid)
        if original != repeated:
            raise ValueError('Repeat candidate routes/resources must remain identical')
        repeats.append(_job(folder/'plan.json', cid, folder/'runs'/cid,
            'seed-'+str(seed)+'__'+cid+'__repeat1', role='identical_trace_training_repeat'))
    save(output/'repeat-training-suite.json', dict(schema='gsdeploy.training_run_repeat_suite.v1',
        jobs=repeats, protocol=protocol_ref, source=receipt(__file__), GPU_started=False,
        split='training', fitting_role='Separate run-variation diagnostics; not passed to frozen seed/candidate fitter'))
    preflight = POLICY['short_preflight']; folder=output/'preflight'
    _freeze_plan(profile_dir, folder, seed=preflight['seed'], duration=preflight['duration_seconds'],
        protocol=protocol_ref, run_label='collection_preflight')
    prejob = _job(folder/'plan.json', preflight['candidate_id'], folder/'runs'/preflight['candidate_id'],
        'collection_preflight', role='preflight_excluded_from_H600_fit')
    main = primary + repeats
    random.Random('h600-training-v14-order-frozen').shuffle(main)
    value = dict(schema=SCHEMA, created_ns=time.perf_counter_ns(), protocol=protocol_ref, source=receipt(__file__),
        primary_suite=receipt(output/'primary-training-suite.json'), repeat_suite=receipt(output/'repeat-training-suite.json'),
        preflight_job=prejob, main_jobs=main, main_run_count=len(main),
        source_receipts=[protocol_ref, receipt(__file__), *[j['plan'] for j in main], prejob['plan']],
        GPU_started=False, no_retries=True, no_seed_replacement=True, validation_sources_used=[],
        expected_request_counts_per_H600_run=dict(text=1110, image=30, asr=30, ocr=30),
        expected_main_requests=12000, repeated_input_seeds_are_not_independent_input_realizations=True,
        estimate=dict(measurement_minutes=100, typical_total_minutes=[120,145],
            drain_upper_additional_minutes=50, collection_preflight_typical_minutes=[2,4]),
        scope='One finite new long-horizon training acquisition, no GPU or inference launched during prepare')
    value['campaign_sha256']=digest(value); save(output/'campaign.json', value); verify_campaign(output/'campaign.json')
    return value


def verify_campaign(path):
    value = read(path)
    if (value.get('schema') != SCHEMA or value.get('campaign_sha256') != digest({k:v for k,v in value.items() if k!='campaign_sha256'})
            or value.get('no_retries') is not True or value.get('no_seed_replacement') is not True):
        raise ValueError('Changed finite training campaign')
    for ref in [*value['source_receipts'], value['primary_suite'], value['repeat_suite']]: checked(ref)
    protocol = read(value['protocol']['path'])
    if protocol.get('policy') != POLICY or protocol['protocol_sha256'] != digest({k:v for k,v in protocol.items() if k!='protocol_sha256'}):
        raise ValueError('Changed training campaign policy')
    primary = read(value['primary_suite']['path'])['jobs']; repeats = read(value['repeat_suite']['path'])['jobs']
    if len(primary)!=8 or len(repeats)!=2 or len(value['main_jobs'])!=10:
        raise ValueError('Exactly eight primary plus two repeated runs required')
    ids=[j['run_id'] for j in value['main_jobs']]; outputs=[j['output'] for j in value['main_jobs']]
    if len(set(ids))!=10 or len(set(outputs))!=10 or sorted(ids)!=sorted(j['run_id'] for j in primary+repeats):
        raise ValueError('Unique complete campaign run identities required')
    expected_order=copy.deepcopy(primary+repeats);random.Random('h600-training-v14-order-frozen').shuffle(expected_order)
    if value['main_jobs']!=expected_order: raise ValueError('Campaign predeclared order changed')
    by_seed={}
    for job in [*value['main_jobs'],value['preflight_job']]:
        plan=read(job['plan']['path'])
        if (plan.get('plan_sha256')!=digest({k:v for k,v in plan.items() if k!='plan_sha256'}) or
                plan.get('phase')!='predictor_training' or plan.get('kind')!='prospective_training' or
                plan.get('split')!='training' or plan.get('validation_tuning') is not False or
                plan.get('protocol_generator')!=value['source'] or plan.get('training_campaign_protocol')!=value['protocol']):
            raise ValueError('Prospective training role and source identity required')
        for field in ('runner_source','context','profile_plan','catalog'):checked(plan[field])
        if job['campaign_role']!='preflight_excluded_from_H600_fit':
            if plan['window_seconds']!=600. or collections.Counter(r['task'] for r in plan['trace'])!=value['expected_request_counts_per_H600_run']:
                raise ValueError('Complete H600 planned cohort required')
            earlier=by_seed.setdefault(plan['seed'],plan['trace'])
            if earlier!=plan['trace']:raise ValueError('Identical seed repeats must retain exact trace')
    return value


def run_campaign(path, phase):
    value=verify_campaign(path); folder=Path(path).resolve().parent
    if phase not in ('preflight','main'):raise ValueError('Explicit preflight or main phase required')
    if phase=='main':
        raw=read(Path(value['preflight_job']['output'])/'raw.json')
        if (raw.get('status')!='completed' or raw.get('collection_complete') is not True or raw.get('qualification_issues')
                or any(raw.get(k) for k in ('errors','runtime_errors','telemetry_errors'))):
            raise ValueError('Clean preflight collection required before main; no retries/replacement')
    marker=folder/('started-'+phase+'.json')
    save(marker,dict(campaign=receipt(path),phase=phase,created_ns=time.perf_counter_ns(),source=receipt(__file__),no_retries=True))
    jobs=[value['preflight_job']] if phase=='preflight' else value['main_jobs']
    for job in jobs:
        verify_campaign(path)
        print('START_TRAINING',job['run_id'],flush=True)
        raw=execute(job['plan']['path'],job['candidate_id'],job['output'])
        if (raw.get('status')!='completed' or raw.get('collection_complete') is not True or raw.get('qualification_issues')
                or any(raw.get(k) for k in ('errors','runtime_errors','telemetry_errors'))
                or set(raw.get('shutdown',{}))!=set(read(job['plan']['path'])['aliases'])
                or any(s.get('forced') or s.get('exit_code')!=0 or s.get('bye_received') is not True for s in raw.get('shutdown',{}).values())):
            raise RuntimeError('Training collection failed; preserve incomplete campaign, no retries or seed replacement')
    save(folder/('completed-'+phase+'.json'),dict(campaign=receipt(path),jobs=len(jobs),completed_ns=time.perf_counter_ns()))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--profile-dir',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('verify');p.add_argument('--campaign',required=True)
    p=sub.add_parser('execute');p.add_argument('--campaign',required=True);p.add_argument('--phase',choices=['preflight','main'],required=True)
    args=parser.parse_args(argv)
    if args.op=='prepare':
        result=prepare_campaign(args.profile_dir,args.output);print('PREPARED',result['main_run_count'],'main runs; GPU_started=False')
    elif args.op=='verify':verify_campaign(args.campaign);print('VERIFIED; GPU_started=False')
    else:run_campaign(args.campaign,args.phase)


if __name__=='__main__':main()
