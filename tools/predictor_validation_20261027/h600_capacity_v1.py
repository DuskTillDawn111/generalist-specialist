"""H600-only input-weighted capacity preparation and prospective qualification.

CPU-only: no collector or quality scorer is invoked. Eight primary training
flows are the only numerical sources; short preflight, repeats, H90 and all
independent outcomes are excluded from estimation.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
from pathlib import Path
import time

from . import capacity_input_weighted_v1 as weighted
from .flow_capacity_catalog import condition, training_evidence
from .flow_capacity_qualification import POLICY as ORIGINAL_QUALIFICATION, _telemetry
from .flow_capacity_transfer import _generation, _plan
from .flow_eligibility import native_stability, RHO_MAX, RHO_TOLERANCE
from .native_flow import checked, receipt, save, prepare as prepare_native_plan
from .probe_validation import read
from .profiling import digest
from gsdeploy.stability import DEFAULT_THRESHOLDS


SCHEMA = 'gsdeploy.h600_input_weighted_capacity_protocol.v1'
TARGET_SEED = 627701
POLICY = dict(
    estimator=weighted.POLICY, numerical_source='Only 8 H600 primary training runs, seeds 627601/627602',
    training_seeds=[627601,627602], primary_routes=sorted(weighted.ROUTES),
    H=600., drain_seconds=300., warmup_seconds=0., target_seed=TARGET_SEED,
    CV_unit='Complete seed held out across all four primary routes',
    preflight_repeats_and_H90_excluded=True,
    mean_service_relative_error_max=ORIGINAL_QUALIFICATION['mean_service_relative_error_max'],
    mean_service_error=ORIGINAL_QUALIFICATION['mean_service_error'],
    rho_threshold=RHO_MAX, rho_tolerance=RHO_TOLERANCE,
    stability_thresholds=copy.deepcopy(DEFAULT_THRESHOLDS),
    qualification_runs_per_candidate=1, candidate_coverage=1., request_coverage=1.,
    qualification='Every offered family/task mean service error <= .2 using frozen predicted denominator; full cohort/telemetry and original native stability; original rho arithmetic <=.8+1e-8',
    missing='Missing positive-weight input or exact condition remains UNKNOWN; never borrow H90, another route or repeat',
    scope='Single new H600 operating run per route; no physical capacity or calibrated confidence bound',
    final_gain_eligible=False,
)


def sources():
    names=('h600_capacity_v1.py','capacity_input_weighted_v1.py','flow_capacity_catalog.py',
           'flow_capacity_qualification.py','flow_capacity_transfer.py','flow_eligibility.py',
           'flow_profiles.py','native_flow.py','training_campaign_v14.py')
    import gsdeploy.stability, gsdeploy.performance, gsdeploy.energy
    return [receipt(Path(__file__).with_name(n)) for n in names]+[
        receipt(m.__file__) for m in (gsdeploy.stability,gsdeploy.performance,gsdeploy.energy)]


def verify_suite(suite_path):
    suite=read(suite_path)
    if (suite.get('schema')!='gsdeploy.continuous_training_suite.v1' or
            suite.get('seeds')!=POLICY['training_seeds'] or len(suite['jobs'])!=8 or
            suite.get('no_validation_data') is not True or suite.get('repeats_excluded') is not True):
        raise ValueError('Exact two-seed/eight-primary H600 training suite required')
    checked(suite['source']); checked(suite['protocol'])
    seen=set();conditions={}
    for job in suite['jobs']:
        plan=read(checked(job['plan']))
        cid=job['candidate_id']; key=(plan['seed'],cid)
        if (key in seen or plan['seed'] not in POLICY['training_seeds'] or cid not in weighted.ROUTES or
                job.get('campaign_role')!='primary_training' or
                plan.get('training_run_label')!='primary' or plan.get('measurement_role')!='H600_training' or
                plan.get('phase')!='predictor_training' or plan.get('kind')!='prospective_training' or
                plan.get('split')!='training' or plan.get('validation_tuning') is not False or
                plan.get('window_seconds')!=600. or plan.get('drain_seconds')!=300. or plan.get('warmup_seconds')!=0. or
                plan.get('training_campaign_protocol')!=suite['protocol'] or
                plan['plan_sha256']!=digest({k:v for k,v in plan.items() if k!='plan_sha256'})):
            raise ValueError('Primary role, H600 horizon, complete identity or frozen training plan mismatch')
        seen.add(key)
        candidate=next(c for c in plan['candidates'] if c['id']==cid)
        spec=condition(plan,candidate); generation=_generation(plan,spec)
        if conditions.setdefault(cid,(spec,generation))!=(spec,generation):
            raise ValueError('Primary seed conditions differ')
    if seen!={(s,c) for s in POLICY['training_seeds'] for c in weighted.ROUTES}:
        raise ValueError('All eight primary seed/route units required')
    return suite


def prepare(suite_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    suite=verify_suite(suite_path)
    output.mkdir(parents=True,exist_ok=False)
    first=read(suite['jobs'][0]['plan']['path'])
    folder=output/'reserved-target-seed-627701'
    plan=prepare_native_plan(Path(first['profile_plan']['path']).parent,folder,
        seed=TARGET_SEED,duration=600.,drain=300.,warmup=0.,text_rate=1.85,background_rate=.05)
    (folder/'plan.json').rename(folder/'native-prepared-plan.json')
    plan.update(kind='prospective_capacity_validation',split='validation',
        phase='predictor_validation',created_ns=time.perf_counter_ns(),
        protocol_generator=receipt(__file__),capacity_estimation_outcomes_used=False,
        scope='Reserved independent H600 capacity operating validation; no model or energy evidence is granted by this plan')
    plan.pop('plan_sha256');plan['plan_sha256']=digest(plan);save(folder/'plan.json',plan)
    refs=[receipt(suite_path),suite['source'],suite['protocol'],*sources(),*[j['plan'] for j in suite['jobs']]]
    protocol=dict(schema=SCHEMA,created_ns=time.perf_counter_ns(),policy=POLICY,
        training_suite=receipt(suite_path),reserved_target_plan=receipt(folder/'plan.json'),
        source_receipts=refs,target_seed=TARGET_SEED,training_outcomes_read=False,
        independent_outcomes_read=False,GPU_started=False,original_failures_preserved=True,
        target_collection_authorized_by_prepare=False)
    protocol['protocol_sha256']=digest(protocol);save(output/'protocol.json',protocol)
    return protocol


def verify_protocol(path):
    value=read(path)
    if (value.get('schema')!=SCHEMA or value.get('policy')!=POLICY or
            value.get('protocol_sha256')!=digest({k:v for k,v in value.items() if k!='protocol_sha256'})):
        raise ValueError('Changed H600 capacity protocol')
    for ref in [*value['source_receipts'],value['reserved_target_plan']]:checked(ref)
    verify_suite(value['training_suite']['path'])
    target=_plan(value['reserved_target_plan']['path'])
    if (target['seed']!=TARGET_SEED or target['window_seconds']!=600. or
            target['drain_seconds']!=300. or target['warmup_seconds']!=0.):
        raise ValueError('Reserved target seed/horizon mismatch')
    return value


def inventory(protocol_path,output):
    """Read only completed primary raws; never starts collectors or quality."""
    if Path(output).exists():raise FileExistsError(output)
    protocol=verify_protocol(protocol_path);suite=read(protocol['training_suite']['path'])
    rows=[]
    from .flow_profiles import map_flow_collection
    for job in suite['jobs']:
        path=Path(job['output'])/'raw.json'
        item={'candidate_id':job['candidate_id'],'plan':job['plan'],'seed':read(job['plan']['path'])['seed']}
        if not path.exists():
            rows.append({**item,'status':'NOT_COMPLETE'});continue
        plan,candidate,raw,derived,observations,peak=training_evidence(path,job['plan'])
        mapped=map_flow_collection(path)
        rows.append({**item,'status':'TRAINING_SOURCE_VERIFIED','raw':receipt(path),
            'requests':len(observations),'mapping_report':mapped['mapping_report'],
            'raw_owned_peak_mib':peak,'numerical_fitting_performed':False})
    value=dict(schema='gsdeploy.h600_capacity_training_inventory.v1',created_ns=time.perf_counter_ns(),
        protocol=receipt(protocol_path),rows=rows,completed=sum(r['status']=='TRAINING_SOURCE_VERIFIED' for r in rows),
        requested=8,GPU_started=False,quality_scoring_performed=False,independent_results_read=False)
    save(output,value);return value


def crossvalidate(protocol_path,output):
    protocol=verify_protocol(protocol_path)
    # Frozen v1 accepts exactly this 2-seed/8-flow suite. Its complete condition
    # key includes H, drain, warmup and resources, so H90 cannot match H600.
    return weighted.crossvalidate(protocol['training_suite']['path'],output)


def seal(protocol_path,output):
    """Freeze all-H600-training prediction before independent target startup."""
    if Path(output).exists():raise FileExistsError(output)
    protocol=verify_protocol(protocol_path)
    plan_path=protocol['reserved_target_plan']['path'];folder=Path(plan_path).parent
    if any((folder/name).exists() for name in ('runs','raw.json','scored-raw.json')):
        raise ValueError('Capacity seal must precede independent target collection')
    plan=_plan(plan_path);bundle=weighted.load_training(protocol['training_suite']['path'])
    catalog=read(checked(plan['catalog']))
    predictions=[{'candidate_id':c['id'],'prediction':weighted.estimate(plan,c,bundle['runs'],catalog)} for c in plan['candidates']]
    refs=[receipt(protocol_path),*protocol['source_receipts'],*bundle['source_receipts'],protocol['reserved_target_plan']]
    for ref in refs:checked(ref)
    value=dict(schema='gsdeploy.h600_input_weighted_capacity_seal.v1',created_ns=time.perf_counter_ns(),
        protocol=receipt(protocol_path),target_plan=protocol['reserved_target_plan'],source_receipts=refs,
        policy=POLICY,candidates=predictions,GPU_started=False,target_outcomes_used=False,
        target_applicability_validated=None,final_gain_eligible=False)
    value['seal_sha256']=digest(value);save(output,value);return value


def service_checks(prediction,observations):
    checks=[]
    by_group=collections.defaultdict(list)
    for row in observations:by_group[(row['family'],row['task'])].append(row['worker_service_ms'])
    for item in prediction['estimates']:
        observed=by_group[(item['family'],item['task'])]
        expected=item['weighted_worker_service_ms']
        actual=sum(observed)/len(observed) if observed else None
        error=abs(actual-expected)/expected if actual is not None and expected is not None else None
        checks.append(dict(family=item['family'],task=item['task'],predicted_worker_mean_ms=expected,
            observed_worker_mean_ms=actual,request_count=len(observed),absolute_relative_mean_error=error,
            threshold=.2,passed=error<=.2 if error is not None else None))
    return checks


def verify_seal(path):
    value=read(path)
    if (value.get('schema')!='gsdeploy.h600_input_weighted_capacity_seal.v1' or value.get('policy')!=POLICY or
            value.get('seal_sha256')!=digest({k:v for k,v in value.items() if k!='seal_sha256'})):
        raise ValueError('Changed H600 capacity prediction seal')
    for ref in value['source_receipts']:checked(ref)
    protocol=verify_protocol(value['protocol']['path'])
    if value['target_plan']!=protocol['reserved_target_plan']:
        raise ValueError('Capacity seal targets another public plan')
    plan=_plan(value['target_plan']['path']);bundle=weighted.load_training(protocol['training_suite']['path'])
    catalog=read(checked(plan['catalog']))
    expected=[{'candidate_id':c['id'],'prediction':weighted.estimate(plan,c,bundle['runs'],catalog)} for c in plan['candidates']]
    if value['candidates']!=expected:raise ValueError('Capacity estimates differ from recomputed H600 training arithmetic')
    return value


def report(sealed_path,raw_paths,output):
    """Evaluate future independent raw; never used by prepare/CV/estimation."""
    if Path(output).exists():raise FileExistsError(output)
    frozen=verify_seal(sealed_path);plan=_plan(frozen['target_plan']['path'])
    catalog=read(checked(plan['catalog']));candidates={c['id']:c for c in plan['candidates']}
    predictions={c['candidate_id']:c['prediction'] for c in frozen['candidates']}
    rows=[];seen=set()
    for path in raw_paths:
        raw=read(path);cid=raw['candidate_id']
        if cid not in candidates or cid in seen:raise ValueError('Duplicate or foreign independent capacity candidate')
        seen.add(cid);candidate=candidates[cid]
        if (raw.get('plan')!=frozen['target_plan'] or raw.get('startup_begin_ns',0)<=frozen['created_ns'] or
                raw.get('resource_configuration')!=candidate.get('resource_configuration')):
            raise ValueError('Independent raw must bind public plan/resources and start after capacity seal')
        stability=native_stability(plan,candidate,raw,catalog)
        observations=[]
        for row in raw['request_records']:
            response=row.get('worker_response',{});start=row.get('observed_worker_start_time_ns');end=response.get('end_perf_ns')
            if (row.get('status')!='completed' or response.get('success') is not True or
                    response.get('start_perf_ns')!=start or type(start) is not int or type(end) is not int or
                    not row['scheduled_time_ns']<=start<end<=row['completion_time_ns']<=raw['observation_end_ns']):
                raise ValueError('Complete successful worker service and valid actual clocks required')
            observations.append(dict(family=plan['aliases'][row['model']]['family'],task=row['task'],
                worker_service_ms=(end-start)/1e6))
        prediction=predictions[cid];checks=service_checks(prediction,observations)
        clean=_telemetry(plan,raw) and not any(raw.get(k) for k in
            ('errors','runtime_errors','telemetry_errors','qualification_issues'))
        shutdown=raw.get('shutdown',{})
        clean=clean and set(shutdown)==set(plan['aliases']) and all(
            not s.get('forced') and s.get('exit_code')==0 and s.get('bye_received') is True for s in shutdown.values())
        complete=raw.get('collection_complete') is True and len(raw['request_records'])==len(plan['trace'])
        applicability=(all(c['passed'] is True for c in checks) and clean and complete) if prediction['status']=='ESTIMATED' else None
        admitted=(applicability and prediction['rho_threshold_satisfied'] and stability['original_stability_passed']) if applicability is not None else None
        rows.append(dict(candidate_id=cid,status='ASSESSED',raw=receipt(path),mean_service_checks=checks,
            statistical_profile_applicable=applicability,clean_full_owned_telemetry=clean,complete_cohort=complete,
            original_stability=stability,original_rho_formula_statistical_profile=prediction['rho_by_family'],
            original_rho_threshold_satisfied=prediction['rho_threshold_satisfied'],
            statistical_original_rho_planning_admission_passed=admitted,
            original_measured_profile_resolver_admission=None,physical_capacity_bound_certified=False,
            final_gain_eligible=False))
    for cid in sorted(set(candidates)-seen):rows.append(dict(candidate_id=cid,status='NOT_RUN',
        statistical_profile_applicable=None,statistical_original_rho_planning_admission_passed=None,final_gain_eligible=False))
    value=dict(schema='gsdeploy.h600_input_weighted_capacity_qualification_report.v1',created_ns=time.perf_counter_ns(),
        seal=receipt(sealed_path),report_source=receipt(__file__),status='ASSESSED' if seen==set(candidates) else 'INCOMPLETE',
        policy=POLICY,candidates=rows,independent_operating_run_count_per_candidate=1,
        GPU_started=False,uncertainty_calibrated=False,physical_capacity_bound_certified=False,
        old_failures_overridden=False,final_gain_eligible=False)
    save(output,value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--suite',required=True);p.add_argument('--output',required=True)
    for name in ('inventory','crossvalidate','seal'):
        p=sub.add_parser(name);p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('report');p.add_argument('--sealed',required=True);p.add_argument('--raw',action='append',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    result=(prepare(args.suite,args.output) if args.op=='prepare' else report(args.sealed,args.raw,args.output)
            if args.op=='report' else globals()[args.op](args.protocol,args.output))
    print(json.dumps({'operation':args.op,'schema':result.get('schema'),'GPU_started':False}))
