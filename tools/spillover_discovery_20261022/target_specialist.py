"""Target metadata/restrictions for proposal tools; no predictor mathematics."""
import math
TARGET_SPECIALISTS = ('asr', 'text', 'image', 'ocr')


def reject_extended_route_domain(config):
    """Legacy certificate tools only understand the original 1..K domain."""
    if config.get('route_pair_denominators_by_task'):
        raise ValueError('PAIR_ROUTE_DOMAIN_UNSUPPORTED_BY_LEGACY_CERTIFICATE_TOOL: use a versioned domain-aware workflow')


def validate_target(catalog, target='asr'):
    if target not in TARGET_SPECIALISTS:
        raise ValueError('Unknown target specialist: '+str(target))
    models=catalog['models']
    record=models.get(target,{})
    if record.get('role')!='specialist' or record.get('supported_tasks')!=[target]:
        raise ValueError('Target must name its dedicated specialist model/task: '+target)
    if any(m!=target and r.get('role')=='specialist' and target in r.get('supported_tasks',[])
           for m,r in models.items()):
        raise ValueError('Ambiguous specialist coverage for target: '+target)
    return target


def recorded_target(record, target='asr'):
    # Historical bundles omitted the target and were ASR-only.
    if record.get('target_specialist','asr')!=target or record.get('target_task',target)!=target:
        raise ValueError('Recorded target specialist differs from requested target')


def one_specialist_restriction(target='asr'):
    if target not in TARGET_SPECIALISTS:raise ValueError('Unknown target specialist')
    return {'replica_options':{target:[1]}}


def target_structure(catalog, plan, target='asr'):
    validate_target(catalog,target)
    row=plan['routing'][target]
    generalist=sum(q for m,q in row.items() if catalog['models'][m]['role']=='generalist')
    specialist=row.get(target,0)
    count=plan['deployment'].get(target,0)
    return dict(target_task=target,target_specialist=target,target_specialist_replicas=count,
        target_generalist_rps=generalist,target_specialist_rps=specialist,
        achieved=count>=2 and generalist>0 and specialist>0)


def assert_one_specialist_plan(plan,target='asr'):
    if plan is not None and plan['deployment'].get(target,0)>1:
        raise ValueError('Strong one-specialist control deploys more than one '+target)


def route_grid_membership(plan,rates,K,*,route_pair_denominators_by_task=None):
    """Check the existing solver domain: union of whole-task denominators1..K.

    Floats are compared at solver serialization tolerance, never rounded or
    substituted. Each task needs one denominator shared by all its model flows.
    """
    reject_extended_route_domain({'route_pair_denominators_by_task':route_pair_denominators_by_task})
    reject_extended_route_domain(plan)
    reject_extended_route_domain(plan.get('problem',{}))
    reject_extended_route_domain(plan.get('domain',{}))
    if type(K) is not int or not 1<=K<=16:raise ValueError('Invalid certified route grid')
    if set(plan['routing'])!=set(rates):raise ValueError('Candidate routing task set differs')
    result={}
    for task,rate in rates.items():
        row=plan['routing'][task]
        if rate<=0 or not row or any(type(q) not in (float,int) or not math.isfinite(q) or q<=0 for q in row.values()):
            raise ValueError('Invalid positive route rates')
        if not math.isclose(sum(row.values()),rate,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('Route conservation failure')
        matches=[]
        for denominator in range(1,K+1):
            units={m:round(q/rate*denominator) for m,q in row.items()}
            if sum(units.values())==denominator and all(u>0 and math.isclose(row[m],rate*u/denominator,rel_tol=1e-10,abs_tol=1e-10) for m,u in units.items()):
                matches.append({'denominator':denominator,'units':units})
        if not matches:raise ValueError('Candidate route outside certified denominator1..K union: '+task)
        result[task]=matches
    return {'K':K,'contract':'union of denominators1..K; original rates retained; no rounding',
        'task_membership':result}
