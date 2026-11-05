"""Prospective phase alignment from frozen training means, never future timing.

Intent is recorded separately from realized exposure. Short first-forward
windows can be missed, and their requests must remain in the training record.
"""
from collections import defaultdict
import itertools

from gsdeploy.forecasting.native_order import NativeOrderLaw


def phase_conditions(template, aliases, sizes):
    """Cross actual owners' phase/task choices; a G2 task is not an owner."""
    options=defaultdict(set)
    for profile in template['profiles']:
        for span in profile['segments']:
            if span['kind']=='gpu':
                options[profile['family']].add((profile['task'],span['native_band_role']))
    if len(set(aliases.values()))!=len(aliases):
        raise ValueError('Initial crossed campaign requires one actual owner per family')
    conditions=[]
    for n in sizes:
        if n<2 or n>len(aliases):raise ValueError('Declared active owner count outside resident roster')
        for roster in itertools.combinations(sorted(aliases),n):
            for phases in itertools.product(*(sorted(options[aliases[a]]) for a in roster)):
                conditions.append([dict(alias=a,family=aliases[a],task=task,role=role)
                                   for a,(task,role) in zip(roster,phases)])
    return conditions


def selected_profiles(template):
    """Both duration extremes per path, selected solely from training means."""
    by_path=defaultdict(list)
    for profile in template['profiles']:
        by_path[profile['family'],profile['task']].append(profile)
    result={}
    for key,profiles in by_path.items():
        ranked=sorted(profiles,key=lambda p:(sum(s['duration_ms']['mean'] for s in p['segments'] if s['kind']=='gpu'),p['pool_id']))
        if len(ranked)<2:raise ValueError('Distinct short/long training input profiles required')
        result[key]={'short':ranked[0],'long':ranked[-1]}
    return result


def align(condition, profiles, template, ordinals, *, fraction=.5):
    """Compute offer offsets before execution, using only mean templates."""
    if not 0<fraction<1:raise ValueError('Strict interior target fraction required')
    if len({c['alias'] for c in condition})!=len(condition):raise ValueError('One request per actual owner required')
    spec=template.get('native_order_law')
    order=NativeOrderLaw(spec,template['execution_context_sha256']) if spec else None
    centers=[];members=[]
    for member,profile in zip(condition,profiles,strict=True):
        if (member['family'],member['task'])!=(profile['family'],profile['task']):
            raise ValueError('Training profile and target path differ')
        cursor=0.;target=None
        for span in profile['segments']:
            duration=span['duration_ms']['mean']
            if order and span['kind']=='gpu':
                duration*=order.factor(profile,dict(id=span['native_root_id']),ordinals[member['alias']])
            if span.get('native_band_role')==member['role']:
                if target is not None:raise ValueError('Target role occurs more than once')
                target=dict(start_ms=cursor,end_ms=cursor+duration,segment_id=span['id'])
            cursor+=duration
        if target is None:raise ValueError('Target phase missing from pinned input template')
        centers.append(target['start_ms']+fraction*(target['end_ms']-target['start_ms']))
        members.append(dict(member,pool_id=profile['pool_id'],profile_id=profile['id'],
            training_band=target,planned_owner_ordinal=ordinals[member['alias']]))
    anchor=max(centers)
    requests=[dict(alias=m['alias'],task=m['task'],pool_id=m['pool_id'],offset_ms=anchor-center)
              for m,center in zip(members,centers)]
    return dict(requests=requests,intended_phase_alignment=dict(
        source='frozen training mean host-band durations with causal planned owner ordinal',
        target_fraction=fraction,target_time_ms=anchor,members=members,
        actual_overlap_guaranteed=False,future_observations_used=False,
        nonattainment_policy='retain every offered request and score realized full exposures; no dropping missed phase overlaps'))
