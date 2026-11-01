"""Separate Gamma-Poisson physical SAVE rates for all 64 holder/handler states.

This CPU-only candidate uses every qualified original v41 risk interval.
Constant intensity conditional on the current four-handler mask is a strong,
unvalidated hypothesis. It changes rates, never marks, restore age or budgets.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from .idle_physical_save_hazard_v53 import posterior, predictive
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .native_collect import receipt
from .owner_lifecycle_mark_v21 import state_features
from .profiling import digest
from .projected_handler_context_v42 import checked, load_dataset, mask, merged_intervals, validate_risk
from .unknown_preflight_archive_v1 import save_synced
from .whole57_shared_observable_risk_v31 import OWNERS

SCHEMA = 'gsdeploy.projected_state_physical_SAVE_counting_process.v57'
ORIGINAL_MANIFEST_SHA256 = '4cbb2725b4ede8e8fc948844feb83daa38b97cd0e8cac81aa16eda93258735b4'
KEYS = tuple(owner+':'+format(bits,'04b') for owner in OWNERS for bits in range(16))
COUNT_FIELDS = ('rows','exposure_ns','events','unresolved_rows','unresolved_exposure_ns','unresolved_events')
PARAMETERS = dict(
    training_population='all original v41 qualified paired risk; positive and zero SAVE states included',
    strata='64 separate held_owner by complete four_owner handler_active mask; no pooling across strata',
    mask_bit_order=list(OWNERS), likelihood='homogeneous_Poisson_in_accumulated_exposure_conditional_on_current_holder_and_handler_mask',
    prior='fixed_improper_Jeffreys_density_proportional_to_lambda_power_minus_one_half',
    posterior_shape='N+0.5',posterior_rate='T_seconds',runtime_point='posterior_mean_per_second_divided_by_1000',
    no_exposure='UNKNOWN',unresolved_risk='accounted_separately_not_used_as_qualified_exposure',
    age_dependence='constant_with_attained_age_conditional_on_mask_UNVALIDATED',
    age_support='union_of_observed_qualified_attained_age_intervals_not_min_max_continuity',
    capture_weighting='exact_pooled_exposure_and_event_counts_no_event_normalization',
    holdout='four leave_one_whole_original_training_capture_out_count_predictive_and_log_probability_no_tuning',
    training_to_runtime_transfer='UNVALIDATED_FIXED_MEDIAN_WORKER_HANDLER_PROJECTION',
    runtime_features='held_owner_and_four_known_handler_booleans_only',
    issued_gpu_ready_and_frontier_used_as_rate_features=False,future_arrivals_used=False,
    mark_replaced=False,cumulative_budget_replaced=False,restore_age_reset=False,
    physical_context_equals_SM_service=False,independent_validation=False)


def empty():
    return dict(rows=0,exposure_ns=0,events=0,unresolved_rows=0,unresolved_exposure_ns=0,
        unresolved_events=0,age_min_ns=None,age_max_ns=None,age_support_intervals_ns=[])


def aggregate(risks,captures):
    counts={c:{key:empty() for key in KEYS} for c in captures};seen=set();events=set()
    for row in risks:
        validate_risk(row)
        if row['capture'] not in counts:raise ValueError('Risk cannot cross declared captures')
        identity=(row['capture'],tuple(row['driver_context']),row['start_ns'],row['end_ns'])
        if identity in seen:raise ValueError('Risk exposure cannot be duplicated')
        seen.add(identity)
        if row['physical_SAVE_END_observed']:
            event=(row['capture'],tuple(row['driver_context']),row['end_ns'])
            if event in events:raise ValueError('Physical SAVE event cannot be duplicated')
            events.add(event)
        value=counts[row['capture']][row['held_owner']+':'+mask(row['shared_left_features'])]
        width=row['end_ns']-row['start_ns'];event=int(row['physical_SAVE_END_observed'])
        if not row['exposure_qualified']:
            value['unresolved_rows']+=1;value['unresolved_exposure_ns']+=width;value['unresolved_events']+=event
            continue
        value['rows']+=1;value['exposure_ns']+=width;value['events']+=event
        value['age_support_intervals_ns'].append([row['age_entry_ns'],row['age_exit_ns']])
    for capture in counts.values():
        for value in capture.values():
            support=merged_intervals(value['age_support_intervals_ns']);value['age_support_intervals_ns']=support
            value['age_min_ns']=support[0][0] if support else None
            value['age_max_ns']=support[-1][1] if support else None
    return counts


def pool(counts,captures,key):
    values=[counts[c][key] for c in captures]
    result={field:sum(v[field] for v in values) for field in COUNT_FIELDS}
    support=merged_intervals([span for v in values for span in v['age_support_intervals_ns']])
    return dict(**result,age_min_ns=support[0][0] if support else None,
        age_max_ns=support[-1][1] if support else None,age_support_intervals_ns=support)


def coverage(fitted):
    return dict(strata=64,exposed_strata=sum(v['exposure_ns']>0 for v in fitted.values()),
        positive_event_strata=sum(v['events']>0 for v in fitted.values()),
        zero_event_exposed_strata=sum(v['exposure_ns']>0 and v['events']==0 for v in fitted.values()),
        unseen_strata=[key for key,v in fitted.items() if not v['exposure_ns']],
        totals={field:sum(v[field] for v in fitted.values()) for field in COUNT_FIELDS},
        by_holder={o:dict(exposed_masks=[key.split(':')[1] for key,v in fitted.items() if key.startswith(o+':') and v['exposure_ns']],
            unseen_masks=[key.split(':')[1] for key,v in fitted.items() if key.startswith(o+':') and not v['exposure_ns']]) for o in OWNERS})


def train(manifest_path,output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    manifest_ref=receipt(manifest_path);manifest=checked(manifest_ref)
    if (manifest_ref['sha256']!=ORIGINAL_MANIFEST_SHA256
            or set(manifest)!={'training_only','validation_labels_used','captures'}
            or manifest['training_only'] is not True or manifest['validation_labels_used'] is not False):
        raise ValueError('Exact frozen original v41 four-capture training-only manifest required')
    data=load_dataset(manifest['captures']);captures=sorted(data['captures'])
    counts=aggregate(data['risks'],captures);fitted={key:posterior(pool(counts,captures,key)) for key in KEYS}
    folds=[]
    for heldout in captures:
        training=[c for c in captures if c!=heldout]
        models={key:posterior(pool(counts,training,key)) for key in KEYS}
        diagnostics={key:predictive(models[key],counts[heldout][key]) for key in KEYS}
        scored=[v for v in diagnostics.values() if 'log_predictive_probability' in v]
        folds.append(dict(held_out_capture=heldout,held_out_route=data['captures'][heldout]['route'],training_captures=training,
            fitted=models,diagnostics=diagnostics,scored_strata=len(scored),
            sum_log_predictive_probability=sum(v['log_predictive_probability'] for v in scored),
            zero_training_exposure_strata=[key for key,v in diagnostics.items() if v['status']=='UNKNOWN_NO_TRAINING_EXPOSURE']))
    implementations=implementation_sources(__file__);sources=strict_references([manifest_ref,data['sources'],implementations])
    for ref in sources:
        if receipt(ref['path'])!=ref:raise ValueError('Training source changed')
    summary=coverage(fitted)
    model=dict(schema=SCHEMA,parameters=PARAMETERS,training_only=True,validation_labels_used=False,
        captures=data['captures'],by_capture=counts,fitted=fitted,coverage=summary,sources=sources,
        original_training_manifest=manifest_ref,implementation=implementations,software=dict(scipy=__import__('scipy').__version__))
    model['model_sha256']=digest(model)
    out.mkdir(parents=True,exist_ok=False)
    save_synced(out/'training-manifest.json',dict(schema=SCHEMA,parameters=PARAMETERS,original=manifest_ref,
        sources=sources,GPU_started=False,validation_labels_read=False))
    save_synced(out/'model.json',model)
    report=dict(schema=SCHEMA,status='CPU_TRAINED_UNVALIDATED_ALL64_STATE_HAZARD_CANDIDATE',
        model=receipt(out/'model.json'),training_manifest=receipt(out/'training-manifest.json'),summary=summary,
        by_state=fitted,leave_one_whole_capture_out=folds,sources=sources,GPU_started=False,validation_labels_read=False,
        independent_prediction_accuracy_validated=False,mark_replaced=False,budget_redrawn=False,
        four_folds_are_overlapping_training_diagnostics=True,
        limitations=['Constant age-independent intensity conditional on current holder/mask is a strong hypothesis, including positive-event states.',
            'Complete exposure is pooled by exact time, assuming route exchangeability and Poisson independent increments; no capture-event normalization.',
            'Jeffreys prior is fixed and improper. Tiny zero-event exposure can yield large positive posterior means; zero events never imply zero hazard.',
            '24 pooled states have no qualified exposure and remain UNKNOWN, including OCR masks listed in the coverage table.',
            'Observed attained-age support is an interval union. Extrapolation outside or across gaps is flagged, without resetting restore age or budget.',
            'Fixed median worker-handler clock projection and its transfer to runtime handler state are uncalibrated.',
            'Issued work, GPU readiness, future arrivals and frontier do not enter this hazard. Physical SAVE is not SM service or useful progress.',
            'This rate model does not identify inactive-target physical marks, same-owner reacquisition, service readiness, queue accuracy or utility.',
            'Leave-one-capture predictive counts/log scores are training diagnostics, not independent validation or hyperparameter selection.'])
    for ref in sources:
        if receipt(ref['path'])!=ref:raise ValueError('Training source changed during fit')
    save_synced(out/'report.json',report);return report


class ProjectedStateSaveHazard:
    """Supply the current state's rate; cumulative v24 owns age and budget."""
    def __init__(self,model):
        if (model.get('schema')!=SCHEMA or model.get('parameters')!=PARAMETERS
                or model.get('training_only') is not True or model.get('validation_labels_used') is not False
                or model.get('model_sha256')!=digest({k:v for k,v in model.items() if k!='model_sha256'})
                or set(model.get('fitted',{}))!=set(KEYS)):
            raise ValueError('Exact immutable all64 v57 candidate required')
        for key,value in model['fitted'].items():
            stats={field:value[field] for field in (*COUNT_FIELDS,'age_min_ns','age_max_ns','age_support_intervals_ns')}
            support=stats['age_support_intervals_ns']
            if (any(type(stats[f]) is not int or stats[f]<0 for f in COUNT_FIELDS)
                    or stats['events']>stats['rows'] or stats['unresolved_events']>stats['unresolved_rows']
                    or any(not isinstance(span,list) or len(span)!=2 or any(type(x) is not int for x in span) or not 0<=span[0]<span[1] for span in support)
                    or support!=merged_intervals(support)
                    or (stats['exposure_ns']>0)!=(bool(support))
                    or stats['age_min_ns']!=(support[0][0] if support else None)
                    or stats['age_max_ns']!=(support[-1][1] if support else None)
                    or posterior(stats)!=value):
                raise ValueError('Exact unpooled exposure/posterior/age support required')
        if 'by_capture' in model:
            if set(model['by_capture'])!=set(model['captures']) or any(set(v)!=set(KEYS) for v in model['by_capture'].values()):
                raise ValueError('Complete per-capture all64 accounting required')
            for key in KEYS:
                if posterior(pool(model['by_capture'],sorted(model['captures']),key))!=model['fitted'][key]:
                    raise ValueError('Per-capture sufficient statistics must reconcile')
        for ref in model['sources']:
            if receipt(ref['path'])!=ref:raise ValueError('v57 source drift')
        self.model=copy.deepcopy(model)

    def query(self,held_owner,pending_frontier,owner_lifecycle,context_age_ns,horizon_ns=0):
        state_features(owner_lifecycle)
        if (held_owner not in OWNERS or type(context_age_ns) is not int or context_age_ns<0
                or type(horizon_ns) is not int or horizon_ns<0):
            raise ValueError('Known holder and nonnegative integer attained age/horizon required')
        if any(owner_lifecycle[o].handler_active is None for o in OWNERS):
            raise UnknownEvidence('PROJECTED_STATE_SAVE_UNKNOWN_HANDLER_MASK')
        bits=''.join('1' if owner_lifecycle[o].handler_active else '0' for o in OWNERS);key=held_owner+':'+bits
        value=self.model['fitted'][key]
        if value['status']=='UNKNOWN_NO_EXPOSURE':raise UnknownEvidence('PROJECTED_STATE_SAVE_NO_HOLDER_MASK_EXPOSURE:'+key)
        flags=['UNCALIBRATED_FIXED_MEDIAN_WORKER_HANDLER_PROJECTION','UNVALIDATED_CONSTANT_PHYSICAL_SAVE_RATE_CONDITIONAL_ON_HOLDER_HANDLER_MASK']
        if not value['events']:flags.append('ZERO_EVENT_STATE_USES_FIXED_JEFFREYS_POSTERIOR_MEAN')
        support=value['age_support_intervals_ns']
        if not any(lo<=context_age_ns<hi for lo,hi in support):flags.append('PROJECTED_STATE_AGE_OUTSIDE_OBSERVED_EXPOSURE_UNION')
        if horizon_ns and not any(lo<=context_age_ns and context_age_ns+horizon_ns<=hi for lo,hi in support):
            flags.append('PROJECTED_STATE_HORIZON_OUTSIDE_OBSERVED_EXPOSURE_UNION')
        rate=value['hazard_per_ms'];seconds=horizon_ns/1e9
        return dict(hazard_per_ms=rate,OOD_reasons=flags,restricted_horizon_ns=horizon_ns,conditional_CDF=None,
            restricted_candidate_CDF=-math.expm1(-rate*horizon_ns/1e6),
            posterior_predictive_CDF=-math.expm1(-value['shape']*math.log1p(seconds/value['rate_seconds'])),
            target='physical_context_SAVE_END_not_SM_service',probability_calibrated=False,
            state_held_fixed_through_horizon=True,projected_state_hazard_version=57,state_key=key,
            model_sha256=self.model['model_sha256'],mark_replaced=False,budget_redrawn=False,restore_age_reset=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--training-manifest',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=train(args.training_manifest,args.output)
    print(json.dumps(dict(status=report['status'],summary=report['summary'])))
