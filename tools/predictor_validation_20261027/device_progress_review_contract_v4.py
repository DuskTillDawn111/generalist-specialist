"""Exact execution descriptors shared by raw and actual CPU reviewers."""
from dataclasses import dataclass,asdict
from pathlib import Path

from .device_progress_smoke_raw_verify_v3 import FROZEN,checked
from .implementation_source_closure_v27 import strict_references
from .native_collect import receipt,context_identity


@dataclass(frozen=True)
class Contract:
    campaign_module:str
    campaign_schema:str
    protocol_prefix:str
    supervisor_module:str
    supervisor_schema:str
    supervisor_context:str
    monitoring_scope:str
    owned_peak_scope:str
    pipeline:tuple

    def document(self):return asdict(self)


V6=Contract(campaign_module='paired_profile_campaign_v6',campaign_schema='gsdeploy.paired_profile_campaign.v6',
    protocol_prefix='paired_profile_v6_',supervisor_module='profile_training_supervisor_v3',
    supervisor_schema='gsdeploy.profile_training_supervision.v3',supervisor_context='profile_training_supervisor_v3_outer_lease_only',
    monitoring_scope='startup_through_final_drain_sample_before_worker_close',
    owned_peak_scope='sampled_live_declared_launcher_descendants_across_PGID_startup_through_final_drain_before_worker_close_not_instantaneous',
    pipeline=(('paired_profile_campaign_v6.py','bc22011c4b12d9c0baaa2d4f2a207ca52ba04f355ecbe6ae6a527ad24644c17d'),
        ('profile_owned_collect_v3.py','1c3222addcafbabbda813ab680822090df966062a9430e7fc3d83dc814c52d83'),
        ('profile_training_supervisor_v3.py','854dfc77b43b2bf7af87935bbe145e7202b81d53359143f61834048d72f4e6a5')))


def qualify_execution_sources(context,contract=V6):
    members=context.get('execution_sources',[]);originals={m['original_path']:m for m in members}
    if len(originals)!=len(members):raise ValueError('Duplicate execution source identity')
    if context.get('sources')!=[m['frozen'] for m in members]:raise ValueError('Exact ordered source closure required')
    refs=[]
    for name,sha in {**FROZEN,**dict(contract.pipeline)}.items():
        actual=str(Path(__file__).with_name(name).resolve());member=originals.get(actual)
        if member is None or member['frozen'].get('sha256')!=sha or receipt(member['frozen']['path'])!=member['frozen'] or receipt(actual)['sha256']!=sha:
            raise ValueError('Exact execution descriptor source required: '+name)
        refs.extend([receipt(actual),member['frozen']])
    return strict_references(refs)


def qualify_roles(raw,contract=V6):
    plan=checked(raw.get('plan'));context=checked(raw.get('context'));mode=plan.get('mode');split=plan.get('split')
    if (plan.get('schema')!=contract.campaign_schema or mode not in ('natural','kernel') or split not in ('smoke','training')
            or plan.get('progress_observer') is not True or plan.get('mode_switch_at_execute') is not False or plan.get('final_gain_eligible') is not False):
        raise ValueError('Exact descriptor non-validation campaign role required')
    if (raw.get('schema')!='gsdeploy.native_phase_collection.v1' or raw.get('kind')!='measured' or raw.get('backend')!='native'
            or raw.get('split',split)!=split or raw.get('not_for_training') or raw.get('validation_labels_used')
            or raw.get('performance_scope')!=('profiler_perturbed' if mode=='kernel' else 'unprofiled_with_phase_markers')
            or raw.get('execution_context_sha256')!=plan.get('execution_context_sha256')):
        raise ValueError('Raw schema/performance/role conflicts with execution descriptor')
    if (context.get('paired_profile_mode')!=mode or context.get('protocol')!=contract.protocol_prefix+mode
            or context.get('supervisor')!=contract.supervisor_context or context.get('worker_variant')!='semantic_progress_worker_v1.py'
            or context.get('monitoring_scope')!=contract.monitoring_scope):raise ValueError('Exact descriptor protocol/supervisor/monitoring context required')
    sources=qualify_execution_sources(context,contract)
    if context_identity(context)!=plan['execution_context_sha256']:raise ValueError('Exact descriptor context identity required')
    jobs=plan.get('jobs',[]);rows=raw.get('request_records',[]);view='correctness_clock_only_not_fit' if split=='smoke' else 'training'
    if not jobs or not rows or any(j.get('split')!=split or j.get('validation_view')!=view for j in jobs):raise ValueError('Every planned job needs exact non-validation role')
    by_job={j.get('id'):j for j in jobs}
    if len(by_job)!=len(jobs) or None in by_job:raise ValueError('Unique declared job identity required')
    if any(r.get('split')!=split or r.get('profile_job_id') not in by_job or r.get('validation_view')!=view for r in rows):raise ValueError('Every measured row needs exact declared non-validation job role')
    return dict(plan_schema=plan['schema'],split=split,mode=mode,role_source='exact_receipted_plan_jobs_and_all_measured_row_splits',
        raw_top_level_split_present='split' in raw,validation_roles_forbidden=True,exact_pipeline_sources=sources,execution_descriptor=contract.document())


def monitoring_boundaries(raw,contract=V6):
    """Retain failures; monitoring ends before close, never samples shutdown."""
    issues=[];boundaries=raw.get('lifecycle_boundaries',[]);samples=raw.get('board_samples',[])
    names=[b.get('boundary') for b in boundaries];times=[b.get('time_ns') for b in boundaries]
    expected=['startup_begin','startup_ready','priming_end','measurement_and_drain_end','monitoring_end','shutdown_begin','shutdown_end']
    if names!=expected or any(type(t) is not int or t<0 for t in times) or any(a>b for a,b in zip(times,times[1:])):issues.append('EXACT_ORDERED_LIFECYCLE_BOUNDARIES_REQUIRED')
    if raw.get('monitoring_scope')!=contract.monitoring_scope or raw.get('owned_peak_scope')!=contract.owned_peak_scope or raw.get('shutdown_memory_sampled') is not False:issues.append('EXACT_STARTUP_THROUGH_FINAL_DRAIN_SCOPE_REQUIRED')
    if issues:return dict(issues=issues)
    b=dict(zip(names,times));valid_samples=bool(samples) and all(type(s.get('time_ns')) is int for s in samples)
    if not valid_samples:issues.append('NONEMPTY_TIMESTAMPED_MONITOR_SAMPLES_REQUIRED')
    else:
        stamps=[s['time_ns'] for s in samples]
        if any(a>b for a,b in zip(stamps,stamps[1:])):issues.append('MONITOR_SAMPLE_CLOCK_ORDER')
        if not (stamps[0]<=b['startup_begin']<=b['startup_ready']<=raw.get('observation_start_ns',-1)<raw.get('observation_end_ns',-1)==b['measurement_and_drain_end']<=stamps[-1]<=b['monitoring_end']<b['shutdown_begin']<=b['shutdown_end']):issues.append('EXACT_STARTUP_DRAIN_FINAL_SAMPLE_MONITOR_BEFORE_CLOSE_CHAIN')
        if samples[0].get('phase')!='startup' or samples[-1].get('phase')!='final_drain' or any(s.get('phase') not in ('startup','priming','measurement','final_drain') for s in samples) or sum(s.get('phase')=='final_drain' for s in samples)!=1:issues.append('MONITOR_PHASE_SCOPE_OR_EXACT_FINAL_SAMPLE')
    return dict(issues=issues,monitoring_scope=contract.monitoring_scope,shutdown_memory_sampled=False,
        boundaries=b,total_samples=len(samples),final_sample_time_ns=samples[-1]['time_ns'] if valid_samples else None)
