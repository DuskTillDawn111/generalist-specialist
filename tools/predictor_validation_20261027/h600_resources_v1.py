"""H600 primary-only resource fitting, whole-seed CV and prediction provenance.

CPU only. Conditional power pools directly observed full family/task contexts
across routes; startup and full-history owned peaks remain route conditioned.
No validation, repeat, H90, preflight or target outcome is a numerical source.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time

from . import background_resources as engine
from . import h600_capacity_v1 as primary
from .flow_profiles import map_flow_collection
from .native_flow import checked, receipt, save
from .probe_validation import read
from .profiling import digest

PROTOCOL_SCHEMA='gsdeploy.h600_resource_protocol.v1'
MANIFEST_SCHEMA='gsdeploy.h600_resource_manifest.v1'
POLICY={
    'sources':'Exactly eight H600 primary runs, two complete seeds; no old profiles or other outcomes',
    'power':'Unchanged background_resources numeric engine; duration-weighted directly observed full family/task context means pooled across H600 routes and retained seeds',
    'memory':'Each context receives one full startup-through-drain owned sampled peak from each retained run in which it appears; mean and empirical max, not instantaneous memory or a capacity bound',
    'startup':'Arithmetic mean of complete startup_begin through phase_start board joules, conditioned on candidate route; no common-startup omission',
    'candidate_peak':'Empirical maximum of same-route retained full startup-through-drain owned sampled peaks; passed as explicit predicted_prephase_peak_mib floor; not a guaranteed upper bound',
    'cross_validation':'Entire seed omitted from every power, memory and startup statistic',
    'resource_cv':'Actual heldout active-context replay is retrospective resource-only diagnostic, not a prospective queue or energy forecast',
    'missing':'Unseen contexts remain UNKNOWN; no family, route or H90 fallback',
    'startup_amortization_seconds':3600.,'safe_factor_applied':False,
    'rho_threshold':primary.RHO_MAX,'rho_tolerance':primary.RHO_TOLERANCE,
    'independent_validation_sources_used':[],'final_gain_eligible':False,
}


def sources():
    names=('h600_resources_v1.py','h600_capacity_v1.py','background_resources.py',
           'empirical_resources.py','flow_profiles.py','native_flow.py')
    return [receipt(Path(__file__).with_name(n)) for n in names]


def unique_refs(refs):
    result={}
    for ref in refs:
        checked(ref)
        if ref['path'] in result and result[ref['path']]!=ref:
            raise ValueError('Conflicting resource source receipts')
        result[ref['path']]=ref
    return list(result.values())


def prepare(suite_path,output):
    if Path(output).exists():raise FileExistsError(output)
    suite=primary.verify_suite(suite_path)
    value=dict(schema=PROTOCOL_SCHEMA,created_ns=time.perf_counter_ns(),policy=POLICY,
        training_suite=receipt(suite_path),source_receipts=unique_refs([
            *sources(),suite['source'],suite['protocol'],*[j['plan'] for j in suite['jobs']]]),
        training_outcomes_read=False,independent_outcomes_read=False,GPU_started=False)
    value['protocol_sha256']=digest(value);save(output,value);return value


def verify_protocol(path):
    value=read(path)
    if (value.get('schema')!=PROTOCOL_SCHEMA or value.get('policy')!=POLICY or
            value.get('protocol_sha256')!=digest({k:v for k,v in value.items() if k!='protocol_sha256'})):
        raise ValueError('Changed H600 resource protocol')
    unique_refs([value['training_suite'],*value['source_receipts']])
    primary.verify_suite(value['training_suite']['path'])
    return value


def fit_run(evidence,traces,residents):
    """Reuse frozen numerical engine; preserve continuous measurement cohort."""
    if evidence.get('split')!='training' or len(traces)!=1 or traces[0]['split']!='training':
        raise ValueError('One mapped training-only continuous resource cohort required')
    window=evidence['measurement_window_ns']
    samples=evidence['board_samples'];left=window['start'];right=window['observation_end']
    selected=[max((s for s in samples if s['time_ns']<=left),key=lambda s:s['time_ns']),
        *[s for s in samples if left<s['time_ns']<right],
        min((s for s in samples if s['time_ns']>=right),key=lambda s:s['time_ns'])]
    raw=dict(status='completed',backend='native',kind='measured',
        board_samples=selected,observation_start_ns=window['start'],
        observation_end_ns=window['observation_end'],
        execution_context_sha256=evidence['execution_context_sha256'])
    document=engine.fit_generalized_resources(raw,traces,residents,sources=evidence['sources'])
    peak=evidence['raw_all_sampled_owned_peak_mib']
    if type(peak) not in (int,float) or not math.isfinite(peak) or peak<0:
        raise ValueError('Finite full-history owned sampled peak required')
    # Preserve engine power integration. Explicitly replace its measurement-only
    # trial peak by the full-history trial peak, once per actually seen context.
    for row in document['task_specific_resource_omission_audit']['active_task_contexts'].values():
        row['mean_trial_sampled_peak_mib']=peak
        row['max_trial_sampled_peak_mib']=peak
    for row in document['contexts'].values():
        for scenario in row['scenarios'].values():scenario['peak_memory_mib']=peak
    return document


def load_runs(protocol_path):
    protocol=verify_protocol(protocol_path);suite=read(protocol['training_suite']['path'])
    if any(not (Path(j['output'])/'raw.json').is_file() for j in suite['jobs']):
        raise ValueError('All eight H600 primary raw files must finish before resource fitting')
    runs=[];identities=set()
    for job in suite['jobs']:
        path=Path(job['output'])/'raw.json';raw=read(path);plan=read(job['plan']['path'])
        if raw.get('plan')!=job['plan'] or raw.get('candidate_id')!=job['candidate_id']:
            raise ValueError('Resource raw differs from declared H600 primary route/plan')
        mapped=map_flow_collection(path) # default forbids validation diagnostics
        evidence=mapped['resource_evidence'];identities.add(evidence['execution_context_sha256'])
        fit=fit_run(evidence,mapped['traces'],plan['aliases'])
        runs.append(dict(seed=plan['seed'],candidate_id=job['candidate_id'],raw=receipt(path),
            plan=job['plan'],evidence=evidence,fit=fit))
    if len(identities)!=1:raise ValueError('One execution context required for H600 resource pooling')
    return protocol,runs


def candidate_overrides(runs):
    result={}
    for cid in sorted(primary.weighted.ROUTES):
        rows=[r for r in runs if r['candidate_id']==cid]
        if not rows:raise ValueError('Every candidate requires retained same-route H600 evidence')
        startup=[r['evidence']['startup_joules'] for r in rows]
        peaks=[r['evidence']['raw_all_sampled_owned_peak_mib'] for r in rows]
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in startup+peaks):
            raise ValueError('Finite startup joules and full-history peaks required')
        result[cid]=dict(predicted_startup_joules=statistics.mean(startup),
            predicted_prephase_peak_mib=max(peaks),full_history_mean_peak_mib=statistics.mean(peaks),
            full_history_max_peak_mib=max(peaks),startup_mean_joules=statistics.mean(startup),
            startup_max_joules=max(startup),training_seeds=sorted({r['seed'] for r in rows}),
            training_run_count=len(rows),sources=[r['raw'] for r in rows],
            raw_training_values=[dict(raw=r['raw'],startup_joules=r['evidence']['startup_joules'],
                full_history_peak_mib=r['evidence']['raw_all_sampled_owned_peak_mib'],
                actual_prephase_peak_mib=r['evidence']['raw_prephase_sampled_owned_peak_mib'],
                measurement_and_drain_peak_mib=r['evidence']['raw_measurement_and_drain_sampled_owned_peak_mib']) for r in rows],
            scope=POLICY['candidate_peak'],capacity_bound=False,safe_factor_applied=False)
    return result


def merge_runs(runs,protocol_ref,omitted_seed=None):
    if not runs or (omitted_seed is not None and any(r['seed']==omitted_seed for r in runs)):
        raise ValueError('Resource fold leaked omitted seed or has no training runs')
    first=runs[0]['fit'];document=copy.deepcopy(first)
    families={};tasks={}
    for run in runs:
        fit=run['fit']
        if fit['execution_context_sha256']!=first['execution_context_sha256']:
            raise ValueError('Mixed resource execution contexts')
        for key,row in fit['contexts'].items():families.setdefault(key,[]).append(row)
        for key,row in fit['task_specific_resource_omission_audit']['active_task_contexts'].items():tasks.setdefault(key,[]).append(row)
    document['contexts']={}
    for key,rows in families.items():
        duration=sum(r['diagnostics']['duration_ns'] for r in rows)
        peaks=[r['scenarios']['mean']['peak_memory_mib'] for r in rows]
        document['contexts'][key]=dict(members=rows[0]['members'],evidence='H600-primary-training:'+key,
            scenarios=dict(mean=dict(watts=sum(r['scenarios']['mean']['watts']*r['diagnostics']['duration_ns'] for r in rows)/duration,
                peak_memory_mib=statistics.mean(peaks)),max=dict(watts=max(r['scenarios']['max']['watts'] for r in rows),peak_memory_mib=max(peaks))),
            diagnostics=dict(duration_ns=duration,independent_batch_peak_count=len(rows)))
    task_rows={}
    for key,rows in tasks.items():
        duration=sum(r['duration_ns'] for r in rows);peaks=[r['mean_trial_sampled_peak_mib'] for r in rows]
        task_rows[key]=dict(members=rows[0]['members'],family_context_id=rows[0]['family_context_id'],
            duration_ns=duration,mean_watts=sum(r['mean_watts']*r['duration_ns'] for r in rows)/duration,
            independent_batch_peak_count=len(rows),mean_trial_sampled_peak_mib=statistics.mean(peaks),
            max_trial_sampled_peak_mib=max(peaks),scope=POLICY['memory'])
    idle_values=[r['fit']['telemetry']['raw_idle_sampled_peak_mib'] for r in runs if r['fit']['telemetry']['raw_idle_sampled_peak_mib'] is not None]
    document['telemetry']=dict(raw_idle_sampled_peak_mib=max(idle_values) if idle_values else None,
        raw_run_owned_sampled_peak_mib=max(r['evidence']['raw_all_sampled_owned_peak_mib'] for r in runs),
        sample_count=sum(r['fit']['telemetry']['sample_count'] for r in runs),
        capacity_bound=False,continuous_peak_known=False,full_history_peaks=True,
        max_gap_ns=max(r['fit']['telemetry']['max_gap_ns'] for r in runs))
    document['task_specific_resource_omission_audit']=dict(active_task_contexts=task_rows,
        unknown_task_mix_may_not_transfer=True,scope='H600 retained primary runs only; direct task contexts pooled across routes')
    document['definition']={**engine.DEFINITION,'H600_primary_policy':POLICY}
    document['definition_sha256']=digest(document['definition'])
    document['task_conditional_definition']={**engine.TASK_CONDITIONAL_DEFINITION,'memory':POLICY['memory']}
    document['task_conditional_definition_sha256']=digest(document['task_conditional_definition'])
    document['source_receipts']=unique_refs([protocol_ref,*sources(),*[s for r in runs for s in r['evidence']['sources']]])
    document['provenance']=dict(kind='measured',split='training',sources=document['source_receipts'])
    document['H600_training_evidence']=dict(protocol=protocol_ref,policy=POLICY,omitted_seed=omitted_seed,
        training_seeds=sorted({r['seed'] for r in runs}),runs=[dict(seed=r['seed'],candidate_id=r['candidate_id'],raw=r['raw'],plan=r['plan']) for r in runs],
        validation_sources_used=[],pooling='Conditional full family/task resource statistics pooled across H600 routes; not causal attribution')
    document['model_sha256']=digest(document)
    return document


def resource_cv(folds,runs):
    rows=[]
    for fold in folds:
        doc=read(fold['resources']['path']);contexts=doc['task_specific_resource_omission_audit']['active_task_contexts']
        for run in runs:
            if run['seed']!=fold['heldout_seed']:continue
            candidate=fold['candidate_overrides'][run['candidate_id']];e=run['evidence']
            actual_contexts=run['fit']['task_specific_resource_omission_audit']['active_task_contexts']
            missing=sorted(set(actual_contexts)-set(contexts))
            replay=None if missing else sum(contexts[k]['mean_watts']*v['duration_ns']/1e9 for k,v in actual_contexts.items())
            actual=e['measurement_joules']+e['drain_joules']
            rows.append(dict(seed=run['seed'],candidate_id=run['candidate_id'],heldout_raw=run['raw'],
                predicted_startup_joules=candidate['predicted_startup_joules'],actual_startup_joules=e['startup_joules'],
                predicted_full_history_peak_mib=candidate['predicted_prephase_peak_mib'],actual_full_history_peak_mib=e['raw_all_sampled_owned_peak_mib'],
                actual_context_replay_predicted_measurement_and_drain_joules=replay,
                actual_measurement_and_drain_joules=actual,unknown_contexts=missing,
                startup_absolute_relative_error=abs(e['startup_joules']-candidate['predicted_startup_joules'])/candidate['predicted_startup_joules'] if candidate['predicted_startup_joules']>0 else None,
                full_history_peak_absolute_relative_error=abs(e['raw_all_sampled_owned_peak_mib']-candidate['predicted_prephase_peak_mib'])/candidate['predicted_prephase_peak_mib'] if candidate['predicted_prephase_peak_mib']>0 else None,
                actual_context_replay_energy_relative_error=abs(replay-actual)/actual if replay is not None and actual>0 else None))
    return dict(schema='gsdeploy.h600_resource_training_CV.v1',rows=rows,
        scope=POLICY['resource_cv'],retrospective_training_cross_validation=True,
        independent_validation_sources_used=[],final_gain_eligible=False,GPU_started=False)


def fit(protocol_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    protocol,runs=load_runs(protocol_path);protocol_ref=receipt(protocol_path)
    output.mkdir(parents=True,exist_ok=False)
    full=merge_runs(runs,protocol_ref);save(output/'full-resources.json',full)
    full_entry=dict(resources=receipt(output/'full-resources.json'),candidate_overrides=candidate_overrides(runs),
        training_seeds=sorted({r['seed'] for r in runs}),omitted_seed=None)
    folds=[]
    for seed in primary.POLICY['training_seeds']:
        retained=[r for r in runs if r['seed']!=seed]
        document=merge_runs(retained,protocol_ref,seed);path=output/f'heldout-seed-{seed}-resources.json';save(path,document)
        folds.append(dict(heldout_seed=seed,resources=receipt(path),candidate_overrides=candidate_overrides(retained),
            training_seeds=sorted({r['seed'] for r in retained}),omitted_seed=seed))
    save(output/'resource-training-cv.json',resource_cv(folds,runs))
    manifest=dict(schema=MANIFEST_SCHEMA,created_ns=time.perf_counter_ns(),protocol=protocol_ref,policy=POLICY,
        full=full_entry,folds=folds,resource_cv=receipt(output/'resource-training-cv.json'),
        source_receipts=unique_refs([protocol_ref,*protocol['source_receipts']]),
        no_validation_data=True,independent_validation_sources_used=[],GPU_started=False,final_gain_eligible=False)
    manifest['manifest_sha256']=digest(manifest);save(output/'manifest.json',manifest)
    return manifest


def verify_manifest(path,*,recompute=True):
    value=read(path)
    if (value.get('schema')!=MANIFEST_SCHEMA or value.get('policy')!=POLICY or
            value.get('no_validation_data') is not True or value.get('independent_validation_sources_used')!=[] or
            value.get('manifest_sha256')!=digest({k:v for k,v in value.items() if k!='manifest_sha256'})):
        raise ValueError('Changed H600 resource manifest')
    protocol=verify_protocol(checked(value['protocol']));unique_refs(value['source_receipts']);checked(value['resource_cv'])
    entries=[value['full'],*value['folds']]
    if len(entries)!=3 or [f.get('heldout_seed') for f in value['folds']]!=primary.POLICY['training_seeds']:
        raise ValueError('Complete full plus two whole-seed resource models required')
    runs=None
    if recompute:_,runs=load_runs(value['protocol']['path'])
    for entry in entries:
        doc=read(checked(entry['resources']));tree=doc.get('H600_training_evidence',{})
        omitted=entry['omitted_seed'];expected=[s for s in primary.POLICY['training_seeds'] if s!=omitted]
        if (doc.get('model_sha256')!=digest({k:v for k,v in doc.items() if k!='model_sha256'}) or
                tree.get('protocol')!=value['protocol'] or tree.get('policy')!=POLICY or
                tree.get('omitted_seed')!=omitted or tree.get('training_seeds')!=expected or
                tree.get('validation_sources_used')!=[] or entry['training_seeds']!=expected):
            raise ValueError('Resource evidence identity or complete seed exclusion mismatch')
        unique_refs(doc['source_receipts'])
        declared={(r['seed'],r['candidate_id']) for r in tree['runs']}
        if len(tree['runs'])!=4*len(expected) or declared!={(s,c) for s in expected for c in primary.weighted.ROUTES}:
            raise ValueError('Resource evidence must contain exactly retained primary runs')
        if runs is not None:
            retained=[r for r in runs if r['seed']!=omitted]
            if entry['candidate_overrides']!=candidate_overrides(retained) or doc!=merge_runs(retained,value['protocol'],omitted):
                raise ValueError('Resource statistics differ from recomputed H600 training evidence')
    if runs is not None and read(value['resource_cv']['path'])!=resource_cv(value['folds'],runs):
        raise ValueError('Resource CV differs from recomputed whole-seed observations')
    return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--suite',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('fit');a.add_argument('--protocol',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('verify');a.add_argument('--manifest',required=True);a.add_argument('--output',required=True)
    args=p.parse_args(argv)
    if args.command=='prepare':prepare(args.suite,args.output)
    elif args.command=='fit':fit(args.protocol,args.output)
    else:
        doc=verify_manifest(args.manifest)
        save(args.output,dict(schema='gsdeploy.h600_resource_audit.v1',manifest=receipt(args.manifest),verified=True,
            recomputed_from_eight_H600_training_sources=True,source=receipt(__file__),GPU_started=False,
            independent_validation_sources_used=[],full=doc['full']['resources'],folds=[f['resources'] for f in doc['folds']]))
    return 0


if __name__=='__main__':raise SystemExit(main())
