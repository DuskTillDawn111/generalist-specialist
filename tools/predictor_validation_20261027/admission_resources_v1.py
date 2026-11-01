"""Admission2-only per-route resource fitting and complete-seed CV, CPU only.

Reuse frozen numerical integration without importing any shared fitted values.
Each route has its own controller-bound conditional power/memory document.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time

from . import admission_capacity_v1 as capacity
from . import background_resources as engine
from .h600_resources_v1 import fit_run as numerical_fit_run
from .native_flow import checked, receipt, save
from .probe_validation import read
from .profiling import digest

SCHEMA = 'gsdeploy.admission2_training_resources.v1'
POLICY = dict(training_seeds=list(capacity.SEEDS), candidate_ids=list(capacity.ROUTES),
    pooling='Only same route and exact admission2 controller; no shared/pilot/independent outcomes',
    power='Duration-weighted directly observed full family/task conditional whole-board means',
    startup='Same-route retained-run arithmetic mean from startup_begin through phase_start',
    peak='Same-route retained-run full startup-through-drain owned sampled maximum; no capacity guarantee',
    memory='Each actually observed task context receives each retained run full-history owned sampled peak once',
    cv='Entire seed omitted from startup, power and owned memory',
    resource_cv='Retrospective heldout actual-context energy replay only; not prospective queue/energy validation',
    missing='Unseen family/task context remains UNKNOWN, no cross-route or shared fallback',
    startup_amortization_seconds=3600., safe_factor_applied=False,
    rho_threshold=capacity.RHO_MAX, rho_tolerance=capacity.RHO_TOLERANCE,
    independent_validation_sources_used=[], capacity_bound=False, final_gain_eligible=False)


def sources():
    return [receipt(p) for p in (__file__,capacity.__file__,engine.__file__,
        numerical_fit_run.__code__.co_filename)] + capacity.sources()


def prepare(campaign_path, output):
    if Path(output).exists():
        raise FileExistsError(output)
    campaign = capacity.training.verify(campaign_path)
    value = dict(schema=SCHEMA+'.protocol',created_ns=time.perf_counter_ns(),policy=POLICY,
        campaign=receipt(campaign_path),controller_binding=campaign['controller_binding'],
        source_receipts=capacity.unique_refs([*sources(),*campaign['source_receipts']]),
        training_outcomes_read=False,independent_outcomes_read=False,GPU_started=False)
    value['protocol_sha256'] = digest(value)
    save(output,value)
    return value


def verify_protocol(path):
    value = read(path)
    if (value.get('schema') != SCHEMA+'.protocol' or value.get('policy') != POLICY
            or type(value.get('created_ns')) is not int
            or value.get('training_outcomes_read') is not False
            or value.get('independent_outcomes_read') is not False
            or value.get('protocol_sha256') != digest({k:v for k,v in value.items() if k!='protocol_sha256'})):
        raise ValueError('Changed admission resource protocol')
    capacity.unique_refs([value['campaign'],*value['source_receipts']])
    campaign = capacity.training.verify(value['campaign']['path'])
    if value['controller_binding'] != campaign['controller_binding']:
        raise ValueError('Admission resource controller differs from campaign')
    required = {(r['path'],r['sha256']) for r in [*sources(),*campaign['source_receipts']]}
    if not required <= {(r['path'],r['sha256']) for r in value['source_receipts']}:
        raise ValueError('Resource protocol omitted required sources')
    return value


def fit_run(evidence, traces, residents, binding):
    identity = binding['controller_sha256']
    if (evidence.get('controller_sha256') != identity or len(traces) != 1
            or identity != digest({k:v for k,v in binding.items() if k!='controller_sha256'})
            or traces[0].get('controller_sha256') != identity
            or evidence.get('resource_configuration') != binding['resource_configuration']
            or traces[0].get('resource_configuration') != binding['resource_configuration']
            or evidence.get('execution_context_sha256') != binding['base_worker_context_sha256']
            or binding.get('resource_configuration') != capacity.training.native_flow.resource_configuration(2)):
        raise ValueError('Resource fit requires exact admission2 training controller')
    # This helper consumes only caller-supplied evidence; it loads no shared data.
    return numerical_fit_run(evidence,traces,residents)


def load_runs(protocol_path):
    protocol = verify_protocol(protocol_path)
    # Use the common strict admission mapper/loader, without constructing or
    # modifying any shared training-suite or validation wrapper.
    bundle = capacity.load_campaign(protocol['campaign']['path'])
    runs = bundle['runs']
    if any(not protocol['created_ns'] < run['startup_begin_ns'] for run in runs):
        raise ValueError('Resource protocol must be frozen before every training startup')
    for run in runs:
        run['fit'] = fit_run(run['evidence'],run['traces'],run['plan']['aliases'],
            run['plan']['controller_binding'])
    return protocol,runs


def overrides(runs):
    result = {}
    for cid in capacity.ROUTES:
        rows = [r for r in runs if r['candidate_id']==cid]
        if not rows:
            raise ValueError('Same-route retained admission evidence required')
        startup = [r['evidence']['startup_joules'] for r in rows]
        peaks = [r['evidence']['raw_all_sampled_owned_peak_mib'] for r in rows]
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in startup+peaks):
            raise ValueError('Finite admission startup/owned peak required')
        result[cid] = dict(predicted_startup_joules=statistics.mean(startup),
            predicted_prephase_peak_mib=max(peaks),training_seeds=sorted(r['seed'] for r in rows),
            sources=[r['raw_source'] for r in rows],controller_sha256=rows[0]['controller_sha256'],
            resource_configuration=rows[0]['plan']['controller_binding']['resource_configuration'],
            raw_values=[dict(seed=r['seed'],startup_joules=r['evidence']['startup_joules'],
                owned_sampled_peak_mib=r['evidence']['raw_all_sampled_owned_peak_mib']) for r in rows],
            safe_factor_applied=False,capacity_bound=False)
    return result


def merge_route(runs, cid, protocol_ref, omitted_seed=None):
    rows = [r for r in runs if r['candidate_id']==cid]
    if not rows or any(r['seed']==omitted_seed for r in rows):
        raise ValueError('Missing route or heldout seed leaked into resource fit')
    first = rows[0]
    if any((r['condition'],r['generation'],r['controller_sha256']) !=
           (first['condition'],first['generation'],first['controller_sha256']) for r in rows):
        raise ValueError('Cannot pool admission resource conditions/controllers')
    document = copy.deepcopy(first['fit'])
    family, task = {}, {}
    for run in rows:
        for key,row in run['fit']['contexts'].items():
            family.setdefault(key,[]).append(row)
        for key,row in run['fit']['task_specific_resource_omission_audit']['active_task_contexts'].items():
            task.setdefault(key,[]).append(row)
    document['contexts'] = {}
    for key,values in family.items():
        duration = sum(v['diagnostics']['duration_ns'] for v in values)
        peaks = [v['scenarios']['mean']['peak_memory_mib'] for v in values]
        document['contexts'][key] = dict(members=values[0]['members'],evidence='admission2:'+cid+':'+key,
            scenarios=dict(mean=dict(watts=sum(v['scenarios']['mean']['watts']*v['diagnostics']['duration_ns'] for v in values)/duration,
                peak_memory_mib=statistics.mean(peaks)),max=dict(watts=max(v['scenarios']['max']['watts'] for v in values),peak_memory_mib=max(peaks))),
            diagnostics=dict(duration_ns=duration,independent_batch_peak_count=len(values)))
    document['task_specific_resource_omission_audit'] = dict(active_task_contexts={
        key:dict(members=values[0]['members'],family_context_id=values[0]['family_context_id'],
            duration_ns=sum(v['duration_ns'] for v in values),
            mean_watts=sum(v['mean_watts']*v['duration_ns'] for v in values)/sum(v['duration_ns'] for v in values),
            independent_batch_peak_count=len(values),
            mean_trial_sampled_peak_mib=statistics.mean(v['mean_trial_sampled_peak_mib'] for v in values),
            max_trial_sampled_peak_mib=max(v['max_trial_sampled_peak_mib'] for v in values),scope=POLICY['memory'])
        for key,values in task.items()},unknown_task_mix_may_not_transfer=True,scope=POLICY['pooling'])
    idle = [r['fit']['telemetry']['raw_idle_sampled_peak_mib'] for r in rows
        if r['fit']['telemetry']['raw_idle_sampled_peak_mib'] is not None]
    document['telemetry'] = dict(raw_idle_sampled_peak_mib=max(idle) if idle else None,
        raw_run_owned_sampled_peak_mib=max(r['evidence']['raw_all_sampled_owned_peak_mib'] for r in rows),
        sample_count=sum(r['fit']['telemetry']['sample_count'] for r in rows),
        max_gap_ns=max(r['fit']['telemetry']['max_gap_ns'] for r in rows),
        continuous_peak_known=False,capacity_bound=False,full_history_peaks=True)
    document['definition'] = {**engine.DEFINITION,'admission2_policy':POLICY}
    document['definition_sha256'] = digest(document['definition'])
    document['task_conditional_definition'] = {**engine.TASK_CONDITIONAL_DEFINITION,'memory':POLICY['memory']}
    document['task_conditional_definition_sha256'] = digest(document['task_conditional_definition'])
    document['source_receipts'] = capacity.unique_refs([protocol_ref,*sources(),
        *[s for r in rows for s in r['evidence']['sources']]])
    document['provenance'] = dict(kind='measured',split='training',sources=document['source_receipts'])
    document['admission_training_evidence'] = dict(protocol=protocol_ref,policy=POLICY,
        controller_binding=first['plan']['controller_binding'],candidate_id=cid,
        condition=first['condition'],generation=first['generation'],omitted_seed=omitted_seed,
        training_seeds=sorted(r['seed'] for r in rows),
        runs=[dict(seed=r['seed'],candidate_id=cid,raw=r['raw_source'],plan=r['plan_source']) for r in rows],
        independent_validation_sources_used=[])
    document.pop('model_sha256',None)
    document['model_sha256'] = digest(document)
    return document


def resource_cv(entries,runs):
    rows = []
    for entry in entries:
        for run in runs:
            if run['seed'] != entry['omitted_seed']:
                continue
            cid = run['candidate_id']; e = run['evidence']
            document = read(entry['route_resources'][cid]['path'])
            trained = document['task_specific_resource_omission_audit']['active_task_contexts']
            actual = run['fit']['task_specific_resource_omission_audit']['active_task_contexts']
            missing = sorted(set(actual)-set(trained))
            replay = None if missing else sum(trained[k]['mean_watts']*v['duration_ns']/1e9 for k,v in actual.items())
            candidate = entry['candidate_overrides'][cid]
            rows.append(dict(seed=run['seed'],candidate_id=cid,heldout_raw=run['raw_source'],
                training_seeds=entry['training_seeds'],
                predicted_startup_joules=candidate['predicted_startup_joules'],actual_startup_joules=e['startup_joules'],
                predicted_owned_sampled_peak_mib=candidate['predicted_prephase_peak_mib'],
                actual_owned_sampled_peak_mib=e['raw_all_sampled_owned_peak_mib'],
                actual_context_replay_predicted_measurement_and_drain_joules=replay,
                actual_measurement_and_drain_joules=e['measurement_joules']+e['drain_joules'],unknown_contexts=missing))
    return dict(schema=SCHEMA+'.training_cv',policy=POLICY,rows=rows,
        scope=POLICY['resource_cv'],independent_validation_sources_used=[],final_gain_eligible=False,GPU_started=False)


def fit(protocol_path,output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    protocol,runs = load_runs(protocol_path)
    ref = receipt(protocol_path)
    # All numerical computations/validation precede output creation.
    documents = {(omitted,cid):merge_route([r for r in runs if r['seed']!=omitted],cid,ref,omitted)
        for omitted in (None,*capacity.SEEDS) for cid in capacity.ROUTES}
    capacity.unique_refs([ref,*protocol['source_receipts'],*[s for r in runs for s in r['evidence']['sources']]])
    output.mkdir(parents=True,exist_ok=False)
    entries = []
    for omitted in (None,*capacity.SEEDS):
        retained = [r for r in runs if r['seed']!=omitted]
        refs = {}
        for cid in capacity.ROUTES:
            path = output/((f'heldout-{omitted}' if omitted else 'full')+'-'+cid+'-resources.json')
            save(path,documents[(omitted,cid)]); refs[cid] = receipt(path)
        entries.append(dict(omitted_seed=omitted,training_seeds=sorted({r['seed'] for r in retained}),
            route_resources=refs,candidate_overrides=overrides(retained)))
    save(output/'resource-training-cv.json',resource_cv(entries[1:],runs))
    value = dict(schema=SCHEMA+'.manifest',created_ns=time.perf_counter_ns(),policy=POLICY,
        protocol=ref,full=entries[0],folds=entries[1:],resource_cv=receipt(output/'resource-training-cv.json'),
        source_receipts=capacity.unique_refs([ref,*protocol['source_receipts'],*[s for r in runs for s in r['evidence']['sources']]]),
        no_validation_data=True,independent_validation_sources_used=[],final_gain_eligible=False,GPU_started=False)
    value['manifest_sha256'] = digest(value)
    save(output/'manifest.json',value)
    return value


def verify_manifest(path,*,recompute=True):
    value = read(path)
    if (value.get('schema') != SCHEMA+'.manifest' or value.get('policy') != POLICY
            or value.get('independent_validation_sources_used') != [] or value.get('no_validation_data') is not True
            or value.get('final_gain_eligible') is not False or value.get('GPU_started') is not False
            or value.get('manifest_sha256') != digest({k:v for k,v in value.items() if k!='manifest_sha256'})):
        raise ValueError('Changed admission resource manifest')
    verify_protocol(checked(value['protocol'])); capacity.unique_refs(value['source_receipts'])
    entries = [value['full'],*value['folds']]
    if [e['omitted_seed'] for e in entries] != [None,*capacity.SEEDS]:
        raise ValueError('Full and both whole-seed resource folds required')
    runs = load_runs(value['protocol']['path'])[1] if recompute else None
    for entry in entries:
        omitted = entry['omitted_seed']; seeds = [s for s in capacity.SEEDS if s!=omitted]
        if entry['training_seeds'] != seeds or set(entry['route_resources']) != set(capacity.ROUTES):
            raise ValueError('Exact retained seed/route resource domain required')
        for cid,ref in entry['route_resources'].items():
            doc = read(checked(ref)); tree = doc.get('admission_training_evidence',{})
            if (doc.get('model_sha256') != digest({k:v for k,v in doc.items() if k!='model_sha256'})
                    or tree.get('protocol') != value['protocol'] or tree.get('policy') != POLICY
                    or tree.get('candidate_id') != cid or tree.get('omitted_seed') != omitted
                    or tree.get('training_seeds') != seeds or tree.get('independent_validation_sources_used') != []):
                raise ValueError('Resource route/controller/seed evidence changed')
            capacity.unique_refs(doc['source_receipts'])
            if runs is not None:
                retained = [r for r in runs if r['seed']!=omitted]
                if doc != merge_route(retained,cid,value['protocol'],omitted) or entry['candidate_overrides'] != overrides(retained):
                    raise ValueError('Resource values differ from recomputed admission training')
    cv = read(checked(value['resource_cv']))
    if runs is not None and cv != resource_cv(value['folds'],runs):
        raise ValueError('Resource CV differs from recomputed admission training')
    return value


def provider(document,plan,candidate):
    """Explicit route/controller applicability before constructing a provider."""
    identity = capacity.controller_identity(plan,candidate)
    tree = document.get('admission_training_evidence',{})
    if (tree.get('controller_binding',{}).get('controller_sha256') != identity
            or tree.get('policy') != POLICY
            or tree.get('candidate_id') != candidate['id']
            or tree.get('condition') != capacity.condition(plan,candidate)
            or tree.get('generation') != capacity._generation(plan,tree['condition'])
            or document.get('model_sha256') != digest({k:v for k,v in document.items() if k!='model_sha256'})):
        raise ValueError('Resource provider requires matching admission route/controller/condition')
    capacity.unique_refs(document['source_receipts'])
    return engine.TaskConditionalResourceProvider(document,candidate['execution_context_sha256'])


def main():
    p = argparse.ArgumentParser(description=__doc__);sub = p.add_subparsers(dest='operation',required=True)
    a=sub.add_parser('prepare');a.add_argument('--campaign',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('fit');a.add_argument('--protocol',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('verify');a.add_argument('--manifest',required=True)
    args=p.parse_args()
    value=prepare(args.campaign,args.output) if args.operation=='prepare' else fit(args.protocol,args.output) if args.operation=='fit' else verify_manifest(args.manifest)
    print(json.dumps(dict(schema=value['schema'],GPU_started=False)))


if __name__=='__main__':
    main()
