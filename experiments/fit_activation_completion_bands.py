"""Isolated completion-boundary fit; frozen v2 fit and raw captures are retained.

The original emitted PHASE messages and raw receipts are never rewritten.
Derived documents carry a separate schema and the coarse normalization source.
"""
import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from experiments.activation_semantic_bands import SCHEMA, virtual_profiles, virtual_order_law
from experiments.activation_completion_bands import completion_band_layout as band_layout, SCHEMA as BOUNDARY_SCHEMA
from experiments.activation_completion_provider import CompletionBandContextProvider as BandContextProvider
from experiments.fit_activation_native import load_bundle
from gsdeploy.forecasting.lifecycle import fit_lifecycle
from gsdeploy.forecasting.native_context import fit_native_progress
from experiments.fit_activation_declared_idle_energy import fit_declared_idle_age_energy
from gsdeploy.forecast_errors import UnknownEvidence

DECLARED_IDLE_EDGES_SECONDS = [0., 2., 10., 20., 121.]
from gsdeploy.forecasting.native_order import NativeOrderLaw, fit_order_baselines, request_orders
from gsdeploy.forecasting.native_training import (build_template, digest, normalize_native,
                                                  stage_layout, validate_capture)
from gsdeploy.paper_training import export_segments


def build_band_template(solo_bundles,pool_items):
    coarse,_=build_template(solo_bundles,pool_items)
    coarse=fit_order_baselines(coarse,solo_bundles)
    template=copy.deepcopy(coarse)
    lookup={(p['family'],p['task'],p['pool_id']):p for p in template['profiles']}
    order=NativeOrderLaw(coarse['native_order_law'],coarse['execution_context_sha256'])
    layouts={}; values=defaultdict(list); run_ids=defaultdict(set)
    for bundle in solo_bundles:
        orders=request_orders(bundle['raw'])
        for row in bundle['raw']['request_records']:
            family=bundle['context']['resident_aliases'][row['model']]
            profile=lookup[family,row['task'],row['pool_id']]
            spans=band_layout(row,family)
            structure=[{k:span[k] for k in span if k not in ('start_ns','end_ns')} for span in spans]
            if profile['id'] in layouts and layouts[profile['id']]!=structure:
                raise ValueError('Derived training band layout differs between independent solo runs')
            layouts[profile['id']]=structure
            for span in spans:
                factor=order.factor(profile,dict(id=span.get('native_root_id',span['id'])),orders[row['request_id']])
                values[profile['id'],span['id']].append((span['end_ns']-span['start_ns'])/1e6/factor)
                run_ids[profile['id'],span['id']].add(bundle['raw']['run_id'])
    evidence=[]
    for profile in template['profiles']:
        segments=[]
        for span in layouts[profile['id']]:
            key=profile['id'],span['id']
            if len(run_ids[key])<2:
                raise ValueError('Each derived band requires independent solo repeats')
            segments.append(dict(span,duration_ms={'mean':statistics.mean(values[key])},
                demand=dict.fromkeys(('sm','dram','l2'),None if span['kind']=='gpu' else 0.)))
            evidence.append(dict(profile_id=profile['id'],segment_id=span['id'],
                independent_run_ids=sorted(run_ids[key]),root_order_adjusted_durations_ms=values[key]))
        profile['segments']=segments
    return template,coarse,evidence


def derive_document(bundle,coarse,template):
    """Replace only an explicitly derived copy, leaving all raw records intact."""
    doc=normalize_native(bundle,coarse)
    # Retain the original normalizer/exporter checks before deriving a distinct
    # schema. Never relabel derived spans as emitted coarse stage evidence.
    export_segments(doc,coarse,bundle['context']['resident_aliases'])
    by_id={row['request_id']:row for row in bundle['raw']['request_records']}
    lookup={p['id']:p for p in template['profiles']}
    manifest=[]
    for request in doc['requests']:
        profile=lookup[request['profile_id']]
        spans=band_layout(by_id[request['request_id']],profile['family'])
        if [(s['id'],s['kind']) for s in spans]!=[(s['id'],s['kind']) for s in profile['segments']]:
            raise ValueError('Derived observed layout differs from pinned training template')
        request['segments']=[dict(segment_id=s['id'],start_ns=s['start_ns'],end_ns=s['end_ns']) for s in spans]
        manifest.append(dict(request_id=request['request_id'],bands=spans))
    doc.update(schema='gsdeploy.derived_host_call_band_observations.v1',boundary_policy=BOUNDARY_SCHEMA,
        derivation_scope='host boundary partition of original mixed native spans; no emitted PHASE substitution',
        derived_band_manifest=manifest)
    return doc


def export_band_observations(doc,template):
    if doc['schema']!='gsdeploy.derived_host_call_band_observations.v1' or doc['split']!='training':
        raise ValueError('Explicit training-only derived band document required')
    profiles={p['id']:p for p in template['profiles']}
    spans=[]
    for row in doc['requests']:
        if [s['segment_id'] for s in row['segments']]!=[s['id'] for s in profiles[row['profile_id']]['segments']]:
            raise ValueError('Complete ordered derived band roster required')
        spans.extend(dict(alias=row['alias'],profile_id=row['profile_id'],request_id=row['request_id'],**s)
                     for s in row['segments'])
    events=defaultdict(list)
    for span in spans:
        events[span['start_ns']].append(('start',span));events[span['end_ns']].append(('end',span))
    active={}; intervals=defaultdict(list);cuts=sorted(events)
    for index,left in enumerate(cuts):
        for kind,span in sorted(events[left],key=lambda pair:pair[0]):
            if kind=='end':
                if active.pop(span['alias']) is not span:
                    raise ValueError('Unmatched derived owner band boundary')
            else:
                if span['alias'] in active:
                    raise ValueError('Overlapping derived owner bands')
                active[span['alias']]=span
        if index+1==len(cuts):
            continue
        right=cuts[index+1]
        members={alias:dict(profile_id=span['profile_id'],segment_id=span['segment_id']) for alias,span in active.items()}
        for span in active.values():
            intervals[span['request_id'],span['segment_id']].append(dict(duration_ms=(right-left)/1e6,members=members))
    if active:
        raise ValueError('Incomplete derived band observations')
    return [dict(id=f'{doc["run_id"]}:{span["request_id"]}:{span["segment_id"]}',split='training',
        execution_context_sha256=doc['execution_context_sha256'],target_alias=span['alias'],
        profile_id=span['profile_id'],segment_id=span['segment_id'],
        intervals=intervals[span['request_id'],span['segment_id']]) for span in spans]


def fit_band_captures(bundles,pool_items):
    if len({b['raw']['run_id'] for b in bundles})!=len(bundles):
        raise ValueError('Distinct original training capture runs required')
    for bundle in bundles:
        validate_capture(bundle)
    solo=[bundle for bundle in bundles if bundle['plan']['mode']=='solo']
    template,coarse,evidence=build_band_template(solo,pool_items)
    virtual,mapping=virtual_profiles(template)
    vtemplate=dict(profiles=virtual,execution_context_sha256=template['execution_context_sha256'])
    vorder=virtual_order_law(template)
    if vorder is not None:
        vtemplate['native_order_law']=vorder
    observations=[]; documents=[]; lifecycle=[]; energy_runs=[]
    for bundle in bundles:
        doc=derive_document(bundle,coarse,template);documents.append(doc)
        lifecycle.extend(doc['lifecycle_observations'])
        rows=export_band_observations(doc,template)
        orders=request_orders(bundle['raw'])
        by_obs={f'{doc["run_id"]}:{row["request_id"]}:{s["segment_id"]}':orders[row['request_id']]
                for row in doc['requests'] for s in row['segments']}
        for row in rows:
            target=mapping.get((row['profile_id'],row['segment_id']))
            if target is None:
                continue
            intervals=[]
            for interval in row['intervals']:
                members={alias:dict(profile_id=mapping[member['profile_id'],member['segment_id']],segment_id='native_band')
                         for alias,member in interval['members'].items()
                         if (member['profile_id'],member['segment_id']) in mapping}
                intervals.append(dict(duration_ms=interval['duration_ms'],members=members))
            observations.append(dict(row,profile_id=target,segment_id='native_band',intervals=intervals,
                                     run_id=doc['run_id'],native_order=by_obs[row['id']]))
        prefix=[s['end_ns'] for row in bundle['raw'].get('priming_request_records',[])
                for s in stage_layout(row) if s['kind']=='gpu']
        if not prefix or max(prefix)>doc['observation_window_ns'][0]:
            raise ValueError('Observed completed priming prefix required for energy fitting')
        kinds={(p['id'],s['id']):s['kind'] for p in template['profiles'] for s in p['segments']}
        energy_runs.append(dict(run_id=doc['run_id'],split='training',window_ns=doc['observation_window_ns'],
            initial_last_native_end_ns=max(prefix),telemetry_errors=bundle['raw']['telemetry_errors'],
            board_samples=bundle['raw']['board_samples'],spans=[dict(alias=row['alias'],profile_id=row['profile_id'],
                kind=kinds[row['profile_id'],s['segment_id']],**s) for row in doc['requests'] for s in row['segments']]))
    law=fit_native_progress(vtemplate,observations)
    try:
        energy,energy_diagnostic=fit_declared_idle_age_energy(
            {p['id']:p for p in template['profiles']},energy_runs,idle_edges=DECLARED_IDLE_EDGES_SECONDS)
    except UnknownEvidence as exc:
        energy=None
        energy_diagnostic=dict(accepted=False,reason=str(exc),scope='training-only domain rejection; no energy dropped')
    energy_diagnostic['declared_idle_edges_seconds']=DECLARED_IDLE_EDGES_SECONDS
    energy_diagnostic['cold_tail_assumption']='one empirical constant power coefficient over 20--121 s; prediction remains bounded by maximum observed idle age'
    template['provenance']['sources']=sorted(bundle['source'] for bundle in bundles)
    peak=max(s['owned_used_mib'] for bundle in bundles for s in bundle['raw']['board_samples'])
    template['derived_host_call_band_model']=dict(schema=SCHEMA,boundary_policy=BOUNDARY_SCHEMA,status='FITTED_UNVALIDATED',
        execution_context_sha256=template['execution_context_sha256'],physical_bound=False,admission_enabled=False,
        decode_stopping_policy='training_mean_band_work_no_future_tokens',progress_law=law,
        observed_peak_memory_mib=peak,native_root_order_law_sha256=digest(template['native_order_law']),
        root_order_assumption='same separately fitted complete-native request-order factor for every constituent band')
    if energy is not None:
        energy['execution_context_sha256']=template['execution_context_sha256']
        template['native_idle_age_energy']=energy
        BandContextProvider(template,template['execution_context_sha256'],allow_cpu_fixture=True)
    accepted=bool(law['fit_diagnostics']) and all(row['accepted'] for row in law['fit_diagnostics'])
    root_diagnostics=template['native_order_law'].get('fit_diagnostics',[])
    accepted=accepted and bool(root_diagnostics) and all(row['accepted'] for row in root_diagnostics)
    return dict(schema='gsdeploy.completion_host_band_fit.v1',
        revision_scope='OCR split at first recognition entry after detector-result consumption; declared long idle tail; unchanged work, timing and energy gates',status='FITTED_UNVALIDATED',
        admission_enabled=False,independent_accuracy_qualified=False,formal_utility_qualified=False,
        training_fit_gates_passed=bool(accepted and energy is not None and energy_diagnostic['accepted']),
        profiles=template,coarse_template=coarse,lifecycle=fit_lifecycle(lifecycle,template['execution_context_sha256']),
        baseline_evidence=evidence,energy_diagnostic=energy_diagnostic,normalized_runs=documents,
        source_receipts=[bundle['source'] for bundle in bundles])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',action='append',required=True,type=Path)
    parser.add_argument('--observer-report',required=True,type=Path)
    parser.add_argument('--pool',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    gate=json.loads(args.observer_report.read_text())
    if gate.get('observer_repeat_check_passed') is not True or gate.get('issues'):
        raise ValueError('Original complete observer gate must pass before fitting fresh semantic data')
    bundles=[load_bundle(path.resolve()) for path in args.capture]
    # Qualified semantic captures must use the exact observer executable context.
    observers=[load_bundle(Path(source.rsplit(':sha256:',1)[0]).parent)
               for name,source in gate['sources'].items() if name.startswith('semantic-')]
    contexts={digest(bundle['context']) for bundle in observers}
    if len(contexts)!=1 or any(digest(bundle['context']) not in contexts for bundle in bundles):
        raise ValueError('Training context differs from the qualified semantic observer')
    import hashlib
    content=args.pool.read_bytes()
    if any(bundle['context']['input_pool_sha256']!=hashlib.sha256(content).hexdigest() for bundle in bundles):
        raise ValueError('Frozen training input pool content changed')
    pool=[json.loads(line) for line in content.decode().splitlines()]
    paths=sorted((ROOT/'gsdeploy/forecasting').glob('*.py'))+[
        Path(__file__),ROOT/'experiments/activation_semantic_bands.py',
        ROOT/'experiments/activation_completion_bands.py',ROOT/'experiments/activation_completion_provider.py',
        ROOT/'experiments/fit_activation_native.py',ROOT/'experiments/fit_activation_declared_idle_energy.py',
        ROOT/'gsdeploy/paper_training.py',ROOT/'gsdeploy/energy.py']
    sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    result=fit_band_captures(bundles,pool)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha for path,sha in sources.items()):
        raise ValueError('Completion fit source changed during execution')
    result['fitting_sources']=sources
    result['observer_report_sha256']=hashlib.sha256(args.observer_report.read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(args.output),training_fit_gates_passed=result['training_fit_gates_passed'],
        rejected_groups=[row for row in result['profiles']['derived_host_call_band_model']['progress_law']['fit_diagnostics']
                         if not row['accepted']],energy_accepted=result['energy_diagnostic']['accepted'])))


if __name__=='__main__':
    main()
