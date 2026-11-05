"""Explain phase exposure gaps without dropping rare vertices or fitting them.

This training-only audit reports the fitter's exact repetition requirements,
examples for targeted new captures, and a zero-coefficient Jacobian diagnostic.
Structurally allowed owner states are not asserted to be physically reachable.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.activation_semantic_bands import virtual_profiles, virtual_order_law
from experiments.fit_activation_semantic_bands_v2 import export_band_observations
from experiments.fit_activation_native import load_bundle
from gsdeploy.forecasting.native_context import categories, counts, design
from gsdeploy.forecasting.native_order import NativeOrderLaw
from gsdeploy.forecasting.native_training import digest, validate_capture


def possible_vertices(names, original_families, aliases, target_family):
    """One phase/task per actual peer owner, including its idle state."""
    roster=Counter(aliases.values())
    if roster[target_family]<1:raise ValueError('Target family absent from resident roster')
    roster[target_family]-=1
    choices=[]
    for family,n in sorted(roster.items()):
        allowed=[None,*[i for i,name in enumerate(names) if original_families[name]==family]]
        choices.extend([allowed]*n)
    if math.prod(map(len,choices))>100000:
        raise ValueError('Explicit structural enumeration too large; use a separately declared audit domain')
    result=set()
    for combination in itertools.product(*choices):
        vector=[0.]*len(names)
        for index in combination:
            if index is not None:vector[index]+=1.
        result.add(tuple(vector))
    return result


def audit(template, documents, aliases):
    virtual,mapping=virtual_profiles(template)
    profiles={p['id']:p for p in virtual};names=categories(profiles)
    original={(p['family'],p['task']):p['original_family'] for p in virtual}
    vorder=virtual_order_law(template)
    order=NativeOrderLaw(vorder,template['execution_context_sha256']) if vorder else None
    groups=defaultdict(list);seen_runs=set()
    for doc in documents:
        if (doc['split']!='training' or doc['execution_context_sha256']!=template['execution_context_sha256']
                or doc['run_id'] in seen_runs):
            raise ValueError('Distinct context-bound training documents required')
        seen_runs.add(doc['run_id']);orders={};owner_counts=defaultdict(int)
        for request in doc['requests']:
            orders[request['request_id']]=owner_counts[request['alias']]
            owner_counts[request['alias']]+=1
        # This is the same complete segment sweep used by both band fitters.
        for row in export_band_observations(doc,template):
            target=mapping.get((row['profile_id'],row['segment_id']))
            if target is None:continue
            profile=profiles[target];segment=profile['segments'][0]
            baseline=segment['duration_ms']['mean']
            request_id=row['id'][len(doc['run_id'])+1:-(len(row['segment_id'])+1)]
            if order:baseline*=order.factor(profile,segment,orders[request_id])
            exposures=[]
            for interval in row['intervals']:
                members={alias:dict(profile_id=mapping[m['profile_id'],m['segment_id']],segment_id='native_band')
                    for alias,m in interval['members'].items() if (m['profile_id'],m['segment_id']) in mapping}
                x=tuple(counts(members,profiles,names,omit=row['target_alias']))
                exposures.append((interval['duration_ms'],x))
            key=(profile['family'],profile['task'])
            groups[key].append(dict(run_id=doc['run_id'],observation_id=row['id'],request_id=request_id,
                profile_id=target,pool_id=profile['pool_id'],baseline_ms=baseline,exposures=exposures))
    reports=[]
    for key,rows in sorted(groups.items()):
        vertices=defaultdict(lambda:dict(runs=set(),observations=set(),duration_ms=0.,examples={}))
        input_runs=defaultdict(set)
        for row in rows:
            input_runs[row['profile_id']].add(row['run_id'])
            for ms,x in row['exposures']:
                v=vertices[x];v['runs'].add(row['run_id']);v['observations'].add(row['observation_id']);v['duration_ms']+=ms
                v['examples'].setdefault(row['run_id'],dict(request_id=row['request_id'],pool_id=row['pool_id']))
        used=[i for i in range(len(names)) if any(x[i] for x in vertices)]
        terms=[[]]+[[i] for i in used]+[[i,j] for i in used for j in used if i<j and any(x[i]*x[j] for x in vertices)]
        matrix=np.array([sum((ms/row['baseline_ms']*design(x,terms) for ms,x in row['exposures']),np.zeros(len(terms))) for row in rows])
        structural=possible_vertices(names,original,aliases,original[key])
        if set(vertices)-structural:raise ValueError('Observed vertex violates actual owner roster')
        def labels(x):return [dict(family=names[i][0],task=names[i][1],count=n) for i,n in enumerate(x) if n]
        details=[dict(peer_counts=list(x),peer_categories=labels(x),independent_run_ids=sorted(v['runs']),
            independent_runs=len(v['runs']),observations=len(v['observations']),total_exposure_ms=v['duration_ms'],
            examples=[dict(run_id=r,**e) for r,e in sorted(v['examples'].items())]) for x,v in sorted(vertices.items())]
        missing_inputs=[dict(profile_id=ident,independent_run_ids=sorted(runs)) for ident,runs in input_runs.items() if len(runs)<2]
        insufficient=[v for v in details if v['independent_runs']<2]
        reports.append(dict(family=key[0],task=key[1],observations=len(rows),parameters=len(terms),
            terms=terms,initial_jacobian_rank=int(np.linalg.matrix_rank(matrix)),
            initial_jacobian_singular_values=np.linalg.svd(matrix,compute_uv=False).tolist(),
            observed_vertex_count=len(details),insufficiently_repeated_vertices=insufficient,
            insufficiently_repeated_inputs=missing_inputs,
            necessary_repetition_checks_passed=len(rows)>=len(terms) and not insufficient and not missing_inputs,
            observed_vertices=details,structurally_allowed_vertex_count=len(structural),
            unobserved_structurally_allowed_vertices=[dict(peer_counts=list(x),peer_categories=labels(x)) for x in sorted(structural-set(vertices))]))
    return dict(schema='gsdeploy.training_band_support_audit.v1',categories=[list(n) for n in names],groups=reports,
        execution_context_sha256=template['execution_context_sha256'],independent_run_ids=sorted(seen_runs),
        limits=['Structural owner states may be physically unreachable; new actual phase exposure must be measured.',
            'Initial Jacobian rank is diagnostic only; full nonlinear fit must meet its original rank and work-error gates.',
            'Every observed vertex is retained regardless of duration; repeated requests in one capture are not independent runs.',
            'No observations, fits, admission thresholds or independent validation data are modified.'],
        physical_predictor_qualified=False,formal_utility_qualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();content=a.fit.read_bytes();fit=json.loads(content)
    if fit['schema'] not in ('gsdeploy.derived_host_call_band_fit.v2','gsdeploy.completion_host_band_fit.v1'):
        raise ValueError('Explicit supported band fit artifact required')
    bundles=[load_bundle(Path(source.rsplit(':sha256:',1)[0]).parent) for source in fit['source_receipts']]
    if [b['source'] for b in bundles]!=fit['source_receipts']:raise ValueError('Original training receipts changed')
    contexts={validate_capture(b) for b in bundles}
    if contexts!={fit['profiles']['execution_context_sha256']}:raise ValueError('Training context differs from fit')
    if fit['schema']=='gsdeploy.completion_host_band_fit.v1':
        from experiments.fit_activation_completion_bands import derive_document
    else:
        from experiments.fit_activation_semantic_bands_v2 import derive_document
    documents=[derive_document(b,fit['coarse_template'],fit['profiles']) for b in bundles]
    if documents!=fit['normalized_runs']:raise ValueError('Derived documents no longer match the saved fit')
    report=audit(fit['profiles'],documents,bundles[0]['raw']['resident_aliases'])
    report.update(fit_source=dict(path=str(a.fit.resolve()),sha256=hashlib.sha256(content).hexdigest()),
        source_receipts=fit['source_receipts'],analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps([dict(family=g['family'],task=g['task'],parameters=g['parameters'],rank=g['initial_jacobian_rank'],
        observed=g['observed_vertex_count'],unrepeated=len(g['insufficiently_repeated_vertices']),
        unobserved=len(g['unobserved_structurally_allowed_vertices'])) for g in report['groups']]))
