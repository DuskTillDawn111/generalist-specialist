"""Lower bounds for the current phase-only law on repeated zero-peer spans.

At zero peer counts, the law has one constant progress value for a target
phase. After the declared root-order adjustment, the same input therefore
has a constant predicted duration. Its exact minimax work residual is
(maximum-minimum)/(maximum+minimum). This is not a bound on richer state models.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_semantic_bands_v2 import export_band_observations
from gsdeploy.forecasting.native_order import NativeOrderLaw


def diagnose(fit):
    template=fit['profiles'];profiles={p['id']:p for p in template['profiles']}
    spans={(p['id'],s['id']):s for p in profiles.values() for s in p['segments']}
    spec=template.get('native_order_law')
    order=NativeOrderLaw(spec,template['execution_context_sha256']) if spec else None
    groups=defaultdict(list)
    for doc in fit['normalized_runs']:
        counts=defaultdict(int);ordinals={}
        for request in doc['requests']:
            ordinals[request['request_id']]=counts[request['alias']];counts[request['alias']]+=1
        for row in export_band_observations(doc,template):
            segment=spans[row['profile_id'],row['segment_id']]
            if segment['kind']!='gpu':continue
            if any(spans[m['profile_id'],m['segment_id']]['kind']=='gpu'
                   for interval in row['intervals'] for alias,m in interval['members'].items()
                   if alias!=row['target_alias']):continue
            request_id=row['id'][len(doc['run_id'])+1:-(len(row['segment_id'])+1)]
            profile=profiles[row['profile_id']]
            factor=order.factor(profile,dict(id=segment['native_root_id']),ordinals[request_id]) if order else 1.
            ms=sum(i['duration_ms'] for i in row['intervals'])
            groups[row['profile_id'],row['segment_id']].append(dict(run_id=doc['run_id'],request_id=request_id,
                native_order=ordinals[request_id],observed_ms=ms,root_order_factor=factor,adjusted_ms=ms/factor))
    results=[]
    for (pid,sid),rows in sorted(groups.items()):
        if len(rows)<2:continue
        profile=profiles[pid];segment=spans[pid,sid];ordered=sorted(rows,key=lambda r:r['adjusted_ms'])
        low,high=ordered[0]['adjusted_ms'],ordered[-1]['adjusted_ms']
        bound=(high-low)/(high+low)
        results.append(dict(family=profile['family'],task=profile['task'],pool_id=profile['pool_id'],
            profile_id=pid,segment_id=sid,role=segment['native_band_role'],observations=len(rows),
            independent_run_ids=sorted({r['run_id'] for r in rows}),minimum_adjusted_ms=low,maximum_adjusted_ms=high,
            exact_minimax_work_error=bound,constant_progress_cannot_meet_point2=bound>.2,
            minimum_witness=ordered[0],maximum_witness=ordered[-1],all_observations=rows))
    return dict(schema='gsdeploy.zero_peer_phase_variation_lower_bound.v1',groups=results,
        infeasible_groups=[{k:r[k] for k in ('family','task','pool_id','segment_id','exact_minimax_work_error')}
                           for r in results if r['constant_progress_cannot_meet_point2']],
        scope='training-only exact lower bound for constant zero-peer phase progress under the existing root-order law',
        unchanged_work_gate=.2,all_training_requests_retained=True,
        limitations=['No impossibility claim for richer causal history/state models or another execution context.',
            'This identifies model-class limitations; it neither qualifies predictions nor identifies cache/DVFS causes.'],
        physical_predictor_qualified=False,formal_utility_qualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True,type=Path)
    p.add_argument('--support-audit',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();content=a.fit.read_bytes();fit=json.loads(content);audit=json.loads(a.support_audit.read_text())
    if (audit['fit_source']['sha256']!=hashlib.sha256(content).hexdigest()
            or audit['source_receipts']!=fit['source_receipts']):
        raise ValueError('Matching raw-rederived training support audit required')
    for receipt in fit['source_receipts']:
        path,sha=receipt.rsplit(':sha256:',1)
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise ValueError('Original raw training receipt changed')
    report=diagnose(fit);report.update(fit_source=audit['fit_source'],
        support_audit_sha256=hashlib.sha256(a.support_audit.read_bytes()).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(repeated_groups=len(report['groups']),infeasible_groups=report['infeasible_groups'])))
