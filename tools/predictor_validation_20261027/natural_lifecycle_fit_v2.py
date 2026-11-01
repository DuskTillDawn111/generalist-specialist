"""Fit host handoff only from distinct, strictly reviewed natural captures.

The explicit clock evidence is a source-bound assertion about local host clocks;
it does not map CUDA clocks or infer worker readiness from subsequent STARTs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.lifecycle import export_training_observations, fit_lifecycle
from .kernel_domain_bridge_review_v3 import load_report
from .native_collect import receipt, save
from .profiling import RESIDENTS
from .smoke_verifier_v3 import check_receipt

CLOCK_SCHEMA='gsdeploy.natural_training_host_clock_evidence.v2'


def fit(report_paths,clock_evidence_path,output):
    clock_path=Path(clock_evidence_path).resolve()
    clock=json.loads(clock_path.read_text())
    if (clock.get('schema')!=CLOCK_SCHEMA or not clock.get('sources')
            or not all(check_receipt(s) for s in clock['sources'])):
        raise ValueError('Explicit unchanged host clock evidence required')
    contract=copy.deepcopy(clock['clock_contract'])
    contract['source']='sha256:'+receipt(clock_path)['sha256']
    output=Path(output).resolve()
    captures=[];observations=[];contexts=set();seen=set();index=[]
    aliases={a:dict(family=f) for a,f in RESIDENTS.items()}
    for path in report_paths:
        report,plan,raw=load_report(path,'natural')
        sha=report['raw']['sha256']
        if sha in seen:raise ValueError('Repeated raw is not an independent capture')
        seen.add(sha)
        if report['raw'] not in clock['sources']:
            raise ValueError('Clock evidence must bind every actual training raw')
        context=json.loads(Path(report['context']['path']).read_text())
        if context['execution'].get('admission')!='default_shared_no_permit':
            raise ValueError('Only actual shared execution has no permit release')
        ctx=plan['execution_context_sha256'];contexts.add(ctx)
        if len(contexts)!=1:raise ValueError('Natural execution contexts differ')
        run_id='raw-sha256:'+sha
        wrapper={**raw,'split':'training','admission_enabled':False}
        rows=export_training_observations(wrapper,aliases,execution_context_sha256=ctx,
            clock_contract=contract,source=run_id)
        observations.extend(rows)
        index.extend(dict(run_id=run_id,request_id=r['request_id'],pool_id=r['pool_id'],task=r['task'])
                     for r in raw['request_records'])
        captures.append(dict(review=receipt(path),raw=report['raw'],run_id=run_id))
    if not captures:raise ValueError('Natural training captures required')
    counts=Counter((r['family'],r['task'],r['pool_id']) for r in observations)
    issues=[];model=None
    if len(captures)<2:issues.append('AT_LEAST_TWO_INDEPENDENT_NATURAL_CAPTURES_REQUIRED')
    else:
        try:model=fit_lifecycle(observations,next(iter(contexts)))
        except UnknownEvidence as exc:issues.append(str(exc))
    output.mkdir(parents=True,exist_ok=False)
    save(output/'observations.json',dict(observations=observations))
    save(output/'training-index.json',dict(schema='gsdeploy.independent_prediction_gate_training_index.v2',
        split='training',records=index))
    if model and any(p['scenarios']['mean'][field] is None for p in model['profiles']
                     for field in ('parent_receipt_ms','worker_next_ready_ms')):
        issues.append('INCOMPLETE_HANDOFF_BOUNDARIES_RETAINED_AS_UNKNOWN')
    result=dict(schema='gsdeploy.natural_training_lifecycle_fit.v2',
        status=('FITTED_PARTIAL' if issues else 'FITTED') if model else 'UNKNOWN',
        issues=issues,captures=captures,independent_capture_count=len(captures),
        execution_context_sha256=next(iter(contexts)),clock_evidence=receipt(clock_path),
        observations=receipt(output/'observations.json'),training_index=receipt(output/'training-index.json'),
        profile_counts=[dict(family=k[0],task=k[1],pool_id=k[2],observations=v) for k,v in sorted(counts.items())],
        sources=[receipt(__file__),receipt(Path(__file__).with_name('kernel_domain_bridge_review_v3.py')),
                 receipt(Path(__file__).parents[2]/'gsdeploy/forecasting/lifecycle.py')],
        GPU_started=False,validation_started=False,physical_accuracy_validated=False,
        limitation='Exact training input mean handoff, not load-independent tails or unseen-input support.')
    if model:
        save(output/'lifecycle.json',model);result['model']=receipt(output/'lifecycle.json')
    save(output/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews',nargs='+',required=True)
    parser.add_argument('--clock-evidence',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=fit(args.reviews,args.clock_evidence,args.output)
    print(json.dumps(dict(status=result['status'],issues=result['issues'],output=str(Path(args.output).resolve()))))


if __name__=='__main__':main()
