"""Describe signed independent timing errors without changing qualification."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gsdeploy.forecasting.native_validation import metrics
from gsdeploy.forecasting.native_training import stage_layout
from gsdeploy.forecasting.native_order import request_orders


def describe(report):
    groups=defaultdict(list);worst=defaultdict(list)
    for run in report['runs']:
        source,sha=run['source'].rsplit(':sha256:',1);path=Path(source)
        content=path.read_bytes()
        if hashlib.sha256(content).hexdigest()!=sha:
            raise ValueError('Original independent raw receipt changed')
        raw=json.loads(content);rows={row['request_id']:row for row in raw['request_records']}
        plan=json.loads((path.parent/'plan.json').read_text());jobs={job['id']:job for job in plan['jobs']}
        orders=request_orders(raw)
        for batch in run['batches']:
            if batch['status']!='predicted':continue
            job=jobs[batch['id']]
            roster=','.join(sorted({item['alias'] for item in job['requests']}))
            for prediction in batch['requests']:
                row=rows[prediction['request_id']];family=raw['resident_aliases'][row['model']]
                actual=prediction['observed_e2e_ms'];predicted=prediction['predicted_e2e_ms']
                signed=(predicted-actual)/actual
                coarse=stage_layout(row)
                item=dict(run_id=raw['run_id'],request_id=row['request_id'],job_id=job['id'],family=family,
                    task=row['task'],pool_id=row['pool_id'],roster=roster,native_order=orders[row['request_id']],
                    observed_e2e_ms=actual,predicted_e2e_ms=predicted,signed_relative_error=signed,
                    observed_native_ms=sum((s['end_ns']-s['start_ns'])/1e6 for s in coarse if s['kind']=='gpu'),
                    observed_host_ms=sum((s['end_ns']-s['start_ns'])/1e6 for s in coarse if s['kind']!='gpu'),
                    cache_reclamation=row['worker_response'].get('cache_reclamation'))
                groups[family,row['task'],'all'].append(item)
                groups[family,row['task'],'job_kind:'+job['id'].split(':')[0]].append(item)
                groups[family,row['task'],'roster:'+roster].append(item)
                worst[family,row['task']].append(item)
    summaries=[]
    for (family,task,condition),items in sorted(groups.items()):
        signed=[item['signed_relative_error'] for item in items]
        summaries.append(dict(family=family,task=task,condition=condition,
            absolute_relative_error=metrics([abs(value) for value in signed]),
            mean_signed_relative_error=statistics.mean(signed),overprediction_fraction=sum(value>0 for value in signed)/len(signed),
            mean_observed_e2e_ms=statistics.mean(item['observed_e2e_ms'] for item in items),
            mean_predicted_e2e_ms=statistics.mean(item['predicted_e2e_ms'] for item in items),
            independent_run_ids=sorted({item['run_id'] for item in items})))
    return dict(schema='gsdeploy.independent_timing_transfer_diagnostic.v1',
        original_gate_passed=report['development_accuracy_passed'],original_reasons=report['reasons'],
        summaries=summaries,worst_by_path=[dict(family=family,task=task,requests=sorted(items,
            key=lambda item:abs(item['signed_relative_error']),reverse=True)[:10]) for (family,task),items in sorted(worst.items())],
        model_refitted=False,formal_utility_qualified=False,
        scope='descriptive heldout error diagnosis only; no thresholds changed or counterfactual device cause inferred')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();content=args.report.read_bytes();result=describe(json.loads(content))
    result.update(report_sha256=hashlib.sha256(content).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps([item for item in result['summaries'] if item['condition']=='all']))


if __name__=='__main__':main()
