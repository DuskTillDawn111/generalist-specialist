"""Training-only quality expectations in original per-request metric units."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import statistics

from gsdeploy.quality import sample_totals
from gsdeploy.tasks import page_ocr_text, score_result
from .kernel_domain_bridge_review_v3 import load_report
from .native_collect import receipt, save
from .profiling import RESIDENTS


def expected_totals(rows,evidence):
    """Expected totals per offered request, not a mean of request CER/WERs."""
    if not rows:raise ValueError('Nonempty quality training group required')
    return dict(basis='expected_per_request_totals',
        numerator=statistics.mean(r['numerator'] for r in rows),
        denominator=statistics.mean(r['denominator'] for r in rows),evidence=evidence)


def fit(reviews,catalog_path,output):
    catalog_path=Path(catalog_path).resolve()
    catalog=json.loads(catalog_path.read_text())
    groups=defaultdict(list);records=[];sources=[];seen=set();contexts=set()
    for path in reviews:
        report,plan,raw=load_report(path,'natural')
        source=report['raw'];run_id='raw-sha256:'+source['sha256']
        if source['sha256'] in seen:raise ValueError('Repeated capture cannot add quality evidence')
        seen.add(source['sha256']);contexts.add(plan['execution_context_sha256'])
        if len(contexts)!=1:raise ValueError('Quality training execution contexts differ')
        for row in raw['request_records']:
            family=RESIDENTS[row['model']];task=row['task'];pool=row['pool_id']
            response=copy.deepcopy(row['worker_response'])
            if task=='ocr':response['prediction']=page_ocr_text(response['prediction'])
            reference=catalog['quality']['references'][task][pool]
            numerator,denominator=sample_totals(task,score_result(reference,response))
            result=dict(run_id=run_id,request_id=row['request_id'],family=family,task=task,pool_id=pool,
                        numerator=numerator,denominator=denominator)
            records.append(result);groups[family,task].append(result)
        sources.extend([receipt(path),source])
    if not records:raise ValueError('Natural training-only completed requests required')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    save(output/'training-totals.json',dict(records=records))
    ref=receipt(output/'training-totals.json');evidence='sha256:'+ref['sha256']
    quality={}
    for (family,task),rows in groups.items():
        pools=defaultdict(list)
        for row in rows:pools[row['pool_id']].append(row)
        quality.setdefault(family,{})[task]=dict(
            by_input={pool:expected_totals(members,evidence) for pool,members in sorted(pools.items())},
            fallback=expected_totals(rows,evidence))
    save(output/'quality-loss.json',quality)
    save(output/'training-index.json',dict(schema='gsdeploy.independent_prediction_gate_training_index.v2',
        split='training',records=[{k:r[k] for k in ('run_id','request_id','pool_id','task')} for r in records]))
    sources.extend([receipt(catalog_path),receipt(__file__),receipt(Path(__file__).parents[2]/'gsdeploy/tasks.py'),
                    receipt(Path(__file__).parents[2]/'gsdeploy/quality.py')])
    result=dict(schema='gsdeploy.natural_training_quality_fit.v2',status='FITTED',
        execution_context_sha256=next(iter(contexts)),quality_loss=receipt(output/'quality-loss.json'),
        training_totals=ref,training_index=receipt(output/'training-index.json'),sources=sources,
        independent_capture_count=len(seen),request_count=len(records),
        expected_totals_basis='Arithmetic mean of per-request numerator and denominator separately; never mean request CER/WER.',
        fallback_assumption='Same family/task training-input distribution; not actual unseen-output quality or an accuracy bound.',
        GPU_started=False,validation_started=False,physical_accuracy_validated=False)
    save(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews',nargs='+',required=True);parser.add_argument('--catalog',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    result=fit(args.reviews,args.catalog,args.output)
    print(json.dumps(dict(status=result['status'],request_count=result['request_count'],output=str(Path(args.output).resolve()))))


if __name__=='__main__':main()
