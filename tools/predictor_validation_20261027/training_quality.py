"""Training-only empirical family/task quality; never reads held-out results."""
from __future__ import annotations
import argparse
import collections
import copy
import json
from pathlib import Path

from gsdeploy.quality import sample_totals
from gsdeploy.tasks import page_ocr_text, score_result
from .probe_validation import receipt, read, save


def fit(raw_path, catalog_path, output):
    raw_path, catalog_path, output = map(lambda p: Path(p).resolve(), (raw_path, catalog_path, output))
    if output.exists():raise FileExistsError(output)
    raw, catalog = read(raw_path), read(catalog_path)
    if (raw.get('kind'), raw.get('backend'), raw.get('status')) != ('measured', 'native', 'completed'):
        raise ValueError('Complete measured native training raw required')
    if any(raw.get(k) for k in ('errors','runtime_errors','telemetry_errors')):
        raise ValueError('Error-free training collection required')
    rows = raw['request_records']
    if not rows or any(r.get('split') != 'training' or r.get('status') != 'completed' for r in rows):
        raise ValueError('Training-only complete request cohort required')
    ids = [r['request_id'] for r in rows]
    if len(ids) != len(set(ids)):raise ValueError('Unique training request identities required')
    context_ref = raw['context']
    if receipt(context_ref['path']) != context_ref:raise ValueError('Changed training context')
    context = read(context_ref['path'])
    residents = context['residents']
    totals, per_request, unique = {}, {}, collections.defaultdict(set)
    for row in rows:
        family, task, pool = residents[row['model']], row['task'], row['pool_id']
        result = copy.deepcopy(row['worker_response'])
        if result.get('task') != task:raise ValueError('Training response task mismatch')
        if task == 'ocr':result['prediction'] = page_ocr_text(result['prediction'])
        reference = catalog['quality']['references'][task][pool]
        numerator, denominator = sample_totals(task, score_result(reference, result))
        target = totals.setdefault(family, {}).setdefault(task, {'numerator':0.,'denominator':0.,'request_count':0})
        target['numerator'] += numerator;target['denominator'] += denominator;target['request_count'] += 1
        unique[(family,task)].add(pool)
        per_request[row['request_id']] = {'family':family,'task':task,'pool_id':pool,'numerator':numerator,'denominator':denominator}
    losses = {}
    for family,tasks in totals.items():
        for task,record in tasks.items():
            record['unique_input_count'] = len(unique[(family,task)])
            losses.setdefault(family,{})[task] = record['numerator']/record['denominator']
    import gsdeploy.tasks, gsdeploy.quality
    sources = [receipt(p) for p in (raw_path,catalog_path,context_ref['path'],__file__,gsdeploy.tasks.__file__,gsdeploy.quality.__file__)]
    value = {'schema':'gsdeploy.training_family_task_quality.v1','kind':'training_aggregate_proxy','split':'training',
        'execution_context_sha256':raw['execution_context_sha256'],'loss_by_family_task':losses,
        'totals_by_family_task':totals,'totals_by_request':per_request,'source_receipts':sources,
        'request_count':len(rows),'heldout_observations_used':False,'GPU_started':False,
        'aggregation':'For each family/task sum native score_result+sample_totals numerators and denominators, then divide; no mean request CER/WER',
        'scope':'Empirical training output-distribution proxy; repeated inputs are not independent quality examples; no assumed improvement or absolute accuracy gate'}
    output.parent.mkdir(parents=True,exist_ok=True);save(output,value)
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','catalog','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args(argv);value=fit(args.raw,args.catalog,args.output)
    print(json.dumps({'requests':value['request_count'],'loss_by_family_task':value['loss_by_family_task'],'output':str(Path(args.output).resolve())}))
    return 0

if __name__=='__main__':raise SystemExit(main())
