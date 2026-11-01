"""Merge disjoint training quality evidence without changing frozen v7 tools."""
from __future__ import annotations
import argparse
import collections
import copy
import json
from pathlib import Path
from .probe_validation import read,receipt,save


def merge(paths,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if len(paths)<2:raise ValueError('At least two separate training quality documents required')
    docs=[read(p) for p in paths];refs=[receipt(p) for p in paths]
    if len({r['sha256'] for r in refs})!=len(refs):raise ValueError('Duplicate training quality source')
    contexts={d['execution_context_sha256'] for d in docs}
    if len(contexts)!=1:raise ValueError('Matching training execution contexts required')
    totals={};per_request={};unique=collections.defaultdict(set);sources=[];raw_sources=set()
    for document,ref in zip(docs,refs):
        if (document.get('schema')!='gsdeploy.training_family_task_quality.v1' or document.get('split')!='training'
                or document.get('heldout_observations_used') is not False):
            raise ValueError('Training-only quality evidence required')
        # The original fit records native raw first, followed by catalog,
        # context and scorers. Reject even differently named copies of it.
        raw_ref=document['source_receipts'][0]
        if raw_ref['sha256'] in raw_sources:raise ValueError('Repeated raw training observations')
        raw_sources.add(raw_ref['sha256'])
        for source in document['source_receipts']:
            if receipt(source['path'])!=source:raise ValueError('Changed training quality source receipt')
            sources.append(source)
        sources.append(ref)
        if len(document['totals_by_request'])!=document['request_count']:raise ValueError('Training request totals coverage mismatch')
        for rid,row in document['totals_by_request'].items():
            family,task=row['family'],row['task'];target=totals.setdefault(family,{}).setdefault(task,dict(numerator=0.,denominator=0.,request_count=0))
            target['numerator']+=row['numerator'];target['denominator']+=row['denominator'];target['request_count']+=1
            unique[(family,task)].add(row['pool_id']);per_request[raw_ref['sha256']+':'+rid]=copy.deepcopy(row)
    losses={}
    for family,tasks in totals.items():
        for task,row in tasks.items():
            row['unique_input_count']=len(unique[(family,task)]);losses.setdefault(family,{})[task]=row['numerator']/row['denominator']
    sources.append(receipt(__file__));sources=list({(r['path'],r['sha256']):r for r in sources}.values())
    value={'schema':'gsdeploy.training_family_task_quality.v1','kind':'training_aggregate_proxy','split':'training',
        'execution_context_sha256':next(iter(contexts)),'loss_by_family_task':losses,'totals_by_family_task':totals,
        'totals_by_request':per_request,'source_receipts':sources,'request_count':len(per_request),
        'training_documents':refs,'heldout_observations_used':False,'GPU_started':False,
        'aggregation':'Sum per-request native training numerators and denominators across distinct raw runs, then divide; no mean of document CER/WER',
        'scope':'Combined empirical training distribution; repeated inputs remain repeated observations, not independent quality examples'}
    output.parent.mkdir(parents=True,exist_ok=True);save(output,value);return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--quality',action='append',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv);value=merge(args.quality,args.output)
    print(json.dumps({'requests':value['request_count'],'loss_by_family_task':value['loss_by_family_task'],'output':str(Path(args.output).resolve())}))
    return 0

if __name__=='__main__':raise SystemExit(main())
