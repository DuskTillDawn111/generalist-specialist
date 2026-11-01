"""Descriptive FIFO decomposition of spill count and specialist input mapping.

The counterfactual alias tapes are not admissible search plans or GPU controls.
They diagnose why the *unchanged* static router's finite traces differ.
"""
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('_mechanism_fine',Path(__file__).with_name('fine_routes.py'))
fine=importlib.util.module_from_spec(spec);spec.loader.exec_module(fine)


def decompose(base, split, vectors, baselines):
    if [(r['arrival_seconds'],r['task'],r['pool_id']) for r in base['rows']] != [(r['arrival_seconds'],r['task'],r['pool_id']) for r in split['rows']]:
        raise ValueError('Cohorts differ')
    counter=[];flows=Counter()
    for original,changed in zip(base['rows'],split['rows']):
        row=dict(changed)
        if row['task']=='text':
            flows[(original['alias'],changed['alias'])]+=1
            if changed['family']=='text':
                row['alias']=original['alias']
        counter.append(row)
    def service(row):
        if row['task']=='text':return vectors[row['alias']][row['pool_id']]
        return baselines[row['family']][row['task']]['1']['samples'][row['pool_id']]['median_service_ms']
    results={label:fine.fifo(rows,service) for label,rows in [('no_split',base['rows']),
        ('spill_keep_original_specialist_alias_COUNTERFACTUAL',counter),('actual_split_router',split['rows'])]}
    text={label:result['per_task']['text'] for label,result in results.items()}
    contributions={metric:dict(spill_with_original_alias=text['spill_keep_original_specialist_alias_COUNTERFACTUAL'][metric]-text['no_split'][metric],
        remaining_alias_remapping=text['actual_split_router'][metric]-text['spill_keep_original_specialist_alias_COUNTERFACTUAL'][metric],
        total=text['actual_split_router'][metric]-text['no_split'][metric]) for metric in ['mean_ms','p95_ms']}
    return dict(seed=base['seed'],rate=base['workload']['text'],share=split['generalist_share'],
        input_alias_flows=[dict(before=a,after=b,count=n) for (a,b),n in sorted(flows.items())],
        text_latency_scenarios=text,ordered_descriptive_contributions=contributions,
        not_admissible_counterfactual=True,U=None)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenarios',required=True,type=Path);p.add_argument('--joint-screen',required=True,type=Path)
    p.add_argument('--baseline-config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=json.loads(args.scenarios.read_text());vectors=json.loads(args.joint_screen.read_text())['service_vectors']
    baselines=json.loads(args.baseline_config.read_text())['catalog']['calibrated_replay_v2']['input_baselines']
    cases=[]
    for case in report['cases']:
        if case['generalist_share']!='1/32' or case['workload']['text']!=1.85:continue
        base=next(c for c in report['cases'] if c['generalist_share']=='0' and c['seed']==case['seed'] and c['workload']==case['workload'])
        cases.append(decompose(base,case,vectors,baselines))
    inputs=[fine.exposure.receipt(p) for p in [args.scenarios,args.joint_screen,args.baseline_config,Path(__file__),Path(fine.__file__)]]
    result=dict(schema='gsdeploy.static_route_mechanism_sensitivity.v1',cases=cases,inputs=inputs,U=None,
        limitations=['Uses unvalidated explicit full-joint text and historical background service scenarios.',
            'Counterfactual tapes preserve original specialist alias for unspilled requests; they are not configurations accepted by the unchanged static router.',
            'Ordered nonlinear decomposition is descriptive and depends on order; it is not a unique causal attribution.',
            'Alias-specific service vectors can contain run-time asymmetry; actual concurrent per-request outcomes are needed for verification.'])
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(cases=cases,output=str(args.output.resolve())),ensure_ascii=False))


if __name__=='__main__':main()
