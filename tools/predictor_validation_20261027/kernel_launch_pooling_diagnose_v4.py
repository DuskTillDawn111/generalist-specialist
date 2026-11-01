"""CPU impossibility diagnostics for exact-geometry pooling, without refitting."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np

from .kernel_baseline_intake_v3 import checked,_verify
from .native_collect import receipt,save


def diagnose(fit_path,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    source=receipt(fit_path);report=checked(source);doc=checked(report['models']);_verify(report['observations'])
    points=defaultdict(list);groups=defaultdict(list)
    with Path(report['observations']['path']).open() as handle:
        for line in handle:
            row=json.loads(line);key=(row['group_id'],tuple(row['features'][f] for f in doc['features']))
            points[key].append(row);groups[row['group_id']].append(row)
    impossible=Counter();family=Counter();phase=Counter();per_group=defaultdict(list);records=[]
    for (group_id,features),rows in points.items():
        values=[r['duration_ns'] for r in rows];low,high=min(values),max(values)
        # A single prediction p must satisfy .75*y <= p <= 1.25*y for every y.
        cannot_fit=0.75*high>1.25*low
        within_run={run:[r['duration_ns'] for r in rows if r['run_id']==run] for run in {r['run_id'] for r in rows}}
        impossible_within_run=any(0.75*max(v)>1.25*min(v) for v in within_run.values())
        if cannot_fit:
            impossible['exact_feature_points']+=1;impossible['observations']+=len(rows)
            impossible['within_run_also_impossible_points']+=impossible_within_run
            family[doc['groups'][group_id]['family']]+=len(rows);phase[doc['groups'][group_id]['semantic_phase']]+=len(rows)
        point=dict(features=dict(zip(doc['features'],features)),observation_count=len(rows),
            min_duration_ns=low,max_duration_ns=high,max_min_ratio=high/low,
            duration_quantiles_ns=np.quantile(values,[0,0.25,0.5,0.75,1]).tolist(),
            any_single_prediction_impossible_at_25_percent=cannot_fit,
            impossible_even_within_one_capture=impossible_within_run)
        per_group[group_id].append(point)
    for group_id,rows in groups.items():
        group=doc['groups'][group_id];model=doc['models'][group_id];bad=[p for p in per_group[group_id] if p['any_single_prediction_impossible_at_25_percent']]
        records.append(dict(group_id=group_id,family=group['family'],semantic_phase=group['semantic_phase'],
            kernel_name=group['kernel_name'],launch_settings=group['launch_settings'],
            observation_count=len(rows),fit_accepted=model['fit_accepted'],
            max_relative_training_error=model.get('max_relative_training_error'),
            impossible_exact_points=len(bad),observations_at_impossible_points=sum(p['observation_count'] for p in bad),
            feature_points=per_group[group_id]))
    records.sort(key=lambda r:r['observations_at_impossible_points'],reverse=True)
    save(output/'groups.json',records)
    result=dict(schema='gsdeploy.launch_geometry_pooling_impossibility_diagnostic.v4',fit=source,
        exact_point_impossibility_counts=dict(impossible),family_impossible_observations=dict(family),
        phase_impossible_observations=dict(phase),groups_with_impossible_points=sum(bool(r['impossible_exact_points']) for r in records),
        failed_fit_groups=sum(not r['fit_accepted'] for r in records),groups=receipt(output/'groups.json'),
        derivation='For error<=25% against every positive target y, p must lie in [0.75*y,1.25*y]. If max(y)/min(y)>5/3 for one exact feature vector, no deterministic predictor on this schema can meet the rule.',
        model_refitted=False,outliers_removed=False,residual_threshold_changed=False,
        interpretation='Identical exposed features conceal timing variation; source-derived semantic work/cache/stride/stage covariates or a justified different statistical target require fresh evidence. More regression flexibility alone cannot solve these points.',
        GPU_workload_executed=False,heldout_timings_consumed=False,implementation_sources=[receipt(__file__)])
    save(output/'report.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=diagnose(a.fit,a.output);print(json.dumps(r))


if __name__=='__main__':main()
