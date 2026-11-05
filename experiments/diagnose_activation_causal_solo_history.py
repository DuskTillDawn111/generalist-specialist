"""Training cross-validation of causal history descriptors; no provider or gate.

All features precede the request's native call. Test-run durations never supply
profile intercepts or feature scaling. This cannot qualify concurrent FIFO.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.activation_semantic_bands import band_layout
from experiments.fit_activation_native import load_bundle
from gsdeploy.forecasting.native_training import stage_layout,validate_capture

FEATURES=['log_owner_ordinal','log_owner_idle_ms','log_board_idle_ms','seen_input',
          'previous_owner_input_log_size_ratio','current_to_max_seen_log_size_ratio',
          'previous_global_text','previous_global_image','previous_global_ocr','previous_global_asr']


def observations(directory,pool):
    bundle=load_bundle(directory.resolve());validate_capture(bundle)
    rows=bundle['raw']['request_records'];prefix=bundle['raw']['priming_request_records']
    owners={};last_global=None;result=[]
    for row in prefix+rows:
        family=bundle['raw']['resident_aliases'][row['model']]
        native=[s for s in stage_layout(row) if s['kind']=='gpu']
        start,end=min(s['start_ns'] for s in native),max(s['end_ns'] for s in native)
        item=pool[row['pool_id']];sample=item['sample']
        size=len(sample['context'])+len(sample['question']) if family=='text' else Path(sample['input']).stat().st_size
        previous=owners.get(row['model'])
        priming=row in prefix
        if previous is not None and last_global is not None:
            if start<last_global['end']:
                raise ValueError('History baseline diagnosis requires genuinely serial native calls')
            features=[np.log1p(previous['ordinal']),np.log1p((start-previous['end'])/1e6),
                np.log1p((start-last_global['end'])/1e6),float(row['pool_id'] in previous['seen']),
                np.log(previous['size']/size),np.log(size/previous['max_size']),
                *[float(last_global['family']==f) for f in ('text','image','ocr','asr')]]
            if not priming:
                for span in band_layout(row,family):
                    if span['kind']=='gpu':
                        result.append(dict(key=(family,row['task'],row['pool_id'],span['id']),
                            family=family,band=span['native_band_role'],features=features,
                            duration_ms=(span['end_ns']-span['start_ns'])/1e6,
                            run_id=bundle['raw']['run_id'],request_id=row['request_id']))
        seen=set() if previous is None else set(previous['seen'])
        seen.add(row['pool_id'])
        owners[row['model']]=dict(end=end,size=size,max_size=max(size,previous['max_size'] if previous else size),
            ordinal=0 if previous is None or priming else previous['ordinal']+1,seen=seen)
        last_global=dict(end=end,family=family)
    return result,bundle['source']


def metrics(values):
    values=np.asarray(values)
    return dict(n=len(values),mean=float(np.mean(values)),p95=float(np.quantile(values,.95)),maximum=float(np.max(values)))


def compare(training,testing):
    groups=defaultdict(list)
    for row in training:groups[row['key']].append(row)
    keys=sorted(groups);lookup={key:i for i,key in enumerate(keys)}
    supported=[r for r in testing if r['key'] in lookup]
    if len(supported)!=len(testing):raise ValueError('Held-out training run contains an untrained input/band')
    raw=np.array([r['features'] for r in training]);mean=raw.mean(axis=0);scale=raw.std(axis=0);scale[scale<1e-8]=1.
    target=np.log([r['duration_ms'] for r in training])
    features=(raw-mean)/scale
    matrix=np.zeros((len(training),len(keys)+len(FEATURES)))
    for i,row in enumerate(training):matrix[i,lookup[row['key']]]=1.
    matrix[:,len(keys):]=features
    # Explicit fixed diagnostic penalty; not tuned to test-run errors.
    penalty=np.zeros((len(FEATURES),matrix.shape[1]));penalty[:,len(keys):]=np.eye(len(FEATURES))*np.sqrt(.1)
    beta=np.linalg.lstsq(np.vstack([matrix,penalty]),np.r_[target,np.zeros(len(FEATURES))],rcond=None)[0]
    baseline={key:float(np.mean([r['duration_ms'] for r in rs])) for key,rs in groups.items()}
    errors=defaultdict(lambda:dict(baseline=[],history=[]));predictions=[]
    for row in testing:
        predicted=float(np.exp(beta[lookup[row['key']]]+((np.array(row['features'])-mean)/scale)@beta[len(keys):]))
        observed=row['duration_ms'];base=baseline[row['key']]
        a,b=abs(observed/base-1),abs(observed/predicted-1)
        path=row['family']+'/'+row['band'];errors[path]['baseline'].append(a);errors[path]['history'].append(b)
        predictions.append(dict(run_id=row['run_id'],request_id=row['request_id'],key=row['key'],
            observed_ms=observed,baseline_ms=base,causal_history_predicted_ms=predicted,
            baseline_relative_work_error=a,history_relative_work_error=b))
    return dict(raw_design_rank=int(np.linalg.matrix_rank(matrix)),parameters=matrix.shape[1],
        penalty=.1,feature_coefficients=beta[len(keys):].tolist(),
        errors={path:{name:metrics(values) for name,values in v.items()} for path,v in errors.items()},predictions=predictions)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round1',type=Path,action='append',required=True)
    p.add_argument('--round2',type=Path,action='append',required=True)
    p.add_argument('--pool',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();pool={r['pool_id']:r for r in (json.loads(line) for line in args.pool.read_text().splitlines())}
    cohorts=[];sources=[]
    for directories in [args.round1,args.round2]:
        rows=[]
        for directory in directories:
            observed,source=observations(directory,pool);rows.extend(observed);sources.append(source)
        cohorts.append(rows)
    # Fit each family separately; both constituent bands share a family-level
    # history descriptor here, an explicit diagnostic simplification.
    comparisons=[]
    for train_index,test_index in [(0,1),(1,0)]:
        for family in sorted({r['family'] for r in cohorts[train_index]}):
            train=[r for r in cohorts[train_index] if r['family']==family]
            test=[r for r in cohorts[test_index] if r['family']==family]
            comparisons.append(dict(training_round=train_index+1,test_round=test_index+1,family=family,
                                    **compare(train,test)))
    report=dict(schema='gsdeploy.causal_solo_history_cross_validation.v1',features=FEATURES,
        source_receipts=sources,comparisons=comparisons,
        prediction_features_use_future_outputs_or_forward_counts=False,
        scope='training-only cross-run diagnosis; no concurrent progress fit or FIFO implementation',
        qualification_criteria_changed=False,physical_predictor_qualified=False,formal_utility_qualified=False,
        limitations=['actual previous completion clocks used as causal training history; autonomous forecasts must predict and propagate them',
                     'regularization can return a diagnostic estimate at deficient rank; it does not grant model identifiability',
                     'same family history coefficients shared across constituent bands',
                     'input size uses static characters/media bytes, not inferred semantic difficulty'],
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(args.output),comparisons=[{k:v for k,v in r.items() if k!='predictions'} for r in comparisons])))


if __name__=='__main__':main()
