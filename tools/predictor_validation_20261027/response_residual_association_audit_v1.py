"""Descriptive residual association; job/context identifiers never fit inputs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from . import macro_joint_response_fit_v1 as base
from .implementation_source_closure_v27 import strict_references


def association(rows,field,*,categorical):
    y=np.asarray([r['log_reference_work_residual'] for r in rows]);result=dict(rows=len(rows))
    if categorical:
        groups=defaultdict(list)
        for row in rows:groups[json.dumps(row[field],sort_keys=True)].append(row)
        total=float(np.sum((y-y.mean())**2))
        between=sum(len(rs)*(np.mean([r['log_reference_work_residual'] for r in rs])-y.mean())**2 for rs in groups.values())
        result.update(groups=len(groups),descriptive_eta_squared=None if total==0 else float(between/total),
            group_summaries=[dict(value=json.loads(k),rows=len(rs),
                mean_log_residual=float(np.mean([r['log_reference_work_residual'] for r in rs]))) for k,rs in groups.items()])
    else:
        x=np.asarray([r[field] for r in rows],dtype=float)
        result['Pearson_r']=None if len(rows)<2 or x.std()==0 or y.std()==0 else float(np.corrcoef(x,y)[0,1])
    return result


def run(training,output):
    ref=base.receipt(training);report=base.checked(ref);diagnostics=base.checked(report['residual_diagnostics'])
    raws={};sources=[ref,report['residual_diagnostics'],base.receipt(__file__)]
    for source in report['sources']:
        if Path(source['path']).name=='raw.json':
            raw=base.checked(source)
            if 'request_records' in raw:
                raws[source['sha256']]={r['request_id']:r for r in raw['request_records']};sources.append(source)
    records=diagnostics['records']
    for r in records:
        source=raws[r['capture']][r['request_id_for_diagnostics_only']]
        r['public_input_id']=source['pool_id'];r['diagnostic_context']=r['diagnostic_job_fields'].get('context')
        r['source_worker_start_rank']=int(r['source_worker_start_rank'])
        for phase in ('entry','prefill','decode','ocr_inference','ready_tail'):
            r['peer_'+phase+'_exposure']=r['peer_phase_exposure_ratios'].get(phase,0)
    groups=defaultdict(list)
    for r in records:groups[r['category']].append(r)
    categorical=('capture','public_input_id','source_job_id','diagnostic_context')
    continuous=('source_worker_start_rank','average_peer_count','average_same_phase_count',
        'peer_entry_exposure','peer_prefill_exposure','peer_decode_exposure','peer_ocr_inference_exposure','peer_ready_tail_exposure')
    summaries={k:dict(categorical={f:association(rs,f,categorical=True) for f in categorical},
        continuous={f:association(rs,f,categorical=False) for f in continuous}) for k,rs in groups.items()}
    result=dict(schema='gsdeploy.response_residual_association_training_audit.v1',
        status='DESCRIPTIVE_TRAINING_RESIDUAL_ASSOCIATION_ONLY',sources=strict_references([sources]),
        macro_denominator=len(records),by_own_family_phase=summaries,
        context_and_job_identifiers_used_as_prediction_features=False,causal_interference_identified=False,
        independent_samples_are_whole_captures=True,replication_units=3,
        multiple_comparisons_or_significance_claims=False,selection_from_residual_associations=False,
        validation_labels_read=False,GPU_started=False,main_queue_provider_eligible=False,final_gain_eligible=False,
        interpretation='Associations are descriptive; category baselines use exact public input/shape nuisance means. Context/job/order correlations may reflect unmeasured warmup, scheduling or response noise and cannot identify physical cause.')
    for source in result['sources']:base.require(base.receipt(source['path'])==source,'Residual audit source changed')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return base.receipt(output/'report.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(run(a.training,a.output)))
