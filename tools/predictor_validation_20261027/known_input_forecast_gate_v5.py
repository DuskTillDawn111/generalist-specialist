"""Execute and seal new known-input conditional hypotheses before collection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from gsdeploy.forecast_errors import UnknownEvidence
from .known_input_gate_v5 import _verify_protocol,receipt,seal_predictions
from .forecast_gate_v3 import execute_bundle,prediction_source_paths,evidence_closure
from .forecast_gate_v2 import _read,_write,_public_cohort,_to_cell

def run_and_seal(protocol_path,jobs,output_dir,*,allow_trained_hypothesis=False,allow_cpu_fixture=False):
    protocol=_read(protocol_path);_verify_protocol(protocol)
    if any(receipt(p) not in protocol['sources'] for p in prediction_source_paths()):
        raise ValueError('Freeze complete v3 package/tools producer source closure before prediction')
    candidates={c['id']:c for c in protocol['candidates']};expected={(r,c) for r in protocol['validation_domain'] for c in candidates}
    keyed={}
    for job in jobs:
        if set(job)!={'run_id','candidate_id','public_inputs_path'}:raise ValueError('Explicit public forecast job required')
        key=job['run_id'],job['candidate_id']
        if key not in expected or key in keyed or receipt(job['public_inputs_path']) not in protocol['sources']:
            raise ValueError('Duplicate/foreign/unfrozen forecast job')
        keyed[key]=job
    if set(keyed)!=expected:raise ValueError('Complete predeclared Cartesian forecast coverage required')
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    cells=[];audits=[];objectives={};shared_cohorts={}
    for index,(key,job) in enumerate(sorted(keyed.items())):
        run,cid=key;candidate=candidates[cid];model_ref=protocol['models'][candidate['model_id']];bundle=_read(model_ref['path'])
        if bundle.get('execution_context_sha256')!=candidate['execution_context_sha256']:
            raise ValueError('Bundle/candidate original target execution mismatch')
        resolution=None;historical_ref=bundle.get('artifacts',{}).get('historical_resolution')
        if historical_ref is not None:
            if receipt(historical_ref['path'])!=historical_ref:raise ValueError('Changed historical resolution manifest')
            from .historical_evidence_v4 import resolve_closure
            resolution=resolve_closure([model_ref['path']],historical_ref['path'])
            closure=resolution['materialized_sources']
        else:closure=evidence_closure([model_ref['path']])
        bound=[*protocol['sources'],*protocol['models'].values(),*protocol['training_data']]
        if any(ref not in bound for ref in closure):raise ValueError('Freeze every model/training/source lineage receipt')
        if not bundle.get('training_indices') or any(ref not in protocol['training_data'] for ref in bundle['training_indices']):
            raise ValueError('Bundle complete training index must be protocol training data')
        if bundle['weights']!=protocol['weights']:raise ValueError('Original six utility weights differ')
        defaults=bundle['forecast_defaults']
        objective={k:defaults[k] for k in ('latency_reference_ms','memory_budget_mib','energy_reference_watts','memory_safe_factor')}
        objective['startup_amortization_seconds']=bundle['startup_amortization_seconds']
        if run in objectives and objectives[run]!=objective:raise ValueError('Paired candidates changed original five normalizers')
        objectives[run]=objective
        public,cohort=_public_cohort(_read(job['public_inputs_path']))
        actual={rid:{k:r[k] for k in ('request_id','task','pool_id')} for rid,r in cohort.items()}
        if actual!={r['request_id']:r for r in protocol['validation_domain'][run]}:
            raise ValueError('Forecast public cohort differs from full validation domain')
        shared={k:public[k] for k in ('phase_start_ns','now_ns','deadline_ns','drain_deadline_ns','future_workload')}
        shared['initial_requests']=sorted(({k:r[k] for k in ('request_id','task','pool_id','arrival_ns')}
            for rows in public['queues'].values() for r in rows),key=lambda r:r['request_id'])
        if run in shared_cohorts and shared_cohorts[run]!=shared:raise ValueError('Paired candidate arrival/workload changed')
        shared_cohorts[run]=shared
        try:
            result=execute_bundle(bundle,public,allow_trained_hypothesis=allow_trained_hypothesis,allow_cpu_fixture=allow_cpu_fixture)
            if set(result['forecast']['cohort'])!=set(cohort):raise ValueError('New producer dropped public cohort requests')
        except UnknownEvidence as exc:
            result=dict(status='UNKNOWN',reason=str(exc),GPU_started=False,full_public_cohort=actual)
        path=output/f'forecast-{index:04d}.json';_write(path,result)
        cells.append(_to_cell(result,cohort,candidate,run,protocol,bundle,public['phase_start_ns']))
        audits.append(dict(run_id=run,candidate_id=cid,model=model_ref,public_inputs=receipt(job['public_inputs_path']),
            result=receipt(path),kind=bundle['kind'],provider_kind=defaults['provider_kind'],model_source_closure=closure,
            historical_evidence_resolution=resolution))
    # Refuse source edits during the CPU producer, before making a seal.
    _verify_protocol(protocol)
    sealed=seal_predictions(protocol,cells,created_unix_ns=time.time_ns());_write(output/'predictions.json',sealed)
    report=dict(schema='gsdeploy.known_input_forecast_production_receipt.v5',protocol=receipt(protocol_path),
        predictions=receipt(output/'predictions.json'),producer=receipt(__file__),jobs=audits,
        validation_scope=protocol['validation_scope'],unseen_input_generalization_claimed=False,
        GPU_started=False,accuracy_established=False,physical_gain_certified=False,
        allow_trained_hypothesis=allow_trained_hypothesis,allow_cpu_fixture=allow_cpu_fixture)
    _write(output/'production-receipt.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--jobs',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--allow-trained-hypothesis',action='store_true');parser.add_argument('--allow-cpu-fixture',action='store_true')
    args=parser.parse_args();result=run_and_seal(args.protocol,_read(args.jobs),args.output,
        allow_trained_hypothesis=args.allow_trained_hypothesis,allow_cpu_fixture=args.allow_cpu_fixture)
    print(json.dumps(result['predictions']))


if __name__=='__main__':main()
