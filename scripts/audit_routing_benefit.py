"""Verify recorded contracts and emit a portable evidence/source manifest."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gsdeploy import SystemConfig
from gsdeploy.continuous.contracts import digest
from experiments.joint_scalability import source_manifest, subset_config, ROOT, save
from experiments.routing_benefit_diagnosis import reconcile_events


def audit(source,report):
    assert json.loads((source/'source-manifest.json').read_text())==source_manifest()
    observations=json.loads((source/'observations.json').read_text())
    configs={seed:SystemConfig.load(source/f'trace{seed}-config.json') for seed in (42,43,45)}
    identities={r['identity'] for r in observations}
    assert len(identities)==sum(json.loads((source/'prediction-budget.json').read_text())['full_predict_scenarios_calls'].values())
    valid=0;infeasible=0
    for key in sorted(identities):
        record=json.loads((source/'points'/f'{key}.json').read_text());c=configs[record['trace_seed']]
        assert key==digest(dict(config=c.as_dict(),deployment=record['deployment'],routing=record['routing']))
        if record['ledger']:reconcile_events(c,record['prediction'],record['ledger'])
        assert set(record['routing'])==set(c.workload)
        for task,row in record['routing'].items():
            assert abs(sum(row.values())-c.workload[task])<1e-9
            assert all(q>0 and m in record['deployment'] and task in c.catalog['models'][m]['supported_tasks'] for m,q in row.items())
        if record['state']=='FEASIBLE':
            valid+=1;p=record['prediction']
            assert abs(p['U']-sum(p[k+'_contribution'] for k in 'QTLPME'))<1e-9
        elif record['state']=='INFEASIBLE':infeasible+=1
    # Fill in explicit executed configuration identities. These metadata
    # fields do not alter any measured result or elapsed time.
    base=SystemConfig.load(source/'frozen-source-config.json')
    comparison=[];predictions=0;contracts={}
    for path in sorted((source/'comparison').glob('*-K*-s*.json')):
        row=json.loads(path.read_text());name=row['case'];K=row['K']
        saved=SystemConfig.load(source/'comparison'/f'{name}-config.json')
        states=json.loads((source/'comparison'/f'{name}-states.json').read_text())
        config=replace(subset_config(base,list(saved.catalog['models']),K,8.,seed=42),milp_max_iterations=200)
        if row['algorithm']!='refine_de':
            assert row['raw']['input_sha256']==digest(dict(config=config.as_dict(),states=states))
        row['config_overrides']=dict(route_grid=K,milp_max_iterations=200)
        row['config_sha256']=digest(config.as_dict())
        save(path,row)
        cfgfile=source/'comparison'/f'{name}-K{K}-executed-config.json'
        if not cfgfile.exists():save(cfgfile,config.as_dict())
        assert row['predictions']<=200
        predictions+=row['predictions']
        contracts.setdefault(name,[]).append(dict(workload=config.workload,weights=config.weights,
            latency_reference_ms=config.latency_reference_ms,gpu_memory_mib=config.gpu_memory_mib,
            catalog=config.catalog,states=states,horizon=config.horizon_seconds,trace_seed=config.random_seed,
            arrival_process=config.arrival_process,drain=config.drain_seconds,policy=config.search_policy))
        comparison.append({k:row[k] for k in row if k not in {'raw','fifo','winner'}}|dict(
            deployment=(row['winner'] or {}).get('deployment'),routing=(row['winner'] or {}).get('routing')))
    assert len(comparison)==28
    assert all(len({digest(c) for c in cs})==1 for cs in contracts.values())
    save(source/'comparison.json',comparison)
    slice_predictions=0;events=0
    for seed in (11,29,47):
        row=json.loads((source/'text-slice'/f'de-s{seed}.json').read_text())
        slice_predictions+=row['budget']['full_prediction_calls']
        assert row['budget']['milp_solve_calls']==0
        for path in sorted((source/'text-slice/events').glob(f's{seed}-*.json')):
            data=json.loads(path.read_text());reconcile_events(configs[42],data['prediction'],data['ledger']);events+=1
    assert events==slice_predictions
    result=dict(unique_diagnostic_predictions=len(identities),diagnostic_observations=len(observations),
        valid_diagnostic_inputs=valid,infeasible_diagnostic_inputs=infeasible,
        unknown_diagnostic_inputs=len(identities)-valid-infeasible,
        comparison_optimizations=len(comparison),comparison_predictions=predictions,
        slice_DE_optimizations=3,slice_DE_predictions=slice_predictions,
        full_prediction_calls_total=len(identities)+predictions+slice_predictions,
        event_ledgers_reconciled=len(identities)+events,
        original_sources_unchanged=True,all_comparison_contracts_equal_within_case=True,
        grid_certificates_finite_domain_only=True,continuous_global_optimality_claimed=False,
        one_dimensional_scan_internal_improvement_cases=1)
    save(source/'audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--report',required=True)
    args=p.parse_args();audit(Path(args.source),Path(args.report))
