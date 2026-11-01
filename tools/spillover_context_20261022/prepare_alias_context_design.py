"""Freeze alias covariates and partitions before fitting; development inputs only."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'artifacts/static-spillover-study-20261022'
FAMILIES=['generalist_1','generalist_2','generalist_3','text','image','ocr','asr']
FEATURES=['alpha_'+m for m in FAMILIES]+['A_self','A_other','M_resident','H_self','centered_log_b_x_A_other',
    'alias_planned_baseline_work_over_horizon','same_family_partner_planned_baseline_work_over_horizon']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def deployment_key(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def rank(rows,columns):
    a=np.asarray([[r[i] for i in columns] for r in rows],dtype=float)
    if not len(a):return dict(rank=0,columns=len(columns),rows=0)
    scales=np.linalg.norm(a,axis=0);s=np.linalg.svd(a/np.where(scales>0,scales,1),compute_uv=False);n=int(np.linalg.matrix_rank(a))
    return dict(rank=n,columns=len(columns),rows=len(a),scaled_singular_values=s.tolist(),
        scaled_condition_number=float(s[0]/s[-1]) if n==len(columns) else None,
        zero_columns=[columns[i] for i,v in enumerate(scales) if v==0],rank_not_independent_run_count=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--inventory-manifest-sha256',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False);inputs={}
    def read(path,role):
        path=Path(path).resolve();inputs[str(path)]=dict(sha256=sha(path),role=role);return json.loads(path.read_text())
    assert sha(args.inventory/'manifest.json')==args.inventory_manifest_sha256
    manifest=read(args.inventory/'manifest.json','pinned_authorized_development_inventory_manifest')
    for name,expected in manifest['files'].items():assert sha(args.inventory/name)==expected
    # This design is written before reconstructing covariates or inspecting outcomes.
    design=dict(schema='gsdeploy.alias_context_design_contract.v1',frozen_utc=datetime.now(timezone.utc).isoformat(),
        inventory=dict(path=str(args.inventory.resolve()),manifest_sha256=args.inventory_manifest_sha256),
        source_contract='All v5 19 valid execution runs including fine107 remain development; all failed/resource-invalid records retained as ineligible. Never read108 or strong-control result raws.',
        old_forecast_and_failure_artifacts='Preserved unchanged; seed108 inspired this feature family, so any future108 evaluation is post-hoc development only.',
        new_features=FEATURES[12:],
        definitions=dict(alias_planned_baseline_work_over_horizon='Sum original per-input same-family/task/n median baseline b over exact pre-execution planned alias cohort /1000/horizon.',
            same_family_partner_planned_baseline_work_over_horizon='Sum corresponding planned baseline work of all other aliases of same family /1000/horizon. Already uses n-state b; no extra n multiplier.'),
        diagnostic_covariates_only=['Per-alias planned b mean/CV','Per-alias planned input-frequency entropy','Per-alias planned task entropy'],
        forbidden_feature_sources=['Actual service','Handler/completion/queue/overlap timing','Generated tokens','Realized energy/memory'],
        existing_model_qualification='Per-input b and exact aliasFIFO already partly encode input assignment; new covariates model alias/partner conditional service residual.',
        aggregation='Mean log residual per run/family/task/n/alias/input cell; each run equal; each path equal within run; each unique input equal within path; split one input weight equally over aliases where observed. Repetitions never inflate run weight.',
        target='log(service_ms/b), positive geometric/median scale by exp; no arithmetic-mean retransformation',
        predeclared_candidates=[dict(name='baseline_offset',extra=[]),dict(name='family_only',extra=[]),
            dict(name='family_self_other',extra=[]),dict(name='base_plus_alias',extra=[12]),
            dict(name='base_plus_partner',extra=[13]),dict(name='base_plus_alias_partner',extra=[12,13])],
        fitted_context_columns=[7,8],diagnostic_only_context_columns=[9,10,11],
        pressure_slope_constraint='All included pressure slopes >=0; no ridge fallback for deficient folds.',
        partitions=['Leave one whole execution run out','Leave one whole deployment (sorted family:replica counts) out'],
        selection='Compare common eligible fold coverage, macro held-group logRMSE and arithmetic totals. Prefer simpler candidate within one descriptive run SE; deployment holdout must improve too before recommending added complexity. This is development only.',
        rank_policy='Record scaled condition and singular values before outcomes; rank-deficient fold predictions UNKNOWN. Marginal support is not a proven applicability domain.',
        missing_policy='Missing any baseline covariate => UNKNOWN; preserve failed/resource-invalid rows without treating them as feasible or using completed subset as evidence of feasibility.',
        GPU_started=False,predictor_activated=False,activity_pair_status='PENDING_DECISION_NOT_QUEUED')
    save(out/'design-contract.json',design);contract_sha=sha(out/'design-contract.json')
    records=read(args.inventory/'requests.json','authorized_development_request_inventory');runs=read(args.inventory/'source-ledger.json','explicit_authorized_terminal_raw_allowlist')
    assert len(runs)==25 and sum(r['run_valid_for_fit'] for r in runs)==19
    assert not any(r['seed'] in [61,62,71,72,73,81,82,108] for r in runs)
    config=read(STUDY/'text185-fine-development-v1/system-config.json','frozen_baseline_config');c=config['catalog']
    baselines={}
    for family,tasks in c['task_replica_profiles'].items():
        for task,states in tasks.items():
            for n,state in states.items():
                grouped=defaultdict(list)
                for filename in state['raw_service_samples']:
                    path=Path(filename).resolve();assert path.is_relative_to(ROOT/'artifacts/retained-replica-profiles-20261021')
                    inputs[str(path)]=dict(sha256=sha(path),role='original_closed_loop_baseline_only')
                    for line in path.read_text().splitlines():
                        r=json.loads(line)
                        if r.get('success') is True and isinstance(r.get('service_ms'),(float,int)) and math.isfinite(r['service_ms']) and r['service_ms']>0:grouped[r['sample_id']].append(r['service_ms'])
                for sid,values in grouped.items():baselines[family,task,int(n),sid]=statistics.median(values)
    source=ROOT/'tools/spillover_capacity_20261022/exposure.py';spec=importlib.util.spec_from_file_location('_alias_design_frozen',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);api=module.frozen_api()
    inputs[str(source)]=dict(sha256=sha(source),role='isolated_cpu_loader')
    for rel in ['gsdeploy/trace.py','gsdeploy/routing.py','gsdeploy/runtime/workload.py']:
        path=module.SOURCE/rel;inputs[str(path)]=dict(sha256=sha(path),role='frozen_cpu_trace_sample_route_source')
    pool_path=ROOT/'artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl'
    inputs[str(pool_path)]=dict(sha256=sha(pool_path),role='frozen_route_independent_sample_pool')
    assert sha(pool_path)==c['quality']['identity']['input_pool_sha256']
    planned={};alias_context=[];run_metadata=[]
    for run in runs:
        path=Path(run['path']).resolve();assert path.name=='run.json' and sha(path)==run['sha256']
        assert not any(any(v in part.lower() for v in ['validation','confirmation','holdout']) for part in path.parts)
        raw=read(path,'explicit_terminal_development_raw_for_plan_identity_only')
        assert raw['status'] in ['completed','failed','aborted'] and raw['transport_config']['random_seed']==run['seed']
        x=run['deployment'];routes=run['routing'];rates=run['workload'];horizon=run['horizon_seconds']
        trace=api['trace'].make_trace(rates,horizon,run['seed'],raw['transport_config']['arrival_process'])
        assert api['trace'].trace_digest(trace)==raw['measurement_window']['trace_sha256']
        aliases,expanded=api['trace'].expand_routes(x,routes);pool=api['workload'].SamplePool(pool_path,list(rates),run['seed']);router=api['routing'].Router(expanded)
        scheduled=[];by_alias=defaultdict(list)
        for index,(arrival,task) in enumerate(trace):
            item=pool.next(task);alias=router.choose(task);family=aliases[alias];sid=item['pool_id'];b=baselines.get((family,task,x[family],sid))
            row=dict(index=index,arrival_seconds=arrival,task=task,alias=alias,family=family,n=x[family],sample_id=sid,baseline_ms=b)
            scheduled.append(row);by_alias[alias].append(row)
        contexts={}
        for alias,family in aliases.items():
            rows=by_alias[alias];partner=[r for a,rr in by_alias.items() if a!=alias and aliases[a]==family for r in rr]
            unknown=[r for r in rows+partner if r['baseline_ms'] is None]
            values=[r['baseline_ms'] for r in rows if r['baseline_ms'] is not None];counts=Counter((r['task'],r['sample_id']) for r in rows);tasks=Counter(r['task'] for r in rows)
            context=dict(run_id=run['run_id'],alias=alias,family=family,n=x[family],planned_requests=len(rows),
                planned_distinct_inputs=len(counts),covariate_status='UNKNOWN_MISSING_BASELINE' if unknown else 'PLAN_KNOWN',
                missing_baseline_inputs=[dict(family=r['family'],task=r['task'],n=r['n'],sample_id=r['sample_id']) for r in unknown],
                alias_planned_baseline_work_over_horizon=sum(values)/1000/horizon if not unknown else None,
                same_family_partner_planned_baseline_work_over_horizon=sum(r['baseline_ms'] for r in partner)/1000/horizon if not unknown else None,
                planned_b_mean_ms=statistics.mean(values) if values else None,
                planned_b_cv=statistics.pstdev(values)/statistics.mean(values) if values else None,
                input_frequency_entropy=-sum(v/len(rows)*math.log(v/len(rows)) for v in counts.values()) if rows else 0.,
                task_frequency_entropy=-sum(v/len(rows)*math.log(v/len(rows)) for v in tasks.values()) if rows else 0.,
                raw_status=run['status'],run_valid_for_fit=run['run_valid_for_fit'])
            contexts[alias]=context;alias_context.append(context)
        lookup={r['arrival_index']:r for r in raw['request_records']}
        for index,rec in lookup.items():
            row=scheduled[index];assert (rec['task'],rec['model'],rec['pool_id'])==(row['task'],row['alias'],row['sample_id'])
            assert abs(rec['scheduled_offset_seconds']-row['arrival_seconds'])<1e-8
            planned[run['run_id'],rec['request_id']]=dict(row=row,context=contexts[row['alias']],deployment_key=deployment_key(x))
        run_metadata.append(dict(run_id=run['run_id'],deployment=x,deployment_key=deployment_key(x),seed=run['seed'],status=run['status'],
            run_valid_for_fit=run['run_valid_for_fit'],planned_requests=len(scheduled),recorded_requests=len(lookup),trace_sha256=api['trace'].trace_digest(trace)))
    augmented=[]
    for r in records:
        proof=planned[r['run_id'],r['request_id']];assert proof['row']['alias']==r['alias']
        context=proof['context'];fs=r['features']+[context[FEATURES[12]],context[FEATURES[13]]] if r['features'] is not None and context['covariate_status']=='PLAN_KNOWN' else None
        augmented.append(dict(**r,deployment_key=proof['deployment_key'],alias_context_status=context['covariate_status'],alias_features=fs,
            alias_valid_for_fit=bool(r['valid_for_fit'] and fs is not None)))
    valid=[r for r in augmented if r['alias_valid_for_fit']];covered=[i for i in range(7) if any(r['features'][i] for r in valid)]
    block_groups=defaultdict(list)
    for r in valid:block_groups[r['run_id'],r['family'],r['task'],r['n'],r['alias']].append(r['alias_features'])
    blocks=[np.mean(v,axis=0).tolist() for v in block_groups.values()]
    candidate_ranks=[]
    for candidate in design['predeclared_candidates']:
        name=candidate['name'];cols=[] if name=='baseline_offset' else covered if name=='family_only' else covered+[7,8]+candidate['extra']
        candidate_ranks.append(dict(name=name,columns=cols,features=[FEATURES[i] for i in cols],run_alias_path_rank=rank(blocks,cols) if cols else None))
    correlation=np.corrcoef(np.asarray(blocks)[:,[7,8,12,13]].T)
    result=dict(schema='gsdeploy.alias_context_design_inventory.v1',design_contract_sha256=contract_sha,features=FEATURES,
        raw_runs=len(runs),qualified_runs=sum(r['run_valid_for_fit'] for r in runs),requests=len(augmented),strict_alias_requests=len(valid),
        missing_covariate_requests=sum(r['alias_features'] is None for r in augmented),run_alias_path_blocks=len(blocks),candidate_ranks=candidate_ranks,
        covariate_correlations=dict(features=[FEATURES[i] for i in [7,8,12,13]],matrix=correlation.tolist()),
        all_failed_and_invalid_preserved=True,failed_or_invalid_fit_records=sum(r['alias_valid_for_fit'] for r in augmented if not next(x for x in runs if x['run_id']==r['run_id'])['run_valid_for_fit']),
        fitted=False,predictor_activated=False,GPU_started=False,seed108_results_read=False,activity_pair_status='PENDING_DECISION_NOT_QUEUED')
    save(out/'report.json',result);save(out/'requests-with-alias-plan-features.json',augmented);save(out/'alias-plan-context.json',alias_context);save(out/'run-groups.json',run_metadata)
    save(out/'opened-inputs.json',inputs)
    for path,receipt in inputs.items():assert sha(path)==receipt['sha256']
    save(out/'manifest.json',dict(script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),inputs=inputs,files={p.name:sha(p) for p in out.iterdir() if p.is_file()},all_inputs_unchanged=True))
    print(json.dumps(result))

if __name__=='__main__':main()
