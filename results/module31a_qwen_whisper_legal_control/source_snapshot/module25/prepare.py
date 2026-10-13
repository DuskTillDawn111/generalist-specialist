"""Freeze one model specification and whole-session/combination validation."""
from .common import *
from module24.calibration import extract,historical_catalog,DIRECTORIES as RAW_DIRECTORIES

def main():
    assert not (OUT/'research_protocol.json').exists(),'Do not overwrite completed research'
    save('historical_metadata_before.json',protected_metadata())
    inventory=[];read_counts={}
    for module,dirname in DIRECTORIES.items():
        count=0
        for p in sorted((ROOT/'results'/dirname).rglob('*')):
            if not p.is_file():continue
            if p.suffix=='.json':read(p)
            elif p.suffix=='.csv':list(csv.DictReader(p.open()))
            else:p.read_bytes()
            inventory.append(ref(p));count+=1
        read_counts[str(module)]=count
    definition,refs=utility_inputs()
    held_pool=checkpoint_key(read(ROOT/'results/module24b_high_validation/frozen_high_manifest.json')['deployment'])
    save('research_protocol.json',dict(frozen_before_fitting=True,CPU_only_until_predictive_gate=True,
        immutable_history=True,utility_definition=ref(ROOT/'results/module20_utility_optimizer/utility_definition.json'),
        reference=ref(ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json'),weights=definition['weights'],refs_ms=refs,
        source_output_counts=read_counts,
        model='Separate checkpoint/task/replica-count ridge regressions for log mean service, 33 log service quantiles and transport residual; group-balanced; route-sensitive own/foreign utilization and idle-gap features',
        fixed_hyperparameters=dict(ridge_penalty=1.,minimum_path_requests=10,minimum_regression_groups=4,
            quantiles=list(map(float,np.linspace(.015,.985,33))),simulation_seeds=list(range(25001,25017)),
            simulation_seconds=120.,uncertainty_alpha=.05,minimum_positive_gain=.01),
        validation=dict(primary='Leave entire checkpoint combination out across every workload/count/module; both comparison arms excluded; all sessions in each combo excluded',
            additional='Final frozen model holds out the entire Module24A/24B checkpoint pool across every count/workload, including preflight',
            no_request_or_repetition_split=True,all_runs_from_same_deployment_workload_together=True,
            repeated_pairs_collapsed_for_independent_group_metrics=True,
            inner_calibration='Deterministic SHA256 combo/workload group partition, 20 percent of each outer training fold; distinct from test',
            historical_known_outcomes='Retrospective grouped out-of-sample validation, not blind prospective trials; attribution reads 24B but its data never fits final model',
            primary_positive_examples='Same-residency, same-nontarget-routing, matched-input GS vs target S-only only',
            original_illegal_comparators='Retain Module19 legacy operationally-illegal diagnostics separately; do not inflate primary validation',
            development_data_can_train_other_folds='Only while its entire tested checkpoint combination is excluded from that fold'),
        final_holdout_checkpoint_pool=held_pool,
        uncertainty='Split-conformal absolute delta error at whole contrast-group level only; finite 95% bound requires enough distinct calibration groups. Missing support/insufficient calibration gives unbounded interval and abstention; simulation variability is not model-error confidence.',
        predictive_gate=dict(minimum_primary_heldout_contrast_groups=3,minimum_checkpoint_combinations=2,
            minimum_positive_primary_contrast_groups=2,minimum_negative_primary_contrast_groups=2,
            minimum_point_direction_accuracy=.8,minimum_interval_coverage=.8,
            zero_false_confident_GS_choices=True,zero_false_confident_stable_for_unstable=True,
            finite_valid_calibration_required=True,all_conditions_required=True),
        collection_policy='If existing matched contrasts are insufficient, a separate preregistered development/test acquisition may be considered. If evaluated out-of-sample signs remain unreliable, stop per user instruction; no outcome-driven GPU extension.',
        high_first_policy=dict(execution_requires_predictive_gate=True,catalog=ref(SOURCE),
            all_existing_admitted_models_and_measured_counts_compete=True,budget_mib=BUDGET,
            target='asr',background=dict(text=4.,image=2.,ocr=5.,lid=.03,visual_qa=.2),
            HIGH_target_qps=[1.52],lower_target_qps=[.02,.80],
            official_optimizer='module20.optimizer.Generator plus complete legal-placement/free-route competition with new scorer and original components',
            minimum_credible_lower_delta_U=.01,no_forced_phase_structure=True,
            no_new_QPS_weight_or_hand_route_success=True,all_failed_candidates_retained=True),
        formal_GPU_policy='Only after predictive gate and credible autonomous HIGH split: three preregistered matched independent pairs, excluded from every model fit; never use exploration as validation'))
    sources=[]
    for module,dirname in RAW_DIRECTORIES.items():
        for p in sorted((ROOT/'results'/dirname/'raw_runs').glob('*.json')):sources.append((module,p))
    for module in ['24A','24B']:
        for p in sorted((ROOT/'results'/DIRECTORIES[module]/'raw_runs').glob('*.json')):sources.append((module,p))
    observations=[];hashes=set();rows=[]
    for module,path in sources:
        raw=read(path);rref=ref(path)
        if rref['sha256'] in hashes:continue
        hashes.add(rref['sha256']);inventory.append(rref)
        d=read(ROOT/'results/module24_utility_calibration/selected_pool_catalog.json') if module in ['24A','24B'] else historical_catalog(module,raw)
        ob=extract(raw,module,path,refs,definition['weights'],d=d,split='UNASSIGNED_WHOLE_SESSION')
        ob['checkpoint_group']=checkpoint_key(ob['deployment']);ob['deployment_workload_group']=group_key(ob)
        ob['is_preflight']=module=='24B' and raw['slot']['arm']=='preflight'
        ob['final_holdout']=ob['checkpoint_group']==held_pool
        ob['service_rows']=[]
        phase=raw.get('phases',{}).get('measurement')
        if phase:
            for pr in ob['paths']:
                requests=[r for r in phase['request_records'] if r['status']=='completed' and r['task']==pr['task'] and raw['replica_to_family'][r['model']]==pr['model']]
                values=[r['worker_latency_ms'] for r in requests]
                residual=[r['e2e_latency_ms']-r['worker_latency_ms']-r['worker_queue_wait_ms'] for r in requests]
                # Only whole-run summaries train; individual requests never form train/test rows.
                row=dict(uid=ob['uid'],module=module,model=pr['model'],task=pr['task'],replicas=pr['replicas'],
                    deployment=ob['deployment'],rates=ob['rates'],routing=ob['routing'],hardware_uuid=ob['hardware_uuid'],
                    arrival_process=ob['arrival_process'],checkpoint_group=ob['checkpoint_group'],
                    deployment_workload_group=ob['deployment_workload_group'],configuration=identity(ob['routing']),
                    count=len(values),mean_ms=float(np.mean(values)),quantiles_ms=list(map(float,np.quantile(values,np.linspace(.015,.985,33)))),
                    transport_mean_ms=float(np.mean(residual)),standalone_service_ms=pr['standalone_s0_ms'],
                    g_isolated=d['capacity_factors'][pr['model']][str(pr['replicas'])],
                    catalog_services={m:md['service_time_ms'] for m,md in d['models'].items()},
                    catalog_roles={m:md['role'] for m,md in d['models'].items()},catalog_capacity=d['capacity_factors'],
                    final_holdout=ob['final_holdout'],is_preflight=ob['is_preflight'],
                    usable_for_service_fit=bool(raw['valid_session'] and not ob['censored'] and len(values)>=10 and not ob['is_preflight']))
                rows.append(row);ob['service_rows'].append(row)
        observations.append(ob)
        print('READ',module,path.name,'U',ob['U'],'unstable',ob['hard_unstable'],flush=True)
    inventory.extend(ref(p) for p in [ROOT/'module20/optimizer.py',ROOT/'module24/calibration.py',ROOT/'module24/replay.py',
        ROOT/'deployment_core/routing/e2e_surrogate.py',ROOT/'joint_deploy/measurement/router.py',SOURCE])
    save('source_inventory.json',list({r['path']:r for r in inventory}.values()))
    save('observations.json',observations);save('service_training_rows.json',rows)
    save('data_availability.json',dict(raw_sessions=len(observations),module_counts=dict(Counter(str(o['module']) for o in observations)),
        duplicate_raw_sha256_removed=len(sources)-len(observations),Module20A_GPU_sessions=0,Module21D_new_GPU_sessions=0,Module23_GPU_sessions=0,
        final_holdout_sessions=sum(o['final_holdout'] for o in observations),whole_group_keys_include_replica_counts=True,
        no_reanalysis_double_count=True))

if __name__=='__main__':main()
