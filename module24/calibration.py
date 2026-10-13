"""Group-balanced, explainable path residuals; archival combination holdout.

Model choices and split are fixed in cpu_protocol.json before fitting. Entire
Module16 combinations (including Module17) never train. Module19/22 always
remain development. Requests are only used to estimate session distributions.
"""
from .common import *
from .replay import replay
from module20.optimizer import components,predict
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS
from types import SimpleNamespace
from collections import defaultdict
from scipy.stats import spearmanr

DIRECTORIES={15:'module15_three_phase',16:'module16_fullpool_gpu',17:'module17_runtime_capacity',
             18:'module18_cross_task',19:'module19_strategy_benchmark',21:'module21_positive_utility',22:'module22_mid_high_utility'}

def historical_catalog(module,raw):
    if module==15:return read(ROOT/'results/module15_three_phase/existing_empirical_catalog.json')
    if module==18:return read(ROOT/'results/module18_cross_task/controlled_common_G_pool.json')
    if module==19:
        s=raw['slot'];cid=s.get('owner_case',s['block_id'])
        cases=read(ROOT/'results/module19_strategy_benchmark/input_protocol_freeze.json')['cases']
        return read(next(c['catalog']['path'] for c in cases if c['case_id']==cid))
    if module==21:return read(ROOT/'results/module21_positive_utility/five_model_catalog.json')
    if module==22:return read(read(ROOT/'results/module22_mid_high_utility/frozen_experiment_manifest.json')['catalog']['path'])
    return read(SOURCE)

def extract(raw,module,path,refs,weights,d=None,split='DEVELOPMENT'):
    s=raw['slot'];x=s['deployment'];rates={t:q for t,q in s['lambda_rps'].items() if q>0}
    routing={t:s['routing'][t] for t in rates};d=d or historical_catalog(module,raw)
    uid=f'M{module}:{s["run_id"]}';ph=raw.get('phases',{}).get('measurement')
    row=dict(uid=uid,module=module,run_id=s['run_id'],combo=combo(x),group=group(x,rates),split=split,
             deployment=x,rates=rates,routing=routing,budget_mib=s.get('safe_memory_budget_mib',BUDGET),
             repetition=s.get('repetition'),config_id=s.get('config_id',s['run_id'].rsplit('_rep',1)[0]),
             case_id=s.get('owner_case',s.get('block_id',s.get('stage',''))),
             hardware_uuid=raw.get('baseline_GPU',{}).get('uuid'),raw=ref(path),
             valid_session=raw.get('valid_session',False),status=raw.get('status'),errors=raw.get('errors',[]),
             eligible=False,U=None,hard_unstable=False,censored=True,paths=[],worker_states={},measurement_available=bool(ph))
    if not ph:
        row['unavailable_reason']='No completed measurement phase; raw failure retained, no normal latency/U'
        return row
    rs=ph['request_records'];w=ph['window'];horizon=w.get('planned_duration_seconds',s.get('measurement_seconds',120.))
    done=[r for r in rs if r['status']=='completed'];mapping=raw['replica_to_family']
    per_task={};pathrows=[];workers=[]
    for t,q in rates.items():
        dd=[r for r in done if r['task']==t]
        if dd:per_task[t]=summarize_samples([r['e2e_latency_ms'] for r in dd])
        for m,n in x.items():
            md=[r for r in dd if mapping[r['model']]==m]
            if not md:continue
            sv=np.array([r['worker_latency_ms'] for r in md]);qm=np.array([r['worker_queue_wait_ms'] for r in md])
            em=np.array([r['e2e_latency_ms'] for r in md]);s0=d['models'][m]['service_time_ms'][t];eff=s0*n/d['capacity_factors'][m][str(n)]
            quantiles=np.quantile(sv,np.linspace(.015,.985,33));shape=quantiles/np.mean(quantiles)
            pr=dict(uid=uid,module=module,run_id=s['run_id'],combo=row['combo'],group=row['group'],split=split,
                    model=m,replicas=n,task=t,count=len(md),standalone_s0_ms=s0,predicted_count_adjusted_service_ms=eff,
                    actual_service=summarize_samples(sv),actual_queue=summarize_samples(qm),actual_e2e=summarize_samples(em),
                    service_ratio=float(np.mean(sv)/eff),service_shape=shape.tolist(),
                    hardware_uuid=row['hardware_uuid'],censored=len(md)<sum(r['task']==t and mapping[r['model']]==m for r in rs))
            pathrows.append(pr)
    for worker,ident in raw['worker_identities'].items():
        wr=[r for r in rs if r['model']==worker];m=mapping[worker]
        if not wr:continue
        tmp=dict(request_records=wr,measurement_window=w,status='completed',errors=raw.get('errors',[]),
                 drain_timeout_seconds=60,worker_identities={worker:ident})
        ev=reconstruct(tmp,{worker:SimpleNamespace(service_time_ms=d['models'][m]['service_time_ms'])},DEFAULT_THRESHOLDS)
        cl=classify(ev,DEFAULT_THRESHOLDS);workers.append(dict(worker=worker,state=cl['stability_state_v2'],
            hard_reasons=cl['hard_reasons'],end_backlog=ev['end_backlog_requests'],
            peak_backlog=ev['peak_backlog_requests'],tail_slope_rps=ev['tail_backlog_slope'],final_backlog=ev['final_backlog_after_drain']))
    hard=ph['stability']['stability_state_v2']=='unstable' or any(wr['state']=='unstable' for wr in workers)
    idle=[float(np.mean([v['owned_used_mib'] for v in raw[k]])) for k in ['postwarm_idle','postmeasurement_idle'] if raw.get(k)]
    memory=d['safe_factor']*max(idle) if len(idle)==2 else None
    peak=max((v['owned_used_mib'] for v in raw.get('telemetry',{}).get('samples',[])),default=None)
    within=sum(r['completion_time_ns']<=w['end_time_ns'] for r in done)
    full=len(done)==len(rs);coverage=set(per_task)==set(rates)
    capability=all(r['task'] in d['models'][mapping[r['model']]]['supported_tasks'] for r in rs)
    semantic_valid=not any(r.get('worker_response',{}).get('valid_closed_set_output') is False for r in done)
    memok=memory is not None and memory<=row['budget_mib']+1e-6 and (peak is None or peak<=row['budget_mib']+1e-6)
    recovery=ph['stability'].get('final_recovery_confirmed',False) if ph['stability']['stability_state_v2']=='recoverable_congested' else True
    eligible=full and coverage and capability and semantic_valid and memok and not hard and recovery and raw.get('valid_session',False) and not raw.get('errors')
    T=1-min(1.,len(done)/len(rs) if rs else 0,within/(sum(rates.values())*horizon))
    norms=components(per_task,rates,refs,T,memory/row['budget_mib'],weights) if coverage and memory is not None else dict(U=None)
    actual=np.array([r['e2e_latency_ms'] for r in done]);row.update(U=norms['U'] if eligible else None,
        diagnostic_completed_subset_U_NOT_sortable=norms['U'] if not eligible else None,eligible=bool(eligible),
        hard_unstable=bool(hard),censored=not full,per_task=per_task,paths=pathrows,workers=workers,
        worker_states={wr['worker']:wr['state'] for wr in workers},offered=len(rs),completed=len(done),within=within,
        planned_seconds=horizon,observed_seconds=w['duration_seconds'],safe_memory_mib=memory,sampled_peak_mib=peak,
        pooled_mean_ms=float(np.mean(actual)) if len(actual) else None,pooled_p95_ms=float(np.percentile(actual,95)) if len(actual) else None,
        task_weighted_mean_ms=sum(q/sum(rates.values())*per_task[t]['mean_ms'] for t,q in rates.items()) if coverage else None,
        arrival_process=raw.get('transport_config',{}).get('arrival_process','unknown'),
        cohort_signature=identity([(r['task'],r.get('pool_id'),r.get('sample_id'),round((r['scheduled_time_ns']-w['start_time_ns'])/1e9,5)) for r in rs]),
        actual_components=norms)
    old,oldpaths=replay(d,x,routing,rates,row['budget_mib'],refs,weights,horizon=horizon)
    check=predict(d,x,routing,rates,row['budget_mib'],refs,weights,horizon=horizon,phase=.37)
    assert old['eligible']==check['eligible']
    if old['U'] is not None:assert abs(old['U']-check['U'])<1e-10,(uid,old['U'],check['U'])
    row.update(prediction_A=old,prediction_A_paths=oldpaths,catalog=d)
    for pr in pathrows:
        pp=oldpaths.get(identity([pr['model'],pr['task']]))
        if pp:pr.update(predicted_queue_mean_ms=pp['queue']['mean_ms'],predicted_queue_p95_ms=pp['queue']['p95_ms'],
                        queue_mean_residual_ms=pr['actual_queue']['mean_ms']-pp['queue']['mean_ms'],
                        service_mean_residual_ms=pr['actual_service']['mean_ms']-pr['predicted_count_adjusted_service_ms'])
    return row

class ResidualModel:
    """Median log service ratios, empirical shape, and median queue offsets.

    Each combo/workload/route group has one vote, then groups have equal votes.
    Unseen combinations fall back to a model/task/count estimate with explicit
    transfer uncertainty. Unknown paths use standalone inputs with no correction.
    """
    def __init__(self):self.exact={};self.path={};self.training_groups=[];self.fit_seconds=0
    def parameter(self,x,m,n,t):
        exact=identity([combo(x),m,n,t]);path=identity([m,n,t])
        if exact in self.exact:return {**self.exact[exact],'scope':'EXACT_COMBINATION'}
        if path in self.path:return {**self.path[path],'scope':'MODEL_TASK_TRANSFER_UNVERIFIED_COMBINATION'}
        return dict(ratio=1.,shape=[1.],queue_offset_ms=0.,scope='NO_MEASURED_PATH_SUPPORT',
                    ratio_range=None,independent_groups=0,hardware_uuids=[])
    def fit(self,rows,refs,weights):
        start=time.perf_counter();training=[r for r in rows if r['split']=='DEVELOPMENT' and r['measurement_available']]
        # A full completed cohort can supply service observations even when its
        # growing queue is unstable. Such queues cannot train stationary offsets.
        prs=[p for r in training if not r['censored'] and r['valid_session'] for p in r['paths'] if p['count']>=3]
        stable_uids={r['uid'] for r in training if not r['hard_unstable'] and not r['censored']}
        def aggregate(keyfn,queue=False):
            by=defaultdict(lambda:defaultdict(list))
            for p in prs:
                gid=identity([p['group'],next(r['routing'] for r in training if r['uid']==p['uid'])])
                by[keyfn(p)][gid].append(p)
            result={}
            for key,groups in by.items():
                ratios=[float(np.median([p['service_ratio'] for p in pp])) for pp in groups.values()]
                shape=np.median([np.median([p['service_shape'] for p in pp],axis=0) for pp in groups.values()],axis=0);shape/=np.mean(shape)
                offsets=[float(np.median([p.get('corrected_queue_residual_ms',0.) for p in pp if p['uid'] in stable_uids]))
                         for pp in groups.values() if any(p['uid'] in stable_uids for p in pp)]
                result[key]=dict(ratio=float(np.exp(np.median(np.log(ratios)))),shape=shape.tolist(),
                    queue_offset_ms=float(np.median(offsets)) if queue and offsets else 0.,ratio_range=[min(ratios),max(ratios)],
                    independent_groups=len(groups),fitting_context_groups=len(groups),
                    distinct_combo_workload_groups=len({p['group'] for pp in groups.values() for p in pp}),
                    session_count=sum(map(len,groups.values())),
                    hardware_uuids=sorted({p['hardware_uuid'] for pp in groups.values() for p in pp if p['hardware_uuid']}))
            return result
        ekey=lambda p:identity([p['combo'],p['model'],p['replicas'],p['task']])
        pkey=lambda p:identity([p['model'],p['replicas'],p['task']])
        self.exact=aggregate(ekey);self.path=aggregate(pkey)
        for r in training:
            if not r.get('prediction_A'):continue
            _,ps=replay(r['catalog'],r['deployment'],r['routing'],r['rates'],r['budget_mib'],refs,weights,self,horizon=r['planned_seconds'])
            for p in r['paths']:
                pp=ps.get(identity([p['model'],p['task']]))
                if pp:p['corrected_queue_residual_ms']=p['actual_queue']['mean_ms']-pp['queue']['mean_ms']
        self.exact=aggregate(ekey,True);self.path=aggregate(pkey,True)
        self.training_groups=sorted({r['group'] for r in training});self.fit_seconds=time.perf_counter()-start
        return self
    def as_dict(self):return dict(method='Group-balanced median log service ratio + 33 empirical quantiles + path queue residual',
        exact=self.exact,path=self.path,training_groups=self.training_groups,fit_seconds=self.fit_seconds,
        universal_interference_coefficient=False,request_random_split=False,
        parameter_group_count_definition='Legacy independent_groups field counts fitting combo/workload/route contexts; paired routes are not independent ranking tests. distinct_combo_workload_groups is separately reported.',
        uncertainty='Training group min/max service ratios are empirical envelopes, not confidence bounds. New combinations, sparse paths, unmeasured paths and new physical GPUs retain uncertainty.')

def scalar_residuals(rows):
    utility=[];service=[]
    for r in rows:
        p=r.get('prediction_A',{});b=r.get('prediction_B',{})
        utility.append({k:r.get(k) for k in ['uid','module','run_id','split','combo','group','eligible','hard_unstable','censored','U','hardware_uuid','status']}|
            dict(predicted_U_A=p.get('U'),predicted_U_B=b.get('U'),
            residual_actual_minus_A=r['U']-p['U'] if r['U'] is not None and p.get('U') is not None else None,
            residual_actual_minus_B=r['U']-b['U'] if r['U'] is not None and b.get('U') is not None else None,
            actual_system_mean_ms=r.get('pooled_mean_ms') if not r['censored'] else None,
            actual_system_P95_ms=r.get('pooled_p95_ms') if not r['censored'] else None,
            predicted_system_mean_A=p.get('pooled_request_mean_ms'),predicted_system_P95_A=p.get('pooled_request_P95_ms'),
            predicted_system_mean_B=b.get('pooled_request_mean_ms'),predicted_system_P95_B=b.get('pooled_request_P95_ms'),
            raw_path=r['raw']['path'],raw_sha256=r['raw']['sha256'],unstable_U_policy='null; completed-subset diagnostics not sortable'))
        for pr in r['paths']:
            bp=r.get('prediction_B_paths',{}).get(identity([pr['model'],pr['task']]),{})
            service.append({k:pr.get(k) for k in ['uid','module','run_id','split','combo','group','model','replicas','task','count','standalone_s0_ms','predicted_count_adjusted_service_ms','service_ratio','service_mean_residual_ms','predicted_queue_mean_ms','predicted_queue_p95_ms','queue_mean_residual_ms','censored','hardware_uuid']}|
                dict(actual_service_mean_ms=pr['actual_service']['mean_ms'],actual_service_p95_ms=pr['actual_service']['p95_ms'],
                actual_queue_mean_ms=pr['actual_queue']['mean_ms'],actual_queue_p95_ms=pr['actual_queue']['p95_ms'],
                actual_e2e_mean_ms=pr['actual_e2e']['mean_ms'],actual_e2e_p95_ms=pr['actual_e2e']['p95_ms'],
                predicted_service_B_ms=bp.get('service',{}).get('mean_ms'),predicted_service_B_p95_ms=bp.get('service',{}).get('p95_ms'),
                predicted_queue_B_ms=bp.get('queue',{}).get('mean_ms'),predicted_queue_B_p95_ms=bp.get('queue',{}).get('p95_ms'),
                predicted_e2e_B_mean_ms=bp.get('e2e',{}).get('mean_ms'),predicted_e2e_B_p95_ms=bp.get('e2e',{}).get('p95_ms'),
                full_session_eligible=r['eligible'],hard_unstable=r['hard_unstable']))
    return utility,service

def pairs(rows):
    """Same workload/budget and repetition; no cross-module/card causal pairs."""
    result=[]
    by=defaultdict(list)
    for r in rows:
        by[identity([r['module'],r['rates'],r['budget_mib'],r['repetition'],r['hardware_uuid']])].append(r)
    for rs in by.values():
        for a,b in itertools.combinations(sorted(rs,key=lambda r:r['uid']),2):
            if identity([a['deployment'],a['routing']])==identity([b['deployment'],b['routing']]):continue
            if a['split']!=b['split']:continue
            actual=a['U']-b['U'] if a['U'] is not None and b['U'] is not None else None
            groupid=identity([a['module'],a['rates'],sorted([a['combo'],b['combo']]),sorted([a['config_id'],b['config_id']])])
            paired=a.get('cohort_signature')==b.get('cohort_signature') and bool(a.get('cohort_signature'))
            p=dict(contrast_group=groupid,split=a['split'],module=a['module'],a_uid=a['uid'],b_uid=b['uid'],
                   repetition=a['repetition'],same_workload=True,same_hardware=True,same_trace=paired,
                   same_residency=a['deployment']==b['deployment'],actual_delta_U=actual,eligible=actual is not None,
                   a_combo=a['combo'],b_combo=b['combo'])
            for alg in ['A','B']:
                pa=a.get('prediction_'+alg,{}).get('U');pb=b.get('prediction_'+alg,{}).get('U')
                delta=pa-pb if pa is not None and pb is not None else None
                p['predicted_delta_U_'+alg]=delta
                p['direction_correct_'+alg]=bool(delta*actual>0) if actual is not None and delta is not None else None
                p['delta_error_'+alg]=delta-actual if actual is not None and delta is not None else None
            result.append(p)
    return result

def compare(rows,pairrows):
    out=[]
    for split in ['DEVELOPMENT','RETROSPECTIVE_COMBINATION_HOLDOUT']:
        rs=[r for r in rows if r['split']==split];prs=[r for r in pairrows if r['split']==split]
        for alg in ['A','B']:
            errors=defaultdict(list);unstable=missed=abstain=0;timings=[]
            for r in rs:
                p=r.get('prediction_'+alg,{});paths=r.get('prediction_'+alg+'_paths',{})
                if r['hard_unstable']:unstable+=1;missed+=int(p.get('eligible',False))
                if r.get('prediction_'+alg+'_seconds') is not None:timings.append(r['prediction_'+alg+'_seconds'])
                if r['U'] is not None:
                    if p.get('U') is not None:errors['U'].append(p['U']-r['U'])
                    else:abstain+=1
                if not r['censored'] and not r['hard_unstable'] and r.get('pooled_mean_ms') is not None:
                    for name,pk,ak in [('system_mean_ms','pooled_request_mean_ms','pooled_mean_ms'),('system_P95_ms','pooled_request_P95_ms','pooled_p95_ms')]:
                        if p.get(pk) is not None:errors[name].append(p[pk]-r[ak])
                    for pr in r['paths']:
                        pp=paths.get(identity([pr['model'],pr['task']]))
                        if pp:
                            errors['service_mean_ms'].append(pp['service']['mean_ms']-pr['actual_service']['mean_ms'])
                            errors['service_P95_ms'].append(pp['service']['p95_ms']-pr['actual_service']['p95_ms'])
                            errors['queue_mean_ms'].append(pp['queue']['mean_ms']-pr['actual_queue']['mean_ms'])
            row=dict(split=split,algorithm=alg,session_count=len(rs),combination_count=len({r['combo'] for r in rs}),
                     workload_groups=len({r['group'] for r in rs}),normal_U_targets=sum(r['U'] is not None for r in rs),
                     U_abstentions=abstain,actual_hard_unstable=unstable,false_predicted_stable=missed,
                     normal_latency_excludes_unstable_and_censored=True,
                     mean_prediction_seconds=float(np.mean(timings)) if timings else None,
                     p95_prediction_seconds=float(np.percentile(timings,95)) if timings else None)
            for name,es in errors.items():row.update({name+'_n':len(es),name+'_MAE':float(np.mean(np.abs(es))),name+'_RMSE':float(np.sqrt(np.mean(np.square(es)))),name+'_bias':float(np.mean(es))})
            # Repetitions collapse to one contrast group; not independent request tests.
            gs=defaultdict(list)
            for p in prs:
                if p['actual_delta_U'] is not None and p['predicted_delta_U_'+alg] is not None:gs[p['contrast_group']].append(p)
            directions=[];ad=[];pd=[]
            for group_rs in gs.values():
                aa=float(np.mean([p['actual_delta_U'] for p in group_rs]));pp=float(np.mean([p['predicted_delta_U_'+alg] for p in group_rs]))
                directions.append(aa*pp>0);ad.append(aa);pd.append(pp)
            row.update(matched_contrast_groups=len(gs),delta_direction_accuracy=float(np.mean(directions)) if directions else None,
                       delta_U_MAE=float(np.mean(np.abs(np.array(pd)-ad))) if ad else None,
                       delta_gain_spearman=float(spearmanr(ad,pd).statistic) if len(set(ad))>1 and len(set(pd))>1 else None,
                       positive_gain_gt_001_groups=sum(a>.01 for a in ad))
            if ad:
                chosen=int(np.argmax(pd));row.update(top_predicted_gain_actual_delta_U=ad[chosen],
                    top_gain_regret_U=max(ad)-ad[chosen],top_predicted_gain_is_actual_best=ad[chosen]==max(ad))
            out.append(row)
    return out

def main():
    assert (OUT/'original_optimizer_audit.json').exists(),'CPU audit must finish before training'
    assert not (OUT/'offline_validation.json').exists(),'Retain completed validation'
    _,refs,weights=inputs();holdout_combos=set()
    for p in (ROOT/'results/module16_fullpool_gpu/raw_runs').glob('*.json'):holdout_combos.add(combo(read(p)['slot']['deployment']))
    sources=[]
    for module,dirname in DIRECTORIES.items():
        for p in sorted((ROOT/'results'/dirname/'raw_runs').rglob('*.json')):sources.append((module,p))
    save('training_protocol.json',dict(frozen_before_fitting=True,holdout_combinations=sorted(holdout_combos),
        grouping='Entire checkpoint combination across all workloads/sessions/modules; no request or repetition split',
        model='Median log ratios and normalized 33-quantile service shapes per combo/model/task/count; model/task/count fallback; median queue residual',
        group_weighting='One vote per combo/workload/routing group, median over sessions first',
        unknown_path_policy='Standalone unchanged, unsupported uncertainty explicit, no common interference multiplier',
        development_negative_modules=[19,22],no_hyperparameter_tuning_on_holdout=True,
        archival_holdout_not_blind_or_prospective=True,raw_sources=[ref(p) for _,p in sources],
        Module20_note='CPU-only optimizer; no raw GPU sessions',Module21D_note='CPU reanalysis of Module21 sessions, not independent GPU observations'))
    rows=[]
    for module,p in sources:
        raw=read(p);split='RETROSPECTIVE_COMBINATION_HOLDOUT' if combo(raw['slot']['deployment']) in holdout_combos else 'DEVELOPMENT'
        row=extract(raw,module,p,refs,weights,split=split);rows.append(row)
    assert not {r['combo'] for r in rows if r['split']=='DEVELOPMENT'} & holdout_combos
    assert all(r['split']=='DEVELOPMENT' for r in rows if r['module'] in [19,22])
    model=ResidualModel().fit(rows,refs,weights);save('residual_model_history.json',model.as_dict())
    for r in rows:
        if not r['measurement_available']:continue
        for alg,mdl in [('A',None),('B',model)]:
            start=time.perf_counter();p,ps=replay(r['catalog'],r['deployment'],r['routing'],r['rates'],r['budget_mib'],refs,weights,mdl,horizon=r['planned_seconds'])
            r['prediction_'+alg]=p;r['prediction_'+alg+'_paths']=ps;r['prediction_'+alg+'_seconds']=time.perf_counter()-start
    pairrows=pairs(rows);comparison=compare(rows,pairrows);utility,service=scalar_residuals(rows)
    csvsave('service_time_residuals.csv',service);csvsave('utility_prediction_residuals.csv',utility)
    csvsave('prediction_model_comparison.csv',comparison)
    # Include an explicit zero-sample holdout row when ranking cannot be tested.
    ranking=list(pairrows)
    if not any(p['split']=='RETROSPECTIVE_COMBINATION_HOLDOUT' for p in ranking):
        ranking.append(dict(split='RETROSPECTIVE_COMBINATION_HOLDOUT',status='NO_MATCHED_ALTERNATIVE_SAME_WORKLOAD',
                            eligible=False,independent_ranking_sample_count=0,actual_delta_U=None))
    csvsave('ranking_holdout_validation.csv',ranking,['split','module','contrast_group','actual_delta_U'])
    # Keep compact training observations for reproducible Stage5 fit; raw source hashes retained.
    for r in rows:r.pop('cohort_signature',None)
    save('historical_observations.json',rows)
    hold=[r for r in comparison if r['split']=='RETROSPECTIVE_COMBINATION_HOLDOUT']
    ranking_count=max((r['matched_contrast_groups'] for r in hold),default=0)
    save('offline_validation.json',dict(CPU_only=True,raw_sessions=len(rows),
        development_sessions=sum(r['split']=='DEVELOPMENT' for r in rows),holdout_sessions=sum(r['split']!='DEVELOPMENT' for r in rows),
        group_split_no_leakage=True,comparison=comparison,independent_ranking_contrast_groups=ranking_count,
        independent_ranking_supported=False,cross_combination_generalization_established=False,
        reason='Archival combination holdout has no matched alternative deployments at identical workload; absolute-error validation cannot establish deployment ranking',
        fit_seconds=model.fit_seconds,negative_module_results={str(m):[p for p in pairrows if p['module']==m] for m in [19,22]},
        probe_data_in_final_independent_test=False))
    print('OFFLINE_COMPLETE',len(rows),'holdout',sum(r['split']!='DEVELOPMENT' for r in rows),'ranking',ranking_count,flush=True)

if __name__=='__main__':main()
