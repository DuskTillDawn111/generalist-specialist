"""Raw system U, paired uncertainty, thermal comparability and Stage B gate."""
from .common import *
from scipy.stats import t as student_t,binom

def p95_bounds(values):
    values=np.sort(values);n=len(values)
    if not n:return dict(lower_ms=None,upper_ms=None,n=0)
    lo=int(binom.ppf(.025,n,.95));hi=int(binom.ppf(.975,n,.95))+1
    return dict(lower_ms=float(values[lo-1]) if lo>=1 else 0.,
        upper_ms=float(values[hi-1]) if hi<=n else None,n=n,
        status='FINITE_IID_CONDITIONAL_ORDER_STATISTIC' if hi<=n else 'UPPER_UNBOUNDED_SMALL_SAMPLE',
        scope='Equal-tail95 binomial order statistic under iid; queue correlation not certified')

def block_difference(a,b,d,refs,weights,seed):
    """Paired scheduled-time block sensitivity, not independent repeat CI."""
    ra=a['phases']['measurement']['request_records'];rb=b['phases']['measurement']['request_records']
    assert cohort(a)==cohort(b)
    horizon=a['slot']['measurement_seconds'];nb=int(horizon/60)
    partitions=[[] for _ in range(nb)]
    for i,r in enumerate(ra):partitions[min(nb-1,int(r['frozen_offset_seconds']/60))].append(i)
    paired={};total=sum(a['slot']['lambda_rps'].values());rng=np.random.default_rng(seed)
    for label,raw in [('a',a),('b',b)]:
        rs=raw['phases']['measurement']['request_records'];end=raw['phases']['measurement']['window']['end_time_ns']
        paired[label]=dict(lat=np.array([r['e2e_latency_ms'] for r in rs]),tasks=np.array([r['task'] for r in rs]),
            within=np.array([r['completion_time_ns']<=end for r in rs]),
            mem=d['safe_factor']*max(np.mean([s['owned_used_mib'] for s in raw[k]]) for k in ['postwarm_idle','postmeasurement_idle'])/BUDGET)
    values=[]
    for _ in range(500):
        ix=np.concatenate([partitions[i] for i in rng.integers(0,nb,nb)]);us=[]
        for label in ['a','b']:
            data=paired[label];L=P=0.;available=True
            for task,q in a['slot']['lambda_rps'].items():
                lat=data['lat'][ix][data['tasks'][ix]==task]
                if not len(lat):available=False;break
                L+=q/total*np.mean(lat)/refs[task];P+=q/total*np.percentile(lat,95)/refs[task]
            if not available:break
            T=1-min(1.,float(np.sum(data['within'][ix]))/(total*horizon))
            us.append(-weights['T']*T-weights['L']*L-weights['P']*P-weights['M']*data['mem'])
        if len(us)==2:values.append(us[0]-us[1])
    return dict(low=float(np.percentile(values,2.5)),high=float(np.percentile(values,97.5)),block_seconds=60.,draws=len(values),
        scope='CONDITIONAL_SINGLE_PAIR_BLOCK_SENSITIVITY_NOT_INDEPENDENT_REPEAT_CI',
        limitations='No new GPU trajectories, no resimulated queues; memory fixed; original window completion indicators resampled; this is sensitivity only')

def main():
    assert (OUT/'replication_campaign.json').exists(),'GPU campaign must finish first'
    assert not (OUT/'replication_result.json').exists(),'No repeated analysis overwrite'
    manifest=read(OUT/'frozen_replication_manifest.json');campaign=read(OUT/'replication_campaign.json');d,refs,weights=inputs()
    runs=[];taskrows=[];workerrows=[];decomposition=[];raws={};observations={};thermalrows=[]
    for name in campaign['executed']:
        path=OUT/'raw_runs'/(name+'.json');raw=read(path);raws[name]=raw;raw['_path']=str(path)
        ob=observe(raw,path);observations[name]=ob;r=arithmetic(raw,d,refs,weights)
        rr=raw.get('phases',{}).get('measurement',{}).get('request_records',[]);done=[v for v in rr if v['status']=='completed']
        asr=[v for v in rr if v['task']=='asr'];nG=sum(raw['replica_to_family'][v['model']]==G for v in asr)
        runs.append(dict(run_id=name,arm=raw['slot']['arm'],repetition=raw['slot']['repetition'],seed=raw['slot']['seed'],
            status=raw['status'],eligible=ob['eligible'],actual_U=ob['U'],hard_unstable=ob['hard_unstable'],censored=ob['censored'],
            ASR_G_fraction=nG/len(asr) if asr else None,ASR_G_count=nG,ASR_count=len(asr),
            raw=ref(path),errors=raw['errors'],**{k:v for k,v in r.items() if k not in ['U','eligible','per_task','hard_unstable']}))
        for k in ['T','L','P','M']:
            decomposition.append(dict(run_id=name,arm=raw['slot']['arm'],repetition=raw['slot']['repetition'],component=k,
                contribution=r.get(k+'_contribution'),eligible=ob['eligible']))
        for t,q in raw['slot']['lambda_rps'].items():
            samples=[v for v in done if v['task']==t];e2e=[v['e2e_latency_ms'] for v in samples]
            responses=[v['worker_response'] for v in samples]
            correct=[v['correct'] for v in responses if isinstance(v.get('correct'),bool)]
            words=[v for v in responses if v.get('word_errors') is not None and v.get('reference_words') is not None]
            word_total=sum(v['reference_words'] for v in words);error_total=sum(v['word_errors'] for v in words)
            taskrows.append(dict(run_id=name,arm=raw['slot']['arm'],repetition=raw['slot']['repetition'],task=t,
                offered=sum(v['task']==t for v in rr),completed=len(samples),E2E=stats(e2e),P95_interval=p95_bounds(e2e),
                service=stats([v['worker_latency_ms'] for v in samples]),queue=stats([v['worker_queue_wait_ms'] for v in samples]),
                quality_admission='UNCHANGED_FROZEN_SUPPORTED_PATH',
                raw_correct_count=sum(v.get('correct') is True for v in responses) if any('correct' in v for v in responses) else None,
                quality_correct_scored_requests=len(correct),accuracy=sum(correct)/len(correct) if correct else None,
                quality_WER_scored_requests=len(words),raw_reference_words=word_total if words else None,
                raw_word_errors=error_total if words else None,weighted_WER=error_total/word_total if word_total else None,
                quality_score_scope='Descriptive raw scored population only; missing metrics not imputed; admission unchanged',
                closed_set_invalid=sum(v.get('valid_closed_set_output') is False for v in responses),
                mean_U_contribution=-weights['L']*q/sum(raw['slot']['lambda_rps'].values())*np.mean(e2e)/refs[t] if e2e else None,
                P95_U_contribution=-weights['P']*q/sum(raw['slot']['lambda_rps'].values())*np.percentile(e2e,95)/refs[t] if e2e else None,
                incomplete_or_unstable_latency_only_diagnostic=not ob['eligible']))
        if raw.get('phases',{}).get('measurement'):
            w=raw['phases']['measurement']['window'];end=w['start_unix_ns']+int(w['planned_duration_seconds']*1e9)
            for worker in raw['worker_identities']:
                md=[v for v in done if v['model']==worker];state=next((v for v in ob['workers'] if v['worker']==worker),{})
                busy=sum(max(0,min(v['worker_response']['end_ns'],end)-max(v['worker_response']['start_ns'],w['start_unix_ns'])) for v in md)/(end-w['start_unix_ns'])
                for task in sorted({v['task'] for v in md}):
                    records=[v for v in md if v['task']==task]
                    workerrows.append(dict(run_id=name,arm=raw['slot']['arm'],repetition=raw['slot']['repetition'],
                        model=raw['replica_to_family'][worker],task=task,completed=len(records),worker_total_busy_fraction=busy,
                        service=stats([v['worker_latency_ms'] for v in records]),queue=stats([v['worker_queue_wait_ms'] for v in records]),**(state or {'worker':worker})))
        for label in ['postwarm_idle','measurement','postmeasurement_idle']:
            samples=raw.get(label,[]) if label!='measurement' else [v for v in raw['telemetry']['samples'] if v['phase']=='measurement']
            missing_thermal=sum('temperature_gpu_c' not in v or 'sm_clock_mhz' not in v or 'power_draw_w' not in v for v in samples)
            samples=[v for v in samples if all(k in v for k in ['temperature_gpu_c','sm_clock_mhz','power_draw_w'])]
            if samples:
                thermalrows.append(dict(run_id=name,arm=raw['slot']['arm'],repetition=raw['slot']['repetition'],phase=label,
                    missing_thermal_samples=missing_thermal,
                    temperature_mean_c=float(np.mean([v['temperature_gpu_c'] for v in samples])),
                    temperature_max_c=max(v['temperature_gpu_c'] for v in samples),
                    sm_clock_mean_mhz=float(np.mean([v['sm_clock_mhz'] for v in samples])),
                    sm_clock_range_mhz=[min(v['sm_clock_mhz'] for v in samples),max(v['sm_clock_mhz'] for v in samples)],
                    power_mean_w=float(np.mean([v['power_draw_w'] for v in samples])),foreign_GPU_samples=sum(bool(v['foreign_pids']) for v in samples)))
    pairs=[]
    for rep in [1,2,3]:
        names={o['arm']:o['run_id'] for o in runs if o['repetition']==rep}
        if set(names)!=set(['proposed','control']):continue
        a=raws[names['proposed']];b=raws[names['control']];oa=observations[names['proposed']];ob=observations[names['control']]
        valid=oa['eligible'] and ob['eligible']
        full_phases=all(r.get('phases',{}).get('measurement') for r in [a,b])
        same=bool(full_phases and cohort(a)==cohort(b))
        tv=[[v['temperature_gpu_c'] for v in r.get('postwarm_idle',[]) if 'temperature_gpu_c' in v] for r in [a,b]]
        temps=[float(np.mean(v)) if v else None for v in tv]
        tempdiff=abs(temps[0]-temps[1]) if all(t is not None for t in temps) else None
        thermal_ok=tempdiff is not None and tempdiff<=manifest['thermal_protocol']['postwarm_pair_temperature_difference_max_c']
        counts=[min((v['count'] for v in o.get('per_task',{}).values()),default=0) for o in [oa,ob]]
        support=min(counts)>=manifest['replication_gate']['per_task_P95_min_requests']
        delta=oa['U']-ob['U'] if valid else None
        parts={k+'_delta':oa['actual_components'][k+'_contribution']-ob['actual_components'][k+'_contribution'] for k in ['T','L','P','M']} if valid else {}
        if valid:assert abs(sum(parts.values())-delta)<1e-12
        sensitivity=block_difference(a,b,d,refs,weights,26200+rep) if valid and same else None
        pairs.append(dict(repetition=rep,proposed_run=names['proposed'],control_run=names['control'],
            proposed_U=oa['U'],control_U=ob['U'],delta_U=delta,**parts,both_legal_stable=valid,
            same_all_inputs_and_trace=same,same_deployment=a['slot']['deployment']==b['slot']['deployment'],
            only_target_routing_differs=all(a['slot']['routing'][t]==b['slot']['routing'][t] for t in a['slot']['routing'] if t!='asr'),
            postwarm_temperatures_c=temps,postwarm_temperature_difference_c=tempdiff,thermal_comparable=thermal_ok,
            smallest_task_request_count=min(counts),P95_min_sample_support=support,
            conditional_block_sensitivity=sensitivity,independent_repeat_CI_not_estimable_from_one_pair=True))
    deltas=[p['delta_U'] for p in pairs if p['delta_U'] is not None];n=len(deltas);CI=None
    if n>=2:
        mean=float(np.mean(deltas));se=float(np.std(deltas,ddof=1)/math.sqrt(n));half=float(student_t.ppf(.975,n-1)*se)
        CI=dict(mean_delta_U=mean,low=mean-half,high=mean+half,n=n,df=n-1,
            method='Approximate paired-effects Student t95; independent sessions, assumed normally distributed effects',
            small_sample_uncertainty='n=3: heavy assumption dependence; report all effects, no broad model-combination generalization')
    comparable=bool(n==3 and all(p['both_legal_stable'] and p['same_all_inputs_and_trace'] and p['same_deployment'] and
        p['only_target_routing_differs'] and p['thermal_comparable'] and p['P95_min_sample_support'] for p in pairs))
    allpos=n==3 and all(v>.01 for v in deltas);replicated=bool(comparable and allpos and CI and CI['low']>0)
    if campaign['stop']:status='PROBE_NOT_REPRODUCIBLE'
    elif n==3 and all(v<=0 for v in deltas):status='PROBE_NOT_REPRODUCIBLE'
    elif replicated:status='POSITIVE_CONFIGURATION_REPLICATED'
    else:status='POSITIVE_REPLICATION_INCONCLUSIVE'
    result=dict(at_utc=now(),status=status,Stage_B_replicated=replicated,Stage_C_permitted=replicated,independent_pairs=len(pairs),
        independent_delta_U=deltas,paired_effect_interval=CI,all_three_over001=allpos,all_pairs_comparable=comparable,
        positive_direction_count=sum(v>0 for v in deltas),negative_direction_count=sum(v<0 for v in deltas),
        exact_one_sided_sign_p_if_all_positive=.125 if n==3 and all(v>0 for v in deltas) else None,
        exact_sign_limit='Three independent positive pairs alone cannot yield a5 percent distribution-free sign significance claim',
        original_24A_not_in_independent_denominator=True,pairs=pairs,runs=runs,
        original_probe_window120_extended_to2400_user_authorized=True,measured_full_system_utility=True,
        original_20A_quality_memory_stability_and_U_unchanged=True,conditional_block_intervals_not_repeat_CI=True,
        no_outcome_driven_retries_or_extra_seeds=True,
        infrastructure_recovery_separately_disclosed=(OUT/'infrastructure_recovery_manifest.json').exists(),
        interrupted_attempt_not_in_independent_denominator=True,original_predictors_not_refitted=True)
    csvsave('gpu_paired_runs.csv',runs);csvsave('system_utility_comparison.csv',pairs)
    csvsave('per_task_latency.csv',taskrows);csvsave('worker_stability.csv',workerrows)
    csvsave('utility_decomposition.csv',decomposition);csvsave('GPU_thermal_state.csv',thermalrows)
    save('replication_result.json',result)
    print('STAGE_B_RESULT',status,'deltas',deltas,'CI',CI,'C permitted',replicated,flush=True)

if __name__=='__main__':main()
