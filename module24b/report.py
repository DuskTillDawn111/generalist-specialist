"""Report normal utility only for fully eligible independent GPU sessions."""
from .common import *
from collections import Counter
from module20.optimizer import components
from module24b.gpu import observation

def block_sensitivity(a,b,ra,rb,manifest):
    """Matched 20s arrival blocks, conditional on the two observed GPU sessions.

    These resamples quantify temporal sensitivity, never new independent tests.
    LID-missing resamples are omitted and the omitted count is explicitly given.
    Independent-session inference uses the three original paired contrasts only.
    """
    if not a['eligible'] or not b['eligible']:return dict(available=False,reason='Ineligible pair; normal utility CI unavailable')
    pa=ra['phases']['measurement'];pb=rb['phases']['measurement']
    bins=[]
    for ph in [pa,pb]:
        bins.append([[r for r in ph['request_records'] if min(5,int(r['frozen_offset_seconds']/20.))==i] for i in range(6)])
    rng=np.random.default_rng(24290+a['repetition']);deltas=[];missing=0
    for _ in range(1000):
        selected=rng.integers(0,6,6);us=[]
        for arm,ph,ob in zip(bins,[pa,pb],[a,b]):
            rows=[r for i in selected for r in arm[i]];pt={}
            for t in manifest['rates']:
                values=[r['e2e_latency_ms'] for r in rows if r['task']==t]
                if values:pt[t]=summarize_samples(values)
            if set(pt)!=set(manifest['rates']):break
            within=sum(r['completion_time_ns']<=ph['window']['end_time_ns'] for r in rows)
            T=1-min(1.,within/(120*sum(manifest['rates'].values())))
            us.append(components(pt,manifest['rates'],manifest['refs_ms'],T,ob['safe_memory_mib']/BUDGET,manifest['weights'])['U'])
        if len(us)==2:deltas.append(us[0]-us[1])
        else:missing+=1
    return dict(available=bool(deltas),method='Matched 20s arrival-block resampling, 1000 draws, empirical percentile 95%',
        conditional_percentile_95=list(map(float,np.percentile(deltas,[2.5,97.5]))) if deltas else None,
        valid_draws=len(deltas),missing_task_draws=missing,block_count=6,block_seconds=20,
        scope='Temporal sensitivity conditional on one GPU session per arm. Carryover queues, sparse LID and six blocks limit validity. Not independent-session confidence or extra test samples.')

def worker_rows(raw,ob):
    slot=raw['slot'];ph=raw.get('phases',{}).get('measurement')
    if not ph:return [dict(run_id=slot['run_id'],arm=slot['arm'],diagnostic_only=slot['arm']=='preflight',
        status='NO_REQUEST_MEASUREMENT',reason=raw['errors'],worker_identities=raw['worker_identities'])]
    rs=ph['request_records'];w=ph['window'];duration=w['planned_duration_seconds']
    wu=w['start_unix_ns'];eu=wu+round(duration*1e9);rows=[];states={v['worker']:v for v in ob['workers']}
    for worker,ident in raw['worker_identities'].items():
        allr=[r for r in rs if r['model']==worker];done=[r for r in allr if r['status']=='completed'];within=[r for r in done if r['completion_time_ns']<=w['end_time_ns']]
        accepted=[r for r in allr if r.get('submit_write_completed_time_ns') is not None or r.get('worker_response') is not None]
        busy=sum(max(0,min(r['worker_response']['end_ns'],eu)-max(r['worker_response']['start_ns'],wu)) for r in done)/1e9
        wr=states.get(worker,dict(state='IDLE_NO_OFFERED_REQUESTS'))
        for t in sorted({r['task'] for r in allr}) or ['IDLE']:
            dd=[r for r in done if r['task']==t];offered=sum(r['task']==t for r in allr)
            rows.append(dict(run_id=slot['run_id'],arm=slot['arm'],repetition=slot['repetition'],
                diagnostic_only=slot['arm']=='preflight',worker=worker,family=raw['replica_to_family'][worker],pid=ident['pid'],task=t,
                **{k:v for k,v in wr.items() if k!='worker'},offered_requests=offered,accepted_requests=sum(r['task']==t for r in accepted),
                completed_requests=len(dd),offered_qps=offered/duration,completed_qps=len(dd)/duration,
                within_window_completed_qps=sum(r['task']==t for r in within)/duration,
                worker_busy_seconds=busy,worker_busy_fraction=busy/duration,
                worker_busy_scope='Completed-response interval overlap with offered window; lower bound if incomplete; repeated across task rows, do not sum task rows',
                service_distribution=summarize_samples([r['worker_latency_ms'] for r in dd]),
                worker_queue_distribution=summarize_samples([r['worker_queue_wait_ms'] for r in dd]),
                host_send_queue_distribution=summarize_samples([(r['actual_submit_time_ns']-r['enqueued_time_ns'])/1e6 for r in dd]),
                normal_E2E=summarize_samples([r['e2e_latency_ms'] for r in dd]) if ob['eligible'] else None,
                completed_subset_E2E_DIAGNOSTIC=summarize_samples([r['e2e_latency_ms'] for r in dd]),
                eligible=ob['eligible'],hard_unstable=ob['hard_unstable'],censored=ob['censored']))
    # There is a family route, independently thinned into two physical worker
    # queues. Aggregate their backlog; do not invent a shared scheduler queue.
    for family,n in slot['deployment'].items():
        wr=[states[k] for k in states if raw['replica_to_family'][k]==family]
        rows.append(dict(run_id=slot['run_id'],arm=slot['arm'],repetition=slot['repetition'],diagnostic_only=slot['arm']=='preflight',
            family=family,kind='FAMILY_AGGREGATE',replicas=n,shared_queue_architecture='No shared family scheduler; independent per-worker FIFO queues after frozen random thinning',
            aggregate_end_backlog=sum(v['end_backlog'] for v in wr),aggregate_final_backlog=sum(v['final_backlog'] for v in wr),
            hard_unstable=any(v['state']=='unstable' for v in wr)))
    return rows

def main():
    manifest=read(OUT/'frozen_high_manifest.json');cpu=read(OUT/'high_cpu_recheck.json')
    pre=read(OUT/'replica_colocation_preflight.json') if (OUT/'replica_colocation_preflight.json').exists() else None
    formal=read(OUT/'formal_observations.json') if (OUT/'formal_observations.json').exists() else []
    rawby={p.stem:read(p) for p in sorted((OUT/'raw_runs').glob('*.json'))} if (OUT/'raw_runs').exists() else {}
    obsby={o['run_id']:o for o in formal};runs=[];system=[];decomp=[];workers=[];comparisons=[];resources=[]
    for name,raw in rawby.items():
        for label in ['load','path_probe','measurement']:
            samples=[s for s in raw.get('telemetry',{}).get('samples',[]) if s['phase']==label]
            if not samples:continue
            row=dict(run_id=name,arm=raw['slot']['arm'],phase=label,samples=len(samples),
                diagnostic_only=raw['slot']['arm']=='preflight',
                per_process_GPU_status=sorted({s['per_process_gpu_utilization_status'] for s in samples}),
                owned_context_PID_count_range=[min(len(s['processes']) for s in samples),max(len(s['processes']) for s in samples)])
            for key in ['gpu_util_percent','memory_util_percent','worker_CPU_core_percent','host_RSS_bytes','host_available_RAM_bytes','owned_used_mib']:
                values=[s[key] for s in samples if s.get(key) is not None]
                if values:row.update({key+'_mean':float(np.mean(values)),key+'_p95':float(np.percentile(values,95)),key+'_max':float(np.max(values))})
            resources.append(row)
    csvsave('resource_competition.csv',resources,['run_id','arm','phase','samples'])
    if pre:
        pre['resource_competition_summary']=[r for r in resources if r['arm']=='preflight']
        pre['full_HIGH_worker_measurements']=worker_rows(rawby['high_preflight_rep0'],pre['full_HIGH_observation'])
        for scaling in pre['replica_scalings']:
            for mode in ['one','two']:
                window=scaling['raw_'+mode]
                per_replica=[]
                for worker in window.get('active_workers',[]):
                    done=[r for r in window['request_records'] if r['model']==worker and r['status']=='completed'
                        and r['completion_time_ns']<window['window']['end_time_ns']]
                    per_replica.append(dict(worker=worker,completed_in_window=len(done),
                        completed_qps=window['per_replica_throughput_rps'][worker],
                        service=summarize_samples([r['worker_latency_ms'] for r in done]),
                        queue=summarize_samples([r['worker_queue_wait_ms'] for r in done])))
                scaling[mode+'_active_per_replica_measurements']=per_replica
        save('replica_colocation_preflight.json',pre)
    slots=manifest.get('slots',[])
    # Required files remain meaningful even when Stage B prevents every formal run.
    formal_slots=[s for s in slots if s['arm']!='preflight'] or [dict(run_id=f'high_{arm}_rep{rep}',arm=arm,repetition=rep,
        seed=24200+rep,diagnostic_only=False) for rep in [1,2,3] for arm in manifest['protocol']['arm_order'][rep-1]]
    for slot in formal_slots:
        name=slot['run_id'];raw=rawby.get(name);ob=obsby.get(name)
        if not raw or not ob:
            runs.append(dict(run_id=name,arm=slot['arm'],repetition=slot['repetition'],seed=slot['seed'],
                status='NOT_EXECUTED',actual_U=None,reason='Stage A/B gate or frozen campaign stop',independent_validation=True))
            continue
        ph=raw.get('phases',{}).get('measurement',{});rs=ph.get('request_records',[]);duration=120.
        accepted=sum(r.get('submit_write_completed_time_ns') is not None or r.get('worker_response') is not None for r in rs)
        asr=[r for r in rs if r['task']=='asr'];g=sum(raw['replica_to_family'][r['model']]==G for r in asr)
        asrdone=[r for r in asr if r['status']=='completed'];gd=sum(raw['replica_to_family'][r['model']]==G for r in asrdone)
        runs.append(dict(run_id=name,arm=slot['arm'],repetition=slot['repetition'],seed=slot['seed'],status=raw['status'],
            independent_validation=True,raw=ref(OUT/'raw_runs'/(name+'.json')),trajectory=slot['trajectories']['measurement'],
            eligible=ob['eligible'],hard_unstable=ob['hard_unstable'],censored=ob['censored'],actual_U=ob['U'],
            offered=ob.get('offered'),accepted=accepted,completed=ob.get('completed'),within=ob.get('within'),
            target_total_qps=sum(manifest['rates'].values()),offered_qps=ob.get('offered',0)/duration,
            accepted_qps=accepted/duration,completed_qps=ob.get('completed',0)/duration,
            within_window_completed_qps=ob.get('within',0)/duration,
            safe_memory_mib=ob.get('safe_memory_mib'),sampled_peak_mib=ob.get('sampled_peak_mib'),
            actual_ASR_G_offered_fraction=g/len(asr) if asr else None,actual_ASR_G_completed_fraction=gd/len(asrdone) if asrdone else None,
            actual_ASR_G_completed=gd,actual_ASR_S_completed=len(asrdone)-gd,
            quality_admission_preserved=all(r['task'] in manifest['quality_admission'][raw['replica_to_family'][r['model']]]['supported_tasks'] for r in rs),
            GPU_release_verified=raw['gpu_release']['verified'],errors=raw['errors']))
        for t in [*manifest['rates'],'POOLED_SYSTEM']:
            pt=ob.get('per_task',{}).get(t,{});taskrs=[r for r in rs if r['task']==t] if t!='POOLED_SYSTEM' else rs
            mean=pt.get('mean_ms') if t!='POOLED_SYSTEM' else ob.get('pooled_mean_ms')
            p95=pt.get('p95_ms') if t!='POOLED_SYSTEM' else ob.get('pooled_p95_ms')
            ac=sum(r.get('submit_write_completed_time_ns') is not None or r.get('worker_response') is not None for r in taskrs)
            system.append(dict(kind='RUN_TASK',run_id=name,arm=slot['arm'],repetition=slot['repetition'],task=t,
                actual_U=ob['U'] if t=='POOLED_SYSTEM' else None,eligible=ob['eligible'],
                normal_mean_ms=mean if ob['eligible'] else None,normal_p95_ms=p95 if ob['eligible'] else None,
                completed_subset_DIAGNOSTIC_mean_ms=mean,completed_subset_DIAGNOSTIC_p95_ms=p95,
                samples=pt.get('count') if t!='POOLED_SYSTEM' else ob.get('completed'),
                offered_qps=len(taskrs)/duration,accepted_qps=ac/duration,
                completed_qps=sum(r['status']=='completed' for r in taskrs)/duration,
                fixed_target_qps=manifest['rates'].get(t,sum(manifest['rates'].values()))))
        if ob.get('actual_components'):
            for key,label in [('T','throughput_loss'),('L','task_normalized_mean'),('P','task_normalized_P95'),('M','safe_memory')]:
                decomp.append(dict(kind='RUN_COMPONENT',run_id=name,arm=slot['arm'],repetition=slot['repetition'],component=key,
                    label=label,weight=manifest['weights'][key],normalized_value=ob['actual_components'][key+'_norm'],
                    contribution=ob['actual_components'][key+'_contribution'] if ob['eligible'] else None,
                    diagnostic_completed_subset_contribution=ob['actual_components'][key+'_contribution'],eligible=ob['eligible']))
            for task,q in manifest['rates'].items():
                pt=ob.get('per_task',{}).get(task)
                if not pt:continue
                for key,stat in [('L','mean_ms'),('P','p95_ms')]:
                    value=-manifest['weights'][key]*q/sum(manifest['rates'].values())*pt[stat]/manifest['refs_ms'][task]
                    decomp.append(dict(kind='RUN_TASK_LATENCY_COMPONENT',run_id=name,arm=slot['arm'],repetition=slot['repetition'],
                        task=task,component=key,contribution=value if ob['eligible'] else None,
                        diagnostic_completed_subset_contribution=value,eligible=ob['eligible']))
        workers.extend(worker_rows(raw,ob))
    if 'high_preflight_rep0' in rawby:
        workers.extend(worker_rows(rawby['high_preflight_rep0'],pre['full_HIGH_observation']))
    for rep in [1,2,3]:
        a=obsby.get(f'high_proposed_rep{rep}');b=obsby.get(f'high_control_rep{rep}')
        row=dict(kind='PAIRED_COMPARISON',repetition=rep,proposed_U=a['U'] if a else None,control_U=b['U'] if b else None,
            delta_U=None,status='NOT_EXECUTED_OR_INCOMPLETE_PAIR',independent_pair=True)
        if a and b:
            ra=rawby[a['run_id']];rb=rawby[b['run_id']]
            sa=ra['slot'];sb=rb['slot'];matched=sa['trajectories']==sb['trajectories'] and sa['deployment']==sb['deployment'] and sa['lambda_rps']==sb['lambda_rps']
            matched=matched and all(sa['routing'][t]==sb['routing'][t] for t in manifest['rates'] if t!='asr')
            matched=matched and a.get('cohort_signature')==b.get('cohort_signature')
            if a['eligible'] and b['eligible'] and matched:
                delta=a['U']-b['U'];row.update(delta_U=delta,status='COMPARABLE',matched=True,
                    above_positive_threshold=delta>.01,conditional_temporal_sensitivity=block_sensitivity(a,b,ra,rb,manifest),
                    independent_per_pair_CI='Not identifiable: one GPU session per arm; repeat-level CI uses original independent pairs only')
                for key in ['T','L','P','M']:
                    da=a['actual_components'][key+'_contribution']-b['actual_components'][key+'_contribution']
                    row['delta_'+key+'_contribution']=da
                    decomp.append(dict(kind='PAIRED_COMPONENT_DELTA',repetition=rep,component=key,contribution_gain=da))
                for task,q in manifest['rates'].items():
                    for key,stat in [('L','mean_ms'),('P','p95_ms')]:
                        da=-manifest['weights'][key]*q/sum(manifest['rates'].values())*(a['per_task'][task][stat]-b['per_task'][task][stat])/manifest['refs_ms'][task]
                        decomp.append(dict(kind='PAIRED_TASK_LATENCY_COMPONENT_DELTA',repetition=rep,task=task,component=key,contribution_gain=da))
                assert abs(sum(row['delta_'+k+'_contribution'] for k in ['T','L','P','M'])-delta)<1e-10
            else:row.update(status='INCOMPARABLE_NO_NORMAL_UTILITY_DELTA',matched=matched,proposed_hard_unstable=a['hard_unstable'],control_hard_unstable=b['hard_unstable'])
        comparisons.append(row)
    ds=[r['delta_U'] for r in comparisons if r['delta_U'] is not None];ci=None;mean=float(np.mean(ds)) if ds else None
    if len(ds)>=2:
        from scipy.stats import t
        margin=float(t.ppf(.975,len(ds)-1)*np.std(ds,ddof=1)/np.sqrt(len(ds)))
        ci=[mean-margin,mean+margin]
    completedpairs=len(ds);actualsplit=all(r.get('actual_ASR_G_completed',0)>0 and r.get('actual_ASR_S_completed',0)>0 for r in runs if r['arm']=='proposed' and r['status']!='NOT_EXECUTED') if formal else False
    if not cpu['passed']:status='HIGH_CPU_NOT_REPRODUCED';reason='Frozen autonomous meaningful HIGH G/S was not reproduced'
    elif not pre or not pre['passed']:status='HIGH_REPLICA_CAPACITY_UNSUPPORTED';reason='Exact frozen replica configuration did not pass GPU colocation safety/capacity preflight'
    elif any(o['hard_unstable'] for o in formal) or any(r.get('errors') for r in runs):status='HIGH_GPU_UNSTABLE';reason='Formal GPU stability/runtime constraints failed; unstable U and paired gains are null'
    elif any(d<=0 for d in ds) or (completedpairs==3 and not all(d>.01 for d in ds)):
        status='HIGH_NO_UTILITY_GAIN';reason='At least one comparable independent pair failed the required positive-utility criterion'
    elif completedpairs==3 and actualsplit and all(d>.01 for d in ds) and ci is not None and ci[0]>0:
        status='HIGH_POSITIVE_UTILITY_VALIDATED';reason='Three legal stable matched independent HIGH pairs all delta_U>0.01; paired t95 CI is positive under its small-sample assumptions'
    else:status='HIGH_UTILITY_INCONCLUSIVE';reason='Insufficient comparable pairs, nonzero executed split, or independent repeat-level confidence for clear HIGH advantage'
    integrity_before=read(OUT/'historical_metadata_before.json');integrity_after=historical_metadata()
    changed=[p for p,v in integrity_before.items() if integrity_after.get(p)!=v];verify(read(OUT/'read_only_input_inventory.json'))
    assert not changed,'Historical files changed: '+repr(changed[:5])
    supported=status=='HIGH_POSITIVE_UTILITY_VALIDATED'
    aggregate=dict(kind='INDEPENDENT_PAIR_SUMMARY',pairs=completedpairs,delta_U_mean=mean,paired_t_95_CI=ci,
        t_CI_assumptions='Independent fresh session pairs, approximately normal paired differences; n<=3 cannot validate normality',
        no_request_random_split=True,sign_test_one_sided_p=.125 if len(ds)==3 and all(d>0 for d in ds) else None,
        uncertainty='Three pairs give limited distribution-free evidence; no cross-combination predictor generalization is established',
        gate=status)
    gate=dict(module='24B',status=status,reason=reason,HIGH_CPU_autonomous_GS_reproduced=cpu['passed'],
        frozen_ASR_G_fraction=manifest['ASR_G_fraction'],HIGH_GS_executed_in_preflight=bool(pre and pre['full_HIGH_observation'].get('measurement_available')),
        HIGH_GS_executed_formally=any(r['arm']=='proposed' and r.get('actual_ASR_G_completed',0)>0 for r in runs),
        real_replica_colocation_capacity_supported=bool(pre and pre['passed']),formal_GPU_runs=len(formal),
        independent_comparable_pairs=completedpairs,independent_positive_HIGH_utility_supported=supported,
        all_three_delta_U_above_001=completedpairs==3 and all(d>.01 for d in ds),actual_nonzero_GS=actualsplit,
        aggregate=aggregate,pairs=comparisons,
        historical_Module24A_gate=ref(HISTORY/'module24a_gate.json'),historical_files_unchanged=True,
        historical_file_count=len(integrity_before),Module24A_gate_modified=False,
        preflight_used_as_independent_gain_evidence=False,development_probe_used_as_success_evidence=False,
        predictor_refit_on_this_experiment=False,independent_ranking_across_combinations_established=False,
        LOW_MID_both_G_only_not_failure=True,Module24C_worth_continuing=supported,Module24C_executed=False,
        stop_after_Module24B=True)
    if (OUT/'formal_campaign.json').exists():gate['formal_stop']=read(OUT/'formal_campaign.json')['stop']
    gate['early_stop_negative_result_is_not_three_pair_effect_estimate']=completedpairs<3 and any(d<0 for d in ds)
    save('module24b_gate.json',gate);save('historical_integrity.json',dict(changed=changed,count=len(integrity_before),scientific_SHA256_verified=True))
    csvsave('high_gpu_paired_runs.csv',runs,['run_id','arm','repetition','status','actual_U'])
    csvsave('system_utility_comparison.csv',system+comparisons+[aggregate],['kind','run_id','repetition','task','actual_U','delta_U','status'])
    csvsave('utility_decomposition.csv',decomp or [dict(kind='NOT_EXECUTED',reason=reason,contribution_gain=None)],['kind','run_id','repetition','component','contribution','contribution_gain'])
    csvsave('worker_stability.csv',workers or [dict(status='NOT_EXECUTED',reason=reason)],['run_id','arm','worker','family','task','state'])
    text=[f'# Module 24B：HIGH 独立真实效用验证\n',f'Gate：**{status}**。{reason}。\n',
        '本轮只验证 HIGH。LOW/MID 均为自主 G-only 不构成失败；Module24A 原 gate 和全部历史文件保持不变。未执行 Module24C。\n',
        '## 冻结与 CPU 复核\n',
        f'全部 {cpu["all_states"]} 个副本部署枚举，其中 {cpu["legal_deployments"]} 个合法部署参与自由路由竞争，{cpu["stages"]["HIGH"]["evaluations"]} 次效用评估。复核通过={cpu["passed"]}，最优部署/路由与24A一致，ASR G份额={manifest["ASR_G_fraction"]:.10%}，原预测U={manifest["prediction"]["U"]:.10f}，同驻留S-only预测U={manifest["control_prediction"]["U"]:.10f}，预测ΔU={manifest["predicted_delta_U_same_residency"]:.10f}。不是数值并列、零请求残差或三阶段强制约束。自由连续路由搜索有多种子/端点/局部精化，不声称全局数学证明。\n',
        '驻留：Qwen×1、Whisper-large-v3×1、RoBERTa×2、ConvNeXt×2、PARSeq×2；8个独立worker进程。QPS：Text=4、Image=2、OCR=5、ASR=1.52、LID=.03、VQA=.2；预算34567.5534 MiB。Text/Image的原优化小份额原样保留，两臂只有ASR路由不同。质量按既有硬准入及运行期输出有效性检查，不新增质量奖励。\n',
        'U=-.25*T_loss-.30*L_norm-.30*P_norm-.15*M_norm；沿用20A各任务参考值和固定offered权重。独立排序样本原为0、双副本共置支持缺失；此次结果不回写预测器，也不证明跨组合泛化。\n',
        '## 双副本共置预检\n']
    if pre:
        text += [f'通过={pre["passed"]}。显存：{json.dumps(pre["memory"],ensure_ascii=False)}。错误：{pre["errors"]}。GPU释放={pre["gpu_release"]["verified"]}。\n',
            '| 模型 | 预测g(2) | 实测g(2) | 单活跃/双活跃完成容量QPS |\n|---|---:|---:|---|\n']
        for s in pre['replica_scalings']:text.append(f'| {s["model"]} | {s["predicted_g2"]} | {s["measured_g2"]} | {s["one_active_capacity_rps"]}/{s["two_active_capacity_rps"]} |\n')
        text.append('\n扩展比使用固定8副本驻留、单家族单活跃/双活跃的各10秒closed-loop诊断，其他模型此时闲置。不能把它当完整HIGH并发下的饱和容量；后者用固定120秒Poisson负载检查完成、busy、队列增长和积压。预检不是独立效用样本。NVML峰值为0.5秒采样峰值，不是分配器瞬时峰值。\n')
    text += ['\n## 正式独立配对\n',
        f'正式运行={len(formal)}/6，合法可比配对={completedpairs}/3；固定新种子24201/24202/24203，臂顺序A-B/B-A/A-B，轨迹和输入在预检前冻结。warmup10秒、measurement120秒、drain最多60秒，未筛种子、延长窗口或重试结果。\n',
        '| 配对 | Proposed U | Control U | ΔU | 状态 |\n|---|---:|---:|---:|---|\n']
    for r in comparisons:text.append(f'| {r["repetition"]} | {r["proposed_U"]} | {r["control_U"]} | {r["delta_U"]} | {r["status"]} |\n')
    for r in runs:
        if r['status']=='NOT_EXECUTED':continue
        text.append(f'\n{r["run_id"]}：全部完成 {r["completed"]}/{r["offered"]}，窗口内完成 {r["within"]}，实际ASR G/S完成={r["actual_ASR_G_completed"]}/{r["actual_ASR_S_completed"]}，G份额={r["actual_ASR_G_completed_fraction"]:.6%}；safe memory={r["safe_memory_mib"]:.6f} MiB，采样实际峰值={r["sampled_peak_mib"]:.6f} MiB。\n')
    for r in comparisons:
        sensitivity=r.get('conditional_temporal_sensitivity',{})
        if sensitivity.get('available'):
            text.append(f'\n配对{r["repetition"]}的20秒匹配时间块条件敏感性95%范围={sensitivity["conditional_percentile_95"]}，有效重采样={sensitivity["valid_draws"]}/1000、缺少任务={sensitivity["missing_task_draws"]}/1000。此范围不是独立重复的效应置信区间。\n')
    text += [f'\n独立配对平均ΔU={mean}；paired t95 CI={ci}。单个配对仅有每臂一次GPU会话，不具备独立重复CI；CSV另有20秒匹配时间块敏感性区间，明确限定为条件描述，不增加独立样本数。n=3仍不能验证正态假设，即便三次方向相同单侧符号检验p=.125。稀疏LID预期每窗口3.6请求，P95的有限样本不确定性必须保留；缺少任务、持续拥塞或不完整运行不计算正常U。\n',
        '每任务和pooled mean/P95、offered/accepted/completed QPS在system_utility_comparison.csv；busy/worker queue/host queue/backlog和family聚合队列在worker_stability.csv；效用分项和配对分项收益在utility_decomposition.csv。无共享家族调度队列，双副本为独立FIFO队列。\n']
    if ds:
        gains={k:float(np.mean([r['delta_'+k+'_contribution'] for r in comparisons if r['delta_U'] is not None])) for k in ['T','L','P','M']}
        text.append(f'\n可比配对收益分解均值：{json.dumps(gains)}（T吞吐、L任务归一化mean、P任务归一化P95、M显存）。收益排序不能只看pooled latency。\n')
        if completedpairs<3 and any(d<0 for d in ds):
            text.append('\n第一组负收益触发预先冻结的停止规则，后两组未执行。该真实负结果足以使本轮成功条件失败，但单个独立配对不能估计总体效应的独立重复置信区间，不能宣称跨会话显著负收益。\n')
    else:text.append('\n没有合法独立配对ΔU，无法判断G/S是否具有真实正效用，也不能归因mean/P95/吞吐收益。开发探针+0.4768没有用于本轮成功判定。\n')
    text += ['\n## 后续决策与完整性\n',f'是否值得进入独立Module24C：{supported}。'+('仅建议另立24C验证LOW/MID与合法G+S对照；本轮停止。' if supported else '当前证据不支持推进该模型池正式后续收益验证。')+'\n',
        f'全部{len(integrity_before)}个历史文件metadata不变，所有24A输出和读取的20A科学输入SHA256不变。复现源码module24b/，新原始GPU/CPU日志完整保留。已完成阶段拒绝静默重跑。\n']
    import re
    document='\n'.join(text)
    document=re.sub(r'(\|[^\n]*\|\n)\n+(?=\|)',r'\1',document)
    (OUT/'module24b_summary.md').write_text(document)
    print('MODULE24B_GATE',status,'formal_runs',len(formal),'pairs',completedpairs,'CI',ci,flush=True)

if __name__=='__main__':main()
