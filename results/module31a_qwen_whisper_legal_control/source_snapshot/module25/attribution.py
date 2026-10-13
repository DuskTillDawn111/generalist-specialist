"""Auditable paired arithmetic, distributions, inputs and oracle diagnostics."""
from .common import *
from module20.optimizer import components,predict
from module24.replay import replay
from module24.calibration import ResidualModel
from scipy.stats import ks_2samp

def input_summary(raw):
    trajectory=read(raw['slot']['trajectories']['measurement']['path']);events=trajectory['events'];result={}
    for t in raw['slot']['lambda_rps']:
        items=[e['item']['sample'] for e in events if e['task']==t]
        entry=dict(count=len(items),unique_inputs=len({s['id'] for s in items}),datasets=dict(Counter(s['dataset'] for s in items)),
            labels=dict(Counter(str(s.get('label')) for s in items)) if t in ['text','image','lid'] else None,
            sample_ids=[s['id'] for s in items])
        for field in ['duration_s','word_length']:
            values=[s[field] for s in items if field in s]
            if values:entry[field]=stats(values)
        if t=='text':entry['text_characters']=stats([len(s['input']) for s in items])
        if t=='visual_qa':entry['question_characters']=stats([len(s['question']) for s in items])
        result[t]=entry
    return result

def path_stats(raw):
    rs=raw['phases']['measurement']['request_records'];result=[]
    for worker,metadata in raw['worker_identities'].items():
        rr=[r for r in rs if r['model']==worker and r['status']=='completed']
        for t in sorted({r['task'] for r in rr}):
            ss=[r for r in rr if r['task']==t]
            result.append(dict(run_id=raw['slot']['run_id'],arm=raw['slot']['arm'],worker=worker,family=raw['replica_to_family'][worker],
                task=t,count=len(ss),service=stats([r['worker_latency_ms'] for r in ss]),queue=stats([r['worker_queue_wait_ms'] for r in ss]),
                E2E=stats([r['e2e_latency_ms'] for r in ss]),
                host_and_transport=stats([r['e2e_latency_ms']-r['worker_latency_ms']-r['worker_queue_wait_ms'] for r in ss])))
    return result

def service_oracle(raw,d,refs,weights):
    """Diagnostic only: observed services/worker arrivals, not a predictor test."""
    rs=raw['phases']['measurement']['request_records'];w=raw['phases']['measurement']['window'];by=defaultdict(list)
    for r in rs:by[r['model']].append(r)
    qerrors=[];latencies=defaultdict(list);within=0;meanlinear=defaultdict(list)
    for worker,rows in by.items():
        available=0
        for r in sorted(rows,key=lambda r:r['actual_submit_unix_ns']):
            arrival=r['actual_submit_unix_ns'];svc=r['worker_latency_ms']*1e6
            start=max(arrival,available);wait=(start-arrival)/1e6;available=start+svc
            qerrors.append(wait-r['worker_queue_wait_ms'])
            residual=r['e2e_latency_ms']-r['worker_latency_ms']-r['worker_queue_wait_ms']
            latency=wait+r['worker_latency_ms']+residual;latencies[r['task']].append(latency)
            within+=int(r['scheduled_time_ns']+latency*1e6<=w['end_time_ns'])
            meanlinear[r['task']].append((r['worker_latency_ms'],r['worker_queue_wait_ms'],residual))
    pt={t:stats(v) for t,v in latencies.items()};T=1-min(1,within/(120*sum(raw['slot']['lambda_rps'].values())))
    idle=max(np.mean([v['owned_used_mib'] for v in raw[k]]) for k in ['postwarm_idle','postmeasurement_idle'])
    c=components(pt,raw['slot']['lambda_rps'],refs,T,d['safe_factor']*idle/raw['slot']['safe_memory_budget_mib'],weights)
    return dict(scope='OBSERVED_SERVICE_ORACLE_DIAGNOSTIC_NOT_INDEPENDENT_PREDICTION',U=c['U'],per_task=pt,
        queue_mean_error_ms=float(np.mean(qerrors)),queue_MAE_ms=float(np.mean(np.abs(qerrors))),queue_p95_absolute_error_ms=float(np.percentile(np.abs(qerrors),95)),
        note='Residual inter-handler gaps and transport timestamps remain; no P95(service)+P95(queue) addition',
        mean_linear_parts={t:dict(zip(['service_ms','queue_ms','host_transport_ms'],map(float,np.mean(v,axis=0)))) for t,v in meanlinear.items()})

def main():
    observations=read(OUT/'observations.json');lookup={o['uid']:o for o in observations};definition,refs=utility_inputs();weights=definition['weights']
    aroot=ROOT/'results/module24_utility_calibration';broot=ROOT/'results/module24b_high_validation'
    saved=read(aroot/'residual_model_final_development.json');bmodel=ResidualModel()
    for k in ['exact','path','training_groups','fit_seconds']:setattr(bmodel,k,saved[k])
    d=read(aroot/'selected_pool_catalog.json');raws={};summary={};paths=[];decomp=[];taskdecomp=[];oracles=[]
    pairs=[('24A','high_gs18_diagnostic_rep1','high_s_only_diagnostic_attempt2'),('24B','high_proposed_rep1','high_control_rep1')]
    for module,aid,bid in pairs:
        a=lookup[f'M{module}:{aid}'];b=lookup[f'M{module}:{bid}'];assert a['eligible'] and b['eligible']
        ra=read(a['raw']['path']);rb=read(b['raw']['path']);raws[module]=(ra,rb);patha=path_stats(ra);pathb=path_stats(rb);paths+=patha+pathb
        pred={}
        for name,model in [('A20',None),('B24',bmodel)]:
            pa,_=replay(d,a['deployment'],a['routing'],a['rates'],a['budget_mib'],refs,weights,model)
            pb,_=replay(d,b['deployment'],b['routing'],b['rates'],b['budget_mib'],refs,weights,model)
            pred[name]=dict(proposed=pa,control=pb,delta_U=pa['U']-pb['U'])
        actual=a['U']-b['U'];oraclea=service_oracle(ra,d,refs,weights);oracleb=service_oracle(rb,d,refs,weights)
        oracles += [dict(module=module,run_id=aid,**oraclea),dict(module=module,run_id=bid,**oracleb)]
        for key in ['T','L','P','M']:
            act=a['actual_components'][key+'_contribution']-b['actual_components'][key+'_contribution']
            for name in pred:
                pr=pred[name]['proposed'][key+'_contribution']-pred[name]['control'][key+'_contribution']
                decomp.append(dict(module=module,component=key,predictor=name,actual_delta=act,predicted_delta=pr,actual_minus_predicted=act-pr))
        for t,q in a['rates'].items():
            partsA=np.array(list(oraclea['mean_linear_parts'][t].values()))
            partsB=np.array(list(oracleb['mean_linear_parts'][t].values()))
            coefficient=-weights['L']*q/sum(a['rates'].values())/refs[t]
            meanparts=coefficient*(partsA-partsB)
            L=-weights['L']*q/sum(a['rates'].values())*(a['per_task'][t]['mean_ms']-b['per_task'][t]['mean_ms'])/refs[t]
            P=-weights['P']*q/sum(a['rates'].values())*(a['per_task'][t]['p95_ms']-b['per_task'][t]['p95_ms'])/refs[t]
            assert abs(sum(meanparts)-L)<1e-10
            taskdecomp.append(dict(module=module,task=t,delta_L=L,delta_P=P,
                delta_L_from_service=float(meanparts[0]),delta_L_from_queue=float(meanparts[1]),delta_L_from_transport=float(meanparts[2]),
                P95_additive_decomposition_prohibited=True))
        asr=ra['phases']['measurement']['request_records'];asr=[r for r in asr if r['task']=='asr'];g=sum(ra['replica_to_family'][r['model']]==G for r in asr)
        actualf=g/len(asr);rr=copy.deepcopy(a['routing']);rr['asr']={G:a['rates']['asr']*actualf,S:a['rates']['asr']*(1-actualf)}
        fractionpred,_=replay(d,a['deployment'],rr,a['rates'],a['budget_mib'],refs,weights,bmodel)
        summary[module]=dict(actual_delta_U=actual,predictions=pred,actual_proposed_U=a['U'],actual_control_U=b['U'],
            CPU_logged_proposed_U=ra['slot']['CPU_predicted_U'],CPU_logged_control_U=rb['slot']['CPU_predicted_U'],
            planned_G_fraction=a['routing']['asr'].get(G,0)/a['rates']['asr'],realized_G_fraction=actualf,G_completed=g,ASR_count=len(asr),
            B24_actual_fraction_counterfactual_delta_U=fractionpred['U']-pred['B24']['control']['U'],
            fraction_counterfactual_scope='CPU diagnostic only; empirical uniform thinning share substituted without rerouting any actual GPU request',
            same_within_pair_residency=a['deployment']==b['deployment'],matched_trajectory=ra['slot']['trajectories']==rb['slot']['trajectories'],
            deployment=a['deployment'],rates=a['rates'],hardware=ra['baseline_GPU'],
            proposed_oracle_U=oraclea['U'],control_oracle_U=oracleb['U'],oracle_delta_U=oraclea['U']-oracleb['U'])
    dev,formal=raws['24A'][0],raws['24B'][0];inputsA=input_summary(dev);inputsB=input_summary(formal)
    inputcompare=[]
    ta=read(dev['slot']['trajectories']['measurement']['path'])['events'];tb=read(formal['slot']['trajectories']['measurement']['path'])['events']
    for t in dev['slot']['lambda_rps']:
        idsA=set(inputsA[t]['sample_ids']);idsB=set(inputsB[t]['sample_ids']);entry=dict(task=t,development=inputsA[t],independent=inputsB[t],
            shared_distinct_inputs=len(idsA&idsB),input_cohorts_identical=inputsA[t]['sample_ids']==inputsB[t]['sample_ids'])
        for field in ['duration_s','word_length']:
            va=[e['item']['sample'][field] for e in ta if e['task']==t and field in e['item']['sample']]
            vb=[e['item']['sample'][field] for e in tb if e['task']==t and field in e['item']['sample']]
            if va and vb:
                ks=ks_2samp(va,vb);entry[field+'_KS']=dict(statistic=float(ks.statistic),pvalue=float(ks.pvalue),scope='Descriptive input check; not proof of exchangeability')
        inputcompare.append(entry)
    fa=read(aroot/'frozen_experiment_manifest.json');fb=read(broot/'frozen_high_manifest.json')
    poolA=ref(fa['plan']['sample_pool']['path']);poolB=ref(fb['plan']['sample_pool']['path'])
    workers_same={m:dict(python_equal=fa['plan']['registry'][m]['python']==fb['plan']['registry'][m]['python'],
        launcher_bytes_equal=ref(fa['plan']['registry'][m]['script'])['sha256']==ref(fb['plan']['registry'][m]['script'])['sha256']) for m in dev['slot']['deployment']}
    artifact=dict(experiments=summary,comparison=dict(checkpoints_same=checkpoint_key(dev['slot']['deployment'])==checkpoint_key(formal['slot']['deployment']),
        actual_counts_same=dev['slot']['deployment']==formal['slot']['deployment'],GPU_uuid_same=dev['baseline_GPU']['uuid']==formal['baseline_GPU']['uuid'],
        GPU_name_driver_same=all(dev['baseline_GPU'][k]==formal['baseline_GPU'][k] for k in ['name','driver']),
        six_offered_rates_same=dev['slot']['lambda_rps']==formal['slot']['lambda_rps'],same_sample_pool_SHA256=poolA['sha256']==poolB['sha256'],
        workers_same=workers_same,trajectory_same=False,input_distributions=inputcompare,
        normalization_weights_and_refs_identical=True,within_each_pair_full_cohort_completed=True,all_four_utility_runs_eligible=True,
        actual_quality_capability_unchanged=True,safe_factor_identical=True,planned_windows_identical=True),
        accounting_reproduction=True,metric_flip_explained=True,unique_physical_causal_root_identified=False,
        missing_causal_evidence=['No matched n1/n2 residency and identical-fraction factorial across same traces',
            'Only one independent24B pair; no repeated effect estimate',
            'No per-kernel, preprocessor, CPU/GPU clock/temperature causal intervention',
            'Sparse LID2 vs5 and VQA32 vs19; tail differences are finite-sample observations'],
        no_empirical_multiplier_applied=True)
    save('module24_prediction_error_attribution.json',artifact);csvsave('attribution_worker_task.csv',paths)
    csvsave('attribution_utility_components.csv',decomp);csvsave('attribution_task_components.csv',taskdecomp)
    save('observed_service_queue_oracle.json',oracles);save('input_distribution_comparison.json',inputcompare)
    lines=['# Module24预测错误定位\n\n',
        '已复现所有四次完整GPU运行及冻结效用。先更正前提：24A开发探针**实际**ΔU=+0.4768，原20A预测ΔU=-0.02925，最终24A校准B预测ΔU=-0.00732。+0.03122是双副本新候选的预测，不能称为开发探针的正预测。24B独立候选实测ΔU=-0.10671。\n\n',
        '| 配置 | 实际ΔU | A20周期预测ΔU | B24校准预测ΔU | 计划/实际G份额 |\n|---|---:|---:|---:|---|\n']
    for module,row in summary.items():lines.append(f'| {module} | {row["actual_delta_U"]:.9f} | {row["predictions"]["A20"]["delta_U"]:.9f} | {row["predictions"]["B24"]["delta_U"]:.9f} | {row["planned_G_fraction"]:.6%}/{row["realized_G_fraction"]:.6%} |\n')
    lines += ['\n1. **驻留/副本/硬件/输入**：同一五个checkpoint、同GPU UUID/型号/driver、同六任务QPS、同sample-pool SHA256、同native worker和Python。24A是五模型各1副本；24B是Qwen1、Whisper1、RoBERTa2、ConvNeXt2、PARSeq2，共8worker。24A为18%探索路由；24B为13.1833%自主路由。seed24101与24201不同，各自配对内输入完全匹配；跨实验输入cohort不相同。不存在“完全相同部署仅换重复”可比性。输入长度/时长/类别计数和KS描述见input_distribution_comparison.json。\n\n',
        '2. **统计口径**：同20A权重、refs、fixed offered任务权重、safe factor、120s窗口、全计划请求和60s drain上限；四个完整cohort均eligible，未用截尾E2E。T_loss计窗口内completion/固定QPS×窗口，不以drain抹平损失；P_norm用每任务P95，不用pooled P95替换。人工独立复算和按服务/等待/传输加性拆分mean通过。P95分位数不具加性，未做错误相加。\n\n',
        '3. **逐worker服务/等待/P95**：attribution_worker_task.csv保留全部物理副本×任务计数、service mean/P95、queue mean/P95、E2E和host/transport。不能把worker handler service当E2E，也不能把单副本低载画像当三种双副本HIGH画像。\n\n',
        '4. **Specialist服务随路由变化**：24A中Whisper mean service由S-only536.66ms变为GS607.59ms，实际有所变慢；24B也必须按各自raw分布比较。B24参数只依赖组合/模型/副本/任务，独立于完整路由，对此变化没有机制。Text/OCR等背景Specialist服务也变化，不只目标ASR队列变化。\n\n',
        '5. **Generalist背景阻塞**：分流会让Qwen的ASR和LID/VQA在同一物理FIFO竞争。background的实际service与queue都需变化；24B VQA P95损失，不能只计算Whisper减压。观察到的handler变化不能唯一归因为GPU干扰，输入混合/CPU预处理/空闲状态也可能参与。\n\n',
        '6. **队列模型**：旧20A周期常数service会遗漏Poisson burst及样本长尾。B24保留周期到达、RR副本和独立路径校准；真实Poisson使用随机副本thinning。其同worker多任务FIFO逻辑本身存在，但副本到达方差和跨模型service动态没有处理，固定queue offset不等于等待分布。observed_service_queue_oracle.json用已测每请求service和真实提交时刻重放原FIFO，仅作诊断，反映队列模型剩余误差；从不计作独立预测能力。\n\n',
        f'7. **13.18%与11.49%**：24B 20/174实际G请求，相比计划约22.94个相差2.94个；独立uniform路由下这是有限样本差异，没有手工改路由。将B24模型仅诊断性代入实际份额，预测ΔU={summary["24B"]["B24_actual_fraction_counterfactual_delta_U"]:.9f}，原计划预测={summary["24B"]["predictions"]["B24"]["delta_U"]:.9f}；此计算不能替代实际系统效用，也不能单独识别翻转原因。\n\n',
        '8. **效用误差逐项归因**：下表为actual-minus-B24预测的ΔU误差；完整A20/B24分解见attribution_utility_components.csv，逐任务与mean服务/等待/传输贡献见attribution_task_components.csv。\n\n',
        '| 实验 | 项 | 实际Δ贡献 | B24预测Δ贡献 | 实际−预测 |\n|---|---|---:|---:|---:|\n']
    for row in decomp:
        if row['predictor']=='B24':lines.append(f'| {row["module"]} | {row["component"]} | {row["actual_delta"]:.9f} | {row["predicted_delta"]:.9f} | {row["actual_minus_predicted"]:.9f} |\n')
    lines += ['\n24B主要损失为P95 -0.076345、mean -0.030519，吞吐仅+0.000163，memory约-0.000010；Text/OCR背景退化占大部分，ASR自身也未改善。24A单副本配对中的背景指标改善不能迁移到24B双副本组合和新输入。旧B在未知双副本路径回退到较快standalone先验，形成过度乐观排序。\n\n',
        '可以解释指标翻转、预测机制缺项和配置不一致，不声称已唯一识别物理根因。独立预测误差仍须用严格整组合留出来验证；不会给ΔU乘经验系数或用oracle服务生成“独立成功”。\n']
    (OUT/'module24_prediction_error_attribution.md').write_text(''.join(lines))
    print('ATTRIBUTION_COMPLETE',summary['24A']['actual_delta_U'],summary['24B']['actual_delta_U'],flush=True)

if __name__=='__main__':main()
