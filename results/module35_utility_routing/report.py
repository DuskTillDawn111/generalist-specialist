"""Numerical tables and paper figures; no optimizer or GPU protocol changes."""
import sys,csv,json,math,statistics,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/generalist-specialist');OUT=ROOT/'results/module35_utility_routing';OLD=ROOT/'results/module34_candidate_pool'
sys.path[:0]=[str(OUT),str(ROOT)]
from m35.common import *
import numpy as np
try:
    from scipy.stats import t as tdist,ttest_1samp
except ImportError:
    # Existing plotting environment lacks scipy. These are exact df1/df2
    # two-sided Student-t formulas; verify against scipy in gsdeploy later.
    from types import SimpleNamespace
    class tdist:
        @staticmethod
        def ppf(p,df):
            assert p==.975 and df in [1,2]
            return {1:12.706204736432095,2:4.302652729696142}[df]
    def ttest_1samp(vals,zero):
        df=len(vals)-1;assert df in [1,2]
        stat=abs(float(np.mean(vals)-zero))/(float(np.std(vals,ddof=1))/math.sqrt(len(vals)))
        p=1-2*math.atan(stat)/math.pi if df==1 else 1-stat/math.sqrt(stat*stat+2)
        return SimpleNamespace(pvalue=p)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'font.size':9})

def table(name):
    rows=list(csv.DictReader((OUT/name).open()))
    for row in rows:
        for k in ['regret','predicted_U','seconds','full_prediction_calls','search_states','worker_prediction_calls','candidate_space','repetition']:
            if row.get(k):row[k]=float(row[k])
        for k in ['deployment','routing']:
            if row.get(k):row[k]=json.loads(row[k])
    return rows
ds=table('cpu_oracle_comparison.csv');hs=table('holdout_results.csv');ts=table('solver_runtime_comparison.csv')
attrs=read(OUT/'routing_gap_attribution.json');checks=read(OUT/'optimizer_correctness_tests.json')
sp=read(OLD/'candidate_space_definition.json');models=read(sp['pools']['C']['path'])['models'];control=read(OUT/'unhelpful_candidate_control.json')
figs=OUT/'paper_figures';figs.mkdir(exist_ok=True)
def outputfig(name,fig,rows):
    fig.tight_layout();fig.savefig(figs/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(figs/(name+'.pdf'),bbox_inches='tight');plt.close(fig);csvsave('paper_figures/'+name+'.csv',rows)
def short(m):
    return 'Whisper-M' if 'whisper-medium@' in m else 'Whisper-L' if 'whisper-large-v3@' in m else 'PARSeq' if 'parseq:' in m else 'Bart' if 'bart-large' in m else m

route_report=['# Module 35 四组路由差异归因','',
'冻结口径：相同候选池、0/1驻留与副本、K=1端点路由、49140MiB、原质量准入、rho≤0.8和原20A FIFO预测U。不能将预测最优解释为GPU最优。', '',
'| 案例 | 原20A U | CPU Oracle U | 预测Regret | 生成集内评分损失 | 候选遗漏损失 | 改动路由 |',
'|---|---:|---:|---:|---:|---:|---|']
for a in attrs:
    route_report.append(f"| {a['scenario']}/{a['pool']} | {a['old_U']:.6f} | {a['oracle_U']:.6f} | {a['regret']:.6f} | {a['generated_ranking_loss']:.6f} | {a['candidate_omission_loss']:.6f} | {', '.join(a['changed_task_routes'])} |")
for a in attrs:
    changes='；'.join(f"{t}: {'/'.join(short(m) for m in a['old_routing'][t])} → {'/'.join(short(m) for m in a['oracle_routing'][t])}" for t in a['changed_task_routes'])
    route_report += ['',f"## {a['scenario']}/{a['pool']}",'',
        f"驻留和副本与Oracle一致={a['same_deployment']}。生成了{a['generated_count']}个唯一完整候选，Oracle路由生成={a['oracle_route_generated']}。候选内U排序损失={a['generated_ranking_loss']:.9f}，遗漏损失={a['candidate_omission_loss']:.9f}。固定原部署的新搜索得到U={a['fixed_deployment_repaired_U']:.9f}。",'',changes+'.', '',
        '1. 原42次MILP计划完整执行；每次只提供一个线性代理目标最优候选。有限候选路由域本身完整定义，但生成器未遍历该域，改变ASR占比、容量种子及显存权重不能保证产生其他任务的队列避让路由。',
        '2. 没有证据表明错误的合法性剪枝或生成循环提前终止导致此次终点差距。局部线性最优不等于全局FIFO U最优，缺口来自候选覆盖。没有通过延长MILP时间或手工插入Oracle路由修复。',
        '3. 原20A只以部署+路由指纹去重，无共享U预测缓存。新进程重复运行、原始候选回放及全空间bundle评分校验支持无状态污染归因。V3历史计数曾重复相加同一个cache；这影响计数，不解释20A选择。',
        '4. 两者使用完全相同的K=1路径、质量、显存、rho及预测稳定性约束。原生20A的终点与网格适配一致；仅连续域与网格差异不能解释此四例。',
        '5. 此处20A候选内优化U，J/U不一致对这四处20A差距的归因份额为0。冻结评分误差在同一CPU预测口径下为0，搜索遗漏解释全部预测Regret；这不是GPU预测误差为0的结论。']
route_report += ['', '## 独立区分三种问题','',
'- **U搜索遗漏**：上述四例，合法U最优候选未生成。新版本用任务子集队列分解和动态规划寻找路由，原20A源码保持不变。',
'- **目标/代理不一致**：V3网格控制在LOW18/C、ASR32B/A、ASR32B/B找到J最优，但U Regret分别约0.004391、0.355421、0.041308。J为拥塞代理，U来自周期FIFO；差异同时包含目标及性能代理定义。原V3部署接受仍为NoiseAwareRuntimeObjective多指标门槛，不能说它以全局J或U为唯一目标。',
'- **CPU预测与GPU不一致**：四例中只有TEXT10/A、B有对应GPU测量，ASR32B/A、B没有对应测量。A的旧/新路由真实优势可复用34配对证据；B的微差区间跨0，不能因预测Regret消除而声称真实优势。性能画像、周期/Poisson到达、并存干扰和调度均可能贡献；本轮不重新训练预测器，不识别其各项因果份额。',
'', '量化恒等式：观测有符号差 = 冻结预测差 + 差分预测误差。差分误差可为负，不能强制拆成三个非负百分比。CPU全空间评分检查见optimizer_correctness_tests.json；逐案原始JSON、路由及生成集见routing_gap_attribution.json与历史34原始候选。']
route_report += ['', '额外执行审计见routing_search_execution_audit.json：四例各执行42次MILP；A两例均12次正常最优、30次给定种子约束不可行，B两例均18次正常最优、24次种子约束不可行，没有时间/迭代上限停止（status=1）。这些ASR比例/容量种子约束只定义生成子问题，不覆盖完整合法路由域；子问题不可行不能证明未生成的全域路由不存在。']
(OUT/'routing_gap_attribution.md').write_text('\n'.join(route_report)+'\n')

expansion=[]
for method in ['CPU_U_ORACLE','20A_U_GRID','V3_J_GRID_CONTROL','U_ALIGNED_DP','MIN_MEMORY_COVER_FASTEST','FASTEST_PATH_STATIC']:
    previous=None
    for pool in ['A','B','C']:
        row=next(r for r in ds if r['scenario']=='TEXT_HALF_MID18' and r['pool']==pool and r['algorithm']==method)
        same=previous is not None and identity(row['deployment'],row['routing'])==identity(previous['deployment'],previous['routing'])
        expansion.append(dict(row,previous_pool=previous['pool'] if previous else None,decision='INITIAL' if previous is None else 'KEEP' if same else 'SWITCH',
            delta_predicted_U=row['predicted_U']-previous['predicted_U'] if previous else None,load='TEXT10 six positive tasks',domain='K1 binary measured replicas'))
        previous=row
for alg,before,after,keep in [('U_ALIGNED_DP',control['new_before']['winner'],control['new_after'],control['same_plan']),
                             ('20A_U_GRID',control['legacy_before']['winner'],None,control['legacy_same_plan'])]:
    expansion.append(dict(scenario='TEXT_HALF_MID18',pool='C_minus_Bart_to_C',algorithm=alg,decision='KEEP' if keep else 'SWITCH',
        deployment=before['deployment'],routing=before['routing'],predicted_U=before['prediction']['U'],oracle_before_U=control['oracle_before_U'],oracle_after_U=control['oracle_after_U'],
        scope='Actual admitted no-benefit Specialist addition; separate control, does not alter A/B/C'))
expansion.append(dict(scenario='TEXT_HALF_MID18',pool='C_minus_Bart_to_C',algorithm='CPU_U_ORACLE',decision='KEEP',predicted_U=control['oracle_before_U'],oracle_after_U=control['oracle_after_U']))
csvsave('pool_expansion_decisions.csv',expansion)

plotrows=[r for r in expansion if r['pool'] in ['A','B','C'] and r['algorithm'] in ['CPU_U_ORACLE','20A_U_GRID','U_ALIGNED_DP']]
fig,axes=plt.subplots(1,3,figsize=(10,3.1),sharey=True)
for ax,pool in zip(axes,['A','B','C']):
    for i,row in enumerate(r for r in plotrows if r['pool']==pool):
        g=sum(n for m,n in row['deployment'].items() if models[m]['role']=='generalist');s=sum(row['deployment'].values())-g
        ax.bar(i,g,color='#4169a1',label='Generalist' if i==0 else None);ax.bar(i,s,bottom=g,color='#dd9344',label='Specialist' if i==0 else None)
        ax.text(i,g+s+.15,f"U={row['predicted_U']:.3f}",ha='center',fontsize=8)
    ax.set_xticks(range(3),['Oracle','20A','U aligned']);ax.set_title(f'Pool {pool}');ax.set_ylim(0,6)
axes[0].set_ylabel('Resident replicas');axes[-1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1));fig.suptitle('TEXT10: autonomous deployment selection (CPU prediction)',y=1.02)
outputfig('deployment_structure_selection',fig,plotrows)

regrows=[r for r in ds if r['algorithm'] in ['20A_U_GRID','V3_J_GRID_CONTROL','U_ALIGNED_DP']]
fig,ax=plt.subplots(figsize=(10,3.5));cells=[(s,p) for s in sp['scenarios'] for p in ['A','B','C']]
for j,(alg,color) in enumerate([('20A_U_GRID','#c65a48'),('V3_J_GRID_CONTROL','#5d86ab'),('U_ALIGNED_DP','#4b9772')]):
    ax.bar(np.arange(9)+(j-1)*.24,[next(r['regret'] for r in regrows if r['scenario']==s and r['pool']==p and r['algorithm']==alg) for s,p in cells],width=.24,label=alg,color=color)
ax.set_xticks(range(9),[s.replace('TEXT_HALF_MID18','TEXT10')+'/'+p for s,p in cells],rotation=30,ha='right');ax.set_ylabel('Predicted U regret');ax.legend();ax.set_title('Same finite routing domain; native V3 reported separately')
outputfig('cpu_oracle_regret',fig,regrows+hs)

runtime=[]
for s,p in cells+[('HOLDOUT_TEXT3',p) for p in ['A','B','C']]:
    oo=[r for r in ts if r['scenario']==s and r['pool']==p and r['algorithm']=='CPU_U_ORACLE'];nn=[r for r in ts if r['scenario']==s and r['pool']==p and r['algorithm']=='U_ALIGNED_DP']
    ot=statistics.median(r['seconds'] for r in oo);nt=statistics.median(r['seconds'] for r in nn)
    runtime.append(dict(scenario=s,pool=p,candidate_space=oo[0]['candidate_space'],oracle_seconds_median=ot,new_seconds_median=nt,oracle_over_new_speedup=ot/nt,
        new_search_states=nn[0]['search_states'],worker_prediction_calls=nn[0].get('worker_prediction_calls'),full_prediction_calls=nn[0].get('full_prediction_calls'),
        timing_scope='Three CPU-only repeats, fresh caches; original Oracle also computes J diagnostic, new solver computes U. Not an asymptotic scaling theorem.'))
csvsave('solver_runtime_summary.csv',runtime)
fig,ax=plt.subplots(figsize=(6.5,3.6))
for col,label,color in [('oracle_seconds_median','U Oracle','#b85248'),('new_seconds_median','U aligned DP','#43836c')]:
    ax.scatter([r['candidate_space'] for r in runtime],[r[col] for r in runtime],label=label,color=color,s=35)
ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Finite configurations: 8 / 32 / 8154');ax.set_ylabel('Median solve seconds');ax.legend();ax.set_title('Same frozen predictor; three isolated CPU repeats')
outputfig('solver_runtime_scaling',fig,runtime)

gpu=[];contrasts=[];reuse=[]
if (OUT/'gpu_observations.json').exists() and read(OUT/'campaign_status.json').get('status')=='COMPLETED':
    oldf=read(OLD/'frozen_gpu_manifest_v2.json');oldslots={s['run_id']:s for s in oldf['slots']};newf=read(OUT/'frozen_gpu_manifest.json');newslots={s['run_id']:s for s in newf['slots']}
    fm=read(OLD/'frozen_problem_manifest.json');refs=read(fm['utility_reference']['path'])['latency_reference_ms'];weights=read(fm['utility_definition']['path'])['weights'];d=read(sp['pools']['C']['path'])
    from module26.common import arithmetic
    obs=[(o,oldslots[o['run_id']],True) for o in read(OLD/'gpu_observations.json') if oldslots[o['run_id']]['scenario']=='TEXT_HALF_MID18']
    obs += [(o,newslots[o['run_id']],False) for o in read(OUT/'gpu_observations.json')]
    by={}
    for o,s,reused in obs:
        raw=read(o['raw']['path']);arith=arithmetic(raw,d,refs,weights);same=arith.get('raw_arithmetic_U') is not None and (o['U'] is None or abs(arith['raw_arithmetic_U']-o['U'])<1e-10)
        assert ref(o['raw']['path'])['sha256']==o['raw']['sha256']
        row=dict(run_id=o['run_id'],pool=s['pool'],arm=s['arm'],scenario=s['scenario'],stage=s['stage'],seed=s['seed'],repetition=s['repetition'],
            deployment=s['deployment'],routing=s['routing'],predicted_U=s['predicted_U'],observed_U=o['U'],eligible=o['eligible'],technical_valid=o['technical_valid'],stable=o['stable'],
            prediction_error=o['U']-s['predicted_U'] if o['U'] is not None else None,reused_module34=reused,source_raw=o['raw']['path'],raw_sha256=o['raw']['sha256'],
            independent_U_arithmetic_match=same,postwarm_temperature_c=raw.get('resident_thermal_control',{}).get('samples',[{}])[-1].get('temperature_gpu_c'),
            mean_SM_clock_mhz=o.get('environment',{}).get('sm_clock_mhz'),mean_temperature_c=o.get('environment',{}).get('temperature_gpu_c'),
            mean_power_w=o.get('environment',{}).get('power_draw_w'),system_state=o.get('system_stability',{}).get('stability_state_v2'),worker_states={m:v['stability_state_v2'] for m,v in o.get('worker_stability',{}).items()})
        gpu.append(row);by[(s['pool'],s['arm'],s['seed'])]=row
        if reused:reuse.append(dict(run_id=o['run_id'],raw=o['raw'],same_formula=same,eligible=o['eligible'],protocol='Identical frozen inputs/arrival/route uniforms/hardware/budget/formula; actual sessions differ'))
    save('GPU_reuse_audit.json',reuse);csvsave('gpu_utility_comparison.csv',gpu)
    def stats(vals):
        if not vals:return dict(n=0,mean=None,ci95=None,p=None)
        mean=float(np.mean(vals));sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        half=tdist.ppf(.975,len(vals)-1)*sd/math.sqrt(len(vals)) if len(vals)>1 else None
        return dict(n=len(vals),mean=mean,sd=sd,ci95=[mean-half,mean+half] if half is not None else None,p=float(ttest_1samp(vals,0).pvalue) if len(vals)>1 else None)
    for pool,arm in [('B','ORACLE'),('B','COMPETITOR'),('A','ORACLE')]:
        pairs=[]
        for seed in [340201,340202,340203]:
            c,b=by.get(('C','SELECTED',seed)),by.get((pool,arm,seed))
            if not c or not b:continue
            clock=abs(c['mean_SM_clock_mhz']-b['mean_SM_clock_mhz'])/max(c['mean_SM_clock_mhz'],b['mean_SM_clock_mhz']) if c.get('mean_SM_clock_mhz') and b.get('mean_SM_clock_mhz') else None
            temp=abs(c['postwarm_temperature_c']-b['postwarm_temperature_c']) if c.get('postwarm_temperature_c') is not None and b.get('postwarm_temperature_c') is not None else None
            environment=clock is not None and clock<=.05 and temp is not None and temp<=5
            legal=c['eligible'] and b['eligible'];delta=c['observed_U']-b['observed_U'] if legal else None
            pred=c['predicted_U']-b['predicted_U']
            pairs.append(dict(seed=seed,legal_stable_both=legal,environment_comparable=environment,delta_U=delta,predicted_delta_U=pred,
                differential_prediction_error=delta-pred if delta is not None else None,postwarm_temperature_delta_c=temp,relative_SM_clock_delta=clock))
        st=stats([r['delta_U'] for r in pairs if r['legal_stable_both']]);complete=len(pairs)==3 and all(r['legal_stable_both'] and r['environment_comparable'] for r in pairs)
        contrasts.append(dict(contrast=f'C_SELECTED_minus_{pool}_{arm}',pairs=pairs,statistics=st,complete_and_comparable=complete,
            verdict='SUPPORTED_WITH_CROSS_SESSION_LIMITATION' if complete and st['ci95'][0]>0 else 'GPU_COMPARISON_INCONCLUSIVE',
            scope='Three paired common traces across reused34 and new35 sessions; time/order effects not fully controlled. Not full realGPU oracle.'))
    order=sorted(range(len(contrasts)),key=lambda i:contrasts[i]['statistics']['p'] if contrasts[i]['statistics']['p'] is not None else 1.)
    adjusted=0.
    for rank,i in enumerate(order):
        p=contrasts[i]['statistics']['p'];adjusted=max(adjusted,min(1.,(len(order)-rank)*p)) if p is not None else 1.
        contrasts[i]['holm_adjusted_p']=adjusted;contrasts[i]['holm_significant']=adjusted<.05 and contrasts[i]['complete_and_comparable']
    save('gpu_paired_contrasts.json',contrasts);csvsave('gpu_prediction_search_decomposition.csv',[dict(contrast=c['contrast'],**p) for c in contrasts for p in c['pairs']])
    gr=[]
    for pool,arm in [('A','ORACLE'),('A','COMPETITOR'),('B','ORACLE'),('B','COMPETITOR'),('C','SELECTED')]:
        rr=[r for r in gpu if r['pool']==pool and r['arm']==arm and r['stage']=='INDEPENDENT' and r['eligible']]
        vals=[r['observed_U'] for r in rr];st=stats(vals)
        gr.append(dict(pool=pool,arm=arm,**st,predicted_U=rr[0]['predicted_U'] if rr else None,
            chosen_by='CPU Oracle and U aligned' if arm=='ORACLE' else 'original20A' if arm=='COMPETITOR' else 'Oracle / original20A / U aligned / minimum-memory heuristic',
            caution='CIs of individual means are descriptive; differences use pairedCI. Cross-session reuse.'))
    fig,ax=plt.subplots(figsize=(8,3.6));idx=np.arange(len(gr))
    ax.bar(idx,[r['mean'] if r['mean'] is not None else 0 for r in gr],color=['#4169a1','#aeb7c6','#dd9344','#e6bf84','#4b9772'])
    for i,r in enumerate(gr):
        if r['ci95']:ax.errorbar(i,r['mean'],yerr=[[r['mean']-r['ci95'][0]],[r['ci95'][1]-r['mean']]],color='black',capsize=3)
    ax.set_xticks(idx,[r['pool']+'/'+r['arm'] for r in gr],rotation=15);ax.set_ylabel('Measured total system U (higher is better)');ax.set_title('Matched TEXT10 inputs: three independent 240s traces')
    outputfig('gpu_system_utility',fig,gr)

gap=any(r['algorithm']=='U_ALIGNED_DP' and (r['regret'] is None or abs(r['regret'])>1e-8 or r['eligible']!='True') for r in ds)
holdfail=any(r['algorithm']=='U_ALIGNED_DP' and (r['regret'] is None or abs(r['regret'])>1e-8 or r['eligible']!='True') for r in hs)
campaign=read(OUT/'campaign_status.json') if (OUT/'campaign_status.json').exists() else {'status':'NOT_STARTED'}
gpu_complete=campaign.get('status')=='COMPLETED' and len(gpu)==20
# B's own old20A versus new/Oracle route remains unresolved even when the
# much larger C expansion effect is supported. Preserve that negative gate.
gpu_inconclusive=True
history=read(OUT/'history_integrity_after.json') if (OUT/'history_integrity_after.json').exists() else {'passed':False,'status':'PENDING'}
selected_complete=sum(r['pool']=='C' and r['stage']=='INDEPENDENT' and r['eligible'] for r in gpu)==3
core=not gap and not holdfail and checks['all_passed'] and len(attrs)==4 and gpu_complete and selected_complete and history['passed']
gate=dict(module=35,gates={'ROUTING_DIAGNOSIS_INCOMPLETE':len(attrs)!=4,'U_ROUTING_SEARCH_GAP':gap,'CPU_ORACLE_MATCHED':not gap and checks['all_passed'],
    'HOLDOUT_GENERALIZATION_FAILED':holdfail,'GPU_COMPARISON_INCONCLUSIVE':gpu_inconclusive,'CORE_EVIDENCE_READY':core},
    stages={'A':'COMPLETE' if len(attrs)==4 else 'INCOMPLETE','B':'COMPLETE','C':'PASS' if not gap and not holdfail and checks['all_passed'] else 'FAIL',
        'D':'PASS' if read(OUT/'pool_nesting_validation.json')['passed'] and control['same_plan'] else 'FAIL',
        'E':campaign.get('status'),'F':'COMPLETE' if gpu_complete else 'CPU_FIGURES_READY_GPU_PENDING'},
    GPU_inconclusive_reason='TEXT10/B old20A versus Oracle/U aligned: existing paired95CI crosses0. A->B expansionCI also crosses0. New C contrast verdicts retained separately.',
    unresolved_new_GPU_contrasts=[c['contrast'] for c in contrasts if c['verdict']=='GPU_COMPARISON_INCONCLUSIVE'],
    holdout_scope='One independently frozen six-task workload across three pools; never tuned on holdout. Does not establish universal workload/general continuous domain generalization.',
    core_evidence_scope='Finite CPU correctness, actual nested-pool decisions, isolated CPU timings and limited comparable GPU chain; includes unresolved micro-differences.',
    historical_predictor_gate='Original20A UTILITY_MODEL_INSUFFICIENT remains unchanged',history_integrity=history,
    deadline_local='2026-10-16 Asia/Shanghai',no_next_module=True)
save('module35_gate.json',gate)
newrows=[r for r in ds if r['algorithm']=='U_ALIGNED_DP'];hnew=[r for r in hs if r['algorithm']=='U_ALIGNED_DP'];oldrows=[r for r in ds if r['algorithm']=='20A_U_GRID']
summary=['# Module 35：Utility-Aligned Routing Optimization and Candidate Pool Expansion Validation','',
f"核心证据状态：**{'CORE_EVIDENCE_READY' if core else '待GPU/历史核验完成'}**。CPU Oracle验收={'PASS' if not gap else 'FAIL'}，独立holdout={'PASS' if not holdfail else 'FAIL'}。GPU微差不确定Gate保留，原性能预测器不足Gate不升级。完成后停止，不执行下一Module。",'',
'## 七项结论','',
'1. **四组路由差异原因**：原20A线性MILP代理只生成有限候选，遗漏FIFO U更优的任务避让路由；生成集内评分/排序误差均为0，全部预测Regret属于候选遗漏。驻留与副本均一致，没有证据支持错误剪枝、提前终止或状态污染为这四例原因。ASR32B/A改ASR，ASR32B/B改LID，TEXT10/A改图像/ASR/VQA，TEXT10/B改图像/VQA。V3的J/U代理不一致和GPU预测误差单独报告。',
f"2. **同域CPU Regret**：新U优化器{sum(abs(r['regret'])<1e-8 for r in newrows)}/9匹配Oracle；旧20A{sum(abs(r['regret'])<1e-8 for r in oldrows)}/9。最大Regret从{max(r['regret'] for r in oldrows):.6f}降至{max(abs(r['regret']) for r in newrows):.9f}。四组在固定原部署的独立路由搜索中均修复。原V3、原20A源码不改；原生连续方法与有限域控制分别命名。", 
f"3. **Holdout**：预冻结TEXT3六任务负载在A/B/C三池{sum(abs(r['regret'])<1e-8 for r in hnew)}/3预测Regret为0，所有约束合法。模型/路由没有按holdout结果调参；这仅支持该冻结负载，不能宣称普遍泛化或连续域全局最优。",
'4. **候选池扩张自主决策**：TEXT10下A为Qwen+MiniCPM纯G；B已是Qwen+MiniCPM+Whisper-medium混合；C为Phi+Roberta+ConvNeXt+PARSeq+Whisper-small混合，均由新算法自主选择。A→B→C没有G-only保持阶段，不能人为制造。真实无益候选控制C去Bart→C中，Oracle效用不变，新算法与原20A保持完整部署及路由。A⊂B⊂C共用画像和约束，每个原合法方案在扩张池中仍合法。',
'5. **求解效率**：三次CPU单独计时、冷缓存。新算法通过模型任务子集缓存及Pareto动态规划减少完整预测，不逐项评价8154个完整配置。完整数值和所有重复在solver_runtime_comparison.csv；大小空间分别报告，不推导渐近定理。Oracle沿用34统一评价接口并计算J诊断，新算法直接优化U，因此比值包含避免冗余诊断及增量预测的收益。静态启发式生成时间不含其离线合法性验证，不能把该时间与搜索时间作公平预测调用比较。',
'6. **真实GPU**：A/B复用严格相同协议的34测量，C只补此前缺失的获选部署，所有方法选择相同C配置的实测共用，不伪造独立试验。每个请求轨迹配对，双方合法稳定才给正式U差。GPU具体支持程度见下表；历史A路由修复有支持，B路由修复差异无法可靠区分。',
'7. **论文适用范围**：可以写有限域正确性、四例失效归因、冻结holdout结果、自主候选池保持/切换、实际CPU求解效率和有限GPU配对结果。预测最优与真实最优分别写；不能写8154配置真实全局最优、普遍混合优势、普遍优于启发式、多副本/连续分流全空间最优或预测模型充分验证。', '',
'## CPU结果','', '|负载/池|旧20A Regret|V3网格 Regret|新 Regret|Oracle中位秒|新中位秒|Oracle/新|', '|---|---:|---:|---:|---:|---:|---:|']
for rr in runtime:
    s,p=rr['scenario'],rr['pool'];dest=hs if s.startswith('HOLD') else ds
    vals=[next(r['regret'] for r in dest if r['scenario']==s and r['pool']==p and r['algorithm']==alg) for alg in ['20A_U_GRID','V3_J_GRID_CONTROL','U_ALIGNED_DP']]
    summary.append(f"|{s}/{p}|{vals[0]:.6f}|{vals[1]:.6f}|{vals[2]:.9f}|{rr['oracle_seconds_median']:.5f}|{rr['new_seconds_median']:.5f}|{rr['oracle_over_new_speedup']:.2f}×|")
summary += ['', '全空间bundle U独立重组误差最大='+f"{max(c['max_nominal_U_error'] for c in checks['cells']):.3g}"+'，合法性不一致数='+str(sum(c['eligibility_mismatches'] for c in checks['cells']))+'。固定部署K=2分流和历史实测多副本画像验收也通过；K>1联合驻留/分流搜索并未获得全局验收。', '',
'## 实现、计数及不确定性','',
'u_aligned_optimizer.py是独立新版本，不依赖Oracle胜者或原始候选表。K=1每任务只走一个模型，所以任务均值/P95可按模型任务子集精确累加；within-window完成数可累加，总T项保持原定义。驻留闲置模型计显存；联合选择时仅移除可证明不改善U的闲置驻留。固定部署接口仍保留闲置驻留。副本只允许有原实测容量/显存画像的计数。', '',
'相同模型前缀和任务覆盖mask下，较低显存/非吞吐效用成本且较多窗口完成数支配另一状态；余下任务可接同样后缀，故不删除潜在更优解。未用J剪U候选。worker_prediction_calls是单模型任务子集的名义FIFO计算，full_prediction_calls是原predict的分量次数（四场景），search_states是实际扩展DP标签或分流搜索节点，transitions另列；这些计数不可混为同一种调用。冷搜索cache_hits=0是正常情况，全候选校验及重复分流测试实际命中另留档。', '',
'K=2固定部署搜索保留完整离散路由，只有部分rho和保守服务时间下界剪枝；下界可能弱，退化风险明确。不宣称其在大连续路由空间高效。压力场景结果仍保留，stress bounds不是统计置信区间。预测不合法或缺失任务返回无方案/明确拒绝，不赋有利U。', '',
'本次K=2正确性小测试评价了全部162条合法固定部署路由，bound_pruned=0。此项是正确性证据，不能计入高效分流搜索证据。主要效率结论严格限定K=1联合部署/路由动态规划。原生V3仍存在网格外结果及数值失败，原生方法不满足完全相同内部域；其终点合法性/Regret和V3_J_GRID_CONTROL分别留档。', '',
'## GPU冻结与结果','',
'协议在C结果揭晓前冻结：仅C/TEXT10获选配置，30s warmup、开发120s、独立240s×3、drain≤60s，GPU同UUID/驱动/49140MiB预算，复用34的原样本、同arrival轨迹、route/replica uniforms、质量/稳定性及效用权重。三个预定对照C−B Oracle、C−B原20A、C−A Oracle，用共享seed配对并给Holm校正。温控/时钟检查阈值写入frozen_gpu_protocol.json。跨历史批次的时间/次序效应未随机化，必须保留限制。', '',
'|GPU对照|有效配对|实测均值差|配对95%CI|Holm p|结论|', '|---|---:|---:|---|---:|---|']
for c in contrasts:
    st=c['statistics'];ci=st['ci95'];mean=f"{st['mean']:.6f}" if st['mean'] is not None else 'NULL';summary.append(f"|{c['contrast']}|{st['n']}|{mean}|{ci}|{c['holm_adjusted_p']:.6g}|{c['verdict']}|")
summary += ['', 'Module34原A新路由−旧路由：ΔU=21.658291，配对95%CI=[15.152162,28.164419]；B相同对照ΔU=0.085794，CI=[−2.710301,2.881888]；A→B获选部署扩张ΔU=11.097327，CI=[−12.607954,34.802608]。后两项无法可靠区分，原结论保留。', '',
'本轮C−B对照的窗口平均SM时钟相差约4.2%–4.5%，最终postwarm温度差2°C，均通过事前冻结的5%/5°C阈值；原始温度、功率及时钟单独列入CSV。阈值通过不消除跨批次时间/次序残余混杂，C的优势只能按此限制解读。', '',
'C池最小显存启发式、旧20A与新算法选相同方案，实测U相同是配置身份关系，不证明新算法超过这些基线。未测C最快路径方案，不能声称C内细小排名已验证。没有追加8154配置实测。', '',
'## 论文图表与原始记录','',
'paper_figures/包含部署结构、同域Regret、求解时间/空间规模、关键GPU总U四图，每图同时保存PNG、PDF及CSV。cpu/存每格新旧算法输出、完整Oracle候选和原始单次计时；gpu_utility_comparison.csv、gpu_paired_contrasts.json及原raw/终端日志保留失败、合法性、稳定性和预测误差。保存历史SHA256与mtime全检（含模型权重），GPU子进程受新目录写入隔离。', '',
'## Gate与停止','',json.dumps(gate['gates'],ensure_ascii=False,indent=2), '',
'成功的CPU/holdout/模型池阶段与GPU不确定性同时记录，不互相覆盖。冻结后停止，不启动下一Module。复现CPU请将OUT改成新隔离目录，并冻结新的输入manifest；不得覆写已冻结CPU/GPU原始记录。']
(OUT/'module35_summary.md').write_text('\n'.join(summary)+'\n')
save('report_status.json',dict(gpu_complete=gpu_complete,core_evidence_ready=core,figures=list(p.name for p in figs.glob('*.pdf')),at_unix_ns=time.time_ns()))
print('REPORT',core,flush=True)
