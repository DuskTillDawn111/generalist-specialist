"""Publication-style, provenance-bound figures from existing results only.

No imports from the optimizer or predictor. No new measurements, optimization,
reference solves, smoothing, confidence intervals or invented missing results.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts'
LEGACY = ART/'bilevel-scalability-ablation-20261105'
CURRENT = ART/'bilevel-scheduler-improvement-20261105'
SEEDS = (11,29,47)
COLORS = dict(legacy='#0072B2',balanced='#D55E00',green='#009E73',purple='#CC79A7',
              lightblue='#56B4E9',yellow='#E69F00',gray='#777777')
SOURCES = {}
FIGURES = []
DATA = []


def receipt(path):
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    path = Path(path)
    SOURCES[str(path.resolve())] = receipt(path)
    return json.loads(path.read_text())


def setup():
    plt.rcParams.update({
        'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':9,'axes.labelsize':9,'axes.titlesize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.5,
        'axes.linewidth':.65,'lines.linewidth':1.3,'lines.markersize':4,
        'xtick.direction':'in','ytick.direction':'in','xtick.major.width':.65,'ytick.major.width':.65,
        'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':True,
        'grid.linewidth':.4,'grid.alpha':.3,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','savefig.facecolor':'white','hatch.linewidth':.45,
    })


def panels(n=2,height=2.55):
    fig,axs=plt.subplots(1,n,figsize=(7.05,height),layout='constrained')
    return fig,np.atleast_1d(axs)


def style(ax,title,ylabel=None):
    ax.set_title(title,loc='left',pad=7)
    if ylabel:ax.set_ylabel(ylabel)
    ax.grid(axis='y',zorder=0)
    ax.set_axisbelow(True)


def row(figure,**values):
    DATA.append(dict(figure=figure,**values))


def savefig(out,fig,name,caption,caption_zh,reference,evidence):
    for fmt in ('pdf','svg','png'):
        fig.savefig(out/f'{name}.{fmt}',dpi=600 if fmt=='png' else None)
    FIGURES.append(dict(name=name,caption=caption,caption_zh=caption_zh,
                        reference=reference,evidence=evidence,size_inches=list(fig.get_size_inches())))
    plt.close(fig)


def median_range(ax,x,values,color,marker='o',offset=0,label=None):
    med=np.array([statistics.median(v) for v in values])
    lo=np.array([min(v) for v in values]);hi=np.array([max(v) for v in values])
    ax.errorbar(np.array(x)+offset,med,yerr=[med-lo,hi-med],color=color,marker=marker,
                capsize=2.5,linewidth=1.2,label=label,zorder=3)


def raw_points(ax,x,values,color,marker='o'):
    for xx,vals in zip(x,values):
        jitter=np.linspace(-.085,.085,len(vals))
        ax.scatter(xx+jitter,vals,marker=marker,color=color,s=16,zorder=4,linewidths=.5)
        ax.plot([xx-.13,xx+.13],[statistics.median(vals)]*2,color=color,lw=1.5,zorder=5)


def hardware(out):
    a=read(ART/'utility-tradeoffs-v3-20261105/results.json')
    assert a['default']['profile']=='four_groups_v3' and a['default']['deadline_multiplier']==2
    ids=['candidate-1','candidate-2','candidate-3']
    labels=['Pure specialists','Off-grid control','50% to G2']
    fig,ax=panels(height=2.65)
    for k,(kind,color,marker,offset) in enumerate((('observed',COLORS['legacy'],'o',-.10),('predicted',COLORS['balanced'],'s',.10))):
        values=[]
        for cid in ids:
            episodes=[e for e in a['episodes'] if e['candidate_id']==cid]
            us=[e['default_scores'][kind]['U'] for e in episodes]
            assert np.isclose(np.mean(us),a['default']['assessment']['mean_U'][kind][cid])
            for e,u in zip(episodes,us):row('fig01_hardware',panel='a',candidate=cid,round=e['round'],kind=kind,U=u)
            values.append(us)
        # Two actual history points and their mean; no population interval.
        for i,vals in enumerate(values):
            ax[0].plot([i+offset]*2,vals,color=color,lw=.7,alpha=.55)
            ax[0].scatter([i+offset]*2,vals,color=color,marker=marker,s=14,zorder=3)
            ax[0].plot([i+offset-.06,i+offset+.06],[np.mean(vals)]*2,color=color,lw=1.8)
        ax[0].plot([],[],color=color,marker=marker,label=kind.capitalize(),linestyle='none')
    ax[0].set_xticks(range(3),labels,rotation=12)
    ax[0].set_ylim(top=0)
    ax[0].legend(frameon=False,loc='lower left')
    style(ax[0],'(a) Fixed-residency candidate utility',r'$U_{v3}$ (higher is better)')
    tasks=['asr','image','ocr','text'];x=np.arange(4);width=.35
    for rnd,color,hatch,shift in ((1,COLORS['legacy'],'',-width/2),(2,COLORS['balanced'],'//',width/2)):
        e=next(e for e in a['episodes'] if e['candidate_id']=='candidate-1' and e['round']==rnd)
        ts=e['metrics_by_deadline_multiplier']['2']['observed']['per_task']
        vals=[100*ts[t]['timely_count']/ts[t]['offered'] for t in tasks]
        ax[1].bar(x+shift,vals,width,color=color,edgecolor='black',linewidth=.4,hatch=hatch,label=f'History {rnd}')
        for i,t in enumerate(tasks):
            count,total=ts[t]['timely_count'],ts[t]['offered']
            ax[1].text(i+shift,vals[i]+2,f'{count}/{total}',ha='center',fontsize=6.6)
            row('fig01_hardware',panel='b',task=t,round=rnd,timely=count,offered=total,percent=vals[i])
    ax[1].set_xticks(x,['ASR','Image','OCR','Text']);ax[1].set_ylim(0,140);ax[1].set_yticks([0,25,50,75,100])
    ax[1].legend(frameon=False,loc='upper center',ncol=2)
    style(ax[1],'(b) Selected route: deadline attainment','On-time requests (%)')
    savefig(out,fig,'fig01_hardware',
        'Physical evidence, retrospectively rescored with v3. Five models remain resident, one replica each. '
        'Panel (a) shows the two actual arrival histories and their means; predictions use matching histories. '
        'Panel (b) shows the pure-specialist route at the frozen default deadline: OCR is 0/24 and image 3/24 in history 2. '
        'These are candidate comparisons, not independent v3 confirmation or an SLO pass.',
        '真实五模型固定驻留结果，按 v3 回顾性重评。左图为两个历史的实测/匹配预测及均值，非置信区间；右图为纯专用路由按时请求数。第二历史 OCR 0/24、image 3/24，不能宣称 SLO 达标。',
        'MuxServe Fig. 5 (PDF p.7); AlpaServe Figs. 12–13 (PDF pp.12–13).', 'physical_retrospective_v3')


def real_search(out):
    a=read(ART/'bilevel-final-acceptance-20261105/outer-comparison.json')
    methods=['BO-MADS','BO-only','MADS-only'];labels=['BO–MADS','BO-only','MADS-only']
    fig,ax=panels()
    us=[];calls=[]
    for m in methods:
        rr=[r for r in a['rows'] if r['method']==m]
        assert len(rr)==3 and all(r['improvement_after_initialization']==0 for r in rr)
        us.append([r['U'] for r in rr]);calls.append([r['budget']['full_prediction_calls'] for r in rr])
        for r in rr:row('fig02_scoped_search',method=m,seed=r['seed'],U=r['U'],initialization_U=r['initialization_U'],predictions=r['budget']['full_prediction_calls'])
    raw_points(ax[0],range(3),us,COLORS['legacy'])
    ax[0].axhline(us[0][0],ls='--',color='black',lw=.8,label='Common initialization')
    ax[0].set_ylim(-.60,-.45);ax[0].legend(frameon=False,loc='lower left')
    ax[0].text(1,-.49,'All final U = −0.517922',ha='center',fontsize=8)
    style(ax[0],'(a) Frozen five-model search',r'Predicted $U_{v3}$')
    raw_points(ax[1],range(3),calls,COLORS['legacy'])
    ax[1].set_ylim(0,68);ax[1].axhline(64,color='gray',ls=':',lw=.8)
    style(ax[1],'(b) Actual complete predictions','Prediction calls')
    for a0 in ax:a0.set_xticks(range(3),labels)
    savefig(out,fig,'fig02_scoped_search',
        'Frozen real-model scope, legacy scheduler. Three optimizer seeds share exactly the same charged initialization '
        'and budget maxima. All final utilities equal the initialization result; raw seed points and median ticks are shown. '
        'The newer balanced scheduler was not rerun against this frozen real predictor. Lower call counts do not establish a utility advantage.',
        '真实五模型冻结预测器、旧调度下的共同初始化对照。三个种子均在初始化得到最终最好值，BO、MADS、组合的效用打平；散点为原始种子，横线为中位数。新调度未在此真实范围重跑。',
        'MuxServe Fig. 8 (PDF p.8): controlled method panels; raw small-sample points are our statistical choice.', 'frozen_real_predictor_search_legacy')


def synthetic(out):
    legacy=read(LEGACY/'summary.json')['rows']
    balanced=read(CURRENT/'comparison.json')['rows']
    fig,ax=panels()
    for kind,rows,ns,color,marker in [('Legacy',legacy,[3,7,14],COLORS['legacy'],'o'),('Balanced',balanced,[7,14],COLORS['balanced'],'s')]:
        groups=[[r for r in rows if r['size']==n and (kind=='Balanced' or r['method']=='BO-MADS')] for n in ns]
        median_range(ax[0],ns,[[r['budget']['full_prediction_calls'] for r in g] for g in groups],color,marker,label=kind)
        median_range(ax[1],ns,[[r['regret_upper'] for r in g] for g in groups],color,marker,label=kind)
        for g in groups:
            for r in g:row('fig03_scaling',scheduler=kind,size=r['size'],seed=r['seed'],predictions=r['budget']['full_prediction_calls'],regret_upper=r['regret_upper'])
    for a0 in ax:a0.set_xticks([3,7,14]);a0.set_xlabel('Number of synthetic models');a0.legend(frameon=False,loc='upper left')
    ax[0].set_ylim(0,67);ax[0].axhline(64,color='gray',ls=':',lw=.8)
    ax[1].set_ylim(0,.013);ax[1].ticklabel_format(axis='y',style='sci',scilimits=(-3,-3),useMathText=True)
    style(ax[0],'(a) Budgeted scaling cost','Complete prediction calls')
    style(ax[1],'(b) Solution quality vs. reference',r'$U^{ref}_{upper}-U_{best}$ (lower is better)')
    savefig(out,fig,'fig03_scaling',
        'Manufactured convex queue-cost benchmark: medians and min–max ranges over three seeds. Balanced results exist only at 7 and 14 models; '
        'the missing 3-model result is not interpolated. Two tasks and a two-replica memory limit yield 7/33/117 feasible deployments at '
        '3/7/14 models. References have numerical intervals narrower than 2.2e-9. This is sparse catalog scaling, not dense 14-model residency.',
        '合成平滑排队成本场景，点为三个种子中位数，误差线为最小—最大值（非置信区间）。新调度只有 7/14 模型数据；固定两任务、最多两个副本，合法部署为 7/33/117 个，不能当作密集大部署验证。',
        'Cascadia Fig. 10 (PDF p.9): separate algorithm scale/cost panels; our x-axis is model count, not GPU count.', 'manufactured_convex_queue')
    return legacy,balanced


def convergence(out):
    fig,ax=panels(height=2.6)
    for panel,n in enumerate((7,14)):
        for name,folder,stem,color,marker in [('Legacy',LEGACY,'BO-MADS',COLORS['legacy'],'o'),('Balanced',CURRENT,'balanced',COLORS['balanced'],'s')]:
            curves=[]
            for seed in SEEDS:
                raw=read(folder/f'{n}-{stem}-{seed}.json')
                initial=raw['initial_signature'][-1]['work']['best_feasible_utility']
                h=raw['convergence'];assert len(h)==32
                vals=[]
                for k in range(4,33):
                    value=max(r['best_feasible_utility'] for r in h if r['outer_evaluations']<=k)
                    vals.append(value-initial)
                    row('fig04_convergence',size=n,scheduler=name,seed=seed,outer=k,gain=value-initial)
                assert all(b>=a-1e-12 for a,b in zip(vals,vals[1:]))
                curves.append(vals)
            arr=np.array(curves);xs=np.arange(4,33)
            ax[panel].fill_between(xs,arr.min(axis=0),arr.max(axis=0),step='post',color=color,alpha=.12,lw=0)
            ax[panel].step(xs,np.median(arr,axis=0),where='post',color=color,label=name,marker=marker,markevery=7,ms=3.4)
        ax[panel].set_xlim(4,32);ax[panel].set_xticks([4,8,12,16,20,24,28,32]);ax[panel].set_ylim(0,.009)
        ax[panel].ticklabel_format(axis='y',style='sci',scilimits=(-3,-3),useMathText=True)
        ax[panel].set_xlabel('Charged route evaluations (including refinement)')
        ax[panel].legend(frameon=False,loc='upper left')
        style(ax[panel],f'({chr(97+panel)}) {n}-model synthetic pool',r'$U_{best}-U_{initial}$')
    savefig(out,fig,'fig04_convergence',
        'Anytime improvement after the four identical charged initialization routes. Step lines are seed medians; '
        'shading is the observed seed range, not a confidence band. All evaluations, including repeated inner refinement, '
        'count on the x-axis. The prediction maxima are equal, but balanced uses 1–5 more actual predictions. '
        'No smoothing, time extrapolation or additional runs are used.',
        '共同计费初始化之后的效用改善。阶梯线为三个种子中位数，阴影为种子范围；横轴包含重复精化的路由评价次数。两版预测上限相同，新版实际预测多 1—5 次；不据此声称更快。',
        'MuxServe Fig. 7 (PDF p.8): paired small-multiple line plots; the anytime axis is specific to this project.', 'manufactured_convex_queue')


def allocation(out):
    categories=['Initialization','EI','Activation','MADS','Refinement']
    cs=[COLORS['gray'],COLORS['balanced'],COLORS['lightblue'],COLORS['legacy'],COLORS['green']]
    hatches=['','//','..','\\\\','xx']
    values=[];labels=[]
    for n in (7,14):
        for name,folder,stem in [('Legacy',LEGACY,'BO-MADS'),('Balanced',CURRENT,'balanced')]:
            samples=[]
            for seed in SEEDS:
                raw=read(folder/f'{n}-{stem}-{seed}.json')
                counts=Counter(e['source'] for e in raw['convergence'])
                assert set(counts)<={'initialization','bo_expected_improvement','activation_exploration','mads_poll','inner_promotion','mads_inner_refinement','pool_refresh'}
                v=[counts['initialization'],counts['bo_expected_improvement'],counts['activation_exploration'],counts['mads_poll'],
                   sum(counts[k] for k in ('inner_promotion','mads_inner_refinement','pool_refresh'))]
                assert sum(v)==32
                samples.append(v)
                for category,count in zip(categories,v):row('fig05_budget',size=n,scheduler=name,seed=seed,category=category,evaluations=count)
            values.append(np.mean(samples,axis=0));labels.append(f'{n} models / {name}')
    vals=np.array(values);fig,ax=panels(n=1,height=2.35);a=ax[0];left=np.zeros(4)
    for j,(name,color,hatch) in enumerate(zip(categories,cs,hatches)):
        a.barh(np.arange(4),vals[:,j],left=left,color=color,edgecolor='white',linewidth=.5,height=.56,hatch=hatch,label=name)
        for i,value in enumerate(vals[:,j]):
            if value>1.5:a.text(left[i]+value/2,i,f'{value:.1f}',ha='center',va='center',fontsize=8,color='white' if j in (0,1,3,4) else 'black')
        left+=vals[:,j]
    assert np.allclose(left,32)
    a.set_yticks(range(4),labels);a.invert_yaxis();a.set_xlim(0,32);a.set_xticks(range(0,33,4))
    a.set_xlabel('Mean charged evaluations (three seeds; total = 32)')
    a.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=5,frameon=False)
    a.grid(axis='x')
    savefig(out,fig,'fig05_budget',
        'Evaluation allocation by recorded source, averaged over three seeds. Each bar totals 32, including initialization. '
        'Legacy spends no evaluations on EI at either scale; balanced executes 7–9 EI proposals per run. '
        'Refinement includes promotion, MADS inner refinement and pool refresh. Source labels do not imply every MADS evaluation is a distinct route.',
        '按日志来源分类的路由评价预算，三个种子取均值，每条总计 32 次。旧版 EI 为零，新版每次 7—9 次；精化包括 promotion、局部内层精化及池刷新。MADS 类别不等于全部为不同路由。',
        'MuxServe Fig. 9 (PDF p.8): compositional allocations; adapted here to charged optimization work.', 'manufactured_convex_queue')


def ablation(out,legacy):
    variants=['BO-only','MADS-only','Grid-K4','No-activation','No-pool','No-progressive']
    labels=['No MADS','No EI','Grid K4','No activation','No pool','Exact inner']
    fig,ax=panels(height=2.7)
    marks=['o','s','^'];seedcolors=[COLORS['legacy'],COLORS['balanced'],COLORS['green']]
    for si,seed in enumerate(SEEDS):
        full=next(r for r in legacy if (r['size'],r['method'],r['seed'])==(7,'BO-MADS',seed))
        delta=[];ratio=[]
        for method in variants:
            r=next(r for r in legacy if (r['size'],r['method'],r['seed'])==(7,method,seed))
            d=r['U']-full['U'];q=r['budget']['full_prediction_calls']/full['budget']['full_prediction_calls']
            delta.append(d);ratio.append(q)
            row('fig06_legacy_ablation',variant=method,seed=seed,delta_U_variant_minus_full=d,prediction_ratio_variant_over_full=q)
        for a,v in zip(ax,(delta,ratio)):
            a.scatter(np.arange(6)+(si-1)*.12,v,s=22,marker=marks[si],color=seedcolors[si],label=f'Seed {seed}',zorder=4,linewidths=.5)
    ax[0].axhline(0,color='gray',ls='--',lw=.8);ax[1].axhline(1,color='gray',ls='--',lw=.8)
    for a in ax:a.set_xticks(range(6),labels,rotation=30,ha='right');a.margins(x=.07)
    ax[0].ticklabel_format(axis='y',style='sci',scilimits=(-3,-3),useMathText=True)
    style(ax[0],'(a) Paired quality effect (legacy)',r'$U_{variant}-U_{full}$')
    style(ax[1],'(b) Paired prediction cost (legacy)','Calls: variant / full')
    handles,labels2=ax[0].get_legend_handles_labels()
    fig.legend(handles,labels2,frameon=False,ncol=3,loc='outside upper center',fontsize=7.5)
    savefig(out,fig,'fig06_legacy_ablation',
        'Historical 7-model ablations under the LEGACY scheduler only. Each point is a paired seed difference or cost ratio. '
        'Positive utility differences favor the variant; cost ratios above one favor full legacy. No-activation loses improvement, '
        'whereas pool/progressive mechanisms trade some solution quality for fewer calls. Grid K4 is a control, not a component ablation. '
        'These results must not be presented as balanced-scheduler component contributions; those experiments have not been run.',
        '仅为旧调度下的 7 模型消融：每个点是同种子差值/调用比。左图正值表示对照更好，右图大于 1 表示完整旧版更省调用；Grid K4 是基线对照。新调度完整消融尚未运行，不能用此图代替。',
        'Cascadia Fig. 9 (PDF p.9); MuxServe Fig. 8 (PDF p.8): controlled component comparisons.', 'manufactured_convex_queue_legacy_only')


def inner(out):
    report=read(ART/'bilevel-final-acceptance-20261105/inner-comparison.json')
    policies=['fresh_exact_milp','shared_exact','shared_budget'];names=['Fresh exact','Shared exact','Shared budget']
    cs=[COLORS['gray'],COLORS['legacy'],COLORS['balanced']];hatches=['','//','..']
    fig,ax=panels(height=2.6);width=.22;x=np.arange(2)
    for j,(policy,name,color,hatch) in enumerate(zip(policies,names,cs,hatches)):
        values=[next(p for p in r['policies'] if p['policy']==policy) for r in report['reports']]
        for panel,metric in enumerate(('full_prediction_calls','milp_solve_calls')):
            counts=[v['budget'][metric] for v in values]
            ax[panel].bar(x+(j-1)*width,counts,width,color=color,edgecolor='black',lw=.4,hatch=hatch,label=name)
            for i,v in enumerate(values):
                cert=sum(r['certified'] for r in v['rows'])
                text=f'{counts[i]}' + (f'\n({cert}/6)' if panel==0 else '')
                ax[panel].text(i+(j-1)*width,counts[i]+1.2,text,ha='center',va='bottom',fontsize=7)
                row('fig07_inner',workload=report['reports'][i]['workload'],policy=policy,metric=metric,calls=counts[i],certified=cert,visits=6)
    for a in ax:a.set_xticks(x,['Low','Text-hot']);a.set_ylim(0,100)
    handles,labels2=ax[0].get_legend_handles_labels()
    fig.legend(handles,labels2,frameon=False,ncol=3,loc='outside upper center',fontsize=7.5)
    style(ax[0],'(a) Complete predictions; certified visits','Prediction calls')
    style(ax[1],'(b) MILP master calls','MILP calls')
    savefig(out,fig,'fig07_inner',
        'Earlier three-model SYNTHETIC FIFO fixture with its ORIGINAL objective, not the manufactured convex benchmark or real v3 utility. '
        'Counts aggregate six fixed-route visits per workload, including one repeated route. Parentheses show certified visits out of six. '
        'All 24 exact outputs agree with independent enumeration; 12 budgeted outputs keep valid bounds and remain uncertified. '
        'Savings combine shared structures, valid bounds and scheduling; this is not isolated pool attribution.',
        '早期三模型合成 FIFO 场景，使用其原目标，不能与其他图的 U 混比。每负载六次路由访问含一次重复；括号为已认证数。24 个精确结果与枚举一致，12 个预算结果仍未认证。节省来自池、界和调度的组合。',
        'Cascadia Fig. 9 (PDF p.9): grouped, hatched component bars, with explicit certification added here.', 'synthetic_FIFO_original_objective')


def routes(out):
    fig,ax=panels(height=2.9)
    models=['g','s_shared','s_text','x00'];names=['g','Shared S','Text S','x00']
    cs=[COLORS['legacy'],COLORS['yellow'],COLORS['green'],COLORS['purple']]
    hatches=['','//','..','xx'];rows=[];labels=[]
    for seed in SEEDS:
        for label,folder,method in [('Legacy',LEGACY,'BO-MADS'),('Balanced',CURRENT,'balanced')]:
            raw=read(folder/f'14-{method}-{seed}.json');rows.append(raw['winner']['routing_shares']);labels.append(f'{seed} / {label}')
    for panel,task in enumerate(('text','image')):
        left=np.zeros(6)
        for m,name,color,hatch in zip(models,names,cs,hatches):
            vals=np.array([r[task].get(m,0.) for r in rows])*100
            ax[panel].barh(range(6),vals,left=left,color=color,edgecolor='white',lw=.45,height=.65,hatch=hatch,label=name)
            for i,v in enumerate(vals):
                if v>=9:ax[panel].text(left[i]+v/2,i,f'{v:.1f}',ha='center',va='center',fontsize=7.1,color='white' if m in ('g','s_text') else 'black')
                row('fig08_routes',size=14,seed=SEEDS[i//2],scheduler='Legacy' if i%2==0 else 'Balanced',task=task,model=m,share=v/100)
            left+=vals
        assert np.allclose(left,100)
        ax[panel].set_yticks(range(6),labels if panel==0 else [])
        ax[panel].invert_yaxis();ax[panel].set_xlim(0,100);ax[panel].set_xticks([0,25,50,75,100]);ax[panel].set_xlabel('Task routing share (%)')
        style(ax[panel],f'({chr(97+panel)}) {task.capitalize()} routes, 14 models')
        ax[panel].grid(axis='x',ls=':',lw=.5)
    handles,labels2=ax[0].get_legend_handles_labels()
    fig.legend(handles,labels2,loc='outside upper center',ncol=4,frameon=False)
    savefig(out,fig,'fig08_routes',
        'All three paired 14-model routing outcomes, including the non-improving-on-grid case for seed 47. '
        'Balanced seeds 11 and 29 select text-to-shared-specialist shares of 14.1104% and 20.3406%; EI directly improved the incumbent in these runs. '
        'Percentages are rounded for display. These are model-routing fractions in a synthetic problem, not measured hardware throughput shares or a proof of BO–MADS synergy.',
        '完整列出 14 模型三个种子的前后路由，而非只展示最好的一个。新版种子 11/29 的 text→共享专用比例为 14.1104%/20.3406%，来自 EI 的直接改善；图内按一位小数展示。属于合成问题，不是硬件吞吐占比。',
        'Cascadia §4.3 routing/resource case study; MuxServe Fig. 9 (PDF p.8): compositional allocation views.', 'manufactured_convex_queue')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Sealed figure bundle; choose a new directory')
    setup()
    hardware(out);real_search(out);legacy,balanced=synthetic(out)
    convergence(out);allocation(out);ablation(out,legacy);inner(out);routes(out)
    keys=list(dict.fromkeys(k for r in DATA for k in r))
    with (out/'plot-data.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(DATA)
    (out/'plot-data.json').write_text(json.dumps(DATA,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'captions.json').write_text(json.dumps(FIGURES,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'sources.json').write_text(json.dumps(list(SOURCES.values()),indent=2)+'\n')
    # Assemble a readable vector PDF, with real prose captions outside the panels.
    import pymupdf as fitz
    from matplotlib.font_manager import findfont
    report_font = findfont('DejaVu Serif')
    book=fitz.open()
    page=book.new_page(width=595,height=842)
    page.insert_font(fontname='reportserif',fontfile=report_font)
    title='Generalist–Specialist Joint Optimization\nResearch Figure Collection'
    page.insert_textbox(fitz.Rect(50,70,545,180),title,fontsize=21,fontname='reportserif',align=1)
    intro=('Status: implementation and bounded evaluations are complete; broader research claims remain open.\n\n'
        'Evidence gaps: (1) real seven-model deployment/replica benefits; (2) full component ablation under the new balanced scheduler; '
        '(3) cross-workload/generalization evidence and business SLO attainment.\n\n'
        'This collection reuses existing records only. No new experiments, predictor calls or optimization runs were performed for plotting. '
        'Physical records, frozen-predictor searches, synthetic convex benchmarks and the earlier synthetic FIFO fixture are kept distinct.\n\n'
        'Statistics: n=3 optimizer seeds or n=2 actual arrival histories. All ranges show observed minima/maxima, never confidence intervals. '
        'Error bars and raw dots are project-specific choices, not claims about the cited papers.\n\n'
        'Design references reviewed from original PDFs: MuxServe (ICML 2024), Figs. 5, 7–9, PDF pp.7–8; '
        'AlpaServe (OSDI 2023), Figs. 12–13, PDF pp.12–13; Cascadia (ICLR 2026), Figs. 9–10, PDF p.9. '
        'We borrow visual organization, not results or statistical evidence.\n\n'
        'Exports: eight vector PDF/SVG figures and 600-dpi PNGs; source script, data, captions and provenance included. '
        'English axes support manuscript use; Chinese captions and a gap assessment accompany the bundle.')
    assert page.insert_textbox(fitz.Rect(55,210,540,770),intro,fontsize=11,fontname='reportserif',lineheight=1.35)>=0
    for i,meta in enumerate(FIGURES,1):
        page=book.new_page(width=595,height=842)
        page.insert_font(fontname='reportserif',fontfile=report_font)
        page.insert_text((45,45),f'Figure {i} | {meta["evidence"].replace("_"," ")}',fontname='reportserif',fontsize=11)
        original=fitz.open(out/f'{meta["name"]}.pdf')
        height=original[0].rect.height/original[0].rect.width*510
        page.show_pdf_page(fitz.Rect(42,80,552,80+height),original,0)
        original.close()
        caption=f'Figure {i}. '+meta['caption']+'\n\nVisual reference: '+meta['reference']
        assert page.insert_textbox(fitz.Rect(50,110+height,545,780),caption,fontsize=10.5,fontname='reportserif',lineheight=1.3)>=0
        page.insert_text((285,815),str(i),fontname='reportserif',fontsize=9)
    book.save(out/'research_figures.pdf');book.close()
    # Small browsing sheet; no replacement of vector publication outputs.
    from PIL import Image,ImageOps,ImageDraw
    canvas=Image.new('RGB',(1600,1800),'white');draw=ImageDraw.Draw(canvas)
    for i,meta in enumerate(FIGURES):
        im=Image.open(out/f'{meta["name"]}.png').convert('RGB')
        im.thumbnail((780,390))
        xx=10+(i%2)*800;yy=20+(i//2)*450
        draw.text((xx,yy),f'Figure {i+1}: {meta["name"]}',fill='black')
        canvas.paste(im,(xx,yy+30))
    canvas.save(out/'contact_sheet.png')
    assert all(receipt(Path(r['path']))==r for r in SOURCES.values())
    print(json.dumps(dict(figures=len(FIGURES),data_rows=len(DATA),source_files=len(SOURCES),
        vector_book=str(out/'research_figures.pdf'),new_experiments=0),ensure_ascii=False))


if __name__=='__main__':main()
