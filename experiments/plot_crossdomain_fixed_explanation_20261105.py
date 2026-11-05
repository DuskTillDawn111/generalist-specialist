"""Offline revised charts and arithmetic explanation of two observed winners.

No searches or new candidates. BO-MADS timing comes from the six stop-fix
regressions; baseline timing is explicitly retained from the preceding batch.
"""
import csv
import json
from pathlib import Path
import statistics
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.bilevel_scalability_benchmark import read,receipt,save,build_problem,QueueCostOracle

OLD=ROOT/'artifacts/crossdomain-strategy-comparison-20261105'
FIX=ROOT/'artifacts/bilevel-prediction-budget-stop-20261105'
OUT=ROOT/'artifacts/crossdomain-fixed-figures-20261105'
METHODS=['BO-MADS','Add-drop-swap','Alternating','Mixed-DE']
LABELS=['BO–MADS','Add/drop/swap','Alternating','Mixed DE']
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    assert read(FIX/'completion.json')['status']=='COMPLETE'
    assert not read(FIX/'comparison.json')['audit_errors']
    for r in read(FIX/'regression-sources.json')['sources']+read(FIX/'regression-sources.json')['inputs']:
        assert receipt(r['path'])==r
    inputs=[];raw={};rows=[]
    for n in (7,14):
        for method in METHODS:
            for seed in (11,29,47):
                source=FIX if method=='BO-MADS' else OLD
                p=source/f'{n}-{method}-{seed}.json';inputs.append(receipt(p));r=read(p)
                raw[n,method,seed]=r
                if method=='BO-MADS':
                    oldp=OLD/p.name;inputs.append(receipt(oldp));old=read(oldp)
                    keys=('index','phase','deployment','routing_qps','U','eligible')
                    assert [{k:c[k] for k in keys} for c in r['calls']]==[{k:c[k] for k in keys} for c in old['calls']]
                rows.append(dict(size=n,method=method,seed=seed,U=r['winner_observed']['U'],
                    gain=r['winner_observed']['U']-r['initial_U'],seconds=r['budget']['elapsed_seconds'],
                    predictions=r['budget']['full_prediction_calls'],source_batch=source.name))
    with (OUT/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plan=read(OLD/'plan.json');problem=build_problem(14,plan['base_protocol'])
    winners={method:raw[14,method,11]['winner_observed'] for method in METHODS[:2]}
    losses={m:QueueCostOracle().score(problem,w['deployment'],w['routing_qps'])['losses'] for m,w in winners.items()}
    contribution={k:problem.config.weights[k]*(losses['BO-MADS'][k]-losses['Add-drop-swap'][k]) for k in losses['BO-MADS']}
    delta=winners['Add-drop-swap']['U']-winners['BO-MADS']['U']
    assert abs(sum(contribution.values())-delta)<1e-12
    coverage=[]
    for seed in (11,29,47):
        r=raw[14,'BO-MADS',seed]
        positive=[c for c in r['calls'] if any(row.get('x09',0)>0 for row in c['routing_qps'].values())]
        discovery=next(c for c in raw[14,'Add-drop-swap',seed]['calls'] if c['U']==winners['Add-drop-swap']['U'])
        coverage.append(dict(seed=seed,BO_MADS_x09_positive_predictions=len(positive),
            winning_pair_registered=any(s['models']==['g','x09'] for s in r['activation_spaces']['spaces']),
            winning_pair_visits=next(s['visits'] for s in r['activation_spaces']['spaces'] if s['models']==['g','x09']),
            facility_discovery_call=discovery['index'],facility_discovery_phase=discovery['phase']))
        assert not positive
    save(OUT/'winner_explanation.json',dict(winners=winners,losses=losses,utility_contributions=contribution,
        delta_U=delta,coverage=coverage,
        model_metadata={m:problem.config.catalog['models'][m] for m in ('g','s_text','x09')},
        calculation='arithmetic decomposition of two already evaluated plans; no counterfactual candidates or predictor calls'))
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,
        'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8.5,'xtick.labelsize':9,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,
        'svg.fonttype':'none','hatch.linewidth':.5,'xtick.direction':'in','ytick.direction':'in'})
    pdf=PdfPages(OUT/'updated_comparison.pdf')
    def finish(fig,name):
        for ext in ('pdf','svg','png'):fig.savefig(OUT/f'{name}.{ext}',dpi=600 if ext=='png' else None)
        pdf.savefig(fig);plt.close(fig)
    def panels():
        fig,axs=plt.subplots(1,2,figsize=(7.4,2.9))
        fig.subplots_adjust(left=.085,right=.985,bottom=.19,top=.78,wspace=.33)
        for ax in axs:ax.grid(axis='y',alpha=.18,linewidth=.5);ax.set_axisbelow(True)
        return fig,axs
    def legend(fig,ax):
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,1.015),
                   ncol=4,frameon=False,columnspacing=1.2,handlelength=1.6)
    fig,axs=panels();width=.18
    for ax,key,scale,title,ylabel in [
        (axs[0],'gain',1000,'(a) Final utility',r'Utility gain ($\times 10^{-3}$)'),
        (axs[1],'seconds',1,'(b) Search time','Optimizer wall time (s)')]:
        for j,(method,color,label,hatch) in enumerate(zip(METHODS,COLORS,LABELS,['','//','..','\\\\'])):
            for i,n in enumerate((7,14)):
                vals=[r[key]*scale for r in rows if (r['size'],r['method'])==(n,method)]
                x=i+(j-1.5)*width
                ax.bar(x,statistics.median(vals),width,color=color,edgecolor=color,alpha=.8,hatch=hatch,lw=.6,
                       label=label if i==0 else None,zorder=2)
                ax.scatter(x+np.array([-.035,0,.035]),vals,s=13,facecolors='white',edgecolors=color,lw=.8,zorder=3)
        ax.set_xticks([0,1],['7','14']);ax.set_xlabel('Model pool size');ax.set_ylabel(ylabel)
        ax.set_title(title,loc='left',pad=7);ax.set_xlim(-.5,1.5)
    axs[0].set_ylim(0,12);axs[0].set_yticks([0,3,6,9,12])
    axs[1].set_ylim(0,1.05);axs[1].set_yticks([0,.25,.5,.75,1.])
    legend(fig,axs[0]);finish(fig,'fig01_quality_cost_fixed')

    fig,axs=panels()
    models=['g','s_text','x09'];modelcolors=[COLORS[0],COLORS[1],COLORS[2]]
    for ax,method,title in zip(axs,METHODS[:2],['(a) BO–MADS','(b) Add/drop/swap']):
        w=winners[method];bottom=np.zeros(2)
        for model,color in zip(models,modelcolors):
            vals=np.array([100*w['routing_qps'][t].get(model,0)/problem.config.workload[t] for t in ('image','text')])
            ax.bar([0,1],vals,.52,bottom=bottom,color=color,alpha=.85,label=model,zorder=2)
            for i,value in enumerate(vals):
                if value>0:ax.text(i,bottom[i]+value/2,f'{value:.0f}%',ha='center',va='center',color='white',fontsize=9)
            bottom+=vals
        ax.set_xticks([0,1],['Image','Text']);ax.set_ylabel('Routing share (%)');ax.set_ylim(0,100)
        ax.set_yticks([0,25,50,75,100]);ax.set_title(title,loc='left',pad=7);ax.set_xlim(-.6,1.6)
    legend(fig,axs[0]);finish(fig,'fig02_winning_routes')

    grouped=[contribution['Q'],sum(contribution[k] for k in ('T','L','P')),contribution['M'],contribution['E']]
    fig,ax=plt.subplots(figsize=(4.5,2.8));fig.subplots_adjust(left=.17,right=.98,bottom=.22,top=.94)
    bars=ax.bar(range(4),np.array(grouped)*1000,color=[COLORS[2] if v>=0 else COLORS[1] for v in grouped],width=.56,alpha=.85)
    ax.axhline(0,color='#555555',lw=.7);ax.set_xticks(range(4),['Quality','Service','Memory','Energy'])
    ax.set_ylabel(r'Utility difference ($\times 10^{-3}$)');ax.set_ylim(-.4,2.8)
    ax.grid(axis='y',alpha=.18,lw=.5);ax.set_axisbelow(True)
    for i,v in enumerate(grouped):
        ax.text(i,v*1000+(.07 if v>=0 else -.07),f'{v*1000:+.3f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=8)
    finish(fig,'fig03_utility_decomposition')

    fig,axs=panels()
    for ax,n in zip(axs,(7,14)):
        start=raw[n,'BO-MADS',11]['initial_calls'];x=np.arange(start,65)
        for method,color,label,style in zip(METHODS,COLORS,LABELS,['-','--','-.',':']):
            ys=[]
            for seed in (11,29,47):
                r=raw[n,method,seed]
                best=np.maximum.accumulate([c['U'] if c['U'] is not None else -np.inf for c in r['calls']])
                ys.append((best[x-1]-r['initial_U'])*1000)
            ys=np.asarray(ys)
            ax.fill_between(x,ys.min(axis=0),ys.max(axis=0),step='post',color=color,alpha=.08,lw=0)
            ax.step(x,np.median(ys,axis=0),where='post',color=color,ls=style,lw=1.3,label=label)
        ax.set_title(f'({"a" if n==7 else "b"}) {n} models',loc='left',pad=7)
        ax.set_xlabel('Cumulative prediction calls');ax.set_ylabel(r'Best utility gain ($\times 10^{-3}$)')
        ax.set_xlim(start,64);ax.set_ylim(0,12);ax.set_yticks([0,3,6,9,12])
        ax.set_xticks([18,32,48,64] if n==7 else [32,40,48,56,64])
    legend(fig,axs[0]);finish(fig,'fig04_anytime')
    pdf.close()
    report=f'''# 修复后图表与 14 模型池胜者解释

图 1 将 BO–MADS 的耗时换为退出修复后的六次记录；其他三种方法沿用前一批记录。两批不是同期计时，耗时只作描述性比较，不提供严格加速比。24 个展示记录均为 64 次预测。六次 BO–MADS 的付费预测轨迹及最终方案与修复前逐项相同，因此效用图不变。本次没有新增搜索或 GPU 实验。

![更新对比](fig01_quality_cost_fixed.png)

## 14 模型池找到了什么

|方法|部署|图像分流|文本分流|最终 U|
|---|---|---|---|---:|
|BO–MADS|g × 1，s_text × 1|g 100%|g 25%，s_text 75%|{winners['BO-MADS']['U']:.9f}|
|增删交换|g × 1，x09 × 1|g 50%，x09 50%|g 50%，x09 50%|{winners['Add-drop-swap']['U']:.9f}|

两者都是两个模型、各一副本，占用相同显存。增删交换更好的地方是模型组合与分流的联合选择，不是副本更多，也不是找到了更细的连续比例：胜者恰好是 50% / 50%。三个种子得到相同的各自胜者，增删交换分别在第 52 / 39 / 43 次预测发现胜者，来源均为 `facility_add_drop_swap`，不是后续分流精修。

这个基线从已有较好部署出发，逐个尝试增加、删除或替换模型，按已定义的规则为新模型分配流量并评价。14 模型目录中新增的 x09 同时支持图像和文本，7 模型目录只有 x00–x03，没有 x09。因此两个规模的胜负不同有明确的候选目录差异。

BO–MADS 的三个修复后日志都已经登记 g+x09 子空间，但访问数为 0，64 次实际预测中 x09 正流量次数均为 0。初始化曾检查 g+x09 的驻留结构，但 x09 流量为零，这不等于评估该组合的共享服务能力。64 次总预测中初始化使用 32 次，后续搜索未在剩余预算内覆盖这个有利组合。这是现有候选覆盖/调度不足的证据；不能单凭此次结果断言 GP 或 MADS 数学机制无效，也不能说放宽预算必定解决。

![胜者分流](fig02_winning_routes.png)

## 为什么效用更高

当前合成目标为 `U=−[Q+(T+L+P)/3+M+E]/4`。这里所有字母是既有合成成本项，不是实测准确率、GPU 时延或能耗。对两个已经评价过的胜者做算术分解：

- 质量成本 Q 从 {losses['BO-MADS']['Q']:.6f} 降到 {losses['Add-drop-swap']['Q']:.6f}，贡献 +{contribution['Q']:.9f} 效用。
- 服务组 T/L/P 合计贡献 {grouped[1]:+.9f}，略微抵消质量收益。
- 显存成本相同，贡献 0。
- 能耗成本贡献 {contribution['E']:+.9f}，也略微抵消收益。
- 净收益为 **+{delta:.9f}**。

x09 的文本质量损失系数为 0.085，s_text 为 0.11；图像上 x09 的损失 0.11 又高于 g 的 0.09。因此这个替换牺牲一点图像质量、改善文本质量，在当前 0.65/0.35 的任务负载和评分下得到较低的综合质量成本。没有证据说明胜出主要来自排队缩短。

![效用来源](fig03_utility_decomposition.png)

## 对我们算法的含义

下一项值得修的是有限预算中的模型替换候选覆盖，例如把有预算的交换候选接入现有搜索，并公平比较效用与开销；不能把 x09 硬编码成优先模型，也不能凭此宣称交换机制必然提高所有场景。本次仅解释和更新图表，没有更改搜索机制或追加实验。

## 文件与口径

- `updated_comparison.pdf`：四页矢量图集；各图另有 SVG 和 600 dpi PNG。
- `results.csv`：逐种子结果，含来源批次，避免把新旧计时冒充同期对照。
- `winner_explanation.json`：两个胜者、成本分解、模型参数和逐种子覆盖检查。
- `manifest.json`：来源与输出 SHA-256。所有图内均无底部解释段落。

主图柱为三个种子的中位数，圆点为每次结果；收敛曲线色带为最小—最大范围，不是置信区间。场景仍为合成候选模型池，不是全部真实七模型或实际服务器部署实验。
'''
    (OUT/'README.md').write_text(report)
    for r in inputs:assert receipt(r['path'])==r
    save(OUT/'manifest.json',dict(inputs=inputs,script=receipt(Path(__file__)),
        arithmetic_source=receipt(ROOT/'experiments/bilevel_scalability_benchmark.py'),
        outputs=[receipt(p) for p in sorted(OUT.iterdir()) if p.is_file()],
        new_searches=0,new_predictor_calls=0,new_gpu_runs=0))
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):z.write(p,arcname=OUT.name+'/'+p.name)
    print(json.dumps(dict(output=str(OUT),utility_contribution=contribution,delta=delta,coverage=coverage)))


if __name__=='__main__':main()
