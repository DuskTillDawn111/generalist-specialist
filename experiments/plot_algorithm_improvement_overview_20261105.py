"""One scientific overview of existing algorithm cost/utility improvements.

Offline only. Missing three-model balanced results are shown as not measured.
Bars show seed medians; every raw seed remains visible. No runtime speed claim.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def receipt(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=[ROOT/'artifacts/bilevel-scalability-ablation-20261105/summary.json',
        ROOT/'artifacts/bilevel-scheduler-improvement-20261105/comparison.json']
    receipts=[receipt(p) for p in sources]
    legacy=[r for r in json.loads(sources[0].read_text())['rows'] if r['method']=='BO-MADS']
    balanced=json.loads(sources[1].read_text())['rows']
    rows=[]
    for name,rs in [('Legacy',legacy),('Improved',balanced)]:
        for r in rs:
            rows.append(dict(scheduler=name,size=r['size'],seed=r['seed'],utility=r['U'],
                initialization_utility=r['initialization_U'],gain=r['U']-r['initialization_U'],
                prediction_calls=r['budget']['full_prediction_calls'],milp_calls=r['budget']['milp_solve_calls']))
    assert len(rows)==15 and {r['initialization_utility'] for r in rows}=={-.38574508028647636}
    comparisons=[]
    for n in (7,14):
        old={r['seed']:r for r in rows if r['size']==n and r['scheduler']=='Legacy'}
        new={r['seed']:r for r in rows if r['size']==n and r['scheduler']=='Improved'}
        assert old.keys()==new.keys()=={11,29,47}
        comparisons.append(dict(size=n,median_legacy_U=statistics.median(r['utility'] for r in old.values()),
            median_improved_U=statistics.median(r['utility'] for r in new.values()),
            median_legacy_calls=statistics.median(r['prediction_calls'] for r in old.values()),
            median_improved_calls=statistics.median(r['prediction_calls'] for r in new.values()),
            paired_deltas=[dict(seed=s,utility_gain=new[s]['utility']-old[s]['utility'],
                extra_predictions=new[s]['prediction_calls']-old[s]['prediction_calls']) for s in sorted(old)]))
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,
        'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':9,'xtick.labelsize':9,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,
        'svg.fonttype':'none','hatch.linewidth':.5,'xtick.direction':'in','ytick.direction':'in'})
    fig,axs=plt.subplots(1,2,figsize=(8.2,4.15))
    fig.subplots_adjust(left=.08,right=.98,bottom=.30,top=.76,wspace=.32)
    colors={'Legacy':'#0072B2','Improved':'#D55E00'}
    width=.32;xs=np.arange(3);sizes=[3,7,14]
    for ax,key,mult,title,ylabel in [
        (axs[0],'prediction_calls',1,'(a) Search cost ↓','Complete prediction calls'),
        (axs[1],'gain',1000,'(b) Solution improvement ↑','Utility gain over initialization (×10⁻³)')]:
        for method,shift,hatch in [('Legacy',-width/2,''),('Improved',width/2,'//')]:
            for x,n in zip(xs,sizes):
                group=[r for r in rows if (r['scheduler'],r['size'])==(method,n)]
                if not group:continue
                vals=[r[key]*mult for r in group];med=statistics.median(vals)
                ax.bar(x+shift,med,width,color=colors[method],alpha=.83,edgecolor='white',hatch=hatch,
                    linewidth=.6,label=method if n==7 else None,zorder=2)
                ax.scatter(np.array([x+shift]*3)+[-.065,0,.065],vals,s=14,facecolors='white',edgecolors=colors[method],linewidths=.85,zorder=4)
                ax.text(x+shift,med*.55,f'{med:.0f}' if key=='prediction_calls' else f'{med:.3f}',
                    ha='center',va='center',fontsize=8,color='white',fontweight='bold')
        ax.set_xticks(xs,['3','7','14']);ax.set_xlabel('Synthetic model pool size');ax.set_ylabel(ylabel)
        ax.set_title(title,loc='left',pad=8);ax.grid(axis='y',alpha=.20,linewidth=.5);ax.set_axisbelow(True)
        ax.set_xlim(-.5,2.5)
        ax.text(width/2,4 if key=='prediction_calls' else .55,'n/a',ha='center',va='bottom',fontsize=8,color='#666666')
    axs[0].set_ylim(0,70);axs[0].set_yticks([0,16,32,48,64]);axs[0].axhline(64,color='#888888',ls=':',lw=.8)
    axs[0].text(2.45,65,'Budget cap: 64',ha='right',fontsize=7,color='#666666')
    axs[1].set_ylim(0,10.2);axs[1].set_yticks([0,2,4,6,8,10])
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.915),ncol=2,frameon=False)
    fig.suptitle('BO–MADS scheduling improvement: cost and utility',fontsize=12,y=.985)
    c7,c14=comparisons
    fig.text(.5,.17,'7 models: +3 predictions, ΔU = +0.003370     |     14 models: +4 predictions, ΔU = +0.000442',
        ha='center',fontsize=9,fontweight='bold')
    fig.text(.5,.105,'Bars: median of 3 seeds; circles: all seeds; n/a: not measured. Changes above compare medians.',ha='center',fontsize=7.5)
    fig.text(.5,.057,'Same initialization and budget caps (64 predictions / 32 route evaluations). Synthetic two-task scenario; ≤2 replicas.',ha='center',fontsize=7.1)
    fig.text(.5,.018,'All 6 paired final utilities improve; actual prediction calls rise. This is a quality–cost trade-off, not a speedup claim.',ha='center',fontsize=7.1)
    for ext in ('pdf','svg','png'):fig.savefig(out/f'algorithm_improvement.{ext}',dpi=600 if ext=='png' else None)
    plt.close(fig)
    with (out/'data.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'data.json').write_text(json.dumps(dict(rows=rows,comparisons=comparisons),ensure_ascii=False,indent=2)+'\n')
    readme='''# 不同模型池下的算法改进

![算法改进](algorithm_improvement.png)

左图是实际完整预测调用量，越少越省；右图是相对共同初始化的效用提升，越大越好。柱为三个种子的中位数，圆点保留全部种子。3 模型只有旧版数据，新版标为未测，未补造。

|模型池|预测次数中位数：旧→新|效用提升中位数：旧→新|最终 U 中位数改善|
|---|---|---|---|
|3|25 → 未测|0.007652 → 未测|不比较|
|7|36 → 39|0.002265 → 0.005635|+0.003370|
|14|49 → 53|0.005635 → 0.006077|+0.000442|

两版上限相同：64 次预测、32 次路由评价、128 次 MILP、60 秒；初始化相同。新版六次配对最终效用都提高，实际预测比对应旧运行多 1—5 次。图下方 +3/+4 是两种规模预测次数中位数之差，不代表每个种子。

这里用实际预测调用衡量开销，不能解释为运行时间加速。效用是负成本，图中用 U_final−U_initial，未对负总效用计算含混的提升百分比。

场景是两任务、最多两副本的合成模型池；不是 14 个模型同时驻留，也不代替真实全七模型 GPU 部署验证。3 模型旧结果的波动较大，三种规模参考最优也不同，不能仅凭跨规模柱高断言某规模更容易优化。

本图只复用已封存结果，没有新的搜索、预测器调用或 GPU 实验。PDF/SVG 为矢量，PNG 为 600 dpi；data.csv/data.json 保留全部逐种子数据。
'''
    (out/'README.md').write_text(readme)
    for r in receipts:assert receipt(Path(r['path']))==r
    (out/'manifest.json').write_text(json.dumps(dict(sources=receipts,script=receipt(Path(__file__)),
        new_experiments=0,new_prediction_calls=0,paired_improvements=6,
        outputs=[receipt(p) for p in sorted(out.iterdir()) if p.is_file()]),indent=2)+'\n')
    print(json.dumps(dict(output=str(out),comparisons=comparisons),ensure_ascii=False))


if __name__=='__main__':main()
