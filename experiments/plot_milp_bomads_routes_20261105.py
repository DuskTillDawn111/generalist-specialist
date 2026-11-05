"""Replot archived Joint-K2 / BO-MADS v3 results; no search or prediction."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/utility-v3-search-20261105'
OUT=ROOT/'artifacts/milp-vs-bomads-routing-20261105'


def main():
    read=lambda p:json.loads(p.read_text())
    rows=read(SOURCE/'comparison.json')['rows']
    assert read(SOURCE/'verification.json')['status']=='PASS_V3_SEARCH_AND_SELECTION_AUDIT'
    methods=['Joint-K2','Activation-BO-MADS'];labels=['Joint-K2','BO–MADS']
    groups={m:sorted([r for r in rows if r['method']==m],key=lambda r:r['seed']) for m in methods}
    assert len(rows)==6
    route_rows=[]
    for m,group in groups.items():
        assert [r['seed'] for r in group]==[11,29,47]
        assert all(r['deployment']==group[0]['deployment'] and r['routing']==group[0]['routing'] for r in group)
        for r in group:
            assert r['deployment']==dict(asr=1,image=1,ocr=1,text=1,generalist_2=1)
            for task,flows in r['routing'].items():
                total=sum(flows.values());assert total>0 and set(flows)<={task,'generalist_2'}
                for model in ('generalist_2',task):
                    route_rows.append(dict(method=m,seed=r['seed'],task=task,model=model,
                        share=flows.get(model,0)/total,flow_qps=flows.get(model,0)))
    OUT.mkdir(exist_ok=False)
    with (OUT/'routing.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(route_rows[0]));w.writeheader();w.writerows(route_rows)
    (OUT/'data.json').write_text(json.dumps(dict(rows=rows,routes=route_rows),indent=2)+'\n')
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'pdf.fonttype':42,'svg.fonttype':'none','xtick.direction':'in','ytick.direction':'in'})
    colors=['#777777','#0072B2']
    with PdfPages(OUT/'comparison_and_routing.pdf') as pdf:
        fig,axs=plt.subplots(1,3,figsize=(8.5,2.65))
        fig.subplots_adjust(left=.09,right=.985,bottom=.22,top=.83,wspace=.48)
        specs=[('utility','(a) Predicted utility','Total utility (v3)',(-.60,-.50)),
               ('full_prediction_calls','(b) Prediction calls','Complete prediction calls',(0,70)),
               ('elapsed_seconds','(c) Search time','Wall time (s)',(0,55))]
        for ax,(key,title,ylabel,limits) in zip(axs,specs):
            for i,m in enumerate(methods):
                vals=[r['utility'] if key=='utility' else r['budget'][key] for r in groups[m]]
                mid=statistics.median(vals)
                if key!='utility':ax.bar(i,mid,.56,color=colors[i],alpha=.75,zorder=2)
                else:ax.plot([i-.2,i+.2],[mid,mid],color=colors[i],lw=2,zorder=2)
                ax.scatter(i+np.array([-.065,0,.065]),vals,s=20,facecolors='white',edgecolors=colors[i],zorder=3)
                ax.text(i,mid+(.003 if key=='utility' else 2.5),f'{mid:.4f}' if key=='utility' else f'{mid:.2f}' if key=='elapsed_seconds' else f'{mid:g}',ha='center',fontsize=9)
            ax.set_xticks([0,1],labels);ax.set_xlim(-.5,1.5);ax.set_ylim(*limits)
            ax.set_ylabel(ylabel);ax.set_title(title,loc='left');ax.grid(axis='y',alpha=.18,lw=.5);ax.set_axisbelow(True)
        for ext in ('png','pdf','svg'):fig.savefig(OUT/f'fig01_comparison.{ext}',dpi=600)
        pdf.savefig(fig);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(7.4,2.95),sharey=True)
        fig.subplots_adjust(left=.09,right=.985,bottom=.21,top=.75,wspace=.18)
        tasks=['asr','image','ocr','text']
        for ax,m,title in zip(axs,methods,['(a) Joint-K2 (old)','(b) BO–MADS (new)']):
            shares=[next(r['share']*100 for r in route_rows if r['method']==m and r['seed']==11 and r['task']==t and r['model']=='generalist_2') for t in tasks]
            others=100-np.array(shares);y=np.arange(4)
            ax.barh(y,shares,height=.6,color='#D55E00',label='Generalist G2')
            ax.barh(y,others,left=shares,height=.6,color='#0072B2',label='Task specialist')
            for i,(a,b) in enumerate(zip(shares,others)):
                if a:ax.text(a/2,i,f'{a:g}%',ha='center',va='center',color='white',fontsize=10)
                if b:ax.text(a+b/2,i,f'{b:g}%',ha='center',va='center',color='white',fontsize=10)
            ax.set_yticks(y,['ASR','Image','OCR','Text']);ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100])
            ax.set_xlabel('Routing share (%)');ax.set_title(title,loc='left');ax.set_ylim(3.6,-.6)
        fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.54,1.))
        for ext in ('png','pdf','svg'):fig.savefig(OUT/f'fig02_routing.{ext}',dpi=600)
        pdf.savefig(fig);plt.close(fig)
    (OUT/'README.md').write_text('''# 旧 Joint-K2 MILP 与新激活 BO–MADS：效用和路由

只重绘 2026-11-05 已有 v3 搜索的六份结果，没有新增优化、预测或 GPU 工作。

图 1 为同一冻结真实预测器、两个开发到达历史下的预测效用、完整预测次数和运行时间。三个点对应 seed 11/29/47，柱/横线为中位数。旧 Joint 是确定性搜索，三次不构成独立随机策略样本。

图 2 为各任务计划路由比例。每个算法的三个种子路由完全一致，故各展示一次；比例由原 routing QPS 除以该任务总 QPS 得到，并非实测完成请求比例。G2 和四专用均驻留、各一副本；G2 零流量不等于卸载。

旧 Joint-K2：ASR/Image/OCR 全部走对应专用，Text 全部走 G2。新 BO–MADS：四任务全部走对应专用。新 U=-0.5179219273，旧 U=-0.5835806104，差值 +0.0656586830。预测调用中位数 44 对 64，耗时约 47 秒，未观察到明显时间优势。

这批采用每次最多 64 次完整预测、32 次外层路由评价、128 次 master、300 秒。旧搜索在该路由预算内没有访问纯专用端点；差异反映有限预算内的搜索访问顺序，不是连续全局最优证明。当前生产优化器后来的预算调度变化未在该真实范围重跑，因此此图明确保留原批次版本，不冒充最新版本实测。

这里不是后来的 7/14 合成模型池比较，也不是 Proteus 论文适配对照。两算法胜者没有同批独立 GPU 头对头实测；纯专用最终方案具有既有实测支持，旧算法 text→G2 方案缺匹配独立确认。

PDF/SVG 为矢量，PNG 为 600 dpi；data.json 与 routing.csv 保留原值。源报告：../../docs/utility_v3_search_20261105.md。
''')
    receipt=lambda p:dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (OUT/'manifest.json').write_text(json.dumps(dict(
        sources=[receipt(SOURCE/n) for n in ('comparison.json','verification.json','plan.json')],
        script=receipt(Path(__file__)),outputs=[receipt(p) for p in sorted(OUT.iterdir())],
        new_searches=0,new_predictions=0,new_gpu_runs=0,routing_conservation_checked=True),indent=2)+'\n')
    print(OUT)


if __name__=='__main__':main()
