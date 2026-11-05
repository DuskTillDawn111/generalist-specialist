"""Offline scientific figures; no new search or reference solving."""
import csv
from pathlib import Path
import statistics
import sys
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.bilevel_scalability_benchmark import read,receipt,save,build_problem,QueueCostOracle

SOURCE=ROOT/'artifacts/paper-baseline-comparison-20261105'
OUT=ROOT/'artifacts/paper-baseline-figures-20261105'
METHODS=['BO-MADS','CherryPick-U','Proteus-U-linear']
LABELS=['BO–MADS','CherryPick-U (adapted)','Proteus-U (linear adapter)']
COLORS=['#0072B2','#D55E00','#009E73']


def main():
    summary=read(SOURCE/'summary.json');plan=read(SOURCE/'plan.json')
    assert read(SOURCE/'completion.json')['status']=='COMPLETE' and not summary['audit_errors']
    for r in plan['sources']+plan['archived']:assert receipt(r['path'])==r
    OUT.mkdir(parents=True,exist_ok=False);rows=summary['rows'];table=[];aggregates=[]
    raw={(n,m,s):read(SOURCE/f'{n}-{m}-{s}.json') for n in (7,14) for m in plan['protocol']['methods'] for s in (11,29,47)}
    for r in rows:
        table.append(dict(size=r['size'],method=r['method'],seed=r['seed'],U=r['U'],gain=r['gain'],
            seconds=r['budget']['elapsed_seconds'],predictions=r['budget']['full_prediction_calls'],
            returned_plan_U=r['returned_plan_U'],target_calls=r['target_calls'],status=r['status']))
    with (OUT/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    for n in (7,14):
        ours={r['seed']:r for r in table if (r['size'],r['method'])==(n,'BO-MADS')}
        for m in plan['protocol']['methods']:
            group=sorted([r for r in table if (r['size'],r['method'])==(n,m)],key=lambda r:r['seed'])
            delta=[ours[r['seed']]['U']-r['U'] for r in group]
            aggregates.append(dict(size=n,method=m,median_U=statistics.median(r['U'] for r in group),
                median_gain=statistics.median(r['gain'] for r in group),median_seconds=statistics.median(r['seconds'] for r in group),
                predictions=[r['predictions'] for r in group],target_success=sum(r['target_calls'] is not None for r in group),
                ours_win_tie_loss=[sum(v>1e-8 for v in delta),sum(abs(v)<=1e-8 for v in delta),sum(v< -1e-8 for v in delta)]))
    # Audit-only decomposition, after all searches ended. Not available to search.
    diagnostics=[]
    for n in (7,14):
        p=build_problem(n,plan['base_protocol']);oracle=QueueCostOracle()
        for m in ('Proteus-Q','Proteus-U-linear'):
            r=raw[n,m,11];detail=r['method_details'];returned=detail['returned_plan']
            routing={t:{model:v*p.config.workload[t] for model,v in row.items() if v>0}
                     for t,row in returned['shares'].items()}
            terms=oracle.score(p,returned['deployment'],routing)
            predicted_U=-(detail['surrogate_objective']+detail['fit']['intercept']) if m=='Proteus-U-linear' else None
            diagnostics.append(dict(size=n,method=m,returned_U=returned['U'],returned_quality_loss=terms['losses']['Q'],
                predicted_U=predicted_U,training_rmse=detail['fit'].get('training_rmse'),deployment=returned['deployment'],routing_qps=routing,
                retained_U=r['winner_observed']['U'],solver_gap=detail['solver_gap']))
    save(OUT/'statistics.json',dict(aggregate=aggregates,rows=table,proteus_diagnostics=diagnostics,
        statistics='Median and all three seeds; deterministic MILP repeats are not independent policy variation. No significance claim.'))
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,
        'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8.5,'xtick.labelsize':9,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,
        'svg.fonttype':'none','xtick.direction':'in','ytick.direction':'in'})
    pdf=PdfPages(OUT/'paper_baseline_comparison.pdf')
    def panels(count=2,width=7.4):
        fig,axs=plt.subplots(1,count,figsize=(width,2.95))
        fig.subplots_adjust(left=.075,right=.985,bottom=.20,top=.78,wspace=.37)
        for ax in axs:ax.grid(axis='y',alpha=.18,linewidth=.5);ax.set_axisbelow(True)
        return fig,axs
    def finish(fig,name):
        for ext in ('pdf','svg','png'):fig.savefig(OUT/f'{name}.{ext}',dpi=600 if ext=='png' else None)
        pdf.savefig(fig);plt.close(fig)
    def legend(fig,ax,ncol=3):
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,1.01),
            ncol=ncol,frameon=False,columnspacing=1.8,handlelength=1.5)
    fig,axs=panels(3,9.5);width=.23
    for ax,key,scale,title,ylabel in zip(axs,['gain','seconds','predictions'],[1000,1,1],
        ['(a) Final utility','(b) Search time','(c) Evaluation cost'],
        [r'Utility gain ($\times10^{-3}$)','Optimizer wall time (s)','Actual prediction calls']):
        for j,(m,label,color) in enumerate(zip(METHODS,LABELS,COLORS)):
            for i,n in enumerate((7,14)):
                vals=[r[key]*scale for r in table if (r['size'],r['method'])==(n,m)];x=i+(j-1)*width
                ax.bar(x,statistics.median(vals),width*.9,color=color,edgecolor=color,alpha=.8,
                    hatch=['','//','..'][j],lw=.6,label=label if i==0 else None,zorder=2)
                ax.scatter(x+np.array([-.035,0,.035]),vals,s=12,facecolors='white',edgecolors=color,lw=.8,zorder=3)
        ax.set_xticks([0,1],['7','14']);ax.set_xlabel('Model pool size');ax.set_ylabel(ylabel)
        ax.set_title(title,loc='left',pad=7);ax.set_xlim(-.5,1.5)
    axs[0].set_ylim(-.3,13);axs[0].set_yticks([0,3,6,9,12])
    axs[1].set_ylim(0,2.15);axs[1].set_yticks([0,.5,1,1.5,2])
    axs[2].set_ylim(0,70);axs[2].set_yticks([0,16,32,48,64])
    legend(fig,axs[0]);finish(fig,'fig01_paper_quality_cost')

    fig,axs=panels()
    for ax,n in zip(axs,(7,14)):
        start=raw[n,'BO-MADS',11]['initial_calls'];x=np.arange(start,65)
        for m,label,color,style in zip(METHODS,LABELS,COLORS,['-','--','-.']):
            vals=[]
            for seed in (11,29,47):
                r=raw[n,m,seed];best=np.maximum.accumulate([c['U'] if c['eligible'] else -np.inf for c in r['calls']])
                vals.append((best[np.minimum(x,len(best))-1]-r['initial_U'])*1000)
                stop=len(best)
                if stop<64:ax.scatter(stop,(best[-1]-r['initial_U'])*1000,s=16,marker='x',color=color,lw=.8,zorder=4)
            vals=np.array(vals)
            ax.fill_between(x,vals.min(axis=0),vals.max(axis=0),step='post',color=color,alpha=.09,lw=0)
            ax.step(x,np.median(vals,axis=0),where='post',color=color,ls=style,lw=1.35,label=label)
        ax.set_title(f'({"a" if n==7 else "b"}) {n} models',loc='left',pad=7)
        ax.set_xlabel('Prediction budget');ax.set_ylabel(r'Best utility gain ($\times10^{-3}$)')
        ax.set_xlim(start,64);ax.set_ylim(-.3,13);ax.set_yticks([0,3,6,9,12])
        ax.set_xticks([18,32,48,64] if n==7 else [32,40,48,56,64])
    legend(fig,axs[0]);finish(fig,'fig02_anytime')

    # Keep the original quality objective OUT of same-objective ranking.
    fig,axs=panels()
    for ax,n in zip(axs,(7,14)):
        for j,(m,label) in enumerate(zip(['Proteus-Q','Proteus-U-linear'],['Proteus-Q','Proteus-U\n(linear)'])):
            r=next(d for d in diagnostics if (d['size'],d['method'])==(n,m))
            for off,key,color,legend_label in [(-.15,'returned_U','#CC79A7','Planner return'),(.15,'retained_U','#777777','Best validated (with init.)')]:
                ax.bar(j+off,r[key],.28,color=color,alpha=.85,label=legend_label if j==0 else None,zorder=2)
        ax.set_xticks([0,1],['Proteus-Q','Proteus-U (linear)']);ax.set_ylabel('Measured total utility')
        ax.set_title(f'({"a" if n==7 else "b"}) {n} models',loc='left',pad=7)
        ax.set_ylim(-.43,-.37);ax.set_yticks([-.43,-.41,-.39,-.37]);ax.set_xlim(-.55,1.55)
    legend(fig,axs[0],2);finish(fig,'fig03_proteus_returned_plans');pdf.close()

    lines=['# 相关论文算法对照：CherryPick 与 Proteus','',
        '**固定 24 次 CPU 搜索已完成，相关论文机制已经实装。** CherryPick-U 在 14 模型池的效用中位数超过 BO–MADS，且三个种子中胜出两次；7 模型池 BO–MADS 三次均较好。本批不支持 BO–MADS 全面占优。','',
        '## 方法与适配','',
        '- CherryPick（NSDI 2017）：保留 Matérn 5/2 GP、期望改进、可行性筛选和 EI 提前停止，替换为本任务的配置空间与总效用。',
        '- Proteus（ASPLOS 2024）：保留二元部署＋连续分流的联合 MILP。Q 版保留质量目标；U-linear 版是本项目新增的线性效用代理适配，不能称为论文原实现。两者各求解一次，未强行补足预算。',
        '- 逐条原文页码、保留/替换机制、参数和不适合直接映射的 InferLine/MuxServe/Cascadia 原因见 [运行前冻结的映射](../../docs/paper_baseline_mapping_20261105.md)。原 PDF 和文本已留存。','',
        '## 同目标适配结果','',
        '所有方法共用初始 U = −0.3857450803；图 1 统计包含共同初始化的完整耗时，柱为中位数，空心点为每次运行。Proteus 保底值不能掩盖其真实规划输出，后表单独列出。','',
        '|模型池|方法|最终 U 中位数|效用增益中位数|耗时中位数 / s|实际预测次数（种子 11/29/47）|',
        '|---|---|---:|---:|---:|---|']
    for a in aggregates:
        if a['method']=='Proteus-Q':continue
        lines.append(f"|{a['size']}|{a['method']}|{a['median_U']:.9f}|{a['median_gain']:.9f}|{a['median_seconds']:.3f}|{' / '.join(map(str,a['predictions']))}|")
    lines += ['',
        '14 模型 seed11 与 seed47 的 CherryPick 都找到 **g+x09**，同时优化两类任务的连续分流，分别达到 −0.374270288 与 −0.374532241；BO–MADS 六次重跑轨迹与先前一致，14 模型仍停在 g+s_text、U=−0.377305615。seed29 的 CherryPick 选 x03+x07、U=−0.377751316，略逊于 BO–MADS。',
        'seed11 的更优分流：图像约 85.40% 交给 g、14.60% 交给 x09；文本约 25.49% 交给 g、74.51% 交给 x09。这进一步表明，联合探索模型组合与连续比例有价值，BO–MADS 目前有限预算内的覆盖仍有缺口。','',
        '## Proteus 原目标与近似误差','',
        '|模型池|版本|规划返回方案 U|共同初始保底 U|质量损失 Q（越低越好）|训练 U 近似 RMSE|',
        '|---|---|---:|---:|---:|---:|']
    for d in diagnostics:
        rmse='—' if d['training_rmse'] is None else f"{d['training_rmse']:.6f}"
        lines.append(f"|{d['size']}|{d['method']}|{d['returned_U']:.9f}|{d['retained_U']:.9f}|{d['returned_quality_loss']:.6f}|{rmse}|")
    lines += ['',
        'Q 版确实求得当前适配约束下的全局质量最优 MILP（求解 gap=0），但其优化目标不包括我们的排队和资源成本，不能用其总效用较低来断言原论文算法差。7 模型返回 g 双副本；14 模型返回 g+x05，文本全交给 g、图像全交给 x05。',
        'U-linear 版两种规模均返回 s_shared+s_text，图像/文本分别专用；方案已经存在于初始化缓存，所以新增完整预测为 0，实际总次数仍为 18/32。训练 RMSE 为 0.073730/0.055298，明显大于本批关注的约 0.01 效用改进。**这一适配的线性模型不足以表示非线性总效用，不作为有竞争力的主基线或 BO–MADS 优越性证据。** 保留结果用于说明直接把线性规划目标替换成效用代理的局限，不重新调参制造更好结果。',
        'Proteus 是确定性规划器，重复三个种子只保持同批计时和协议一致，其方案没有随机变化；不能把这三次视为三个独立策略样本。','',
        '## 预算、图表与验收','',
        '- 7/14 合成目录、3 种子、4 方法、最多 64 次完整预测（初始化占 18/32 次），60 秒和 128 次 MILP 上限；固定打乱执行顺序，单进程单线程 BLAS。',
        '- BO–MADS 六次均使用 64 次。CherryPick 三次触发 EI 提前停止，另三次用满预算。Proteus 一次规划结束；14 模型 Q 版增加一次验证，其他结果命中缓存。不再把“耗尽预算”误当作所有方法必须完成的动作。',
        '- 图 1：同目标适配的效用、耗时和实际评价次数。图 2：中位数与最小—最大范围，× 标记提前停止；停止后水平延长仅表示保留 incumbent，未虚构新预测。图 3：Proteus 原生返回与共同初始化保底值，单独展示不同目标/近似造成的差异。',
        '- PDF/SVG 为矢量图，PNG 为 600 dpi；图内只有标题、轴、图例和数据，没有底部长段解释。原始 CSV、逐次日志、配置、源码与论文哈希齐全。',
        f"- 18 项机制与预算测试通过；{summary['actual_calls']} 次已付费预测的合法性、流量、两套独立目标计算通过审计；{summary['checked_inner_results']} 个内层结果通过独立核对；审计错误为 0。",'','## 结论边界','',
        '这里只验证论文优化机制在当前合成部署问题中的表现，不是原论文软件或完整系统复现，不是真实七模型/GPU/FIFO 性能结论。7→14 增大的是候选目录，最多同时两个副本。共同初始化替代论文原生起点、类别编码、EI 候选离散近似、静态容量以及线性效用近似都限制外推。',
        '已有经典 DE/DA、随机搜索以及 14 模型上胜出的设施增删交换结果继续保留。当前主张应为：BO–MADS 在小目录表现较好且本批比 CherryPick 适配更快；大目录下 CherryPick 的更好组合/分流说明仍存在搜索覆盖不足。不能宣称全面优于相关论文算法。',
        '本批没有真实预测器重训，没有新增 GPU 运行，没有改生产优化器，也没有等待执行的扩展实验。','',
        '[图集 PDF](paper_baseline_comparison.pdf) · [逐种子 CSV](results.csv) · [完整统计](statistics.json)','']
    (OUT/'README.md').write_text('\n'.join(lines))
    inputs=[receipt(SOURCE/p) for p in ('plan.json','summary.json','completion.json')]
    inputs +=[receipt(SOURCE/f'{n}-{m}-{s}.json') for n,m,s in raw]
    save(OUT/'manifest.json',dict(inputs=inputs,script=receipt(Path(__file__)),
        outputs=[receipt(p) for p in sorted(OUT.iterdir()) if p.is_file()],new_searches=0,new_gpu_runs=0))
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):z.write(p,arcname=OUT.name+'/'+p.name)
    print(OUT)


if __name__=='__main__':main()
