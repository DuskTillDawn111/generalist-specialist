"""Publication-style figures and auditable report for the sealed 24-run batch.

This script only reads observations. No searches or predictor calls.
"""
import argparse
import csv
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.bilevel_scalability_benchmark import read, receipt, save

METHODS = ['BO-MADS', 'Add-drop-swap', 'Alternating', 'Mixed-DE']
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
LABELS = ['BO–MADS', 'Add/drop/swap', 'Alternating', 'Mixed DE']
HATCHES = ['', '//', '..', '\\\\']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.input.resolve(), args.output.resolve()
    summary, plan = read(source/'summary.json'), read(source/'plan.json')
    assert read(source/'completion.json')['status'] == 'COMPLETE'
    assert not summary['audits']
    for r in plan['sources']+plan['inputs']:assert receipt(r['path']) == r
    out.mkdir(parents=True, exist_ok=False)
    rows = summary['rows'];sizes = plan['protocol']['sizes'];seeds = plan['protocol']['seeds']
    raw = {(n,m,s):read(source/f'{n}-{m}-{s}.json') for n in sizes for m in METHODS for s in seeds}
    inputs = [receipt(source/name) for name in ('summary.json','plan.json','completion.json')]
    inputs += [receipt(source/f'{n}-{m}-{s}.json') for n,m,s in raw]
    table = []
    for r in rows:
        record = raw[r['size'],r['method'],r['seed']]
        last = record['calls'][-1]['elapsed_seconds']
        table.append(dict(size=r['size'], method=r['method'], seed=r['seed'], U=r['U'],
            gain=r['gain'], regret_upper=r['regret_upper'], prediction_calls=r['budget']['full_prediction_calls'],
            milp_calls=r['budget']['milp_solve_calls'], route_evaluations=r['budget']['outer_evaluations'],
            total_seconds=r['budget']['elapsed_seconds'], last_prediction_seconds=last,
            post_last_prediction_seconds=r['budget']['elapsed_seconds']-last,
            target_reached=r['target_reached'], target_calls=r['target_calls'],
            target_seconds=r['target_seconds'], status=r['status']))
    with (out/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    aggregate=[]
    for n in sizes:
        for method in METHODS:
            group=[r for r in table if (r['size'],r['method'])==(n,method)]
            ours={r['seed']:r for r in table if (r['size'],r['method'])==(n,'BO-MADS')}
            deltas=[ours[r['seed']]['U']-r['U'] for r in group]
            aggregate.append(dict(size=n,method=method,n=len(group),
                median={key:statistics.median(r[key] for r in group) for key in
                        ('U','gain','regret_upper','total_seconds','last_prediction_seconds','post_last_prediction_seconds')},
                target_success=sum(r['target_reached'] for r in group),
                target_calls=[r['target_calls'] for r in sorted(group,key=lambda r:r['seed'])],
                ours_wins=sum(d>1e-8 for d in deltas),ours_ties=sum(abs(d)<=1e-8 for d in deltas),
                ours_losses=sum(d < -1e-8 for d in deltas),paired_ours_minus_baseline=deltas))
    save(out/'statistics.json',dict(aggregate=aggregate,target_gain=plan['protocol']['target_gain'],
        rows=table,statistics='median and individual seeds / observed min-max; no confidence intervals',
        runtime='single same-batch measurement per seed, includes common initialization and post-budget controller work'))
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,
        'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8.5,'xtick.labelsize':9,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,
        'svg.fonttype':'none','hatch.linewidth':.5,'xtick.direction':'in','ytick.direction':'in'})
    pdf = PdfPages(out/'crossdomain_comparison.pdf')
    figures=[]
    def finish(fig,name):
        for ext in ('pdf','svg','png'):fig.savefig(out/f'{name}.{ext}',dpi=600 if ext=='png' else None)
        pdf.savefig(fig);plt.close(fig);figures.append(name)
    def panels():
        fig,axs=plt.subplots(1,2,figsize=(7.4,2.9))
        fig.subplots_adjust(left=.085,right=.985,bottom=.19,top=.78,wspace=.33)
        for ax in axs:
            ax.grid(axis='y',alpha=.18,linewidth=.5);ax.set_axisbelow(True)
        return fig,axs
    def legend(fig,ax):
        handles,labels=ax.get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,1.015),ncol=4,
                   frameon=False,columnspacing=1.2,handlelength=1.6)

    fig,axs=panels();width=.18
    for ax,key,mult,title,ylabel in [
        (axs[0],'gain',1000,'(a) Final utility',r'Utility gain ($\times 10^{-3}$)'),
        (axs[1],'total_seconds',1,'(b) Search time','Optimizer wall time (s)')]:
        for j,(method,color,label,hatch) in enumerate(zip(METHODS,COLORS,LABELS,HATCHES)):
            for i,n in enumerate(sizes):
                values=[r[key]*mult for r in table if (r['size'],r['method'])==(n,method)]
                x=i+(j-1.5)*width
                ax.bar(x,statistics.median(values),width,color=color,alpha=.8,
                       edgecolor=color,hatch=hatch,linewidth=.6,label=label if i==0 else None,zorder=2)
                ax.scatter(x+np.array([-.035,0,.035]),values,s=13,facecolors='white',edgecolors=color,lw=.8,zorder=3)
        ax.set_xticks(range(len(sizes)),sizes);ax.set_xlabel('Model pool size')
        ax.set_title(title,loc='left',pad=7);ax.set_ylabel(ylabel)
        ax.set_xlim(-.5,len(sizes)-.5)
    axs[0].set_ylim(0,12);axs[0].set_yticks([0,3,6,9,12])
    axs[1].set_yscale('log');axs[1].set_ylim(.08,10)
    axs[1].yaxis.set_major_locator(FixedLocator([.1,1,10]))
    axs[1].yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
    legend(fig,axs[0]);finish(fig,'fig01_quality_cost')

    fig,axs=panels()
    for ax,n in zip(axs,sizes):
        start=raw[n,METHODS[0],seeds[0]]['initial_calls'];x=np.arange(start,65)
        for method,color,label in zip(METHODS,COLORS,LABELS):
            ys=[]
            for seed in seeds:
                r=raw[n,method,seed]
                best=np.maximum.accumulate([v['U'] if v['U'] is not None else -np.inf for v in r['calls']])
                assert len(best)==64
                ys.append((best[x-1]-r['initial_U'])*1000)
            ys=np.asarray(ys)
            ax.fill_between(x,ys.min(axis=0),ys.max(axis=0),color=color,alpha=.09,step='post',lw=0)
            ax.step(x,np.median(ys,axis=0),where='post',color=color,label=label,lw=1.35)
        ax.set_title(f'({"a" if n==sizes[0] else "b"}) {n} models',loc='left',pad=7)
        ax.set_xlabel('Cumulative prediction calls');ax.set_ylabel(r'Best utility gain ($\times 10^{-3}$)')
        ax.set_xlim(start,64);ax.set_ylim(0,12);ax.set_yticks([0,3,6,9,12])
        ax.set_xticks([start,32,48,64] if start<32 else [32,40,48,56,64])
    legend(fig,axs[0]);finish(fig,'fig02_anytime')

    fig,axs=panels()
    for ax,n in zip(axs,sizes):
        x=np.arange(0,65)
        for method,color,label in zip(METHODS,COLORS,LABELS):
            group=[r for r in table if (r['size'],r['method'])==(n,method)]
            y=[100*sum(r['target_calls'] is not None and r['target_calls']<=k for r in group)/len(group) for k in x]
            ax.step(x,y,where='post',color=color,label=label,lw=1.3)
        ax.set_title(f'({"a" if n==sizes[0] else "b"}) {n} models',loc='left',pad=7)
        ax.set_xlabel('Cumulative prediction calls');ax.set_ylabel('Target reached (% of seeds)')
        ax.set_xlim(0,64);ax.set_ylim(-3,106);ax.set_xticks([0,16,32,48,64]);ax.set_yticks([0,100/3,200/3,100],['0','33','67','100'])
    legend(fig,axs[0]);finish(fig,'fig03_target_attainment')

    fig,ax=plt.subplots(figsize=(3.6,2.7));fig.subplots_adjust(left=.17,right=.98,bottom=.20,top=.77)
    before,after=[],[]
    for n in sizes:
        group=[r for r in table if (r['size'],r['method'])==(n,'BO-MADS')]
        # Means keep the stacked components additive; raw total points retained.
        before.append(statistics.mean(r['last_prediction_seconds'] for r in group))
        after.append(statistics.mean(r['post_last_prediction_seconds'] for r in group))
    ax.bar(range(len(sizes)),before,.5,color=COLORS[0],alpha=.8,label='Through last prediction')
    ax.bar(range(len(sizes)),after,.5,bottom=before,color='#999999',hatch='//',alpha=.8,label='After last prediction')
    for i,n in enumerate(sizes):
        values=[r['total_seconds'] for r in table if (r['size'],r['method'])==(n,'BO-MADS')]
        ax.scatter(i+np.array([-.06,0,.06]),values,s=15,facecolors='white',edgecolors='#333333',lw=.8,zorder=3)
    ax.set_xticks(range(len(sizes)),sizes);ax.set_xlabel('Model pool size');ax.set_ylabel('BO–MADS wall time (s)')
    ax.set_ylim(0,6);ax.set_yticks([0,2,4,6]);ax.grid(axis='y',alpha=.18,lw=.5);ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,1.02),frameon=False,ncol=1)
    finish(fig,'fig04_exit_overhead');pdf.close()

    lines=['# 跨领域配置策略对照：完成报告','',
        '固定 24 次 CPU 搜索全部完成；没有 GPU 实验、真实预测器调用、重训或结果驱动的追加试验。',
        '研究对象是启用、容量与任务分配的共同优化结构。当前证据仅来自既有合成凸排队成本场景，不是实际服务器工作负载或真实 FIFO/GPU 验证。','',
        '## 主要结果','',
        '**BO–MADS 在 7 模型池下的最终效用中位数最高；14 模型池下，增删交换更好。没有全面领先结论。**',
        '所有运行恰好使用 64 次完整预测。最终效用采用所有已支付、合法候选中最好者，包括内层结束前已发现的可行候选；不要求它的固定路由最优值已经认证。','',
        '|规模|方法|最终 U 中位数|相对初始化增益|返回耗时中位数（秒）|达到预定目标|',
        '|---|---|---:|---:|---:|---:|']
    for a in aggregate:
        med=a['median'];lines.append(f"|{a['size']}|{a['method']}|{med['U']:.9f}|{med['gain']:.9f}|{med['total_seconds']:.3f}|{a['target_success']}/3|")
    lines += ['',
        '7 模型中，BO–MADS 对增删交换为 3 胜，对交替优化及 Mixed DE 各为 2 胜 1 负；14 模型中，对增删交换为 3 负，对另两种方法均为 3 胜。三个种子不足以作普遍或显著性结论。',
        '交替优化在两个规模下分别只需 19/33 次预测就达到预定效用目标，BO–MADS 为 21/35 次。因此更高的最终质量不等于达到较低目标更快。','',
        '## 方法来源与适配','',
        '1. **Add/drop/swap**：Arya 等，*Local Search Heuristics for k-Median and Facility Location Problems*，2004，[DOI](https://doi.org/10.1137/S0097539702416402)。本次核对了出版方摘要和书目信息，未取得全文。借用增加、删除、交换设施的邻域；这里改为单副本增删/转移，用能力和显存筛选，再对候选分流进行修复、评价和局部微调。目标、容量限制及分流子问题均已改变，不继承论文的近似比。',
        '2. **Alternating**：Wright，*Coordinate Descent Algorithms*，2015，[原文](https://arxiv.org/abs/1502.04759)，核对原文第 1–2 页。固定部署进行流量坐标轮询，再固定流量改进部署。四个共同初值轮流推进，缩步到阈值后进行有预算的随机重启，防止固定零流量导致永远无法激活新模型。属于分块交替思想的工程适配，不是凸优化收敛结论的复现。',
        '3. **Mixed DE**：Storn–Price，1997，[DOI](https://doi.org/10.1023/A:1008202821328)。本次通过 [SciPy 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) 核对 rand/1/bin、强制一个交叉维度、选择及引用；未取得原论文全文。自实现标准 DE 算子，随机键编码模型身份，离散解码副本数/集合大小，按任务归一化分流。种群 8、F=0.8、CR=0.7、延迟代际更新、边界截断；参数在结果前固定，未调优。这里的混合解码是本项目适配，不称为某个原版混合 DE 系统。',
        '4. **BO–MADS**：保留当前 balanced 控制器、激活子空间、共享部署池、渐进 MILP 和已认证观测 GP；生产源码没有改变。',
        '',
        '方法来源的阅读范围、原始证据和 SHA-256 见 `artifacts/crossdomain-method-references-20261105/references.json`。这里只比较机制适配，不能写成超过三个论文原系统。',
        '',
        '## 公平性与固定边界','',
        '- 同一合成目标、模型池、任务负载、能力/显存约束及 oracle；没有给基线使用参考最优或梯度。结构筛选、混合编码修复不查效用。',
        '- 相同四个初始路由，由相同精确内层求解并计费；逐项核对旧档案的路由、上下界、认证、累计预测/MILP 与最好效用。所有已观测初始化候选和缓存同样可用。',
        '- 总上限：64 次完整预测、60 秒、128 次 MILP。外层/候选评价上限从历史实验的 32 改为 256，作为循环保护；这不是跨方法等价成本。本次四种方法均重跑，同一进程、单线程 BLAS、预先随机化顺序。不能把本批绝对值直接拼接到旧 32 路由实验。',
        '- 7/14 模型初始化分别花费 18/32 次预测，剩余搜索预算为 46/32。两种规模都最多容纳两副本；扩展的是候选目录，不是同时运行模型数，也不是任务维数或多服务器规模。',
        '- 三种基线均是本项目明确参数的适配实现，不是广泛调优后的最佳版本。DE 的 23/44 维编码和 8 个个体在此预算下探索较少，应保留此限制。',
        '- 本合成 oracle 的实际算术开销很小；CPU 耗时主要反映优化管理开销，不代表真实预测器耗时，更不能外推 GPU 服务吞吐。每次只有一次计时，不作精确加速比或统计显著性结论。',
        '',
        '## 发现的退出开销','',
        '**六次 BO–MADS 均已耗尽 64 次预测，但仍继续外层调度，直到 256 次外层上限才返回。** 完整耗时保留在主图，不能删除这部分来声称速度优势。',
        '7/14 模型的最后一次预测时刻中位数约为 0.738/0.828 秒，而实际返回时刻中位数为 3.953/4.975 秒。预测结束后的部分是无新增预测的收尾/继续调度时间，不能全部严格归因于某一行代码。原始事件可核对预测计数已经停在 64。',
        '本次没有修改冻结控制器或重跑。后续具体改进应先检查总预测预算耗尽时的停止传播，保留已发现最好候选，并允许明确的缓存结果处理；不能仅因内层返回未完成就继续反复启动无法获得新证据的求解。',
        '',
        '## 目标命中与图表口径','',
        '目标在运行前固定为 `U_initial + 0.005 = -0.38074508028647636`。记录首次达到目标的累计预测次数及实际时间；失败保持空值，不把失败当作 64 次成功，也不只对成功者取平均。','',
        '|规模|方法|种子 11 / 29 / 47 的首次目标调用数|',
        '|---|---|---|']
    for a in aggregate:lines.append(f"|{a['size']}|{a['method']}|{' / '.join(str(v) if v is not None else '未达到' for v in a['target_calls'])}|")
    lines += ['',
        '- 图 1：最终效用增益和实际完整优化耗时；柱为中位数、圆点为三个种子，耗时采用对数轴。',
        '- 图 2：每个付费预测后可得到的最好效用；线为中位数、色带是三个种子的最小—最大范围，不是置信区间。横轴包含初始化，曲线从初始化结束开始显示。',
        '- 图 3：截至给定调用次数，达到预定目标的种子比例；包括未达标运行。',
        '- 图 4：BO–MADS 最后一次预测前/后的时间分解；堆叠柱使用均值以保证可加，圆点为各次完整时间。',
        '图内仅保留图例、坐标和面板标题；没有底部解释文字。PDF/SVG 为矢量，PNG 为 600 dpi。','',
        '## 验证与交付','',
        f"全部 {summary['raw_calls_audited']} 次付费预测核对合法性、请求流量守恒和两套独立目标算术；{summary['checked_inner_results']} 个内层结果通过有限部署枚举核对上下界和认证状态。审计错误为 0。",
        '6 项语义测试覆盖共同初始化及计费、缓存不重复预测、预算不可突破、邻域覆盖、混合编码修复和 DE 强制交叉。测试仅验证语义，没有额外性能搜索。',
        '原始结果：`artifacts/crossdomain-strategy-comparison-20261105/`。本目录保存四张图、合并 PDF、逐种子 CSV、统计 JSON 和来源清单。',
        '结论允许后续比较和有针对性的改进，但本批实验已经结束，没有自动追加任务。','']
    report='\n'.join(lines)
    (out/'README.md').write_text(report)
    save(out/'manifest.json',dict(source_inputs=inputs,script=receipt(Path(__file__)),
        outputs=[receipt(p) for p in sorted(out.iterdir()) if p.is_file()],
        new_searches=0,new_predictor_calls=0,figures=figures))
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):z.write(p,arcname=out.name+'/'+p.name)
    print(json.dumps(dict(output=str(out),figures=figures,aggregate=aggregate),ensure_ascii=False))


if __name__=='__main__':main()
