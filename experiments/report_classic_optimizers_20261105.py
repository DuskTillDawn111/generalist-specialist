"""Offline publication-style report of actual SciPy optimizer comparisons."""
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
from experiments.bilevel_scalability_benchmark import read,receipt,save

SOURCE=ROOT/'artifacts/classic-optimizer-comparison-20261105'
OUT=ROOT/'artifacts/classic-optimizer-figures-20261105'
METHODS=['BO-MADS','SciPy-DE','SciPy-DA','Random']
LABELS=['BO–MADS','DE (SciPy)','DA (SciPy)','Random']
COLORS=['#0072B2','#D55E00','#009E73','#777777']


def main():
    summary=read(SOURCE/'summary.json');plan=read(SOURCE/'plan.json')
    assert read(SOURCE/'completion.json')['status']=='COMPLETE' and not summary['audit_errors']
    for r in plan['sources']+plan['archived']:assert receipt(r['path'])==r
    OUT.mkdir(parents=True,exist_ok=False)
    rows=summary['rows'];raw={(n,m,s):read(SOURCE/f'{n}-{m}-{s}.json')
        for n in (7,14) for m in METHODS for s in (11,29,47)}
    table=[];aggregates=[]
    for r in rows:
        table.append(dict(size=r['size'],method=r['method'],seed=r['seed'],U=r['U'],gain=r['gain'],
            regret_upper=r['regret_upper'],predictions=r['budget']['full_prediction_calls'],
            seconds=r['budget']['elapsed_seconds'],milp_calls=r['budget']['milp_solve_calls'],
            objective_attempts=r['objective_attempts'],target_calls=r['target_calls'],target_seconds=r['target_seconds'],status=r['status']))
    with (OUT/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    for n in (7,14):
        ours={r['seed']:r for r in table if (r['size'],r['method'])==(n,'BO-MADS')}
        for m in METHODS:
            group=sorted((r for r in table if (r['size'],r['method'])==(n,m)),key=lambda r:r['seed'])
            deltas=[ours[r['seed']]['U']-r['U'] for r in group]
            aggregates.append(dict(size=n,method=m,median_U=statistics.median(r['U'] for r in group),
                median_gain=statistics.median(r['gain'] for r in group),median_seconds=statistics.median(r['seconds'] for r in group),
                target_calls=[r['target_calls'] for r in group],target_success=sum(r['target_calls'] is not None for r in group),
                ours_win_tie_loss=[sum(d>1e-8 for d in deltas),sum(abs(d)<=1e-8 for d in deltas),sum(d< -1e-8 for d in deltas)]))
    save(OUT/'statistics.json',dict(rows=table,aggregate=aggregates,statistics='median and all three seeds, no significance claims'))
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,
        'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8.5,'xtick.labelsize':9,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,
        'svg.fonttype':'none','hatch.linewidth':.5,'xtick.direction':'in','ytick.direction':'in'})
    pdf=PdfPages(OUT/'classic_optimizer_comparison.pdf')
    def panels():
        fig,axs=plt.subplots(1,2,figsize=(7.4,2.9))
        fig.subplots_adjust(left=.085,right=.985,bottom=.19,top=.78,wspace=.33)
        for ax in axs:ax.grid(axis='y',alpha=.18,linewidth=.5);ax.set_axisbelow(True)
        return fig,axs
    def finish(fig,name):
        for ext in ('pdf','svg','png'):fig.savefig(OUT/f'{name}.{ext}',dpi=600 if ext=='png' else None)
        pdf.savefig(fig);plt.close(fig)
    def legend(fig,ax):
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,1.015),
            ncol=4,frameon=False,columnspacing=1.2,handlelength=1.6)
    fig,axs=panels();width=.18
    for ax,key,scale,title,ylabel in [
        (axs[0],'gain',1000,'(a) Final utility',r'Utility gain ($\times10^{-3}$)'),
        (axs[1],'seconds',1,'(b) Search time','Optimizer wall time (s)')]:
        for j,(m,label,color,hatch) in enumerate(zip(METHODS,LABELS,COLORS,['','//','..','\\\\'])):
            for i,n in enumerate((7,14)):
                vals=[r[key]*scale for r in table if (r['size'],r['method'])==(n,m)];x=i+(j-1.5)*width
                ax.bar(x,statistics.median(vals),width,color=color,edgecolor=color,alpha=.8,hatch=hatch,lw=.6,label=label if i==0 else None,zorder=2)
                ax.scatter(x+np.array([-.035,0,.035]),vals,s=13,facecolors='white',edgecolors=color,lw=.8,zorder=3)
        ax.set_xticks([0,1],['7','14']);ax.set_xlabel('Model pool size');ax.set_ylabel(ylabel)
        ax.set_title(title,loc='left',pad=7);ax.set_xlim(-.5,1.5)
    axs[0].set_ylim(-.3,14);axs[0].set_yticks([0,3,6,9,12])
    axs[1].set_ylim(0,1.);axs[1].set_yticks([0,.25,.5,.75,1.])
    legend(fig,axs[0]);finish(fig,'fig01_quality_cost')

    fig,axs=panels()
    for ax,n in zip(axs,(7,14)):
        start=raw[n,'BO-MADS',11]['initial_calls'];x=np.arange(start,65)
        for m,label,color,style in zip(METHODS,LABELS,COLORS,['-','--','-.',':']):
            vals=[]
            for s in (11,29,47):
                r=raw[n,m,s];assert len(r['calls'])==64
                best=np.maximum.accumulate([c['U'] if c['U'] is not None else -np.inf for c in r['calls']])
                vals.append((best[x-1]-r['initial_U'])*1000)
            vals=np.array(vals)
            ax.fill_between(x,vals.min(axis=0),vals.max(axis=0),step='post',color=color,alpha=.08,lw=0)
            ax.step(x,np.median(vals,axis=0),where='post',color=color,ls=style,lw=1.35,label=label)
        ax.set_title(f'({"a" if n==7 else "b"}) {n} models',loc='left',pad=7)
        ax.set_xlabel('Cumulative prediction calls');ax.set_ylabel(r'Best utility gain ($\times10^{-3}$)')
        ax.set_xlim(start,64);ax.set_ylim(-.3,14);ax.set_yticks([0,3,6,9,12])
        ax.set_xticks([18,32,48,64] if n==7 else [32,40,48,56,64])
    legend(fig,axs[0]);finish(fig,'fig02_anytime')

    fig,axs=panels()
    for ax,n in zip(axs,(7,14)):
        x=np.arange(65)
        for m,label,color,style in zip(METHODS,LABELS,COLORS,['-','--','-.',':']):
            group=[r for r in table if (r['size'],r['method'])==(n,m)]
            y=[100*sum(r['target_calls'] is not None and r['target_calls']<=k for r in group)/3 for k in x]
            ax.step(x,y,where='post',color=color,ls=style,lw=1.3,label=label)
        ax.set_title(f'({"a" if n==7 else "b"}) {n} models',loc='left',pad=7)
        ax.set_xlabel('Cumulative prediction calls');ax.set_ylabel('Target reached (% of seeds)')
        ax.set_xlim(0,64);ax.set_ylim(-3,106);ax.set_xticks([0,16,32,48,64]);ax.set_yticks([0,100/3,200/3,100],['0','33','67','100'])
    legend(fig,axs[0]);finish(fig,'fig03_target_attainment');pdf.close()

    lines=['# 经典优化算法正式对照：SciPy DE / Dual Annealing','',
        '**已完成固定 24 次 CPU 搜索，使用 SciPy 1.15.3 的真实算法实现。** 没有把旧自写 Mixed-DE 改名，也没有把通用算法冒充 CherryPick、InferLine 或 MuxServe 的系统复现。用户允许相关论文方法或经典优化算法，本次采用后者。','',
        '## 方法与来源','',
        '|名称|实际执行器|依据与适配|','|---|---|---|',
        '|SciPy-DE|`scipy.optimize.differential_evolution`|Storn–Price 差分进化；`best1bin`、随机变异系数 0.5–1、交叉率 0.7、即时更新；只适配编码/目标/预算|',
        '|SciPy-DA|`scipy.optimize.dual_annealing`|广义模拟退火结合局部优化；保留默认温度、访问及接受参数和局部搜索|',
        '|Random|同一五维编码上的独立均匀采样|校准对照，用来判断搜索是否优于简单采样，不作为系统论文方法|',
        '|BO–MADS|当前修复后的 balanced 控制器|相同激活子空间、GP/EI、MADS 风格轮询、部署池及渐进内层|','',
        '[Storn & Price, 1997](https://doi.org/10.1023/A:1008202821328)；[Tsallis & Stariolo, 1996](https://doi.org/10.1016/S0378-4371(96)00271-3)。本次核对的是安装版本的官方文档、参考文献和实际库源码，没有声称重新核对原论文全部实验。',
        '[SciPy DE 文档](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.differential_evolution.html)；[SciPy Dual Annealing 文档](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.dual_annealing.html)。原文档和源码哈希保存在 `artifacts/classic-optimizer-references-20261105/`。','',
        '## 公平协议','',
        '- 7/14 模型池、11/29/47 三种子，四方法共 24 次，运行前固定并打乱顺序。同一进程、单线程 BLAS；本次四种方法同期重跑，BO–MADS 使用退出修复后的实现。',
        '- 相同目标、任务负载、显存/能力约束、完整可行部署域，以及四个精确计费初始路由；所有初始预测记录与缓存都可用。初始化分别用掉 18/32 次预测，剩余 46/32 次。',
        '- 每次上限 64 次完整预测、60 秒、128 次 MILP，256 次外层评价作循环保护。SciPy 的软 `maxfun` 不承担硬预算职责，统一评价器在全局预测预算耗尽时立即停止并保留最好已观测结果。缓存查询不额外计预测。',
        '- DE、DA、Random 共用五维紧凑编码：两个模型位置、单模型一副本/单模型双副本/双模型各一副本三个模式、两类任务的比例。由元数据修复重复位置与能力不覆盖，不查质量或效用。每种子相同模型顺序随机置换，避免固定目录名的偶然顺序成为优势。',
        '- 编码测试穷尽模型位置和部署模式，确认为 7/14 模型覆盖全部 33/117 个合法部署；包含零流量驻留模型和比例端点。两个连续比例能表达每个两模型部署内的合法分流。编码解码并非算法原生的类别优化，整数分箱会引入不连续和邻接偏置，修复会改变采样分布；随机对照使用相同映射。',
        '- DE 用 10 个个体（五维的 2 倍），四个位置放共同初始路由最优部署。没有采用默认 75 个体，因为预算不足以完成其初始种群；这一设置是小预算适配，不代表充分调优的 DE。`polish=False` 防止另一阶段自动增加工作；硬预算仍强制执行。',
        '- DA 从共同初始化的最好方案开始，保留库默认局部优化；没有替换或手写退火接受规则。有限差分查询也统一计费，不使用解析梯度。',
        '- 候选尝试上限 2048，仅防止重复/缓存导致死循环。本批所有方法都实际用满 64 次预测，无失败后重跑、调参、增加种子、扩展负载或 GPU 工作。','',
        '## 完整结果','',
        '效用越大越好；增益为最终 U 减共同初始 U。柱为三个种子的中位数，所有种子均保留。',
        '|规模|方法|最终 U 中位数|效用增益中位数|耗时中位数（秒）|达到目标|','|---|---|---:|---:|---:|---:|']
    for a in aggregates:lines.append(f"|{a['size']}|{a['method']}|{a['median_U']:.9f}|{a['median_gain']:.9f}|{a['median_seconds']:.3f}|{a['target_success']}/3|")
    lines+=['','**本批 BO–MADS 在两个规模下的效用中位数最高，但耗时也最高，且不是所有种子都获胜。**',
        '7 模型：BO–MADS 对 DE 和 Random 均为 3 胜；对 DA 为 2 胜 1 负。14 模型：对 DE 为 2 胜 1 负；对 DA 和 Random 均为 3 胜。',
        '7 模型 seed47 的 DA 找到 U=-0.376582639，优于对应 BO–MADS 的 -0.380109937。14 模型 seed47 的 DE 找到 U=-0.373008172，优于 BO–MADS 的 -0.377305615；其部署为 x09+s_text，图像全部交给 x09，文本约 19.27% 交给 x09，其余交给 s_text。说明经典方法能找到我们错过的模型组合和连续分流。',
        'BO–MADS 六次付费预测轨迹与已封存的退出修复复核逐项相同，因此这次没有靠调整自身算法获得胜出。','',
        '## 不能从本批推出的结论','',
        '- 这不是“BO–MADS 普遍优于经典优化”。只有两个合成目录、三个随机种子、64 次预算，没有统计显著性或广泛调优证据。',
        '- 前一批增删交换在 14 模型池的三次效用都优于 BO–MADS，该结论继续有效。新基线不替代或删除原有强对手；旧 Mixed-DE 与当前 SciPy-DE 的实现、编码、参数均不同，不能通过两批差异单独归因于库实现优劣。',
        '- 当前模型池规模只扩大候选目录，同时驻留最多两个副本、任务仍为两类。使用的是合成凸排队成本公式，不是冻结真实预测器的 FIFO 回放，也不代表全七真实模型或服务器系统实验。',
        '- 相同初始化虽然控制了已有信息，但来自我们原算法的精确内层，需要较多初始化成本；不能据此推断这些经典算法在各自原生初始化下的表现。',
        '- 当前函数评价极便宜，运行时间主要反映搜索管理开销；没有用此结果推断真实昂贵预测或 GPU 服务性能。每个种子只计时一次，不给出精确加速比。','',
        '## 图表与验证','',
        '图 1：最终效用增益和完整优化耗时，柱为中位数，圆点为各次运行。图 2：最好已知效用随累计预测次数变化，色带为最小—最大范围，不是置信区间。图 3：达到预定 `U_initial+0.005` 目标的种子比例，未达到的保留，不把失败记成第 64 次成功。图内没有底部说明文字。',
        f"11 项语义/预算测试通过；全部 {summary['actual_calls']} 次付费预测验证合法性、流量守恒、两套目标算术，{summary['checked_inner_results']} 个内层结果通过独立有限部署枚举核对。审计错误为 0。",
        '源码与实际 SciPy 实现 SHA-256、固定参数、调度顺序、全部候选和计费日志均已保存。没有待执行的自动实验。','',
        '[图集 PDF](classic_optimizer_comparison.pdf) · [逐种子 CSV](results.csv) · [统计 JSON](statistics.json)','']
    (OUT/'README.md').write_text('\n'.join(lines))
    inputs=[receipt(SOURCE/p) for p in ('plan.json','summary.json','completion.json')]
    inputs += [receipt(SOURCE/f'{n}-{m}-{s}.json') for n,m,s in raw]
    save(OUT/'manifest.json',dict(inputs=inputs,script=receipt(Path(__file__)),
        outputs=[receipt(p) for p in sorted(OUT.iterdir()) if p.is_file()],new_searches=0,new_gpu_runs=0))
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):z.write(p,arcname=OUT.name+'/'+p.name)
    print(OUT)


if __name__=='__main__':main()
