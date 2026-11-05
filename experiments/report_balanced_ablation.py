"""Offline paired ablation tables and updated publication figure collection."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.plot_research_results_20261105 import setup, panels, style, COLORS
import matplotlib.pyplot as plt
import numpy as np

METHODS=['BO-MADS','BO-only','MADS-only','No-activation','No-pool','No-progressive','Grid-K4']
LABELS=['No MADS','No EI','No activation','No pool','Exact inner','Grid K4']
SEEDS=[11,29,47]
INPUTS={}


def receipt(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def load(p):
    INPUTS[str(p.resolve())]=receipt(p)
    return json.loads(p.read_text())


def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def savefig(out,fig,name,caption,zh):
    for ext in ('pdf','svg','png'):fig.savefig(out/f'{name}.{ext}',dpi=600 if ext=='png' else None)
    plt.close(fig)
    return dict(name=name,caption=caption,caption_zh=zh,evidence='balanced_synthetic_component_ablation',
        reference='Cascadia Fig. 9 (PDF p.9); MuxServe Figs. 7–8 (PDF p.8). Visual organization only.')


def paired_figure(out,rows):
    fig,axes=panels(height=2.9)
    for ax,n in zip(axes,(7,14)):
        for i,(seed,c,mark) in enumerate(zip(SEEDS,[COLORS['legacy'],COLORS['balanced'],COLORS['green']],['o','s','^'])):
            vals=[next(r for r in rows if (r['size'],r['method'],r['seed'])==(n,m,seed))['delta_U_variant_minus_full'] for m in METHODS[1:]]
            ax.scatter(np.arange(6)+(i-1)*.13,vals,s=20,c=c,marker=mark,label=f'Seed {seed}',zorder=4)
        ax.axhline(0,color='gray',ls='--',lw=.8)
        ax.set_xticks(range(6),LABELS,rotation=30,ha='right');ax.set_ylim(-.0095,.0045)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,-3),useMathText=True)
        style(ax,f'({"a" if n==7 else "b"}) {n} synthetic models',r'$U_{variant}-U_{full}$')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False)
    return savefig(out,fig,'fig09_balanced_ablation',
        'Current balanced scheduler: paired seed utility differences (variant minus full). Negative values favor the full method. '
        'No MADS loses in all six cases; No EI ties at 7 models and loses twice, wins slightly once at 14 models. '
        'No pool and Exact inner sometimes improve quality while spending more predictions. No activation stops after four initialization '
        'evaluations because full-domain proposals are filtered by the memory bound; this is a proposal-generation limitation, not a universal '
        'impossibility result. Grid K4 is a budgeted baseline, not a component ablation. n=3 seeds; no confidence intervals.',
        '新调度配对消融，纵轴为对照减完整版，负值有利于完整版。去掉 MADS 六次都变差；去掉 EI 在 7 模型打平，14 模型两降一微升。'
        '去池/精确内层有时以更多调用换来更好效用。去激活搜索全部停在四次初始化，因全维候选被显存过滤；不是对任意无激活算法的不可能性结论。')


def cost_figure(out,rows):
    fig,axs=plt.subplots(2,2,figsize=(7.05,4.7),layout='constrained')
    for j,n in enumerate((7,14)):
        for i,(metric,label) in enumerate([('prediction_ratio','Prediction calls: variant / full'),('milp_ratio','MILP calls: variant / full')]):
            ax=axs[i,j]
            for k,(seed,c,mark) in enumerate(zip(SEEDS,[COLORS['legacy'],COLORS['balanced'],COLORS['green']],['o','s','^'])):
                vals=[next(r for r in rows if (r['size'],r['method'],r['seed'])==(n,m,seed))[metric] for m in METHODS[1:]]
                ax.scatter(np.arange(6)+(k-1)*.13,vals,s=20,c=c,marker=mark,label=f'Seed {seed}',zorder=4)
            ax.axhline(1,color='gray',ls='--',lw=.8)
            ax.set_xticks(range(6),LABELS,rotation=30,ha='right')
            ax.set_ylim((0.35,1.4) if i==0 else (.35,2.5))
            style(ax,f'({chr(97+2*i+j)}) {n} models',label)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False)
    return savefig(out,fig,'fig10_balanced_cost',
        'Actual work ratios at the same budget maxima. Ratios above one favor full balanced in cost only. No activation uses only the shared '
        'four initialization evaluations and therefore is not an efficiency win. Full uses 39/53 predictions at 7/14 models, versus 46/60 '
        'without pool reuse or with exact inner solves. Candidate-pool removal retains route-local caching/bounds and initial observations. '
        'Quality and certification differ: removing pool reuse and exact-inner both certify 32/32 visits; full certifies 18–21/32. '
        'These are seed-paired component effects, not wall-clock speedups.',
        '相同预算上限下的真实预测/MILP 调用比，大于 1 表示完整版更省该类调用。去激活只用初始化，不能称为高效。'
        '完整版预测为 39/53 次，去池或精确内层为 46/60 次；后两者 32/32 次认证，完整版 18—21/32 次，必须结合效用与认证读图。')


def anytime_figure(out,raw):
    fig,axes=panels(height=2.7)
    styles=[('BO-MADS','Full',COLORS['legacy'],'-'),('BO-only','No MADS',COLORS['balanced'],'--'),
        ('MADS-only','No EI',COLORS['green'],'-.'),('Grid-K4','Grid K4',COLORS['purple'],':')]
    data=[]
    for ax,n in zip(axes,(7,14)):
        initial_calls=raw[n,'BO-MADS',11]['initial_signature'][-1]['work']['full_prediction_calls']
        xs=np.arange(initial_calls,65)
        for method,label,color,ls in styles:
            curves=[]
            for seed in SEEDS:
                r=raw[n,method,seed];initial=r['initial_signature'][-1]['work']['best_feasible_utility']
                curve=[max(h['best_feasible_utility'] for h in r['convergence'] if h['full_prediction_calls']<=k)-initial for k in xs]
                assert all(b>=a-1e-12 for a,b in zip(curve,curve[1:]))
                curves.append(curve)
                data.extend(dict(size=n,method=method,seed=seed,prediction_allowance=int(k),gain=float(v)) for k,v in zip(xs,curve))
            ax.step(xs,np.median(curves,axis=0),where='post',c=color,ls=ls,label=label)
        style(ax,f'({"a" if n==7 else "b"}) {n} synthetic models',r'Median $U_{best}-U_{initial}$')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,-3),useMathText=True)
        ax.set_ylim(0,.009);ax.set_xlim(initial_calls,64);ax.set_xlabel('Prediction allowance (completed observations)')
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,ncol=4,loc='outside upper center',frameon=False)
    meta=savefig(out,fig,'fig11_balanced_anytime',
        'Observed median improvement versus actual prediction allowance, for full, no-MADS, no-EI and budgeted Grid K4. '
        'Only completed inner evaluations are visible; stopped runs retain their incumbent. This is post hoc truncation of fixed runs, '
        'not rerunning or retuning at each allowance. All 3 raw seed curves are provided in CSV. Crossovers and overlap are retained; '
        'the full method is not the best at every intermediate allowance. No smoothing or time-speedup claim.',
        '按已完成内层评价的实际预测账本回看搜索进展，线为三个种子中位数；停止后的水平延伸仅表示保留已有方案。'
        '不是在每个预算点重新运行；全部原始种子曲线附 CSV。不平滑，也不声称完整版在任意中间预算都最好。')
    return meta,data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();inp=args.input.resolve();old=args.previous.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    summary=load(inp/'summary.json');plan=load(inp/'plan.json');rows=summary['rows']
    assert summary['new_searches']==36 and summary['reused_full_searches']==6 and not summary['audit_errors']
    archived=ROOT/plan['protocol']['balanced_archive']
    raw={}
    for r in rows:
        name=f"{r['size']}-balanced-{r['seed']}.json" if r['method']=='BO-MADS' else f"{r['size']}-{r['method']}-{r['seed']}.json"
        raw[r['size'],r['method'],r['seed']]=load((archived if r['method']=='BO-MADS' else inp)/name)
    groups=[]
    for n in (7,14):
        for m in METHODS:
            group=[r for r in rows if (r['size'],r['method'])==(n,m)]
            delta=[r['delta_U_variant_minus_full'] for r in group]
            groups.append(dict(size=n,method=m,U_median=statistics.median(r['U'] for r in group),
                U_min=min(r['U'] for r in group),U_max=max(r['U'] for r in group),
                predictions_median=statistics.median(r['budget']['full_prediction_calls'] for r in group),
                milp_median=statistics.median(r['budget']['milp_solve_calls'] for r in group),
                variant_wins=sum(v>1e-8 for v in delta),ties=sum(abs(v)<=1e-8 for v in delta),variant_losses=sum(v< -1e-8 for v in delta),
                certified_counts=[r['certified_evaluations'] for r in group],evaluation_counts=[r['evaluations'] for r in group],
                termination_counts=dict(Counter(r['status'] for r in group))))
    write(out/'analysis.json',dict(groups=groups,new_searches=36,reused_searches=6,
        new_inner_checks=summary['new_inner_results_checked'],reused_inner_checks=summary['reused_inner_results_checked'],
        no_universal_synergy_claim=True,no_activation_proposal_degeneracy=True,hardware_evidence=False))
    columns=['size','method','seed','U','full_U','delta_U_variant_minus_full','regret_upper','prediction_ratio','milp_ratio',
        'certified_evaluations','evaluations','winner_route_inner_certified','non_K4','status','full_prediction_calls','milp_solve_calls']
    with (out/'ablation-data.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in rows:writer.writerow({k:r[k] if k in r else r['budget'][k] for k in columns})
    write(out/'ablation-data.json',rows)
    setup();meta=load(old/'captions.json')
    for m in meta:
        for ext in ('pdf','svg','png'):
            src=old/f'{m["name"]}.{ext}';INPUTS[str(src)]=receipt(src);shutil.copyfile(src,out/src.name)
    meta[5]['caption']=meta[5]['caption'].replace('those experiments have not been run.','current balanced ablations are reported separately in Figures 9–11.')
    meta[5]['caption_zh']=meta[5]['caption_zh'].replace('新调度完整消融尚未运行，不能用此图代替。','新调度消融另见图 9—11；本图只保留历史对照。')
    meta += [paired_figure(out,rows),cost_figure(out,rows)]
    last,data=anytime_figure(out,raw);meta.append(last)
    with (out/'anytime-data.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    write(out/'captions.json',meta)
    import pymupdf as fitz
    from matplotlib.font_manager import findfont
    font=findfont('DejaVu Serif');book=fitz.open()
    page=book.new_page(width=595,height=842);page.insert_font(fontname='serif',fontfile=font)
    page.insert_textbox(fitz.Rect(50,60,545,170),'Generalist–Specialist Joint Optimization\nResults and Balanced Ablations',fontname='serif',fontsize=20,align=1)
    intro=('Updated collection: 11 figures. The fixed balanced-scheduler ablation batch is complete: 36 new CPU searches, '
        '6 reused full runs, 984 new and 192 reused independently checked inner results. No GPU runs, real predictor calls or refits.\n\n'
        'Main results: removing MADS loses utility in all six paired cases. Removing EI ties at seven models; at fourteen models, '
        'two cases lose and one gains slightly. Pool reuse and progressive inner solving save calls, but sometimes sacrifice utility. '
        'Removing activation search leaves only shared initialization because the current full-domain proposal generator fails to produce '
        'memory-feasible candidates. This limitation is reported explicitly.\n\n'
        'Scope: manufactured two-task convex queue costs, at most two replicas, three seeds. Component switches start after identical charged '
        'initialization. This is descriptive evidence, not a universal synergy or significance claim.\n\n'
        'Remaining evidence gaps: real full seven-model deployment/replica benefits, cross-workload generalization and physical SLO attainment. '
        'The new component study closes the specified synthetic ablation gap only.\n\n'
        'Figures 1–8 retain historical physical, frozen-predictor, synthetic and legacy results, with their evidence labels. Figures 9–11 '
        'report the current balanced ablations. Different objectives are not pooled.\n\n'
        'Visual references: MuxServe (ICML 2024), Figs. 5, 7–9; AlpaServe (OSDI 2023), Figs. 12–13; Cascadia (ICLR 2026), Figs. 9–10. '
        'Vector PDF/SVG and 600-dpi PNG exports; raw data and Chinese report accompany this collection.')
    assert page.insert_textbox(fitz.Rect(50,185,545,790),intro,fontname='serif',fontsize=10.5,lineheight=1.3)>=0
    for i,m in enumerate(meta,1):
        page=book.new_page(width=595,height=842);page.insert_font(fontname='serif',fontfile=font)
        page.insert_text((42,42),f'Figure {i} | '+m['evidence'].replace('_',' '),fontname='serif',fontsize=10)
        with fitz.open(out/f'{m["name"]}.pdf') as original:
            height=original[0].rect.height/original[0].rect.width*510
            page.show_pdf_page(fitz.Rect(42,70,552,70+height),original,0)
        text=f'Figure {i}. '+m['caption']+'\n\nVisual reference: '+m['reference']
        assert page.insert_textbox(fitz.Rect(45,95+height,550,790),text,fontname='serif',fontsize=9.5,lineheight=1.25)>=0
        page.insert_text((285,817),str(i),fontname='serif',fontsize=9)
    book.save(out/'research_figures_updated.pdf');book.close()
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(1600,1350),'white');draw=ImageDraw.Draw(canvas)
    for i,m in enumerate(meta[-3:]):
        im=Image.open(out/f'{m["name"]}.png').convert('RGB');im.thumbnail((1500,410))
        draw.text((20,i*450+8),f'Figure {i+9}: {m["name"]}',fill='black');canvas.paste(im,((1600-im.width)//2,i*450+32))
    canvas.save(out/'ablation_preview.png')
    for src in INPUTS.values():assert receipt(Path(src['path']))==src
    write(out/'sources.json',list(INPUTS.values()))
    print(json.dumps(dict(figures=len(meta),new_figures=3,pdf_pages=12,paired_runs=len(rows),output=str(out))))


if __name__=='__main__':main()
