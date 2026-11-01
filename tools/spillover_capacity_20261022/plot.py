#!/usr/bin/env python3
"""Plot descriptive capacity analysis only; never read experiment raw or rank U."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def receipt(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def number(value):
    return type(value) in (int,float) and math.isfinite(value) and value>0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path,help='New artifact directory')
    args=parser.parse_args(argv)
    source=receipt(args.report);report=json.loads(args.report.read_text())
    if report.get('schema')!='gsdeploy.balanced_capacity_analysis.v1' or report.get('U') is not None or report.get('ranking_performed') is not False:
        raise ValueError('Expected diagnostic analysis with U=null and no ranking')
    if args.output.exists():raise FileExistsError('Never overwrite figure artifacts')
    aliases=['text#0','text#1','generalist_2#0']
    colors={'text#0':'#2563a6','text#1':'#36a3a8','generalist_2#0':'#d17b24'}
    short={'text#0':'T0','text#1':'T1','generalist_2#0':'G2'}
    groups=[('A','text#0'),('A','text#1'),('B','generalist_2#0'),('C','text#0'),('C','text#1'),('C','generalist_2#0')]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,(left,right)=plt.subplots(1,2,figsize=(12.6,5.9),gridspec_kw={'width_ratios':[1.12,1]})
    points=[]
    for index,(state,alias) in enumerate(groups):
        rows=[r for r in report.get('common_input_comparisons',[]) if r['comparison']==f'state{state}_block1_to_block2' and r['alias']==alias]
        full=[r for r in rows if number(r.get('right_over_left')) and r.get('left_completed')==3 and r.get('right_completed')==3]
        partial=[r for r in rows if number(r.get('right_over_left')) and r not in full]
        for cohort,marker,color in [(full,'o',colors[alias]),(partial,'x','#888888')]:
            offset=np.linspace(-.18,.18,len(cohort)) if cohort else []
            values=[r['right_over_left'] for r in cohort]
            left.scatter(index+np.asarray(offset),values,color=color,s=17,marker=marker,alpha=.72,linewidths=.7)
        values=[r['right_over_left'] for r in full]
        if values:left.plot([index-.23,index+.23],[np.median(values)]*2,color='#202020',lw=2)
        left.text(index,.02,f'{len(full)}/32\n(+{len(partial)} partial)',ha='center',va='bottom',transform=left.get_xaxis_transform(),fontsize=7)
        points.append(dict(state=state,alias=alias,complete_input_pairs=len(full),partial_input_pairs=len(partial),
            missing_or_unknown_input_pairs=32-len(full)-len(partial),median_of_complete_input_ratios=float(np.median(values)) if values else None))
    left.axhline(1,color='#777777',lw=.9,ls='--');left.set_xticks(range(len(groups)),[f'{s}\n{short[a]}' for s,a in groups])
    left.set_ylabel('Block 2 / block 1 median service time')
    left.set_title('(a) Same state, alias and input',loc='left',fontweight='bold')
    left.grid(axis='y',alpha=.17);left.margins(y=.2)
    left.legend(handles=[Line2D([],[],color='#202020',lw=2,label='Median of complete input ratios'),
        Line2D([],[],color='#888888',marker='x',ls='',label='Incomplete paired input')],loc='upper left',fontsize=7,frameon=False)
    markers={'A':'o','B':'s','C':'^'};series=[]
    for job in report.get('jobs',[]):
        for drift in job.get('repetition_drift',[]):
            alias=drift['alias'];observations=drift.get('repetitions',[])
            x=[int(r['repetition']) for r in observations];y=[r.get('median_service_over_repetition0') for r in observations]
            valid=[(i,v) for i,v in zip(x,y) if number(v)]
            if not valid:continue
            right.plot(x,[v if number(v) else np.nan for v in y],color=colors[alias],marker=markers[job['state']],
                ls='-' if job['block']==1 else '--',lw=1.25,markersize=4.5,alpha=.8)
            series.append(dict(job_id=job['job_id'],alias=alias,points=[dict(repetition=i,ratio=v) for i,v in valid],
                complete_repetitions=sum(r['service_available_requests']==32 for r in observations)))
    right.axhline(1,color='#777777',lw=.9,ls=':');right.set_xticks([0,1,2]);right.set_xlabel('Balanced 32-input repetition')
    right.set_ylabel('Median service time / repetition 0');right.set_title('(b) Within-run repeated-cohort drift',loc='left',fontweight='bold')
    right.grid(axis='y',alpha=.17);right.margins(y=.2)
    handles=[Line2D([],[],color=colors[a],lw=2,label=short[a]) for a in aliases]
    handles.extend(Line2D([],[],color='#555555',marker=markers[s],ls='',label=f'State {s}') for s in ['A','B','C'])
    handles.extend([Line2D([],[],color='#555555',ls='-',label='Block 1'),Line2D([],[],color='#555555',ls='--',label='Block 2')])
    right.legend(handles=handles,ncol=3,fontsize=7,frameon=False,loc='upper left')
    status=report.get('status','UNKNOWN')
    display_status={'ALL_SIX_DEVELOPMENT_DIAGNOSTICS_COMPLETE':'All six complete','INCOMPLETE':'Incomplete'}.get(status,status)
    fig.suptitle(f'Text closed-loop development diagnostics — {display_status}',fontsize=12,fontweight='bold',y=.97)
    caption=(
        'T0/T1: text specialist aliases; G2: generalist alias. Fixed residents: text ×2, generalist_2 ×1, ocr ×1.\n'
        '(a) Each dot compares one input\'s 3-observation median across blocks; counts disclose complete/partial coverage.\n'
        '(b) Each line is one job/alias; repeated inputs and two development blocks are not independent validation.\n'
        'Activity limitation: whole-cohort service includes tails after other aliases finish; common lifetime is not simultaneous handler activity.\n'
        'Handler overlap is retrospective, not GPU utilization or a predictor. Singleton state B has trivial all-active coverage.\n'
        'Finite closed-loop cohorts do not establish open-loop stable capacity. No utility, fitting, deployment ranking or optimality claim.'
    )
    fig.text(.055,.025,caption,fontsize=7.5,ha='left',va='bottom',linespacing=1.5)
    fig.subplots_adjust(left=.065,right=.985,top=.88,bottom=.36,wspace=.28)
    if receipt(args.report)!=source:raise ValueError('Analysis report changed while plotting')
    args.output.mkdir(parents=True,exist_ok=False)
    stem=args.output/'text-capacity-diagnostic'
    fig.savefig(stem.with_suffix('.png'),dpi=180,facecolor='white');fig.savefig(stem.with_suffix('.pdf'),facecolor='white');plt.close(fig)
    data=dict(schema='gsdeploy.capacity_diagnostic_plot_data.v1',input=source,status=status,groups=points,repetition_series=series,U=None,
        scope='DESCRIPTIVE_DEVELOPMENT_DIAGNOSTIC_NOT_FORMAL_VALIDATION')
    (args.output/'plot-data.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    manifest=dict(schema='gsdeploy.capacity_diagnostic_plot_manifest.v1',input=source,tool=receipt(__file__),
        matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,U=None,
        outputs=[receipt(path) for path in sorted(args.output.iterdir()) if path.is_file()])
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='DIAGNOSTIC_FIGURE_WRITTEN',outputs=manifest['outputs'],manifest=str((args.output/'manifest.json').resolve()),U=None)))


if __name__=='__main__':main()
