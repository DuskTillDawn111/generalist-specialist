"""Descriptive figures from an existing prediction report; no fit or GPU work."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .native_flow import checked, receipt, save
from .probe_validation import read


def generate(report_path, output):
    source=receipt(report_path); report=read(checked(source))
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    rows=report['candidates']; labels=[r['candidate_id'] for r in rows]
    x=np.arange(len(rows)); width=.34
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for offset,key,name in [(-width/2,'service','Service'),(width/2,'e2e','Arrival to receipt')]:
        values=[100*r['timing_errors'][key]['mean_absolute_relative_error']
                if r.get('timing_errors',{}).get(key,{}).get('mean_absolute_relative_error') is not None
                else np.nan for r in rows]
        axes[0,0].bar(x+offset,values,width,label=name)
    axes[0,0].axhline(20,color='firebrick',ls='--',label='Original 20% limit')
    axes[0,0].set(ylabel='Mean absolute relative error (%)',title='Mean timing errors')
    axes[0,0].legend(fontsize=8)
    values=[r.get('queue_error',{}).get('mae',np.nan) for r in rows]
    axes[0,1].bar(x,values,color='#377eb8')
    axes[0,1].axhline(1,color='firebrick',ls='--',label='Original limit: 1 request')
    axes[0,1].set(ylabel='Queue depth MAE (requests)',title='Full-horizon queue error')
    axes[0,1].legend(fontsize=8)
    tasks=('text','image','asr','ocr'); data=np.full((4,len(rows)),np.nan)
    for j,row in enumerate(rows):
        for i,task in enumerate(tasks):
            p=(row.get('prediction_metrics') or {}).get('per_task',{}).get(task,{}).get('p95_ms')
            a=(row.get('observed_metrics') or {}).get('per_task',{}).get(task,{}).get('p95_ms')
            if p is not None and a is not None and a>0:data[i,j]=100*abs(p-a)/a
    axes[1,0].imshow(data,aspect='auto',cmap='YlOrRd',vmin=0,vmax=max(20,np.nanmax(data)) if np.isfinite(data).any() else 20)
    axes[1,0].set(yticks=range(4),yticklabels=tasks,title='Task P95 ARE (%) — limit 20%')
    for i in range(4):
        for j in range(len(rows)):
            label='UNKNOWN' if not np.isfinite(data[i,j]) else f'{data[i,j]:.1f}'+(' FAIL' if data[i,j]>20 else '')
            axes[1,0].text(j,i,label,ha='center',va='center',fontsize=8)
    for offset,key,name in [(-width/2,'predicted_comparison_U','Predicted'),(width/2,None,'Observed')]:
        values=[r.get(key) if key else -(r.get('observed_metrics') or {}).get('cost',float('nan')) for r in rows]
        axes[1,1].bar(x+offset,[np.nan if v is None else v for v in values],width,label=name)
    axes[1,1].set(ylabel='Six-term comparison U (higher is better)',title='Utility and candidate ranking')
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xticks(x,labels,rotation=20,ha='right');ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f"Independent prediction report | delta sign: {report.get('delta_sign_accuracy')} | pair rank: {report.get('rank_accuracy')}\nDescriptive evidence; no fit, uncertainty calibration, or deployment certification",fontsize=12)
    fig.savefig(output/'prediction-errors.png',dpi=180)
    fig.savefig(output/'prediction-errors.pdf')
    plt.close(fig);checked(source)
    result=dict(source=source,script=receipt(__file__),figures=[receipt(output/'prediction-errors.png'),receipt(output/'prediction-errors.pdf')],
        descriptive_only=True,independent_outcomes_used_for_fit=False,GPU_started=False)
    save(output/'receipt.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();generate(args.report,args.output)
