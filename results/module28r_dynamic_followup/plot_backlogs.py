"""Standalone scientific plots from recorded backlog events; CPU only."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'backlog_timeseries.csv').open()))
ids=list(dict.fromkeys(r['run_id'] for r in rows))
for rid in ids:
    rr=[r for r in rows if r['run_id']==rid]
    x=[float(r['offset_seconds']) for r in rr]
    fig,ax=plt.subplots(figsize=(12,4))
    for key,label in [('submitted:system','Submitted backlog (original criterion)'),('offered:system','Scheduled-arrival backlog')]:
        ax.step(x,[float(r.get(key) or 0) for r in rr],where='post',label=label,alpha=.85,lw=1)
    ax.axvline(600,color='black',ls='--',lw=1,label='Measurement end')
    ax.set(xlabel='Time from measurement start (s)',ylabel='Uncompleted requests',title=rid,xlim=(0,max(x)))
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    stem='module28_backlog' if rid=='C_steady_s_only' else rid+'_backlog'
    fig.savefig(ROOT/(stem+'.png'),dpi=160);fig.savefig(ROOT/(stem+'.pdf'));plt.close(fig)
print('PLOTS',len(ids))
