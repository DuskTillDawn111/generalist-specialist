"""Standalone CPU plots. Failed prefixes are never presented as full windows."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('/root/autodl-tmp/generalist-specialist/results/module29_dynamic_replication')
f=json.loads((OUT/'frozen_replication_manifest.json').read_text())
S='host:'+f['parameters']['S_worker']
def rows(p):
    with p.open() as stream:return list(csv.DictReader(stream))
def backlog(ax,rr,title,aborted=None):
    x=np.array([float(r['offset_seconds']) for r in rr])
    system=np.array([float(r.get('host:system') or 0) for r in rr]);specialist=np.array([float(r.get(S) or 0) for r in rr])
    ax.step(x,system,where='post',lw=.7,c='#606770',label='System host outstanding')
    ax.step(x,specialist,where='post',lw=.65,c='#1765b0',label='Specialist host outstanding')
    ax.set(xlabel='Seconds from planned measurement start',ylabel='Requests',title=title,xlim=(0,2405))
    ax.set_ylim(bottom=0);ax.grid(alpha=.17);ax.axvline(2400,c='#343434',ls='--',lw=.8)
    if aborted is None:ax.axvspan(1800,2400,color='#b9c7dc',alpha=.18,label='Frozen final 25% interval')
    else:
        ax.axvline(aborted,c='#b62323',ls='--',lw=1.2,label=f'Telemetry abort at {aborted:.3f}s')
        ax.axvspan(aborted,2400,color='#f1c8c8',alpha=.45,label='Remaining registered interval unobserved')
    ax.legend(fontsize=8,loc='upper left')
fig,axes=plt.subplots(2,2,figsize=(14,8),sharex=True)
for ax,rid in zip(axes.flat,['G1_s_only','G1_fixed10','G1_dynamic','G2_dynamic']):
    r=json.loads((OUT/'run_summaries'/(rid+'.json')).read_text())['run']
    assert r['eligible'] and r['formal_seconds']==2400.
    backlog(ax,rows(OUT/'per_run'/(rid+'_backlog.csv')),rid+' | legal 2400s | '+r['system_state'])
fig.suptitle('Module 29: four complete GPU measurement windows; original stability criteria',fontsize=14)
fig.tight_layout()
for suffix in ['png','pdf']:fig.savefig(OUT/('completed_backlogs.'+suffix),dpi=170)
plt.close(fig)
failed=json.loads((OUT/'technical_stop_audit.json').read_text())[0]
fig,ax=plt.subplots(figsize=(13,4.5))
backlog(ax,rows(OUT/'aborted_backlog_timeseries.csv'),'G2 S-only: technically aborted, U = NULL; prefix is diagnostic only',failed['observed_seconds_until_abort'])
fig.tight_layout()
for suffix in ['png','pdf']:fig.savefig(OUT/('aborted_backlog.'+suffix),dpi=170)
plt.close(fig)
pairs=rows(OUT/'paired_utility_comparison.csv');assert len(pairs)==2 and {p['group'] for p in pairs}=={'1'}
fig,ax=plt.subplots(figsize=(8,4.7))
ax.axhline(0,c='#4b4b4b',lw=.9);ax.axhline(.01,c='#b4851b',ls='--',lw=1.,label='Frozen success threshold +0.01')
for x,arm in enumerate(['s_only','fixed10']):
    p=next(p for p in pairs if p['control_arm']==arm);y=float(p['delta_U'])
    ax.scatter([x],[y],s=100,c='#b42b2b' if y<0 else '#227841',zorder=4)
    ax.annotate(f'{y:+.6f}',(x,y),xytext=(10,10),textcoords='offset points',fontsize=12)
ax.set(xticks=[0,1],xticklabels=['Dynamic - S-only','Dynamic - Fixed10'],ylabel='Observed system utility difference',
    title='Module 29: one complete independent triplet of three planned\nNo three-pair mean, SD, or 95% interval',xlim=(-.4,1.7),ylim=(-.32,.76))
ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',fontsize=9);fig.tight_layout()
for suffix in ['png','pdf']:fig.savefig(OUT/('independent_pair_deltas.'+suffix),dpi=170)
plt.close(fig)
print('CPU_PLOTS_DONE; no GPU inference; no censored utility comparisons')
