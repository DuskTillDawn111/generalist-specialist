"""Standalone plotting entry point (base Python matplotlib; no model environment changes)."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(sys.argv[1]); summary=json.loads((out/'profiling_summary.json').read_text())
conditions=summary['conditions']; tasks=['text','image','ocr','asr','visual_qa','lid']
metrics={'mean_worker_latency_ms':'Worker','mean_e2e_latency_ms':'E2E mean','p95_e2e_latency_ms':'E2E P95','mean_worker_queue_wait_ms':'Queue wait'}
a=[c for c in conditions if c['family']=='load_axis' and c['repetitions']]
fig,axes=plt.subplots(2,3,figsize=(15,9))
for task,ax in zip(tasks,axes.flat):
    for key,label in metrics.items():
        good=[c for c in a if c['per_task'][task][key]['mean'] is not None]
        line,=ax.plot([c['u'] for c in good],[c['per_task'][task][key]['mean'] for c in good],'o-',label=label)
        for c in good:
            values=c['per_task'][task][key]['values']
            ax.scatter([c['u']+(i-1)*.0015 for i in range(len(values))],values,color=line.get_color(),s=16,alpha=.5,marker='x')
    for c in a:
        if c['unstable_runs']: ax.axvline(c['u'],color='red',alpha=.2)
    ax.set(title=task,xlabel='Qwen standalone service demand u',ylabel='Latency (ms, log scale)',yscale='log'); ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.suptitle('A: Fixed mix; circles = average across runs, crosses = individual runs\nSuccessful requests; red lines = unstable condition; sample std in report; lines are visual guides')
fig.tight_layout(); fig.savefig(out/'load_axis.png',dpi=160); plt.close(fig)
b=[c for c in conditions if (c['name']=='A_u_0.40' or c['family']=='task_mix_axis') and c['repetitions']]
fig,axes=plt.subplots(2,3,figsize=(15,9))
for task,ax in zip(tasks,axes.flat):
    for i,(key,label) in enumerate(metrics.items()):
        good=[c for c in b if c['per_task'][task][key]['mean'] is not None]
        line,=ax.plot([b.index(c)+(i-1.5)*.08 for c in good],[c['per_task'][task][key]['mean'] for c in good],'o',label=label)
        for c in good:
            values=c['per_task'][task][key]['values']
            ax.scatter([b.index(c)+(i-1.5)*.08+(j-1)*.012 for j in range(len(values))],values,color=line.get_color(),s=16,alpha=.5,marker='x')
    ax.set_xticks(range(len(b)),['balanced' if c['name'].startswith('A_') else c['name'][2:] for c in b],rotation=20)
    ax.set(title=task,ylabel='Latency (ms, log scale)',yscale='log'); ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.suptitle('B: Matched Qwen u=0.4; circles = average across runs, crosses = individual runs\nSuccessful requests only, including censored runs; sample std and incomplete counts in report')
fig.tight_layout(); fig.savefig(out/'task_mix_axis.png',dpi=160); plt.close(fig)
rank=summary['resource_pressure_associations']; dimensions=[r['feature'] for r in rank[:6]]
if not dimensions: dimensions=['kernel_time_ms_per_request','offline_power_w','offline_gpu_util_pct','offline_memory_util_pct','kernel_fraction_attn','kernel_fraction_gemm']
fig,axes=plt.subplots(2,3,figsize=(15,9))
for feature,ax in zip(dimensions,axes.flat):
    for i,c in enumerate(summary['colocation_paired_changes']):
        stat=c['mean_e2e_delta_ms']; x=c['pressure'].get(feature)
        if stat['mean'] is not None and x is not None:
            ax.errorbar(x,stat['mean'],yerr=stat['std'] or 0,fmt='o',label=c['condition'][2:])
    ax.axhline(0,color='gray',linewidth=.7); ax.set(xlabel=feature+' × req/s',ylabel='Paired Qwen mean E2E increase (ms)'); ax.grid(alpha=.2)
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
fig.suptitle('C: Fixed Qwen u/mix/residency; exploratory top |rank correlations|, n=4 conditions')
fig.tight_layout(); fig.savefig(out/'colocation_axis.png',dpi=160); plt.close(fig)
