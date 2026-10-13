"""Standalone audit figures (no model or optimizer execution)."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/quality_workload_audit'
TASKS=['text','image','ocr','asr','visual_qa','lid']
COLORS=['#4e79a7','#59a14f','#f28e2b','#e15759','#b07aa1','#76b7b2']
def main():
 with (OUT/'workload_qwen_demand_share.csv').open() as f:original=list(csv.DictReader(f))
 names=list(dict.fromkeys(r['workload'] for r in original))
 proposals=json.loads((OUT/'proposed_demand_normalized_workloads.json').read_text())['proposals']
 fig,axs=plt.subplots(1,3,figsize=(16,5.2),gridspec_kw={'width_ratios':[1,1,1.5]})
 data=[(names,[[float(next(r for r in original if r['workload']==n and r['task']==k)['raw_request_share']) for k in TASKS] for n in names], 'Original request shares'),
       (names,[[float(next(r for r in original if r['workload']==n and r['task']==k)['normalized_qwen_demand_share']) for k in TASKS] for n in names], 'Original Qwen demand shares'),
       ([p['name'] for p in proposals],[[p['qwen_reference_demand_share'][k] for k in TASKS] for p in proposals], 'Frozen proposed demand shares')]
 for ax,(labels,values,title) in zip(axs,data):
  values=np.array(values);left=np.zeros(len(labels))
  for i,k in enumerate(TASKS):
   ax.barh(labels,values[:,i],left=left,color=COLORS[i],label=k);left+=values[:,i]
  ax.set_xlim(0,1);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('Share');ax.grid(axis='x',alpha=.2)
 fig.legend(TASKS,loc='lower center',ncol=6,frameon=False)
 fig.suptitle('Module 8A: workload demand audit — proposals frozen before planning')
 fig.tight_layout(rect=(0,.07,1,.94));fig.savefig(OUT/'workload_demand_audit.png',dpi=180);fig.savefig(OUT/'workload_demand_audit.pdf');plt.close(fig)
 matrix=json.loads((OUT/'quality_matrix.json').read_text())
 fig,axs=plt.subplots(2,3,figsize=(15,9))
 for task,ax in zip(TASKS,axs.flat):
  metric='cer' if task=='ocr' else 'wer' if task=='asr' else 'top1_accuracy' if task=='image' else 'accuracy'
  rows=[r for r in matrix['rows'] if r['task']==task and r['evaluation_status']=='COMPLETE']
  for r in rows:
   color='#4e79a7' if r['role']=='generalist' else '#f28e2b'
   ax.scatter(r['s0_ms'],r[metric],c=color,s=75,marker='o' if r['pareto_nondominated'] else 'x')
   offset=(5,5)
   if task=='image':offset=(5,15) if r['model']=='resnet50' else (5,-15) if r['model']=='convnext_tiny' else (5,5)
   if task=='text':offset=(5,-16) if r['model']=='minicpm' else (5,15) if r['model']=='phi4mm' else (5,5)
   ax.annotate(r['model'],(r['s0_ms'],r[metric]),xytext=offset,textcoords='offset points',fontsize=8)
  ax.set_xlim(min(r['s0_ms'] for r in rows)*.7,max(r['s0_ms'] for r in rows)*2.2)
  ax.set_xscale('log');ax.set_title('ocr (Pareto also uses exact match)' if task=='ocr' else task);ax.set_xlabel('Frozen standalone s0 (ms, log scale)');ax.set_ylabel(metric+' (lower better)' if task in ('ocr','asr') else metric+' (higher better)');ax.grid(alpha=.2)
  if task not in ('ocr','asr'):ax.set_ylim(-.03,1.08)
 fig.suptitle('Module 8A: quality audit over same frozen 200 IDs per task\nFrozen v2 paths only; VQA revalidation is separate. Blue=Generalist; orange=Specialist; circles=Pareto.\ns0 and quality have different sample scopes.')
 fig.tight_layout(rect=(0,0,1,.93));fig.savefig(OUT/'quality_latency_pareto.png',dpi=180);fig.savefig(OUT/'quality_latency_pareto.pdf');plt.close(fig)
if __name__=='__main__':main()
