"""Standalone scientific figure; run only after the six measurements."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent
g=json.loads((out/'module32c_gate.json').read_text());s=g['statistics']
p=list(csv.DictReader((out/'gpu_independent_pairs.csv').open()));d=list(csv.DictReader((out/'per_task_utility_decomposition.csv').open()));paths=list(csv.DictReader((out/'gpu_worker_service_wait.csv').open()));q=list(csv.DictReader((out/'Whisper_paired_queue_service.csv').open()))
fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True);colors=['#1f77b4','#ff7f0e','#2ca02c'];valid=[v for v in p if v['delta_U']];x=np.arange(len(valid));a=axs[0,0]
if valid:
 a.bar(x,[float(v['delta_U']) for v in valid],color=colors[:len(valid)]);a.axhline(.01,color='black',ls=':',label='Frozen +0.01 threshold');a.axhline(0,color='gray',lw=.8);a.set_xticks(x,[f'Pair {v["pair"]}' for v in valid]);a.set_ylabel('Official six-task Delta U');a.set_title('Independent matched contrasts');a.legend(fontsize=8)
else:
 a.axis('off');a.text(.05,.9,'Official Dynamic U = NULL\nOfficial paired Delta U = NULL\n\nFrozen routing-overhead acceptance failed\nP95 1.222 ms > 1 ms\n\n2 / 6 runs measured; P2/P3 not executed\nNo three-pair mean or confidence interval',transform=a.transAxes,va='top',fontsize=12);a.set_title('Inconclusive replication; no legal utility contrast')
if s:a.text(.02,.98,f'Mean {s["mean_delta_U"]:+.5f}\n95% t CI [{s["CI95_low"]:+.5f}, {s["CI95_high"]:+.5f}]\nn=3 pairs, df=2',transform=a.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
a=axs[0,1];groups=['text','asr','visual_qa','other'];width=.22
if valid:
 for i,pair in enumerate(range(1,4)):
  vals=[sum(float(v['L_P_delta_U']) for v in d if v['L_P_delta_U'] and int(v['pair'])==pair and (v['task']==t if t!='other' else v['task'] not in ['text','asr','visual_qa'])) for t in groups]
  a.bar(np.arange(4)+(i-1)*width,vals,width,color=colors[i],label=f'Pair {pair}')
 a.set_xticks(range(4),['Text','ASR','VQA','Other tasks']);a.set_ylabel('Contribution to Delta U (L + P)');a.set_title('All tasks retained; Text sensitivity is diagnostic');a.legend(fontsize=8)
else:
 a.bar(range(len(d)),[float(v['D_p95_ms'])-float(v['S_p95_ms']) for v in d]);a.set_xticks(range(len(d)),[v['task'] for v in d],rotation=25);a.set_ylabel('Measured E2E P95 change, ms');a.set_title('Full-cohort latency diagnostic; invalid utility comparison')
a.axhline(0,color='gray',lw=.8)
a=axs[1,0];xx=[];labels=[];queue=[];service=[];p95=[]
for pair in range(1,4):
 for arm in ['S','D']:
  rows=[v for v in paths if v['run_id']==f'P{pair}_{arm}' and v['task']=='visual_qa']
  if rows:
   v=rows[0];xx.append(len(xx));labels.append(f'{pair}{arm}');service.append(json.loads(v['service'])['mean']);queue.append(json.loads(v['wait'])['mean']);p95.append(json.loads(v['e2e'])['p95'])
a.bar(xx,service,label='Mean handler service');a.bar(xx,queue,bottom=service,label='Mean native queue wait');a.plot(xx,p95,'ko-',label='P95 E2E (separate statistic)');a.set_xticks(xx,labels);a.set_ylabel('VQA latency, ms');a.set_title('VQA service, queue and E2E are distinct');a.legend(fontsize=8)
a=axs[1,1];a.bar(np.arange(len(q)),[float(v['S_wait_reduction_mean_ms']) for v in q],color=colors[:len(q)]);a.axhline(0,color='gray',lw=.8);a.set_xticks(np.arange(len(q)),[f'Pair {v["pair"]}\nG ASR n={v["diverted_G_n"]}' for v in q]);a.set_ylabel('Matched retained-S wait reduction, ms');a.set_title('Whisper queue relief (diagnostic when invalid)')
fig.suptitle('Module32C: '+g['gate']+'\nFrozen policy; same sample pool; '+str(len(p))+' matched pair(s) measured',fontsize=12)
fig.savefig(out/'independent_replication.png',dpi=180);fig.savefig(out/'independent_replication.pdf');print('PLOTS_SAVED')
