import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('/root/autodl-tmp/generalist-specialist/results/module32b_vqa_aware_routing');read=lambda n:json.loads((p/n).read_text());o={x['run_id']:x for x in read('run_observations.json')};g=read('module32b_gate.json')
if len(o)!=3 or not all(x['eligible'] for x in o.values()):raise SystemExit('No complete legal three-arm plot')
ids=['A0','A1','A2'];labels=['S-only','Fixed10','Dynamic'];x=np.arange(3);fig,ax=plt.subplots(1,4,figsize=(17,4));plt.rcParams.update({'font.size':9})
v=[next(x for x in o[a]['paths'] if x['task']=='visual_qa') for a in ids];sv=np.array([x['service']['mean']/1000 for x in v]);qw=np.array([x['wait']['mean']/1000 for x in v]);ax[0].bar(x,sv,label='Mean handler execution');ax[0].bar(x,qw,bottom=sv,label='Mean native queue wait');ax[0].plot(x,[r['e2e']['p95']/1000 for r in v],'ko-',label='P95 E2E (separate statistic)');ax[0].set(xticks=x,xticklabels=labels,ylabel='Seconds',title='Phi VQA: distinct timing definitions');ax[0].legend(fontsize=7)
vals=[o[a]['U']-o['A0']['U'] for a in ids[1:]];ax[1].bar([0,1],vals);ax[1].axhline(.01,color='black',linestyle=':');ax[1].axhline(0,color='grey',linewidth=.7);ax[1].set(xticks=[0,1],xticklabels=labels[1:],ylabel='Delta U vs S-only',title='Frozen six-task utility')
s=[next(x for x in o[a]['paths'] if x['task']=='asr' and x['model']=='openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1') for a in ids];ax[2].bar(x,[r['wait']['mean']/1000 for r in s]);ax[2].set(xticks=x,xticklabels=labels,ylabel='Native mean S wait, seconds',title='Whisper queue pressure')
for i,a in enumerate(ids):ax[2].text(i,s[i]['wait']['mean']/1000,f"G ASR n={o[a]['G_completed']}",ha='center',va='bottom',fontsize=8)
tasks=list(o['A0']['per_task']);xx=np.arange(6)
for j,a in enumerate(ids[1:]):ax[3].bar(xx+(j-.5)*.36,[(o[a]['per_task'][t]['p95_ms']-o['A0']['per_task'][t]['p95_ms']) for t in tasks],width=.36,label=labels[j+1])
ax[3].axhline(0,color='black',linewidth=.7);ax[3].set(xticks=xx,xticklabels=[t.replace('visual_qa','VQA') for t in tasks],ylabel='P95 E2E delta, ms',title='All six task tails');ax[3].tick_params(axis='x',rotation=40);ax[3].legend(fontsize=8)
fig.suptitle('Module32B: '+g['gate']+' | One exploratory matched trajectory, no independent replication',fontsize=11);fig.tight_layout();fig.savefig(p/'vqa_aware_exploration.png',dpi=160);fig.savefig(p/'vqa_aware_exploration.pdf');plt.close(fig);print('PLOTS_SAVED')
