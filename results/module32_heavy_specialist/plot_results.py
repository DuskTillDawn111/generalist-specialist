import sys
from pathlib import Path
OUT=Path('/root/autodl-tmp/generalist-specialist/results/module32_heavy_specialist');sys.path[:0]=[str(OUT),str(OUT.parents[1])]
from m32.common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
obs=read(OUT/'run_observations.json');by={o['run_id']:o for o in obs}
if len(obs)!=3 or not all(o['eligible'] for o in obs):raise SystemExit('No complete legal three-arm plot')
plt.rcParams.update({'font.size':10,'figure.dpi':160})
fig,ax=plt.subplots(1,3,figsize=(14,4))
ids=['A0','A1','A2'];x=np.arange(3)
sv=[];wt=[]
for a in ids:
 p=next(p for p in by[a]['paths'] if p['task']=='asr' and p['model']==S);sv.append(p['service']['mean']/1000);wt.append(p['wait']['mean']/1000)
ax[0].bar(x,sv,label='S execution');ax[0].bar(x,wt,bottom=sv,label='S queue wait');ax[0].set(xticks=x,xticklabels=['0%','5%','10%'],ylabel='Mean seconds',title='Native Specialist ASR');ax[0].legend()
cs=['T','L','P','M'];xx=np.arange(2);pos=np.zeros(2);neg=np.zeros(2)
for k in cs:
 vals=np.array([by[a]['components'][k+'_contribution']-by['A0']['components'][k+'_contribution'] for a in ids[1:]])
 ax[1].bar(xx,vals,bottom=np.where(vals>=0,pos,neg),label=k)
 pos+=np.maximum(vals,0);neg+=np.minimum(vals,0)
ax[1].axhline(.01,color='black',linestyle=':',label='+0.01 gate');ax[1].set(xticks=xx,xticklabels=['5%','10%'],ylabel='Delta utility',title='Frozen Module20A utility');ax[1].legend()
tasks=list(by['A0']['per_task']);xx=np.arange(len(tasks));_,refs=__import__('module25.common',fromlist=['utility_inputs']).utility_inputs()
for j,a in enumerate(ids[1:]):
 values=[(by[a]['per_task'][t]['p95_ms']-by['A0']['per_task'][t]['p95_ms'])/refs[t] for t in tasks]
 ax[2].bar(xx+(j-.5)*.36,values,width=.36,label=a)
ax[2].axhline(0,color='black',linewidth=.7);ax[2].set(xticks=xx,xticklabels=[t.replace('visual_qa','VQA') for t in tasks],ylabel='P95 delta / frozen task reference',title='All six task tails');ax[2].legend();ax[2].tick_params(axis='x',rotation=35)
fig.suptitle('Module32: one matched exploratory trajectory; no independent replication')
fig.tight_layout();fig.savefig(OUT/'utility_and_queue_analysis.png');fig.savefig(OUT/'utility_and_queue_analysis.pdf');plt.close(fig)
print('PLOTS_SAVED')
