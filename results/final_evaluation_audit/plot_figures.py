"""Render existing audited data with matplotlib; CPU only; no model imports."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

O=Path(__file__).resolve().parent
def read(name):return json.loads((O/name).read_text())
def csvsave(name, rows):
    with (O/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def finish(fig,name):
    fig.savefig(O/(name+'.png'),dpi=220,bbox_inches='tight',facecolor='white')
    fig.savefig(O/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold','pdf.fonttype':42})
colors={'B0':'#656d78','B1_seed_1103':'#d6922a','B1_seed_2207':'#af6a15','B2':'#087f8c','B3':'#7962ac'}
methods=['B0','B1_seed_1103','B1_seed_2207','B2','B3']
labels=['Seed','Random\n1103','Random\n2207','Proposed','LP-only']
static=read('rq2_static_search.json')['records']
workloads=sorted({r['workload'] for r in static})
figure_data=[]
for r in static:
    for stage,k in [('search-time measurements','search_final'),('post-search confirmation','confirmation_final')]:
        v=r[k]
        figure_data.append(dict(workload=r['workload'],method=r['algorithm'],measurement_stage=stage,
            runtime_evaluations=r['runtime_evaluations'],mean_e2e_ms=v['mean'],sample_std_ms=v['sample_std'],
            repetitions=v['n'],individual_means=json.dumps(v['individual_means']),
            raw_sources=';'.join(s['path'] for s in v['raw_sources'])))
csvsave('figure_static_search_data.csv',figure_data)
csvsave('figure_search_efficiency_data.csv',figure_data)
fig,axes=plt.subplots(1,3,figsize=(13.1,4.6),sharey=True)
for ax,w in zip(axes,workloads):
    for j,m in enumerate(methods):
        r=next(r for r in static if r['workload']==w and r['algorithm']==m)
        for off,k,marker,filled in [(-.12,'search_final','o',False),(.12,'confirmation_final','D',True)]:
            v=r[k];color=colors[m]
            ax.errorbar(j+off,v['mean'],yerr=v['sample_std'],fmt=marker,color=color,
                mfc=color if filled else 'white',ms=6,capsize=3,lw=1.2,zorder=3)
            ax.scatter(np.full(v['n'],j+off),v['individual_means'],s=10,color=color,alpha=.3,zorder=2)
    ax.set_title(w.split('_')[0]);ax.set_xticks(range(5),labels);ax.grid(axis='y',alpha=.15)
    ax.set_ylim(0,365)
axes[0].set_ylabel('Mean E2E (ms)')
handles=[plt.Line2D([],[],marker='o',mfc='white',color='#333',ls='',label='Search (3 repetitions)'),
         plt.Line2D([],[],marker='D',color='#333',ls='',label='Independent confirmation (2 repetitions)')]
fig.legend(handles=handles,loc='upper center',ncol=2,bbox_to_anchor=(.5,.94),frameon=False)
fig.suptitle('Static deployment search across W1 / W2 / W3',y=1.01,fontweight='bold')
fig.text(.5,-.03,'Error bars: between-repetition sample std, not confidence intervals. Policies can reuse identical search-cache measurements.',ha='center',fontsize=9)
fig.subplots_adjust(top=.79,wspace=.13)
finish(fig,'fig_static_search')

fig,axes=plt.subplots(1,3,figsize=(12.9,4.7),sharey=True)
markers={'B0':'s','B1_seed_1103':'^','B1_seed_2207':'v','B2':'o','B3':'D'}
for ax,w in zip(axes,workloads):
    for m in methods:
        r=next(r for r in static if r['workload']==w and r['algorithm']==m);v=r['confirmation_final']
        ax.scatter(r['runtime_evaluations'],v['mean'],marker=markers[m],color=colors[m],s=78,
            facecolors='none' if m=='B3' else colors[m],linewidths=1.7,zorder=4 if m=='B3' else 3)
        ax.errorbar(r['runtime_evaluations'],v['mean'],yerr=v['sample_std'],fmt='none',color=colors[m],capsize=3,alpha=.6)
    ax.set_title(w.split('_')[0]);ax.set_xticks([1,2,3,4,5,6]);ax.set_xlim(.5,6.6)
    ax.set_xlabel('Runtime state evaluations (including seed)');ax.grid(alpha=.15)
axes[0].set_ylabel('Final mean E2E: confirmation (ms)')
fig.legend(handles=[plt.Line2D([],[],marker=markers[m],color=colors[m],mfc='none' if m=='B3' else colors[m],ls='',label=l.replace('\n',' '))
                    for m,l in zip(methods,labels)],loc='upper center',ncol=5,bbox_to_anchor=(.5,.95),frameon=False)
fig.suptitle('Search cost versus independently confirmed latency',y=1.01,fontweight='bold')
fig.text(.5,-.03,'All searched policies use the same 6-state budget; this endpoint plot does not establish a cost frontier. Cache accesses are not new physical GPU runs.',ha='center',fontsize=8.5)
fig.subplots_adjust(top=.79,wspace=.13)
finish(fig,'fig_search_efficiency')

rq3=read('rq3_guidance.json');fig,ax=plt.subplots(figsize=(7.3,5.1))
for w,c,m in zip(workloads,['#087f8c','#d6922a','#7962ac'],['o','^','s']):
    points=[p for p in rq3['unique_transition_points'] if p['workload']==w]
    ax.scatter([p['delta_z'] for p in points],[p['delta_J_ms'] for p in points],color=c,marker=m,s=45,
               alpha=.8,edgecolors='white',linewidths=.4,label=f'{w.split("_")[0]} (n={len(points)})')
ax.axhline(0,color='#999',lw=.8);ax.axvline(0,color='#999',lw=.8)
ax.set_xlabel(r'LP pressure relief $\Delta z = z_{current}-z_{candidate}$')
ax.set_ylabel(r'Measured improvement $\Delta J$ (ms)')
ax.set_title('LP analytical relief versus measured improvement');ax.legend(frameon=False);ax.grid(alpha=.12)
fig.text(.5,-.035,'Unique current/candidate deployment transitions; evaluated candidates only. Search measurements; no fit or causal inference.',ha='center',fontsize=8.4)
finish(fig,'fig_lp_vs_runtime')

rq4=read('rq4_dynamic_adaptation.json');dyn=rq4['records'];transitions=sorted({r['transition'] for r in dyn})
translabels=['W1 → W2','W2 → W3','W3 → W1'];dynamic_data=[]
for r in dyn:
    for stage,k in [('search-time measurements','search_final'),('post-search confirmation','confirmation_final')]:
        if r[k] is None:continue
        rc=r['reconfiguration']
        dynamic_data.append(dict(transition=r['transition'],method=r['method'],measurement_stage=stage,
            runtime_evaluations=r['search_cost']['valid_runtime_evaluations'],mean_e2e_ms=r[k]['mean'],
            weighted_p95_ms=r[k]['weighted_p95'],reconfiguration_time_seconds=rc['measured_switch_time_ms']/1000 if rc['measured_switch_time_ms'] is not None else None,
            accepted_switches=rc['accepted_switches'],action=r['action'],repetitions=r[k]['n']))
csvsave('figure_dynamic_adaptation_data.csv',dynamic_data)
fig,axes=plt.subplots(1,3,figsize=(13.8,4.7));xx=np.arange(3)
for method,color,off in [('D1','#d6922a',-.18),('D2','#087f8c',.18)]:
    rs=[next(r for r in dyn if r['transition']==t and r['method']==method) for t in transitions]
    lab='Cold' if method=='D1' else 'Warm'
    evals=[r['search_cost']['valid_runtime_evaluations'] for r in rs]
    timing=[r['reconfiguration']['measured_switch_time_ms'] for r in rs]
    axes[0].bar(xx+off,evals,width=.33,color=color,label=lab)
    axes[1].bar(xx+off,[v/1000 if v is not None else 0 for v in timing],width=.33,color=color,label=lab)
    for j,v in enumerate(timing):
        if v is None:axes[1].text(j+off,.35,'NO_SWITCH\nno timing',ha='center',fontsize=7.4,color=color)
    for j,r in enumerate(rs):
        for k,marker,filled,offset in [('search_final','o',False,-.045),('confirmation_final','D',True,.045)]:
            v=r[k]
            axes[2].errorbar(j+off+offset,v['mean'],yerr=v['sample_std'],fmt=marker,color=color,mfc=color if filled else 'white',capsize=2,ms=5)
for ax,title,ylabel in zip(axes,['Search cost','Isolated accepted-path reconfiguration','Final serving performance'],
                           ['Runtime state evaluations','Seconds (cold includes seed reset)','Mean E2E (ms)']):
    ax.set_title(title,fontsize=10.5);ax.set_ylabel(ylabel);ax.set_xticks(xx,translabels);ax.grid(axis='y',alpha=.12)
axes[0].set_ylim(0,7.4);axes[1].set_ylim(0,13.8);axes[2].set_ylim(25,67)
for i,(c,w) in enumerate(zip([6,6,6],[5,3,3])):
    axes[0].text(i-.18,c+.12,str(c),ha='center',fontsize=9);axes[0].text(i+.18,w+.12,str(w),ha='center',fontsize=9)
fig.legend(handles=[plt.Rectangle((0,0),1,1,color='#d6922a',label='Cold'),plt.Rectangle((0,0),1,1,color='#087f8c',label='Warm'),
    plt.Line2D([],[],marker='o',mfc='white',color='#333',ls='',label='Search (n=3)'),
    plt.Line2D([],[],marker='D',color='#333',ls='',label='Confirmation (n=2)')],loc='upper center',ncol=4,bbox_to_anchor=(.5,.97),frameon=False)
fig.suptitle('Cold versus warm dynamic adaptation',y=1.035,fontweight='bold')
fig.text(.5,-.04,'Evaluation saving: 7/18 = 38.89%. Isolated accepted-path time: 32.843 s / 3.344 s; these are different metrics, not wall-clock adaptation speedup.',ha='center',fontsize=9)
fig.subplots_adjust(top=.80,wspace=.32)
finish(fig,'fig_dynamic_adaptation')

trace=rq4['sequence_trace'];modelnames=['qwen','distilbert','paddleocr','resnet50','whisper']
matrix=np.array([[float(m in p['deployment'].split('+')) for p in trace] for m in modelnames])
fig,ax=plt.subplots(figsize=(10.2,4.3))
ax.imshow(matrix,cmap=matplotlib.colors.ListedColormap(['#f0f1f2','#087f8c']),vmin=0,vmax=1,aspect='auto')
ax.set_yticks(range(5),['Qwen','DistilBERT','PaddleOCR','ResNet50','Whisper'])
ax.set_xticks(range(4),[f'Phase {i}: {p["workload"]}' for i,p in enumerate(trace)])
for j,p in enumerate(trace):
    label='INITIAL' if j==0 else ('ACCEPTED SWITCH\nADD DistilBERT' if p['action']=='SWITCHED' else 'NO_SWITCH')
    ax.text(j,-.77,label,ha='center',va='center',fontsize=9,color='#087f8c' if j==1 else '#333')
    for i,m in enumerate(modelnames):
        ax.text(j,i,'resident' if matrix[i,j] else 'absent',ha='center',va='center',color='white' if matrix[i,j] else '#666',fontsize=10)
ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True)
ax.grid(which='minor',color='white',lw=3);ax.tick_params(which='minor',bottom=False,left=False)
ax.set_title('Warm sequence: W1 → W2 → W3 → W1',pad=55)
ax.set_xlabel('Discrete workload phases and final accepted deployment; phase duration is not shown')
fig.text(.5,-.04,'Accepted-path activation occurs after search freezes. This phase trace does not represent continuous-arrival or zero-downtime service.',ha='center',fontsize=9)
finish(fig,'fig_dynamic_trace')

save_metadata={'fig_static_search':'figure_static_search_data.csv','fig_search_efficiency':'figure_search_efficiency_data.csv',
 'fig_lp_vs_runtime':'figure_lp_vs_runtime_data.csv','fig_dynamic_adaptation':'figure_dynamic_adaptation_data.csv','fig_dynamic_trace':'figure_dynamic_trace_data.csv'}
(O/'figure_manifest.json').write_text(json.dumps({'figures':save_metadata,'no_fitted_regression':True,'error_bars':'sample std, not confidence intervals',
 'primary_static_presentation':'search and confirmation separately displayed','primary_dynamic_presentation':'search and confirmation separately displayed',
 'routing_percentile_semantics':'weighted task P95, not pooled percentile','pdf_versions_available':True},indent=2)+'\n')
print('Generated five PNG figures, five vector PDFs and figure data files.')
