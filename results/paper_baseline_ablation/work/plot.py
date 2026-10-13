import sys,os,json,csv,math,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/generalist-specialist');OUT=Path(os.environ.get('PAPER_EVAL_OUT',str(ROOT/'results/paper_baseline_ablation')));sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'results/module35_utility_routing'),str(ROOT)]
sys.stdout=(OUT/'work/plot.log').open('a',buffering=1);sys.stderr=sys.stdout
from m35 import isolation
isolation.ROOT=OUT;isolation.install('supervisor')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
read=lambda p:json.loads(Path(p).read_text())
base=read(OUT/'baseline_cpu_results.json');abl=read(OUT/'ablation_cpu_results.json');eff=read(OUT/'solver_efficiency.json');gpu=read(OUT/'gpu_baseline_results.json');summ=read(OUT/'gpu_paired_summary.json')
scens=['LOW18','ASR32B','TEXT_HALF_MID18','HOLDOUT_TEXT3'];labels=['LOW/A','LOW/B','LOW/C','ASR/A','ASR/B','ASR/C','TEXT10/A','TEXT10/B','TEXT10/C','TEXT3/A','TEXT3/B','TEXT3/C'];cells=[(s,p) for s in scens for p in ['A','B','C']]
def finish(fig,name):
 fig.savefig(OUT/f'paper_figures/{name}.png',dpi=240,bbox_inches='tight');fig.savefig(OUT/f'paper_figures/{name}.pdf',bbox_inches='tight');plt.close(fig)
methods=['CPU_U_ORACLE','U_ALIGNED_DP','20A_U_GRID','V3_J_GRID_CONTROL','FASTEST_PATH_STATIC','MIN_MEMORY_COVER_FASTEST']
names=['Finite U Oracle','Module 35','20A grid control','V3 J grid control','Fastest path','Min-memory + fastest']
z=np.array([[next(r['regret'] for r in base if r['scenario']==s and r['pool']==p and r['algorithm']==m) for s,p in cells] for m in methods])
fig,ax=plt.subplots(figsize=(12.8,5.3));im=ax.imshow(np.log1p(z),cmap='YlOrRd',aspect='auto',vmin=0,vmax=np.log1p(z.max()))
for i in range(len(methods)):
 for j in range(len(cells)):
  value=z[i,j];ax.text(j,i,'0' if abs(value)<1e-8 else f'{value:.3f}',ha='center',va='center',fontsize=8,color='white' if value>7 else 'black')
ax.set_xticks(range(12),labels,rotation=40,ha='right');ax.set_yticks(range(6),names);ax.axvline(8.5,color='#3182bd',linewidth=2);ax.set_title('Figure 1. Predicted U regret in the same finite domain (K=1, binary replicas)')
cbar=fig.colorbar(im,ax=ax,pad=.02);cbar.set_label('Color: log(1 + regret); cells show actual U regret')
fig.subplots_adjust(bottom=.27)
fig.text(.13,.02,'Nine development cells | TEXT3: one historically frozen holdout workload, three pools.\nNative continuous methods remain in source CSV; strict regret undefined.',fontsize=8)
finish(fig,'figure1_regret')
fig,axes=plt.subplots(2,1,figsize=(12.8,7.3),sharex=True)
for ax,full,variant,title in [(axes[0],'A_FIXED_20A_DEPLOYMENT_FULL_ROUTE','A_OLD_ROUTE_COVERAGE_DIAGNOSTIC','A. Fixed original-20A deployment: complete routing vs old generated candidates'),(axes[1],'B_FIXED_M35_DEPLOYMENT_FULL_ROUTE','B_FIXED_M35_DEPLOYMENT_FASTEST','B. Fixed Module-35 deployment: U routing vs legal fastest routing')]:
 ys=[]
 for name in [full,variant]:ys.append([next(r['predicted_U'] for r in abl if r['scenario']==s and r['pool']==p and r['ablation']==name) for s,p in cells])
 x=np.arange(12);ax.plot(x,ys[0],'o-',color='#2166ac',label='U-aligned complete routing');ax.plot(x,ys[1],'s--',color='#d6604d',label='Old candidate set' if variant.startswith('A') else 'Fastest routing')
 ax.axvline(8.5,color='#777777',linewidth=1,linestyle=':');ax.set_title(title,loc='left',fontsize=10);ax.set_ylabel('Predicted total utility U');ax.grid(axis='y',alpha=.2);ax.legend(loc='lower left')
axes[1].set_xticks(range(12),labels,rotation=40,ha='right');fig.suptitle('Figure 2. Legal fixed-deployment routing diagnostics',fontsize=12,y=.98)
fig.text(.125,.005,'Not a global deployment-search switch ablation. Historical holdout is reused for diagnosis; no blind ablation claim. Co-location component absent (C skipped).',fontsize=8)
fig.tight_layout(rect=(0,.025,1,.95));finish(fig,'figure2_ablation_utility')
colors=['#2166ac','#1b9e77','#d95f02','#7570b3'];fig,ax=plt.subplots(figsize=(8.8,5.5))
for scen,color in zip(scens,colors):
 for method,style in [('FINITE_U_ONLY_ORACLE','o-'),('MODULE35_FROZEN','s--')]:
  rr=sorted([r for r in eff if r['scenario']==scen and r['method']==method],key=lambda r:r['oracle_candidates']);x=[r['oracle_candidates'] for r in rr];y=[r['seconds_median'] for r in rr]
  ax.plot(x,y,style,color=color,label=f"{scen.replace('TEXT_HALF_MID18','TEXT10').replace('HOLDOUT_','')} / {'U-only Oracle' if 'ORACLE' in method else 'Module 35'}")
  ax.errorbar(x,y,yerr=np.array([[r['seconds_median']-r['seconds_min'] for r in rr],[r['seconds_max']-r['seconds_median'] for r in rr]]),fmt='none',color=color,capsize=3,alpha=.6)
ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks([8,32,8154],['8 (A: 3 models)','32 (B: 5 models)','8,154 (C: 10 models)']);ax.set_xlabel('Complete route candidates after memory and coverage filtering');ax.set_ylabel('CPU solve time (seconds)');ax.set_title('Figure 3. Same-domain U-only enumeration and frozen Module 35');ax.grid(alpha=.2);ax.legend(fontsize=7.8,ncol=2,loc='upper left')
fig.text(.13,.005,'Three cold-cache repetitions: median and min/max. K=1 only. Input loading and result serialization excluded; final prediction validation included.',fontsize=8)
finish(fig,'figure3_solver_time')
fig,axes=plt.subplots(1,2,figsize=(11.5,4.8));groups=[r['group'] for r in summ];x=np.arange(3)
for arm,offset,color,label in [('ORACLE',-.18,'#2166ac','Module 35 selected config'),('COMPETITOR',.18,'#d6604d','Fastest selected config')]:
 means=[float(np.mean([r['U'] for r in gpu if r['group']==g and r['arm']==arm])) for g in groups]
 axes[0].bar(x+offset,means,width=.34,color=color,label=label)
 for i,g in enumerate(groups):axes[0].scatter([i+offset]*3,[r['U'] for r in gpu if r['group']==g and r['arm']==arm],s=12,c='black',alpha=.7)
axes[0].set_xticks(x,['A / TEXT10','B / TEXT10','C / ASR']);axes[0].set_ylabel('Observed total utility U');axes[0].legend(fontsize=8);axes[0].set_title('Raw paired arms: means and three independent traces')
y=[r['delta_U'] for r in summ];errors=np.array([[r['delta_U']-r['CI95_low'] for r in summ],[r['CI95_high']-r['delta_U'] for r in summ]])
axes[1].errorbar(x,y,yerr=errors,fmt='o',color='#2166ac',capsize=6);axes[1].axhline(0,color='black',linewidth=1);axes[1].set_xticks(x,['A / TEXT10','B / TEXT10','C / ASR']);axes[1].set_ylabel('Paired U difference: Module 35 config − fastest config');axes[1].set_title('95% paired t intervals (n=3)')
for i,r in enumerate(summ):axes[1].annotate('supported*' if r['verdict'].startswith('SUPPORTED') else 'inconclusive',(i,r['CI95_high']),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
axes[1].set_xlim(-.5,2.5);axes[1].set_ylim(-5,max(r['CI95_high'] for r in summ)+8);fig.suptitle('Figure 4. Audited historical within-pool GPU configuration comparisons',fontsize=12)
fig.text(.1,.005,'*Original three-comparison Holm correction. Shared exact-config measurements; not independent runs for each algorithm alias. B→C expansion excluded.',fontsize=8)
fig.tight_layout(rect=(0,.035,1,.94));finish(fig,'figure4_gpu_utility')
# Final figureacceptance and output hashes; no scientific evidencefields changed.
for name in ['figure1_regret','figure2_ablation_utility','figure3_solver_time','figure4_gpu_utility']:
 for ext in ['png','pdf','csv']:assert (OUT/f'paper_figures/{name}.{ext}').stat().st_size>0
 (OUT/f'paper_figures/{name}_caption.md').write_text({'figure1_regret':'Finite-domain predicted U regret for every fixed cell. Full native-domain outcomes and failures are preserved outside strict regret. Zero values are actual ties.',
 'figure2_ablation_utility':'Fixed-deployment routing diagnostics, not global single-variable switch ablations. Both arms retain hard constraints. C co-location ablation is unavailable and skipped.',
 'figure3_solver_time':'U-only same-domain enumeration and Module35 solve times, three fresh-cache repetitions, median and min/max. Scope K=1 and binary replica domain; absolute seconds shown.',
 'figure4_gpu_utility':'Historical34 within-pool paired configuration measurements reattributed by exact fullconfiguration match. Only A supports a difference after original Holm correction; B/C inconclusive. n=3. No full-GPU-Oracle claim.'}[name]+'\n')
gate=read(OUT/'final_gate.json');gate['stages']['F']='COMPLETE';gate['PAPER_ASSETS_READY']=True;gate['figure_formats']=['PNG','PDF','CSV','complete_raw_data_index'];(OUT/'final_gate.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2)+'\n')
paths=[p for p in OUT.rglob('*') if p.is_file() and 'work' not in p.relative_to(OUT).parts and p.name not in ['artifact_sha256.json','result_freeze.json']]
refs=[]
for p in paths:refs.append(dict(path=str(p.relative_to(OUT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
(OUT/'artifact_sha256.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2)+'\n');(OUT/'result_freeze.json').write_text(json.dumps(dict(status='FROZEN_STOP',artifacts=len(refs),next_experiment=False,strict_global_ablation_A_B=False,scope='FiniteK1 baseline; legal fixeddeploymentdiagnostics; rawGPUreuse; U-onlyefficiency'),indent=2)+'\n')
print('PAPER_ASSETS_READY',len(refs))
