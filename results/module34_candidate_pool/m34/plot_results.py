from .common import *
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def finish(fig,name):
 p=OUT/'figures';p.mkdir(exist_ok=True);fig.savefig(p/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(p/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def main():
 ds=read(OUT/'optimizer_decisions.json');oo=read(OUT/'cpu_oracles.json');scenarios=['LOW18','ASR32B','TEXT_HALF_MID18'];methods=['CPU_U_ORACLE','V3_J_GRID_CONTROL','20A_U_GRID','FASTEST_PATH_STATIC','MIN_MEMORY_COVER_FASTEST'];labels=['U Oracle','V3 J grid control','20A U grid','Fastest-path','Min-memory + fastest']
 fig,axes=plt.subplots(1,3,figsize=(13,4));
 for ax,scenario in zip(axes,scenarios):
  a=np.zeros((len(methods),3));txt=[]
  for i,m in enumerate(methods):
   row=[]
   for j,p in enumerate(['A','B','C']):
    r=next(r for r in ds if r['scenario']==scenario and r['pool']==p and r['algorithm']==m);a[i,j]={'G_ONLY':0,'MIXED':1}.get(r.get('structure'),2);row.append(r.get('structure','FAIL'))
   txt.append(row)
  ax.imshow(a,vmin=0,vmax=2,cmap=matplotlib.colors.ListedColormap(['#bad7e9','#abe1bb','#f5afa9']),aspect='auto');ax.set_xticks(range(3),['A (3)','B (5)','C (10)']);ax.set_yticks(range(len(labels)),labels if ax is axes[0] else ['']*len(labels));ax.set_title(scenario.replace('TEXT_HALF_MID18','TEXT10'))
  for i in range(len(methods)):
   for j in range(3):ax.text(j,i,'G' if txt[i][j]=='G_ONLY' else 'Mixed' if txt[i][j]=='MIXED' else 'Fail',ha='center',va='center',fontsize=9)
 fig.suptitle('Autonomous deployment selection in fixed finite CPU forecast domain');finish(fig,'deployment_selection')
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for ax,scenario in zip(axes,scenarios):
  arr=np.array([[next(r['regret'] for r in ds if r['scenario']==scenario and r['pool']==p and r['algorithm']==m) for p in ['A','B','C']] for m in methods[1:]])
  ax.imshow(np.log10(1+arr),cmap='YlOrRd',aspect='auto',vmin=0,vmax=np.log10(12));ax.set_xticks(range(3),['A','B','C']);ax.set_yticks(range(4),labels[1:] if ax is axes[0] else ['']*4);ax.set_title(scenario.replace('TEXT_HALF_MID18','TEXT10'))
  for i in range(4):
   for j in range(3):ax.text(j,i,f'{arr[i,j]:.4g}',ha='center',va='center',fontsize=9,color='white' if arr[i,j]>3 else 'black')
 fig.suptitle('Predicted U regret relative to exhaustive finite U Oracle');finish(fig,'oracle_regret')
 times=list(csv.DictReader((OUT/'optimizer_runtime_benchmark.csv').open()));fig,axes=plt.subplots(1,3,figsize=(13,4))
 for ax,scenario in zip(axes,scenarios):
  xo=[o['evaluated_routes'] for o in oo if o['scenario']==scenario];yo=[o['seconds'] for o in oo if o['scenario']==scenario];ax.loglog(xo,yo,'o-',label='Exhaustive U Oracle')
  for m,l in [('20A_U_GRID','20A U grid'),('V3_J_GRID_CONTROL','V3 J grid control')]:
   yy=[np.median([float(r['seconds']) for r in times if r['pool']==p and r['scenario']==scenario and r['algorithm']==m]) for p in ['A','B','C']];ax.loglog(xo,yy,'o-',label=l)
  ax.set_title(scenario.replace('TEXT_HALF_MID18','TEXT10'));ax.set_xlabel('Memory/coverage-admissible route configurations');ax.grid(alpha=.25)
 axes[0].set_ylabel('CPU decision / enumeration seconds');axes[0].legend(fontsize=8);finish(fig,'search_time_space')
 if (OUT/'gpu_contrasts.json').exists():
  cs=read(OUT/'gpu_contrasts.json');fig,axes=plt.subplots(1,3,figsize=(12,4),squeeze=False)
  for ax,c in zip(axes[0],cs):
   ss=c['delta_statistics'];mean=ss['mean'];ci=ss['ci95']
   if mean is not None:ax.errorbar(0,mean,yerr=[[mean-ci[0]],[ci[1]-mean]] if ci else None,fmt='o',color='#1565c0',capsize=4)
   ax.scatter(1,c['predicted_delta_U'],marker='x',color='#c62828');ax.axhline(0,color='black',lw=1);ax.set_xticks([0,1],['Observed','Predicted']);ax.set_title(c['group'].replace('TEXT_HALF_MID18','TEXT10'));ax.grid(axis='y',alpha=.2)
   ax.text(.02,.98,f"n={ss['n']}; Holm p={c.get('holm_adjusted_p', 'pending')}",transform=ax.transAxes,va='top',fontsize=7)
  axes[0,0].set_ylabel('U(Oracle-selected plan) − U(competitor)');fig.suptitle('Independent matched GPU traces; 95% t CI; each panel has its own U scale');finish(fig,'gpu_optimizer_heuristic')
  rrall=read(OUT/'gpu_results_table.json');groups=sorted({r['group'] for r in rrall});fig,axes=plt.subplots(1,len(groups),figsize=(4*len(groups),4),squeeze=False)
  for ax,group in zip(axes[0],groups):
   for i,arm in enumerate(['ORACLE','COMPETITOR']):
    values=[r['U'] for r in rrall if r['group']==group and r['stage']=='INDEPENDENT' and r['arm']==arm and r['eligible']];dev=[r['U'] for r in rrall if r['group']==group and r['stage']=='DEVELOPMENT' and r['arm']==arm and r['eligible']]
    if values:ax.scatter(np.full(len(values),i)+np.linspace(-.07,.07,len(values)),values,color='#1565c0',alpha=.6);ax.scatter(i,np.mean(values),marker='D',color='black')
    if dev:ax.scatter(i,dev[0],marker='x',color='#c62828')
   ax.set_xticks([0,1],['20A / U Oracle','Fastest (Qwen)'] if group.startswith('C_') else ['V3 / U Oracle','20A / fastest'],rotation=10);ax.set_title(group.replace('TEXT_HALF_MID18','TEXT10'));ax.grid(axis='y',alpha=.2)
  axes[0,0].set_ylabel('Observed GPU U');fig.suptitle('System utility: blue independent seeds, red development, black mean');finish(fig,'gpu_system_utility_comparison')
  rr=read(OUT/'gpu_results_table.json');rr=[r for r in rr if r['eligible']];fig,axes=plt.subplots(1,2,figsize=(11,5));ax=axes[0];ax.scatter([r['predicted_U'] for r in rr],[r['U'] for r in rr],c=['#1565c0' if r['stage']=='INDEPENDENT' else '#aaaaaa' for r in rr]);lim=[min([r['U'] for r in rr]+[r['predicted_U'] for r in rr]),max([r['U'] for r in rr]+[r['predicted_U'] for r in rr])] if rr else [-1,0];ax.plot(lim,lim,'k--');ax.set_xlabel('Predicted U');ax.set_ylabel('Observed GPU U');ax.set_title('Unchanged predictor: valid configurations');ax=axes[1]
  if rr:ax.scatter([-r['predicted_U'] for r in rr],[-r['U'] for r in rr],c=['#1565c0' if r['stage']=='INDEPENDENT' else '#aaaaaa' for r in rr]);costs=[-v for v in lim];ax.plot(costs,costs,'k--')
  ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Predicted penalty −U');ax.set_ylabel('Observed GPU penalty −U');ax.set_title('Same data, log penalty axes');finish(fig,'prediction_error')
 print('FIGURES_WRITTEN',flush=True)
if __name__=='__main__':main()
