"""Standalone scientific figures for the finite benchmark, not global claims."""
from .common import *
import numpy as np
os.environ['MPLCONFIGDIR']=str(OUT/'work/cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    candidates=read(OUT/'gpu_candidates.json');by={c['candidate_id']:c for c in candidates}
    dev=read(OUT/'development_observations.json');best=max((o for o in dev if o['U'] is not None),key=lambda o:o['U'])
    fig,ax=plt.subplots(1,2,figsize=(11,4.2),layout='constrained');colors={0.:'tab:blue',.5:'tab:orange',1.:'tab:red'}
    for index,c in enumerate(candidates,1):
        o=next((v for v in dev if v['candidate_id']==c['candidate_id']),None)
        if not o or o['U'] is None:continue
        frac=c['routing']['asr'].get('phi4mm',0)/sum(c['routing']['asr'].values());ocr=next(m for m in c['deployment'] if m.startswith('baudm/parseq:parseq@'));n=c['deployment'][ocr]
        ax[0].scatter(c['predicted_U'],o['U'],color=colors[frac],marker='o' if n==1 else '^',s=65,edgecolors='black' if o['candidate_id']==best['candidate_id'] else 'none')
        offsets={1:(8,10),2:(-25,-2),3:(10,8),4:(10,-3),5:(15,-14),6:(12,-3),7:(-15,-15),8:(-12,15)}
        ax[0].annotate(str(index),(c['predicted_U'],o['U']),xytext=offsets[index],textcoords='offset points',fontsize=9,arrowprops=dict(arrowstyle='-',color='gray',lw=.5))
    a=ax[0];lim=a.get_xlim();a.plot(lim,lim,':',color='gray',label='Prediction = observation');a.set_xlabel('Frozen20A predicted U');a.set_ylabel('Development measured U');a.set_title('All8 finite configurations; valid U only');a.grid(alpha=.2)
    handles=[plt.Line2D([],[],marker='o',linestyle='',color=colors[f],label=f'ASR Phi fraction {f:g}') for f in colors]
    a.legend(handles=handles,fontsize=8,loc='best');a.text(.43,.06,'Circle: OCR1; triangle: OCR2\nBlack edge: observed empirical maximum\nPoint maxima do not imply strict population rankings',transform=a.transAxes,fontsize=7,va='bottom')
    oracles=read(OUT/'cpu_oracles.json');rt=list(csv.DictReader((OUT/'native_optimizer_speedup.csv').open()));x=np.arange(len(oracles));width=.2
    a=ax[1];a.bar(x-1.5*width,[o['seconds'] for o in oracles],width,label='Exact finite enumeration')
    for offset,alg,label in [(-.5,'20A_NATIVE_CONTINUOUS','20A native continuous'),(.5,'20A_FINITE_DOMAIN_ADAPTER','20A finite adapter'),(1.5,'FORMAL_V3_CPU_BACKEND','Formal V3 CPU backend')]:
        vals=[float(next(r['seconds'] for r in rt if r['pool']==o['pool'] and r['load']==o['load'] and r['algorithm']==alg)) for o in oracles]
        a.bar(x+offset*width,vals,width,label=label)
    a.set_yscale('log');a.set_xticks(x,[o['pool']+'\n'+o['load'] for o in oracles],fontsize=8);a.set_ylabel('CPU decision wall time (s, log scale)');a.set_title('First recorded CPU runs, before GPU');a.legend(fontsize=7);a.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'module33_benchmark.png',dpi=200);fig.savefig(OUT/'module33_benchmark.pdf');plt.close(fig)
    save('plot_candidate_labels.json',{str(i):c['candidate_id'] for i,c in enumerate(candidates,1)})
    print('FIGURES_WRITTEN',flush=True)
if __name__=='__main__':main()
