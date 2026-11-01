"""Plot audited, eligible native observations from the development ledger."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def receipt(path):
    return dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    manifest=json.loads((args.comparison.parent/'manifest.json').read_text())
    assert receipt(args.comparison)==manifest['report']
    report=json.loads(args.comparison.read_text())
    rows=[r for r in report['observations'] if r['audit_status']=='VERIFIED_TERMINAL_EVIDENCE' and
          r['utility'].get('eligible') and r['memory_pass'] and r['energy_pass'] and r['shutdown_pass']]
    if not rows:raise ValueError('No eligible audited observations to plot')
    rows.sort(key=lambda r:(r['seed'],r['label']))
    labels=[]
    for row in rows:
        share=row['plan']['routing']['text'].get('generalist_2',0)/sum(row['plan']['routing']['text'].values())
        if row['plan']['deployment']==dict(text=2,generalist_2=1,ocr=1):
            name='G2 share 1/32' if share else 'No split'
        elif set(row['plan']['deployment'])=={'text','image','ocr','asr'}:name='All specialists'
        else:name=row['label']
        labels.append(f"Seed {row['seed']}: {name}")
    fig,axes=plt.subplots(1,2,figsize=(12,max(3.5,len(rows)*.55+1.4)),gridspec_kw=dict(width_ratios=[1.3,1]))
    colors=['#9333ea','#94a3b8','#0284c7','#0d9488','#d97706','#dc2626'];left=[0.]*len(rows)
    for term,color in zip('QTLPME',colors):
        values=[-r['utility'][term+'_contribution'] for r in rows]
        axes[0].barh(range(len(rows)),values,left=left,color=color,label=term)
        left=[a+b for a,b in zip(left,values)]
    for i,row in enumerate(rows):axes[0].text(left[i],i,f"  U={row['utility']['U']:.4f}",va='center',fontsize=8)
    axes[0].set_yticks(range(len(rows)),labels);axes[0].invert_yaxis()
    axes[0].set_xlabel('Six weighted losses = -U (smaller is better)');axes[0].legend(ncol=6,fontsize=8,loc='lower right')
    axes[0].set_xlim(0,max(left)*1.30)
    means=[r['per_task_metrics']['text']['mean_e2e_latency_ms']/1000 for r in rows]
    p95=[r['per_task_metrics']['text']['p95_e2e_latency_ms']/1000 for r in rows]
    axes[1].barh([i-.17 for i in range(len(rows))],means,height=.32,label='Mean',color='#0284c7')
    axes[1].barh([i+.17 for i in range(len(rows))],p95,height=.32,label='P95',color='#0d9488')
    axes[1].set_yticks(range(len(rows)),labels);axes[1].invert_yaxis();axes[1].set_xlabel('Actual text E2E latency (s)');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    pending=sum(r['audit_status']!='VERIFIED_TERMINAL_EVIDENCE' for r in report['observations'])
    fig.suptitle('Audited native development observations — text 1.85 rps',fontsize=12)
    fig.text(.5,.015,f'Development only. {pending} declared conditions lack verified terminals. No algorithmic/full-pool optimum claim.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.93));args.output.mkdir(parents=True,exist_ok=False)
    for suffix in ['png','pdf']:fig.savefig(args.output/f'native-development-comparison.{suffix}',dpi=180)
    plt.close(fig)
    (args.output/'manifest.json').write_text(json.dumps(dict(source=receipt(args.comparison),tool=receipt(__file__),
        files=[receipt(p) for p in args.output.iterdir() if p.is_file()]),indent=2)+'\n')
    print(json.dumps(dict(observations=len(rows),output=str(args.output.resolve()))))


if __name__=='__main__':main()
