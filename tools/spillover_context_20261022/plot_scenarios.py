"""Plot explicit FIFO sensitivity deltas; these are not GPU results."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=json.loads(args.report.read_text());args.output.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharex=True)
    for seed,color in [(107,'#1565c0'),(108,'#b45309')]:
        rows=sorted((c for c in report['cases'] if c['seed']==seed and c['generalist_share']=='1/32'),key=lambda c:c['workload']['text'])
        x=[c['workload']['text'] for c in rows]
        for ax,key in zip(axes,['L_delta_from_no_split','P_delta_from_no_split']):
            ax.plot(x,[c['scenarios'][0][key] for c in rows],marker='o',label=f'Development seed {seed}',color=color)
    for ax,label in zip(axes,['Mean-latency term L','P95-latency term P']):
        ax.axhline(0,color='#64748b',linewidth=1);ax.axvline(1.85,color='#64748b',linestyle=':',linewidth=1)
        ax.set_xlabel('Text arrival rate (requests/s)');ax.set_ylabel(f'Split minus no split: {label}')
        ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('UNVALIDATED FIFO SENSITIVITY — G2 text share 1/32',fontsize=12)
    fig.text(.5,.015,'Observed joint-text scenario + historical background service; U is unknown. Negative deltas favor split.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.93))
    for suffix in ['png','pdf']:fig.savefig(args.output/f'fine-share-latency-sensitivity.{suffix}',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output.resolve()),not_native=True,U=None)))


if __name__=='__main__':main()
