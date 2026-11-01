"""Plot actual training queue/service evidence; never a prediction or fit."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .native_flow import checked, receipt, save
from .probe_validation import read


def plot(audit_path, output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    audit=read(audit_path);raw=read(checked(audit['raw']));plan=read(checked(audit['plan']))
    if plan.get('phase')!='predictor_training' or audit.get('independent_validation') is not False:
        raise ValueError('Training diagnostics only')
    if not audit['stability']['original_stability_passed']:
        raise ValueError('Complete training diagnostics required')
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    for alias in plan['aliases']:
        rows=[r for r in raw['request_records'] if r['model']==alias]
        times=[(r['scheduled_time_ns']-raw['phase_start_ns'])/1e9 for r in rows]
        waits=[(r['observed_worker_start_time_ns']-r['scheduled_time_ns'])/1e9 for r in rows]
        services=[(r['worker_response']['end_perf_ns']-r['observed_worker_start_time_ns'])/1e9 for r in rows]
        axes[0].scatter(times,waits,s=7,alpha=.65,label=alias)
        axes[1].scatter(times,services,s=7,alpha=.65,label=alias)
    axes[0].set(title=f"Actual H600 training: {audit['candidate_id']} / seed {audit['seed']}",ylabel='Arrival to worker START (s)')
    axes[1].set(xlabel='Declared arrival time in measurement window (s)',ylabel='Worker service END - START (s)')
    axes[0].legend(ncol=4,loc='upper left',fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2);ax.set_xlim(0,plan['window_seconds']);ax.set_ylim(bottom=0)
    output.mkdir(parents=True,exist_ok=False)
    fig.savefig(output/'actual-queue-and-service.png',dpi=180)
    fig.savefig(output/'actual-queue-and-service.pdf')
    plt.close(fig)
    save(output/'sources.json',{'audit':receipt(audit_path),'raw':audit['raw'],'plan':audit['plan'],
        'source':receipt(__file__),'scope':'Actual training only; full offered cohort, drain-completed requests retained',
        'predictions_shown':False,'error_bars_or_confidence_intervals':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();plot(a.audit,a.output)
