"""Report sealed Poisson GPU prediction errors, never refit or re-predict.

Request absolute percentage error (APE), error of task mean, and error of
task P95 are separate statistics. Every plotted group includes mean/min/max,
counts, and machine-readable data. No clipping, trimming or confidence claims.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.plot_research_results_20261105 import setup, COLORS
import matplotlib.pyplot as plt

SOURCES={}
TASKS=['asr','image','ocr','text']


def receipt(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(p):
    SOURCES[str(p.resolve())]=receipt(p)
    return json.loads(p.read_text())


def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def csv_save(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stats(values):
    if not values or any(not math.isfinite(x) or x<0 for x in values):raise ValueError('Invalid nonnegative errors')
    return dict(n=len(values),mean_pct=statistics.mean(values),min_pct=min(values),max_pct=max(values))


def ape(predicted,observed):
    if not math.isfinite(observed) or observed<=0 or not math.isfinite(predicted) or predicted<0:
        raise ValueError('Finite nonnegative prediction and strictly positive observation required')
    return 100*abs(predicted-observed)/observed


def range_table(ax,labels,values,title,xlabel,counts=None):
    """Readable min/max line plus mean point and explicit numbers for every row."""
    summaries=[stats(v) for v in values];y=np.arange(len(labels))
    means=np.array([s['mean_pct'] for s in summaries]);lo=np.array([s['min_pct'] for s in summaries]);hi=np.array([s['max_pct'] for s in summaries])
    ax.hlines(y,lo,hi,color=COLORS['legacy'],lw=1.3)
    ax.scatter(lo,y,marker='|',s=70,color=COLORS['green'],zorder=4,label='Minimum')
    ax.scatter(hi,y,marker='|',s=70,color=COLORS['balanced'],zorder=4,label='Maximum')
    ax.scatter(means,y,marker='o',s=25,color=COLORS['legacy'],zorder=5,label='Mean')
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_ylim(len(labels)-.5,-1.05)
    ax.set_xlim(0,max(max(hi)*1.08,.1));ax.set_xlabel(xlabel);ax.set_title(title,loc='left',pad=12)
    ax.grid(axis='x',alpha=.25);ax.set_axisbelow(True)
    transform=ax.get_yaxis_transform()
    columns=[(1.10,'Mean'),(1.27,'Min'),(1.44,'Max'),(1.61,'n')]
    for x,label in columns:ax.text(x,-.65,label,transform=transform,ha='right',va='center',fontsize=8,fontweight='bold')
    for i,s in enumerate(summaries):
        for x,key in [(1.10,'mean_pct'),(1.27,'min_pct'),(1.44,'max_pct')]:
            ax.text(x,i,f'{s[key]:.2f}',transform=transform,ha='right',va='center',fontsize=8)
        ax.text(1.61,i,str(counts[i] if counts else s['n']),transform=transform,ha='right',va='center',fontsize=8)
    return summaries


def figure(out,name,labels,values,title,xlabel,caption,counts=None,height=3):
    fig,ax=plt.subplots(figsize=(7.05,height))
    fig.subplots_adjust(left=.15,right=.64,top=.72 if height<2.6 else .80,bottom=.23 if height<2.6 else .18)
    summaries=range_table(ax,labels,values,title,xlabel,counts)
    handles,labs=ax.get_legend_handles_labels();fig.legend(handles,labs,loc='upper center',ncol=3,frameon=False)
    for ext in ('pdf','svg','png'):fig.savefig(out/f'{name}.{ext}',dpi=600 if ext=='png' else None)
    plt.close(fig)
    return dict(name=name,caption=caption,groups=[dict(label=l,**s) for l,s in zip(labels,summaries)])


def analyze(base):
    from experiments.fit_activation_native import load_bundle
    from experiments.replay_activation_aggregate import score_prediction
    from gsdeploy.trace import make_trace
    protocol=read(base/'protocol.json');policy=read(base/'policy-frozen.json')
    state=read(base/'capture-state.json')
    assert state['status']=='capture_plan_completed' and not state['errors']
    assert protocol['maximum_gpu_episodes']==2 and len(protocol['captures'])==2
    for r in protocol['source_receipts']+[protocol['fit'],protocol['original_fit'],protocol['policy'],protocol['queue']]:
        assert receipt(Path(r['path']))==r
        SOURCES[r['path']]=r
    audit=read(base/'capture-audit.json')
    assert len(audit['runs'])==2
    integrity_issues=[dict(run_id=r['run_id'],issues=r['issues']) for r in audit['runs'] if r['issues']]
    records=[];aggregates=[];episodes=[];assessments=[]
    for seal in protocol['predictions']:
        d=Path(seal['directory'])
        for key in ('plan','prediction'):assert receipt(Path(seal[key]['path']))==seal[key]
        plan=read(d/'plan.json');pred=read(d/'prediction-sealed.json');raw=read(d/'raw.json');hw=read(d/'hardware.json')
        assert hw['started_unix']*1e9>protocol['prediction_sealed_unix_ns']
        assert hw['raw_sha256']==receipt(d/'raw.json')['sha256'] and hw['same_device_during_capture']
        assert pred['complete'] and not pred['observed_timestamps_used']
        assert plan['not_for_training'] and plan['split']=='development_validation'
        job=plan['jobs'][0];design=job['design'];assert design['arrival_regime']=='poisson'
        trace=make_trace(protocol['task_rates_rps'],protocol['horizon_seconds'],plan['seed'],'poisson')
        assert [(offset*1000,t) for offset,t in trace]==[(r['offset_ms'],r['task']) for r in job['requests']]
        lookup={r['request_id']:r for r in pred['jobs'][0]['requests']}
        assert len(lookup)==len(raw['request_records'])==len(job['requests'])
        assert set(lookup)=={r['request_id'] for r in raw['request_records']}
        current=[]
        for r in raw['request_records']:
            p=lookup[r['request_id']]
            assert r['status']=='completed' and r['worker_response']['success']
            assert (r['model'],r['task'],r['pool_id'])==(p['alias'],p['task'],p['pool_id'])
            o=(r['completion_time_ns']-r['scheduled_time_ns'])/1e6
            predicted=p['predicted_e2e_ms'];assert o>0 and predicted>=0
            current.append(dict(round=plan['round'],seed=plan['seed'],request_id=r['request_id'],task=r['task'],
                alias=r['model'],pool_id=r['pool_id'],observed_e2e_ms=o,predicted_e2e_ms=predicted,
                absolute_error_ms=abs(predicted-o),absolute_relative_error_pct=ape(predicted,o),
                signed_relative_error_pct=100*(predicted-o)/o))
        records.extend(current)
        assessment=score_prediction(pred,load_bundle(d),policy);assessments.append(assessment)
        assert assessment['complete_cohort']
        for t in TASKS:
            rr=[r for r in current if r['task']==t];obs=[r['observed_e2e_ms'] for r in rr];ps=[r['predicted_e2e_ms'] for r in rr]
            for metric,fn in [('mean_latency',np.mean),('p95_latency',lambda x:np.percentile(x,95))]:
                ov,pv=float(fn(obs)),float(fn(ps))
                aggregates.append(dict(round=plan['round'],task=t,metric=metric,n_requests=len(rr),observed_ms=ov,predicted_ms=pv,
                    absolute_relative_error_pct=ape(pv,ov),signed_relative_error_pct=100*(pv-ov)/ov))
        e=assessment['episode'];th=assessment['throughput'][0]
        episodes.append(dict(round=plan['round'],n_requests=len(current),
            energy_observed_j=e['observed_board_joules'],energy_predicted_j=e['predicted_board_joules'],
            energy_error_pct=100*e['relative_energy_error'],throughput_error_pct=100*th['relative_error'],
            observed_completed=th['observed_completed'],predicted_completed=th['predicted_completed'],
            window_seconds=th['window_seconds']))
    return protocol,records,aggregates,episodes,assessments,integrity_issues


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.input.resolve();out=args.output.resolve()
    protocol,requests,aggregates,episodes,assessments,issues=analyze(base)
    out.mkdir(parents=True,exist_ok=False)
    csv_save(out/'request-errors.csv',requests);csv_save(out/'aggregate-errors.csv',aggregates);csv_save(out/'episode-errors.csv',episodes)
    save(out/'assessments.json',assessments)
    setup();figures=[]
    labels=['ASR','Image','OCR','Text','All requests']
    vals=[[r['absolute_relative_error_pct'] for r in requests if t=='all' or r['task']==t] for t in TASKS+['all']]
    figures.append(figure(out,'fig01_request_errors',labels,vals,'Poisson arrivals: end-to-end prediction error',
        'Per-request absolute relative error (%)',
        'Two prospectively sealed Poisson histories, fixed five-model residency and 50% G2 routing. '
        'Point: arithmetic mean of per-request absolute percentage errors; endpoints: observed minimum and maximum. '
        'All includes every request with equal weight. No error cancellation, clipping or confidence intervals. '
        'n is the number of requests, not independent inputs or repeated experiments. Latency includes FIFO waiting and completion.',height=3.1))
    paths=[(t,model) for t in TASKS for model in (t,'generalist_2')]
    labels=[f'{t.upper() if t in ("asr","ocr") else t.title()} / {"G2" if m=="generalist_2" else "Specialist"}' for t,m in paths]
    vals=[[r['absolute_relative_error_pct'] for r in requests if (r['task'],r['alias'])==(t,m+'#0')] for t,m in paths]
    figures.append(figure(out,'fig02_path_errors',labels,vals,'Poisson arrivals: errors by execution path',
        'Per-request absolute relative error (%)',
        'The same requests split by task and assigned model. Mean/min/max APE and request count are shown for all eight paths. '
        'The routing policy is fixed before measurement; these are descriptive strata, not independent experiments.',height=3.9))
    for metric,name,title in [('mean_latency','fig03_mean_latency_errors','Poisson arrivals: task mean-latency error'),
        ('p95_latency','fig04_p95_latency_errors','Poisson arrivals: task P95-latency error')]:
        labels=['ASR','Image','OCR','Text']
        vals=[[r['absolute_relative_error_pct'] for r in aggregates if r['task']==t and r['metric']==metric] for t in TASKS]
        figures.append(figure(out,name,labels,vals,title,'Absolute relative error of aggregate latency (%)',
            f'Compute {metric} independently for each task/run, then compare predicted and observed values. '
            'The point is the mean of two absolute aggregate errors; endpoints are their minimum and maximum. '
            'n=2 runs per task. This is not the mean or P95 of individual request errors. No confidence interval.',height=2.8))
    figures.append(figure(out,'fig05_energy_throughput_errors',['GPU energy','Throughput'],
        [[r['energy_error_pct'] for r in episodes],[r['throughput_error_pct'] for r in episodes]],
        'Poisson arrivals: episode-level prediction errors','Absolute relative error (%)',
        'Mean/min/max across two runs. Energy covers GPU-board startup through observation end, including priming, drain and tail, '
        'excluding host and teardown. Throughput compares completions within the same declared offered window; it is not saturation capacity.',height=2.35))
    save(out/'captions.json',figures)
    save(out/'summary.json',dict(protocol=protocol,requests=len(requests),episodes=episodes,figures=figures,
        integrity_issues=issues,statistical_scope='two histories, known inputs, fixed route; descriptive only',
        predictor_refits=0,new_gpu_episodes=2,prospective_predictions=2,new_predictions_during_analysis=0))
    import pymupdf as fitz
    from matplotlib.font_manager import findfont
    font=findfont('DejaVu Serif');book=fitz.open()
    for i,m in enumerate(figures,1):
        page=book.new_page(width=595,height=842);page.insert_font(fontname='serif',fontfile=font)
        page.insert_text((45,45),f'Figure {i} | Frozen predictor, Poisson-arrival GPU validation',fontname='serif',fontsize=11)
        with fitz.open(out/f'{m["name"]}.pdf') as d:
            height=d[0].rect.height/d[0].rect.width*510
            page.show_pdf_page(fitz.Rect(42,75,552,75+height),d,0)
        cap=m['caption']+'\n\nError (%) = 100 × |prediction - observation| / observation. '
        cap+='The same Poisson generator as the earlier static/routing experiments is used; rates and horizon follow the frozen five-model validation. '
        cap+='Historical stratified-jitter and paired-burst runs are excluded from these figures.'
        assert page.insert_textbox(fitz.Rect(48,105+height,548,795),cap,fontname='serif',fontsize=10.5,lineheight=1.3)>=0
    book.save(out/'poisson_prediction_errors.pdf');book.close()
    for r in SOURCES.values():assert receipt(Path(r['path']))==r
    save(out/'sources.json',list(SOURCES.values()))
    print(json.dumps(dict(figures=5,requests=len(requests),output=str(out),task_request_errors=figures[0]['groups']),ensure_ascii=False))


if __name__=='__main__':main()
