"""CPU-only matplotlib figures from completed WLS/Gamma H600 training reports.

Reads training reports and their public heldout plans, hashes their declared
sources, and recomputes report arithmetic only. No fit, forecast, independent
raw, runtime collection, or GPU experiment is performed.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

from .duration_selection import evaluate
from .native_flow import checked, receipt, save
from .ocr_trained_validation import LIMITS
from .probe_validation import read
from .profiling import digest

SEEDS=(627601,627602)
ROUTES=('shared_balanced','shared_skewed','shared_spill_low','shared_spill_high')
METRICS=('service_mare','e2e_mare','queue_mae','text_p95_are','image_p95_are','asr_p95_are','ocr_p95_are')
LABELS=('Service\nMARE / .20','E2E\nMARE / .20','Queue\nMAE / 1 req','Text P95\nARE / .20',
    'Image P95\nARE / .20','ASR P95\nARE / .20','OCR P95\nARE / .20')
COLORS={'WLS':'#2474a6','Gamma':'#c45c25'}
SCOPE='Whole-seed training CV only; no independent validation or final gain claim.'


def refs_in(value):
    refs={}
    def visit(x):
        if isinstance(x,dict):
            if set(x)=={'path','sha256'}: refs[(x['path'],x['sha256'])]=x
            else:
                for v in x.values():visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    visit(value)
    return list(refs.values())


def audit_reports(baseline_path,gamma_path):
    """Verify frozen input identity and independently recalculate plotted values."""
    reports={'WLS':read(baseline_path),'Gamma':read(gamma_path)}
    expected_schema={'WLS':'gsdeploy.run_aware_H600_numerical_v2_training_report.v1',
        'Gamma':'gsdeploy.Gamma_H600_training_CV.v1'}
    all_refs=[receipt(baseline_path),receipt(gamma_path)]
    contracts={};actual=[];summaries={}
    for name,d in reports.items():
        if (d.get('schema')!=expected_schema[name] or d.get('independent_validation') is not False
                or d.get('independent_results_read') is not False or d.get('final_gain_eligible') is not False
                or d.get('thresholds_tuned') is not False or d.get('known')!=9600 or d.get('requested')!=9600
                or d.get('complete_prediction_coverage') is not True):
            raise ValueError('Complete unchanged training-only H600 reports required')
        rows={(r['seed'],r['candidate_id']):r for r in d['candidates']}
        if len(d['candidates'])!=8 or set(rows)!={(s,c) for s in SEEDS for c in ROUTES}:
            raise ValueError('Two whole H600 seeds and exactly four routes required')
        observed={}
        for key,row in rows.items():
            if row['known']!=1200 or row['requested']!=1200 or row.get('independent_validation') is not False:
                raise ValueError('Full per-case training request denominator required')
            plan=read(checked(row['heldout_plan']))
            if (plan.get('split')!='training' or plan.get('phase')!='predictor_training'
                    or plan['seed']!=row['seed'] or plan['window_seconds']!=600. or len(plan['trace'])!=1200):
                raise ValueError('Training-only public heldout plan required')
            contract=dict(baseline=plan['baseline_candidate_id'],tie_tolerance=plan.get('utility_tie_tolerance',0.),
                candidate_ids=sorted(c['id'] for c in plan['candidates']),tasks=sorted(plan['workload_rps']))
            if contracts.setdefault(row['seed'],contract)!=contract:
                raise ValueError('Changed same-seed candidate/objective contract')
            observed[key]={k:row[k] for k in ('heldout_plan','heldout_raw','heldout_scored','observed_U','observed_metrics')}
        assessment=evaluate(d,dict(seeds=list(SEEDS),routes=sorted(ROUTES)),LIMITS,contracts)
        if assessment!=d['original_threshold_assessment'] or len(assessment['metrics'])!=56:
            raise ValueError('Training threshold/ranking arithmetic changed')
        if any(r['normalized_error'] is None for r in assessment['metrics']):
            raise ValueError('Figure v1 requires explicit complete 56-check temporal coverage')
        a=assessment
        summaries[name]=dict(requested=d['requested'],known=d['known'],temporal_checks=56,
            temporal_failures=a['temporal_failed_or_unknown'],normalized_temporal_error_mean=a['normalized_temporal_error_mean'],
            delta_sign_correct=sum(sum(p['sign_correct'] for p in s['paired_deltas']) for s in a['by_seed'].values()),
            delta_sign_requested=6,rank_correct=sum(sum(p['correct'] for p in s['candidate_pairs']) for s in a['by_seed'].values()),
            rank_requested=12,decision_errors=a['decision_errors'],training_cv_thresholds_passed=a['original_training_cv_thresholds_passed'])
        all_refs.extend(refs_in(d));actual.append(observed)
    if actual[0]!=actual[1]:raise ValueError('Identical heldout actual measurements required')
    refs={}
    for ref in all_refs:
        # Verify byte identity; never parse or use raw outcome payloads here.
        checked(ref);refs[(ref['path'],ref['sha256'])]=ref
    return reports,summaries,list(refs.values())


def footer(fig,extra=''):
    fig.text(.5,.025,SCOPE+('  '+extra if extra else ''),ha='center',fontsize=9,color='#4c4c4c')


def export(fig,output,name):
    for suffix in ('png','pdf'):
        fig.savefig(output/(name+'.'+suffix),dpi=190,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def temporal_heatmap(reports,summaries,output):
    data={name:{(r['seed'],r['candidate_id'],r['metric']):r for r in d['original_threshold_assessment']['metrics']}
        for name,d in reports.items()}
    maximum=max(r['normalized_error'] for values in data.values() for r in values.values())
    cmap=LinearSegmentedColormap.from_list('threshold',['#d8e9f3','#fffdf8','#d67b65'])
    norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=max(2.8,maximum))
    fig,axes=plt.subplots(1,2,figsize=(15.3,7.3),sharey=True)
    for ax,(name,values) in zip(axes,data.items()):
        grid=np.array([[values[(s,c,k)]['normalized_error'] for k in METRICS] for s in SEEDS for c in ROUTES])
        im=ax.imshow(grid,cmap=cmap,norm=norm,aspect='auto')
        for y in range(8):
            for x in range(7):
                value=grid[y,x];failed=value>1
                ax.text(x,y,f'{value:.2f}',ha='center',va='center',fontsize=10,
                    color='#7e2116' if failed else '#17344a',fontweight='bold' if failed else 'normal')
                if failed:ax.add_patch(Rectangle((x-.48,y-.48),.96,.96,fill=False,edgecolor='#a62f22',lw=1.8))
        ax.set_xticks(range(7),LABELS,fontsize=9)
        ax.set_yticks(range(8),[str(s)+'  '+c.removeprefix('shared_') for s in SEEDS for c in ROUTES],fontsize=10)
        ax.axhline(3.5,color='white',lw=3)
        ax.set_title(f'{name}: {summaries[name]["temporal_failures"]}/56 temporal checks fail\n'
            f'Mean normalized error = {summaries[name]["normalized_temporal_error_mean"]:.6f}',fontsize=12,pad=14)
        ax.tick_params(length=0)
    fig.subplots_adjust(left=.16,right=.91,bottom=.19,top=.80,wspace=.08)
    cax=fig.add_axes([.93,.25,.015,.48]);bar=fig.colorbar(im,cax=cax)
    bar.set_label('Error / original threshold (>1 fails)',fontsize=10)
    fig.suptitle('All 56 temporal checks per estimator: Gamma improves the mean, but does not clear the gate',
        fontsize=15,y=.97)
    fig.text(.5,.875,'Both: 9,600/9,600 predicted requests | delta signs 6/6 | route pairs 12/12 | 0 decision errors',
        ha='center',fontsize=11)
    footer(fig,'Red boxes mark failures; values are not clipped.')
    export(fig,output,'01_all_temporal_checks')


def failure_tradeoffs(reports,summaries,output):
    values={name:{(r['seed'],r['candidate_id'],r['metric']):r for r in d['original_threshold_assessment']['metrics']}
        for name,d in reports.items()}
    keys=[key for key,r in values['WLS'].items() if r['failed_or_unknown']]
    fig,axes=plt.subplots(1,len(keys),figsize=(13.8,5.9))
    records=[]
    for ax,key in zip(axes,keys):
        rows=[values[name][key] for name in ('WLS','Gamma')]
        heights=[r['normalized_error'] for r in rows]
        ax.bar([0,1],heights,color=[COLORS[n] for n in ('WLS','Gamma')],width=.58)
        ax.axhline(1,color='#333',ls='--',lw=1.3,label='Original threshold')
        ax.set_ylim(0,3.2);ax.set_xticks([0,1],['WLS','Gamma']);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        metric=key[2];unit='requests' if metric=='queue_mae' else 'fraction'
        change=(rows[1]['value']/rows[0]['value']-1)*100
        title=key[1].removeprefix('shared_')+'\n'+metric.replace('_',' ')
        ax.set_title(title,fontsize=12,pad=13)
        for x,r in enumerate(rows):
            raw=f'{r["value"]:.3f} req' if unit=='requests' else f'{r["value"]*100:.2f}%'
            ax.text(x,r['normalized_error']+.07,f'{r["normalized_error"]:.2f}x\n{raw}',ha='center',fontsize=11)
        direction='worse' if change>1e-9 else 'better' if change < -1e-9 else 'unchanged'
        ax.text(.5,.96,f'Gamma: {abs(change):.1f}% {direction}' if direction!='unchanged' else 'Gamma: unchanged',
            transform=ax.transAxes,ha='center',fontsize=11,color='#a32e20' if direction=='worse' else '#284b60')
        records.append(dict(seed=key[0],candidate_id=key[1],metric=metric,WLS=rows[0],Gamma=rows[1],
            relative_change_percent=change,direction=direction))
    axes[0].set_ylabel('Error / original threshold',fontsize=11)
    fig.subplots_adjust(left=.07,right=.98,bottom=.22,top=.72,wspace=.32)
    fig.suptitle('The same three failures remain (heldout training seed 627601)',fontsize=15,y=.97)
    mean_change=(summaries['Gamma']['normalized_temporal_error_mean']/summaries['WLS']['normalized_temporal_error_mean']-1)*100
    fig.text(.5,.86,f'Mean normalized error improves {abs(mean_change):.2f}%; OCR P95 error worsens. Training gate remains FAIL.',
        ha='center',fontsize=12)
    footer(fig,'All three panels show original WLS failure cases; no case was removed.')
    export(fig,output,'02_original_failures_and_tradeoffs')
    return records


def utility_scatter(reports,output):
    rows={name:{(r['seed'],r['candidate_id']):r for r in d['candidates']} for name,d in reports.items()}
    fig,axes=plt.subplots(2,2,figsize=(12.7,10.5))
    for j,seed in enumerate(SEEDS):
        for i,zoom in enumerate((False,True)):
            ax=axes[i,j];domain=[c for c in ROUTES if not zoom or c!='shared_skewed']
            values=[v for name in rows for c in domain for v in
                (rows[name][(seed,c)]['observed_U'],rows[name][(seed,c)]['predicted_comparison_U'])]
            lo,hi=min(values),max(values);pad=(hi-lo)*.14
            ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color='#777',ls='--',lw=1,label='Prediction = actual')
            for name in ('WLS','Gamma'):
                x=[rows[name][(seed,c)]['observed_U'] for c in domain]
                y=[rows[name][(seed,c)]['predicted_comparison_U'] for c in domain]
                ax.scatter(x,y,c=COLORS[name],marker='o' if name=='WLS' else 'x',
                    s=65 if name=='WLS' else 95,lw=1.8,label=name,zorder=3,
                    facecolors='none' if name=='WLS' else None)
            for c in domain:
                r=rows['Gamma'][(seed,c)]
                ax.annotate(c.removeprefix('shared_'),(r['observed_U'],r['predicted_comparison_U']),
                    textcoords='offset points',xytext=(5,-14 if c=='shared_balanced' else 7),fontsize=9)
            ax.set_xlim(lo-pad,hi+pad);ax.set_ylim(lo-pad,hi+pad);ax.set_aspect('equal',adjustable='box')
            ax.grid(alpha=.2);ax.set_xlabel('Observed six-term U');ax.set_ylabel('Predicted six-term U')
            ax.set_title(f'Seed {seed}: '+('all four routes' if not zoom else 'zoom: three non-skewed routes'),fontsize=12)
            if i==0 and j==0:ax.legend(loc='upper left',fontsize=9)
    fig.subplots_adjust(left=.08,right=.96,bottom=.10,top=.86,hspace=.32,wspace=.28)
    fig.suptitle('Six-term utility predictions versus the same heldout actual measurements',fontsize=15,y=.975)
    fig.text(.5,.925,'Higher U is better. Both estimators preserve all 6 delta signs and all 12 route-pair orderings.',
        ha='center',fontsize=11)
    footer(fig,'Lower panels expose the close routes; skewed remains visible in the upper panels.')
    export(fig,output,'03_utility_predictions_two_seeds')


def run(baseline_path,gamma_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    reports,summaries,refs=audit_reports(baseline_path,gamma_path)
    tool_refs=[receipt(__file__),receipt(Path(__file__).with_name('duration_selection.py')),
        receipt(Path(__file__).with_name('ocr_trained_validation.py')),
        receipt(Path(__file__).with_name('flow_crossfit_report.py'))]
    refs.extend(tool_refs)
    for ref in refs:checked(ref)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    output.mkdir(parents=True,exist_ok=False)
    temporal_heatmap(reports,summaries,output)
    failures=failure_tradeoffs(reports,summaries,output)
    utility_scatter(reports,output)
    data=dict(schema='gsdeploy.H600_training_CV_figure_data.v1',summaries=summaries,
        temporal_checks={n:d['original_threshold_assessment']['metrics'] for n,d in reports.items()},
        original_WLS_failure_tradeoffs=failures,
        utilities={n:[{k:r[k] for k in ('seed','candidate_id','predicted_comparison_U','observed_U')}
            for r in d['candidates']] for n,d in reports.items()},
        independent_validation=False,independent_outcomes_read=False,final_gain_eligible=False,GPU_started=False)
    save(output/'figure-data.json',data)
    readme=('Training CV only. Source reports are unchanged; raw observations are not parsed by this script.\n'
        '01: all 56 temporal checks per estimator, normalized by unchanged original thresholds.\n'
        '02: all original WLS failure cases; same three remain, image improves and OCR worsens.\n'
        '03: two-seed utility scatter with full range and explicit close-route zoom.\n'
        'Both: coverage 9600/9600, delta signs 6/6, route pairs 12/12; temporal gate not passed.\n'
        'This is two whole-seed training CV, not independent validation, an uncertainty calibration, or final gain.\n')
    (output/'README.txt').write_text(readme)
    # No source may drift between reading and writing the figures.
    for ref in refs:checked(ref)
    manifest=dict(schema='gsdeploy.H600_training_CV_matplotlib_figures.v1',created_ns=time.perf_counter_ns(),
        inputs={'WLS':receipt(baseline_path),'Gamma':receipt(gamma_path)},entry_source=receipt(__file__),
        source_receipts=refs,summary=summaries,data=receipt(output/'figure-data.json'),
        outputs=[receipt(p) for p in sorted(output.iterdir())],
        runtime=dict(python=platform.python_version(),matplotlib=matplotlib.__version__,numpy=np.__version__),
        plot_scope=SCOPE,training_assessment_recomputed=True,numerical_model_refit=False,
        raw_payloads_parsed=False,independent_outcomes_read=False,independent_validation=False,
        thresholds_tuned=False,final_gain_eligible=False,GPU_started=False)
    manifest['manifest_sha256']=digest(manifest);save(output/'sources.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('baseline','gamma','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();v=run(a.baseline,a.gamma,a.output)
    print(json.dumps(dict(status='TRAINING_FIGURES_COMPLETE',output=str(Path(a.output).resolve()),
        output_count=len(v['outputs']),GPU_started=False,independent_outcomes_read=False)))


if __name__=='__main__':main()
