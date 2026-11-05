"""Export auditable curve/FIFO/mapping diagnostics and standalone figures."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))


def write(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def analyze(source):
    observations=json.loads((source/'observations.json').read_text())
    raw={r['identity']:json.loads((source/r['raw']).read_text()) for r in observations}
    plan=json.loads((source/'curve-plan.json').read_text())
    byname={p['name']:p for p in plan}
    points=[];models=[];paths=[]
    for row in observations:
        record=raw[row['identity']];p=record['prediction'];ledger=record['ledger']
        r=dict(row)
        for k in ('pooled_request_mean_ms','pooled_request_P95_ms','safe_memory_mib','max_rho',
                  'offered_requests','within_window_requests','end_backlog_requests','incomplete_after_drain',
                  'predicted_within_window_QPS','constraints_eligible','eligible','diagnostic_partial_U'):
            r[k]=p.get(k)
        for k in 'QT LPME'.replace(' ',''):
            r[f'{k}_norm']=p.get(f'{k}_norm');r[f'{k}_contribution']=p.get(f'{k}_contribution')
        r['watts']=(p.get('energy_evidence') or {}).get('watts')
        r['run_joules']=(p.get('energy_evidence') or {}).get('run_joules')
        for task,stat in p.get('per_task',{}).items():
            for k in ('mean_ms','p95_ms','request_count'): r[f'{task}_{k}']=stat[k]
            r[f'{task}_quality_loss']=p.get('quality_evidence',{}).get('per_task',{}).get(task,{}).get('loss')
        points.append(r)
        if ledger:
            for m,stats in ledger['per_model'].items():
                models.append(dict(identity=row['identity'],stage=row['stage'],curve=row.get('curve'),
                    ratio=row.get('ratio'),trace_seed=row['trace_seed'],model=m,replicas=row['deployment'][m],
                    offered_qps=math.fsum(rs.get(m,0) for rs in row['routing'].values()),rho=p['rho'][m],**stats))
            for m,ts in ledger['per_path'].items():
                for t,stats in ts.items():
                    paths.append(dict(identity=row['identity'],stage=row['stage'],curve=row.get('curve'),
                        ratio=row.get('ratio'),trace_seed=row['trace_seed'],model=m,task=t,
                        offered_qps=row['routing'][t].get(m,0),pool_service_ms=ledger['pool_service_ms'][m][t],**stats))
    curves=[]
    for name,curve in byname.items():
        rows=sorted({r['ratio']:r for r in points if r['stage'] in ('A','A_extra') and r.get('curve')==name}.values(),key=lambda r:r['ratio'])
        eligible=[r for r in rows if r['utility'] is not None]
        if not eligible: continue
        best=max(eligible,key=lambda r:r['utility'])
        endpoints=[next(r for r in rows if r['ratio']==v) for v in (0,1)]
        endpoints_valid=all(r['utility'] is not None for r in endpoints)
        interior=[r for r in eligible if 0<r['ratio']<1]
        best_interior=max(interior,key=lambda r:r['utility'],default=None)
        finer={}
        # Evaluate K membership on observed points only. Solver comparisons
        # below, rather than this partial slice, establish grid optima.
        for K in (1,4,8):
            sampled=[r for r in eligible if any(abs(r['ratio']*d-round(r['ratio']*d))<1e-10 for d in range(1,K+1))]
            finer[f'best_sampled_K{K}']=max((r['utility'] for r in sampled),default=None)
        curves.append(dict(curve=name,task=curve['task'],deployment=curve['deployment'],samples=len(rows),
            feasible_samples=len(eligible),infeasible_samples=sum(r['state']=='INFEASIBLE' for r in rows),
            unknown_samples=sum(r['state']=='UNKNOWN' for r in rows),best_ratio=best['ratio'],best_U=best['utility'],
            best_identity=best['identity'],endpoint_0_U=endpoints[0]['utility'],endpoint_1_U=endpoints[1]['utility'],
            best_interior_U=best_interior['utility'] if best_interior else None,
            best_interior_ratio=best_interior['ratio'] if best_interior else None,
            internal_beats_both_valid_endpoints=endpoints_valid and 0<best['ratio']<1
                and best['utility']>max(r['utility'] for r in endpoints)+1e-12,
            gain_over_best_valid_endpoint=best['utility']-max((r['utility'] for r in endpoints if r['utility'] is not None),default=-math.inf),
            valid_sample_ratio_min=min(r['ratio'] for r in eligible),valid_sample_ratio_max=max(r['ratio'] for r in eligible),
            same_mapping_adjacent=sum(a['assignment_sha256']==b['assignment_sha256'] for a,b in zip(rows,rows[1:])),
            utility_decreases_toward_S=sum(b['utility']<a['utility']-1e-12 for a,b in zip(rows,rows[1:])
                if a['utility'] is not None and b['utility'] is not None),**finer))
    mapping=[];bands=[];mapping_groups=[]
    for name in ('g2_asr','g2_image_ocr__ocr'):
        rows=sorted([r for r in points if r['stage']=='C' and r['curve']==name],key=lambda r:r['ratio'])
        previous=None;start=None
        for r in rows:
            record=raw[r['identity']];es=record['ledger']['events']
            workers=[e['worker'] for e in es]
            counts={w:workers.count(w) for w in sorted(set(workers))}
            pr=raw[previous['identity']] if previous else None
            changed=sum(e['worker']!=p['worker'] for e,p in zip(es,pr['ledger']['events'])) if pr else None
            p95delta=r['pooled_request_P95_ms']-previous['pooled_request_P95_ms'] if previous else None
            mapping.append(dict(curve=name,ratio=r['ratio'],identity=r['identity'],assignment_sha256=r['assignment_sha256'],
                changed_requests_from_previous=changed,counts=counts,
                same_counts_previous=counts==last_counts if previous else None,
                U=r['utility'],delta_U=r['utility']-previous['utility'] if previous else None,
                mean_ms=r['pooled_request_mean_ms'],p95_ms=r['pooled_request_P95_ms'],delta_p95_ms=p95delta))
            if previous and previous['assignment_sha256']!=r['assignment_sha256']:
                bands.append(dict(curve=name,first_sample_ratio=start['ratio'],last_sample_ratio=previous['ratio'],
                    samples=count,assignment_sha256=previous['assignment_sha256'],
                    U_range_in_same_mapping=max(us)-min(us),observed_samples_only=True))
                start=None
            if start is None: start=r;count=0;us=[]
            count+=1;us.append(r['utility']);previous=r;last_counts=counts
        if rows:
            bands.append(dict(curve=name,first_sample_ratio=start['ratio'],last_sample_ratio=previous['ratio'],samples=count,
                assignment_sha256=previous['assignment_sha256'],U_range_in_same_mapping=max(us)-min(us),observed_samples_only=True))
        groups={}
        for r in mapping:
            if r['curve']==name: groups.setdefault(json.dumps(r['counts'],sort_keys=True),[]).append(r)
        for count_key,rs in groups.items():
            if len({r['assignment_sha256'] for r in rs})>1:
                mapping_groups.append(dict(curve=name,counts=json.loads(count_key),samples=len(rs),
                    distinct_assignments=len({r['assignment_sha256'] for r in rs}),
                    p95_range=max(r['p95_ms'] for r in rs)-min(r['p95_ms'] for r in rs),
                    U_range=max(r['U'] for r in rs)-min(r['U'] for r in rs),
                    example_ratios=[r['ratio'] for r in rs[:10]]))
    controls=[r for r in points if r['stage']=='B' and r.get('control') not in (None,'representative')]
    idle=[]
    for task in ('asr','ocr','image','text'):
        a=next(r for r in points if r.get('contrast')==f'idle_{task}_resident')
        b=next(r for r in points if r.get('contrast')==f'idle_{task}_removed')
        idle.append(dict(task=task,resident_U=a['utility'],removed_U=b['utility'],
            removal_gain=b['utility']-a['utility'],memory_reduction=a['safe_memory_mib']-b['safe_memory_mib'],
            same_assignment=a['assignment_sha256']==b['assignment_sha256'],
            contribution_delta={k:b[f'{k}_contribution']-a[f'{k}_contribution'] for k in 'QTLPME'}))
    holdouts=[]
    for name in byname:
        for seed in (43,45):
            rs=[r for r in points if r['stage'] in ('E','E_extra') and r.get('curve')==name and r['trace_seed']==seed]
            if not rs: continue
            valid=[r for r in rs if r['utility'] is not None]
            best=max(valid,key=lambda r:r['utility'],default=None)
            holdouts.append(dict(curve=name,trace_seed=seed,evaluated_ratios=[r['ratio'] for r in rs],
                best_checked_ratio=best['ratio'] if best else None,best_checked_U=best['utility'] if best else None,
                all_checked_points_feasible=len(rs)==len(valid)))
    plane=[r for r in points if r['stage']=='A2D' and r['utility'] is not None]
    plane_best=max(plane,key=lambda r:r['utility'])
    summary=dict(unique_prediction_inputs=len(raw),observations=len(points),curves=curves,controls=controls,
        idle_deletion=idle,holdouts=holdouts,
        plane_best={k:plane_best[k] for k in ('asr_ratio','ocr_ratio','utility','identity')},
        internal_improvements=sum(c['internal_beats_both_valid_endpoints'] for c in curves),
        mapping=dict(adjacent_pairs=len(mapping)-2,same_full_assignment_pairs=sum(r['changed_requests_from_previous']==0 for r in mapping),
            changed_full_assignment_pairs=sum((r['changed_requests_from_previous'] or 0)>0 for r in mapping),
            changed_assignment_with_p95_change=sum((r['changed_requests_from_previous'] or 0)>0 and abs(r['delta_p95_ms'])>1e-9 for r in mapping),
            same_assignment_with_U_change=sum(r['changed_requests_from_previous']==0 and abs(r['delta_U'])>1e-12 for r in mapping),
            same_count_distinct_assignment_groups=len(mapping_groups)))
    folder=source/'text-slice'
    if folder.exists():
        grid=json.loads((folder/'grid.json').read_text())
        grid_best={str(K):max((r for r in grid if r['K']==K),key=lambda r:r['utility']) for K in (1,4,8)}
        de=[]
        for seed in (11,29,47):
            r=json.loads((folder/f'de-s{seed}.json').read_text());w=r['winner']
            de.append(dict(seed=seed,ratio=w['routing_shares']['text']['text'],U=w['best_feasible_utility'],
                delta_K8=w['best_feasible_utility']-grid_best['8']['utility'],
                seconds=r['budget']['elapsed_seconds'],predictions=r['budget']['full_prediction_calls'],status=r['status']))
        checks=json.loads((folder/'winner-audit.json').read_text())
        holdout=[]
        for seed in (11,29,47):
            for trace in (43,45):
                rr=[r for r in checks if r['optimizer_seed']==seed and r['trace_seed']==trace]
                interior=next(r for r in rr if 0<r['ratio']<1)
                endpoint=max(r['utility'] for r in rr if r['ratio'] in (0,1))
                holdout.append(dict(seed=seed,trace_seed=trace,ratio=interior['ratio'],U=interior['utility'],
                    best_endpoint_U=endpoint,delta_best_endpoint=interior['utility']-endpoint))
        summary['text_slice']=dict(grid_best=grid_best,de=de,holdouts=holdout,
            interpretation='only text varies; forced resident deployment; not the unrestricted Joint domain')
    return summary,points,models,paths,mapping,bands,mapping_groups,raw


def figures(source,out,summary,points,raw,mapping):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    curves=summary['curves']
    def rows(name):
        return sorted({r['ratio']:r for r in points if r['stage'] in ('A','A_extra') and r.get('curve')==name}.values(),key=lambda r:r['ratio'])
    fig,axs=plt.subplots(math.ceil(len(curves)/3),3,figsize=(15,3.2*math.ceil(len(curves)/3)),squeeze=False)
    for ax,curve in zip(axs.flat,curves):
        rs=rows(curve['curve']);vs=[r for r in rs if r['utility'] is not None]
        ax.plot([r['ratio'] for r in vs],[r['utility'] for r in vs],'.-',ms=3)
        bad=[r for r in rs if r['state']!='FEASIBLE']
        if bad: ax.plot([r['ratio'] for r in bad],[min(r['utility'] for r in vs)]*len(bad),'rx',label='invalid sampled route')
        ax.plot(curve['best_ratio'],curve['best_U'],'*',ms=12,color='orange')
        ax.set_title(curve['curve']+f" | best sampled r={curve['best_ratio']:g}",fontsize=9)
        ax.set(xlabel='Specialist fraction',ylabel='Valid U (higher is better)');ax.grid(alpha=.2)
    for ax in list(axs.flat)[len(curves):]:ax.set_visible(False)
    fig.tight_layout();fig.savefig(out/'utility-curves.png',dpi=150);plt.close(fig)
    selected=['g2_asr','g2_ocr','g2_image_ocr__ocr','g2_image_ocr_asr__asr','g2_text','g2_image_ocr_text__text']
    plan={p['name']:p for p in json.loads((source/'curve-plan.json').read_text())}
    for kind in ('latency','service','waiting','components'):
        fig,axs=plt.subplots(3,2,figsize=(13,11))
        for ax,name in zip(axs.flat,selected):
            rs=rows(name);x=[r['ratio'] for r in rs]
            curve=plan[name];task=curve['task'];g=curve['generalist'];s=curve['specialist']
            if kind=='latency':
                for field,label in [(f'{task}_mean_ms',f'{task} mean'),(f'{task}_p95_ms',f'{task} P95'),
                        ('pooled_request_mean_ms','all-task mean'),('pooled_request_P95_ms','all-task P95')]:
                    ax.plot(x,[r[field] for r in rs],label=label)
                ax.set_ylabel('Latency ms (includes invalid routes)')
            elif kind=='service':
                for m,label in ((g,'G'),(s,'S')):
                    ax.plot(x,[raw[r['identity']]['ledger']['pool_service_ms'][m][task] for r in rs],label=label+' full-pool service')
                    ax.plot(x,[raw[r['identity']]['ledger']['per_path'][m][task]['mean_service_ms'] for r in rs],
                        '--',label=label+' assigned-input mean')
                ax.set_ylabel(f'{task} service ms')
            elif kind=='waiting':
                for m,label in ((g,'G'),(s,'S')):
                    ax.plot(x,[raw[r['identity']]['ledger']['per_model'][m]['mean_waiting_ms'] for r in rs],label=label+' all-task mean wait')
                    ax.plot(x,[raw[r['identity']]['ledger']['per_path'][m][task]['mean_waiting_ms'] for r in rs],
                        '--',label=label+f' {task} mean wait')
                ax.set_ylabel('FIFO waiting ms; idle worker = missing')
            else:
                for k in 'QTLPME':ax.plot(x,[r[f'{k}_contribution'] for r in rs],label=k)
                ax.set_ylabel('Negative weighted cost (invalid routes diagnostic only)')
            ax.set_title(name);ax.set_xlabel('Specialist fraction');ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/f'{kind}-curves.png',dpi=170);plt.close(fig)
    plane=[r for r in points if r['stage']=='A2D']
    z=np.full((7,7),np.nan)
    for r in plane:
        if r['utility'] is not None:z[round(r['ocr_ratio']*6),round(r['asr_ratio']*6)]=r['utility']
    fig,ax=plt.subplots(figsize=(7,5));im=ax.imshow(z,origin='lower',extent=(-1/12,13/12,-1/12,13/12),aspect='auto')
    best=summary['plane_best'];ax.plot(best['asr_ratio'],best['ocr_ratio'],'r*',ms=14)
    ax.set(xlabel='ASR specialist fraction',ylabel='OCR specialist fraction',title='G2 + ASR + OCR, fixed single replicas; grey = invalid')
    ax.set_facecolor('lightgrey');fig.colorbar(im,ax=ax,label='Valid U');fig.tight_layout();fig.savefig(out/'asr-ocr-heatmap.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(15,7))
    for axes,name in zip(axs,['g2_asr','g2_image_ocr__ocr']):
        rs=[r for r in mapping if r['curve']==name];x=[r['ratio'] for r in rs]
        axes[0].plot(x,[(r['changed_requests_from_previous'] or 0) for r in rs]);axes[0].set_ylabel('Changed workers vs previous ratio')
        axes[1].plot(x,[r['p95_ms'] for r in rs]);axes[1].set_ylabel('All-task P95 ms')
        axes[2].plot(x,[r['U']-rs[0]['U'] for r in rs]);axes[2].set_ylabel('U minus r=0 U')
        first_change=next(r['ratio'] for r in rs if (r['changed_requests_from_previous'] or 0)>0)
        cutoff=max(r['ratio'] for r in rs if r['ratio']<first_change)
        for ax in axes:
            ax.axvspan(0,cutoff,color='orange',alpha=.15)
            ax.axvline(first_change,color='grey',ls=':',lw=1)
        for ax in axes:ax.set_title(name);ax.set_xlabel('Specialist fraction');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'mapping-zoom.png',dpi=170);plt.close(fig)
    if 'text_slice' in summary:
        s=summary['text_slice'];fig,axs=plt.subplots(1,2,figsize=(13,5))
        rs=rows('g2_image_ocr_text__text')
        axs[0].plot([r['ratio'] for r in rs],[r['utility'] for r in rs],'.-',label='coarse + local scan',lw=.7)
        grid=s['grid_best']['8']
        axs[0].axhline(grid['utility'],ls='--',color='grey',label='best complete K8 slice')
        for row in s['de']:axs[0].plot(row['ratio'],row['U'],'*',ms=13,label=f"DE seed {row['seed']}")
        baseline=next(r['utility'] for r in json.loads((source/'comparison.json').read_text())
                      if r['case']=='full5_n1' and r['algorithm']=='joint' and r['K']==1)
        axs[0].axhline(baseline,ls=':',color='black',label='unrestricted Joint winner')
        axs[0].set(xlabel='Text specialist fraction',ylabel='U',title='Internal gains under forced resident Text specialist')
        for row in s['de']:
            rr=[r for r in s['holdouts'] if r['seed']==row['seed']]
            axs[1].plot([42,43,45],[row['U']-s['grid_best']['1']['utility']]+[r['delta_best_endpoint'] for r in rr],
                '.-',label=f"frozen seed {row['seed']} route")
        axs[1].axhline(0,color='grey');axs[1].set(xticks=[42,43,45],xlabel='Trace seed (42 discovery)',
            ylabel='U minus best same-trace endpoint U',title='Route fixed before new-trace checks')
        for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/'text-interior-and-holdout.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();args_source=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    summary,points,models,paths,mapping,bands,groups,raw=analyze(args_source)
    for filename,rows in [('points.csv',points),('model-metrics.csv',models),('path-metrics.csv',paths),
                         ('curves.csv',summary['curves']),('mapping.csv',mapping),('mapping-bands.csv',bands),
                         ('same-count-different-mapping.csv',groups),('holdouts.csv',summary['holdouts']),
                         ('idle-deletion.csv',summary['idle_deletion'])]:write(out/filename,rows)
    if (args_source/'comparison.json').exists():write(out/'algorithms.csv',json.loads((args_source/'comparison.json').read_text()))
    if 'text_slice' in summary:
        write(out/'text-slice-de.csv',summary['text_slice']['de']);write(out/'text-slice-holdouts.csv',summary['text_slice']['holdouts'])
    save(out/'analysis.json',summary)
    figures(args_source,out,summary,points,raw,mapping)
