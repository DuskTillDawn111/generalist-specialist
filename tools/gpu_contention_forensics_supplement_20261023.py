"""Finalize host timeline, matched-input examples and independent invariants."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools.gpu_contention_forensics_20261023 import BASE, receipt,load_compact,intervals


def exact_work_features(path, wanted):
    result={};section=False;buf=[]
    with Path(path).open() as stream:
        for line in stream:
            if not section:
                if line.startswith('  "request_records": ['):section=True
                continue
            if line.startswith('  ],'):break
            if line.startswith('    {'):buf=[line]
            elif buf:buf.append(line)
            if buf and (line.startswith('    },') or line.rstrip()=='    }'):
                r=json.loads(''.join(buf).rstrip().rstrip(','));buf=[]
                if r['request_id'] not in wanted:continue
                w=r['worker_response']
                result[r['request_id']]={k:w.get(k) for k in ('input_tokens','generated_tokens','effective_generation','finish_reason','request_sha256','protocol_sha256','cache_reclamation')}
                for k in ('prediction','raw_output'):
                    value=w.get(k)
                    result[r['request_id']][k+'_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    assert set(result)==set(wanted)
    return result


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    cases=[json.loads((out/f'seed402-{label}-measurement-{rid}-timeline.json').read_text()) for label,rid in [('good',1039),('bad',932)]]
    fig,axes=plt.subplots(2,1,figsize=(13,6.8),constrained_layout=True)
    lanes={'generalist_2#0':3,'text#0':2,'text#1':1,'ocr#0':0}
    for ax,case in zip(axes,cases):
        event=case['events'][0]
        origin=event['wall_ns']-round(event['measurement_offset_seconds']*1e9)
        gs,ge=case['spill_service_start_ns'],case['spill_service_end_ns']
        pad=(ge-gs)*.08+50_000_000
        low,high=gs-pad,ge+pad
        for row in case['intervals']:
            if row['end_ns']<=low or row['start_ns']>=high:continue
            left,right=max(low,row['start_ns']),min(high,row['end_ns'])
            color='#d87539' if row['request_id']==case['request_id'] else ('#467caa' if row['family']=='text' else '#769e55')
            y=lanes[row['alias']]
            ax.broken_barh([((left-origin)/1e9,(right-left)/1e9)],(y-.28,.56),facecolors=color,edgecolors='white',linewidth=.6)
            if row['family']=='text' and gs<row['start_ns']<ge:
                ax.plot((row['start_ns']-origin)/1e9,y+.34,marker='v',color='#242424',markersize=4)
        for t in (gs,ge):ax.axvline((t-origin)/1e9,color='#b85d29',linestyle='--',linewidth=.8)
        ax.set_xlim((low-origin)/1e9,(high-origin)/1e9)
        ax.set_yticks(list(lanes.values()),list(lanes))
        ax.set_ylim(-.6,3.65)
        ax.grid(axis='x',alpha=.18)
        ax.set_axisbelow(True)
        ax.set_xlabel('Seconds since run start (Unix wall timestamps)')
        outcome=case['paired_outcome']
        renew=sum(e['event']=='other_service_start' and e['alias'].startswith('text#') and gs<e['wall_ns']<ge for e in case['events'])
        ax.set_title(f"Seed402 {case['outcome']}: {case['request_id']}; G2 service {outcome['on_service_ms']:.1f} ms; matched E2E change {outcome['e2e_change_ms']:+.1f} ms; later specialist starts={renew}",fontsize=10,loc='left')
    fig.suptitle('Host handler service timelines — CUDA kernel overlap is UNKNOWN',fontsize=13)
    handles=[Patch(facecolor='#d87539',label='G2 text spill service'),Patch(facecolor='#467caa',label='Specialist text service'),Patch(facecolor='#769e55',label='OCR service'),Line2D([0],[0],marker='v',color='none',markerfacecolor='#242424',label='Later specialist handler start')]
    axes[0].legend(handles=handles,loc='upper right',fontsize=8,ncol=2)
    fig.savefig(out/'host-handler-good-bad-timelines.png',dpi=180)
    fig.savefig(out/'host-handler-good-bad-timelines.pdf')
    plt.close(fig)
    return cases


def main():
    out=BASE/'gpu-contention-forensics-v1'
    if (out/'supplement-validation.json').exists():raise SystemExit('Refusing overwrite')
    forensic=json.loads((out/'analysis.json').read_text())
    reprice=json.loads((out/'actual-context-reprice.json').read_text())
    work_path=BASE/'gpu-contention-workload-decomposition-v1/spill-work.csv'
    work={(int(r['seed']),r['request_id']):r for r in csv.DictReader(work_path.open())}
    checks=[]
    def check(name,passed,details=None):
        checks.append(dict(name=name,passed=bool(passed),**({'details':details} if details is not None else {})))
    check('forensic source receipt unchanged',receipt(forensic['source']['path'])==forensic['source'])
    check('reprice source receipt unchanged',receipt(reprice['source']['path'])==reprice['source'])
    model=json.loads(Path(reprice['model']['path']).read_bytes())
    check('fixed model receipt unchanged',receipt(reprice['model']['path'])==reprice['model'])
    for run in forensic['results']:
        seed=run['seed'];raws=[]
        for role,ref in zip(('coupled_spill','matched_no_spill'),run['raw_receipts']):
            check(f'{seed} {role} audited raw SHA unchanged',receipt(ref['path'])==ref)
            raw=load_compact(ref['path']);raws.append(raw)
            for alias in sorted(raw['replica_to_family']):
                ints=sorted([r for r in intervals(raw) if r['alias']==alias],key=lambda r:r['start_ns'])
                check(f'{seed} {role} {alias} serial service intervals',all(a['end_ns']<=b['start_ns'] for a,b in zip(ints,ints[1:])))
        a,b=[{r['request_id']:r for r in raw['request_records']} for raw in raws]
        check(f'{seed} exact request and input/arrival cohort',set(a)==set(b) and all(all(r[k]==b[i][k] for k in ('task','pool_id','sample_id','scheduled_offset_seconds')) for i,r in a.items()))
        for spill in run['spill_requests']:
            s=spill['segments'];name=f"{seed} {spill['request_id']}"
            check(name+' segment endpoints and contiguous cover',s[0]['start_ns']==spill['start_ns'] and s[-1]['end_ns']==spill['end_ns'] and all(x['end_ns']==y['start_ns'] for x,y in zip(s,s[1:])))
            check(name+' segment duration closes to service',abs(sum(x['duration_ms'] for x in s)-spill['on_service_ms'])<1e-6)
            check(name+' text count range and same G2 serial',all(0<=x['specialist_active_count']<=2 and not x['same_G2_worker_active_tasks'] for x in s))
        for group in run['spill_context_summaries'].values():
            check(f'{seed} subgroup context fractions sum1 n{group["n"]}',abs(sum(v['fraction'] for v in group['time_by_specialist_active_count'].values())-1)<1e-12)
    for r in reprice['rows']:
        fit=model['family_models'][r['family']]['fit'];unknown=defaultdict(float)
        for seg in r['unsupported_segments']:
            unknown[json.dumps(seg['other_active_counts'],sort_keys=True,separators=(',',':'))]+=seg['duration_ms']
        work_sum=sum((duration-unknown[key])*fit['progress'].get(key,0.) for key,duration in r['active_state_duration_ms'].items())
        check(f"{r['seed']} {r['role']} {r['request_id']} actual W closes",abs(work_sum-r['known_partial_work_W_ms'])<1e-6)
        if r['fully_supported']:
            check(f"{r['seed']} {r['role']} {r['request_id']} W/b identity",r['unknown_duration_ms']==0 and abs(r['W_ms']/r['solo_input_baseline_b_ms']-r['W_over_b'])<1e-12)
        else:
            check(f"{r['seed']} {r['role']} {r['request_id']} missing background not imputed",r['W_ms'] is None and r['W_over_b'] is None)
    rr=[r for run in forensic['results'] if run['seed']==402 for r in run['spill_requests']]
    ids={r['request_id']:r for r in rr}
    wanted=['measurement:483','measurement:613','measurement:546','measurement:741','measurement:769','measurement:932']
    rawref=next(run['raw_receipts'][0] for run in forensic['results'] if run['seed']==402)
    exact=exact_work_features(rawref['path'],wanted)
    pairs=[]
    for low,high in [('measurement:613','measurement:483'),('measurement:741','measurement:546'),('measurement:769','measurement:932')]:
        flag={k:exact[low][k]==exact[high][k] for k in ('input_tokens','generated_tokens','effective_generation','finish_reason','request_sha256','protocol_sha256','prediction_sha256','raw_output_sha256')}
        check(f'402 same pool repeated G2 {low}/{high} all work flags',all(flag.values()) and ids[low]['pool_id']==ids[high]['pool_id'])
        records=[]
        for rid in (low,high):
            r=ids[rid];times={n:sum(s['duration_ms'] for s in r['segments'] if s['specialist_active_count']==n) for n in range(3)}
            records.append(dict(seed=402,request_id=rid,pool_id=r['pool_id'],outcome_group=r['outcome_group'],
                                input_tokens=r['input_tokens'],generated_tokens=r['generated_tokens'],service_ms=r['on_service_ms'],
                                two_specialist_service_fraction=times[2]/r['on_service_ms'],
                                later_specialist_service_starts=r['later_specialist_service_start_count'],
                                cache_reclamation=exact[rid]['cache_reclamation'],e2e_change_ms=r['e2e_change_ms']))
        pairs.append(dict(pool_id=ids[low]['pool_id'],all_work_equal_flags=flag,records=records))
    (out/'same-input-controlled-examples.json').write_text(json.dumps(dict(source=receipt(__file__),token_work_source=receipt(work_path),raw=rawref,pairs=pairs,scope='Same-run same G2 family and pool, request/protocol hashes, actual output text hashes and generation features equal. Context and service differ; descriptive within-run evidence, not randomized CUDA intervention.'),indent=2,ensure_ascii=False)+'\n')
    fields=['seed','request_id','pool_id','outcome_group','input_tokens','generated_tokens','service_ms','two_specialist_service_fraction','later_specialist_service_starts','cache_performed','e2e_change_ms']
    with (out/'same-input-controlled-examples.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for pair in pairs:
            for r in pair['records']:
                writer.writerow({**{k:r[k] for k in fields if k!='cache_performed'},'cache_performed':r['cache_reclamation']['performed']})
    plot(out)
    complete_rows=list(csv.DictReader((out/'actual-context-reprice-requests-complete.csv').open()))
    check('complete reprice CSV includes all4384 rows',len(complete_rows)==len(reprice['rows'])==4384)
    check('spill CSV includes all50 rows',len(list(csv.DictReader((out/'spill-requests.csv').open())))==50)
    result=dict(schema='gsdeploy.host_contention_forensics_validation.v1',source=receipt(__file__),
                sources=[receipt(out/'analysis.json'),receipt(out/'actual-context-reprice.json'),receipt(out/'actual-context-reprice-requests-complete.csv'),receipt(work_path)],
                checks=checks,passed=sum(c['passed'] for c in checks),checks_total=len(checks),verified=all(c['passed'] for c in checks),
                excluded_partial_artifacts=['actual-context-reprice-requests.csv'],
                exclusion_reason='Initial CSV exporter expected a misnamed partial-work field and stopped after header; complete JSON was closed successfully. Re-exported complete CSV directly from that immutable JSON, independently checked4384 rows. Do not include header-only CSV as data.',
                GPU_started=False,frozen_sources_modified=False)
    (out/'supplement-validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=result['passed'],checks_total=result['checks_total'],verified=result['verified'],pairs=pairs),ensure_ascii=False))


if __name__=='__main__':main()
