"""Posthoc actual-context work reconciliation, not an online predictor."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools.gpu_contention_forensics_20261023 import BASE, load_compact, receipt, intervals, partition, intersection


def summary(values):
    values=[v for v in values if v is not None]
    return dict(n=len(values),mean=statistics.fmean(values) if values else None,
                median=statistics.median(values) if values else None,
                minimum=min(values) if values else None,maximum=max(values) if values else None,
                above2=sum(v>2 for v in values))


def main():
    out=BASE/'gpu-contention-forensics-v1'
    target=out/'actual-context-reprice.json'
    if target.exists():
        raise SystemExit('Refusing to overwrite '+str(target))
    model_path=ROOT/'artifacts/queue-coupled-study-20261023/controlled-resource-fit-v2/resources.json'
    model=json.loads(model_path.read_bytes())
    audit_path=BASE/'independent-runtime-audit-stream-v1/terminal.json'
    audit=json.loads(audit_path.read_bytes())
    observed={o['label']:o for o in audit['observations']}
    results=[];all_rows=[];spill_pool_rows=[]
    for seed in (401,402):
        for role in ('coupled_spill','matched_no_spill'):
            ref=observed[f'{role}-seed{seed}']['raw']
            assert observed[f'{role}-seed{seed}']['verified'] and receipt(ref['path'])==ref
            raw=load_compact(ref['path']);ints=intervals(raw);byid={r['request_id']:r for r in ints}
            original={r['request_id']:r for r in raw['request_records']}
            spills=[r for r in ints if r['family']=='generalist_2' and r['task']=='text']
            details=[]
            for row in ints:
                if row['task']!='text' or row['family'] not in model['family_models']:
                    continue
                fit=model['family_models'][row['family']]['fit'];baseline=fit['baselines_ms'].get(row['pool_id'])
                segs=partition(row,ints);known_work=0.;unknown_ms=0.;states=defaultdict(float)
                known=[];unknown=[]
                other_ids=set()
                for s in segs:
                    active=[byid[rid] for rid in s['all_other_active'].values()]
                    counts=dict.fromkeys(model['deployment'],0)
                    for x in active:counts[x['family']]+=1
                    key=json.dumps(counts,sort_keys=True,separators=(',',':'))
                    # Never interpret nontext activity through a text-only fit.
                    progress=fit['progress'].get(key) if all(x['task']=='text' for x in active) else None
                    states[key]+=s['duration_ms']
                    other_ids.update(x['request_id'] for x in active)
                    if progress is None:
                        unknown_ms+=s['duration_ms']
                        unknown.append(dict(duration_ms=s['duration_ms'],other_active_counts=counts,
                                            other_tasks=[x['task'] for x in active]))
                    else:
                        known_work+=s['duration_ms']*progress
                        known.append(dict(duration_ms=s['duration_ms'],other_active_counts=counts,progress=progress))
                supported=unknown_ms==0 and baseline is not None
                later=[byid[i] for i in other_ids if byid[i]['family']=='text' and byid[i]['start_ns']>row['start_ns']]
                mixed=[byid[i] for i in other_ids if byid[i]['family']=='text' and byid[i]['pool_id']!=row['pool_id']]
                overlap=any(intersection(row,s)>0 for s in spills if s['request_id']!=row['request_id'])
                d=dict(seed=seed,role=role,request_id=row['request_id'],pool_id=row['pool_id'],family=row['family'],alias=row['alias'],
                       actual_service_ms=row['service_ms'],solo_input_baseline_b_ms=baseline,
                       fully_supported=supported,known_partial_work_W_ms=known_work,
                       unknown_duration_ms=unknown_ms,W_ms=known_work if supported else None,
                       W_over_b=known_work/baseline if supported else None,
                       known_partial_W_over_b=known_work/baseline if baseline else None,
                       overlapped_G2_text=overlap,later_other_specialist_service_starts=len(later),
                       mixed_other_specialist_inputs=len(mixed),input_tokens=row['input_tokens'],generated_tokens=row['generated_tokens'],
                       active_state_duration_ms=dict(states),unsupported_segments=unknown)
                details.append(d)
                if row['family']=='generalist_2':
                    spill_pool_rows.append(d)
            all_rows.extend(details)
            groups={}
            for family in ('generalist_2','text'):
                rr=[r for r in details if r['family']==family]
                if not rr:continue
                groups[family]=dict(requests=len(rr),supported=sum(r['fully_supported'] for r in rr),
                                    unsupported_request_ids=[r['request_id'] for r in rr if not r['fully_supported']],
                                    W_over_b=summary([r['W_over_b'] for r in rr]),
                                    G2_overlap=summary([r['W_over_b'] for r in rr if r['overlapped_G2_text']]),
                                    no_G2_overlap=summary([r['W_over_b'] for r in rr if not r['overlapped_G2_text']]),
                                    later_specialist_renewal_count=summary([r['later_other_specialist_service_starts'] for r in rr]))
            results.append(dict(seed=seed,role=role,raw=ref,groups=groups))
    forensic=json.loads((out/'analysis.json').read_bytes())
    outcome={(r['seed'],s['request_id']):s['outcome_group'] for r in forensic['results'] for s in r['spill_requests']}
    spill_groups={}
    for seed in (401,402):
        spill_groups[str(seed)]={}
        for group in ('all','faster','slower_or_equal'):
            rows=[r for r in spill_pool_rows if r['seed']==seed and (group=='all' or outcome[(seed,r['request_id'])]==group)]
            spill_groups[str(seed)][group]=dict(n=len(rows),supported=sum(r['fully_supported'] for r in rows),
                                               W_over_b=summary([r['W_over_b'] for r in rows]),
                                               mixed_input_request_count=sum(r['mixed_other_specialist_inputs']>0 for r in rows),
                                               renewal_count=summary([r['later_other_specialist_service_starts'] for r in rows]))
    result=dict(schema='gsdeploy.actual_context_progress_reconciliation.v1',source=receipt(__file__),
                interval_tool=receipt(ROOT/'tools/gpu_contention_forensics_20261023.py'),
                model=receipt(model_path),audit=receipt(audit_path),results=results,spill_outcome_groups=spill_groups,
                rows=all_rows,method='W=sum(actual segment duration_ms * fixed progress(context)), b=fixed same-pool solo baseline; W/b expected near1 only if this family-count work model transfers to that actual execution. Nontext background or absent state makes full W UNKNOWN; partial W is diagnostic only.',
                interpretation='Actual-context posthoc reconciliation uses future realized service events unavailable online. W/b residual can reveal mismatch even after conditioning on realized family-count trajectory, but does not identify a causal fraction of GPU interference or isolate kernel contention. Mixed co-runner inputs, sustained renewal, prefill/decode phases, host scheduling, clocks, timer alignment and allocator/history can all remain outside the family-count model.',
                GPU_started=False,frozen_sources_modified=False)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    columns=['seed','role','request_id','pool_id','family','alias','actual_service_ms','solo_input_baseline_b_ms','fully_supported','W_ms','W_over_b','known_partial_W_ms','unknown_duration_ms','overlapped_G2_text','later_other_specialist_service_starts','mixed_other_specialist_inputs','input_tokens','generated_tokens']
    with (out/'actual-context-reprice-requests.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows({k:r[k] for k in columns} for r in all_rows)
    print(json.dumps(dict(results=results,spill_outcome_groups=spill_groups),ensure_ascii=False))


if __name__=='__main__':main()
