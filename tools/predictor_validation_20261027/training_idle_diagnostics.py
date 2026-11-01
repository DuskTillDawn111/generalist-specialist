"""Observed worker idle history in audited training, without model changes.

The fixed minimum cell count is descriptive coverage only, never an admission,
model-selection or prediction gate. No confidence or causal claim is made.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics

from .native_flow import checked, receipt, save
from .probe_validation import read
from .training_composition_diagnostics import sample_stats

MIN_CELL_N = 5
GAP_UPPER_MS = (1., 10., 100., 1000.)


def _correlation(rows):
    if len(rows) < MIN_CELL_N:
        return None, 'BELOW_FIXED_MIN_CELL_N'
    x=[math.log1p(r['idle_gap_ms']) for r in rows];y=[r['service_ms'] for r in rows]
    xm,ym=statistics.mean(x),statistics.mean(y)
    xx=sum((v-xm)**2 for v in x);yy=sum((v-ym)**2 for v in y)
    if xx == 0 or yy == 0:
        return None, 'ZERO_OBSERVED_VARIATION'
    return sum((a-xm)*(b-ym) for a,b in zip(x,y))/math.sqrt(xx*yy), None


def summarize(raw, plan):
    history=[*raw.get('priming_request_records',[]),*raw.get('warmup_request_records',[]),*raw['request_records']]
    by_alias=defaultdict(list)
    for row in history:
        if row.get('status') != 'completed':
            raise ValueError('Complete real worker history required')
        by_alias[row['model']].append(row)
    previous={}
    for alias,rows in by_alias.items():
        prior=None
        for row in sorted(rows,key=lambda r:r['observed_worker_start_time_ns']):
            start=row['observed_worker_start_time_ns'];end=row['worker_response']['end_perf_ns']
            if prior is not None and start < prior['worker_response']['end_perf_ns']:
                raise ValueError('Same-alias history cannot overlap')
            if row['phase']=='measurement':
                previous[row['request_id']]=prior
            prior=row
    cells=defaultdict(list);thirds=defaultdict(Counter);records=[]
    width=raw['deadline_ns']-raw['phase_start_ns']
    for row in raw['request_records']:
        offset=row['scheduled_time_ns']-raw['phase_start_ns']
        if not 0 <= offset < width:
            raise ValueError('Scheduled arrival outside measurement window')
        alias,task,pool=row['model'],row['task'],row['pool_id']
        third=min(2,3*offset//width);thirds[(alias,task,pool)][third]+=1
        prior=previous[row['request_id']];start=row['observed_worker_start_time_ns']
        end=row['worker_response']['end_perf_ns'];service=(end-start)/1e6
        if service <= 0:
            raise ValueError('Positive actual service required')
        gap=(start-prior['worker_response']['end_perf_ns'])/1e6 if prior else None
        record=dict(request_id=row['request_id'],alias=alias,task=task,pool_id=pool,
            arrival_third=third,service_ms=service,idle_gap_ms=gap,
            prior_request_id=prior['request_id'] if prior else None,
            prior_phase=prior['phase'] if prior else None,
            prior_task=prior['task'] if prior else None,
            prior_pool_id=prior['pool_id'] if prior else None)
        records.append(record)
        if gap is not None:cells[(alias,task,pool)].append(record)
    groups=[]
    for (alias,task,pool),rows in sorted(cells.items()):
        bins=defaultdict(list)
        for row in rows:
            idx=next((i for i,bound in enumerate(GAP_UPPER_MS) if row['idle_gap_ms']<=bound),len(GAP_UPPER_MS))
            bins[idx].append(row['service_ms'])
        correlation,reason=_correlation(rows)
        groups.append(dict(alias=alias,task=task,pool_id=pool,n=len(rows),
            arrival_third_counts={str(i):thirds[(alias,task,pool)][i] for i in range(3)},
            each_arrival_third_has_minimum_n=all(thirds[(alias,task,pool)][i]>=MIN_CELL_N for i in range(3)),
            idle_gap_ms=dict(minimum=min(r['idle_gap_ms'] for r in rows),maximum=max(r['idle_gap_ms'] for r in rows),
                mean=statistics.mean(r['idle_gap_ms'] for r in rows)),
            service=sample_stats([r['service_ms'] for r in rows]),
            correlation_log1p_idle_service=correlation,correlation_unknown_reason=reason,
            idle_gap_bins=[dict(index=i,lower_exclusive_ms=None if i==0 else GAP_UPPER_MS[i-1],
                upper_inclusive_ms=GAP_UPPER_MS[i] if i<len(GAP_UPPER_MS) else None,
                n=len(bins[i]),minimum_n_satisfied=len(bins[i])>=MIN_CELL_N,
                observed_service=sample_stats(bins[i]) if bins[i] else None) for i in range(len(GAP_UPPER_MS)+1)]))
    return dict(request_count=len(records),minimum_cell_n=MIN_CELL_N,idle_gap_upper_bounds_ms=list(GAP_UPPER_MS),
        groups=groups,records=records,missing_prior_count=sum(r['idle_gap_ms'] is None for r in records),
        coverage=dict(input_alias_task_strata=len(thirds),strata_with_minimum_n_each_arrival_third=sum(
            all(counts[i]>=MIN_CELL_N for i in range(3)) for counts in thirds.values())),
        semantics='Same owning alias previous real handler END to current real worker START, including priming/warmup predecessors; not request arrival wait or GPU idle-counter time',
        interpretation='Descriptive same-input associations and observed ranges only; idle, alias, context, queued work and input history are not causally separated; no fallback for sparse cells, no fitting or model change')


def run(audit_path,output):
    if Path(output).exists():raise FileExistsError(output)
    audit=read(audit_path)
    if audit.get('schema')!='gsdeploy.training_run_postcollection_audit.v1' or audit.get('status')!='PASS':
        raise ValueError('Passed training source/quality audit required')
    checked(audit['source']);raw=read(checked(audit['raw']));plan=read(checked(audit['plan']))
    if plan.get('phase')!='predictor_training' or plan.get('split')!='training':
        raise ValueError('Training-only evidence required')
    value=dict(schema='gsdeploy.training_idle_history_diagnostics.v1',audit=receipt(audit_path),
        raw=audit['raw'],plan=audit['plan'],source=receipt(__file__),candidate_id=audit['candidate_id'],seed=audit['seed'],
        source_receipts=[receipt(__file__),receipt(sample_stats.__code__.co_filename)],
        **summarize(raw,plan),GPU_started=False,independent_validation=False,model_changed=False,final_gain_eligible=False)
    save(output,value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();v=run(a.audit,a.output)
    print('COMPLETE',v['request_count'],'requests;',v['missing_prior_count'],'missing predecessors; GPU_started=False')
