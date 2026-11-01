"""Describe time/input variation in signed failed-diagnostic support records.

Support never increases independent input/sample count. All bins are fixed100s
from common release, assigned by observed native start, and completed after the
declared deadline is excluded. No new calibrated model is activated.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def rec(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def summary(values):
    return dict(count=len(values),mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,
                minimum=min(values) if values else None,maximum=max(values) if values else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();b=args.study.resolve()
    if args.output.exists():raise FileExistsError('Never overwrite development evidence')
    qpath=b/'runs/queue-progress.json';q=json.loads(qpath.read_text())
    if q['status']!='PAUSED_ON_FAILED_ATTEMPT_NO_RETRY':raise ValueError('Only the declared failed development batch')
    r=next(r for r in q['jobs'] if r['job_id']=='block1_C_seed103');rawpath=Path(r['raw']['path'])
    if rec(rawpath)['sha256']!=r['raw']['sha256']:raise ValueError('Terminal receipt differs')
    raw=json.loads(rawpath.read_text());w=raw['measurement_window'];start=w['start_time_ns'];end=start+600_000_000_000
    offset=w['start_unix_ns']-start
    verified=b/'failed-attempt-independent-audit-v1/audit.json';audit=json.loads(verified.read_text())
    if audit['status']!='EXPECTED_WALL_TIMEOUT_EVIDENCE_VERIFIED' or audit['job']['raw']!=r['raw']:
        raise ValueError('Independent failed evidence audit required')
    accepted=[];excluded=Counter()
    for row in raw['request_records']:
        response=row.get('worker_response') or {}
        if row['status']!='completed' or response.get('success') is not True:
            excluded['incomplete_or_native_failure']+=1;continue
        if row['completion_time_ns']>end or response['end_ns']>end+offset:
            excluded['completed_after_declared_cutoff']+=1;continue
        time=(response['start_ns']-w['start_unix_ns'])/1e9
        assert 0<=time<600
        accepted.append(dict(alias=row['model'],role=row['role'],pool_id=row['pool_id'],
            request_id=row['request_id'],start_seconds=time,bin_index=int(time//100),
            service_ms=response['service_ms'],generated_tokens=response.get('generated_tokens'),
            prediction=response.get('prediction')))
    assert len(accepted)==1471 #192 specialist primary +76 G2 primary +1203 support
    rows=[];input_rows=[]
    for alias in raw['job']['active_aliases']:
        for index in range(6):
            group=[r for r in accepted if r['alias']==alias and r['bin_index']==index]
            inputs=defaultdict(list)
            for r in group:inputs[r['pool_id']].append(r)
            per_input={pid:summary([x['service_ms'] for x in v]) for pid,v in sorted(inputs.items())}
            rows.append(dict(alias=alias,start_seconds=index*100,end_seconds=(index+1)*100,
                role_counts=dict(Counter(r['role'] for r in group)),unique_inputs=len(inputs),
                service_ms=summary([r['service_ms'] for r in group]),
                equal_input_mean_of_medians_ms=statistics.mean(v['median'] for v in per_input.values()) if len(inputs)==32 else None,
                missing_inputs=sorted(set(raw['job']['input_pool_ids'])-set(inputs)),
                generated_tokens=summary([r['generated_tokens'] for r in group if isinstance(r['generated_tokens'],(int,float))]),
                per_input=per_input))
        all_input=defaultdict(list)
        for r in accepted:
            if r['alias']==alias:all_input[r['pool_id']].append(r)
        for pid,v in sorted(all_input.items()):
            groups=defaultdict(list)
            for x in v:groups[x['bin_index']].append(x['service_ms'])
            first=summary(groups[0]);later=[x['service_ms'] for x in v if x['bin_index']>=1]
            input_rows.append(dict(alias=alias,pool_id=pid,role_counts=dict(Counter(x['role'] for x in v)),
                per_bin={str(i):summary(values) for i,values in sorted(groups.items())},
                late_over_first_median=statistics.median(later)/first['median'] if later and first['median'] else None,
                prediction_variants=len({json.dumps(x['prediction'],ensure_ascii=False,sort_keys=True) for x in v}),
                token_variants=len({x['generated_tokens'] for x in v})))
    sources=[rec(qpath),rec(rawpath),rec(verified),rec(__file__)]
    result=dict(schema='gsdeploy.support_drift_context.v1',status='DESCRIPTIVE_DEVELOPMENT_ONLY',
        fixed_bins_seconds=100,within_declared600_completed_records=len(accepted),excluded=dict(excluded),
        independent_input_clusters_per_alias=32,bins=rows,per_input_time_comparisons=input_rows,
        U=None,formal_rho=None,predictor_activated=False,GPU_started=False,
        scope='Observed activity context only; bin summaries vary in input coverage and primary/support composition. Support observations share one evolving run and are not independent calibration blocks.',
        limitations=['No cross-block replication','G2 primary third repetition censored','Input services and generation state may drift','Bin coverage missing is UNKNOWN; no partial-pool means presented as full-pool means','No measured saturation table is assumed a load-independent predictor or capacity bound'],sources=sources)
    for item in sources:
        if rec(item['path'])!=item:raise ValueError('Source changed')
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'report.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    lines=['# 已有支持请求的时间分层','','只读已登记终态，严格600秒截止；support不增加独立输入。','',
           '| alias | 秒区间 | 请求数 | 独立输入 | 服务均值ms | 全32输入等权中位数均值ms |','|---|---|---:|---:|---:|---:|']
    for x in rows:
        v=x['equal_input_mean_of_medians_ms'];s='UNKNOWN' if v is None else f'{v:.2f}'
        lines.append(f"| {x['alias']} | {x['start_seconds']}–{x['end_seconds']} | {x['service_ms']['count']} | {x['unique_inputs']} | {x['service_ms']['mean']:.2f} | {s} |")
    lines+=['','不完整输入区间不填全池均值；各角色、逐输入变化和输出变化见JSON。该表不能直接成为已验证的联合服务、容量或效用预测器。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    with (args.output/'manifest.json').open('x') as f:json.dump(dict(inputs=sources,files={x.name:rec(x)['sha256'] for x in args.output.iterdir() if x.is_file()}),f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],accepted=len(accepted),excluded=dict(excluded),bins=len(rows))))


if __name__=='__main__':main()
