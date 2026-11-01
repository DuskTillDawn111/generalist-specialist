"""Compare registered terminal development observations across explicit packets.

Ongoing queues may supply *completed terminal receipts* only. This ledger never
reads live raw files or promotes the best observed condition to a search optimum.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import traceback

spec=importlib.util.spec_from_file_location('_development_comparison_audit',Path(__file__).with_name('analyze_fine_development.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def comparison_contract(packet):
    config=packet['config']
    fields=['workload','weights','latency_reference_ms','gpu_memory_mib','horizon_seconds',
            'drain_seconds','arrival_process','energy_reference_watts','startup_amortization_seconds','utility_schema']
    return {**{k:config.get(k) for k in fields},
            'quality_identity':config['catalog']['quality']['identity']}


def compare(registrations, output):
    output=Path(output)
    if output.exists():raise FileExistsError('Never overwrite measured comparisons')
    modules=audit.api(); observations=[]; sources=[]
    queue_snapshots=[];comparison_contracts=[]
    for packet_path,queue_path in registrations:
        packet_path=Path(packet_path);queue_path=Path(queue_path)
        packet=json.loads(packet_path.read_text());queue_bytes=queue_path.read_bytes();queue=json.loads(queue_bytes)
        assert modules['static_study'].validate_packet(packet)['valid']
        assert queue['packet']==audit.exposure.receipt(packet_path)
        comparison_contracts.append(comparison_contract(packet))
        if comparison_contracts[-1]!=comparison_contracts[0]:
            raise ValueError('Cannot compare different workload/scoring/quality contracts')
        snapshot_name=f'queue-snapshot-{len(queue_snapshots)+1}.json';queue_snapshots.append((snapshot_name,queue_bytes))
        sources.append(dict(packet=audit.exposure.receipt(packet_path),
            queue=dict(path=str(queue_path.resolve()),sha256=hashlib.sha256(queue_bytes).hexdigest(),
                       historical_snapshot=snapshot_name),queue_status=queue['status']))
        registered={row['job_id']:row for row in queue['jobs']}
        assert len(registered)==len(queue['jobs']) and set(registered)<={j['id'] for j in packet['jobs']}
        for job in packet['jobs']:
            record=registered.get(job['id'])
            if record is None:
                observations.append(dict(job_id=job['id'],label=job['labels'][0],seed=job['seed'],
                    plan=job['plan'],audit_status='DECLARED_NO_TERMINAL_RECEIPT',U=None));continue
            path=Path(record['raw']['path']);assert audit.exposure.receipt(path)==record['raw']
            if path.name!='run.json':raise ValueError('Only immutable terminal raw accepted')
            raw=json.loads(path.read_text())
            try:result=audit.analyze_job(packet,job,raw,path,modules)
            except (AssertionError,KeyError,ValueError,TypeError,IndexError,OSError) as exc:
                result=dict(job_id=job['id'],label=job['labels'][0],seed=job['seed'],plan=job['plan'],
                    audit_status='AUDIT_INCOMPLETE_OR_FAILED',U=None,audit_error=f'{type(exc).__name__}: {exc}',
                    audit_traceback=traceback.format_exc(),raw_status=raw.get('status'),raw_errors=raw.get('errors'))
            result['raw_receipt']=record['raw'];observations.append(result)
    assert len({r['job_id'] for r in observations})==len(observations),'Same physical identity duplicated across packets'
    comparisons=[]
    for seed in sorted({r['seed'] for r in observations}):
        rows=[r for r in observations if r['seed']==seed]
        verified=[r for r in rows if r['audit_status']=='VERIFIED_TERMINAL_EVIDENCE']
        cohorts={r['cohort_sha256'] for r in verified}
        if len(cohorts)>1:raise ValueError('Cannot compare different task/input/arrival cohorts')
        eligible=[r for r in verified if r['utility'].get('eligible') and r['memory_pass'] and r['energy_pass'] and r['shutdown_pass']]
        if any(type(r['utility'].get('U')) not in (int,float) or not math.isfinite(r['utility']['U']) for r in eligible):
            raise ValueError('Eligible observations require finite measured U')
        eligible.sort(key=lambda r:r['utility']['U'],reverse=True)
        ranking=[dict(job_id=r['job_id'],label=r['label'],deployment=r['plan']['deployment'],routing=r['plan']['routing'],
            U=r['utility']['U'],stability=r['stability']['stability_state_v2'],
            six_terms={t:r['utility'][t+'_norm'] for t in 'QTLPME'}) for r in eligible]
        comparisons.append(dict(seed=seed,declared_conditions=len(rows),audited_terminals=len(verified),
            eligible_conditions=len(eligible),ranking_among_observed_eligible=ranking,
            exact_cohort_equal=len(cohorts)==1,
            pairwise_deltas=[dict(left=a['job_id'],right=b['job_id'],U_left_minus_right=a['utility']['U']-b['utility']['U'],
                term_left_minus_right={t:a['utility'][t+'_norm']-b['utility'][t+'_norm'] for t in 'QTLPME'})
                for i,a in enumerate(eligible) for b in eligible[i+1:]],
            ineligible_or_pending=[dict(label=r['label'],audit_status=r['audit_status'],
                raw_status=r.get('raw_status'),stability=r.get('stability'),U=None) for r in rows if r not in eligible]))
    result=dict(schema='gsdeploy.explicit_development_comparison.v1',observations=observations,comparisons=comparisons,
        sources=sources,algorithm_selected=False,predictor_validated=False,full_pool_optimality=False,
        comparison_contract=comparison_contracts[0] if comparison_contracts else None,
        limitations=['Only explicitly declared development conditions; incomplete, failed, and ineligible conditions retained.',
            'Best observed eligible condition is not an algorithmic or physical all-pool optimum.',
            'Single-seed differences are descriptive; no confidence interval or final replication claim.',
            'Unknown/unexecuted alternatives cannot be ruled out by this comparison.'])
    output.mkdir(parents=True,exist_ok=False)
    for name,content in queue_snapshots:(output/name).write_bytes(content)
    (output/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['# 已声明开发方案的实测比较','',
        '只列已登记终态；队列进行中时保留未完成条件。U越大越好，不代表算法已经自主选出最优。','',
        '| seed | 具体方案 | 部署 | U | 稳定性 |','|---|---|---|---:|---|']
    for group in comparisons:
        for row in group['ranking_among_observed_eligible']:
            lines.append(f"| {group['seed']} | {row['label']} | {json.dumps(row['deployment'],ensure_ascii=False)} | {row['U']:.9f} | {row['stability']} |")
        for row in group['ineligible_or_pending']:
            lines.append(f"| {group['seed']} | {row['label']} | — | UNKNOWN/INELIGIBLE | {row['audit_status']} |")
    (output/'README.zh.md').write_text('\n'.join(lines)+'\n')
    bound=[Path(__file__),Path(audit.__file__),Path(audit.base.__file__),Path(audit.exposure.__file__)]
    (output/'manifest.json').write_text(json.dumps(dict(sources=sources,tools=[audit.exposure.receipt(p) for p in bound],
        report=audit.exposure.receipt(output/'report.json'),readme=audit.exposure.receipt(output/'README.zh.md')),indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',nargs=2,action='append',required=True,metavar=('PACKET','QUEUE'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=compare(args.registration,args.output)
    print(json.dumps(dict(output=str(args.output.resolve()),observations=len(report['observations']),
        seeds=[{k:v for k,v in g.items() if k in ['seed','declared_conditions','audited_terminals','eligible_conditions']} for g in report['comparisons']]),ensure_ascii=False))


if __name__=='__main__':main()
