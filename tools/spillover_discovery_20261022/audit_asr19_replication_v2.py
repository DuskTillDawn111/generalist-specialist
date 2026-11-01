#!/usr/bin/env python3
"""ASR1.9 development audit preserving invalid attempts and missing pairs.

All seven declared raw paths must exist before full audit. Invalid attempts
remain validity=false/U=null, never receive a diagnostic substitute. Only
fully valid within-seed pairs contribute to descriptive CI/sign-flip results.
This tool never fits, reruns, changes gates, or starts GPU work.
"""
import argparse
from collections import Counter
import copy
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_asr19_replication as v1
from audit_interim_data import receipt,read,write,csv_write,close,TERMS
from gsdeploy.config import SystemConfig
from gsdeploy.evaluation import evaluate_run
from gsdeploy.calibration import _baseline_identity
from gsdeploy.quality import quality_digest
from gsdeploy.trace import make_trace,trace_digest,expand_routes
from gsdeploy.routing import Router
from scipy.stats import t as student_t


def import_preserving_invalid(entry):
    raw=read(entry['run_path']);packet=read(entry['packet_path']);job=next(j for j in packet['jobs'] if j['labels']==[entry['label']])
    config=SystemConfig(**raw['config']);identity=copy.deepcopy(raw['executed_measurement_identity'])
    if quality_digest(identity)!=raw['executed_measurement_id'] or raw['experiment_job_id']!=job['id'] or raw['experiment_phase']!='calibration':
        raise ValueError('Invalid execution/job identity')
    actual_workers=copy.deepcopy(identity.get('workers',{}))
    for worker in actual_workers.values():worker.get('env',{}).pop('GS_CHECKPOINT_FILES_JSON',None)
    if actual_workers!=job['identity']['workers']:raise ValueError('Worker identity changed')
    for key,value in job['identity'].items():
        if key not in ('workers','hardware_contract') and identity.get(key)!=value:raise ValueError('Executed identity changed: '+key)
    if identity['hardware_contract'].get('device_uuid')!=raw['preflight']['gpu']['uuid']:raise ValueError('GPU identity changed')
    original=read(v1.ORIGINAL)['config']
    for key in ('models','quality','safe_factor','capacity_factors','memory_table_mib','task_replica_profiles'):
        if config.catalog[key]!=original['catalog'][key]:raise ValueError('Profile identity changed: '+key)
    if _baseline_identity(config.catalog)!=_baseline_identity(original['catalog']):raise ValueError('Baseline identity changed')
    if config.workload!=job['identity']['workload'] or config.random_seed!=entry['seed'] or config.horizon_seconds!=600 or raw['plan']!=job['plan']:
        raise ValueError('Execution config/plan differs')
    transport=raw['transport_config']
    for key,value in dict(random_seed=entry['seed'],measurement_seconds=600,arrival_process=config.arrival_process,warmup_seconds=job['identity']['warmup_seconds'],drain_timeout_seconds=job['identity']['drain_seconds']).items():
        if transport[key]!=value:raise ValueError('Transport changed: '+key)
    trace=make_trace(config.workload,600,entry['seed'],config.arrival_process);records=raw['request_records'];window=raw['measurement_window']
    aliases,routes=expand_routes(raw['plan']['deployment'],raw['plan']['routing']);router=Router(routes)
    if len(trace)!=len(records) or window['trace_sha256']!=trace_digest(trace):raise ValueError('Cohort trace changed')
    for i,((offset,task),record) in enumerate(zip(trace,records)):
        if record['request_id']!=f'measurement:{i}' or record['task']!=task or record['model']!=router.choose(task) or record['scheduled_time_ns']!=window['start_time_ns']+round(offset*1e9):
            raise ValueError('Cohort route/timestamp changed')
    computed=evaluate_run(config,copy.deepcopy(raw));utility=computed['utility']
    cached=raw['evaluation']['utility']
    if utility['U'] is not None:
        condition,rows=v1.import_one(entry)
        condition.update(validity=True,strict_import_passed=True,invalid_reasons=[],identity_verified=True)
        for row in rows:row.update(run_valid=True,status='completed',error=None)
        return condition,rows
    if cached.get('U') is not None or utility.get('eligible'):raise ValueError('Invalid attempt has finite cached U or eligibility')
    for key in [k+'_norm' for k in TERMS]:
        if utility.get(key) is not None and cached.get(key) is not None:close(utility[key],cached[key],'invalid diagnostic/'+key)
    rows=[]
    for record in records:
        completed=record['status']=='completed';submitted=record.get('actual_submit_time_ns')
        rows.append(dict(run_id=entry['id'],seed=entry['seed'],asr_replicas=entry['replicas'],split=entry['split'],
            request_id=record['request_id'],task=record['task'],pool_id=record['pool_id'],model=record['model'],family=aliases[record['model']],
            scheduled_offset_seconds=record['scheduled_offset_seconds'],
            e2e_latency_ms=(record['completion_time_ns']-record['scheduled_time_ns'])/1e6 if completed else None,
            runtime_latency_ms=(record['completion_time_ns']-submitted)/1e6 if completed and submitted else None,
            worker_latency_ms=record.get('worker_latency_ms'),queue_wait_ms=record.get('worker_queue_wait_ms'),
            quality_numerator=None,quality_denominator=None,within_window=completed and record['completion_time_ns']<=window['end_time_ns'],
            run_valid=False,status=record['status'],error=record.get('error')))
    condition=dict(**entry,source=dict(id=entry['id'],**receipt(entry['run_path']),packet_path=entry['packet_path'],packet_sha256=receipt(entry['packet_path'])['sha256']),
        validity=False,strict_import_passed=False,identity_verified=True,
        strict_import_failure='Strict successful-measurement importer rejects this invalid attempt; source identity checked separately, never bypasses utility gate.',
        invalid_reasons=[raw['status'],*raw.get('errors',[]),computed['stability'].get('stability_reason_v2','')],
        utility=utility,U=None,terms={k:utility.get(k+'_norm') for k in TERMS},contributions={k:None for k in TERMS},
        diagnostic_partial_U=utility.get('diagnostic_partial_U'),diagnostic_is_not_utility=True,
        quality_document_sha256=quality_digest(config.catalog['quality']),
        cohort=dict(trace_sha256=window['trace_sha256'],offered=len(records),task_input_sha256=quality_digest([(r['task'],r['pool_id']) for r in records])),
        plan=raw['plan'],status_counts=dict(Counter(r['status'] for r in records)),stability=computed['stability'],weights=config.weights)
    return condition,rows


def statistics_for_valid_pairs(conditions):
    pairs=[];differences=[]
    for seed in v1.PAIR_SEEDS:
        a=next(c for c in conditions if c['seed']==seed and c['replicas']==2 and not c['split'])
        b=next(c for c in conditions if c['seed']==seed and c['replicas']==2 and c['split'])
        if a['cohort']!=b['cohort'] or a['quality_document_sha256']!=b['quality_document_sha256']:raise ValueError('Pair input/trace differs')
        valid=a['validity'] and b['validity'];delta=b['U']-a['U'] if valid else None
        pairs.append(dict(seed=seed,nosplit_valid=a['validity'],split_valid=b['validity'],delta_U_split_minus_nosplit=delta,
            status='VALID_PAIRED_BLOCK' if valid else 'INVALID_PAIRED_BLOCK',
            capacity_rescue_observed=b['validity'] and not a['validity'],
            interpretation='At most this fixed trace supports completion/feasibility rescue; missing nosplit U forbids utility gain comparison' if b['validity'] and not a['validity'] else None))
        if valid:differences.append(delta)
    n=len(differences);mean=statistics.fmean(differences) if n else None
    ci=None;p=None
    if n>=2:
        half=float(student_t.ppf(.975,n-1))*statistics.stdev(differences)/math.sqrt(n)
        ci=[mean-half,mean+half]
        p=sum(abs(math.fsum(s*d for s,d in zip(signs,differences)))>=abs(math.fsum(differences))-1e-12 for signs in itertools.product((-1,1),repeat=n))/2**n
    return dict(pairs=pairs,valid_pair_seeds=[p['seed'] for p in pairs if p['status']=='VALID_PAIRED_BLOCK'],n_valid_pairs=n,
        mean_split_minus_nosplit_U=mean,student_95_interval=ci,exact_two_sided_sign_flip_p=p,
        minimum_possible_two_sided_p=2/2**n if n>=1 else None,
        status='ALL_THREE_VALID' if n==3 else 'INCOMPLETE_VALID_PAIRED_EVIDENCE',
        caveat='Development selected after seed51 observation; invalid blocks are excluded only from numeric pairing and retained in report, not silently discarded. n<=3; no reliable-gain or final-confirmation claim.',
        assumptions='t interval assumes independent normal seed differences; sign flip requires symmetric independent null differences; neither corrects development selection.',final_confirmation=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    entries=v1.roster();ready=v1.readiness(entries);ready['invalid_attempt_policy']='Preserve all existing invalid runs; U remains null; statistics use fully valid same-seed pairs only'
    if args.output:args.output.mkdir(parents=True,exist_ok=False);write(args.output/'readiness.json',ready)
    if args.check_only or ready['missing_raw_paths']:
        print(json.dumps(ready,ensure_ascii=False,indent=2));raise SystemExit(2 if ready['missing_raw_paths'] else 0)
    if not args.output:parser.error('--output required')
    try:
        conditions=[];requests=[]
        for entry in entries:
            c,rows=import_preserving_invalid(entry);conditions.append(c);requests.extend(rows)
        stats=statistics_for_valid_pairs(conditions);n3=next(c for c in conditions if c['replicas']==3);controls=[]
        for other in (c for c in conditions if c['seed']==51 and c['replicas']==2):
            if n3['cohort']!=other['cohort']:raise ValueError('n3 cohort differs')
            controls.append(dict(reference=other['id'],n3_valid=n3['validity'],reference_valid=other['validity'],
                delta_U_n3_minus_reference=n3['U']-other['U'] if n3['validity'] and other['validity'] else None,
                scope='One seed51 replica-structure contrast only; no U comparison when either invalid'))
        report=dict(schema='gsdeploy.asr19_replication_development_audit.v2',status='ALL_SEVEN_ATTEMPTS_AUDITED',
            readiness=ready,conditions=conditions,paired=stats,n3_structural_controls=controls,
            invalid_ids=[c['id'] for c in conditions if not c['validity']],final_confirmation=False,full_pool_optimality=False,
            purpose='Development decision support: continue ASR or inspect text; no fitting or gate activation',gpu_started=False)
        write(args.output/'audit.json',report)
        csv_write(args.output/'plans-six-terms.csv',[dict(id=c['id'],seed=c['seed'],asr_replicas=c['replicas'],split=c['split'],validity=c['validity'],
            **{k+'_norm':c['terms'][k] for k in TERMS},**{k+'_contribution':c['contributions'][k] for k in TERMS},U=c['U'],
            diagnostic_partial_U=c.get('diagnostic_partial_U'),diagnostic_is_not_utility=not c['validity']) for c in conditions])
        csv_write(args.output/'request-latencies.csv',requests)
        lines=['# ASR1.9 开发重复审计（保留失败运行）','',f"七条预声明 attempt 全部保留；无效运行：{report['invalid_ids']}。无效 Q/U 维持 null，不补零，不用 diagnostic_partial_U 参与效用差。",'',
            f"完整有效配对 seed={stats['valid_pair_seeds']}，n={stats['n_valid_pairs']}，平均 split−nosplit U={stats['mean_split_minus_nosplit_U']}，95% t区间={stats['student_95_interval']}，双侧符号翻转 p={stats['exact_two_sided_sign_flip_p']}。"]
        for pair in stats['pairs']:lines.append(f"- seed{pair['seed']}：{pair['status']}，ΔU={pair['delta_U_split_minus_nosplit']}，完成性救援观察={pair['capacity_rescue_observed']}。")
        lines.extend(['','失败不分流与有效分流可描述相同trace下的完成性/容量救援；因前者没有合法 U，不能计算效用提升。统计只用完整有效seed对，失败保留在明细中；样本极小且seed51已用于开发选择，不据此声称可靠收益或最终确认。','',f"n3结构对照：{controls}",'','仅用于继续 ASR 或转向 text 的开发决策，不声称全模型池最优，不改变门禁，不重新拟合，不重跑、不启动 GPU。'])
        (args.output/'report.zh.md').write_text('\n'.join(lines)+'\n')
        inputs=[ready['original_packet'],*ready['packets'],*[dict(path=c['source']['path'],sha256=c['source']['sha256']) for c in conditions]]
        if any(receipt(row['path'])!=row for row in inputs):raise ValueError('Audit inputs changed')
        write(args.output/'manifest.json',dict(inputs=inputs,code=[receipt(__file__),receipt(v1.__file__),receipt(Path(__file__).with_name('audit_interim_data.py'))],
            outputs=[receipt(p) for p in sorted(args.output.iterdir()) if p.is_file()],gpu_started=False))
        print(json.dumps(dict(status=report['status'],invalid_ids=report['invalid_ids'],paired=stats,gpu_started=False),ensure_ascii=False))
    except Exception as error:
        write(args.output/'failure.json',dict(status='AUDIT_FAILED',error_type=type(error).__name__,error=str(error),gpu_started=False));raise SystemExit(3) from error


if __name__=='__main__':main()
