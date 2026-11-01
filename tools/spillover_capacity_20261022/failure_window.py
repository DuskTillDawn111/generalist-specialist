#!/usr/bin/env python3
"""Deadline-qualified descriptive slice of signed terminal analysis; no raw read."""
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path


def module(name):
    spec=importlib.util.spec_from_file_location('_failure_window_'+name,Path(__file__).with_name(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


analysis=module('analyze_sustained');plot=module('plot_sustained');legacy=analysis.legacy


def slice_job(job):
    original=copy.deepcopy(job);rows=original['request_diagnostics'];window=original['measurement_timing']['measurement_window']
    start,end=window['start_time_ns'],window['end_time_ns'];cutoff=min(end,start+600_000_000_000)
    unix_cutoff=window['start_unix_ns']+(cutoff-start);unix_start=window['start_unix_ns']
    intervals=[];classified=[]
    for row in rows:
        rec=row['original_record'];completion=rec.get('completion_time_ns');released=type(rec.get('scheduled_time_ns')) is int
        available=row['service_status']=='AVAILABLE';native=row['quality']['status']=='AVAILABLE'
        if available and type(completion) is int and completion<=cutoff and native:
            classification='QUALIFIED_COMPLETED_WITHIN_DEADLINE'
        elif available and type(completion) is int and completion>cutoff:
            classification='COMPLETED_AFTER_DEADLINE_NOT_QUALIFIED'
        elif not released:classification='UNRELEASED_PRIMARY_UNKNOWN'
        else:classification='RELEASED_WITHOUT_QUALIFIED_COMPLETION_UNKNOWN'
        row['deadline_qualification']=classification
        row['terminal_service_ms']=row['service_ms'];row['terminal_quality']=row['quality']
        if available:
            left=max(unix_start,row['worker_start_unix_ns']);right=min(unix_cutoff,row['worker_end_unix_ns'])
            if right>left:intervals.append(dict(alias=row['model'],start_ns=left,end_ns=right,role=row['role'],request_id=row['request_id']))
        if classification!='QUALIFIED_COMPLETED_WITHIN_DEADLINE':
            row.update(service_status='UNKNOWN',service_ms=None,handler_ms=None,
                quality=dict(status='UNKNOWN',reason=classification,loss=None),activity=None)
        classified.append(row)
    activity=legacy.common_activity(intervals,original['active_aliases'],sum(r['service_status']!='AVAILABLE' for r in classified))
    for row in classified:
        if row['service_status']=='AVAILABLE':row['activity']=legacy.interval_exposure(row['worker_start_unix_ns'],row['worker_end_unix_ns'],
            activity['segments'],original['active_aliases'],row['model'])
    aliases={};per_input=[]
    for alias in original['active_aliases']:
        seq=[r for r in classified if r['model']==alias];p=[r for r in seq if r['role']=='primary'];s=[r for r in seq if r['role']=='support']
        aliases[alias]=dict(primary=legacy.summarize_requests(p),support=legacy.summarize_requests(s),
            primary_deadline_classes=dict(Counter(r['deadline_qualification'] for r in p)),support_deadline_classes=dict(Counter(r['deadline_qualification'] for r in s)))
        for pool_id in original['input_pool_ids']:
            part=[r for r in p if r['pool_id']==pool_id]
            per_input.append(dict(alias=alias,pool_id=pool_id,**legacy.summarize_requests(part),
                covered_primary_count=sum(r.get('activity') is not None and r['activity']['all_active_aliases_busy_fraction']>=.9 for r in part)))
    return dict(schema='gsdeploy.sustained_failure_window.job.v1',job_id=original['job_id'],state=original['state'],block=original['block'],
        status='INSUFFICIENT_DIAGNOSTIC_EVIDENCE',raw_status=original['raw_status'],active_aliases=original['active_aliases'],
        input_pool_ids=original['input_pool_ids'],aliases=aliases,per_input=per_input,request_diagnostics=classified,
        coverage=analysis.coverage_gate(classified,original['active_aliases'],original['state']),common_activity=activity,
        deadline=dict(start_time_ns=start,cutoff_time_ns=cutoff,cutoff_unix_ns=unix_cutoff,duration_seconds=(cutoff-start)/1e9,
            policy='min(measurement_end, measurement_start + frozen600s); qualify by host terminal completion within cutoff'),
        observed_handler_intervals_clipped_to_deadline=intervals,original_resource_declarations=original['resource_evidence'],
        original_analysis_contract_flags=original['contract_flags'],raw_files_read=0,U=None,
        limitations=['Terminal completions after cutoff remain preserved but are UNKNOWN for deadline-qualified service/quality/coverage.',
            'In-flight handler intervals are clipped at deadline and contribute only observed context, never fabricated completion.',
            'Completed-only statistics are censored; full96-primary coverage and cross-block repeatability remain insufficient.',
            'Original runner resource declaration does not prove full-window sampling coverage; independent failure audit must be read separately.'])


def paired_rows(base,joint):
    return analysis.repetition_pairs(base,joint)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();directory=args.analysis.resolve();published=plot.load_plot_data(directory)
    manifest=json.loads((directory/'manifest.json').read_text());report=json.loads((directory/'report.json').read_text())
    jobs=[slice_job(json.loads((directory/item['detail_file']).read_text())) for item in report['jobs']]
    for item in report['jobs']:
        if plot.sha(directory/item['detail_file'])!=manifest['files'][item['detail_file']]:raise ValueError('Analysis changed during slice')
    audit=json.loads(args.audit.read_text())
    audit_receipt=legacy.file_receipt(args.audit)
    by={(j['block'],j['state']):j for j in jobs};pairs=[]
    for block in [1,2]:
        if (block,'B') in by and (block,'C') in by:pairs.extend(paired_rows(by[(block,'B')],by[(block,'C')]))
    if args.output.exists():raise FileExistsError('Never overwrite failure evidence')
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,value):(args.output/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    overview=[]
    for job in jobs:
        name=job['job_id']+'.json';save(name,job);overview.append({k:v for k,v in job.items() if k not in ['request_diagnostics','common_activity','observed_handler_intervals_clipped_to_deadline']});overview[-1]['detail_file']=name
    save('report.json',dict(schema='gsdeploy.sustained_failure_window.v1',status='INSUFFICIENT_DIAGNOSTIC_EVIDENCE',jobs=overview,
        paired_primary_repetitions=pairs,source_analysis=published['source_analysis'],independent_audit=audit_receipt,
        memory_full_window_coverage=audit.get('memory_full_window_coverage'),audit_research_gate=audit.get('research_gate'),
        raw_files_read=0,U=None,fitted_parameters=False,ranking_performed=False,independent_formal_validation=False))
    lines=['# 持续共同活动失败批次：截止窗口内描述','',
        '状态：INSUFFICIENT_DIAGNOSTIC_EVIDENCE。原始终态与analysis-v1保持原样，本报告仅增加严格截止切片。', '',
        '| job | alias | 窗口内primary完成/计划 | 截止后primary完成 | primary未释放 | 窗口内support完成 | 截止后support完成 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for j in jobs:
        for alias,x in j['aliases'].items():
            pc=x['primary_deadline_classes'];sc=x['support_deadline_classes'];lines.append(f"| {j['job_id']} | {alias} | {x['primary']['service_available_requests']}/{x['primary']['planned_requests']} | {pc.get('COMPLETED_AFTER_DEADLINE_NOT_QUALIFIED',0)} | {pc.get('UNRELEASED_PRIMARY_UNKNOWN',0)} | {x['support']['service_available_requests']} | {sc.get('COMPLETED_AFTER_DEADLINE_NOT_QUALIFIED',0)} |")
    lines.extend(['','截止后完成仍保留terminal_service_ms/terminal_quality；窗口内统计将其记UNKNOWN，不补齐primary，不加入有效服务、质量、B/C比值或覆盖。真实handler在截止前的部分仅用于观察活动上下文。','',
        'C的完整96条primary与两块重复性均未通过。每输入已有两次高重叠观察不能替代96条全部成功与正交叠门槛。','',
        '独立失败审计发现显存尾段约0.316秒未被采样括住，因此全窗口资源覆盖UNKNOWN；观测安全峰值合法不能消除该缺口。电能覆盖另有独立验证。原runner resource_valid声明仅保留为来源信息。','',
        '这些是失败开发诊断的删失统计，不是开放负载容量、总效用或分流收益。U=null。'])
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    save('manifest.json',dict(source_analysis=published['source_analysis'],audit=audit_receipt,tool=legacy.file_receipt(__file__),
        files={x.name:plot.sha(x) for x in args.output.iterdir() if x.is_file()},raw_files_read=0,U=None))
    print(json.dumps(dict(status='INSUFFICIENT_DIAGNOSTIC_EVIDENCE',jobs=len(jobs),output=str(args.output.resolve()),raw_files_read=0,U=None)))


if __name__=='__main__':main()
