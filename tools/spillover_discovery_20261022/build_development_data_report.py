#!/usr/bin/env python3
"""Summarize explicit audited development JSON only; never opens native raw files."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'artifacts/static-spillover-study-20261022'
SCHEMAS={
    'five':'spillover_five_development_point_audit.v1',
    'asr':'gsdeploy.asr19_replication_development_audit.v3',
    'text':'text26_development_audit.v1'}
COMPLETE={'five':'COMPLETE_DEVELOPMENT_AUDIT','asr':'SEVEN_CONDITIONS_EIGHT_ATTEMPTS_AUDITED','text':'ALL_PREDECLARED_ATTEMPTS_AUDITED'}
TERMS='QTLPME'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def compact(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def finite(value):return type(value) in (int,float) and math.isfinite(value)


def request_completion(c,u):
    """Keep request completion independent of utility eligibility and stability."""
    count=lambda v:v if type(v) is int and v>=0 else None
    offered=count(c.get('cohort',{}).get('offered'))
    offered_evidence='cohort.offered' if offered is not None else None
    if offered is None:
        offered=count(c.get('measurement_window',{}).get('offered_requests'))
        if offered is not None:offered_evidence='measurement_window.offered_requests'
    statuses=c.get('status_counts',{})
    statuses=statuses if isinstance(statuses,dict) and statuses and all(count(v) is not None for v in statuses.values()) else None
    completed=None;completed_evidence=None
    if statuses is not None:
        completed=statuses.get('completed',0);completed_evidence='status_counts.completed'
        if offered is None:offered=sum(statuses.values());offered_evidence='sum(status_counts)'
        if sum(statuses.values())!=offered:raise ValueError('Audited status counts differ from offered cohort')
    summaries=c.get('request_summaries',{}).get('task',{})
    if summaries and all(count(s.get('offered')) is not None and count(s.get('completed')) is not None for s in summaries.values()):
        summary_offered=sum(s['offered'] for s in summaries.values())
        summary_completed=sum(s['completed'] for s in summaries.values())
        if offered is not None and offered!=summary_offered:raise ValueError('Audited task summary offered count differs')
        if completed is not None and completed!=summary_completed:raise ValueError('Audited task summary completed count differs')
        if offered is None:offered=summary_offered;offered_evidence='sum(request_summaries.task.offered)'
        if completed is None:completed=summary_completed;completed_evidence='sum(request_summaries.task.completed)'
    # Five-point audits expose explicit window/drain completion counts by task.
    tasks=c.get('per_task',{})
    if completed is None and tasks and all(count(t.get('completed_within_window')) is not None and count(t.get('completed_during_drain')) is not None for t in tasks.values()):
        completed=sum(t['completed_within_window']+t['completed_during_drain'] for t in tasks.values())
        completed_evidence='sum(per_task.completed_within_window + completed_during_drain)'
    # Direct utility counters are supported; utility_complete/eligible never imply request counts.
    if offered is None and count(u.get('offered_requests')) is not None:
        offered=u['offered_requests'];offered_evidence='utility.offered_requests'
    if completed is None and count(u.get('completed_requests')) is not None:
        completed=u['completed_requests'];completed_evidence='utility.completed_requests'
    if offered is not None and completed is not None and completed>offered:raise ValueError('Completed count exceeds offered cohort')
    missing=offered-completed if offered is not None and completed is not None else None
    completeness='UNKNOWN' if missing is None else '完整' if missing==0 else f'未完成{missing}'
    status=c.get('status') or 'UNKNOWN'
    if status=='UNKNOWN':
        reasons=c.get('invalid_reasons',[])
        if reasons and reasons[0] in {'completed','failed','aborted'}:status=reasons[0]
    return dict(offered_requests=offered,completed_requests=completed,not_completed_requests=missing,
        request_status_counts=statuses,run_status=status,completeness=completeness,
        completion_evidence={'offered':offered_evidence,'completed':completed_evidence},
        utility_eligible=u.get('eligible') if type(u.get('eligible')) is bool else None,
        utility_complete=u.get('utility_complete') if type(u.get('utility_complete')) is bool else None,
        utility_reason=u.get('reason'))


def normalize(kind,c,selected=True):
    if kind=='five':
        u=c['measured_six_term_utility'];rates=c['workload'];valid=c['strict_import_passed'] and u.get('eligible') is True
        seed=51;identifier=c['id'];plan=c['plan'];quality=u.get('quality_evidence',{})
    elif kind=='asr':
        u=c['utility'];plan=c['plan'];rates={t:math.fsum(row.values()) for t,row in plan['routing'].items()}
        valid=c['validity'] is True and c['strict_import_passed'] is True and u.get('eligible') is True
        seed=c['seed'];identifier=c['id']+('' if selected else '__preserved_user_pause_attempt1')
        quality=u.get('quality_evidence',{})
    else:
        u=c['utility'];plan=c.get('plan');rates=c.get('workload');valid=c.get('utility_comparison_allowed') is True and c.get('strict_import_passed') is True and u.get('eligible') is True
        seed=c.get('seed');identifier=c['id'];quality=c.get('actual_output_quality',{}).get('evidence',{})
    stability=c.get('stability',{}).get('stability_state_v2','UNKNOWN')
    valid=bool(valid and selected and stability in {'steady','recoverable_congested'} and finite(u.get('U'))
        and all(finite(u.get(k+'_norm')) for k in TERMS))
    source=c.get('source',{})
    memory=c.get('memory_evidence',{})
    memory_valid=memory.get('resource_valid') if type(memory.get('resource_valid')) is bool else None
    reason=u.get('reason')
    if memory_valid is False:
        reason=(reason+'; ' if reason else '')+'MEMORY_RESOURCE_INVALID'
    return dict(dataset=kind,id=identifier,selected_for_condition=selected,seed=seed,
        workload=rates,deployment=plan.get('deployment') if plan else None,routing=plan.get('routing') if plan else None,
        **request_completion(c,u),stability=stability,valid_utility=valid,
        memory_resource_valid=memory_valid,safe_memory_mib=memory.get('safe_memory_mib'),
        utility_ineligibility_evidence=reason,
        **{k+'_norm':u.get(k+'_norm') for k in TERMS},U=u.get('U') if valid else None,
        raw_sha256=source.get('sha256'),raw_source_path=source.get('path'),
        actual_quality_status=quality.get('status','UNKNOWN'),
        source_role=c.get('role'),diagnostic_U_used=False,
        boundary='仅开发证据；失败/缺测无合法U；非独立验证、最终确认或全池最优')


def pair(kind,byid,a,b,claimed,paired,role):
    ca,cb=byid.get(a),byid.get(b)
    allowed=bool(ca and cb and ca['valid_utility'] and cb['valid_utility'] and paired)
    delta=cb['U']-ca['U'] if allowed else None
    if allowed and (not finite(claimed) or not math.isclose(delta,claimed,rel_tol=1e-9,abs_tol=1e-10)):
        raise ValueError('Audited paired effect differs from valid row values')
    return dict(dataset=kind,a=a,b=b,direction='b minus a',role=role,
        same_trace_verified_by_source=paired,comparable=allowed,delta_U=delta,
        reason='双方有效且原审计确认同trace' if allowed else 'PENDING或失败/不完整/原审计未确认可比；禁止诊断U替代',
        boundary='单seed结构对照或开发seed配对；不证明可靠收益/全池最优')


def load_audit(kind,path):
    path=Path(path).resolve()
    if path.name!='audit.json':raise ValueError('Only explicit audit.json documents are accepted')
    if any('validation' in p or 'confirmation' in p for p in path.parts):raise ValueError('Heldout/confirmation inputs forbidden')
    if not path.is_file():return dict(dataset=kind,status='PENDING',reason='明确审计文件尚未生成',path=str(path)),[],[],None
    doc=json.loads(path.read_text())
    if doc.get('schema')!=SCHEMAS[kind]:raise ValueError('Unsupported/mismatched audit schema: '+str(path))
    before=sha(path)
    source=dict(dataset=kind,path=str(path),sha256=before,schema=doc['schema'],audit_status=doc['status'])
    # A report may carry partial rows, but they never become complete evidence.
    complete=doc.get('status')==COMPLETE[kind]
    if complete and kind=='five' and len(doc['conditions'])!=5:raise ValueError('Five-point audit must contain five conditions')
    if complete and kind=='asr' and (len(doc['conditions'])!=7 or len(doc['excluded_user_paused_attempts'])!=1 or doc['all_attempt_count']!=8):
        raise ValueError('ASR v3 audit must retain seven selected conditions and one paused attempt')
    if complete and kind=='text' and {c['label'] for c in doc['conditions']}!={
        'text26_n2_specialist_share_4of7','text26_n2_specialist_share_1of2','text26_n1_specialist_share_1of2','text26_n3_specialist_share_4of7'}:
        raise ValueError('Text audit must retain all four predeclared plan labels')
    rows=[normalize(kind,c) for c in doc.get('conditions',[])]
    if kind=='asr':rows += [normalize(kind,c,False) for c in doc.get('excluded_user_paused_attempts',[])]
    byid={r['id']:r for r in rows};pairs=[]
    if len(byid)!=len(rows):raise ValueError('Duplicate condition IDs inside one audit')
    if kind=='five':
        for p in doc.get('same_load_paired_comparisons',[]):
            pairs.append(pair(kind,byid,p['a'],p['b'],p['measured_delta_U'],p.get('same_ordered_task_sample_schedule') is True,'same-load route'))
    elif kind=='asr':
        for p in doc.get('paired',{}).get('pairs',[]):
            seed=p['seed'];a=next((r['id'] for r in rows if r['selected_for_condition'] and r['seed']==seed and r['deployment'].get('asr')==2 and r['routing']['asr'].get('generalist_2',0)==0),None)
            b=next((r['id'] for r in rows if r['selected_for_condition'] and r['seed']==seed and r['deployment'].get('asr')==2 and r['routing']['asr'].get('generalist_2',0)>0),None)
            pairs.append(pair(kind,byid,a,b,p.get('delta_U_split_minus_nosplit'),p.get('status')=='VALID_PAIRED_BLOCK','same-seed split minus no split'))
        n3=next((r['id'] for r in rows if r['selected_for_condition'] and r['seed']==51 and r['deployment'].get('asr')==3),None)
        for p in doc.get('n3_structural_controls',[]):
            pairs.append(pair(kind,byid,p['reference'],n3,p.get('delta_U_n3_minus_reference'),p.get('n3_valid') is True and p.get('reference_valid') is True,'seed51 replica expansion'))
    else:
        for p in doc.get('same_trace_comparisons',[]):
            pairs.append(pair(kind,byid,p['a'],p['b'],p.get('delta_U'),p.get('same_ordered_task_sample_schedule') is True and p.get('same_trace_sha256',True) is True,'four-plan same-trace contrast'))
    if not complete:
        for r in rows:r.update(valid_utility=False,U=None,boundary='源审计尚未完成，保留部分诊断数据但不比较U')
        for p in pairs:p.update(comparable=False,delta_U=None,reason='PENDING: 源审计尚未完成')
    if sha(path)!=before:raise ValueError('Audit changed during read')
    return dict(**source,status='AUDITED' if complete else 'PENDING',rows=len(rows)),rows,pairs,doc.get('paired') if kind=='asr' else None


def write_csv(path,rows,fields):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:compact(row[k]) if isinstance(row.get(k),(dict,list)) else row.get(k) for k in fields})


def build(paths,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    sources=[];rows=[];pairs=[];asr_stats=None
    for kind in ('five','asr','text'):
        source,r,p,stats=load_audit(kind,paths[kind]);sources.append(source);rows.extend(r);pairs.extend(p)
        if kind=='asr':asr_stats=stats
    # Mark overlapping evidence; never pretend the original seed51 pair is a new repetition.
    seen={}
    for row in rows:
        digest=row['raw_sha256'];row['same_raw_already_shown_as']=seen.get(digest) if digest else None
        if digest:seen.setdefault(digest,row['dataset']+'/'+row['id'])
    report=dict(schema='gsdeploy.development_data_report.v1',status='COMPLETE_AUDITED_SOURCES' if all(s['status']=='AUDITED' for s in sources) else 'PENDING',
        sources=sources,conditions=rows,paired_comparisons=pairs,source_asr_development_statistics=asr_stats,
        only_audited_json_read=True,native_raw_files_opened=False,heldout_results_read=False,
        failed_or_missing_comparison_forbidden=True,diagnostic_U_used=False,production_math_changed=False,
        full_pool_optimality_claim=False,final_confirmation=False,gpu_started=False)
    save(output/'report.json',report)
    write_csv(output/'conditions.csv',rows,['dataset','id','selected_for_condition','seed','workload','deployment','routing','completeness','offered_requests','completed_requests','not_completed_requests','request_status_counts','completion_evidence','run_status','utility_eligible','utility_complete','utility_reason','utility_ineligibility_evidence','memory_resource_valid','safe_memory_mib','stability','valid_utility',*[k+'_norm' for k in TERMS],'U','same_raw_already_shown_as','actual_quality_status','raw_sha256','raw_source_path','boundary'])
    write_csv(output/'paired-comparisons.csv',pairs,['dataset','a','b','direction','role','same_trace_verified_by_source','comparable','delta_U','reason','boundary'])
    write_csv(output/'sources.csv',sources,['dataset','status','schema','audit_status','path','sha256','rows','reason'])
    fmt=lambda v:'—' if v is None else f'{v:.6f}' if finite(v) else str(v)
    lines=['# 开发实验数据一页', '',f"状态：{report['status']}。Q/T/L/P/M/E为既定六项归一化成本，越小越好；U越大越好。仅汇总已审计JSON，不读取raw、heldout或正在采集的数据。",'',
        '| 开发资料 | 审计状态 |','|---|---|']
    lines += [f"| {s['dataset']} | {s['status']}：{s.get('audit_status',s.get('reason'))} |" for s in sources]
    lines += ['', '完整性、运行状态、源效用资格和稳定性分别展示：完整请求不等于效用有效；eligible不用于推断完成数。明确计数缺失时完整性为UNKNOWN。', '',
        '| 条件 | seed | 完整性 | offered/completed/未完成 | 运行状态 | 源eligible | 效用有效 | 效用原因 | 稳定性 | Q | T | L | P | M | E | U |',
        '|---|---:|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    number=lambda v:'UNKNOWN' if v is None else str(v)
    for r in rows:
        counts='/'.join(number(r[k]) for k in ('offered_requests','completed_requests','not_completed_requests'))
        eligible='UNKNOWN' if r['utility_eligible'] is None else str(r['utility_eligible'])
        lines.append(f"| {r['dataset']}/{r['id']} | {r['seed']} | {r['completeness']} | {counts} | {r['run_status']} | {eligible} | {'有效' if r['valid_utility'] else '无效'} | {r['utility_ineligibility_evidence'] or '—'} | {r['stability']} | "+' | '.join(fmt(r[k+'_norm']) for k in TERMS)+f" | {fmt(r['U'])} |")
    for r in rows:
        if r['memory_resource_valid'] is False:
            lines += ['',f"{r['dataset']}/{r['id']}：请求{r['completeness']}，稳定性{r['stability']}；原审计memory.resource_valid=False，safe_memory_mib={r['safe_memory_mib']}，M_norm={r['M_norm']}。内存资源不合格使U为空，不能归因为unstable；诊断算术不得参与比较。"]
    lines += ['', '负载、部署和精确绝对路由（JSON数值原样保留，无四舍五入）：','', '| 条件 | 负载rps | 部署 | 精确路由rps |','|---|---|---|---|']
    for r in rows:lines.append(f"| {r['dataset']}/{r['id']} | {compact(r['workload'])} | {compact(r['deployment'])} | {compact(r['routing'])} |")
    lines += ['', '| 同条件对照 b−a | 可比较 | ΔU |','|---|---|---:|']
    for p in pairs:lines.append(f"| {p['dataset']}: {p['b']} − {p['a']} | {p['comparable']} | {fmt(p['delta_U'])} |")
    if asr_stats:
        lines += ['',f"ASR重复开发有效seed对：{asr_stats.get('valid_pair_seeds')}；n={asr_stats.get('n_valid_pairs')}；原审计平均分流ΔU={fmt(asr_stats.get('mean_split_minus_nosplit_U'))}，95% t区间={asr_stats.get('student_95_interval')}，双侧符号翻转p={fmt(asr_stats.get('exact_two_sided_sign_flip_p'))}。只引用源审计统计，不把缺失/失败对补入分母。"]
    lines += ['', '失败、未完成、稳定性不通过或缺测的U为空，不能用diagnostic U代替比较。ASR原seed51对在五点资料和重复资料中可能重复展示，CSV的same_raw_already_shown_as明确标注，不作为新增重复样本。', '',
        '这些是开发数据：seed51曾参与候选/追加实验选择，重复seed数很少；有限部署/路由、单seed扩容对照均不证明独立确认、可靠普遍收益或全模型池最优。text资料缺失时保持PENDING，不自动推断赢家。完整CSV和JSON保留所有失败尝试及来源指纹。','']
    (output/'README.zh.md').write_text('\n'.join(lines))
    save(output/'manifest.json',dict(script={'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        inputs=[s for s in sources if 'sha256' in s],files={p.name:sha(p) for p in output.iterdir() if p.is_file()},gpu_started=False))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--five-audit',type=Path,default=STUDY/'five-development-point-audit-v1/audit.json')
    parser.add_argument('--asr-audit',type=Path,default=STUDY/'asr19-replication-development-audit-v3/audit.json')
    parser.add_argument('--text-audit',type=Path,default=STUDY/'text26-development-audit-v1/audit.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=build({'five':args.five_audit,'asr':args.asr_audit,'text':args.text_audit},args.output)
    print(json.dumps({'status':report['status'],'conditions':len(report['conditions']),'comparable_pairs':sum(p['comparable'] for p in report['paired_comparisons']),'output':str(args.output.resolve())},ensure_ascii=False))


if __name__=='__main__':main()
