#!/usr/bin/env python3
"""Read-only v3: exactly one user-pause replacement, all failed attempts kept.

Selection accepts gsdeploy.user_pause_attempt_selection.v1 (the resume driver)
or gsdeploy.asr19_attempt_selection.v1. Both require the same job/path/pause
semantics, reason=user_requested_pause_retry and a bound resume declaration.
Use selected_before_observation=true / selection_depends_on_U=false, or the
legacy selection_policy=user_pause_only_not_utility; contradictory flags fail.
previous_run/user_pause_receipt/declaration carry path+sha256. selected_run
carries path+sha256 for full audit. Only seed53 nosplit attempt1 -> attempt2
is allowed. --check-only checks existence without opening selected raw;
it reads the preserved interrupted attempt to verify its pause association.
--declaration-only provides readiness before the driver publishes selection.
"""
import argparse
import copy
import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_asr19_replication_v2 as v2
from audit_interim_data import receipt,read,write,csv_write,TERMS


def checked(row):
    if receipt(row['path'])!={k:row[k] for k in ('path','sha256')}:raise ValueError('Attempt/pause receipt changed')
    return read(row['path'])


def selection_entries(selection_path):
    selection=read(selection_path);entries=v2.v1.roster()
    entry=next(e for e in entries if e['id']=='seed53_n2_nosplit');previous=Path(entry['run_path']).resolve()
    selected=previous.parent.with_name(previous.parent.name.replace('-attempt1','-attempt2'))/'run.json'
    packet=read(entry['packet_path']);job=next(j for j in packet['jobs'] if j['labels']==[entry['label']])
    if (selection.get('schema') not in {'gsdeploy.asr19_attempt_selection.v1','gsdeploy.user_pause_attempt_selection.v1'}
            or selection.get('target_job_id')!=job['id']
            or selection.get('reason')!='user_requested_pause_retry'
            or ('selected_before_observation' in selection and selection['selected_before_observation'] is not True)
            or ('selection_depends_on_U' in selection and selection['selection_depends_on_U'] is not False)
            or ('selection_policy' in selection and selection['selection_policy']!='user_pause_only_not_utility')
            or not (selection.get('selection_policy')=='user_pause_only_not_utility'
                    or (selection.get('selected_before_observation') is True and selection.get('selection_depends_on_U') is False))
            or Path(selection['previous_run']['path']).resolve()!=previous
            or Path(selection['selected_run']['path']).resolve()!=selected):
        raise ValueError('Only explicit user-paused seed53 nosplit attempt1 -> attempt2 may be selected')
    old=checked(selection['previous_run']);pause=checked(selection['user_pause_receipt'])
    if (old.get('status')!='failed' or old.get('experiment_job_id')!=job['id']
            or not any('KeyboardInterrupt' in str(e) for e in old.get('errors',[]))
            or old.get('measurement_window',{}).get('aborted') is not True
            or old.get('evaluation',{}).get('utility',{}).get('reason')!='WINDOW_ABORTED'
            or old['evaluation']['utility'].get('U') is not None
            or pause.get('raw_evidence_preserved') is not True
            or 'pause' not in pause.get('reason','').lower()
            or not pause.get('interrupted_active_runtime')):
        raise ValueError('Previous attempt is not a documented user-pause WINDOW_ABORTED/KeyboardInterrupt')
    window=old['measurement_window'];when=datetime.datetime.fromisoformat(pause['paused_utc']).timestamp()
    begin=window['start_unix_ns']/1e9;finish=begin+(window['drain_end_time_ns']-window['start_time_ns'])/1e9
    if not begin-5<=when<=finish+10:raise ValueError('Pause timestamp does not match interrupted observation')
    declaration_receipt=selection.get('declaration') or selection.get('declaration_receipt')
    if not declaration_receipt:raise ValueError('Pre-observation resume declaration receipt required')
    declaration=checked(declaration_receipt)
    declared=next((r for r in declaration.get('selected_attempts',[]) if r.get('job_id')==job['id']),None)
    if (declaration.get('schema')!='gsdeploy.user_pause_resume_declaration.v1'
            or declaration.get('stability_rules_changed') is not False
            or declaration.get('user_pause_receipt')!=selection['user_pause_receipt']
            or declaration.get('interrupted_run')!=selection['previous_run']
            or declaration.get('packet')!=receipt(entry['packet_path'])
            or declared is None or declared.get('attempt')!=2
            or Path(declared['output']).resolve()!=selected.parent):
        raise ValueError('Resume declaration does not preselect only this paused condition')
    old_entry=copy.deepcopy(entry);old_entry['id']+='_user_paused_attempt1'
    entry['run_path']=str(selected)
    metadata=dict(selection_receipt=receipt(selection_path),selection=selection,
        previous_attempt=dict(source=receipt(previous),status=old['status'],errors=old['errors'],
            measurement_window=window,utility=old['evaluation']['utility'],job_id=job['id']),
        pause_receipt=selection['user_pause_receipt'],declaration_receipt=declaration_receipt,timestamp_association_verified=True,
        selection_based_on_U=False,only_replaced_condition='seed53_n2_nosplit',other_failures_never_replaced=True)
    return entries,old_entry,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt-selection',required=True,type=Path)
    parser.add_argument('--declaration-only',action='store_true',help='CPU readiness from pre-observation declaration before selection receipt is published')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.declaration_only:
        # In-memory selection contains no resumed measurements; only the old
        # interrupted run and the signed-by-hash preselection are inspected.
        declaration=read(args.attempt_selection)
        target=next(r for r in declaration['selected_attempts'] if r['attempt']==2)
        synthetic=dict(schema='gsdeploy.asr19_attempt_selection.v1',target_job_id=target['job_id'],
            reason='user_requested_pause_retry',selection_policy='user_pause_only_not_utility',
            previous_run=declaration['interrupted_run'],user_pause_receipt=declaration['user_pause_receipt'],
            declaration=receipt(args.attempt_selection),selected_run=dict(path=str(Path(target['output'])/'run.json')))
        # Avoid persisting a fake selection; validate via a temporary CPU-only
        # input, and expose the real declaration as the readiness source.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            selection_path=Path(tmp)/'selection.json';selection_path.write_text(json.dumps(synthetic))
            entries,old_entry,metadata=selection_entries(selection_path)
        metadata['selection_receipt']=None
        metadata['selection_not_yet_published']=True
        args.check_only=True
    else:
        entries,old_entry,metadata=selection_entries(args.attempt_selection.resolve())
    ready=v2.v1.readiness(entries);ready.update(schema='gsdeploy.asr19_replication_readiness.v3',
        attempt_selection=metadata,raw_files_read='Only preserved user-paused attempt1; selected/current raw contents not read',
        invalid_attempt_policy='All failed52 and ineligible runs retained; Q retained when known, U null; no diagnostic substitution')
    if args.output:args.output.mkdir(parents=True,exist_ok=False);write(args.output/'readiness.json',ready)
    if args.check_only or ready['missing_raw_paths']:
        print(json.dumps(ready,ensure_ascii=False,indent=2));raise SystemExit(2 if ready['missing_raw_paths'] else 0)
    if not args.output:parser.error('--output required')
    try:
        selected=metadata['selection']['selected_run']
        if not selected.get('sha256') or receipt(selected['path'])!=selected:raise ValueError('Selected run receipt missing or changed')
        conditions=[];rows=[]
        for entry in entries:
            c,requests=v2.import_preserving_invalid(entry);conditions.append(c)
            for row in requests:row.update(attempt=2 if entry['id']=='seed53_n2_nosplit' else 1,selected_for_condition=True)
            rows.extend(requests)
        paused,paused_rows=v2.import_preserving_invalid(old_entry)
        if paused['U'] is not None or paused['validity']:raise ValueError('Interrupted attempt unexpectedly valid')
        for row in paused_rows:row.update(attempt=1,selected_for_condition=False)
        rows.extend(paused_rows)
        stats=v2.statistics_for_valid_pairs(conditions)
        n3=next(c for c in conditions if c['replicas']==3);controls=[]
        for c in (c for c in conditions if c['seed']==51 and c['replicas']==2):
            if c['cohort']!=n3['cohort']:raise ValueError('n3 cohort mismatch')
            controls.append(dict(reference=c['id'],n3_valid=n3['validity'],reference_valid=c['validity'],
                delta_U_n3_minus_reference=n3['U']-c['U'] if n3['validity'] and c['validity'] else None,
                scope='Single seed51 replica structure comparison; no significant or full-pool inference'))
        report=dict(schema='gsdeploy.asr19_replication_development_audit.v3',status='SEVEN_CONDITIONS_EIGHT_ATTEMPTS_AUDITED',
            readiness=ready,conditions=conditions,excluded_user_paused_attempts=[paused],all_attempt_count=8,
            attempt_selection=metadata,paired=stats,n3_structural_controls=controls,
            invalid_selected_ids=[c['id'] for c in conditions if not c['validity']],
            final_confirmation=False,full_pool_optimality=False,stability_rules_unchanged=True,gpu_started=False)
        write(args.output/'audit.json',report)
        csv_write(args.output/'plans-six-terms.csv',[dict(id=c['id'],seed=c['seed'],asr_replicas=c['replicas'],split=c['split'],
            selected_for_condition=c is not paused,attempt=2 if c['id']=='seed53_n2_nosplit' else 1,validity=c['validity'],
            **{k+'_norm':c['terms'][k] for k in TERMS},U=c['U'],diagnostic_partial_U=c.get('diagnostic_partial_U'),
            diagnostic_is_not_utility=not c['validity']) for c in [*conditions,paused]])
        csv_write(args.output/'request-latencies.csv',rows)
        lines=['# ASR1.9 开发重复审计 v3','',
            '七个条件、八条原始 attempt 全部披露。仅 seed53 不分流因用户暂停后恢复而采用 attempt2；attempt1 的 KeyboardInterrupt、WINDOW_ABORTED、暂停记录 SHA 和时间关联均校验。选择不依据 U，seed52失败/不稳定记录不能替换。', '',
            f"无效已选条件：{report['invalid_selected_ids']}。Q 已知时保留，U 空值不补造；diagnostic_partial_U 不参与比较。",'',
            f"完整有效配对 seed={stats['valid_pair_seeds']}，n={stats['n_valid_pairs']}，均值ΔU={stats['mean_split_minus_nosplit_U']}，95% t区间={stats['student_95_interval']}，双侧符号翻转p={stats['exact_two_sided_sign_flip_p']}。",'',
            '稳定性规则保持原样；全部完成但 unstable 的运行仍不合格。失败/无效条件保留在明细中；只有双方有效才计算效用差。小样本且seed51已用于开发选择，不能据此宣称可靠收益或最终确认。', '',
            f'n3结构对照：{controls}', '',
            '仅支持继续 ASR 或研究 text 的开发决策。不重跑、不改原始数据/画像/门禁、不拟合、不启动 GPU，不声称全池最优。']
        (args.output/'report.zh.md').write_text('\n'.join(lines)+'\n')
        inputs=[ready['original_packet'],*ready['packets'],metadata['selection_receipt'],metadata['pause_receipt'],
            metadata['previous_attempt']['source'],metadata['declaration_receipt'],*[dict(path=c['source']['path'],sha256=c['source']['sha256']) for c in conditions]]
        if any(receipt(r['path'])!=r for r in inputs):raise ValueError('Input bytes changed during audit')
        write(args.output/'manifest.json',dict(inputs=inputs,code=[receipt(__file__),receipt(v2.__file__),receipt(v2.v1.__file__)],
            outputs=[receipt(p) for p in sorted(args.output.iterdir()) if p.is_file()],gpu_started=False))
        print(json.dumps(dict(status=report['status'],invalid_ids=report['invalid_selected_ids'],paired=stats,gpu_started=False)))
    except Exception as exc:
        write(args.output/'failure.json',dict(status='AUDIT_FAILED',error_type=type(exc).__name__,error=str(exc),gpu_started=False));raise SystemExit(3) from exc


if __name__=='__main__':main()
