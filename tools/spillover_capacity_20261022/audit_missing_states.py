#!/usr/bin/env python3
"""Read-only CPU inventory of nine missing standalone path/replica states.

Writes new documents only. A historical failed protocol never becomes a global
low-rate exclusion. Full input coverage is separate from valid profile status.
"""
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess

ROOT=Path(__file__).resolve().parents[2]
REQUIREMENTS=ROOT/'docs/seven_pool_recalibration_requirements_20261022.json'
BASE=ROOT/'artifacts/retained-replica-profiles-20261021'
OUTPUT=ROOT/'docs/seven_pool_missing_state_evidence_audit_20261022.json'
MARKDOWN=OUTPUT.with_suffix('.md')


def sha_bytes(payload):return hashlib.sha256(payload).hexdigest()


def receipt(path):
    path=Path(path).resolve();before=path.stat();payload=path.read_bytes();after=path.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Input changed '+str(path))
    return dict(path=str(path),sha256=sha_bytes(payload),bytes=len(payload))


def load(path,receipts):
    r=receipt(path);receipts[str(Path(path).resolve())]=r
    return json.loads(Path(path).read_text())


def raw_rows(path,receipts):
    r=receipt(path);receipts[r['path']]=r
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def valid(row,task):
    value=row.get('service_ms')
    return (row.get('status')=='RESULT' and row.get('success') is True and row.get('task')==task and
        type(value) in (int,float) and math.isfinite(value) and value>0 and bool(row.get('sample_id')) and
        bool(row.get('request_id')) and bool(row.get('request_sha256')) and bool(row.get('protocol_sha256')))


def main():
    if OUTPUT.exists() or MARKDOWN.exists():raise FileExistsError('New evidence documents only')
    sources={};requirements=load(REQUIREMENTS,sources);config=load(requirements['source_config']['path'],sources)
    if sources[requirements['source_config']['path']]['sha256']!=requirements['source_config']['sha256']:raise ValueError('Requirements config SHA differs')
    summary=load(BASE/'combined-summary.json',sources);releases={};states={}
    for package in [BASE,BASE/'protocol-binding-correction-v2']:
        releases[str(package)]=load(package/'release.json',sources);states[str(package)]=load(package/'state.json',sources)
        if states[str(package)]['status']!='COMPLETE_RESOURCE_GRID':raise ValueError('Historical profiling package not terminal')
    needed={(r['model'],r['task'],r['replicas']) for r in requirements['missing_path_states']}
    filenames=subprocess.run(['rg','--files','-uu','artifacts'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.splitlines()
    excluded=['static-spillover-study','confirmation','qa-development','supervision','validation','holdout']
    profiles=[];matches=defaultdict(list)
    # Metadata inventory only. Ongoing root experiments and reserved validations
    # are excluded before opening anything; exact raw opened only from frozen
    # historical terminal resource packages below.
    for filename in filenames:
        if not filename.endswith('/profile.json') or any(term in filename for term in excluded):continue
        path=ROOT/filename
        try:data=load(path,sources)
        except (OSError,json.JSONDecodeError):continue
        profiles.append(sources[str(path.resolve())]);model=data.get('model') or {};model=model if isinstance(model,dict) else {}
        family=model.get('candidate',{}).get('slot');n=data.get('replicas')
        if not family or not isinstance(n,int):continue
        tasks={r.get('task') for r in data.get('rounds',[]) if isinstance(r,dict)}
        if not tasks:tasks={t for t in ['text','image','ocr','asr'] if '-'+t in str(path.parent)}
        for task in tasks:
            if (family,task,n) in needed:matches[(family,task,n)].append(path)
    results=[]
    for missing in requirements['missing_path_states']:
        family,task,n=missing['model'],missing['task'],missing['replicas'];key=(family,task,n)
        catalog=config['catalog'];expected=list(catalog['quality']['input_order'][task]);entry_id=family+'-'+task
        item=dict(model=family,task=task,replicas=n,expected_inputs=len(expected),domain_status='UNKNOWN_NOT_INFEASIBLE',
            requirements_original_status=missing['status'],complete_valid_native_baseline_found=False,
            global_or_low_rate_infeasibility_proved=False,source_receipts=[],profile_metadata_matches=[str(p) for p in matches[key]],
            physical_state_excluded=False,U=None)
        stop=summary['stop_receipts'].get(entry_id);item['profiling_stop_receipt']=stop
        if not matches[key]:
            item.update(conclusion='UNKNOWN_UNTESTED',formal_returned=0,formal_valid=0,formal_unique_inputs=0,
                coverage_fraction=0,missing_input_ids=expected,complete_input_service_candidate=False,
                reason='No matching terminal standalone profile in scoped423-profile metadata inventory; old controller stopped before this state.',
                failure_scope='NONE_FOR_THIS_EXACT_N_TASK',next_action='Retain state in original domain. New state-specific evidence or a mathematically applicable exclusion is needed; do not inherit plateau/failure as global infeasibility.')
            if stop and stop['reason']=='TWO_INCREMENT_THROUGHPUT_PLATEAU':
                item['reason']='Historical resource controller stopped at throughput plateau; this exact n/task was never run. Plateau is neither native failure nor infeasibility proof.'
            elif stop and stop['reason']=='MEASURED_FAILURE':
                item['reason']='Historical controller stopped after a smaller replica count failed. No execution for this exact n/task and no universal monotonic resource proof.'
            item['source_receipts']=[sources[str((BASE/'combined-summary.json').resolve())],sources[str((BASE/'state.json').resolve())],sources[str((BASE/'release.json').resolve())]]
            results.append(item);continue
        if len(matches[key])!=1:raise ValueError('Multiple exact-state profiles require explicit review')
        path=matches[key][0];profile=load(path,sources)
        if profile.get('status') not in {'FAILED','COMPLETE','INTERRUPTED'}:raise ValueError('Profile is not terminal')
        package=BASE/'protocol-binding-correction-v2' if path.is_relative_to(BASE/'protocol-binding-correction-v2') else BASE
        release=releases[str(package)];job_entry=next(e for e in release['entries'] if e['id']==entry_id)
        job_path=Path(job_entry['folder'])/'job.json';job=load(job_path,sources);spec_path=Path(job['spec']);spec=load(spec_path,sources)
        cohort=load(BASE/'cohorts'/f'{task}.json',sources)
        if {r['id'] for r in cohort}!=set(expected):raise ValueError('Historical/target input sets differ')
        if profile['replicas']!=n or profile['model']['candidate']['slot']!=family or profile['model']!=spec:raise ValueError('Profile/spec/state identity differs')
        if profile['manifest_sha256']!=sha_bytes(json.dumps(job['samples'],sort_keys=True).encode()):raise ValueError('Historical manifest differs')
        raws=[];rounds=[];paths=[path,job_path,spec_path,BASE/'cohorts'/f'{task}.json',package/'release.json',package/'state.json',BASE/'combined-summary.json']
        for raw_path in sorted(path.parent.glob(task+'-round*.jsonl')):
            rows=raw_rows(raw_path,sources);raws.extend(rows);paths.append(raw_path)
            completed=[r for r in rows if valid(r,task)]
            rounds.append(dict(path=str(raw_path),sha256=sources[str(raw_path.resolve())]['sha256'],returned=len(rows),valid=len(completed),
                unique_inputs=len({r['sample_id'] for r in completed}),missing_input_ids=sorted(set(expected)-{r['sample_id'] for r in completed}),
                native_status_counts=dict(Counter(r.get('status') for r in rows))))
        formal=[r for r in raws if valid(r,task)];counts=Counter(r['sample_id'] for r in formal)
        if not set(counts)<=set(expected):raise ValueError('Unexpected historical input')
        warm_path=path.parent/'warmup.json';warm=load(warm_path,sources);paths.append(warm_path)
        startup_path=path.parent/'startup.json';startup=load(startup_path,sources);paths.append(startup_path)
        telemetry_path=path.parent/'telemetry.json';telemetry=load(telemetry_path,sources);paths.append(telemetry_path)
        peak=max(row['used_mib'] for row in telemetry);factor=release['safe_factor'];budget=release['budget_mib']
        if not math.isclose(peak,profile['peak_board_used_mib'],abs_tol=1e-6):raise ValueError('Profile peak differs from raw telemetry')
        reference_path=path.parent.parent/(entry_id+'-n1')/'profile.json';reference=load(reference_path,sources);paths.append(reference_path)
        base=[]
        for rp in sorted(reference_path.parent.glob(task+'-round*.jsonl')):base.extend(raw_rows(rp,sources));paths.append(rp)
        index=defaultdict(list)
        for row in base:index[row.get('sample_id')].append(row)
        matches_request=sum(any(r.get('request_sha256')==x.get('request_sha256') and r.get('protocol_sha256')==x.get('protocol_sha256') and r.get('effective_generation')==x.get('effective_generation') for x in index[r['sample_id']]) for r in formal)
        worker_current=ROOT/'artifacts/static-study-v6-20261021/inputs/workers'/family/'worker.json'
        current_spec=load(worker_current,sources);paths.append(worker_current)
        existing_sha=summary.get('artifact_sha256',{})
        perinput=[]
        for pool_id in expected:
            seq=[r for r in formal if r['sample_id']==pool_id]
            perinput.append(dict(pool_id=pool_id,observations=len(seq),replicas=sorted({r['replica'] for r in seq}),
                service_ms=[r['service_ms'] for r in seq],observed_median_ms=median(r['service_ms'] for r in seq) if seq else None,
                raw_request_ids=[r['request_id'] for r in seq],request_sha256=sorted({r['request_sha256'] for r in seq}),
                native_protocol_sha256=sorted({r['protocol_sha256'] for r in seq})))
        full=set(counts)==set(expected)
        item.update(conclusion='FULL_INPUT_SERVICE_CANDIDATE_FROM_FAILED_RUN' if full else 'UNKNOWN_INCOMPLETE_INPUT_SERVICE',
            complete_input_service_candidate=full,profile_status=profile['status'],failure_stage=profile.get('failure_stage'),error=profile.get('error'),
            failure_detail=profile.get('failure_detail'),workload=profile['workload'],native_rounds=rounds,
            formal_returned=len(raws),formal_valid=len(formal),formal_unique_inputs=len(counts),coverage_fraction=len(counts)/len(expected),
            missing_input_ids=sorted(set(expected)-set(counts)),per_input_observation_count_histogram=dict(Counter(counts.values())),
            per_replica_unique_inputs={str(i):len({r['sample_id'] for r in formal if r['replica']==i}) for i in range(n)},
            per_input=perinput,completed_profile_rounds=len(profile['rounds']),
            warmup=dict(returned=len(warm),success=sum(r.get('success') is True for r in warm),unique_inputs=len({r.get('sample_id') for r in warm}),
                failures=[{k:r.get(k) for k in ['request_id','sample_id','status','success','error']} for r in warm if r.get('success') is not True]),
            startup_ready=sum(r.get('status')=='READY' for r in startup),
            resource_evidence=dict(telemetry_peak_board_mib=peak,safe_factor=factor,recomputed_safe_peak_mib=peak*factor,budget_mib=budget,
                observed_safe_peak_over_budget=peak*factor>budget,reported_safe_memory_mib=profile.get('safe_memory_mib'),
                reported_safe_memory_scope='Warm-residency screen only, not measurement peak; recomputed peak shown separately.'),
            identity=dict(historical_gpu_uuid=profile.get('gpu_uuid'),target_gpu_uuid=catalog['measurement_context']['gpu_uuid'],
                exact_gpu_uuid_match=profile.get('gpu_uuid')==catalog['measurement_context']['gpu_uuid'],hardware=profile.get('hardware'),
                historical_model_spec_equals_n1=profile['model']==reference['model'],
                observed_request_protocol_and_generation_match_n1=matches_request,total_formal_requests_checked=len(formal),
                current_worker_candidate_equals_profile=current_spec['candidate']==profile['model']['candidate'],
                current_worker_execution_equals_profile=current_spec['execution']==profile['model']['execution'],
                input_ids_equal_target_quality_pool=True,profile_manifest_matches_declared_samples=True,
                full_current_checkpoint_weights_rehashed=False,source_semantic_equivalence_to_current_runtime_not_certified=True),
            failure_scope='Historical saturated standalone n/task measurement, not arbitrary low-rate routing or full-domain exclusion.',
            next_action='Retain UNKNOWN. Preserve partial diagnostic measurements; new evidence must address the exact state and intended activity/identity context, without recycling failed data as a valid resource profile.')
        if full:
            item['reuse_scope']='Complete32-input observations may populate an explicitly unvalidated, failed-context diagnostic input-service inventory. Does not restore legal memory,3 complete rounds,all-input-per-alias coverage,current-device equivalence,open-loop capacity,or mixed-model service.'
            item['next_action']='CPU-audit candidate identity and preserve per-input medians/coverage. Domain remains UNKNOWN; do not activate as validated baseline or justify rho/resource pruning without compatible legal-context evidence.'
        if family=='generalist_1' and task=='image':
            item['failure_scope']='Exact historical sequential load-and-prime protocol failed at warmup-3-0 on largest image with four loaded replicas and one active inference. Not merely saturated-measurement OOM; no global low-rate/new-device proof.'
        for source in paths:
            source=str(source.resolve());r=sources[source];entry=dict(r)
            if source in existing_sha:entry['combined_summary_sha_match']=existing_sha[source]==r['sha256']
            if source in release['bindings']:entry['release_binding_sha_match']=release['bindings'][source]==r['sha256']
            item['source_receipts'].append(entry)
        results.append(item)
    assert len(results)==9 and sum(r['complete_valid_native_baseline_found'] for r in results)==0
    for path,r in sources.items():
        if receipt(path)['sha256']!=r['sha256']:raise ValueError('Audit source changed')
    report=dict(schema='gsdeploy.seven_pool_missing_state_evidence_audit.v1',status='NINE_STATES_REMAIN_UNKNOWN_ONE_PARTIAL_FULL_INPUT_CANDIDATE',
        requirements=sources[str(REQUIREMENTS)],source_config=sources[requirements['source_config']['path']],
        gpu_started=False,existing_evidence_modified=False,production_math_modified=False,U=None,
        scope='Existing standalone native profiling metadata and four exact failed states; no current running GPU output, reserved validation/final result or new experiment read.',
        inventory=dict(profile_metadata_files_scanned=len(profiles),exact_state_matches=sum(len(v) for v in matches.values()),
            excluded_path_terms=excluded,profile_receipts=profiles,absence_scope='Scoped native resource-profile inventory; no claim that unindexed arbitrary files prove universal absence.'),
        counts=dict(missing_states=9,full_input_partial_failed_candidates=sum(r['complete_input_service_candidate'] for r in results),
            valid_complete_baselines=0,globally_proved_infeasible_states=0,untested=sum(r['conclusion']=='UNKNOWN_UNTESTED' for r in results)),
        items=results,limitations=['Observed high-concurrency failure is not universal low-rate infeasibility.',
            'Sequential priming OOM is scoped to its exact startup/warmup and historical hardware.',
            'Plateau is a profiling stop heuristic, not exclusion proof.',
            'Full input pool union does not imply every replica/input observed or complete safe profile.',
            'Historical GPU UUID differs from current target; no weight rehash or current-runtime semantic certification performed.',
            'Missing states remain in seven-model domain and block full-domain certificate until resolved.'],
        auditor=receipt(__file__),sources=list(sources.values()))
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 七池9个缺失状态的已有证据审计（2026-10-22）','',
        '只读CPU审计，未启动GPU、未改既有证据/画像/评分。9状态均保持UNKNOWN；0条完整合格baseline，0条全域不可行证明。唯一可补充的完整输入诊断候选来自G1/text/n3失败运行；它不解除资源及迁移门禁。','',
        f"扫描{len(profiles)}份非当前GPU/非保留验证的原生资源profile元数据，精确匹配4个终态FAILED；其余5状态未执行。逐项SHA、覆盖、原生请求身份和完整输入列表见同名JSON。",'',
        '|模型/任务/n|正式成功/返回|有效输入覆盖|既有记录/停止原因|结论|', '|---|---:|---:|---|---|']
    for r in results:
        detail=r.get('error') or (r.get('profiling_stop_receipt') or {}).get('reason','无精确状态记录')
        label='可补充32输入失败上下文诊断；全域仍UNKNOWN' if r['complete_input_service_candidate'] else 'UNKNOWN，不排除'
        lines.append(f"|{r['model']}/{r['task']}/{r['replicas']}|{r['formal_valid']}/{r['formal_returned']}|{r['formal_unique_inputs']}/{r['expected_inputs']}|{detail}|{label}|")
    lines.extend(['','## G1/text/n3：覆盖能补，合格baseline不能直接补','',
        '首轮44/44成功，32/32输入全覆盖；第二轮中断前7/7成功，合计51条。各输入仅1–3次，各alias分别覆盖17、13、16个输入，不是每alias完整32输入×3轮。51/51请求与既有n1相同input的request SHA、native protocol SHA和effective generation一致，模型spec也一致。可保留逐输入观测/中位数作为明确标记failed-context的未验证诊断清单，不能重新登记为合格资源画像。', '',
        '测量峰值47245.6875MiB，乘safe_factor后52483.060942MiB>49140。profile里的safe_memory_mib=32421.037702是预热后驻留值，不能拿它遮盖测量峰值超限。全局状态继续UNKNOWN：当前GPU/运行语义迁移及低频合法路径均未证明。','',
        '## 其他失败及未测状态的范围','',
        'G1/image/n4：0/64正式输入；startup4个READY，前3个进程共9个串行warmup成功，第四进程第一个最大图像warmup-3-0 OOM。冻结runner是每进程load后逐个串行warmup，因此不是单纯测量期四路同时请求失败。该启动/priming/驻留上下文是精确失败证据；但历史设备UUID不同，且它不是对所有初始化、低频静态路由或新设备的通用不可行证明。', '',
        'G1/ocr/n4及G3/ocr/n4：各仅1/32正式输入；safe peak分别49858.529510、53293.290078MiB，均超49140。最多保留单输入服务诊断，不能补全池服务或把饱和失败推广到所有低频。', '',
        'G1/text/n4：因n3失败而未测；不能没有单调性证明就继承n3失败。G2/text/n4、n5、image/n5、ocr/n5：因吞吐平台启发式停止而未测；平台既不是OOM也不是不可行证明。来自G2/asr/n5或同族其他task的成功/失败都不能补这些具体n/task的服务值。','',
        '## 身份与下一动作','',
        '4个失败profile均来自GPU-533a287b-0e15-a7c8-8e58-2b21bdcbb13a；当前要求设备为GPU-8a262fe0-598c-46b7-c953-41d9da84a47e。相同型号/显存不等于精确设备身份。还需明确当前worker/source/检查点/任务协议的适用性，不能仅凭一个protocol_sha字段就断言等价或不等价。此审计未重哈希全模型权重，未认证新共驻或开放负载服务。', '',
        '优先在CPU保存G1/text/n3的32输入观测与身份链，作为可复用诊断候选；其余8状态无完整目标输入服务证据。若需解除全域证书缺口，应另行声明特定n/task与目标活动上下文的合法测量或可证明排除，不修改现有61条画像，不静默删状态，不直接把4个历史FAILED纳入不可行剪枝。本文没有安排新GPU。', '',
        '机器报告：'+str(OUTPUT)+f"；SHA256={receipt(OUTPUT)['sha256']}。所有读取source路径和SHA在JSON逐项保留，输出auditor脚本也有SHA。"])
    MARKDOWN.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'],json=str(OUTPUT),markdown=str(MARKDOWN),counts=report['counts'],U=None)))


if __name__=='__main__':main()
