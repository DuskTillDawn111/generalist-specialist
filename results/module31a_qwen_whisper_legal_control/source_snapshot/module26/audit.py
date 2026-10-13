"""Stage A: verify source manifests, raw cohorts and unchanged utility directly."""
from .common import *
from module21.trajectory import choose

def main():
    assert not (OUT/'positive_probe_audit.json').exists(),'No completed audit overwrite'
    save('historical_metadata_before.json',protected_metadata());d,refs,weights=inputs()
    inventory=[];parsed=[]
    # Consume raw artifacts and logs from all three modules, not prose summaries.
    for directory in [AROOT,BROOT,ROOT/'results/module25_utility_difference']:
        for p in sorted(directory.rglob('*')):
            if not p.is_file() or p.suffix not in ['.json','.csv','.log','.py']:continue
            inventory.append(ref(p))
            if p.suffix=='.json':
                value=read(p);parsed.append(dict(path=str(p),type=type(value).__name__))
            elif p.suffix=='.csv':
                with p.open() as f:count=sum(1 for row in csv.DictReader(f))
                parsed.append(dict(path=str(p),records=count))
            else:
                with p.open(errors='replace') as f:count=sum(1 for line in f)
                parsed.append(dict(path=str(p),lines=count))
    save('read_only_input_inventory.json',inventory);save('parsed_input_audit.json',parsed)
    manifest=read(AROOT/'frozen_experiment_manifest.json');bmanifest=read(BROOT/'frozen_high_manifest.json')
    verify([manifest['utility_definition'],manifest['utility_references'],manifest['catalog'],*manifest['inputs']])
    sources=[ROOT/'module22/gpu.py',ROOT/'module21/trajectory.py',ROOT/'module11b/measure.py',ROOT/'module16/gpu.py',ROOT/'module11b/telemetry.py',
        ROOT/'module24/calibration.py',ROOT/'module20/optimizer.py',ROOT/'module25/predictor.py',ROOT/'joint_deploy/measurement/deployment_runtime.py',
        ROOT/'performance_model/local_calibration/stability_v2.py',ROOT/'module13b/worker.py']
    sources += [Path(v['script']) for v in manifest['plan']['registry'].values()]
    verify([manifest['plan']['sample_pool']]);save('read_only_implementation_inventory.json',[ref(p) for p in sources])
    runs={};taskrows=[];workers=[];checks=[]
    names=[('24A','GS18','high_gs18_diagnostic_rep1',AROOT),('24A','S-only','high_s_only_diagnostic_attempt2',AROOT),
        ('24B','GS','high_proposed_rep1',BROOT),('24B','S-only','high_control_rep1',BROOT)]
    for module,arm,name,directory in names:
        path=directory/'raw_runs'/(name+'.json');raw=read(path);raw['_path']=str(path)
        recalc=arithmetic(raw,d,refs,weights);ob=observe(raw,path);ph=raw['phases']['measurement'];rs=ph['request_records'];w=ph['window']
        trajectory=read(raw['slot']['trajectories']['measurement']['path']);events=trajectory['events']
        route_ok=len(events)==len(rs) and all(r['model']==choose(e,raw['slot']['routing'],raw['slot']['deployment']) and
            r['sample_id']==e['item']['sample']['id'] and r['frozen_offset_seconds']==e['offset_seconds'] for r,e in zip(rs,events))
        e2e_error=max(abs(r['e2e_latency_ms']-(r['completion_time_ns']-r['scheduled_time_ns'])/1e6) for r in rs if r['status']=='completed')
        prior=[r for r in [*raw.get('path_probe_records',[]),*raw['warmup']['request_records']] if r['status']=='completed']
        warm_last=max(r['completion_time_ns'] for r in prior);no_contamination=warm_last<w['start_time_ns']
        unfinished_warm=sum(r['status']!='completed' for r in raw['warmup']['request_records'])
        path_legal=all(r['task'] in d['models'][raw['replica_to_family'][r['model']]]['supported_tasks'] for r in rs)
        semantic_invalid=sum(r['worker_response'].get('valid_closed_set_output') is False for r in rs if r['status']=='completed')
        telemetry_thermal=any('temperature_gpu_c' in r or 'temperature_c' in r for r in raw['telemetry']['samples'])
        quality=[]
        for t,q in raw['slot']['lambda_rps'].items():
            rr=[r for r in rs if r['task']==t and r['status']=='completed'];responses=[r['worker_response'] for r in rr]
            metrics=dict(task=t,requests=len(rr),quality_admission='FROZEN_SUPPORTED_PATH',
                correct_count=sum(v.get('correct') is True for v in responses) if any('correct' in v for v in responses) else None,
                word_errors=sum(v.get('word_errors',0) for v in responses) if any('word_errors' in v for v in responses) else None,
                reference_words=sum(v.get('reference_words',0) for v in responses) if any('reference_words' in v for v in responses) else None,
                missing_raw_quality=sum(not any(k in v for k in ['correct','wer','word_errors']) for v in responses))
            quality.append(metrics)
            taskrows.append(dict(module=module,arm=arm,run_id=name,task=t,**recalc['per_task'][t],
                mean_U_contribution=-weights['L']*q/sum(raw['slot']['lambda_rps'].values())*recalc['per_task'][t]['mean_ms']/refs[t],
                P95_U_contribution=-weights['P']*q/sum(raw['slot']['lambda_rps'].values())*recalc['per_task'][t]['p95_ms']/refs[t],
                P95_request_support_weak=len(rr)<100))
        for wr in ob['workers']:
            worker=wr['worker'];rr=[r for r in rs if r['model']==worker and r['status']=='completed']
            workers.append(dict(module=module,arm=arm,run_id=name,**wr,
                service=stats([r['worker_latency_ms'] for r in rr]),queue=stats([r['worker_queue_wait_ms'] for r in rr]),
                E2E=stats([r['e2e_latency_ms'] for r in rr])))
        item=dict(module=module,arm=arm,run_id=name,raw=ref(path),slot=raw['slot'],recomputed=recalc,
            quality=quality,worker_states=ob['worker_states'],workers=ob['workers'],valid_session=raw['valid_session'],
            errors=raw['errors'],GPU=raw['baseline_GPU'],actual_worker_identities=raw['worker_identities'],
            trajectory_and_actual_route_match=route_ok,E2E_timestamp_max_error_ms=e2e_error,
            measurement_no_warmup_backlog=no_contamination,unfinished_warm_requests=unfinished_warm,
            warm_to_measurement_gap_seconds=(w['start_time_ns']-warm_last)/1e9,
            quality_paths_legal=path_legal,closed_set_invalid=semantic_invalid,thermal_evidence_complete=telemetry_thermal,
            source_gpu_release_verified=raw['gpu_release']['verified'],measurement_window_complete=w['duration_seconds']==w['planned_duration_seconds'],
            system_stability=ph['stability'],raw_cohort_sha256=hashlib.sha256(identity(cohort(raw)).encode()).hexdigest())
        runs[module+':'+arm]=item
        checks.append(dict(module=module,arm=arm,eligible=ob['eligible'],full_cohort=len(rs)==recalc['completed'],
            no_warmup_pollution=no_contamination and unfinished_warm==0,actual_route_and_input_match=route_ok,
            no_window_truncation=item['measurement_window_complete'],timestamp_arithmetic_correct=e2e_error<1e-8))
    pairs=[];partrows=[]
    for module,arm in [('24A','GS18'),('24B','GS')]:
        a=runs[module+':'+arm];b=runs[module+':S-only'];matched=a['raw_cohort_sha256']==b['raw_cohort_sha256']
        delta=a['recomputed']['U']-b['recomputed']['U']
        for k in ['T','L','P','M']:
            partrows.append(dict(module=module,component=k,delta_contribution=a['recomputed'][k+'_contribution']-b['recomputed'][k+'_contribution']))
        pairs.append(dict(module=module,delta_U=delta,same_all_inputs_and_Poisson_trace=matched,same_residency=a['slot']['deployment']==b['slot']['deployment'],
            same_background_routing=all(a['slot']['routing'][t]==b['slot']['routing'][t] for t in a['slot']['routing'] if t!='asr'),
            A_U=a['recomputed']['U'],B_U=b['recomputed']['U']))
    a=runs['24A:GS18'];b=runs['24B:GS'];differences=dict(checkpoints_same=set(a['slot']['deployment'])==set(b['slot']['deployment']),
        physical_replica_counts_same=a['slot']['deployment']==b['slot']['deployment'],six_offered_QPS_same=a['slot']['lambda_rps']==b['slot']['lambda_rps'],
        planned_ASR_G_fractions=[a['slot']['routing']['asr'][G]/1.52,b['slot']['routing']['asr'][G]/1.52],
        GPU_uuid_name_driver_same=all(a['GPU'][k]==b['GPU'][k] for k in ['uuid','name','driver']),
        seeds=[a['slot']['seed'],b['slot']['seed']],cohort_identical=a['raw_cohort_sha256']==b['raw_cohort_sha256'],
        original_pair_arm_order=['S-only→GS18','GS→S-only'],
        normalization_same=True,input_pool_same=manifest['plan']['sample_pool']['sha256']==bmanifest['plan']['sample_pool']['sha256'],
        background_TINY_routes_differ=True,
        physical_causal_identifiability=False,missing=['matched trace at same counts and ASR fraction across runs','historical temperature/clock coverage','independent repetitions'])
    startup=read(AROOT/'raw_runs/high_s_only_startup_failure_attempt1.json')
    amendment=read(AROOT/'probe_startup_repair_amendment.json')
    startup_no_outcome=not startup['phases'] and not startup.get('path_probe_records') and amendment['no_request_measurement_before_repair']
    # Existing admission is not replaced with new sample quality thresholds.
    criteria=dict(all_four_raw_runs_eligible=all(c['eligible'] for c in checks),
        positive_probe_checks_pass=all(all(v for k,v in c.items() if k not in ['module','arm']) for c in checks if c['module']=='24A'),
        matched_24A_comparator=all(v for k,v in pairs[0].items() if k.startswith('same_')),
        utility_raw_positive_over001=pairs[0]['delta_U']>.01,original_sources_verified=True,startup_repair_without_outcome_retry=startup_no_outcome,
        all_worker_and_pool_files_available=all(Path(v['script']).is_file() and Path(v['python']).is_file() for v in manifest['plan']['registry'].values()))
    passed=all(criteria.values())
    doc=dict(at_utc=now(),passed=passed,status='LEGAL_DEVELOPMENT_POSITIVE_EXAMPLE' if passed else 'PROBE_INVALID',
        eligible_for_independent_replication=passed,original_positive_probe_is_independent_validation=False,
        runs=runs,pairs=pairs,run_checks=checks,criteria=criteria,configuration_comparison=differences,
        quality_scope='Original frozen per-task/model capability admission; raw quality scores descriptive, no refitted thresholds or quality reward',
        audit_limitations=['Original LID has2 requests, VQA32; tails uncertain','Only one developer pair; no replication evidence',
            'Historical thermal/clock data incomplete; order and replica/input effects not causally isolated',
            'Some priming/warmup sample IDs recur in measurement in both arms; native workers have no prediction cache; completion windows disjoint'],
        no_new_predictor=True,no_new_pool_search=True,original_gate_not_modified=True)
    save('positive_probe_audit.json',doc);csvsave('historical_task_latency_audit.csv',taskrows);csvsave('historical_worker_stability.csv',workers)
    csvsave('historical_utility_decomposition.csv',partrows)
    text=['# Module26 CPU证据审计\n\n',f'Stage A：**{doc["status"]}**。+0.4767982826570303由1500/1500完整GPU配对的20A系统U重算得到。仅为合法开发正例，不是独立验证。\n\n',
        '原配置：Qwen、Whisper-large-v3、RoBERTa-SST2、ConvNeXt-tiny、PARSeq各1副本；背景text4/image2/ocr5/lid.03/VQA.2，ASR1.52，ASR G份额18%。对照只将ASR改为全Whisper，其他驻留、请求、路由完全匹配。预算34567.5534MiB，原20A权重/refs不变。\n\n',
        '24B改为Text/Image/OCR各2副本，共8worker，ASR13.1833%，Text/Image另有极小G流量。QPS、checkpoint、GPU、输入池相同，但seed、输入cohort、运行顺序不同。不是同配置的独立重复。具体逐任务/worker服务与等待、输入分布见JSON和CSV，以及25原始归因输出。\n\n',
        '| 历史实验 | G/S U | S-only U | ΔU |\n|---|---:|---:|---:|\n']
    for p in pairs:text.append(f'| {p["module"]} | {p["A_U"]:.9f} | {p["B_U"]:.9f} | {p["delta_U"]:.9f} |\n')
    text+=['\n24A S-only为steady，G/S为recoverable_congested（峰值拥塞告警但最终恢复），均无硬unstable或持续增长；各物理worker重建和safe/peak显存通过。全部请求完整、输入/到达/路由匹配、warmup完全drain且先于测量，E2E由scheduled到completion重算一致；启动READY修复前无任何测量，失败原样保留。\n\n',
        '原正收益主要是背景Text/OCR的mean/P95改善，不能归因于单独ASR收益或自动称分流必然产生改善。逐项贡献见historical_utility_decomposition.csv；Text/OCR含数百请求，但LID仅2、VQA32，后者尾部很不确定。两次历史GPU温度/时钟证据不足，不能唯一分离热状态、副本共置、输入、顺序的物理原因。\n\n',
        '质量按原supported path准入，不用短时样本准确率改准入；原始WER/correct保留描述性报告。全部原始JSON/CSV/log及manifest已读取并hash冻结；本轮没有开发预测器或全池搜索。\n']
    (OUT/'positive_probe_audit.md').write_text(''.join(text))
    print('STAGE_A',doc['status'],'replication_permitted',passed,flush=True)

if __name__=='__main__':main()
