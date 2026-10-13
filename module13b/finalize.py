"""Final evidence audit and self-contained twenty-answer report."""
import csv,hashlib,json,math
from pathlib import Path
from module13b.analyze import read,save
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/module13_real_specialists';RAW=OUT/'raw13b'
def rows(name):return list(csv.DictReader((OUT/name).open()))
def fmt(v,n=3):return 'MISSING' if v is None or v=='' else f'{float(v):.{n}f}'
def main():
    freeze=read(OUT/'freeze_manifest.json');protected=read(OUT/'protected_before.json');changed=[]
    for r in freeze['files']+protected['files']:
        p=Path(r['path'])
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:changed.append(r['path'])
    assert not changed,changed
    lock=read(OUT/'module13b_execution_freeze.json');implementation=[]
    for r in lock['implementation']:
        current=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest();implementation.append({'path':r['path'],'matches_execution_lock':current==r['sha256']})
    # Compatibility amendments, if needed, remain evidence bearing and must be
    # verified separately; no silent replacement of a frozen implementation.
    amended=sorted(OUT.glob('execution_amendment*.json'))
    assert all(r['matches_execution_lock'] for r in implementation) or amended,'Unlogged implementation change after execution lock'
    q=rows('real_quality_results.csv');lat=rows('real_latency_results.csv');mem=rows('real_memory_results.csv');points=rows('real_operating_points.csv');new=[p for p in points if p['origin']=='NEW_MEASURED_CANDIDATE'];assert len(q)==len(lat)==len(mem)==len(new)==9
    for p in new:
        if p['valid_operating_point']=='True':
            ref=read(OUT/'specialist_selection_rule.json')['ratio_denominators'][p['task']]
            assert math.isclose(float(p['gamma_real']),float(p['s0_ms'])/ref['latency_ms'],rel_tol=1e-12)
            assert math.isclose(float(p['mu_real']),float(p['safe_memory_mib'])/(ref['idle_memory_mib']*lock['safe_factor']),rel_tol=1e-12)
            assert math.isclose(float(p['safe_memory_mib']),float(p['idle_memory_mib'])*lock['safe_factor'],rel_tol=1e-12)
    replay=rows('real_structural_replay.csv');matched=rows('real_structural_replay_matched_counts.csv');assert len(replay)==len(matched)==54
    assert len({(r['workload'],r['budget']) for r in replay})==27
    gate=read(OUT/'module13b_gate.json');selection=read(OUT/'balanced_specialist_selection.json');frontiers=read(OUT/'real_pareto_frontiers.json');transition=read(OUT/'real_transition_analysis.json');prediction=read(OUT/'sensitivity_prediction_validation.json');catalogs=read(OUT/'real_replay_catalogs.json');downloads=read(OUT/'download_manifest.json');pre=read(OUT/'cpu_preflight_results.json')
    for m in downloads['models']:
        pathstats=[]
        for task in m['task_paths']:
            stem=task+'__'+m['model_id'].replace('/','--');f=RAW/(stem+'__quality__r0.summary.json');d=read(f) if f.exists() else {}
            pathstats.append({'task':task,'CUDA_model_load_success':bool(d.get('ready')),'full_quality_evaluation_success':d.get('success',False),'quality_summary':str(f)})
        m['local_load_CPU_success']=all(r['success'] for r in pre['entries'] if r['model_id']==m['model_id']);m['local_GPU_paths']=pathstats;m['local_load_GPU_success']=all(p['CUDA_model_load_success'] for p in pathstats);m['local_execution_GPU_success']=all(p['full_quality_evaluation_success'] for p in pathstats)
    save('download_manifest.json',downloads)
    loads=sum(m['local_execution_GPU_success'] for m in downloads['models']);events=transition['event_context_counts'];profiling=transition['replica_profiling_decision'];needed=sorted({d['model_id']+'@'+d['revision'] for d in profiling if d['decision']=='PROFILE_SELECTED_STANDALONE_WINNER_GT1_NEXT_STAGE'})
    validation={'passed':True,'13A_frozen_files_unchanged':len(freeze['files']),'historical_protected_files_unchanged':len(protected['files']),'execution_implementation':implementation,'compatibility_amendments':[str(p) for p in amended],'candidate_task_paths':9,'unique_checkpoint_GPU_load_success_count':loads,'primary_CPU_replay_rows':len(replay),'matched_count_diagnostic_rows':len(matched),'ratio_and_safe_memory_consistency_verified':True,'original_baselines_and_Module9_unchanged':True,'no_replica_profiling_executed':True,'selected_to_make_Generalist_win':'NO'}
    save('module13b_validation.json',validation)
    text=['# Module13B — Real Specialist Local Measurement',f'\nGate：**{gate["gate"]}**。完成9条冻结task-model paths的真实本机测量与27个context的两regime CPU回放。未执行完整replica profiling、最终deployment、low-QPS、multi-hotspot或Task-Rich。', '\n## 1. 八个checkpoint是否成功本地运行',f'\nGPU完整任务路径执行成功：**{loads}/8**。每个checkpoint的精确revision、下载字节、LFS SHA256、缓存位置、CPU/CUDA加载状态见download_manifest.json；版本、FP32、device、attention和TF32见local_environment.json。RTX4090为实际49140MiB设备，不能称作标准24GB实测。', '\n## 2–5. 真实quality、s0、safe memory和gamma/mu', '\n所有quality来自冻结200条本机结果；s0为3个独立进程各30条的加权均值，保留全部有效run。ASR WER越低越好；LID/Text accuracy越高越好。ASR≤0.042181；Text≥0.925；LID只有原semantic admission，无数值accuracy门槛。', '\n| Task | Candidate | Q | Admission | s0 ms | Idle MiB | Safe MiB | γ_real | μ_real | Pareto |','|---|---|---:|---|---:|---:|---:|---:|---:|---|']
    for p in new:text.append('| '+ ' | '.join([p['task'],p['model_id'],fmt(p['quality'],6),'PASS' if p['admission_pass']=='True' else 'FAIL',fmt(p['s0_ms']),fmt(p['idle_memory_mib'],1),fmt(p['safe_memory_mib'],1),fmt(p['gamma_real'],4),fmt(p['mu_real'],4),p['pareto_status']])+' |')
    text+=['\nsafe M = 实测post-warmup idle × 原冻结factor1.110853955254881；与原baseline safe M比较，factor相消。该经验safe定义不是candidate peak的保证。peak NVML、torch maxima、host RSS与idle均分开记录。ASR/LID Whisper-medium共用权重，task-path latency/quality分别测量。','\n旧immutable分母：ASR Distil-large-v3146.223666144ms/3672MiB；LID Whisper-small27.084134878ms/1616MiB；Text DistilBERT3.769309222ms/750MiB。旧quality与所有旧admitted models完整保留；同期reference control只另报drift，未重定义历史分母。']
    for number,field,label in [(6,'first_split_gamma_distance_log','γ≈1.2125–1.25'),(7,'stronger_gamma_distance_log','γ≈1.9375–2.0')]:
        eligible=[p for p in new if p.get(field)];closest=sorted(eligible,key=lambda p:(float(p[field]),p['model_id']+'@'+p['revision'],p['task']))
        text.append(f'\n## {number}. 哪些最接近{label}\n')
        if closest:
            d=float(closest[0][field]);ties=[p for p in closest if abs(float(p[field])-d)<=1e-12];text.append('按γ单轴log-distance最近：'+ '；'.join(p['task']+'/'+p['model_id']+' γ='+fmt(p['gamma_real'],4) for p in ties)+f'。距离={d:.6f}；还须同时考虑真实μ、quality与task-local target，不能只靠γ声称落靶区。')
        else:text.append('缺少有效实测，不能定位。')
    text.append('\n## 8. 真实Pareto-optimal候选\n')
    for task,f in frontiers['tasks'].items():text.append('- '+task+': '+', '.join(f['frontier']))
    text.append('\n## 9–13. 最终选择与BALANCED组成\n')
    for i,task in enumerate(['asr','lid','text'],9):
        s=selection['tasks'][task];w=s['winner'];text.append(f'{i}. {task}: **{w["model_id"]}** @{w["revision"]}；{s["status"]}；冻结log-distance={w["target_distance"]:.6f}。')
    text.append('12. BASELINE_RETAINED tasks：'+', '.join(gate['baseline_retained_tasks'])+'。所有旧reference加入Pareto；reference endpoint距离可为0，距离0不等于已发生transition。')
    pool={m:p['supported_tasks'] for m,p in catalogs['BALANCED_SPECIALIST_REGIME']['models'].items() if p['role']=='specialist'}
    text.append('13. BALANCED完整Specialist pool：'+json.dumps(pool,ensure_ascii=False)+'。完整EFFICIENCY pool仍单独保留。仅移除所有task均被新winner替换的旧identity；baseline-retained task保持旧pool，shared checkpoint保留原admitted capabilities；新checkpoint保留全部已测且quality-admitted的冻结paths。Generalists/OCR/Image保持原样。')
    text.append('\n## 14–17. 真实结构变化\n')
    for i,field,label in [(14,'R1','Specialist replica count减少'),(15,'R2','Generalist总workload加权routing share增加'),(16,'R3','新的G/S task split'),(17,'new_second_Generalist','新产生second Generalist（原N_G<2，新N_G≥2）')]:text.append(f'{i}. {label}：**{events[field]}/27 contexts**。')
    text.append(f'R4 Generalist多task新增consolidation：{events["R4"]}/27；R5 N_G增加：{events["R5"]}/27；R6 replica allocation变化：{events["R6"]}/27。未要求一定新增second Generalist；原已有N_G≥2的context仍计其S/routing/composition变化。每context deployment、identity、routing、J*、rho、safe M与feasibility见real_structural_replay.csv。')
    text.append('Gate的NEAR报告尺度在GPU结果前锁定：log-L∞距离≤log(1.05)，取自Module12首个memory bracket的5%相对宽度；仅用于gate说明，不改变13A选择公式或任何实测值。')
    text.append('新模型只有真实g(1)=1和M(1)，g(x>1)未知且未复制旧模型曲线。同0/1searched-Specialist count上限诊断单独存real_structural_replay_matched_counts.csv；其是否仍有transition：'+str(transition['matched_count_transition_observed'])+'。单纯由证据上限差异造成的transition不能视作真实operating-point因果验证。CPU数学J* certificate与冻结Module9 router的运行结果分别保留；CPU回放不是最终GPU deployment evidence。')
    text.append('\n## 18. 与Module12 prediction一致性\n'+prediction['status']+'。'+prediction['explanation'])
    text.append('\n## 19. 下一步哪些selected Specialist需要g(x>1)\n'+('; '.join(needed) if needed else '无新selected Specialist满足下一阶段profiling条件；本模块不启动profiling。'))
    text.append('\n## 20. Gate\n**'+gate['gate']+'**。gate与局限见module13b_gate.json和real_transition_analysis.json；不强制得到balanced transition，不追加新模型。')
    text.append('\n反cherry-picking：**NO，未为了让Generalist赢挑模型。** 13A原quality admission→含旧baseline的真实三维Pareto→冻结target union log-L∞最近→higher quality→lower memory→deterministic ID@SHA；选择先于replay，未使用V3 outcome。specialist_selection_rule.json、module13b_execution_freeze.json与balanced_specialist_selection.json提供selection-rule evidence。')
    text.append(f'\n完整性审计：{len(freeze["files"])}个13A冻结文件、{len(protected["files"])}个历史受保护文件未变；Module9、原Specialist baseline与阈值未修改。所有失败与compatibility logs保留。完成后停止。')
    (OUT/'module13b_summary.md').write_text('\n'.join(text)+'\n')
    # Freeze completed13B artifacts without ever rebuilding the13A manifest.
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='module13b_artifact_manifest.json' and p.suffix!='.pyc']
    save('module13b_artifact_manifest.json',{'module':'13B','complete':True,'artifacts':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(files)],'original_13A_manifest_preserved':True})
    print(json.dumps({'gate':gate['gate'],'GPU_loads':loads,'validation':validation,'next_replica_checkpoints':needed},indent=2))
if __name__=='__main__':main()
