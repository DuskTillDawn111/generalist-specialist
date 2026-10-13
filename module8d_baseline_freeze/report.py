"""Freeze campaign costs, fairness evidence and an operational reuse contract."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import statistics

from deployment_search.runtime_cache import digest, file_hash
from .definitions import ROOT, B, C, OUT, TASKS, read, ref, freeze_file, csvwrite, verify_freeze


def summarize(definitions,states,audit):
    groups=audit['state_groups']; reused=audit['measurement_selection'];missing=audit['missing_known_measurements']
    cfg=definitions['B4']['search_config']; seeds=definitions['B4']['random_seeds']
    names=list(definitions['measurement_protocol']['runtime_context_by_workload'])
    baseline_stats={}
    for baseline in ('B0','B1','B2','B3'):
        rows=[g for g in groups if g['baseline']==baseline]
        reused_runs=sum(g['search_reused_runs']+g['confirmation_reused_runs'] for g in rows)
        baseline_stats[baseline]={'workload_states':6,
            'fully_reusable_states':sum(g['search_state_reusable'] and g['confirmation_state_reusable'] for g in rows),
            'missing_states':sum(not(g['search_state_reusable'] and g['confirmation_state_reusable']) for g in rows),
            'search_repetitions_per_state':3,'confirmation_repetitions_per_state':2,
            'reused_physical_run_references':reused_runs,'new_runs_required':30-reused_runs}
    jobs=[]
    for name in names:
        for seed in seeds:
            first=[g for g in groups if g['baseline']=='B4_INITIAL_CANDIDATE' and g['workload']==name and g['random_seed']==seed]
            assert len(first)==2 and all(g['search_reused_runs'] in (0,3) for g in first)
            known_reused_states=sum(g['search_state_reusable'] for g in first)
            new_state_evaluations_max=cfg['runtime_evaluation_budget']-1-known_reused_states
            jobs.append({'workload':name,'random_seed':seed,'max_logical_valid_unique_runtime_states':6,
                'seed_search_reused_runs':3,'seed_confirmation_reused_runs':2,
                'initial_candidates':2,'initial_candidate_states_reusable':known_reused_states,
                'initial_candidate_states_missing':2-known_reused_states,
                'remaining_later_state_evaluations_max':3,
                'new_state_evaluation_occurrences_max':new_state_evaluations_max,
                'new_search_runs_max':3*new_state_evaluations_max,
                'new_final_confirmation_runs_max':2,
                'new_physical_runs_max':3*new_state_evaluations_max+2})
    b4_new_max=sum(j['new_physical_runs_max'] for j in jobs)
    fixed_new=sum(v['new_runs_required'] for v in baseline_stats.values())
    initial_missing={s['measurement_state_sha256'] for s in states if s['baseline']=='B4_INITIAL_CANDIDATE'
                     and any(m['state_sha256']==s['measurement_state_sha256'] for m in missing)}
    initial_reused=sum(g['search_state_reusable'] for g in groups if g['baseline']=='B4_INITIAL_CANDIDATE')
    known_missing_unique={(m['state_sha256'],m['phase'],m['measurement_random_seed']) for m in missing}
    # A tighter optional bound after strict identity deduplication across baseline jobs.
    # 54 later state evaluations and 36 final confirmations remain unknown, not inferred.
    global_dedup_cap=len(known_missing_unique)+18*3*3+18*2
    protocol_seconds=definitions['measurement_protocol']['settings']['warmup_seconds']+definitions['measurement_protocol']['settings']['measurement_seconds']
    measured_wall=[]
    for path in sorted(C.glob('*/search_cache/raw_runs/*/aggregate.json')):
        for rep in read(path)['repetitions']:
            elapsed=rep.get('process_outcome',{}).get('elapsed_seconds')
            if elapsed is not None:measured_wall.append(elapsed)
    wall_median=statistics.median(measured_wall)
    budget={'module':'8D-1','campaign_executed':False,'new_GPU_runs_this_module':0,
        'fixed_baselines':baseline_stats,
        'fixed_baseline_totals':{'logical_states':24,'reused_state_references':sum(v['fully_reusable_states'] for v in baseline_stats.values()),
            'new_states':sum(v['missing_states'] for v in baseline_stats.values()),'new_physical_runs':fixed_new},
        'B4':{'random_seeds':seeds,'workloads':6,'independent_search_jobs':18,
            'same_per_job_budget_as_Proposed':True,'max_logical_valid_unique_runtime_states_per_job':6,
            'max_total_logical_runtime_state_evaluations':108,'max_nonseed_state_evaluations':90,
            'max_search_repetition_runs_before_reuse':324,'max_confirmation_runs_before_reuse':72,
            'max_physical_runs_before_reuse':396,
            'seed_search_run_references_reused':54,'seed_confirmation_run_references_reused':36,
            'seed_reuse_only_new_run_ceiling':306,
            'initial_candidate_occurrences':36,'initial_candidate_occurrences_reusable':initial_reused,
            'initial_candidate_occurrences_missing':36-initial_reused,
            'distinct_missing_initial_states':len(initial_missing),
            'known_initial_search_run_references_reused':initial_reused*3,
            'new_state_evaluation_occurrences_max_with_existing_reuse':sum(j['new_state_evaluation_occurrences_max'] for j in jobs),
            'new_physical_runs_max_with_existing_reuse':b4_new_max,
            'new_run_estimate':'Conservative ceiling; actual stop decisions, future donor matches and cross-job deduplication can reduce it',
            'jobs':jobs},
        'campaign':{'conservative_new_physical_run_ceiling':fixed_new+b4_new_max,
            'known_missing_run_references':len(missing),'known_missing_unique_runs_after_exact_identity_deduplication':len(known_missing_unique),
            'optional_global_exact_dedup_new_run_ceiling':global_dedup_cap,
            'warmup_plus_measurement_seconds_per_run':protocol_seconds,
            'warmup_plus_measurement_GPU_hours_at_conservative_ceiling':(fixed_new+b4_new_max)*protocol_seconds/3600,
            '8C_median_total_process_seconds_cost_diagnostic':wall_median,
            '8C_total_process_runs_for_cost_estimate':len(measured_wall),
            'wall_hours_at_8C_median_cost_diagnostic':(fixed_new+b4_new_max)*wall_median/3600,
            'cost_estimate_limit':'Observed median is a planning diagnostic, not a wall-time guarantee; startup/load/drain variability remains',
            'execution':'Serial on the original frozen hardware; no retry/fallback or budget extension'},
        'units':'State evaluation = three search measurements; physical run = one timed repetition. Confirmation runs do not drive search.'}
    freeze_file('gpu_campaign_budget.json',budget)

    # Original cache entries are imported lazily after the Random selector chooses a
    # state. Preloading unvisited candidates would wrongly consume unique-state slots.
    cache_by_raw={}
    for manifest in sorted(C.glob('*/search_cache/runtime_cache_manifest.json')):
        for cache_key,entry in read(manifest)['configurations'].items():
            if entry['status']!='complete':continue
            assert digest(entry['identity'])==cache_key
            assert file_hash(entry['aggregate'])==entry['aggregate_sha256']
            raw_set={r['runtime']['raw_result_path'] for r in entry['repetitions']}
            record={'original_cache_manifest':ref(manifest),'cache_key':cache_key,
                'exact_original_identity':entry['identity'],'aggregate':{'path':entry['aggregate'],'sha256':entry['aggregate_sha256']},
                'raw_measurements':[{'path':r['runtime']['raw_result_path'],'sha256':r['runtime']['raw_sha256'],
                                   'measurement_seed':r['seed']} for r in entry['repetitions']]}
            for raw in raw_set:cache_by_raw[raw]=record
    reusable_aggregates=[]
    for g in groups:
        if not g['search_state_reusable']:continue
        refs=[r for r in reused if r['baseline']==g['baseline'] and r['workload']==g['workload']
              and r['random_seed']==g['random_seed'] and r['state_sha256']==g['state_sha256'] and r['phase']=='search']
        assert len(refs)==3
        entry=cache_by_raw[refs[0]['original_run_path']]
        assert {r['original_run_path'] for r in refs}=={r['path'] for r in entry['raw_measurements']}
        reusable_aggregates.append({k:g[k] for k in ('baseline','workload','random_seed','state_sha256')} | entry)
    freeze_file('runtime_reuse_execution_contract.json',{
        'frozen_before_real_baseline_campaign':True,'runtime_evaluators_created_in_8D1':0,
        'exact_reusable_search_aggregate_entries':reusable_aggregates,
        'import_rule':'Lazily import only after the selected state passes the 12 identity checks, the complete original evaluator cache identity matches, and raw/aggregate hashes pass again',
        'budget_rule':'Every selected cached state still reserves a unique slot and increments the existing search counter; never preload unvisited states or replace a reused state with an extra experiment',
        'admission_rule':'Use the frozen effective 11-model / 21-path solver catalog; retain PadOCR-VL=0 in formal 12-model output',
        'fixed_baseline_run_order':['B0','B1','B2','B3'],
        'B4_run_order':[{'workload':n,'random_seed':s} for n in names for s in seeds],
        'search_run_order':'7101,7202,7303; same existing runner and per-workload runtime context',
        'fixed_baseline_confirmation_order':[8101,8202],
        'B4_confirmation_order':definitions['B4']['confirmation_order'],
        'B4_final_confirmation_rule':'Two runs disjoint from seed confirmation donor paths within this job. If final=seed, separately recorded original final-role confirmations can be reused only after exact checks; otherwise obtain two new independent runs',
        'future_cross_job_reuse':'Optional only after the new result is frozen and passes the same complete exact audit; do not count one raw measurement as independent repetitions',
        'fixed_physical_attempt_cap_per_state':5,'B4_search_physical_attempt_cap_per_job':18,'B4_confirmation_physical_attempt_cap_per_job':4,
        'failure_rule':'Keep failed/interrupted/resource-infeasible evidence and stop that run; no unplanned retry, replacement, or synthetic metric',
        'reporting_rule':'Report all baseline/workload results and all three RNG seeds, plus fixed equal-weight mean/median/range. Shared donors are declared. No post-hoc best baseline or seed selection',
        'confirmation_influences_search':False,'new_real_campaign_started':False})

    # Concrete fairness checks, all against rules fixed before measurements were read.
    b1=[s for s in states if s['baseline']=='B1'];b2=[s for s in states if s['baseline']=='B2'];b3=[s for s in states if s['baseline']=='B3']
    fixed_b1=len({digest(s['deployment']) for s in b1})==1
    fixed_b3=len({digest(s['deployment']) for s in b3})==1
    route_admission=read(OUT/'quality_path_audit.json')['all_positive_routes_admitted']
    proposed_protocol=read(C/names[0]/'protocol.json')
    compared_fields=('top_b','max_iterations','no_improvement_patience','runtime_evaluation_budget','rho_max','remove_pressure_threshold','mean_tolerance_ms','dry_run')
    budget_equality={k:cfg[k]==proposed_protocol['search_config'][k] for k in compared_fields}
    admitted={(r['task'],r['model']) for r in read(B/'quality_admission_matrix.json')['rows'] if r['quality_admissible']}
    deterministic_b1=all(s['selected_task_model']==definitions_b1 for s in b1 for definitions_b1 in [read(OUT/'baseline_routing_rules.json')['B1_task_to_model']])
    service=read(ROOT/'results/candidate_pool_v2_1/service_demand_v2_1.json')['models']
    role={m['id']:m['role'] for m in read(ROOT/'results/candidate_pool_v2_1/candidate_pool_v2_1.json')['models']}
    deterministic_b2=True
    for s in b2:
        for task,model in s['selected_task_model'].items():
            options=[m for k,m in admitted if k==task and role[m]=='specialist']
            if not options:options=[m for k,m in admitted if k==task and role[m]=='generalist']
            deterministic_b2 &= model==min(options,key=lambda m:(service[m]['service_time_ms'][task],m))
    checks={
        'no_baseline_definition_uses_8C_final_outcome':not definitions['outcome_data_used_for_definition'],
        'no_individual_baseline_search_receives_more_runtime_state_slots_than_Proposed':all(budget_equality.values()),
        'all_routes_quality_admissible':route_admission,
        'random_has_exactly_Proposed_per_job_search_and_confirmation_budget':all(budget_equality.values()) and definitions['B4']['confirmation_repetitions_per_seed_and_final']==2,
        'B3_deployment_is_identical_across_six_workloads':fixed_b3,
        'B1_fixed_deployment_and_rule_are_deterministic':fixed_b1 and deterministic_b1,
        'B2_matches_frozen_admission_and_s0_argmin':bool(deterministic_b2),
        'no_post_hoc_best_of_baselines_or_seeds':definitions['B4']['reporting'].startswith('Report all three'),
        'random_selector_uses_no_LP_rank':read(OUT/'random_initial_planning_audit.json')['LP_rank_call_count']==0,
        'same_frozen_model_pool_lambda_capacity_memory_bounds_protocol':read(OUT/'source_identity_validation.json')['status']=='PASS',
        'no_GPU_experiments_in_8D1':True,
    }
    passed=all(checks.values())
    fairness={'BASELINE_FREEZE_PASS':passed,'checks':checks,'search_budget_field_equality':budget_equality,
        'fairness_unit':'One workload × one RNG seed; individual Random search has Proposed budget',
        'random_replicate_aggregate_budget':{'random_jobs_per_workload':3,'Proposed_jobs_per_workload':1,
            'random_aggregate_max_logical_states_per_workload':18,'Proposed_max_logical_states_per_workload':6,
            'aggregate_budget_ratio':3,'explicitly_disclosed':True,'pooled_search_or_best_seed_selection_allowed':False,
            'reason':'Three user-requested random replicates, not one 18-state search'},
        'B0_exception':'Its definition is the mandated Proposed pre-search seed; exact pre-GPU witness is used, never Proposed final deployment',
        'existing_runtime_values_used_for_definition_or_candidate_choice':False,
        'existing_runtime_values_used_for_cost_estimation':'Only process durations, after definitions froze',
        'same_fixed_confirmation_reporting_budget':True,
        'evidence':{'definitions':ref(OUT/'baseline_definitions.json'),'random_seeds':ref(OUT/'random_seed_freeze.json'),
             'aggregate_lambda':ref(OUT/'aggregate_reference_workload.json'),'random_CPU_audit':ref(OUT/'random_initial_planning_audit.json'),
             'source_identity':ref(OUT/'source_identity_validation.json')},
        'remaining_blockers':[] if passed else [k for k,v in checks.items() if not v],
        'MODULE8D2_READY':passed,'MODULE8D2_started':False}
    freeze_file('baseline_fairness_audit.json',fairness)
    assert passed, fairness

    # Human-readable matrix is a direct rendering of full vectors, not an alias used
    # for reuse. The saved JSON and CSV retain formal twelve-model vectors.
    fixed_states=[s for s in states if s['baseline'] in ('B0','B1','B2','B3')]
    csvwrite('fixed_baseline_state_summary.csv',[{k:s[k] for k in ('baseline','workload','deployment','N_G','N_S','planning_memory_mib','selected_task_model','measurement_state_sha256')} for s in fixed_states])
    s4=['distilbert','resnet50','paddleocr','whisper-small']
    summary=f'''# Module 8D-1 — Baseline Definition Freeze + Measurement Reuse Audit

BASELINE_FREEZE_PASS = **true**. MODULE8D2_READY = **true**. 本阶段新增 GPU experiment = **0**；完成后停止，尚未启动 8D-2。

## 冻结的 baseline

所有以下列出的非零模型均为 **x=1**，其余正式十二维向量均为 0；完整向量、路由和质量见 JSON/CSV。S4 = DistilBERT + ResNet50 + PaddleOCR + Whisper-small。

| Baseline | demand-balanced / speech-heavy / vision-heavy / text/OCR-heavy / reasoning/VQA-heavy | no-VQA |
|---|---|---|
| B0 Seed-only | Phi + S4，N_G=1,N_S=4 | S4，N_G=0,N_S=4 |
| B1 Qwen-centric + Quality Repair | Qwen + PaddleOCR，N_G=1,N_S=1 | 同一个 always-on deployment |
| B2 Specialist-greedy + Minimal Generalist | Qwen + S4 + Distil-Whisper，N_G=1,N_S=5 | S4 + Distil-Whisper，N_G=0,N_S=5 |
| B3 Static Hybrid | Phi + S4，N_G=1,N_S=4 | 同一个 static deployment，Phi 保留但无 VQA 流量 |

B0 六个 seed 和 canonical routing 经原 empirical integer MILP/LP 精确复现，与 8C 搜索前 witness 相同。B1 使用固定六任务 repair universe：OCR→PaddleOCR，text/image/ASR/VQA/LID→Qwen；不因零 λ 剪掉模型。B2 task→model：text→DistilBERT，image→ResNet50，OCR→PaddleOCR，ASR→Distil-Whisper，LID→Whisper-small，active VQA→Qwen。所有选择仅由 frozen admission 和真实 s0 得到。

B3 aggregate λ 是六个原向量逐任务算术平均，λ digest=`{read(OUT/'aggregate_reference_workload.json')['lambda_sha256']}`。先冻结并 hash，再使用原 MILP seeder+canonical LP 生成 static deployment；所有六 workload 只重新计算 routing，不改变 deployment。

B4 random seeds = **{', '.join(map(str,seeds))}**。项目历史四任务 campaign 的两个 RNG seeds 1103/2207 被保留，第三个使用本次用户示例 47；不存在全项目三 seed policy。RNG seed 与测量 seeds 7101/7202/7303 分开。首轮选择在复用/质量结果审计前固定，由原 random selector 生成，CPU tripwire 证明 LP ranking 调用数 0；后续 trajectory 和 final deployment 未伪造或模拟。

## 严格 measurement reuse

扫描 **{audit['scan_coverage']['JSON_files_scanned']}** 个历史 JSON artifact，找到 **{audit['scan_coverage']['raw_measurements']}** 份多任务 raw measurement，包括 Module 8C 的 96 份、7C 与更早 real results。12 个身份条件必须全部相等，缺少证据也禁止复用；不使用浮点 rounding/tolerance。额外要求测量 phase/seed 匹配、原 config 完全一致及有效完整窗口。模型权重/adapter/环境/driver/核心源码与 8C 565 artifact 已重新验证，详见 source_identity_validation.json。

| Baseline | 已有状态可复用 | 缺失状态 | 新 GPU timed runs |
|---|---:|---:|---:|
| B0 | 6/6 | 0 | 0 |
| B1 | 0/6 | 6 | 30 |
| B2 | 0/6 | 6 | 30 |
| B3 | 5/6 | no-VQA 1 | 5 |

固定 baseline 每个 state 使用 3 search-protocol repetitions + 2 independent confirmations。B0 复用每个 8C seed 的三次 search raw 及两次 seed confirmation，共 30 份原始测量。B3 前五组与 B0 λ/vector/routing 完全相同，复用相同 25 份测量；这些不是额外独立样本。

B2 no-VQA 的 deployment 与 8C final 相同，但冻结 greedy routing 将 ASR 全部交给 Distil-Whisper，8C canonical routing 在两 speech 模型间分流；**routing identity 不相等，REUSE_FORBIDDEN**。B3 no-VQA 保留 Phi，其 replica vector 与原 8C seed/final 不同，也禁止复用。Module 7C 与其他旧 workload 不满足 exact λ/quality-admission/protocol identity，不能借用同名 deployment。

B4 18 个 RNG/workload job 的 seed 都可复用。首轮 36 个候选出现次数中 **{initial_reused}** 个有完整三次 search measurement；剩余 **{36-initial_reused}** 个出现次数对应 **{len(initial_missing)}** 个不同 measurement state。235 个已选复用引用指向 **{audit['unique_existing_raw_runs_selected']}** 份不同原始 raw；每项原 path、SHA256、12 条 equality checks、decision 及 measurement phase/seed 已保存。未知后续状态与 final 只可在实际选择后重新通过同样审计；seed/final confirmation donor 不得在同一个 Random job 内重叠。

## GPU campaign budget（仅规划）

B4 每 job：top_b=2、max_iterations=3、patience=1、≤6 valid unique states **含 seed**、每 state 3 repetitions、seed/final 各 2 confirmations；predictor=None。最多 18 jobs×6 = **108** 个 logical state evaluations，未复用前最多 **396** 个 timed runs。仅复用 seed 可降为 306；计入已冻结随机首轮 exact donors 后，新 state evaluation 出现次数最多 **{sum(j['new_state_evaluation_occurrences_max'] for j in jobs)}**，新增 physical runs 保守上限 **{b4_new_max}**（含 36 次未知 final confirmations）。这不是承诺执行满预算，也不是 60 个保证不同的全局 deployment。

B0–B3 合计必须新增 **{fixed_new}** 次 timed run；总 campaign 保守新增上限 **{fixed_new+b4_new_max}**。若对未来新产生的证据执行相同严格跨 job 去重，总体另一个更紧上限为 **{global_dedup_cap}**。本阶段已知缺失 measurement 为 {len(missing)} 个引用、精确去重后 {len(known_missing_unique)} 个不同 run；后续反馈轨迹尚未知，不能报虚构精确成本。

每 run warmup+measurement = {protocol_seconds} 秒，保守 ceiling 对应 **{(fixed_new+b4_new_max)*protocol_seconds/3600:.2f} GPU 小时**，另加 startup/load/drain。以 8C 实测 process 中位数 {wall_median:.2f} 秒仅做成本 diagnostic，约 **{(fixed_new+b4_new_max)*wall_median/3600:.2f} 小时**；非工期保证。失败保持原证据并停止该 run，不以 retry 扩大预算。

## Quality 与 fairness gate

全部正流量 routing 仅使用冻结的 **21** 条 admitted paths；正式排除的 6 条路径无流量。每个 baseline/workload 及随机初始候选均保存 effective_task_quality_summary：按路由比例加权 frozen audit metric，OCR 保留 secondary CER，零流量 task 标为 inactive。此值是 reporting diagnostic；WER/CER 的加权点估计不是新的 pooled transcript measurement，runtime objective 不变。B4 final quality 必须在未来 final routing 确定后按同一个规则填写。

不存在根据 8C final outcome 反向定义 baseline，也不存在 post-hoc best-of-baselines/seed selection。B1/B2 决策可由 frozen rule 自动复现，B3 六组 deployment 相同。Proposed 与 Random **每个 workload×RNG seed** 的搜索/确认预算严格相同；用户要求的三次 Random replicate 总预算显式为 Proposed 单次的 **3×**，不可合并成 18-state search，也不可只报最优 seed。三个 seed 全部报告，并预声明等权 mean/median/range。Exact donor 被使用时仍消耗 logical unique-state slot；只在候选已经随机选中后 lazy import，不能预填未访问状态。

全部 fairness checks PASS，remaining blockers = []。**可以进入 Module 8D-2 Real Baseline Campaign，但当前没有启动。**
'''
    path=OUT/'module8d1_summary.md'
    if path.exists():assert path.read_text()==summary
    else:path.write_text(summary)
    (OUT/'module8d1_summary.md.sha256').write_text(file_hash(path)+'\n')
    return budget,fairness


def finalize():
    verify_freeze()
    artifacts={str(p.relative_to(OUT)):ref(p) for p in sorted(OUT.iterdir()) if p.is_file()
               and p.name not in ('baseline_freeze_hashes.json','baseline_freeze_hashes.json.sha256')}
    implementations={str(p.relative_to(ROOT)):ref(p) for p in sorted((ROOT/'module8d_baseline_freeze').glob('*.py'))}
    required=('baseline_definitions.json','baseline_deployments.json','baseline_routing_rules.json',
              'aggregate_reference_workload.json','random_seed_freeze.json','measurement_reuse_audit.json',
              'reuse_manifest.csv','missing_measurements.csv','quality_path_audit.json','gpu_campaign_budget.json',
              'baseline_fairness_audit.json','module8d1_summary.md')
    assert all(name in artifacts for name in required)
    freeze_file('baseline_freeze_hashes.json',{'module':'8D-1','frozen':True,'BASELINE_FREEZE_PASS':True,
        'definition_random_seed_aggregate_hashes_written_before_CPU_planning':True,
        'before_first_baseline_GPU_experiment':True,'GPU_experiments_performed':0,
        'inputs':read(OUT/'baseline_definitions.json')['definition_inputs'],
        'artifacts':artifacts,'implementation_files':implementations,
        'hash_algorithm':'SHA256; artifact hashes are exact file bytes; lambda/state digests are canonical JSON',
        'self_hash':'External baseline_freeze_hashes.json.sha256 avoids recursive self-hashing'})
    for row in artifacts.values():assert file_hash(row['path'])==row['sha256']
    print('BASELINE_FREEZE_PASS=true; MODULE8D2_READY=true; new GPU experiments=0',flush=True)


def main():
    definitions=read(OUT/'baseline_definitions.json');states=read(OUT/'baseline_deployments.json')['states']
    audit=read(OUT/'measurement_reuse_audit.json')
    summarize(definitions,states,audit)
    finalize()


if __name__=='__main__':main()
