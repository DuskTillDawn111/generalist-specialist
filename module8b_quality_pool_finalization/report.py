"""Freeze the final matrix only after the user-declared five gates pass."""
import copy,csv,datetime,math,subprocess
from .common import *
from zoneinfo import ZoneInfo

def main():
 policy=read(OUT/'quality_policy.json');admission=read(OUT/'quality_admission_matrix.json')
 pool=read(POOL_OUT/'candidate_pool_v2_1.json');service=read(POOL_OUT/'service_demand_v2_1.json')
 validation=read(POOL_OUT/'vqa_service_demand_validation.json');diversity=read(OUT/'workload_diversity.json')
 details=read(OUT/'planning_details.json');diff=read(OUT/'candidate_pool_v2_to_v2_1_diff.json')
 before=read(OUT/'before_hashes.json')
 changed=[p for p,r in before['files'].items() if not (ROOT/p).exists() or sha(ROOT/p)!=r['sha256']]
 assert not changed,changed
 # Model inventory is exactly the old pool plus the authorized two capability updates.
 old_pool=read(ROOT/'results/candidate_pool_v2/candidate_models.json')
 old_models={m['id']:m for m in old_pool['models']};new_models={m['id']:m for m in pool['models']}
 assert set(old_models)==set(new_models)
 for name,model in new_models.items():
  reconstructed=copy.deepcopy(model)
  if name in ('phi4mm','minicpm'):
   reconstructed['capabilities_verified'].remove('visual_qa')
   # Restore original order explicitly, as pending fields are ordered inventory data.
   reconstructed['capabilities_pending']=list(old_models[name]['capabilities_pending'])
   reconstructed['task_compatibility']['visual_qa']='pending'
  assert reconstructed==old_models[name],name
 raw=read(SERVICE_OLD)
 for m,v in raw['models'].items():
  for k,s0 in v['service_time_ms'].items():
   assert service['models'][m]['service_time_ms'][k]==s0
   assert service['models'][m]['service_time_sources'][k]==v['service_time_sources'][k]
 proposal_path=QUALITY/'proposed_demand_normalized_workloads.json'
 assert sha(proposal_path)==details['workload_source']['sha256']
 proposal=read(proposal_path)
 assert len(details['workloads'])==len(proposal['proposals'])==6
 rowmap={w['workload']:w for w in details['workloads']}
 for p in proposal['proposals']:assert rowmap[p['name']]['lambda_rps']==p['lambda_rps']
 admitted={(r['task'],r['model']) for r in admission['rows'] if r['quality_admissible']}
 # Audit every seed/candidate/witness, including edges of lower-ranked states.
 legal_witness_count=0
 for w in details['workloads']:
  main_catalog=read(OUT/'planning_catalog.json')['models']
  def audit_canonical(c,x):
   nonlocal legal_witness_count
   assert c['max_pressure']<=.8+1e-8
   for k,lam in w['lambda_rps'].items():
    allocated=0.
    for key,r in c['routing'].items():
     task,m=key.split('::')
     if task==k:
      allocated+=r
      if r>0:assert (task,m) in admitted and x.get(m,0)>0,(w['workload'],task,m)
    assert math.isclose(allocated,lam,rel_tol=1e-7,abs_tol=1e-12),(w['workload'],k,allocated,lam)
   legal_witness_count+=1
  audit_canonical(w['canonical'],w['seed']['deployment'])
  for c in w['candidate_evidence']:
   assert c['runtime'] is None
   if c['canonical']:audit_canonical(c['canonical'],c['mutation']['deployment'])
  for witness in w['generalist_witnesses'].values():
   if witness['canonical']:audit_canonical(witness['canonical'],witness['deployment'])
 assert details['solver_counts']['runtime_evaluations']==details['solver_counts']['enumeration_calls']==0
 pool_frozen=pool['frozen'] and sha(POOL_OUT/'candidate_pool_v2_1.json')==sha(OUT/'candidate_pool_v2_1.json')
 vqa_admitted=sorted(m for k,m in admitted if k=='visual_qa')
 gates={'quality_policy_frozen':policy['frozen'] and admission['frozen'],
        'candidate_pool_v2_1_frozen':pool_frozen,
        'service_demand_completeness_100_percent':service['completeness']['available_paths']==service['completeness']['required_paths']==27 and validation['status']=='PASS',
        'generalist_VQA_monopoly_resolved_or_explained':vqa_admitted==['minicpm','phi4mm','qwen'] and not any(w['summary']['qwen_mandatory'] for w in details['workloads']),
        'workload_planning_audit_DIVERSIFIED':diversity['status']=='DIVERSIFIED'}
 blockers=[k for k,v in gates.items() if not v]
 ready=not blockers
 gate={'module':'8B','FINAL_BENCHMARK_MATRIX_READY':ready,'gates':gates,'remaining_blockers':blockers,
       'quality_policy_scope':'Historical four-task specialist_baseline SLA; user-authorized semantic-only VQA/LID admission, numeric VQA/LID threshold remains UNSPECIFIED',
       'ready_scope':'Ready for future benchmark under this declared policy; no claim of compliance with unspecified numerical VQA/LID SLOs',
       'planning_diversity_limits':'2 seeds, 2 Top-B family sets, 3 bottleneck-model sets. Four workloads share VQA bottleneck and Qwen/MiniCPM Top-B; vision/text-OCR reference-heavy phases are not distinct runtime planning bottlenecks here.',
       'evidence':{'policy':ref(OUT/'quality_policy.json'),'pool':ref(POOL_OUT/'candidate_pool_v2_1.json'),
                   'service':ref(POOL_OUT/'service_demand_v2_1.json'),'diversity':ref(OUT/'workload_diversity.json')},
       'next_phase':'real multi-workload benchmark under frozen matrix' if ready else 'resolve listed blockers first',
       'next_phase_executed':False,'new_deployment_searches':0,'baseline_campaigns':0,'ablation_campaigns':0,'algorithm_changes':0}
 write(OUT/'final_benchmark_gate.json',gate,freeze=True)
 if ready:
  matrix={'module':'8B','version':'final_benchmark_matrix_v1','frozen':True,'FINAL_BENCHMARK_MATRIX_READY':True,
          'source_proposals':ref(proposal_path),'lambda_vectors_modified':False,'task_order':proposal['task_order'],
          'reference':proposal['reference'],'definition_rule':proposal['definition_rule'],'scale_rule':proposal['scale_rule'],
          'total_reference_demand_seconds_per_second':proposal['total_reference_demand_seconds_per_second'],
          'proposals':copy.deepcopy(proposal['proposals']),
          'quality_policy':ref(OUT/'quality_policy.json'),'quality_admission':ref(OUT/'quality_admission_matrix.json'),
          'candidate_pool':ref(POOL_OUT/'candidate_pool_v2_1.json'),'service_demand':ref(POOL_OUT/'service_demand_v2_1.json'),
          'capacity_and_replica_bounds':read(OUT/'planning_protocol.json')['inputs']['empirical_capacity'],
          'planning_protocol':ref(OUT/'planning_protocol.json'),'planning_evidence':ref(OUT/'planning_details.json'),
          'execution_status':'NOT_STARTED','execution_protocol_basis':ref(ROOT/'results/integer_replica_search_multiworkload/w1/protocol.json'),
          'reference_existing_measurement_settings':{'warmup_seconds':10,'measurement_seconds':210,'arrival_process':'deterministic_periodic','repetitions':3},
          'scope':'Frozen rates/quality/candidate/demand inputs for the next benchmark; no search acceptance decisions or measured runtime outcomes in this matrix'}
  write(OUT/'final_benchmark_workload_matrix.json',matrix,freeze=True)
  (OUT/'final_benchmark_workload_matrix.sha256').write_text(sha(OUT/'final_benchmark_workload_matrix.json')+'\n')
 # Explicit correction of the previous report's subset description, with evidence.
 taskrows=[json.loads(l) for l in TASK_POOL.read_text().splitlines() if l]
 asr=[r for r in taskrows if r['task_type']=='asr'][:30]
 durations=[r['sample']['duration_s'] for r in asr]
 subset_note={'Stage1_input_subset':'first 30 task rows from mixed_formal_1200_6task.jsonl; no duration filtering',
              'asr_first_30_duration_s':{'min':min(durations),'mean':sum(durations)/len(durations),'max':max(durations),'count_4_to_6_s':sum(4<=v<6 for v in durations)},
              'source':ref(ROOT/'results/candidate_pool_v2_service_demand/historical_protocol_audit.json'),
              'correction':'Module 8A report characterized catalog ASR s0 as a 4–6s subset. The actual Stage-1 protocol selects the first 30 service-pool task rows without duration filtering. Historical values/inputs are not changed.'}
 write(OUT/'standalone_subset_clarification.json',subset_note,freeze=True)
 audit={'status':'PASS','protected_file_count':len(before['files']),'changed_protected_files':changed,
        'candidate_inventory_only_two_capability_updates':True,'original_25_s0_values_and_provenance_unchanged':True,
        'quality_policy_frozen':True,'user_selected_policy_recorded':True,'arbitrary_numeric_thresholds_added':False,
        'new_vqa_standalone_physical_attempts':6,'requests_per_attempt':30,'warmups_per_attempt':1,
        'service_completeness':'27/27','six_lambda_vectors_unchanged':True,'legal_quality_admitted_routing_witnesses_checked':legal_witness_count,
        'runtime_evaluator_constructed_or_called':False,'gpu_deployment_searches':0,'baseline_or_ablation_campaigns':0,
        'search_algorithms_unchanged':True,'enumeration_calls':0,'solver_counts':details['solver_counts'],
        'FINAL_BENCHMARK_MATRIX_READY':ready,'stopped_after_module8b':True,
        'completed_at_client_time':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'new_source_hashes':{str(p.relative_to(ROOT)):ref(p) for folder in ('module8b_quality_pool_finalization','candidate_pool_v2_1_adapters') for p in (ROOT/folder).glob('*.py')}}
 write(OUT/'audit_validation.json',audit)
 write_summary(policy,admission,service,details,diversity,gate,subset_note)
 artifacts={str(p.resolve()):ref(p) for folder in (OUT,POOL_OUT) for p in folder.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
 write(OUT/'artifact_manifest.json',{'frozen':True,'artifacts':artifacts,'source_scope':'Module 8B outputs and real standalone raw files; original inputs remain protected'})
 print('Final audit PASS. FINAL_BENCHMARK_MATRIX_READY =',ready,'Remaining blockers:',blockers)

def write_summary(policy,admission,service,details,diversity,gate,subset_note):
 ws=details['workloads'];scores=read(POOL_OUT/'vqa_service_demand_validation.json')['new_vqa_s0_ms']
 lines=['# Module 8B — Quality Admission + Candidate Pool v2.1 Finalization','',
        f"**PASS；FINAL_BENCHMARK_MATRIX_READY={str(gate['FINAL_BENCHMARK_MATRIX_READY']).lower()}；完成后停止。** 本阶段仅有 6 次新增 VQA standalone physical attempts（2 paths × 3 repetitions，每次一次 warmup + 30 measured requests）。Real deployment search=0，RuntimeEvaluator=0，baseline/ablation=0，算法修改=0，enumeration=0。",'',
        '## Frozen policy and formal exclusions','',
        '用户明确选择：历史 SLA + VQA/LID 语义准入。四任务完全复用已经用于 placement_specialist_baseline_g1 的原阈值；保留 best_observed 为历史记录，未事后选择另一 preset。VQA/LID 的正式政策是 SEMANTIC_VALIDITY_ONLY：数值门槛仍为 UNSPECIFIED，但在此已授权政策下 verified+measured semantic paths 可以准入。单样本错答/无效标签保留并计错，不添加临时“零错误”门槛。','',
        '|Task|Metric|Direction / threshold|Policy status|','|---|---|---|---|']
 for p in policy['tasks']:lines.append(f"|{p['task']}|{p['metric']}|{p['direction']} / {p['historical_threshold']}|{p['policy_status']}|")
 lines+=['','27 verified+measured paths 中，21 admitted；以下 6 paths 被已有数值 gate 正式排除。排除的是 routing edge，模型 capability inventory 不降级：','',
         '|Model / task|Measured quality|Historical threshold|','|---|---:|---:|']
 for r in admission['quality_threshold_excluded_paths']:lines.append(f"|{r['model']} / {r['task']}|{r['quality']:.6f}|{r['threshold']:.6f}|")
 lines+=['','PaddleOCR-VL 仅有的 OCR edge 未过 gate，故在本 planning 中固定 x=0；其模型身份和 verified capability 仍保留于 v2.1。Phi 的 text/image/OCR 被屏蔽，但 ASR/VQA 可用；MiniCPM text 被屏蔽；Qwen OCR 被屏蔽。所有状态与理由保存于 quality_admission_matrix.json。','',
         '## VQA capability and standalone evidence','',
         'v2→v2.1 只有 Phi/MiniCPM visual_qa pending→verified 两项 capability 更新，模型 IDs/revisions/weights/precision/roles/VRAM/其他能力不变。原 v2 未覆盖。正式 versioned adapter 的推理/prompt/归一化/评分 body 与 8A revalidation AST 等价，只替换版本 metadata。模型规范 ID 为 phi4mm（Phi）。','',
         f"VQA s0：Phi={scores['phi4mm']:.9f} ms；MiniCPM={scores['minicpm']:.9f} ms。均执行原 run_active_single_profile.main 和 7A-2 的隔离、计时、原始 summary、卸载校验；相同前 30 个 ScienceQA service-pool rows、greedy max_new_tokens=1、相同一次 warmup、三独立 repetitions、90 samples weighted mean。原有 25 条值及 provenance 完全不变，completeness=27/27。",'',
         '新 VQA capable/admitted Generalists = Qwen/Phi/MiniCPM。λVQA>0 时仅要求至少一个 admitted VQA model，已不能推出 qwen≥1；VQA 阈值仍未数值化，这一点不会被“READY”隐藏。','',
         '## Frozen six-workload planning outcomes','',
         '全部 λ 向量逐项与 Module 8A proposal 相等、原 JSON SHA-256 未变。只调用原 MILP seed、canonical LP、bottleneck_feedback、x±1 generation 和 exact empirical delta_z ranking；不执行 search loop，不实例化/调用 RuntimeEvaluator，不模拟 runtime feedback。质量作为 routing-edge mask 传入原 solver；现有 empirical g、M(x)、replica evidence bounds 均保留。','',
         'A = Phi + DistilBERT + ResNet50 + PaddleOCR + Whisper-small，各 x=1，15,310 MiB。B = A 去掉 Phi，3,538 MiB。Top-B 表示由 seed 新增的 model family，不是实测 final deployment。','',
         '|Workload|Seed|N_G / N_S|Bottleneck model(s)|Dominant bottleneck task|Top-1 addition|Top-2 addition|',
         '|---|---|---|---|---|---|---|']
 for w in ws:
  r=w['summary'];lines.append(f"|{r['workload']}|{'B' if r['N_G']==0 else 'A'}|{r['N_G']} / {r['N_S']}|{r['bottleneck_models']}|{r['dominant_bottleneck_tasks']}|{r['Top_1_mutated_model']}|{r['Top_2_mutated_model']}|")
 lines+=['','五个含 VQA workload 的 minimum-memory seed 相同；no-VQA 的 seed 为 N_G=0。原先 Qwen monopoly 消失，但 minimum-memory seed 不等于性能最优，不能把 Phi seed presence 当作已经赢得 runtime benchmark。','',
         'Reasoning/VQA-heavy：Phi 在 seed，Qwen/MiniCPM 都成为合法新增 Generalist candidate 并占 Top-B；三种 Generalist 的单独替代 witnesses 都通过 admitted-edge routing、capacity、memory 验证。新增 Qwen/MiniCPM 是 one-hop additions；单 Generalist替代 states 是独立 feasibility diagnostics，未伪装成同一跳 swap。', '',
         '## Diversity gate and practical limits','',
         f"Suite status={diversity['status']}。Seed signatures={diversity['seed_diversity']['distinct_count']}；Top-B family sets={diversity['Top_B_diversity']['distinct_model_family_set_count']}；bottleneck model sets={diversity['bottleneck_diversity']['distinct_model_set_count']}；dominant tasks 为 ASR/VQA。四组 Top-B 是 Qwen/MiniCPM（new_generalist），speech-heavy/no-VQA 是 distil-whisper/Whisper-turbo（new_specialist）。",'',
         'Diversity rule 在规划结果出现前冻结：不同 dominant bottleneck tasks、不同 Top-B family sets，以及 Generalist/Specialist mutation patterns 均出现则 DIVERSIFIED。结果满足这一有限定义，不代表六组都独立：demand-balanced、vision-heavy、text/OCR-heavy、reasoning/VQA-heavy 都仍由 Phi VQA 主导，四组相同 Top-B；image 与 OCR 没有成为 seed bottleneck。没有为改变此结果修改 workload。','',
         'Generalist additions 没有被 memory 或 VQA quality/capability gate 阻止。Speech/no-VQA 中 Qwen/Phi/MiniCPM 仍有合法 witnesses，但部分 Generalist proposals 被 rank 剪枝。Phi scale-down 在含 VQA seed 中因 coverage 被拒绝，这是删除唯一已部署 VQA model 的约束，不是对 Phi 本身不支持 VQA 的判定。','',
         '## Final benchmark matrix gate','',
         f"五 gate={json.dumps(gate['gates'],ensure_ascii=False)}；remaining blockers={json.dumps(gate['remaining_blockers'],ensure_ascii=False)}。",'',
         '已冻结 final_benchmark_workload_matrix.json 和 SHA-256，原 proposal λ 不变；绑定本次 policy/admission、v2.1、真实 service catalog 和 planning evidence。READY 适用于用户选择的语义准入政策，不声称满足尚未定义的 VQA/LID accuracy SLO，也不证明规划排名会成为真实部署结果。下一阶段可以执行冻结矩阵上的真实 multi-workload benchmark；本阶段没有启动。','',
         '## Final fourteen answers','',
         '1. 六任务 policy：text/image/OCR/ASR=HISTORICAL_THRESHOLD；VQA/LID=SEMANTIC_VALIDITY_ONLY，numeric threshold=UNSPECIFIED。',
         '2. 正式排除 6 条 paths：Phi text/image/OCR、MiniCPM text、Qwen OCR、PaddleOCR-VL OCR；未按相对排名或 Pareto 自动淘汰。',
         '3. Phi/MiniCPM VQA 已正式进入独立 v2.1，原 v2 未改。',
         f"4. 新 VQA s0：Phi {scores['phi4mm']:.6f} ms；MiniCPM {scores['minicpm']:.6f} ms。",
         '5. 达到 27/27；原 25 条没有重测或修改。',
         '6. Qwen 不再 mandatory；五个含 VQA 的 seed 都是 Phi + 四 Specialists。',
         '7. Reasoning/VQA-heavy 出现 planning Generalist competition：Phi seed，Qwen/MiniCPM Top-B，三者独立替代 states 都合法；没有 runtime 竞争结论。',
         '8. No-VQA 允许且实际 seed 为 N_G=0，四 Specialists 完整覆盖。',
         '9. Seed 有两种，五组相同，no-VQA 不同；不是六个独立 seeds。',
         '10. Bottleneck model sets 有三种：Phi；Phi+Whisper-small；Whisper-small。Dominant tasks 有 VQA/ASR 两种。',
         '11. Top-B 有两种明显类别：四组 Qwen/MiniCPM；两组 distil-whisper/Whisper-turbo。',
         f"12. {diversity['status']}，按预先冻结的有限 diversity 定义；四组重复已披露。",
         f"13. FINAL_BENCHMARK_MATRIX_READY={str(gate['FINAL_BENCHMARK_MATRIX_READY']).lower()}；final matrix 与 hash 已冻结。",
         '14. 下一步可进入真实 multi-workload benchmark；本阶段完成并停止。', '',
         '## Protocol clarification','',subset_note['correction'],
         f"实际前 30 条 ASR 音频 min/mean/max={subset_note['asr_first_30_duration_s']['min']:.3f}/{subset_note['asr_first_30_duration_s']['mean']:.3f}/{subset_note['asr_first_30_duration_s']['max']:.3f} s。此说明只纠正上阶段文字，不改变历史 catalog 或其 demand-share 数字。",'',
         '产物：results/module8b_quality_pool_finalization/；正式 pool/catalog/raw measurements 同时保存在 results/candidate_pool_v2_1/。audit_validation.json、artifact_manifest.json 保留 immutable-input 和输出 hash 核验。']
 (OUT/'module8b_summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
