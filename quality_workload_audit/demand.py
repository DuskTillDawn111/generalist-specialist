"""Read frozen demand/search evidence and freeze demand-normalized proposals."""
import csv,hashlib,itertools,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/quality_workload_audit'
TASKS=['text','image','ocr','asr','visual_qa','lid']
def read(p):return json.loads((ROOT/p).read_text())
def write(name,d): (OUT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def csvwrite(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def distances(vectors):
 rows=[]
 for a,b in itertools.combinations(vectors,2):
  x,y=vectors[a],vectors[b]
  rows.append({'a':a,'b':b,'L1_distance':sum(abs(x[k]-y[k]) for k in TASKS),
               'cosine_similarity':sum(x[k]*y[k] for k in TASKS)/math.sqrt(sum(v*v for v in x.values())*sum(v*v for v in y.values()))})
 return rows
def active(d):return ', '.join(f'{m}={v}' for m,v in d.items() if v)
def main():
 workloads=read('results/deployment_search_multiworkload/benchmark_workloads.json')['workloads']
 catalog=read('results/candidate_pool_v2_service_demand/updated_service_demand_catalog.json')['models']
 factors=read('results/replica_capacity_calibration/replica_capacity_factors.json')['models']
 qwen=catalog['qwen']['service_time_ms']
 best={}
 for task in TASKS:
  eligible={m:v['service_time_ms'][task] for m,v in catalog.items() if v['role']=='specialist' and task in v['capabilities_verified']}
  best[task]={'status':'VERIFIED','model':min(eligible,key=eligible.get),'s0_ms':min(eligible.values()),'all_verified_specialists_s0_ms':eligible} if eligible else {'status':'NO_SPECIALIST','model':None,'s0_ms':None,'all_verified_specialists_s0_ms':{}}
 raw,request,demand,offload,model_pressure=[],[],[],[],[]
 reqvectors,demvectors,convergence={},{},{}
 for w in workloads:
  name=w['name'];lam=w['arrival_rates_rps'];r=sum(lam.values());d={k:lam[k]*qwen[k]/1000 for k in TASKS};D=sum(d.values())
  reqvectors[name]={k:lam[k]/r for k in TASKS};demvectors[name]={k:d[k]/D for k in TASKS}
  raw.append({'name':name,'lambda_rps':lam,'total_rps':r,'total_qwen_reference_demand_seconds_per_second':D})
  for k in TASKS:
   common={'workload':name,'task':k,'lambda_rps':lam[k],'s0_qwen_ms':qwen[k],'raw_request_share':lam[k]/r}
   request.append(common)
   demand.append(dict(common,absolute_qwen_service_demand_ms_per_second=lam[k]*qwen[k],absolute_qwen_service_demand_seconds_per_second=d[k],normalized_qwen_demand_share=d[k]/D))
   b=best[k]
   offload.append(dict(common,specialist_status=b['status'],best_specialist=b['model'],best_specialist_s0_ms=b['s0_ms'],
                       absolute_specialist_reference_demand_ms_per_second=lam[k]*b['s0_ms'] if b['s0_ms'] else None,
                       offload_leverage=qwen[k]/b['s0_ms'] if b['s0_ms'] else None,
                       weighted_offload_pressure_ms_per_second=lam[k]*(qwen[k]-b['s0_ms']) if b['s0_ms'] else None))
  for m,v in catalog.items():
   if v['role']!='specialist':continue
   ts=v['capabilities_verified'];original=sum(lam[k]*qwen[k] for k in ts);special=sum(lam[k]*v['service_time_ms'][k] for k in ts)
   model_pressure.append({'workload':name,'model':m,'tasks':','.join(ts),'qwen_offloadable_demand_ms_per_second':original,
                         'specialist_demand_ms_per_second':special,'weighted_offload_pressure_ms_per_second':original-special})
  short=name[:2].lower()
  tracepath=ROOT/f'results/integer_replica_search_multiworkload/{short}/search_trace.json'
  # W2 is explicitly inherited from 7C3; resolve saved summary reference.
  if not tracepath.exists():
   row=next(v for v in read('results/integer_replica_search_multiworkload/multiworkload_summary.json')['workloads'] if v['short']==short)
   tracepath=Path(row['source_trace']['path'])
  trace=json.loads(tracepath.read_text())
  rounds=[]
  for it in trace['iterations']:
   round_record={k:it[k] for k in ['iteration','deployment','ranking','selected_top_b','accepted','pruned_candidates']}
   round_record['candidate_evidence']=[{'deployment':r['mutation']['deployment'],'mutation':r['mutation']['operation'],
                                      'model':r['mutation']['model'],'rank':r.get('rank'),'status':r['status'],'reason':r.get('reason'),
                                      'delta_z':r.get('delta_z'),'canonical_max_pressure':(r.get('canonical') or {}).get('max_pressure'),
                                      'runtime_mean_ms':(r.get('runtime') or {}).get('weighted_mean_e2e'),
                                      'runtime_p95_ms':(r.get('runtime') or {}).get('weighted_p95_e2e'),
                                      'stability':(r.get('runtime') or {}).get('stability_state_v2')}
                                     for r in it['candidates']]
   rounds.append(round_record)
  convergence[name]={'total_qwen_reference_demand_seconds_per_second':D,'ASR_share':d['asr']/D,'LID_share':d['lid']/D,
                     'ASR_LID_share':(d['asr']+d['lid'])/D,'ASR_largest_single_task':d['asr']==max(d.values()),
                     'ASR_majority_of_total':d['asr']>D/2,
                     'whisper_small_standalone_speech_demand_seconds_per_second':sum(lam[k]*catalog['whisper-small']['service_time_ms'][k]/1000 for k in ['asr','lid']),
                     'source_trace':str(tracepath),'source_trace_sha256':hashlib.sha256(tracepath.read_bytes()).hexdigest(),
                     'seed_pressure':trace['seed']['utilization']['qwen'],'final_canonical_pressure':trace['canonical']['max_pressure'],
                     'final_model_pressure':trace['canonical']['model_pressure'],'final_routing_rps':trace['canonical']['routing'],
                     'final_deployment':trace['deployment'],'runtime_mean_ms':trace['runtime']['weighted_mean_e2e'],
                     'runtime_p95_ms':trace['runtime']['weighted_p95_e2e'],'rounds':rounds,'stop_reason':trace['status']}
 write('workload_raw_rates.json',{'tasks':TASKS,'rate_units':'requests/second','frozen_source':'results/deployment_search_multiworkload/benchmark_workloads.json','workloads':raw})
 csvwrite('workload_request_share.csv',request);csvwrite('workload_qwen_demand_share.csv',demand);csvwrite('offload_leverage.csv',offload)
 csvwrite('specialist_model_offload_pressure.csv',model_pressure)
 write('specialist_reference_demand.json',{'tasks':best,'note':'Exploratory cheapest verified specialist reference; quality not constrained; VQA excluded rather than treated as zero cost'})
 distance={'task_order':TASKS,'raw_request_share_space':distances(reqvectors),'qwen_demand_share_space':distances(demvectors),
           'interpretation':'L1 in [0,2]; L1/2=total variation. Cosine=1 means identical direction. Normalized distances hide overall load scale; absolute demand retained separately.'}
 write('workload_distance.json',distance)
 write('convergence_evidence.json',{'workloads':convergence,'empirical_g':{m:factors[m]['g'] for m in ['qwen','whisper-small','resnet50']}})
 lines=['# Module 8A W1/W2/W3 convergence audit','', '冻结目录中的 s0 是 standalone worker latency；λ×s0/1000 的单位为参考 GPU 秒/秒，仅是服务成本 proxy。未运行新 search。','',
        '|Workload|Total rps|Total Qwen demand s/s|ASR share|LID share|ASR+LID|Whisper-small speech demand s/s|',
        '|---|---:|---:|---:|---:|---:|---:|']
 for w in raw:
  c=convergence[w['name']]
  lines.append(f"|{w['name']}|{w['total_rps']:.6f}|{c['total_qwen_reference_demand_seconds_per_second']:.6f}|{c['ASR_share']:.2%}|{c['LID_share']:.2%}|{c['ASR_LID_share']:.2%}|{c['whisper_small_standalone_speech_demand_seconds_per_second']:.6f}|")
 lines += ['', 'ASR 在三组中确实是最大的单项 task demand，但没有超过总 demand 的 50%。不能把“最大单项”写成“绝对多数”。Whisper-small 同时覆盖 ASR 与 LID；二者合计贡献及小模型 LID 成本共同解释其价值。', '',
           '|Pair|Raw L1|Raw cosine|Qwen-demand L1|Qwen-demand cosine|','|---|---:|---:|---:|---:|']
 for x,y in zip(distance['raw_request_share_space'],distance['qwen_demand_share_space']):
  lines.append(f"|{x['a']} / {x['b']}|{x['L1_distance']:.6f}|{x['cosine_similarity']:.6f}|{y['L1_distance']:.6f}|{y['cosine_similarity']:.6f}|")
 lines += ['', '三组 request mix 本来就接近；W2 主要增加总量，不能凭 workload 名称认定覆盖了不同 regimes。Demand 空间的距离见上表：W1/W2 几乎同方向，W3 speech 比例略低，并未建立 vision/text/VQA 主导 regime。','',
           '## Offload explanation','', f"Qwen ASR s0={qwen['asr']:.3f} ms。ASR specialists: "+', '.join(f'{m}={s:.3f} ms' for m,s in best['asr']['all_verified_specialists_s0_ms'].items())+'.',
           f"Whisper-small LID={catalog['whisper-small']['service_time_ms']['lid']:.3f} ms，turbo LID={catalog['whisper_turbo']['service_time_ms']['lid']:.3f} ms；Qwen LID={qwen['lid']:.3f} ms。turbo ASR 比 small 更快，但 small 的 LID 更快且内存更少。distil-whisper ASR 最快，但不覆盖 LID。",'',
           '下面是每类 Specialist 对其全部覆盖任务的探索性 λ(sQ−sS)；它衡量总服务成本下降，**并非 canonical min-max objective 的排名**。在新增模型后 Qwen 剩余压力、分流约束、内存和并发干扰仍需由冻结 trace 解释。','',
           '|Workload|Specialist|Task(s)|Pressure relief ms/s|','|---|---|---|---:|']
 for row in model_pressure:lines.append(f"|{row['workload']}|{row['model']}|{row['tasks']}|{row['weighted_offload_pressure_ms_per_second']:.3f}|")
 lines += ['', '## Frozen search evidence and limits','']
 for name,c in convergence.items():
  lines += [f'### {name}', '', f"原 seed pressure={c['seed_pressure']:.6f}；final canonical max pressure={c['final_canonical_pressure']:.6f}。Final routing 将 ASR/LID 全部交给 Whisper-small，Qwen 保留其余四类。剩余 Qwen pressure={c['final_model_pressure']['qwen']:.6f}，Whisper pressure={c['final_model_pressure']['whisper-small']:.6f}。",'']
  for it in c['rounds']:
   lines.append(f"Round {it['iteration']}: incumbent {active(it['deployment'])}；Top-B={json.dumps(it['selected_top_b'],ensure_ascii=False)}；accepted={json.dumps(it['accepted'],ensure_ascii=False)}。")
  lines += ['', '|Round|Candidate|Rank|Δz|Mean E2E ms|P95 ms|Status / reason|', '|---|---|---:|---:|---:|---:|---|']
  for it in c['rounds']:
   for r in it['candidate_evidence']:
    if r['runtime_mean_ms'] is not None:
     lines.append(f"|{it['iteration']}|{active(r['deployment'])}|{r['rank']}|{r['delta_z']:.6f}|{r['runtime_mean_ms']:.3f}|{r['runtime_p95_ms']:.3f}|{r['status']} / {r['reason']}|")
  lines += ['', f"Stop={c['stop_reason']}；所有候选排名与 decision 原始字段保存于 convergence_evidence.json，source={c['source_trace']}。",'']
 lines += [f"Empirical g(2)：Qwen={factors['qwen']['g']['2']:.6f}，Whisper-small={factors['whisper-small']['g']['2']:.6f}，ResNet50={factors['resnet50']['g']['2']:.6f}。Qwen/Whisper 因子来自 ASR representative task；对 mixed tasks 的迁移仍为 surrogate。复制不提供新的任务覆盖；小模型卸载同时减轻两类 speech 负载，但实际接受仍以冻结 runtime 为准。三组 Top-B 中没有任何 x=2 deployment，因而 no accepted x=2 在此处主要表示未被选入实测，不能写成“实测副本更差”。",'',
           'Frozen v2 只有 Qwen verified visual_qa，三个 λVQA>0，故 Qwen presence 是 coverage 约束。不能将其解释为在所有 Generalists 中以性能胜出。', '',
           '相同 final 的证据解释：相近 mix、相同 mandatory coverage、同样 ASR/LID 卸载机会，并在有限 Top-B 的实际 runtime 决策中选择 small。不能仅凭 standalone cost 推导“small 必然全局最优”；其他 Specialist 未被全部实际评估，停止是预算内局部无改善而非穷举最优。']
 (OUT/'w1_w2_w3_convergence_analysis.md').write_text('\n'.join(lines)+'\n')
 # Freeze target shares and load scale before any planning (planning is omitted).
 targets={'demand_balanced':[1/6]*6,'speech_heavy':[.075,.075,.075,.45,.075,.25],
          'vision_heavy':[.08,.60,.08,.08,.08,.08], 'text_ocr_heavy':[.35,.075,.35,.075,.075,.075],
          'reasoning_vqa_heavy':[.08,.08,.08,.08,.60,.08], 'no_vqa_phase':[.2,.2,.2,.2,0,.2]}
 D=statistics.median(w['total_qwen_reference_demand_seconds_per_second'] for w in raw)
 proposals=[]
 for name,shares in targets.items():
  assert abs(sum(shares)-1)<1e-12
  target=dict(zip(TASKS,shares));lam={k:D*1000*target[k]/qwen[k] for k in TASKS};total=sum(lam.values())
  actual={k:lam[k]*qwen[k]/1000/D for k in TASKS}
  dominant=[k for k in TASKS if math.isclose(target[k],max(shares),abs_tol=1e-12)]
  proposals.append({'name':name,'target_qwen_demand_share':target,'lambda_rps':lam,'total_rps':total,
                    'raw_request_share':{k:lam[k]/total for k in TASKS},'qwen_reference_demand_share':actual,
                    'total_reference_demand_seconds_per_second':D,'dominant_task':dominant,
                    'expected_bottleneck_planning_diagnostic':{'reference_model':'qwen single worker','dominant_task_contribution':dominant,
                                                              'note':'Qwen reference demand only; actual deployment bottleneck unknown without routing/co-residency evidence'},
                    'semantic_scope':'Existing five non-VQA tasks, ScienceQA requests absent in this phase' if name=='no_vqa_phase' else 'Same six frozen task semantics'})
 proposal={'module':'8A','status':'FROZEN_PROPOSAL_NOT_FINAL_BENCHMARK','task_order':TASKS,'reference':'qwen frozen standalone s0',
           'definition_rule':'lambda[k] = D_ref * target_share[k] / (s0_qwen[k]/1000)',
           'scale_rule':'D_ref = median of original W1/W2/W3 total Qwen reference demands; common for all regimes, chosen without optimizer outputs',
           'total_reference_demand_seconds_per_second':D,'target_rules':'balanced=equal sixths; speech=ASR45%+LID25%; vision=image60%; text/OCR=35% each; reasoning=VQA60%; no-VQA=equal fifths over other tasks',
           'freeze_precedes_planning':True,'planning_sanity_check':'NOT_RUN; optional, skipped; no GPU deployment search',
           'proposals':proposals,'final_matrix_status':'PROVISIONAL_PENDING_QUALITY_POLICY_AND_V2_1_DECISION'}
 path=OUT/'proposed_demand_normalized_workloads.json';payload=json.dumps(proposal,ensure_ascii=False,indent=2)+'\n'
 if path.exists() and path.read_text()!=payload:raise RuntimeError('Proposals already frozen')
 path.write_text(payload);(OUT/'proposed_demand_normalized_workloads.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
 design=['# Demand-normalized workload design audit','', 'Target shares 冻结后 λ 由解析公式唯一确定。各 regime 的总 Qwen reference demand 相同，等于旧三组总 demand 中位数；没有依据 deployment/planning outcome 反向调参。没有运行 canonical planning 或 real search。','',
         '|Proposal|Target demand shares (text,image,ocr,asr,VQA,lid)|λ vector rps|Total rps|Dominant task|', '|---|---|---|---:|---|']
 for p in proposals:design.append(f"|{p['name']}|{', '.join(f'{v:.4f}' for v in p['target_qwen_demand_share'].values())}|{', '.join(f'{v:.6f}' for v in p['lambda_rps'].values())}|{p['total_rps']:.6f}|{', '.join(p['dominant_task'])}|")
 design += ['',f'Common reference demand={D:.9f} GPU-reference s/s。Raw share 与实际反算 demand share 全量见 JSON；浮点误差范围内完全匹配 target。相同 reference load 不代表相同真实 E2E 或不同 deployment。', '',
            'Vision regime 以 Imagenette image 为主；reasoning/VQA 以当前 ScienceQA-IMG multiple choice 为主，不能描述为覆盖 generic reasoning。Speech 同时包含转写与独立语言识别。每个 regime 保留原 dataset/split/prompt/normalization。','',
            'No-VQA phase 语义合理：真实流量阶段可以没有 ScienceQA 请求；既不是删去其他任务的视觉语义，也没有改变能力/评分。它移除唯一 VQA coverage 约束，可研究 Generalist choice 是否改变，但不预言一定切换到 Phi/MiniCPM；LID 仍由 Qwen/Whisper 覆盖。现有 Specialists 共同覆盖其余五类，所以该 phase 连 Generalist presence 都不是结构性必需，N_G=0 也具备 capability coverage；是否实际采用仍未测。', '',
            '当前冻结的是可复现 proposals 和 hash。最终 workload matrix 暂不能作为最终 benchmark 冻结：需先冻结六任务质量政策、确认 candidate_pool_v2.1 的能力/adapter/service-demand 注册策略；未来执行协议可延用现有 10s warmup / 210s window、deterministic periodic、三次重复，但本阶段不启动。建议保留这六种 regime；禁止依据未来 planning/search 结果改变 target shares。']
 (OUT/'workload_design_audit.md').write_text('\n'.join(design)+'\n')
 print(json.dumps({'distances':distance,'shares':{n:{k:v for k,v in c.items() if 'share' in k.lower()} for n,c in convergence.items()},'D_ref':D},indent=2))
if __name__=='__main__':main()
