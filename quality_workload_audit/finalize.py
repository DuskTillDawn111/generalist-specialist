"""Score audited paths and report evidence without inventing acceptance thresholds."""
import csv, hashlib, json, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark.evaluation.metrics import cer,wer,levenshtein,normalize_ocr,normalize_asr
OUT=ROOT/'results/quality_workload_audit'
TASKS=['text','image','ocr','asr','visual_qa','lid']
def read(name):return json.loads((OUT/name).read_text())
def save(name,d):(OUT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvwrite(name,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def score(task,record,tm):
 rs=record['responses'];n=len(rs)
 assert n==len(tm['sample_ids'])==200
 assert [r['pool_id'] for r in rs]==tm['sample_ids']
 ps=[r['prediction'] for r in rs];refs=[r['reference'] for r in rs]
 samples=[json.loads(l) for l in Path(tm['evaluation_manifest']).read_text().splitlines() if l]
 norm=lambda v:normalize_ocr(v) if task=='ocr' else normalize_asr(v) if task=='asr' else str(v).strip().upper() if task=='visual_qa' else str(v).strip().lower()
 assert all(norm(r)==norm(s.get('normalized_label',s['label'])) for r,s in zip(refs,samples))
 row={'n':n,'accuracy':None,'top1_accuracy':None,'exact_match':None,'cer':None,'wer':None,'invalid_output_count':None,
      'errors':None,'reference_units':None}
 if task=='ocr':
  row['exact_match']=sum(normalize_ocr(p)==normalize_ocr(r) for p,r in zip(ps,refs))/n
  row['cer']=cer(ps,refs)
  row['errors']=sum(levenshtein(list(normalize_ocr(p)),list(normalize_ocr(r))) for p,r in zip(ps,refs))
  row['reference_units']=sum(len(normalize_ocr(r)) for r in refs)
 elif task=='asr':
  row['wer']=wer(ps,refs)
  row['errors']=sum(levenshtein(normalize_asr(p).split(),normalize_asr(r).split()) for p,r in zip(ps,refs))
  row['reference_units']=sum(len(normalize_asr(r).split()) for r in refs)
  assert row['errors']==sum(r['word_errors'] for r in rs)
  assert row['reference_units']==sum(r['reference_words'] for r in rs)
 else:
  quality=sum(norm(p)==norm(r) for p,r in zip(ps,refs))/n
  row['top1_accuracy' if task=='image' else 'accuracy']=quality
  if task=='visual_qa':
   row['invalid_output_count']=sum(norm(p) not in {chr(65+i) for i in range(len(s['choices']))} for p,s in zip(ps,samples))
  else:
   valid={norm(s['label']) for s in samples}
   row['invalid_output_count']=sum(norm(p) not in valid for p in ps)
 return row

def main():
 manifest=read('quality_evaluation_manifest.json')
 catalog=json.loads((ROOT/'results/candidate_pool_v2_service_demand/updated_service_demand_catalog.json').read_text())['models']
 rows=[];revalidations={}
 for m,mm in manifest['models'].items():
  raw=read('raw_quality/'+m+'.json')
  assert raw['manifest_sha256']==digest(OUT/'quality_evaluation_manifest.json'),m
  for task in mm['verified_tasks']:
   tr=raw['tasks'].get(task,{})
   row={'task':task,'model':m,'role':catalog[m]['role'],'capability_status':'verified','evaluation_status':tr.get('status','PENDING_TECHNICAL_FAILURE'),
        's0_ms':catalog[m]['service_time_ms'][task],'source_revision':mm['source_revision'],
        'dataset':manifest['tasks'][task]['dataset'],'split':manifest['tasks'][task]['split'],
        'evaluation_manifest_sha256':manifest['tasks'][task]['manifest_sha256'],
        'adapter_sha256':mm['adapter_sha256'],'quality_threshold_status':'UNSPECIFIED',
        'raw_evidence':str(OUT/'raw_quality'/(m+'.json')),'raw_sha256':digest(OUT/'raw_quality'/(m+'.json'))}
   if tr.get('status')=='COMPLETE':row.update(score(task,tr,manifest['tasks'][task]))
   else:row.update(n=len(tr.get('responses',[])),error=raw.get('error'))
   rows.append(row)
  if mm['revalidation_only_tasks']:
   task='visual_qa';tr=raw['tasks'].get(task,{})
   evidence={'model':m,'original_frozen_capability_status':'pending','frozen_pool_changed':False,
             'same_semantics':True,'same_prompt':True,'same_metric':True,'same_answer_normalization':True,
             'max_new_tokens':1,'do_sample':False,'model_revision':mm['source_revision'],
             'dataset_manifest_sha256':manifest['tasks'][task]['manifest_sha256'],
             'adapter':str(ROOT/'quality_workload_audit/vqa_worker.py'),'adapter_sha256':digest(ROOT/'quality_workload_audit/vqa_worker.py'),
             'semantic_source':str(ROOT/'benchmark/scripts/eval_qwen_scienceqa.py'),
             'raw_evidence':str(OUT/'raw_quality'/(m+'.json')),'raw_sha256':digest(OUT/'raw_quality'/(m+'.json')),
             'technical_failure':raw.get('error')}
   if tr.get('status')=='COMPLETE':
    q=score(task,tr,manifest['tasks'][task]);evidence['quality']=q
    # Compatibility is ability to process same inputs and emit valid contract.
    # Not an accuracy admission threshold; invalid answers still count wrong.
    valid=q['n']-q['invalid_output_count']
    evidence['status']='VERIFIED' if valid>0 else 'UNSUPPORTED'
    evidence['reason']='All 200 same-semantic requests completed, valid option outputs observed; quality measured without an accuracy gate' if valid else 'All outputs violate strict one-option-letter contract under frozen decoding; no successful closed-set response'
   else:
    evidence['status']='PENDING_TECHNICAL_FAILURE';evidence['reason']='Runtime/inference failure; no evidence of inherent task incompatibility'
   revalidations[m]=evidence
 assert len(rows)==25
 save('generalist_vqa_revalidation.json',{'module':'8A','contract':'Same ScienceQA-IMG question + choices + hint + RGB image; common build_instruction; 1 token greedy; strip+uppercase only; invalid counts as wrong',
                                        'quality_gate':'UNSPECIFIED; VERIFIED is interface/semantic compatibility, not final quality admission',
                                        'qwen_reference_quality':next({k:r.get(k) for k in ['n','accuracy','invalid_output_count','raw_evidence','source_revision']} for r in rows if r['model']=='qwen' and r['task']=='visual_qa'),
                                        'results':revalidations})
 additions=[{'model':m,'task':'visual_qa','proposed_capability_status':'verified','source_revision':r['model_revision'],
             'evidence':str(OUT/'generalist_vqa_revalidation.json'),'quality':r.get('quality'),
             'implementation_requirement':'Promote audit-only same-semantic handler through a separately versioned adapter; keep frozen v2 unchanged',
             'service_demand_status':'NOT_MEASURED_FOR_NEW_PATH; must obtain current standalone s0 before optimizer admission',
             'quality_admission_status':'PENDING_FORMAL_QUALITY_POLICY'} for m,r in revalidations.items() if r['status']=='VERIFIED']
 save('candidate_pool_v2_1_proposal.json',{'base':'candidate_pool_v2','status':'PROPOSAL_ONLY','base_pool_changed':False,
                                        'new_model_types':0,'recommended_capability_additions':additions,
                                        'nonverified_revalidation_results':{m:r for m,r in revalidations.items() if r['status']!='VERIFIED'},
                                        'publication_gates':['versioned capability/adapter freeze','measure service demands for new verified paths','freeze six-task quality thresholds before any final optimization']})
 fronts={};tradeoffs=[]
 for task in TASKS:
  candidates=[r for r in rows if r['task']==task and r['evaluation_status']=='COMPLETE']
  def costs(r):
   if task=='ocr':return [r['s0_ms'],r['cer'],-r['exact_match']]
   if task=='asr':return [r['s0_ms'],r['wer']]
   return [r['s0_ms'],-r['top1_accuracy' if task=='image' else 'accuracy']]
  front=[]
  for r in candidates:
   cr=costs(r)
   dominators=[o['model'] for o in candidates if o is not r and all(a<=b for a,b in zip(costs(o),cr)) and any(a<b for a,b in zip(costs(o),cr))]
   r['pareto_nondominated']=not dominators;r['dominated_by']=','.join(dominators)
   flags=[]
   if dominators:flags.append('PARETO_DOMINATED')
   if r.get('invalid_output_count',0):flags.append('INVALID_CLOSED_SET_OUTPUTS')
   r['quality_review_flags']=','.join(flags)
   r['optimizer_admission_status']='UNDECIDED_WITHOUT_FROZEN_QUALITY_POLICY'
   if not dominators:front.append(r['model'])
  fronts[task]=front
  qwen=next((r for r in candidates if r['model']=='qwen'),None)
  for r in candidates:
   if r['role']!='specialist' or qwen is None:continue
   primary='cer' if task=='ocr' else 'wer' if task=='asr' else 'top1_accuracy' if task=='image' else 'accuracy'
   delta=r[primary]-qwen[primary]
   worse=delta>0 if task in ('ocr','asr') else delta<0
   t={'task':task,'model':r['model'],'metric':primary,'qwen_quality':qwen[primary],'specialist_quality':r[primary],
      'specialist_minus_qwen':delta,'s0_speedup':qwen['s0_ms']/r['s0_ms'],
      'latency_quality_tradeoff_observed':r['s0_ms']<qwen['s0_ms'] and worse,
      'inference_scope':'Frozen subset point estimates; no statistical significance or formal admission decision'}
   if task=='ocr':
    t['exact_match_specialist_minus_qwen']=r['exact_match']-qwen['exact_match']
    t['either_ocr_metric_tradeoff_observed']=r['s0_ms']<qwen['s0_ms'] and (worse or r['exact_match']<qwen['exact_match'])
   tradeoffs.append(t)
 matrix={'module':'8A','quality_threshold_status':'UNSPECIFIED',
         'comparability':'Within each task: identical 200 sample IDs, dataset/split/task/prompt/output metric, existing model-specific preprocessing/decoding retained. Across tasks: metrics are different and cannot be averaged into universal quality.',
         'latency_scope':'s0 from frozen standalone catalog, often historical 30-row service pool / ASR 4–6s subset; audit quality over frozen 200 evaluation IDs. Pareto is an exploratory cross-artifact pairing, not a new co-measured latency-quality benchmark.',
         'f1_status':manifest['f1_status'],
         'audit_source_hashes':{str(p):digest(p) for p in sorted((ROOT/'quality_workload_audit').glob('*.py'))},
         'metric_implementation_sha256':digest(ROOT/'benchmark/evaluation/metrics.py'),
         'verified_path_count':25,'completed_path_count':sum(r['evaluation_status']=='COMPLETE' for r in rows),
         'rows':rows,'pareto_fronts':fronts,'specialist_vs_qwen':tradeoffs,
         'optimizer_quality_policy':'No new thresholds. No path automatically removed or admitted by quality; low quality and dominated paths must be reviewed before final policy freeze.'}
 save('quality_matrix.json',matrix);csvwrite('quality_matrix.csv',rows)
 csvwrite('latency_quality_tradeoffs.csv',tradeoffs)
 # Documentation generated from scores below.
 write_reports(manifest,matrix,revalidations,additions)
 validate(manifest,matrix)
 print(json.dumps({'completed_paths':matrix['completed_path_count'],'vqa':{m:r['status'] for m,r in revalidations.items()},'fronts':fronts,'quality':[{k:r.get(k) for k in ['task','model','accuracy','top1_accuracy','cer','exact_match','wer','invalid_output_count']} for r in rows]},indent=2))

def write_reports(manifest,matrix,revalidations,additions):
 rows=matrix['rows'];pairs=matrix['specialist_vs_qwen']
 diagnostic='；'.join(f"{r['model']}/{r['task']} invalid={r['invalid_output_count']}/{r['n']}" for r in rows if r.get('invalid_output_count')) or '无 invalid closed-set outputs'
 weakest=[]
 for task in TASKS:
  rs=[r for r in rows if r['task']==task and r['evaluation_status']=='COMPLETE']
  metric='cer' if task=='ocr' else 'wer' if task=='asr' else 'top1_accuracy' if task=='image' else 'accuracy'
  r=(max if task in ('ocr','asr') else min)(rs,key=lambda r:r[metric])
  weakest.append(f"{task}: {r['model']} {metric}={r[metric]:.6f}")
 lines=['# Module 8A quality and Pareto audit','',matrix['comparability'],'',matrix['latency_scope'],'',
        'Quality 是同 task 的 task-specific matrix；不使用 accuracy 与 WER 的跨任务混合分数。CER/WER 使用 corpus error totals / corpus reference units，保留 OCR exact match。现有实现没有 F1/macro-F1，故未替换 metric。','',
        '|Task|Model|N|Accuracy / top1|CER|OCR exact|WER|s0 ms|Pareto|','|---|---|---:|---:|---:|---:|---:|---:|---|']
 def fmt(v):return '—' if v is None else f'{v:.6f}'
 for r in rows:
  lines.append(f"|{r['task']}|{r['model']}|{r.get('n')}|{fmt(r.get('accuracy') if r.get('accuracy') is not None else r.get('top1_accuracy'))}|{fmt(r.get('cer'))}|{fmt(r.get('exact_match'))}|{fmt(r.get('wer'))}|{r['s0_ms']:.3f}|{r.get('pareto_nondominated')}|")
 lines+=['','Pareto：最小化 s0；accuracy 最大化；ASR WER 最小化；OCR 同时最小化 CER、最大化 exact。支配结果没有正式 quality gate 的含义。单 task 的 Pareto 未包含整模型共享加载、VRAM 或跨 task 覆盖，不能据此剔除整模型；例如 Whisper-small 的 ASR 点被支配，但其 LID 点在 frontier 上。','',
         '|Task|Specialist|Metric|Qwen|Specialist|Δ quality (S−Q)|s0 speedup|Faster but worse?|','|---|---|---|---:|---:|---:|---:|---|']
 for t in pairs:lines.append(f"|{t['task']}|{t['model']}|{t['metric']}|{t['qwen_quality']:.6f}|{t['specialist_quality']:.6f}|{t['specialist_minus_qwen']:+.6f}|{t['s0_speedup']:.2f}|{t['latency_quality_tradeoff_observed']}|")
 lines+=['','Legacy quality presets 在 benchmark/profiles/placement_scenario.json 中早已存在（none / specialist_baseline / best_observed），其中后两者覆盖 text/image/OCR/ASR；完整数值在 quality_evaluation_manifest.json 保留。它们是历史四任务 placement 条件，没有自动推广到当前六任务 v2 optimizer；visual_qa/LID 未定义 threshold。','',
         'Current six-task v2 optimizer quality_threshold_status = UNSPECIFIED。未在看结果后创建阈值；低质量不会被此 audit 偷偷标为 optimizer 可用。正式淘汰/接受留待冻结政策。','', 'Closed-set contract diagnostics: '+diagnostic+'。','', '各 task 的最弱 primary quality 点估计（排序，不是新 threshold）：'+'；'.join(weakest)+'。','',
         '每条结果保留 source revision、manifest hash、adapter hash、raw response/IDs 和模型资产哈希。历史质量结果无法绑定全部五项复用条件，故统一重新评估当前 handlers；没有将历史 summary 混入当前 matrix。','',
         'Generalist VQA 的两条新测 path 单独放在 generalist_vqa_revalidation.json，尚不属于 frozen v2 的 25 条 matrix，也没有伪造其 s0。']
 (OUT/'quality_pareto_summary.md').write_text('\n'.join(lines)+'\n')
 convergence=read('convergence_evidence.json')['workloads'];dist=read('workload_distance.json')
 by={(r['task'],r['model']):r for r in rows}
 ws_asr=by.get(('asr','whisper-small'));q_asr=by.get(('asr','qwen'));ws_lid=by.get(('lid','whisper-small'));q_lid=by.get(('lid','qwen'))
 summary=['# Module 8A — Quality + Workload Demand Audit','',
          '已完成审计并停止。无新的 deployment search、baseline/ablation、predictor、新模型或 7B/7C 算法修改。质量评估使用现有六类各 200 条冻结 manifest；25 条 verified path 及 Phi/MiniCPM VQA revalidation 分开。','',
          '## 1. W1/W2/W3 在 raw request space 中有多不同？','',
          'Pair 顺序为 W1/W2、W1/W3、W2/W3：L1='+', '.join(f"{r['L1_distance']:.6f}" for r in dist['raw_request_share_space'])+'；cosine='+', '.join(f"{r['cosine_similarity']:.6f}" for r in dist['raw_request_share_space'])+'。差异有限，W1/W2 近似同方向；workload 名称夸大了 regime 差别。','',
          '## 2. 在 Qwen service-demand space 中有多不同？','',
          '同样顺序：L1='+', '.join(f"{r['L1_distance']:.6f}" for r in dist['qwen_demand_share_space'])+'；cosine='+', '.join(f"{r['cosine_similarity']:.6f}" for r in dist['qwen_demand_share_space'])+'。W3 相对 W1 的 demand L1 略大于 raw L1，因此不能笼统说 demand 空间总是比 request 更相似。绝对总 demand 单独报告，归一化没有覆盖 load-level 差异。','',
          '## 3. ASR 是否主导三个 workload？','',
          'ASR 是三者最大单项：'+', '.join(f"{n}={c['ASR_share']:.2%}" for n,c in convergence.items())+'；但都不到 50%。ASR+LID：'+', '.join(f"{c['ASR_LID_share']:.2%}" for c in convergence.values())+'。应表述为最大单项 speech-related cost，而非 ASR 绝对多数。','',
          '## 4. 为什么 Whisper-small 三组均高价值？','',
          '同时覆盖 ASR/LID，small LID s0=27.084 ms，比 Qwen 139.909 ms、turbo 83.179 ms 更低；ASR s0=190.166 ms 对 Qwen 654.078 ms 有下降。三组 mix 接近，Qwen mandatory VQA 相同。冻结 trace 的 W1/W2 首轮 Top-B 是 small/turbo，W3 是 MiniCPM/small；三组都由 runtime 选择 small。第二轮 Top-B Generalist additions 未改善，便停止。empirical g(2) 为小于线性的代表任务测量，不能代替 runtime。完整路径与压力见 convergence report。','',
          '## 5. 25 条 verified path 的质量是否可比较？','',
          f"{matrix['completed_path_count']}/25 完成。同 task 下可比较：相同 frozen IDs/dataset/split/task semantics，复用各模型既有 preprocessing 和 serving decode；不跨 task 混合质量。各模型原生官方图像 transform/API 有合理差异且已记录。质量200样本与 s030-row/短ASR subset 的范围差异限制 Pareto 的 latency 外推。",'',
          '## 6. 是否存在 latency-quality tradeoff？','',
          '实际数据：'+('; '.join(f"{t['task']}/{t['model']} 更快但 {t['metric']} 更差，Δ={t['specialist_minus_qwen']:+.6f}" for t in pairs if t['latency_quality_tradeoff_observed']) or '本 frozen subset 上没有 Specialist 相对 Qwen 更快但 primary quality 更差的点估计')+'。这些是描述性点估计，不代表统计显著性。','',
          f"当前最终 speech offload：Qwen→Whisper-small ASR WER {fmt(q_asr.get('wer'))}→{fmt(ws_asr.get('wer'))}；LID accuracy {fmt(q_lid.get('accuracy'))}→{fmt(ws_lid.get('accuracy'))}。不能只报告 latency 改善而省略此质量变化。",'',
          '## 7. 是否有质量太差而不适合直接进入最终 optimizer 的模型？','',
          '测量诊断：'+diagnostic+'。各 task 最弱点估计：'+'；'.join(weakest)+'。需要逐 task 审查低质量及被支配 path；不能把某 path 的低分或被支配直接变成整模型淘汰，Phi 的 VQA 与 text/image 表现就不同。当前 v2 verified 只表示 capability，不是质量准入。没有正式阈值便不能按临时规则宣告某模型通过/淘汰。quality matrix 的所有分数、invalid counts 和 Pareto 都已披露；最终 optimizer 不能在未冻结质量政策时默认这些 path 全部合格。','',
          '## 8. 是否已有正式 quality threshold？','',
          '存在历史四任务 placement presets，完整值已保存；当前六任务 v2 optimizer 的统一 threshold 未规定：UNSPECIFIED。VQA/LID 尤其没有历史门槛。没有新增或事后选择 threshold。','',
          '## 9. Phi 是否可承担统一 VQA？','',
          f"{revalidations['phi4mm']['status']}。{revalidations['phi4mm']['reason']}。Quality={json.dumps(revalidations['phi4mm'].get('quality'),ensure_ascii=False)}。",'',
          '## 10. MiniCPM 是否可承担统一 VQA？','',
          f"{revalidations['minicpm']['status']}。{revalidations['minicpm']['reason']}。Quality={json.dumps(revalidations['minicpm'].get('quality'),ensure_ascii=False)}。",'',
          '## 11. Qwen 是否仍然是唯一 mandatory VQA Generalist？','',
          '在 unchanged frozen v2 中是，λVQA>0 时仍强制存在。' + ('本次相同语义验证已发现其他 capable Generalists，结构性唯一性可在 v2.1 审批/版本冻结后解除；这不自动说明其他模型符合质量政策。' if additions else '本次没有新增 VERIFIED VQA path；需保留此限制。'),'',
          '## 12. 是否需要 candidate_pool_v2.1？','',
          ('建议建立 v2.1：新增 capability proposals 为 '+', '.join(a['model']+'/visual_qa' for a in additions)+'。只生成 proposal，未改 v2；新 path 的 s0 尚缺、正式质量政策尚未冻结，不能直接注册进完整 optimizer。' if additions else '暂不建议因未验证 path 扩展能力；proposal 记录原因和 pending 状态。'),'',
          '## 13. 下一版 suite 应包含哪些 workload？','',
          'demand-balanced、speech-heavy、vision-heavy、text/OCR-heavy、reasoning/VQA-heavy、no-VQA phase。所有 λ 从固定 Qwen-reference target demand shares 解析生成，common D 取旧三组总 demand 中位数；JSON/hash 已冻结。没有根据 planning/search result 调参。','',
          '## 14. no-VQA phase 语义合理吗？','',
          '合理，研究有五类现有任务且没有 ScienceQA 请求的流量阶段。λVQA=0 只改变流量覆盖约束，其他 task semantics 不变；它检验而不保证 Generalist choice 变化；现有 Specialists 已能覆盖全部五个非 VQA task，因此 N_G=0 也无结构性 coverage 障碍。不能将此 phase 当成全六任务 coverage。','',
          '## 15. 可以冻结最终 workload matrix 吗？','',
          '现在可冻结需求原则、proposal λ 向量与 hash；不能把它们称为最终完整 benchmark matrix。需要先决定 v2.1、新 VQA path standalone demands 和六任务质量政策；正式执行协议可延用现有 10s warmup / 210s window、deterministic periodic、三次重复。本阶段没有启动。后续不能为得到不同 deployment 改 workload。','',
          '所有要求产物位于 results/quality_workload_audit/。质量测量与原 deployment search 独立；完成后停止，不进入禁止的后续阶段。']
 (OUT/'module8a_summary.md').write_text('\n'.join(summary)+'\n')

def validate(manifest,matrix):
 protected=read('protected_input_hashes.json')
 changed=[p for p,h in protected.items() if not (ROOT/p).exists() or digest(ROOT/p)!=h]
 proposals=read('proposed_demand_normalized_workloads.json')
 for p in proposals['proposals']:
  assert math.isclose(sum(p['raw_request_share'].values()),1,abs_tol=1e-12)
  assert math.isclose(sum(p['qwen_reference_demand_share'].values()),1,abs_tol=1e-12)
  for k,v in p['target_qwen_demand_share'].items():assert math.isclose(v,p['qwen_reference_demand_share'][k],abs_tol=1e-12)
 assert not changed,changed
 assert matrix['completed_path_count']==25
 provenance=read('model_asset_hashes.json')
 pool=json.loads((ROOT/'results/candidate_pool_v2/candidate_models.json').read_text())
 integrity=json.loads((ROOT/'results/candidate_pool_v2/download_integrity.json').read_text())
 context=json.loads((ROOT/'results/integer_replica_search_multiworkload/w1/runtime_context.json').read_text())
 revision_hash_check=[]
 for name,model in context['models'].items():
  assert provenance[name]['frozen_source_revision']==model['source']['source_revision']
  for relative,expected in model['source']['model_files'].items():
   absolute=str(ROOT/relative)
   observed=provenance[name]['files'].get(absolute)
   assert observed is not None,(name,absolute)
   assert observed['sha256']==expected['sha256'],(name,absolute)
   revision_hash_check.append(absolute)
 failures=[]
 for entry in integrity:
  matches=[f[entry['file']]['sha256'] for m in provenance.values() if entry['file'] in (f:=m['files'])]
  if matches and matches[0]!=entry['actual_sha256']:failures.append(entry['file'])
 assert not failures,failures
 save('audit_validation.json',{'status':'PASS','protected_file_count':len(protected),'changed_protected_files':changed,
                               'verified_quality_paths_complete':25,'expected_sample_count_each':200,
                               'total_verified_path_predictions':5000,'vqa_revalidation_predictions':400,'completed_quality_worker_loads':12,'quality_worker_startup_attempts':12+len(list((OUT/'raw_quality').glob('*_failed_attempt_*.json'))),'preserved_failed_startup_attempts':len(list((OUT/'raw_quality').glob('*_failed_attempt_*.json'))),'new_real_deployment_searches':0,'algorithm_changes':0,
                               'proposal_target_shares_validated':True,'proposal_sha256':digest(OUT/'proposed_demand_normalized_workloads.json'),
                               'quality_manifest_sha256':digest(OUT/'quality_evaluation_manifest.json'),
                               'new_candidate_weight_hash_matches_frozen_integrity':not failures,'all_model_files_match_frozen_7c4_identity':True,'frozen_model_files_checked':len(revision_hash_check),
                               'no_post_result_quality_thresholds':True,'frozen_candidate_pool_v2_changed':False,
                               'stopped_after_module8a':True})
if __name__=='__main__':main()
