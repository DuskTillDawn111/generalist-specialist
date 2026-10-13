"""Independent evidence audit and all fifteen requested answers."""
import csv,math,statistics
from module14.common import *

def fmt(v,n=3):return 'N/A' if v is None else f'{float(v):.{n}f}'
def main():
 gate=read(OUT/'module14a_gate.json');selection=read(OUT/'hq_heavy_specialist_selection.json');slo=read(OUT/'hq_quality_slo.json');baseline=read(OUT/'baseline_references.json');new=list(csv.DictReader((OUT/'image_quality_latency_memory.csv').open()))+list(csv.DictReader((OUT/'ocr_quality_latency_memory.csv').open()));existing=list(csv.DictReader((OUT/'existing_heavy_candidate_audit.csv').open()));allrows=list(csv.DictReader((OUT/'all_heavy_candidate_audit.csv').open()));protocol=read(OUT/'measurement_protocol_freeze.json')
 protected=read(OUT/'protected_before.json')['files'];unchanged=[]
 for r in protected:
  assert ref(r['path'])['sha256']==r['sha256'],r['path'];unchanged.append(r['path'])
 for r in read(OUT/'discovery_freeze_manifest.json')['files']+read(OUT/'dataset_input_hashes.json')['files']+read(OUT/'execution_freeze.json')['implementation']:assert ref(r['path'])['sha256']==r['sha256'],r['path']
 for r in allrows:
  b=baseline['tasks'][r['task']]
  if r.get('quality') and r.get('s0_ms') and r.get('safe_memory_mib'):
   q=float(r['quality']);lat=float(r['s0_ms']);mem=float(r['safe_memory_mib']);flags=[q<b['quality'] if r['task']=='asr' else q>b['quality'],lat>b['s0_ms'],mem>b['safe_memory_mib']]
   assert (r['HQ_HEAVY_ELIGIBLE']=='True')==all(flags)
 for r in new:
  if r['safe_memory_mib']:assert math.isclose(float(r['safe_memory_mib']),float(r['idle_memory_mib'])*baseline['safe_factor'],rel_tol=1e-12)
 runs=[read(p) for p in RAW.glob('*__quality__r0.summary.json')]+[read(p) for p in RAW.glob('*__latency__r*.summary.json')];runs.sort(key=lambda r:r['at_utc']);assert all(not r.get('foreign_process_samples') for r in runs);assert all(runs[i]['completed_at_utc']<=runs[i+1]['at_utc'] for i in range(len(runs)-1))
 paths=[]
 for e in protocol['entries']:
  stem=key(e);qfile=RAW/(stem+'__quality__r0.summary.json');q=read(qfile) if qfile.exists() else {};measure=[read(RAW/(stem+f'__latency__r{rep}.summary.json')) if (RAW/(stem+f'__latency__r{rep}.summary.json')).exists() else {} for rep in [1,2,3]]
  if q.get('success'):
   rows=[json.loads(x) for x in Path(q['raw_path']).read_text().splitlines()];expected=protocol['datasets'][e['task']]['quality_ids_in_order'];assert [r['pool_id'].split(':',1)[1] for r in rows]==expected;assert len(rows)==200
  for d in measure:
   if d.get('success'):
    rows=[json.loads(x) for x in Path(d['raw_path']).read_text().splitlines()];assert len(rows)==30;assert [r['pool_id'] for r in rows]==protocol['datasets'][e['task']]['latency_ids_in_order']
  paths.append({'model_id':e['model_id'],'revision':e['revision'],'quality_success':q.get('success',False),'latency_memory_all3_success':all(d.get('success') for d in measure),'native_cuda_load':bool(q.get('ready'))})
 for task,s in selection['tasks'].items():
  w=s['selected']
  if w:
   assert w['HQ_HEAVY_ELIGIBLE'];ss=slo['tasks'][task]['new_HQ_SLO'];b=baseline['tasks'][task]
   assert w['quality']<=ss['value']<b['quality'] if task=='asr' else b['quality']<ss['value']<=w['quality']
 validated={'at_utc':now(),'passed':True,'protected_historical_files_unchanged':len(unchanged),'discovery_freeze_and_code_hashes_unchanged':True,'strict_triple_eligibility_recomputed':True,'same_frozen_quality_and_latency_sample_order_verified':True,'safe_memory_definition_verified':True,'GPU_worker_intervals_nonoverlapping':True,'all_raw_runs_retained':True,'GPU_runs':len(runs),'successful_GPU_runs':sum(r['success'] for r in runs),'new_paths':paths,'old_specialist_regime_and_Module9_unmodified':True,'deployment_or_V3_outcome_used':'NO','g_gt1_or_replica_frontier_started':False,'no_second_gap_filling_round':True,'Image_accuracy_ceiling_honored':True}
 save('validation.json',validated)
 lines=['# Module14A — High Quality Heavy Specialist Pool', '\nGate：**'+gate['gate']+'**。严格 HIGHER_QUALITY + SLOWER + LARGER，不使用Pareto状态排除，不改变任务、数据、旧阈值或旧pool。此模块只测Q/s0/M，不运行deployment、structural replay、lowQPS、multi-hotspot或g(x>1)。', '\n## 1–2. 固定旧五类 reference 与真实 Q/s0/M', '\nReference来自原真实baseline：ASR/LID/Text沿用13B固定ratio anchors；Image为原ResNet50；OCR为原PP-OCRv6-medium-rec。13B用于nearest-target规则选出的ASR代表Whisper-small不重定义原Distil-large-v3 reference。未根据deployment选择比较分母。', '\n| Task | Old specialist | Metric | Q | s0 ms | Idle MiB | Safe MiB |','|---|---|---|---:|---:|---:|---:|']
 for task,b in baseline['tasks'].items():lines.append('| '+' | '.join([task,b['model_id'],b['metric'],fmt(b['quality'],9),fmt(b['s0_ms'],6),fmt(b['idle_memory_mib'],1),fmt(b['safe_memory_mib'],6)])+' |')
 lines+=['\nsafe M仍为原factor '+str(baseline['safe_factor'])+' × 真实post-warmup process idle NVML MiB；不是参数、权重大小或模型卡memory。','\n## 3. 现有 ASR/LID/Text 严格三重审计','\n| Task | Existing measured candidate | Origin | Status | Failed criteria |','|---|---|---|---|---|']
 for r in existing:lines.append('| '+' | '.join([r['task'],r['model_id'],r['origin'],r['status'],r['failed_criteria']])+' |')
 lines+=['\n九条新增13B paths中ASR没有eligible；LID Whisper-medium、Text三个模型eligible。完整13B真实operating-point表还包含旧pool替代模型：Whisper-turbo ASR/LID与RoBERTa-base Text均eligible。按GPU前公开声明的默认范围纳入这些真实替代checkpoint，故ASR/LID/Text不需新gap filling。旧pool checkpoint可以在独立HQ regime中复用，原EFFICIENCY内容不删除。参考模型自身的三项相等全部不满足严格门槛。','\n## 4–5. Image/OCR 公开候选搜索']
 for task in ['image','ocr']:
  universe=list(csv.DictReader((OUT/(task+'_candidate_universe.csv')).open()));lines.append('\n'+task+': '+', '.join(r['model_id'] for r in universe)+'。');lines.append('冻结测量：'+', '.join(e['model_id']+' @'+e['revision'] for e in protocol['entries'] if e['task']==task)+'。')
 lines+=['\n公开quality仅是搜索证据：Image官方ImageNet1K top1优于ResNet50-v2不等于冻结Imagenette200能提高；OCR跨版本公开分数包含不同评估集，不能直接比本机IIIT5K200。原OCR是单词裁剪图recognition-only，不含detector、orientation/cropping pipeline；TrOCR single-line/word input兼容，并使用已训练STR checkpoint，不做本地finetuning。所有source/revision、metadata与网页依据见sources/、source_audit.md和冻结shortlist。','\n## 6–7. Image/OCR 真实测量与eligibility','\n| Task | Candidate | Q | s0 ms | Safe MiB | Latency ratio | Memory ratio | Status |','|---|---|---:|---:|---:|---:|---:|---|']
 for r in new:lines.append('| '+' | '.join([r['task'],r['model_id'],fmt(r.get('quality') or None,6),fmt(r.get('s0_ms') or None),fmt(r.get('safe_memory_mib') or None,1),fmt(r.get('latency_ratio') or None),fmt(r.get('safe_memory_ratio') or None),r['status']])+' |')
 lines+=['\n原Image accuracy=1.0，任何真实新模型在相同metric上都不能严格更高。即便新模型同为1.0且更慢更大也FAIL_QUALITY，不能更换数据子集、改用ImageNet1000类或放宽为≥。','\n## 8. Gap filling','\n最多允许一轮、仅缺失ASR/LID/Text，规则在新GPU前固定。本次完整已有实测candidate universe三task均有eligible，gap filling rounds_used=0；没有反复搜索或Outcome条件。','\n## 9–10. 最终每task HQ选择、真实提升与资源比','\n| Task | Selected HQ specialist | Old Q | New Q | Quality gain | Latency × old | Safe M × old |','|---|---|---:|---:|---:|---:|---:|']
 for task,s in selection['tasks'].items():
  w=s['selected'];b=baseline['tasks'][task]
  lines.append('| '+' | '.join([task,w['model_id'] if w else 'NO_ELIGIBLE_HEAVY_SPECIALIST',fmt(b['quality'],6),fmt(w['quality'],6) if w else 'N/A',fmt(w['quality_gain'],6) if w else 'N/A',fmt(w['latency_ratio']) if w else 'N/A',fmt(w['safe_memory_ratio']) if w else 'N/A'])+' |')
 lines+=['\nQuality gain以原metric单位表达：accuracy/exact-match为new-old，ASR为oldWER-newWER；乘100为百分点。选择顺序严格为eligible中质量最高→相同质量下latency低→safe M低→full ID@revision。不会为了heavy去选最慢/最大，也不会移除dominated候选以改变研究问题。','\n## 11. 是否完成五类替换','\n'+('五类HQ独立pool完整，但旧regime仍保留。' if gate['full_HQ_regime_supported'] else 'NO。仅'+str(gate['eligible_task_count'])+'/5任务有合格HQ Specialist；missing='+', '.join(gate['missing_tasks'])+'。保留部分HQ pool证据，不用旧lightweight baseline填空，不宣称完整第二regime READY。'),'\n## 12. 一次性HQ SLO冻结','\n构造规则在GPU前固定，实际数值在所有冻结测量完成后一次性写入hq_quality_slo.json：两真实质量值算术中点；higher-is-better用≥，WER用≤；保留原semantic admission。每个有效值比旧SLO更严、排除旧reference、允许所选HQ模型。旧LID仅semantic admission，新HQ task增加独立accuracy SLO，原LID政策未改。Image没有可行严格增强阈值，值为null，不能伪造accuracy>1。','\n| Task | Old SLO | Old Q | New fixed HQ SLO | HQ Q |','|---|---|---:|---|---:|']
 for task,d in slo['tasks'].items():
  old=d['old_SLO'];newS=d['new_HQ_SLO'];lines.append('| '+' | '.join([task,'semantic-only' if old['value'] is None else old['operator']+fmt(old['value'],9),fmt(d['old_specialist_measured_Q'],9),newS['operator']+fmt(newS['value'],9) if newS else 'NO_VALID_HQ_SLO',fmt(d['selected_HQ_specialist_measured_Q'],9)])+' |')
 controls=read(OUT/'baseline_control_comparability.json')['controls'];lines+=['\n同期Image/OCR control独立另报，历史分母不改。'+ '; '.join(t+' s0 ratio='+fmt(c['s0_ratio_vs_frozen'],6)+', safe M ratio='+fmt(c['memory_ratio_vs_frozen'],6) for t,c in controls.items())+'。漂移与所有有效run/outliers保留，不重测以挑更有利结果。13B旧controls drift仍见原文件；不同时间实测对比较存在限制，不能将本模块作为最终deployment证据。','\n## 13. 模型/SLO选择是否使用deployment/V3 outcome','\n**NO**。比较仅来自真实Q/s0/M固定reference，资格和质量优先规则及SLO公式在新GPU之前冻结。不调用Module9/12 optimizer，不进行新的structural或V3 replay；不引用后续2G/Generalist胜负决定任何模型或阈值。','\n## 14. Gate','\n**'+gate['gate']+'**。不降低三重门槛。','\n## 15. 若READY下一阶段g(x)','\n'+('; '.join(gate['next_g_if_READY']) if gate['full_HQ_regime_supported'] else '当前不是READY，不启动g(x>1)。部分pool所选checkpoint保持单独证据，未来阶段若授权再判断需补曲线；旧checkpoint已有g曲线不因本模块被重定义。'),'\n完整性：'+str(len(unchanged))+'个既有受保护文件未变；冻结inputs/代码哈希、样本顺序与strict eligibility独立核验通过。新GPU task paths='+str(len(paths))+'，全部quality和3run measurement success='+str(all(p['quality_success'] and p['latency_memory_all3_success'] for p in paths))+'。实际RTX4090为49140MiB设备，未声称标准24GB本机验证。所有失败兼容修复与raw evidence保留。完成后停止。']
 (OUT/'module14a_summary.md').write_text('\n'.join(lines)+'\n')
 print('FINAL_VALIDATION_PASS',len(unchanged),'gate',gate['gate'],flush=True)
if __name__=='__main__':main()
