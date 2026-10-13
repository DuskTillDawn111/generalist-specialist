"""CPU-only failure diagnosis; no policy/threshold change or GPU rerun."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32c.common import *
from vqa_aware_dynamic_router import VQAAwareRouter
from module26.common import arithmetic
from module25.common import utility_inputs
f=read(OUT/'experiment_manifest.json');raw=read(OUT/'raw_runs/P1_D.json');control=read(OUT/'raw_runs/P1_S.json');ph=raw['phases']['measurement'];dec=ph['routing_decisions'];rs=ph['request_records'];p=read(OUT/'frozen_routing_parameters.json');meta=read(OUT/'audio_metadata_cache.json');trace=read(raw['slot']['trajectories']['measurement']['path'])['events'];blocks=[];pure=[];projection=[];replay=[]
router=VQAAwareRouter(p,meta,2400.)
for d in dec:
 tick=time.perf_counter_ns();v=router.decide(d['decision_perf_ns'],trace[int(d['request_id'].split(':')[1])]['item']['sample']['input'],d['past_state_snapshot']);pure.append((time.perf_counter_ns()-tick)/1e6);replay.append(all(native(v[k])==d[k] for k in v))
for lo in range(0,2400,300):
 ds=[d for d in dec if lo<=(d['decision_perf_ns']-ph['window']['start_time_ns'])/1e9<lo+300]
 blocks.append(dict(start_seconds=lo,end_seconds=lo+300,n=len(ds),observed_routing_compute_ms=stats(d['routing_compute_ms'] for d in ds),mean_prior_history_records=float(np.mean([int(d['request_id'].split(':')[1]) for d in ds]))))
 d=ds[-1];hist=[*raw['path_probe_records'],*raw['warmup']['request_records'],*rs[:int(d['request_id'].split(':')[1])]]
 # Model the original projection's completed-history scan using completed
 # records plus the observed active snapshot. No worker execution is replayed.
 active=[dict(model=G if x['role']=='G' else S,status='submitted' if x['submit_perf_ns'] else 'queued',task=x['task'],actual_submit_time_ns=x['submit_perf_ns'],enqueued_time_ns=x['enqueue_perf_ns'],router_S_service_estimate_ms=x['S_service_estimate_ms']) for x in d['past_state_snapshot']]
 pending={str(i):x for i,x in enumerate(hist+active)};elapsed=[]
 for _ in range(31):
  tick=time.perf_counter_ns();proj=[dict(role='G' if x['model']==G else 'S',task=x['task'],submit_perf_ns=x.get('actual_submit_time_ns'),enqueue_perf_ns=x['enqueued_time_ns'],S_service_estimate_ms=x.get('router_S_service_estimate_ms',max(z['S_p75_ms'] for z in p['cost_bins']))) for x in pending.values() if x['model'] in [G,S] and x['status'] in ('queued','submitted')];elapsed.append((time.perf_counter_ns()-tick)/1e6)
 projection.append(dict(start_seconds=lo,completed_history_records=len(hist),observed_active_states=len(active),projection_CPU_ms=stats(elapsed),scope='CPU-only microprofile of original completed-history scan, not same GPU-run locks/scheduling or counterfactual utility'))
csvsave('router_overhead_by_300s_block.csv',blocks);csvsave('completed_history_projection_CPU_profile.csv',projection)
save('long_window_router_failure.json',dict(frozen_P95_limit_ms=1.,observed_ms=stats(d['routing_compute_ms'] for d in dec),decisions_over_1ms=sum(d['routing_compute_ms']>=1 for d in dec),pure_router_CPU_replay_ms=stats(pure),all_decisions_exactly_replayed=all(replay),whole_wrapper_timing='lock acquisition + full runtime.pending history scan + router.decide + enqueue, excluding initial pressure safety scan; not pure decide alone',source_mechanism='DeploymentRuntime.pending retains completed request records; original32B phase projection scans all dictionary values at each ASR arrival. History grows throughout2400s, even when real outstanding state is small.',source_files=[ref(ROOT/'joint_deploy/measurement/deployment_runtime.py'),ref(OUT/'m32c/runtime.py')],GPU_reexecution=False,parameters_unchanged=True,remaining_groups='P2/P3 were not launched under pre-frozen STOP_ENTIRE_CAMPAIGN_NO_RETRY technical failure policy',causal_wall_time_decomposition_not_directly_instrumented=True))
D,refs=utility_inputs();catalog=read(OUT/'selected_pool_catalog.json');ar={k:arithmetic(r,catalog,refs,D['weights']) for k,r in [('S',control),('D',raw)]};rows=[];rates=raw['slot']['lambda_rps'];total=sum(rates.values())
for t,q in rates.items():
 b=ar['S']['per_task'][t];a=ar['D']['per_task'][t];lu=-D['weights']['L']*q/total*(a['mean_ms']-b['mean_ms'])/refs[t];pu=-D['weights']['P']*q/total*(a['p95_ms']-b['p95_ms'])/refs[t]
 rows.append(dict(pair=1,task=t,measured_mean_change_ms=a['mean_ms']-b['mean_ms'],measured_P95_change_ms=a['p95_ms']-b['p95_ms'],DIAGNOSTIC_L_contribution=lu,DIAGNOSTIC_P_contribution=pu,DIAGNOSTIC_L_P_contribution=lu+pu,valid_legal_delta_U=None,official_U_Dynamic=None,invalid_for_replication=True))
diagnostic=ar['D']['raw_arithmetic_U']-ar['S']['raw_arithmetic_U'];text=next(x for x in rows if x['task']=='text');save('ineligible_full_cohort_arithmetic_diagnostic.json',dict(official_U_S=read(OUT/'formal_acceptance/P1_S.json')['U'],official_U_Dynamic=None,official_pair_delta_U=None,full_cohort_raw_arithmetic=ar,DIAGNOSTIC_arithmetic_difference=diagnostic,DIAGNOSTIC_Text_P95_contribution=text['DIAGNOSTIC_P_contribution'],DIAGNOSTIC_without_Text_P95=diagnostic-text['DIAGNOSTIC_P_contribution'],not_a_legal_comparison=True,no_truncated_requests=True,not_an_executed_counterfactual=True,reason='Dynamic frozen router P95 overhead acceptance failed; average GPU temperature diagnostic differs7.006C; no valid replication delta/CI'))
csvsave('ineligible_task_arithmetic_diagnostic.csv',rows)
slots=[]
for s in f['slots']:
 attempted=(OUT/'run_attempts'/(s['run_id']+'.json')).exists();slots.append(dict(pair=s['pair'],run_id=s['run_id'],seed=s['seed'],planned_seconds=2400.,status='MEASURED_FULL_COHORT' if attempted else 'NOT_EXECUTED',reason=None if attempted else 'Pre-frozen stop on P1_D technical invalidity; no replacement seed or threshold change',trajectory=s['trajectories']['measurement']))
save('planned_run_disposition.json',slots)
lines=['# 长窗口路由开销与停止诊断','', 'P1_S与P1_D均完成完整2400s，分别3419/3419请求，均steady、原生时钟/journal/质量/显存检查通过。Dynamic真实执行10/1070个Phi ASR，实际份额0.934579%。','',f'但原32B在线wrapper路由开销P95={np.percentile([d["routing_compute_ms"] for d in dec],95):.6f}ms，超过原冻结1ms；原validation返回False并触发预定技术错误停止，正式Dynamic U=NULL，正式ΔU=NULL。该阈值来自原32B m32b/campaign.py，不是事后新增停止标准；32C完整复用了它。','', '不能只把纯router.decide的小CPU开销拿来替换GPU期间的原完整wrapper计时以挽救验收。完整计时包含取锁、投影、决定与enqueue；原计时和原源码保持不变。','', '|时间段s|ASR决策数|wrapper mean ms|wrapper P95 ms|','|---|---:|---:|---:|']
for b in blocks:lines.append(f'|{b["start_seconds"]}–{b["end_seconds"]}|{b["n"]}|{b["observed_routing_compute_ms"]["mean"]:.6f}|{b["observed_routing_compute_ms"]["p95"]:.6f}|')
lines+=['', '原DeploymentRuntime.pending是累积记录表，完成记录未删除；原state projection每次扫描所有历史记录后过滤active。实际决策投影很小，但扫描规模随窗口增长。源代码与逐段开销增长支持这一宿主集成扩展问题；独立CPU微画像另列，其纯CPU时间不含原GPU运行的锁/调度/内存状态，不能精确分摊原壁钟开销。纯decide重放保持全部决定一致，不能说明整个在线实现已通过。','', '组内平均GPU温度S=43.735°C、D=50.741°C，差7.006°C，原5°C环境诊断也未通过；SM均值差仅0.282%。新Phi工作、热历史或其他宿主漂移没有被因果隔离，不能把所有服务变化唯一归为分流。没有外来GPU进程或异常/截尾请求。','', 'P2/P3新种子与轨迹均仍保留为NOT_EXECUTED；没有生成假测量、换seed、改阈值或GPU补跑。不存在三组有效ΔU，不能报告预定均值或置信区间，也不能标记独立复现成功。','', f'仅完整cohort代数诊断差={diagnostic:+.9f}，Text P95诊断贡献={text["DIAGNOSTIC_P_contribution"]:+.9f}；正式U/ΔU仍为NULL，以上不用于Gate或成功判定。独立首组Text P95从58.638增至62.398ms，原开发Text尾改善没有在这一已测轨迹上出现。','', '本轮不修原冻结wrapper、不修改正式优化器。如果后续另行研究，应先以维护有界active状态索引的办法解决历史扫描扩展问题并单独冻结/验收，而不是放宽1ms阈值；之后再设计新的独立配对及环境控制。这只是研究建议，本轮未实现/未执行。']
(OUT/'long_window_failure_diagnosis.md').write_text('\n'.join(lines)+'\n')
hist=read(OUT/'historical_utility_recalculation.json');s0=hist['A0']['per_task']['text'];d0=hist['A2']['per_task']['text'];note=f'\n## 原Text mean口径补充\n\n原Dynamic Text mean由{s0["mean_ms"]:.6f}增至{d0["mean_ms"]:.6f}ms（变化{d0["mean_ms"]-s0["mean_ms"]:+.6f}ms），mean实际上恶化；正贡献来自P95下降，不能把它称作mean/P95共同改善。\n'
for n in ['utility_attribution.md','text_tail_audit.md']:
 with (OUT/n).open('a') as h:h.write(note)
print('CPU_FAILURE_DIAGNOSIS',stats(d['routing_compute_ms'] for d in dec),'RAW_ONLY_DIAGNOSTIC',diagnostic,'NO_FORMAL_DELTA',flush=True)
