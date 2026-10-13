import sys,itertools,time,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32b.common import *
from module20.optimizer import components
from module25.common import utility_inputs
D,refs=utility_inputs();manifest=read(OLD/'experiment_manifest.json');rates=manifest['slots'][0]['lambda_rps'];total=sum(rates.values());raws={a:read(OLD/'raw_runs'/f'{a}.json') for a in ['A0','A1','A2']};obs={o['run_id']:o for o in read(OLD/'run_observations.json')};base={r['request_id']:r for r in raws['A0']['phases']['measurement']['request_records']};allrows=[];vq=[];div=[];gpu=[];clock=[];loss=[];infl=[];summary={};tails=[]
# Read every raw, journal, and text log, including historical failed/archived windows.
inventory=[]
for p in sorted(OLD.rglob('*.json')):
 if p.parent.name!='raw_runs':continue
 r=read(p);phases=[r.get('warmup')]+list(r.get('phases',{}).values())+[r.get('capacity_probe'),r.get('aborted_phase')];rr=[x for ph in phases if ph for x in ph.get('request_records',[])];tt=r.get('telemetry',{}).get('samples',[])
 inventory.append(dict(source=ref(p),status=r.get('status'),requests_in_all_phases=len(rr),telemetry_samples=len(tt),worker_identities=r.get('worker_identities'),errors=r.get('errors'),stability={k:v.get('stability') for k,v in r.get('phases',{}).items()}))
 for phase in phases:
  if not phase:continue
  for x in phase.get('request_records',[]):
   w=x.get('worker_response',{});allrows.append(dict(source=str(p.relative_to(OLD)),run_id=r['slot']['run_id'],phase=x.get('phase'),request_id=x['request_id'],task=x['task'],worker=x['model'],status=x['status'],sample_id=x.get('sample_id'),scheduled_perf_ns=x.get('scheduled_time_ns'),submit_perf_ns=x.get('actual_submit_time_ns'),completion_perf_ns=x.get('completion_time_ns'),submit_unix_ns=x.get('actual_submit_unix_ns'),completion_unix_ns=x.get('completion_unix_ns'),worker_start_unix_ns=w.get('start_ns'),worker_end_unix_ns=w.get('end_ns'),service_ms=x.get('worker_latency_ms'),queue_ms=x.get('worker_queue_wait_ms'),e2e_ms=x.get('e2e_latency_ms')))
 gpu.extend(dict(source=str(p.relative_to(OLD)),run_id=r['slot']['run_id'],**s) for s in tt)
journals=[]
for p in sorted(OLD.rglob('*.jsonl')):
 if p.parent.name!='request_journals':continue
 kinds=collections.Counter();n=0
 with p.open() as f:
  for line in f:kinds[readline_kind:=__import__('json').loads(line)['kind']]+=1;n+=1
 journals.append(dict(source=ref(p),lines=n,kinds=dict(kinds)))
logs=[]
for p in sorted(OLD.rglob('*.log')):
 txt=p.read_text(errors='replace');logs.append(dict(source=ref(p),lines=len(txt.splitlines()),warnings=sum('warning' in l.lower() for l in txt.splitlines()),traceback_mentions=txt.count('Traceback')))
save('module32_complete_log_inventory.json',dict(raw=inventory,journals=journals,text_logs=logs,archives_retained=True));csvsave('all_historical_request_timings.csv',allrows);csvsave('all_historical_GPU_CPU_telemetry.csv',gpu)
for a,r in raws.items():
 rr=r['phases']['measurement']['request_records'];ga=[x for x in rr if x['model']==G and x['task']=='asr'];gv=[x for x in rr if x['model']==G and x['task']=='visual_qa'];offsets=[];tail=sorted(gv,key=lambda x:x['e2e_latency_ms'],reverse=True)[:math.ceil(len(gv)*.05)]
 for x in rr:
  w=x['worker_response'];assert abs((w['end_ns']-w['start_ns'])/1e6-x['worker_latency_ms'])<1e-6;assert abs((w['start_ns']-w['submit_ns'])/1e6-x['worker_queue_wait_ms'])<1e-6;assert w['submit_ns']==x['actual_submit_unix_ns'];assert w['end_ns']<=x['completion_unix_ns'];assert w['start_ns']>=w['submit_ns']
  offsets.extend([x['actual_submit_unix_ns']-x['actual_submit_time_ns'],x['completion_unix_ns']-x['completion_time_ns']])
 clock.append(dict(arm=a,records=len(rr),same_host_unix=True,native_formula_passed=True,unix_perf_offset_range_ms=(max(offsets)-min(offsets))/1e6,Phi_intertask_native_overlap_pairs=sum(overlap(interval(x),interval(y))>0 for x in ga for y in gv)))
 for x in gv:
  w=x['worker_response'];wait=(w['submit_ns'],w['start_ns']);aas=[(y,overlap(wait,interval(y))) for y in ga];vv=[(y,overlap(wait,interval(y))) for y in gv if y is not x];asm=sum(z for y,z in aas);vwm=sum(z for y,z in vv);q=x['worker_queue_wait_ms'];act=[y for y in ga if interval(y)[0]<=w['submit_ns']<interval(y)[1]];b=base[x['request_id']]
  rec=dict(arm=a,request_id=x['request_id'],sample_id=x['sample_id'],arrival_offset_seconds=x['frozen_offset_seconds'],service_ms=x['worker_latency_ms'],queue_ms=q,e2e_ms=x['e2e_latency_ms'],service_delta_ms=x['worker_latency_ms']-b['worker_latency_ms'],queue_delta_ms=q-b['worker_queue_wait_ms'],e2e_delta_ms=x['e2e_latency_ms']-b['e2e_latency_ms'],ASR_handler_overlap_in_wait_ms=asm,VQA_handler_overlap_in_wait_ms=vwm,uncovered_wait_ms=q-asm-vwm,ASR_active_at_submit=bool(act),blocking_ASR_ids=[y['request_id'] for y,z in aas if z>0],is_E2E_top5percent=x in tail)
  vq.append(rec)
  if x in tail:tails.append(rec)
 for x in ga:
  now=x['actual_submit_unix_ns'];scheduled=now-(x['actual_submit_time_ns']-x['scheduled_time_ns']);pending=[y for y in rr if y is not x and y['model']==G and y['actual_submit_unix_ns']<now<y['completion_unix_ns']];active=[y for y in pending if interval(y)[0]<=now<interval(y)[1]];phiV=[y for y in pending if y['task']=='visual_qa'];pendingS=[y for y in rr if y['model']==S and y['actual_submit_unix_ns']<now<y['completion_unix_ns']];futureV=[y for y in gv if now<y['actual_submit_unix_ns']<x['worker_response']['end_ns']];beforeV=[y for y in gv if y['actual_submit_unix_ns']<=now];lastV=max((y['actual_submit_unix_ns'] for y in beforeV),default=None)
  div.append(dict(arm=a,request_id=x['request_id'],sample_id=x['sample_id'],arrival_offset_seconds=x['frozen_offset_seconds'],G_service_ms=x['worker_latency_ms'],G_queue_ms=x['worker_queue_wait_ms'],same_input_S_service_ms=base[x['request_id']]['worker_latency_ms'],same_input_S_queue_ms=base[x['request_id']]['worker_queue_wait_ms'],Phi_observable_inflight=len(pending),Phi_observable_VQA_inflight=len(phiV),Phi_native_active_task=active[0]['task'] if active else None,Phi_native_queued=len(pending)-len(active),Whisper_observable_inflight=len(pendingS),VQA_elapsed_since_last_arrival_seconds=(now-lastV)/1e9 if lastV else None,VQA_arrivals_during_ASR=futureV and len(futureV) or 0,future_VQA_ids=[y['request_id'] for y in futureV],future_information_policy_forbidden=True))
 for x in rr:
  if x['model']==G:continue
  b=base[x['request_id']]
  if b['model']!=x['model']:continue
  ol=sum(overlap(interval(x),interval(y)) for y in ga);infl.append(dict(arm=a,task=x['task'],worker=x['model'],request_id=x['request_id'],sample_id=x['sample_id'],service_ms=x['worker_latency_ms'],paired_base_service_ms=b['worker_latency_ms'],service_delta_ms=x['worker_latency_ms']-b['worker_latency_ms'],service_ratio=x['worker_latency_ms']/b['worker_latency_ms'],Phi_ASR_native_overlap_ms=ol,has_Phi_ASR_overlap=ol>0,overlap_is_wall_handler_not_kernel=True))
 # Mean contributions and algebraic Shapley P95 decomposition (no executed counterfactual).
 for t,q in rates.items():
  xs=[x for x in rr if x['task']==t];bs=[base[x['request_id']] for x in xs];mean=lambda key,ys:float(np.mean([x[key] for x in ys]));p95=lambda key,ys:float(np.percentile([x[key] for x in ys],95));coefL=-D['weights']['L']*q/total/refs[t];coefP=-D['weights']['P']*q/total/refs[t];md=mean('e2e_latency_ms',xs)-mean('e2e_latency_ms',bs);pd=p95('e2e_latency_ms',xs)-p95('e2e_latency_ms',bs);sv=mean('worker_latency_ms',xs)-mean('worker_latency_ms',bs);qw=mean('worker_queue_wait_ms',xs)-mean('worker_queue_wait_ms',bs);parts={k:0. for k in ['service','queue','host_residual']};bvec=np.array([x['e2e_latency_ms'] for x in bs]);deltas={'service':np.array([x['worker_latency_ms']-b['worker_latency_ms'] for x,b in zip(xs,bs)]),'queue':np.array([x['worker_queue_wait_ms']-b['worker_queue_wait_ms'] for x,b in zip(xs,bs)])};deltas['host_residual']=np.array([x['e2e_latency_ms']-b['e2e_latency_ms'] for x,b in zip(xs,bs)])-deltas['service']-deltas['queue']
  for perm in itertools.permutations(parts):
   v=bvec.copy()
   for k in perm:
    old=float(np.percentile(v,95));v+=deltas[k];parts[k]+=(float(np.percentile(v,95))-old)/6
  assert abs(sum(parts.values())-pd)<1e-8
  loss.append(dict(arm=a,task=t,n=len(xs),lambda_qps=q,reference_ms=refs[t],mean_E2E_ms=mean('e2e_latency_ms',xs),p95_E2E_ms=p95('e2e_latency_ms',xs),mean_service_ms=mean('worker_latency_ms',xs),p95_service_ms=p95('worker_latency_ms',xs),mean_queue_ms=mean('worker_queue_wait_ms',xs),p95_queue_ms=p95('worker_queue_wait_ms',xs),mean_delta_ms=md,p95_delta_ms=pd,L_U_delta=coefL*md,P_U_delta=coefP*pd,L_P_U_delta=coefL*md+coefP*pd,mean_service_U_delta=coefL*sv,mean_queue_U_delta=coefL*qw,mean_host_U_delta=coefL*(md-sv-qw),P95_service_ShAP_ms=parts['service'],P95_queue_ShAP_ms=parts['queue'],P95_host_ShAP_ms=parts['host_residual'],P95_service_U_delta=coefP*parts['service'],P95_queue_U_delta=coefP*parts['queue'],P95_host_U_delta=coefP*parts['host_residual'],Shapley_scope='Algebraic attribution, not simulated/executed routing counterfactual'))
 vv=[x for x in vq if x['arm']==a];dd=[x for x in div if x['arm']==a];svq=sum(x['ASR_handler_overlap_in_wait_ms'] for x in vv);sumwait=sum(x['queue_ms'] for x in vv);nativeG=[x for x in rr if x['model']==G]
 summary[a]=dict(U=obs[a]['U'],delta_U=obs[a]['U']-obs['A0']['U'],VQA_n=len(gv),Phi_ASR_n=len(ga),VQA_ASR_blocked_n=sum(x['ASR_handler_overlap_in_wait_ms']>0 for x in vv),VQA_total_wait_ms=sumwait,VQA_wait_covered_by_ASR_ms=svq,VQA_wait_ASR_coverage=svq/sumwait if sumwait else 0,Phi_native_serial_verified=all(interval(x)[1]<=interval(y)[0] for x,y in zip(sorted(nativeG,key=lambda r:interval(r)[0]),sorted(nativeG,key=lambda r:interval(r)[0])[1:])),ASR_to_busy_Phi=sum(x['Phi_observable_inflight']>0 for x in dd),ASR_to_Phi_with_VQA_pending=sum(x['Phi_observable_VQA_inflight']>0 for x in dd),ASR_to_idle_Phi_with_future_VQA=sum(x['Phi_observable_inflight']==0 and x['VQA_arrivals_during_ASR']>0 for x in dd),VQA_top_tail_ASR_blocked=sum(x['is_E2E_top5percent'] and x['ASR_handler_overlap_in_wait_ms']>0 for x in vv),VQA_stats={k:stats(x[k] for x in vv) for k in ['service_ms','queue_ms','e2e_ms']})
 forcomp=sum(x['L_P_U_delta'] for x in loss if x['arm']==a);tm=sum(obs[a]['components'][k+'_contribution']-obs['A0']['components'][k+'_contribution'] for k in ['T','M']);assert abs(forcomp+tm-summary[a]['delta_U'])<1e-10
 summary[a]['T_M_U_delta']=tm
csvsave('per_task_utility_decomposition.csv',loss);csvsave('phi_execution_overlap_analysis.csv',vq);csvsave('Phi_ASR_arrival_state.csv',div);csvsave('cross_worker_service_overlap.csv',infl);csvsave('VQA_tail_requests.csv',tails);save('clock_and_cuda_timing_audit.json',dict(clock=clock,Phi_handler_synchronized=True,pure_kernel_time_available=False,source_refs=[ref(ROOT/x) for x in ['joint_deploy/worker_protocol.py','joint_deploy/measurement/deployment_runtime.py','joint_deploy/measurement/metrics.py','joint_deploy/phi_worker.py','candidate_pool_v2_1_adapters/vqa_extension.py','benchmark/scripts/eval_phi4_imagenette.py','benchmark/scripts/eval_phi4_librispeech.py']],cross_worker_intervals='Unix wall timestamps on the same machine. Never subtract worker Unix directly from host perf_counter. Alignment spread checked.',limitation='No CUDA events, per-kernel/context tracing or controlled causal interventions; wall overlap is not GPU kernel overlap. Native queue includes serial-worker input pipe backlog, not solely host Queue.'))
save('attribution_summary.json',summary);save('frozen_utility_context.json',dict(definition=read(ROOT/'results/module20_utility_optimizer/utility_definition.json'),refs=refs,rates=rates,Module20A_source=ref(ROOT/'module20/optimizer.py')))
lines=['# Module 32B：VQA 干扰归因','', 'CPU-only；读取Module32全部原始raw/journal/遥测/文本日志（含归档失败），正式逐请求比较仅A0/A1/A2完整匹配600秒cohort。归档重复数据不重复加入估计。未执行反事实路由，未将代数时延替换视作真实收益。','', '## 时钟与CUDA','', 'E2E=(host perf completion−scheduled)，worker服务=(Unix end−start)，worker等待=(Unix start−host Unix submit)。跨worker区间仅同机Unix口径。Unix/perf偏移散布及native公式逐请求核验见clock_and_cuda_timing_audit.json。Phi原ASR/VQA生成在开始、输入转GPU后、generate后显式cuda synchronize；服务是包含I/O、处理、GPU及同步的handler墙钟时间，不是CUDA event的纯kernel时间。Phi协议同步串行，VQA无法抢占在执行的ASR。','', '## VQA排队与执行','', '|臂|VQA数|服务mean/P95 ms|等待mean/P95 ms|E2E mean/P95 ms|等待与Phi ASR执行区间覆盖|VQA被ASR阻塞数|','|---|---:|---|---|---|---:|---:|']
for a,z in summary.items():
 s=z['VQA_stats'];lines.append(f"|{a}|{z['VQA_n']}|{s['service_ms']['mean']:.3f}/{s['service_ms']['p95']:.3f}|{s['queue_ms']['mean']:.3f}/{s['queue_ms']['p95']:.3f}|{s['e2e_ms']['mean']:.3f}/{s['e2e_ms']['p95']:.3f}|{z['VQA_wait_ASR_coverage']:.2%}|{z['VQA_ASR_blocked_n']}|")
lines+=['','逐请求等待区间与同Phi ASR handler的交集是直接日志证据；同Phi ASR/VQA handler无并行，不能称GPU内两个任务同时执行。VQA未来到达也会被已接纳的ASR挡住，当前空队列不能保证背景安全。','', '## 完整系统效用损失','', '|臂|任务|mean贡献ΔU|P95贡献ΔU|合计ΔU|','|---|---|---:|---:|---:|']
for x in loss:
 if x['arm']!='A0':lines.append(f"|{x['arm']}|{x['task']}|{x['L_U_delta']:+.6f}|{x['P_U_delta']:+.6f}|{x['L_P_U_delta']:+.6f}|")
for a in ['A1','A2']:
 x=next(x for x in loss if x['arm']==a and x['task']=='visual_qa');z=summary[a];lines+=['',f"{a}系统ΔU={z['delta_U']:+.6f}，T/M贡献变化={z['T_M_U_delta']:+.6f}。VQA P95差={x['p95_delta_ms']:.3f}ms；对配对E2E向量的三项Shapley代数分解：排队{x['P95_queue_ShAP_ms']:.3f}ms，服务{x['P95_service_ShAP_ms']:.3f}ms，host残差{x['P95_host_ShAP_ms']:.3f}ms。三项相加等于真实P95差；不得相加各自P95冒充E2E P95。",f"Phi ASR={z['Phi_ASR_n']}，路由到忙Phi={z['ASR_to_busy_Phi']}，已有VQA在途={z['ASR_to_Phi_with_VQA_pending']}；路由时Phi空闲但ASR期间新VQA到达={z['ASR_to_idle_Phi_with_future_VQA']}。最高5% VQA尾样本中被ASR阻塞={z['VQA_top_tail_ASR_blocked']}，全部尾请求与blocking ID见VQA_tail_requests.csv。"]
lines+=['','## 共置执行关联与可观测状态','', 'cross_worker_service_overlap.csv完整保留同输入其他worker的服务变化及与Phi ASR handler区间的交集。交集取决于请求持续时间，并可能受输入难度、热状态、CPU与次序混杂，不能唯一识别GPU资源争用因果。正式G/S中保留S的配对服务均值没有膨胀；对真实重叠子群与非重叠子群分别报告，不能将总体均值直接推广至某时刻。','', '运行时能够因果获取：到达时已入队/发送尚未收到RESULT的Phi VQA/ASR数、S在途数、已完成请求统计、最新有限频率遥测、当前输入的元数据。原协议没有START ACK，因此“正在执行哪一任务”只能用在途保守近似，后台pending快照能保守禁止在忙Phi加ASR，不能读取事后native start/end。Phi空闲时未来VQA不可知；须以冻结到达率与ASR服务尾部风险保护。','', '现有日志可直接识别内部串行排队阻塞，但无法唯一归因服务膨胀中的GPU、CPU、同步等待部分。若要kernel级因果，需要独立CUDA events/NSys与受控扰动；本轮不把缺失量补为零。机制分类和是否进入原型由dynamic_opportunity_gate.json另行冻结。']
(OUT/'vqa_interference_attribution.md').write_text('\n'.join(lines)+'\n');print('ATTRIBUTION',summary,flush=True)
