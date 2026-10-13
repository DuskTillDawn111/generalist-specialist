import sys,collections,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32b.common import *
import soundfile as sf
f=read(OLD/'experiment_manifest.json');events=read(f['slots'][0]['trajectories']['measurement']['path'])['events'];samples={e['item']['sample']['id']:e['item']['sample'] for e in events};duration={s['input']:sf.info(s['input']).duration for s in samples.values() if s['task_type']=='asr'} if all('task_type' in s for s in samples.values()) else {}
for e in events:
 if e['task']=='asr':duration[e['item']['sample']['input']]=sf.info(e['item']['sample']['input']).duration
raws={a:read(OLD/'raw_runs'/f'{a}.json') for a in ['A0','A1','A2']};rrs={a:r['phases']['measurement']['request_records'] for a,r in raws.items()};base=rrs['A0'];allD=[duration[samples[r['sample_id']]['input']] for r in base if r['task']=='asr'];cuts=[float(np.percentile(allD,k)) for k in [33.3333333333,66.6666666667]]
def binid(d):return int(d>cuts[0])+int(d>cuts[1])
bins=[]
for i in range(3):
 gs=[r['worker_latency_ms'] for a in ['A1','A2'] for r in rrs[a] if r['task']=='asr' and r['model']==G and binid(duration[samples[r['sample_id']]['input']])==i];ss=[r['worker_latency_ms'] for r in base if r['task']=='asr' and binid(duration[samples[r['sample_id']]['input']])==i]
 bins.append(dict(bin=i,G_n=len(gs),S_n=len(ss),G_second_moment_seconds2=float(np.mean((np.array(gs)/1000)**2)),G_mean_ms=float(np.mean(gs)),G_p75_ms=float(np.percentile(gs,75)),G_p95_ms=float(np.percentile(gs,95)),S_p75_ms=float(np.percentile(ss,75)),S_mean_ms=float(np.mean(ss))))
pol=dict(duration_cuts_seconds=cuts,cost_bins=bins,metadata_lookup='current input audio header, statically cached; no labels/reference words/request identity',S_pressure_on_ms=2500.,S_pressure_off_ms=1250.,Phi_max_ASR_inflight=1,Phi_total_inflight_required=0,VQA_arrival_rate_qps=.2,ASR_arrival_rate_qps=.46,ASR_ref_ms=read(OUT/'frozen_utility_context.json')['refs']['asr'],VQA_ref_ms=read(OUT/'frozen_utility_context.json')['refs']['visual_qa'],future_VQA_mean_risk_multiplier=1.5,future_VQA_mean_risk_model='lambda_VQA * E[ASR_duration^2]/2, with 1.5 uncertainty multiplier; p95 separate admission and exposure cap',positive_margin_ms=100.,expected_VQA_blocked_fraction_budget=.03,max_G_ASR_share=.05,max_G_burst=1,min_G_cooldown_seconds=10.,G_tail_max_ms=bins[0]['G_p95_ms'],freeze_reason='Queue-major mechanism; gate conservative tail-estimated future-VQA risk and preserve only high-S-pressure opportunities. Thresholds chosen once from historical demand, not per-request IDs; no GPU outcomes used.',nonpreemptive=True,no_future_information=True)
# This read-only replay only asks whether observable opportunities occur on historical S-only states.
# It does not update those states as if diversion happened, and must not compute counterfactual utility.
ops=[];armed=False;lastG=-1e99;risk=0.;nc=0;admitted=0;quota=max(1,math.floor(.05*280));ref_ratio=pol['ASR_ref_ms']/pol['VQA_ref_ms']
for x in base:
 if x['task']!='asr':continue
 nc+=1;now=x['actual_submit_time_ns'];pend=[y for y in base if y['model']==S and y['actual_submit_time_ns']<now<y['completion_time_ns']];phi=[y for y in base if y['model']==G and y['actual_submit_time_ns']<now<y['completion_time_ns']];w=0.
 for j,y in enumerate(sorted(pend,key=lambda r:r['actual_submit_time_ns'])):
  b=bins[binid(duration[samples[y['sample_id']]['input']])];v=b['S_p75_ms']
  if j==0:v=max(0,v-(now-y['actual_submit_time_ns'])/1e6)
  w+=v
 if w>=pol['S_pressure_on_ms']:armed=True
 elif w<=pol['S_pressure_off_ms']:armed=False
 b=bins[binid(duration[samples[x['sample_id']]['input']])];g=b['G_p75_ms'];gp=b['G_p95_ms'];s=b['S_p75_ms'];riskpen=pol['future_VQA_mean_risk_multiplier']*.2*b['G_second_moment_seconds2']/2*ref_ratio*1000;score=w+s-g-riskpen;why='eligible_state'
 if not armed:why='S_low_pressure'
 elif phi:why='Phi_background_busy'
 elif gp>pol['G_tail_max_ms']:why='G_execution_tail_risk'
 elif score<=pol['positive_margin_ms']:why='nonpositive_risk_adjusted_margin'
 elif risk+.2*gp/1000> .03*.2*600:why='VQA_risk_budget'
 elif admitted>=quota:why='G_share_budget'
 elif (now/1e9-lastG)<10:why='cooldown'
 allow=why=='eligible_state'
 if allow:admitted+=1;risk+=.2*gp/1000;lastG=now/1e9
 ops.append(dict(request_id=x['request_id'],arrival_offset_seconds=x['frozen_offset_seconds'],S_observable_inflight=len(pend),Phi_observable_inflight=len(phi),predicted_S_wait_ms=w,current_input_duration_seconds=duration[samples[x['sample_id']]['input']],G_p75_ms=g,G_p95_ms=gp,expected_future_VQA_cost_ASR_equivalent_ms=riskpen,risk_adjusted_margin_ms=score,decision=why,eligible_state=allow,scope='Read-only observed S-only state replay; no counterfactual state updates or U'))
csvsave('historical_observable_opportunities.csv',ops);save('draft_routing_parameters.json',pol);save('audio_metadata_cache.json',duration)
# Separate associated GPU service effects by overlap, never infer pure-GPU causality.
irs=list(__import__('csv').DictReader((OUT/'cross_worker_service_overlap.csv').open()));groups=[]
for a in ['A1','A2']:
 for task in f['slots'][0]['lambda_rps']:
  for ov in [False,True]:
   z=[r for r in irs if r['arm']==a and r['task']==task and (r['has_Phi_ASR_overlap']=='True')==ov]
   if z:groups.append(dict(arm=a,task=task,has_overlap=ov,n=len(z),paired_service_mean_delta_ms=float(np.mean([float(r['service_delta_ms']) for r in z])),paired_service_total_ratio=sum(float(r['service_ms']) for r in z)/sum(float(r['paired_base_service_ms']) for r in z),scope='Association with native wall-handler overlap, not GPU causal attribution'))
csvsave('cross_worker_overlap_group_summary.csv',groups)
at=read(OUT/'attribution_summary.json');loss=list(__import__('csv').DictReader((OUT/'per_task_utility_decomposition.csv').open()));queue_primary=all(at[a]['VQA_wait_ASR_coverage']>.9 for a in ['A1','A2']) and all(float(next(r for r in loss if r['arm']==a and r['task']=='visual_qa')['P95_queue_ShAP_ms'])>float(next(r for r in loss if r['arm']==a and r['task']=='visual_qa')['p95_delta_ms']) for a in ['A1','A2']);passed=queue_primary and bins[0]['G_n']>=10 and admitted>0
save('dynamic_opportunity_gate.json',dict(at_unix_ns=time.time_ns(),mechanism_class='A' if queue_primary else 'D',mechanism='Phi internal nonpreemptive serial ASR blocking dominates VQA tails; GPU-only service effects not uniquely identifiable.',stage_B_passed=passed,observed_state_eligible_opportunities=admitted,read_only_replay_not_counterfactual=True,predicted_positive_U=None,real_GPU_gain_claimed=False,controllable=['Do not enqueue ASR behind or ahead of pending VQA','At most one G ASR outstanding and no second queued G ASR','Use short-input cost/tail prior plus observed S pressure','Cap expected future VQA exposure and route frequency'],uncontrollable=['New VQA arrivals during accepted nonpreemptive ASR','Unknown realized ASR duration','GPU scheduling/kernel contention without per-kernel logs'],no_future_or_request_id_policy=True,parameter_evidence=ref(OUT/'draft_routing_parameters.json'),formal_GPU_max_runs=3,decision='PROCEED_TO_CAUSAL_STATE_ROUTER_PROTOTYPE_AND_ONE_FROZEN_THREE_ARM_EXPLORATION' if passed else 'STOP_NO_CREDIBLE_CONTROLLED_OPPORTUNITY',independent_replication=False))
print('BINS',bins,'CUTS',cuts,'ELIGIBLE_STATES',admitted,'REASONS',dict(collections.Counter(x['decision'] for x in ops)),flush=True)
