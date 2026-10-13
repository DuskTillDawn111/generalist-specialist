"""Verify selection maximality and document pre-runtime exact-signature reuse."""
import bisect,csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];TOL=1e-9

def read(n):return json.loads((OUT/n).read_text())
def write(n,d):(OUT/n).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
validation=read('validation_results.json');assert validation['status']=='PASS'
for r in read('artifact_hashes.json').values():assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
final=read('final_memory_budgets.json');budgets=[v['budget_mib'] for v in final['budgets']];levels=read('frozen_intensity_levels.json');limits=defaultdict(dict);mem={}
for r in csv.DictReader((OUT/'deployment_capacity_limits.csv').open()):limits[r['mixture']][int(r['state_id'])]=float(r['alpha_limit']);mem[int(r['state_id'])]=float(r['M_safe_mib'])
reference=levels['alpha_max_reference'];v={};pop={B:[i for i in mem if mem[i]<=B] for B in budgets}
for m in limits:
 for B in budgets:v[m,B]=sorted(limits[m][i]/reference[m] for i in pop[B])
points=[]
for value in sorted(a/reference[m] for m,d in limits.items() for a in d.values()):
 if not points or value-points[-1]>TOL*max(1,value,points[-1]):points.append(value)
bsg=final['anchors']['B_SG_global'];majority=len(budgets)//2+1

def n(m,B,b):vs=v[m,B];return len(vs)-bisect.bisect_left(vs,b-TOL*max(1,b))
def satisfies(label,b):
 for m in limits:
  if label=='I1':
   if sum(3*n(m,B,b)>=2*len(pop[B]) for B in budgets)<majority or any(n(m,B,b)<1 for B in budgets):return False
  elif label=='I2':
   if sum(3*n(m,B,b)>=len(pop[B]) for B in budgets)<majority or any(n(m,B,b)<2 for B in budgets if B>=bsg):return False
  elif sum(n(m,B,b)>=2 for B in budgets)<majority:return False
 return True
proof=[]
for l in levels['levels']:
 b=l['beta'];assert satisfies(l['id'],b);nextpoint=next((p for p in points if p>b+TOL*max(1,b,p)),None);assert nextpoint is None or not satisfies(l['id'],nextpoint);proof.append({'level':l['id'],'selected_beta':b,'selected_satisfies_rule':True,'next_capacity_breakpoint_beta':nextpoint,'next_breakpoint_fails_rule':nextpoint is None or not satisfies(l['id'],nextpoint)})
# Prove the memory replacement is the FIRST qualifying empirical point.
curve=list(csv.DictReader((ROOT/'results/module8f1a_memory_breakpoints/memory_breakpoint_curve.csv').open()));bps=[float(r['breakpoint_mib']) for r in curve];fractions=[float(r['fraction_feasible']) for r in curve]
def fraction(B):return fractions[bisect.bisect_right(bps,B)-1]
log=next(v for v in final['deduplication_log'] if v['old_label']=='B3');original=log['original_mib'];chosen=log['selected_mib'];survivors=[B for B in budgets if B!=chosen]
qualifying=[]
for b in bps:
 if b<original or b>=49140 or b in survivors:continue
 lo=max((p for p in survivors if p<b),default=None);hi=min((p for p in survivors if p>b),default=None)
 if (lo is None or fraction(b)-fraction(lo)>=.05) and (hi is None or fraction(hi)-fraction(b)>=.05):qualifying.append(b)
assert chosen==min(qualifying)
write('selection_maximality_verification.json',{'status':'PASS','intensity_levels':proof,'memory_replacement_is_first_qualifying_empirical_breakpoint':True,'grid_changed':False,'runtime_read':False})
# Identify exact known lambda/deployment/routing matches. Environment/context and
# phase contracts still require prelaunch verification; no speculative discount.
campaign=read('campaign_cells.json');records=[]
for c in campaign['cells']:
 for method in ['static_v2','RC_SG']:
  p=c['preflight'][method]
  if p['status']!='READY':continue
  signature={'lambda_rps':c['lambda_rps'],'deployment':p['deployment'],'routing_rps':p['canonical_routing_rps']};encoded=json.dumps(signature,sort_keys=True,separators=(',',':'));h=hashlib.sha256(encoded.encode()).hexdigest()
  records.append({'signature_sha256':h,'cell_id':c['cell_id'],'method':method,'role':'cold_seed' if method=='static_v2' else 'fixed','memory_budget_mib':c['memory_budget_mib'],'signature':signature})
groups=defaultdict(list)
for r in records:groups[r['signature_sha256']].append({k:value for k,value in r.items() if k!='signature'})
repeated=[{'signature_sha256':h,'members':members,'potential_search_phase_raw_run_savings':3*(len(members)-1)} for h,members in sorted(groups.items()) if len(members)>1]
savings=sum(g['potential_search_phase_raw_run_savings'] for g in repeated)
reuse={'frozen_before_any_runtime':True,'known_execution_signature_groups':repeated,'potential_search_phase_saving_upper_bound':savings,'confirmed_reused_runs':0,'conservative_run_estimate_discount':0,'must_still_match':['full runtime context and source identity','physical hardware/environment/weights/adapters/precision/backend','settings and exact random seed','phase and role contract','each cell independently commits the state and passes its own safe-memory gate'],'forbidden':['approximate float/alpha/routing matches','search-confirmation sharing','seed-final sharing within a method','baseline fixed-confirmation sharing with v2 seed/final role','treating one shared raw run as independent samples'],'A_B_overlap_already_deduplicated':True,'future_v2_nonseed_states_known':False}
write('reuse_opportunities.json',reuse)
estimate=read('gpu_budget_estimate.json');estimate['known_preflight_execution_signature_reuse_source']='reuse_opportunities.json';estimate['potential_search_run_savings_not_discounted']=savings;write('gpu_budget_estimate.json',estimate)
validation['intensity_rule_maximality_verified']=True;validation['memory_replacement_first_qualifying_point_verified']=True;validation['known_execution_signature_reuse_groups']=len(repeated);write('validation_results.json',validation)
summary=(OUT/'module8f1b_summary.md').read_text();summary+='\n独立验证 PASS：原有 canonical pressure LP 复核全部 5772 limits，最大相对差 4.237038865532085e-16；947 curve rows、复制 intervals、RC-SG 每步 gain/tie-break、所有 ready routing/memory 与冷 seed 最小显存均已验证。选定 intensity 满足规则且下一个 capacity breakpoint 不满足，B3 replacement 为首个合格 empirical point。\n'
summary+=f'\n已知 seed/RC-SG exact lambda/deployment/canonical-route signature 重复组={len(repeated)}，潜在 search raw-run savings 上界={savings}；仍需完整 context/phase identity 校验，保守 324-run 估计不预先打折。见 reuse_opportunities.json。\n'
(OUT/'module8f1b_summary.md').write_text(summary)
# Freeze the finalized inventory after all validations and supplemental files.
gate=read('resource_campaign_gate.json');gate['independent_validation']='PASS';gate['validation_source']={'path':str(OUT/'validation_results.json'),'sha256':hashlib.sha256((OUT/'validation_results.json').read_bytes()).hexdigest()};gate['selection_maximality_validation']='PASS';write('resource_campaign_gate.json',gate)
art={}
for p in sorted(OUT.iterdir()):
 if p.is_file() and p.name not in ('artifact_hashes.json','execution.log','verification.log'):
  b=p.read_bytes();art[p.name]={'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
write('artifact_hashes.json',art)
print(json.dumps({'status':'PASS','selection_maximality':'PASS','reuse_candidate_groups':len(repeated),'potential_search_savings_undiscounted':savings,'RESOURCE_STATIC_GPU_CAMPAIGN_READY':gate['RESOURCE_STATIC_GPU_CAMPAIGN_READY']}))
