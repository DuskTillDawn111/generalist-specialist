"""Split and cost-bounded stratified selection before any new pair outcome."""
import csv,itertools,subprocess
from collections import defaultdict
from .common import *

def identity(r):return f"{r['target_model']}::{r['target_task']}<-{r['co_model']}::{r['co_task']}"
def main():
 assert not (OUT/'pair_validation_manifest.json').exists()
 c=load(OUT/'catalog.json');profiles=load(OUT/'profile_protocol.json');roles=c['roles'];paths={(r['model'],r['task']):r for r in c['paths']}
 with (OUT/'pair_predictions.csv').open() as f:predictions=list(csv.DictReader(f))
 old=load(ROOT/'results/module9a1_task_conditioned_pilot/measurement_protocol_freeze.json')
 oldanchors={(m,k,j,l) for m,ks in old['tasks'].items() for k in ks for j,ls in old['co_tasks'].items() for l in ls}
 rows=[]
 for r in predictions:
  m,k,j,l=(r[key] for key in ['target_model','target_task','co_model','co_task']);ident=identity(r);h=digest({'split_seed':'9AF_SPLIT_V1','pair':ident});anchor=(m,k,j,l) in oldanchors
  memory=paths[(m,k)]['safe_memory_mib_by_replicas']['1']+paths[(j,l)]['safe_memory_mib_by_replicas']['1'];direction=roles[m][0].upper()+'<-'+roles[j][0].upper()
  rows.append({'identity':ident,'target_model':m,'target_task':k,'co_model':j,'co_task':l,'split':'CALIBRATION' if anchor or int(h[:8],16)%2==0 else 'VALIDATION','split_hash':h,'anchor_override':anchor,'safe_memory_mib':memory,'runnable':memory<=49140,'direction':direction,'mu_hat':float(r['mu_hat']),'selection_hash':digest({'selection_seed':'9AF_SELECT_V1','pair':ident})})
 run=sorted([r for r in rows if r['runnable']],key=lambda r:(r['mu_hat'],r['identity']))
 for i,r in enumerate(run):r['overlap_stratum']=['low','medium','high'][min(2,3*i//len(run))];r['structure_rank']=i+1
 for r in rows:
  if not r['runnable']:r['overlap_stratum']='nonresident';r['structure_rank']=None
 save('pair_split_manifest.json',{'frozen_at_utc':now(),'hash_rule':'sha256(canonical JSON {split_seed:9AF_SPLIT_V1,pair:identity}), first32bits mod2; even CALIBRATION, odd VALIDATION','group_unit':'directional pair identity; all repetitions remain in one split','anchor_rule':'every constituent path of every 9A1/2 co-task mixture forced CALIBRATION, conservatively excluding composite-anchor leakage','sample_level_split':False,'rows':rows})
 # Cover all requested direction/overlap strata and at least one new task contrast.
 # Find the six-identity set with fewest calibration model pairs and target paths.
 # The candidate representatives are the first hash in each structural stratum for
 # each model pair; retained task contrasts are also hash-first. No serving outcomes.
 available=[r for r in rows if r['runnable'] and r['split']=='VALIDATION' and r['direction'] in ['G<-S','S<-G','G<-G']]
 calibration=[r for r in rows if r['runnable'] and r['split']=='CALIBRATION']
 groups=defaultdict(list)
 for r in available:groups[(r['target_model'],r['co_model'])].append(r)
 candidates=[]
 for modelpair,rr in groups.items():
  cal=[r for r in calibration if (r['target_model'],r['co_model'])==modelpair]
  if not cal:continue
  for stratum in ['low','medium','high']:
   group=sorted([r for r in rr if r['overlap_stratum']==stratum],key=lambda r:r['selection_hash'])
   if group:candidates.append(group[0])
 # Ordering contrasts fix both co model AND co task, stronger than task-agnostic test.
 contrasts=[]
 for a,b in itertools.combinations(available,2):
  if a['direction'].startswith('G') and (a['target_model'],a['co_model'],a['co_task'])==(b['target_model'],b['co_model'],b['co_task']) and a['target_task']!=b['target_task']:
   pairkey=(a['target_model'],a['co_model'])
   if any((r['target_model'],r['co_model'])==pairkey for r in calibration):contrasts.append((a,b))
 contrasts.sort(key=lambda ab:digest(sorted(r['identity'] for r in ab)))
 # Greedy constrained set cover, with explicit cost first and deterministic hash tie.
 best=None
 for a,b in contrasts:
  selected={r['identity']:r for r in [a,b]}
  while len(selected)<6:
   have_dir={r['direction'] for r in selected.values()};have_strata={r['overlap_stratum'] for r in selected.values()};mp={(r['target_model'],r['co_model']) for r in selected.values()};tp={(r['target_model'],r['target_task']) for r in selected.values()}
   pool=[r for r in candidates if r['identity'] not in selected]
   if not pool:break
   pick=min(pool,key=lambda r:(-(int(r['direction'] not in have_dir)+int(r['overlap_stratum'] not in have_strata)),int((r['target_model'],r['co_model']) not in mp),int((r['target_model'],r['target_task']) not in tp),r['selection_hash']))
   selected[pick['identity']]=pick
  ss=list(selected.values())
  if len(ss)!=6 or {r['direction'] for r in ss}!={'G<-S','S<-G','G<-G'} or {r['overlap_stratum'] for r in ss}!={'low','medium','high'}:continue
  mp={(r['target_model'],r['co_model']) for r in ss};tp={(r['target_model'],r['target_task']) for r in ss};score=(len(mp),len(tp),digest(sorted(selected)))
  if best is None or score<best[0]:best=(score,ss)
 assert best,'cannot cover requested strata with six held-out pairs'
 selected=sorted(best[1],key=lambda r:r['selection_hash']);modelpairs={(r['target_model'],r['co_model']) for r in selected};cal_selected=[]
 for mp in sorted(modelpairs):
  group=[r for r in calibration if (r['target_model'],r['co_model'])==mp]
  # Prefer target already measured in validation to reduce independent baseline runs.
  targets={(r['target_model'],r['target_task']) for r in selected};cal_selected.append(min(group,key=lambda r:(int((r['target_model'],r['target_task']) not in targets),r['selection_hash'])))
 chosen=cal_selected+selected;targetpaths=sorted({(r['target_model'],r['target_task']) for r in chosen});states={};itembank=profiles['items'];registry=profiles['registry']
 for m,k in targetpaths:
  key=f'{m}__{k}__none';warm=[{'worker':m,'item':itembank[k][i%3]} for i in range(6)]
  states[key]={'state_key':key,'protocol_family':'current_active_pair','model':m,'task':k,'co_model':None,'target_worker':m,'worker_specs':{m:registry[m]},'target_items':itembank[k],'warmup_sequence':warm,'warmup_count':len(warm),'target_schedule_seconds':[i*7 for i in range(30)],'measurement_seconds':210,'co_tasks':[],'co_items':itembank,'co_backlog':64,'minimum_overlap_ratio':.9,'budget_mib':49140}
 for r in chosen:
  m,k,j,l=[r[key] for key in ['target_model','target_task','co_model','co_task']];key=f'{m}__{k}__{j}__{l}';warm=[]
  for i in range(6):warm.extend([{'worker':m,'item':itembank[k][i%3]},{'worker':j,'item':itembank[l][i%3]}])
  base=states[f'{m}__{k}__none'];states[key]={**base,'state_key':key,'co_model':j,'co_task':l,'worker_specs':{m:registry[m],j:registry[j]},'co_tasks':[l],'warmup_sequence':warm,'warmup_count':len(warm),'pair_identity':r['identity'],'split':r['split']}
 save('pair_validation_manifest.json',{'frozen_at_utc':now(),'selection_uses_new_pair_outcomes':False,'unique_directional_validation_pairs':len(selected),'process_repetitions':3,'selection_rule':'six identities: first-hash representatives per modelpair/structure tertile, constrained set cover with a task-ordering contrast; minimize distinct modelpairs then target baseline paths then deterministic set hash; per-addition missing coverage first, cost then hash','strata_rule':'tertiles of all runnable pair mu predictions, ties identity; full analytical kernel simulation only, no measured slowdown','validation_pairs':selected,'calibration_pairs':cal_selected,'new_standalone_paths':targetpaths,'maximum_validation_pairs':12,'no_pair_reselection_or_topup':True,'covered_directions':sorted({r['direction'] for r in selected}),'covered_overlap':sorted({r['overlap_stratum'] for r in selected}),'GPU_window_seconds':210,'new_ground_truth_protocol':'exact IDs/order, independent process reset, 30 target requests at i*7s, warmup fixed six per worker, co-backlog64 refill, single co-task; aggregate ratio retained'})
 save('protocol_identity.json',{'frozen_at_utc':now(),'states':states,'GPU':profiles['GPU'],'environment_variables':profiles['environment'],'bootstrap':{'resamples':10000,'seed':94002},'GROUND_TRUTH_PROTOCOL':'AGGREGATE_PROTOCOL_RETAINED','legacy_warmup_difference':'new target/co warmup exactly six requests each, so legacy 9A1/2 pair measurements are excluded from I1 fitted means; retain them only for development sanity','historical_environment_overrides':{}})
 save('historical_protocol_audit.json',{'9A1':'different repetition inputs and time-terminated warmup; not pooled in I1','9A2':'exact deterministic seed/IDs/cadence/backlog match, but state-specific larger frozen warmup counts and whisper mixed co tasks differ; not pooled in I1','8F2':'different multi-model workload/arrival/routing; development per-path sanity only, never pair accuracy','I1_source':'new protocol-identical calibration identities only; arithmetic modelpair average of calibration mu across three repetitions','selected_modelpairs':sorted(modelpairs),'compatibility_checks':['same runtime stack/native handler','same input bank/seed/order','same target cadence/window','same per-worker six-request warmup rule','same co backlog/refill','independent process resets']})
 save('scheduler_policy_audit.json',{'frozen_at_utc':now(),'runtime':'independent native-worker processes/CUDA primary contexts; default stream priorities; no MPS server','MPS_process_listing':subprocess.check_output(['pgrep','-a','-f','^(/[^ ]*/)?nvidia-cuda-mps'],text=True) if subprocess.run(['pgrep','-f','^(/[^ ]*/)?nvidia-cuda-mps'],capture_output=True).returncode==0 else 'NONE','retained_policies':['FCFS_CONTEXT_SERIAL'],'excluded_policies':{'EQUAL_PARTITION':'cross-context spatial concurrent kernels unavailable without MPS','PRIORITY_BASED':'no explicit priority shared context or process priority configured'},'selected_policy':'FCFS_CONTEXT_SERIAL','source':'https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html','selection_rule':'only physically compatible candidate retained; no error-based selection necessary. If >1 policy were compatible, calibration identity leave-one-pair-out MAE would select lowest error, tie policy name, frozen before validation','internal_context_quantum_unknown':True,'approximation':'alternate completed kernels; both initial phases included in median; exact driver time slicing not replicated','scheduler_policy_uncertainty':True,'uses_8F2_final_E2E':False,'uses_validation_outcomes':False})
 save('gate_rule_freeze.json',{'SUPPORTED':'paired pair-identity error difference I2-I1 CI95 entirely below zero; no systematic task-order direction errors; multi-model sanity does not show significant opposite direction to finite analytical predictions across all source repetition blocks; CPU evaluator fits bounded search cost','PARTIAL':'CI spans zero or multi-model aggregation is materially inconsistent, unless clear negative evidence triggers NOT_SUPPORTED','NOT_SUPPORTED':'CI95 entirely above zero or severe analytical mismatch evidenced by confidence-interval-separated task ordering reversals and systematic per-path errors','ordering_systematic_rule':'among independently CI-resolved task contrasts, exact one-sided sign test vs 1/2 shows reversal majority at p<.05; do not count unresolved ties as correct','multimodel_catastrophe_rule':'nonnegative delay estimator predicts slowing while real service-time slowdown CI lies entirely below 1, for a majority of observed active paths in a source state, OR predicted slowdown exceeds all empirical repetition means with no corresponding ordering (report magnitudes separately; no tuned percent cutoff)','runtime_sufficiency_rule':'CPU evaluate <1 second per candidate, inherited practical Roomie decision scale; report measured cost and 100/1000 candidate projections','no_fixed_improvement_percentage':True,'enter_9B_only_if_SUPPORTED':True,'frozen_before_new_validation_runs':True})
 print(json.dumps({'validation_pairs':[(r['identity'],r['overlap_stratum']) for r in selected],'calibration_pairs':[r['identity'] for r in cal_selected],'baseline_paths':targetpaths,'timed_runs':len(states)*3}))
if __name__=='__main__':main()
