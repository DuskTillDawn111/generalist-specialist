"""14F: enumerate frozen single/pair/triple directions; no invented frontiers."""
import itertools,os
from .common import ROOT,OUT as DOUT,OLD,read,ref,now,verify
from module14c.common import catalog
from module14c.analyze import minimum_real_memory_cover
from module11.envelope import solve
from module11b.common import factor
from . import common
OUT=ROOT/'results/module14f_heavy_stable_overflow'
def save(name,data):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(__import__('json').dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert read(ROOT/'results/module14e_heavy_gpu_validation/module14e_gate.json')['continue_stable_overflow_search']
 h=read(DOUT/'heavy_empirical_replica_table.json');c,roles,cal=catalog(h);f=read(OLD/'screen_protocol_freeze.json');bg=next(w['rates'] for w in f['workloads'] if w['id']=='mix_03_I1')
 tasks=sorted({k for m,p in c.items() if roles[m]=='specialist' for k in p.supported_tasks if any(roles[n]=='generalist' and k in q.supported_tasks for n,q in c.items())});sets=[list(v) for n in [1,2,3] for v in itertools.combinations(tasks,n)];assert len(sets)==25
 OUT.mkdir(exist_ok=True)
 if not (OUT/'search_freeze.json').exists():save('search_freeze.json',{'at_utc':now(),'task_pool':tasks,'sets':sets,'background':bg,'budgets':f['budgets'],'direction_rule':'Per-budget headroom from Heavy single-task S-only operational empirical envelope; same frozen background; normalize every direction before any hotspot GPU outcome.','rho_limit':.8,'modes':['S','1G','2G'],'candidate_window':'C_S^op < workload < stable_system_boundary','no_new_gpu_outcomes_for_direction_budget_or_landmarks':True,'no_unmeasured_g_or_M':True,'source_H':ref(DOUT/'heavy_empirical_replica_table.json'),'source_14E_gate':ref(ROOT/'results/module14e_heavy_gpu_validation/module14e_gate.json'),'implementation':[ref(ROOT/'module14d/overflow.py'),ref(ROOT/'module11/envelope.py')]})
 freeze=read(OUT/'search_freeze.json');verify(freeze['implementation']);rows=read(OUT/'windows.json')['rows'] if (OUT/'windows.json').exists() else [];fits=read(OUT/'capacity_witnesses.json')['rows'] if (OUT/'capacity_witnesses.json').exists() else [];directions={}
 # Missing counts never become physical capacity limits. A legal exhaustion
 # window needs empirical terminal evidence or memory impossibility, not a
 # next-count extrapolation. We explicitly report evidence gaps in the frozen pool.
 unknown={m:cal.evidence_bounds[m]+1 for m in c if roles[m]=='specialist'}
 terminal={}
 for m in unknown:
  points=cal.model_evidence[m].get('Module11B_points',[])
  stop=next((p for p in reversed(points) if p.get('classification') in ['saturated','unstable','memory_infeasible']),None)
  if stop:terminal[m]=stop['classification']
 for budget in freeze['budgets']:
  bid=budget['id'];B=budget['safe_memory_budget_mib'];heads={}
  for task in tasks:
   single=solve(c,roles,cal,B,bg,{task:1.},'S',factor())
   if single['alpha'] is not None and single['alpha']>0:heads[task]=single['alpha']
  directions[bid]=heads;save('normalized_directions.json',{'background':bg,'heads_by_budget':directions,'source':'Heavy empirical single-task S envelope','fixed_before_GPU':True})
  for subset in sets:
   sid='+'.join(subset)
   if any(r['budget']==bid and r['set_id']==sid for r in rows):continue
   if any(k not in heads for k in subset):
    rows.append({'budget':bid,'set_id':sid,'tasks':subset,'status':'BACKGROUND_OR_HEADROOM_INFEASIBLE','legal_stable_window':False});save('windows.json',{'rows':rows});continue
   direction={k:heads[k] for k in subset};pair={}
   for mode in ['S','1G','2G']:
    fit=solve(c,roles,cal,B,bg,direction,mode,factor());pair[mode]=fit;fits.append({'budget':bid,'set_id':sid,'tasks':subset,'mode':mode,'direction':direction,**fit})
   s,a,b=[pair[m]['alpha'] for m in ['S','1G','2G']];window=a-s if a is not None and s is not None else None;second=b-a if b is not None and a is not None else None
   unresolved=[m for m in unknown if set(subset)&set(c[m].supported_tasks) and m not in terminal]
   # There is no safe basis to certify absent frontier counts without measured
   # g(next)/M(next). No artificial count cap is called Specialist exhaustion.
   legal=bool(window is not None and window>1e-7 and not unresolved)
   row={'budget':bid,'set_id':sid,'tasks':subset,'direction':direction,'alpha_S_op':s,'alpha_1G_op':a,'alpha_2G_op':b,'conditional_W1G':window,'conditional_W2G':second,'legal_stable_window':legal,'stable_system_boundary_GPU_verified':False,'missing_empirical_terminal_frontiers':unresolved,'first_unknown_counts':{m:unknown[m] for m in unresolved},'status':'CPU_CANDIDATE_REQUIRES_STABLE_GPU_BOUNDARY' if legal else 'NO_LEGAL_CERTIFIED_WINDOW_REPLICA_EVIDENCE_GAP' if window and window>1e-7 else 'NO_POSITIVE_OVERFLOW_WINDOW','conditional_H0_H1_H2_H3':{'H0':.8*s,'H1':.98*s,'H2':s+.3*window,'H3':s+.7*window} if window is not None and s is not None else None,'no_GS_substitution_success_claim':True}
   rows.append(row);save('windows.json',{'rows':rows});save('capacity_witnesses.json',{'rows':fits});print('OVERFLOW',bid,sid,'W1',window,'W2',second,'legal',legal,flush=True)
 legal=[r for r in rows if r['legal_stable_window']]
 status='NO_LEGAL_STABLE_OVERFLOW_WINDOW' if not legal else 'CPU_CANDIDATES_PENDING_GPU_STABLE_BOUNDARY'
 save('module14f_gate.json',{'at_utc':now(),'gate':status,'directions':len(sets),'budget_contexts':len(rows),'capacity_envelopes':len(fits),'legal_candidates':len(legal),'conditional_positive_windows':sum((r.get('conditional_W1G') or 0)>1e-7 for r in rows),'conditional_second_G_windows':sum((r.get('conditional_W2G') or 0)>1e-7 for r in rows),'stop_required':not legal,'stop_reason':'Unknown Heavy Specialist next replica counts can change S operational capacity; measured x<=2 search bounds are not empirical terminal capacity. No legal stable overflow or second-G capacity/latency window certified. No extrapolation/threshold/budget/model/hotspot changes.','GPU_hotspot_runs':0,'second_G_GPU_runs':0,'second_G_legal_window':False,'workload_direction_moved_after_GPU':False})
 (OUT/'module14f_summary.md').write_text('# Module14F\n\n'+status+'\n\nEnumerated all 25 frozen task single/pair/triple directions over the original B1/B3/B6 budgets. Positive empirical-count-space windows remain conditional while Heavy Specialist terminal frontiers are unmeasured. Unknown g/M counts cannot establish Specialist exhaustion; no hotspot GPU or second-G success is claimed.\n')
 print('OVERFLOW_GATE',status,flush=True)
if __name__=='__main__':main()
