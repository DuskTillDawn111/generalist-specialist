"""Same 27 workloads/budgets and frozen optimizer; corrected empirical tables."""
import concurrent.futures,os
from .common import *
from module14c.common import catalog
from module12.solver import StructuralSolver
from module12.metrics import describe

def context(job):
 work,budget=job;name=work['id']+'__'+budget['id']+'.json';path=OUT/'witnesses'/name
 if path.exists():return read(path)['rows']
 solver=StructuralSolver();rows=[];fits={}
 for regime,source in [(E,OUT/'regime_e_reconciled.json'),(H,OUT/'heavy_empirical_x3_reconciled.json')]:
  c,roles,cal=catalog(read(source));fit=solver.solve(work['rates'],c,roles,cal,budget['safe_memory_budget_mib']);fit['evidence_scope']='CPU_J_RHO_CERTIFICATE_WITH_EMPIRICAL_X3_AND_RECONCILED_POSTWARM_MEMORY_NOT_GPU_STABLE_BOUNDARY';fits[regime]=fit
  row={'workload':work['id'],'workload_regime':work['regime'],'budget':budget['id'],'memory_budget_mib':budget['safe_memory_budget_mib'],'Specialist_regime':regime,'count_scope':'14F_RECONCILED_MEMORY_OCR_X3',**describe(fit,work,roles,c),'frozen_Module9_solver':fit.get('frozen_Module9_solver')}
  if row['feasible']:
   row['Generalist_routed_fraction']=math.fsum(work['rates'][k]*v for k,v in row['task_G_fraction'].items())/math.fsum(work['rates'].values());row['G_served_task_sets']={m:sorted({p.split('::')[0] for p,v in row['routing'].items() if v>1e-9 and p.split('::')[1]==m}) for m in row['Generalist_identities']};row['certificate_passed']=fit['certificate']['passed'];row['memory_slack_mib']=row['memory_budget_mib']-row['safe_memory_mib']
  rows.append(row)
 save('witnesses/'+name,{'rows':rows,'fits':fits});return rows

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 f=read(C/'screen_protocol_freeze.json');verify(f['immutable_inputs']+f['implementation']+[f['plan_reference']]);jobs=[(w,b) for w in f['workloads'] for b in f['budgets']];assert len(jobs)==27
 (OUT/'witnesses').mkdir(exist_ok=True)
 save('structural_replay_freeze.json',{'at_utc':now(),'source_contexts':ref(C/'screen_protocol_freeze.json'),'E_table':ref(OUT/'regime_e_reconciled.json'),'H_table':ref(OUT/'heavy_empirical_x3_reconciled.json'),'workloads':f['workloads'],'budgets':f['budgets'],'rho_max':.8,'objective_and_Module9_unchanged':True,'implementation':[ref(ROOT/'module12/solver.py'),ref(ROOT/'module11b/convex_audit.py'),ref(ROOT/'module14f/replay.py')]})
 rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(context,j) for j in jobs]
  for future in concurrent.futures.as_completed(futures):
   r=future.result();rows.extend(r);print('REPLAY',r[0]['workload'],r[0]['budget'],len(rows),flush=True)
 rows.sort(key=lambda r:(r['workload'],r['budget'],r['Specialist_regime']));save('structural_replay_rows.json',rows);csvsave('structural_replay_empirical.csv',rows);print('REPLAY_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
