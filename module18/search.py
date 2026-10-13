"""Both legal pools, frozen preferred points then finite formula candidates."""
from .common import *
from .solver import Solver
from module15.baseline import run as baseline

def phase_check(fit,rates,task,stage,solver):
 w=fit.get('winner')
 if not w:return {'passed':False,'reason':fit.get('status'),'winner':False}
 x=w['deployment'];route=w['routing'];q=rates[task];gs=[m for m in x if solver.roles[m]=='generalist'];target_s=[m for m in x if solver.roles[m]=='specialist' and task in solver.c[m].supported_tasks];rg=sum(v for p,v in route.items() if p.startswith(task+'::') and solver.roles[p.split('::')[1]]=='generalist');rs=q-rg
 remain=sum(max(0,.8*solver.cal.capacity_factors[m][x[m]]-w['rho'][m]*solver.cal.capacity_factors[m][x[m]])/(solver.c[m].service_time_ms[task]/1000) for m in x if task in solver.c[m].supported_tasks)
 speech={m:sum(v*solver.c[m].service_time_ms[k]/1000 for p,v in route.items() for k,name in [p.split('::')] if name==m and k in ['asr','lid'])/solver.cal.capacity_factors[m][x[m]] for m in x if any(k in solver.c[m].supported_tasks for k in ['asr','lid'])};active_medium=any(m.startswith('openai/whisper-medium') and speech[m]>=.4 for m in x);active_turbo='whisper_turbo' in x and speech.get('whisper_turbo',0)>=.4;known=active_medium and active_turbo
 conditions={'one_G_for_validation':len(gs)==1 and x[gs[0]]==1,'quality_and_coverage':all(k in solver.c[m].supported_tasks for p,v in route.items() if v>0 for k,m in [p.split('::')]),'flow_conservation':all(abs(sum(v for p,v in route.items() if p.startswith(k+'::'))-r)<1e-8 for k,r in rates.items()),'rho_gate':w['J_operational_certificate']['passed'],'no_known_heavy_speech_pair_bottleneck':not known,'not_old_failed_speech_workload':not (rates['asr']==1. and rates['lid']==12.)}
 if stage=='LOW':conditions['phase']=not target_s and rg>=q-1e-8
 elif stage=='MID':conditions['phase']=bool(target_s) and rs/q>=.8
 else:conditions['phase']=bool(target_s) and rg>1e-9 and rs>1e-9 and remain>=.01*q and min(rg,rs)*120>=3
 return {'passed':all(conditions.values()),'conditions':conditions,'reason':','.join(k for k,v in conditions.items() if not v) or None,'G':gs,'target_S_resident':target_s,'target_G_rps':rg,'target_S_rps':rs,'target_G_fraction':rg/q,'target_S_fraction':rs/q,'joint_operational_target_capacity_remaining_rps':remain,'joint_physical_fraction_headroom_min':1-w['max_rho'],'speech_demands':speech}

def main():
 f=read(OUT/'cpu_protocol_freeze.json');verify(f['implementation']+[f['full_pool'],f['controlled_pool'],f['pool_definition']]);rows=[];results={};selected={};cache={}
 for pool in ['FULL_HEAVY','CONTROLLED_COMMON_G']:
  data=read(f['full_pool' if pool=='FULL_HEAVY' else 'controlled_pool']['path']);s=Solver(data)
  def solve(task,stage,q):
   rates={**f['fixed_background_base'],task:q};key=(pool,task,stage,q)
   if key not in cache:
    fit=s.solve(rates,f['budget_mib']);ce=phase_check(fit,rates,task,stage,s);cache[key]={'stage':stage,'rates':rates,'fit':fit,'check':ce};w=fit.get('winner');rows.append({'pool':pool,'task':task,'stage':stage,'target_QPS':q,'status':fit['status'],'phase_passed':ce['passed'],'reason':ce['reason'],'deployment':w['deployment'] if w else None,'safe_mib':w['safe_memory_mib'] if w else None,'max_rho':w['max_rho'] if w else None,'target_G_fraction':ce.get('target_G_fraction'),'target_S_fraction':ce.get('target_S_fraction'),'joint_operational_target_margin_rps':ce.get('joint_operational_target_capacity_remaining_rps'),'tied_minimum_states':len(fit.get('tied_minimum_states',[]))})
   return cache[key]
  for task in f['task_priority']:
   pref=f['preferred_points'][task];points=[solve(task,stage,pref[stage]) for stage in ['LOW','MID','HIGH']];attempts=[{'source':'USER_PREFERRED_TRIPLE','points':points}];winner=None
   def complete(p):return all(z['check']['passed'] for z in p) and p[0]['check']['G']==p[1]['check']['G']==p[2]['check']['G']
   if complete(points):winner={'point_source':'USER_PREFERRED_TRIPLE','points':points}
   elif points[0]['check']['passed'] and points[1]['check']['passed'] and points[0]['check']['G']==points[1]['check']['G']:
    candidates=[]
    for g,gp in data['models'].items():
     if gp['role']!='generalist' or task not in gp['supported_tasks']:continue
     bg=sum(q*gp['service_time_ms'][k]/1000 for k,q in f['fixed_background_base'].items() if k!=task and k in gp['supported_tasks']);CG=(.8*data['capacity_factors'][g]['1']-bg)/(gp['service_time_ms'][task]/1000)
     for m,sp in data['models'].items():
      if sp['role']!='specialist' or task not in sp['supported_tasks']:continue
      CS=.8*data['capacity_factors'][m]['1']/(sp['service_time_ms'][task]/1000);cost=data['safe_factor']*(data['memory_table_mib'][g]['1']+data['memory_table_mib'][m]['1'])
      if CG>0 and CS>CG and cost<=f['budget_mib']:candidates.append((cost,g,m,CG,CS))
    for cost,g,m,CG,CS in sorted(candidates):
     for frac in f['derived_HIGH_fraction_order']:
      q=CS+frac*CG;high=solve(task,'HIGH',q);trip=[points[0],points[1],high];attempts.append({'source':'FROZEN_CAPACITY_DERIVED_HIGH','generator':{'G':g,'S':m,'CG':CG,'CS':CS,'fraction':frac},'points':trip})
      if complete(trip):winner={'point_source':'FROZEN_CAPACITY_DERIVED_HIGH','generator':attempts[-1]['generator'],'points':trip};break
     if winner:break
   result={'task':task,'pool':pool,'theoretical_witness':winner is not None,'witness':winner,'attempts':attempts,'all_legal_models_competed':list(data['models']),'Specialists_excluded':[]};results[pool,task]=result;save(f'cpu/{pool.lower()}_{task}.json',result);print('CPU',pool,task,'WITNESS',bool(winner),[(p['stage'],p['rates'][task],p['check']['passed'],p['check']['reason']) for p in points],flush=True)
   if winner and task not in selected:selected[task]={'pool':pool,'task':task,**winner}
 # Strengthen each selected point competition and preserve true original V3 baseline.
 for task,witness in selected.items():
  data=read(f['full_pool' if witness['pool']=='FULL_HEAVY' else 'controlled_pool']['path']);s=Solver(data)
  for point in witness['points']:
   fit=s.solve(point['rates'],f['budget_mib'],competition=3);assert fit['winner']['deployment']==point['fit']['winner']['deployment'];assert abs(fit['winner']['J_seconds']-point['fit']['winner']['J_seconds'])<1e-8;point['fit']=fit;point['original_V3_CPU_baseline']=baseline(point['rates'],f['budget_mib'],s)
 save('cpu_search_results.json',{'at_utc':now(),'selected':selected,'pool_results':[r for r in results.values()],'CPU_unique_solves':len(cache),'GPU_started':False,'new_models':0});csvsave('cpu_pool_comparison.csv',rows);print('CPU_SELECTED',[(t,w['pool'],[(p['stage'],p['rates'][t]) for p in w['points']]) for t,w in selected.items()],flush=True)
if __name__=='__main__':main()
