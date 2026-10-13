"""All deployments, continuous all-task Poisson routing; no GPU imports."""
from .common import *
from module20.optimizer import components
import numpy as np
from scipy.optimize import differential_evolution,minimize,linprog
import itertools

class PoissonUtility:
 def __init__(self,d,refs,weights,dt=.001,bins=16384):
  self.d=d;self.refs=refs;self.weights=weights;self.dt=dt;self.bins=bins;self.kernel={};self.services={};self.evals=0
  for m in MODELS:
   for ns in d['capacity_factors'][m]:
    n=int(ns)
    if not n:continue
    for t in d['models'][m]['supported_tasks']:
     s=d['models'][m]['service_time_ms'][t]/1000*n/d['capacity_factors'][m][ns];self.services[m,n,t]=s
     # Residual kernel mass = integral over ((j-1)dt,jdt] inside [0,s].
     k=np.zeros(bins);length=int(math.ceil(s/dt));k[1:length+1]=np.minimum(dt,np.maximum(0,s-np.arange(length)*dt));self.kernel[m,n,t]=np.fft.rfft(k)
 def domain(self,x,rates,fixed=None):
  fixed=fixed or {};bounds=[];tasks=[]
  for t in ['text','image','lid']:
   S=next(m for m in MODELS if m!=G and t in self.d['models'][m]['supported_tasks'])
   if t in fixed:bounds.append((fixed[t],fixed[t]))
   elif S in x:bounds.append((0.,1.))
   else:bounds.append((1.,1.))
   tasks.append(t)
  return tasks,bounds
 def routing(self,x,rates,z):
  rr={t:{} for t in rates}
  for ti,t in enumerate(['text','image','lid']):
   S=next(m for m in MODELS if m!=G and t in self.d['models'][m]['supported_tasks']);f=float(z[ti]);q=rates[t]
   if f>1e-12:rr[t][G]=q*f
   if f<1-1e-12:rr[t][S]=q*(1-f)
  for t in ['asr','visual_qa']:rr[t][G]=rates[t]
  rr['ocr'][MODELS[3]]=rates['ocr'];return rr
 def evaluate(self,x,rates,z,detail=False):
  self.evals+=1;rr=self.routing(x,rates,z);rho={};dist={};means={};mem=self.d['safe_factor']*sum(self.d['memory_table_mib'][m][str(n)] for m,n in x.items())
  for m,n in x.items():
   lam={t:row[m]/n for t,row in rr.items() if row.get(m,0)>1e-12};r=sum(q*self.services[m,n,t] for t,q in lam.items());rho[m]=r
   if r>=1:return {'U':None,'eligible':False,'rho':rho} if detail else 1000.+1000.*(r-1)
   if not lam:means[m]=0.;dist[m]=None;continue
   mw=sum(q*self.services[m,n,t]**2 for t,q in lam.items())/(2*(1-r));means[m]=mw
   denom=1-sum(q*self.kernel[m,n,t] for t,q in lam.items());pdf=np.fft.irfft((1-r)/denom,n=self.bins)
   # Circular tail alias is separately checked at refinement; clamp FFT roundoff only.
   pdf=np.maximum(pdf,0.);cdf=np.cumsum(pdf);cdf/=cdf[-1];dist[m]=cdf
  def Wcdf(m,a):
   if a<0:return 0.
   if dist[m] is None:return 1.
   u=a/self.dt;i=int(u)
   if i>=self.bins-1:return 1.
   return float(dist[m][i]+(u-i)*(dist[m][i+1]-dist[m][i]))
  stats={};T=0.;total=sum(rates.values());max_tail=0.
  for t,q in rates.items():
   paths=[(m,v/q) for m,v in rr[t].items() if v>0];mean=sum(f*(self.services[m,x[m],t]+means[m]) for m,f in paths)
   lo=min(self.services[m,x[m],t] for m,f in paths);hi=max(self.services[m,x[m],t] for m,f in paths)+(self.bins-1)*self.dt
   for _ in range(24):
    mid=(lo+hi)/2;c=sum(f*Wcdf(m,mid-self.services[m,x[m],t]) for m,f in paths)
    if c>=.95:hi=mid
    else:lo=mid
   p95=(lo+hi)/2
   # E[min(latency,120)]/120 gives expected window completion loss for stationary arrivals.
   latency_clip=0.
   for m,f in paths:
    s=self.services[m,x[m],t];cdf=dist[m]
    integral=0. if cdf is None else float(np.sum(1-cdf)*self.dt)
    latency_clip+=f*min(120.,s+integral)
    if cdf is not None:max_tail=max(max_tail,float(1-cdf[int(.9*self.bins)]))
   T+=q/total*latency_clip/120
   stats[t]={'mean_ms':mean*1000,'p95_ms':p95*1000}
  norms=components(stats,rates,self.refs,T,mem/BUDGET,self.weights)
  if not detail:return -norms['U']
  return {**norms,'eligible':True,'rho':rho,'max_rho':max(rho.values()),'operational_rho_08':max(rho.values())<=.8,'safe_memory_mib':mem,'routing':rr,'per_task':stats,'predicted_stationary_within_window_QPS':total*(1-T),'FFT_tail_mass_last_decile':max_tail,'model_uncertainty':'Unmeasured co-location/service variability; CPU prediction only','tail_grid_seconds':self.dt,'replica_arrival_assumption':'Independent random Poisson thinning per replica'}
 def optimize(self,x,rates,fixed=None):
  _,bounds=self.domain(x,rates,fixed);free=[i for i,b in enumerate(bounds) if b[0]!=b[1]];base=np.array([b[0] for b in bounds]);starts=[];solves=[]
  def expand(v):
   z=base.copy();z[free]=v;return z
  def objective(v):return self.evaluate(x,rates,expand(v))
  # Linear stability feasibility over continuous shares, no .8 hard gate added.
  A=[];b=[]
  for m,n in x.items():
   c0=self.routing(x,rates,base);r0=sum(row.get(m,0)*self.services[m,n,t]/n for t,row in c0.items() if row.get(m,0));cs=[]
   for i in free:
    z=base.copy();z[i]=1.;route=self.routing(x,rates,z);r1=sum(row.get(m,0)*self.services[m,n,t]/n for t,row in route.items() if row.get(m,0));cs.append(r1-r0)
   A.append(cs);b.append(1-1e-9-r0)
  if free:
   feas=linprog(np.zeros(len(free)),A_ub=np.array(A),b_ub=b,bounds=[bounds[i] for i in free],method='highs')
   if not feas.success:return {'status':'NO_NOMINAL_STABLE_CONTINUOUS_ROUTING','deployment':x,'winner':None}
   starts.append(feas.x)
  elif any(v<0 for v in b):return {'status':'NO_NOMINAL_STABLE_CONTINUOUS_ROUTING','deployment':x,'winner':None}
  if not free:
   p=self.evaluate(x,rates,base,True);return {'status':'FIXED_BY_CAPABILITY','deployment':x,'winner':p,'shares':base.tolist(),'evaluations':1}
  for v in itertools.product([0.,1.],repeat=len(free)):starts.append(np.array(v))
  starts.append(np.full(len(free),.5))
  for seed in [21001,21002]:
   fit=differential_evolution(objective,[bounds[i] for i in free],seed=seed,popsize=7,maxiter=45,tol=1e-7,polish=False,updating='immediate');starts.append(fit.x);solves.append({'method':'continuous_DE','seed':seed,'success':bool(fit.success),'fun':float(fit.fun),'evaluations':fit.nfev})
  scored=sorted([(objective(v),v) for v in starts],key=lambda a:a[0])
  bestf,best=scored[0]
  for _,v in scored[:3]:
   fit=minimize(objective,v,method='Powell',bounds=[bounds[i] for i in free],options={'xtol':1e-7,'ftol':1e-9,'maxiter':100});solves.append({'method':'Powell','success':bool(fit.success),'fun':float(fit.fun),'evaluations':fit.nfev})
   if fit.fun<bestf:bestf=float(fit.fun);best=fit.x
  z=expand(best);p=self.evaluate(x,rates,z,True)
  return {'status':'CONTINUOUS_NUMERICAL_OPTIMUM_NOT_GLOBAL_CERTIFICATE','deployment':x,'shares':z.tolist(),'winner':p,'solver_diagnostics':solves,'endpoint_best_objective':float(min(objective(v) for v in itertools.product([0.,1.],repeat=len(free))))}
