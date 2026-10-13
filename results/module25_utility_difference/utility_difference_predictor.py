"""Route-dependent service distributions + project Poisson FIFO utility.

No GPU imports. Same Module20 components/quality/replica/memory constraints.
Independent validation fits exclude entire checkpoint combinations. The queue
kernel generalizes module21.optimizer.PoissonUtility to empirical service
distributions, arbitrary admitted paths and replica counts. J is unchanged.
"""
from pathlib import Path
import sys
for parent in Path(__file__).resolve().parents:
    if (parent/'module20/optimizer.py').exists():
        sys.path.insert(0,str(parent));break
from module25.common import *
from module20.optimizer import components
from deployment_core.routing.e2e_surrogate import evaluate_surrogate
from types import SimpleNamespace

FEATURE_NAMES=['intercept','own_path_load_log1p','other_tasks_same_worker_load',
    'foreign_generalist_load','foreign_specialist_load','resident_worker_count_log1p','idle_gap_over_service_log1p']

def features(deployment,routing,services,roles,capacity,model,task):
    n=deployment[model];g=capacity[model][str(n)];loads={}
    for m,nm in deployment.items():
        loads[m]=sum(row.get(m,0)*services[m][t]/1000/capacity[m][str(nm)] for t,row in routing.items() if row.get(m,0)>0)
    own=routing.get(task,{}).get(model,0)*services[model][task]/1000/g
    return np.array([1.,math.log1p(own),loads[model]-own,
        sum(v for m,v in loads.items() if m!=model and roles[m]=='generalist'),
        sum(v for m,v in loads.items() if m!=model and roles[m]=='specialist'),
        math.log1p(sum(deployment.values())),math.log1p(1/max(own,1e-12))])

class UtilityDifferencePredictor:
    def __init__(self):self.paths={};self.memory={};self.training_uids=[];self.training_groups=[];self.training_combinations=[]

    def fit(self,observations):
        start=time.perf_counter();by=defaultdict(lambda:defaultdict(list))
        self.training_uids=sorted(o['uid'] for o in observations)
        self.training_groups=sorted({o['deployment_workload_group'] for o in observations})
        self.training_combinations=sorted({o['checkpoint_group'] for o in observations})
        for ob in observations:
            for r in ob.get('service_rows',[]):
                if not r['usable_for_service_fit']:continue
                key=identity([r['model'],r['task'],r['replicas']])
                context=identity([r['deployment_workload_group'],r['configuration']])
                by[key][context].append(r)
        for key,contexts in by.items():
            xs=[];ys=[];groups=[];identities=[];hardware=set()
            for context,rows in contexts.items():
                r=rows[0];xs.append(features(r['deployment'],r['routing'],r['catalog_services'],r['catalog_roles'],r['catalog_capacity'],r['model'],r['task']))
                s0=r['standalone_service_ms']*r['replicas']/r['g_isolated']
                targets=[]
                for v in rows:
                    shape=np.array(v['quantiles_ms'])/v['mean_ms'];shape/=np.mean(shape)
                    targets.append(np.r_[math.log(v['mean_ms']/s0),np.log(shape),math.log1p(max(0,v['transport_mean_ms']))])
                ys.append(np.median(targets,axis=0));groups.append(r['deployment_workload_group']);identities.append(r['checkpoint_group'])
                hardware.update(v['hardware_uuid'] for v in rows if v['hardware_uuid'])
            X=np.array(xs);Y=np.array(ys);groupcounts=Counter(groups);weights=np.array([1/groupcounts[g] for g in groups])
            mean=np.average(X,axis=0,weights=weights);scale=np.sqrt(np.average((X-mean)**2,axis=0,weights=weights));scale[scale<1e-8]=1.
            mean[0]=0.;scale[0]=1.;Z=(X-mean)/scale
            enough=len(set(groups))>=4
            if enough:
                penalty=np.eye(Z.shape[1]);penalty[0,0]=0.
                coef=np.linalg.solve(Z.T@(weights[:,None]*Z)+penalty,Z.T@(weights[:,None]*Y))
            else:
                coef=np.zeros((Z.shape[1],Y.shape[1]));coef[0]=np.median(Y,axis=0)
            fitted=Z@coef
            self.paths[key]=dict(coefficients=coef.tolist(),feature_mean=mean.tolist(),feature_scale=scale.tolist(),
                feature_min=X.min(axis=0).tolist(),feature_max=X.max(axis=0).tolist(),
                feature_names=FEATURE_NAMES,targets=['log_mean_ratio',*['log_shape_q'+str(i) for i in range(33)],'log1p_transport_ms'],
                model='RIDGE_ROUTE_CONDITIONED' if enough else 'SPARSE_CONTEXT_MEDIAN_NO_ROUTE_SLOPE',
                distinct_deployment_workload_groups=len(set(groups)),distinct_checkpoint_combinations=len(set(identities)),
                training_contexts=len(contexts),hardware_uuids=sorted(hardware),
                mean_log_service_fit_RMSE=float(np.sqrt(np.average((fitted[:,0]-Y[:,0])**2,weights=weights))),
                measured_replica_count=int(json.loads(key)[2]))
        mem=defaultdict(list)
        for ob in observations:
            if ob.get('safe_memory_mib') is None or not ob['valid_session']:continue
            d=ob['catalog'];nominal=d['safe_factor']*sum(d['memory_table_mib'][m][str(n)] for m,n in ob['deployment'].items())
            mem[deployment_key(ob['deployment'])].append(ob['safe_memory_mib']/nominal)
        self.memory={k:dict(median_ratio=float(np.median(v)),observed_range=[min(v),max(v)],sessions=len(v)) for k,v in mem.items()}
        self.fit_seconds=time.perf_counter()-start
        return self

    def as_dict(self):
        return dict(type='RouteConditionalServiceQuantileRidge_PoissonFIFO_Module20Utility',paths=self.paths,memory=self.memory,
            training_uids=self.training_uids,training_groups=self.training_groups,training_combinations=self.training_combinations,
            fit_seconds=self.fit_seconds,feature_names=FEATURE_NAMES,shared_universal_interference_coefficient=False,
            known_limits=['Stationary M/G/1 predicts iid per-path services, not transient correlated GPU stalls',
                'No new hardware/input-length/thermal extrapolation certificate',
                'Historical isolated g(n) is a conditional service prior, never certain joint capacity',
                'Sparse-context path medians have no identified route-response slope',
                'Unknown replica counts or residencies require new grouped evidence'],
            queue_engine='Generalized existing Module21 PoissonUtility transform; independent Poisson thinning per physical replica; all tasks share one FIFO per replica',
            objective='Unchanged module20.optimizer.components and frozen quality/memory/replica/rho constraints')

    @classmethod
    def load(cls,path):
        saved=read(path);obj=cls()
        for k in ['paths','memory','training_uids','training_groups','training_combinations','fit_seconds']:setattr(obj,k,saved[k])
        return obj

    def service(self,d,x,routing,m,t,hardware_uuid=None):
        n=x[m];s0=d['models'][m]['service_time_ms'][t]*n/d['capacity_factors'][m][str(n)]
        key=identity([m,t,n]);p=self.paths.get(key);warnings=[]
        if not p:
            return dict(model=m,task=t,replicas=n,mean_ms=s0,quantiles_ms=[s0],transport_ms=0.,
                support='NO_MEASURED_CHECKPOINT_TASK_COUNT',uncertain_joint_replica_scaling=True,
                g_isolated_conditional_prior=d['capacity_factors'][m][str(n)],support_warnings=['No measured path at this replica count; deterministic nominal prior is not a validated service distribution'])
        f=features(x,routing,{m:v['service_time_ms'] for m,v in d['models'].items()},
            {m:v['role'] for m,v in d['models'].items()},d['capacity_factors'],m,t)
        z=(f-np.array(p['feature_mean']))/np.array(p['feature_scale']);y=z@np.array(p['coefficients'])
        if np.any(y>15) or np.any(y< -15):warnings.append('REGRESSION_EXTRAPOLATION_EXTREME')
        y=np.clip(y,-15,15);mean=s0*math.exp(float(y[0]));shape=np.sort(np.exp(y[1:34]));shape/=np.mean(shape)
        low=np.array(p['feature_min']);high=np.array(p['feature_max']);extrap=np.flatnonzero((f<low-1e-6)|(f>high+1e-6))
        if len(extrap):warnings.append('FEATURE_EXTRAPOLATION:'+','.join(FEATURE_NAMES[i] for i in extrap))
        if checkpoint_key(x) not in self.training_combinations:warnings.append('UNSEEN_CHECKPOINT_COMBINATION')
        if hardware_uuid and hardware_uuid not in p['hardware_uuids']:warnings.append('UNSEEN_PHYSICAL_GPU')
        if p['model'].startswith('SPARSE'):warnings.append('ROUTE_RESPONSE_SLOPE_NOT_IDENTIFIED')
        return dict(model=m,task=t,replicas=n,mean_ms=mean,quantiles_ms=(mean*shape).tolist(),
            transport_ms=max(0,math.expm1(float(y[34]))),support='MEASURED_PATH_CONDITIONAL' if not warnings else 'TRANSFER_OR_EXTRAPOLATION_UNVERIFIED',
            support_warnings=warnings,distinct_training_groups=p['distinct_deployment_workload_groups'],
            route_sensitive=p['model']=='RIDGE_ROUTE_CONDITIONED',uncertain_joint_replica_scaling=bool(warnings),
            g_isolated_conditional_prior=d['capacity_factors'][m][str(n)])

    def predict(self,d,x,routing,rates,budget,refs,weights,arrival_process='poisson',hardware_uuid=None,detail=True):
        start=time.perf_counter();legal=True;errors=[]
        for m,n in x.items():
            if m not in d['models'] or str(n) not in d['capacity_factors'].get(m,{}):errors.append('UNMEASURED_OR_UNKNOWN_REPLICA_COUNT:'+m)
        for t,q in rates.items():
            if abs(sum(routing.get(t,{}).values())-q)>1e-6:errors.append('ROUTE_CONSERVATION:'+t)
            for m,v in routing.get(t,{}).items():
                if v>1e-12 and (m not in x or t not in d['models'][m]['supported_tasks']):errors.append('QUALITY_CAPABILITY:'+m+'/'+t)
        if errors:return dict(U=None,eligible=False,support=False,hard_constraint_errors=errors,seconds=time.perf_counter()-start)
        services={};rho={};worker={};warnings=[]
        for m,n in x.items():
            for t,q in rates.items():
                if routing[t].get(m,0)>1e-12:
                    p=self.service(d,x,routing,m,t,hardware_uuid);services[m,t]=p
                    if p['support']!='MEASURED_PATH_CONDITIONAL':warnings.append(dict(model=m,task=t,support=p['support'],reasons=p['support_warnings']))
            rho[m]=sum(routing[t].get(m,0)/n*services[m,t]['mean_ms']/1000 for t in rates if (m,t) in services)
        nominal_memory=d['safe_factor']*sum(d['memory_table_mib'][m][str(n)] for m,n in x.items())
        memprofile=self.memory.get(deployment_key(x));memory=nominal_memory*(memprofile['median_ratio'] if memprofile else 1.)
        if not memprofile:warnings.append(dict(support='UNMEASURED_RESIDENCY_MEMORY',reasons=['Safe table is nominal; actual selected colocation memory unsupported']))
        if max(rho.values(),default=0)>=1:
            return dict(U=None,eligible=False,support=False,predicted_stability='NOMINAL_UNSTABLE',rho=rho,
                safe_memory_mib=memory,service_paths={identity(k):v for k,v in services.items()},
                support_warnings=warnings,seconds=time.perf_counter()-start)
        stats_by_task={};pooled=[]
        if arrival_process=='deterministic_periodic':
            prediction=self._finite_periodic(d,x,routing,rates,budget,refs,weights,services,rho,memory)
        else:
            prediction=self._poisson(d,x,routing,rates,budget,refs,weights,services,rho,memory)
        prediction.update(support=not warnings and prediction.get('queue_tail_resolved',True),
            service_paths={identity(k):v for k,v in services.items()},support_warnings=warnings,
            predicted_stability='NOMINAL_STABLE_UNCERTIFIED' if prediction.get('eligible') else 'NOMINAL_INFEASIBLE_OR_TAIL_UNRESOLVED',
            predicted_effective_capacity={m:dict(per_replica_busy_fraction=rho[m],aggregate_completion_capacity_rps=
                (sum(routing[t].get(m,0) for t in rates)/rho[m]) if rho[m]>0 else None,
                capacity_scope='Route-mix conditional capacity from predicted mean service; not certain joint g(n)') for m in x},
            seconds=time.perf_counter()-start,arrival_process=arrival_process,
            inference_uses_actual_test_service_or_queue=False)
        if not detail:prediction.pop('service_paths',None)
        return prediction

    def _finite_periodic(self,d,x,routing,rates,budget,refs,weights,services,rho,memory):
        from joint_deploy.measurement.load_generator import periodic_arrivals
        from joint_deploy.measurement.router import Router
        aliases={t:{f'{m}#{i}':q/x[m] for m,q in row.items() if q>0 for i in range(x[m])} for t,row in routing.items()}
        values={t:[] for t in rates};queuevalues=defaultdict(list);pooled=[];offered=within=late=0
        for seed in range(25001,25017):
            router=Router(aliases);available=defaultdict(float);indexes=defaultdict(int)
            for a,t in periodic_arrivals(rates,120.,seed):
                worker=router.choose(t);m=worker.rsplit('#',1)[0];p=services[m,t];quantiles=p['quantiles_ms'];j=indexes[worker,t]
                service=quantiles[(j*17+seed)%len(quantiles)];indexes[worker,t]+=1
                wait=max(0,available[worker]-a*1000);finish=a*1000+wait+service;available[worker]=finish
                latency=wait+service+p['transport_ms'];values[t].append(latency);queuevalues[worker].append(wait);pooled.append(latency)
                offered+=1;within+=int(a+latency/1000<=120.);late+=int(a+latency/1000>180.)
        pt={t:dict(mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95))) for t,v in values.items() if v}
        T=1-min(1.,(offered-late)/offered,within/(16*120*sum(rates.values())))
        c=components(pt,rates,refs,T,memory/budget,weights) if set(pt)==set(rates) else dict(U=None)
        eligible=memory<=budget and late==0 and c['U'] is not None
        return dict(**{k:v for k,v in c.items() if k!='U'},U=c['U'] if eligible else None,eligible=eligible,rho=rho,
            per_task=pt,pooled_request_mean_ms=float(np.mean(pooled)),pooled_request_P95_ms=float(np.percentile(pooled,95)),
            worker_queues={k:stats(v) for k,v in queuevalues.items()},safe_memory_mib=memory,queue_tail_resolved=late==0,
            queue_method='Native periodic_arrivals/Router, physical replica FIFO, 16 fixed phases, route-conditioned service quantiles',
            finite_window_expected_within_QPS=within/(16*120))

    def _poisson(self,d,x,routing,rates,budget,refs,weights,services,rho,memory):
        # Existing Module21 waiting transform, generalized from constant S to E[S].
        dt=.001;distributions={};waitmeans={};tail_ok=True;worker={}
        for m,n in x.items():
            if rho[m]==0:distributions[m]=(np.array([1.]),dt);waitmeans[m]=0.;continue
            bins=32768;cdf=None
            while bins<=262144:
                kernel=np.zeros(bins)
                for t in rates:
                    if (m,t) not in services:continue
                    q=routing[t][m]/n;s=np.array(services[m,t]['quantiles_ms'])/1000
                    length=min(bins-1,int(math.ceil(float(np.max(s))/dt)))
                    grid=np.arange(length)*dt
                    residual=np.mean(np.minimum(dt,np.maximum(0,s[:,None]-grid[None,:])),axis=0)
                    kernel[1:length+1]+=q*residual
                pdf=np.maximum(0,np.fft.irfft((1-rho[m])/(1-np.fft.rfft(kernel)),n=bins));cdf=np.cumsum(pdf);cdf/=cdf[-1]
                if 1-cdf[int(.9*bins)]<1e-5:break
                bins*=2
            resolved=bool(1-cdf[int(.9*len(cdf))]<1e-5);tail_ok=tail_ok and resolved
            distributions[m]=(cdf,dt)
            waitmeans[m]=sum(routing[t][m]/n*np.mean((np.array(services[m,t]['quantiles_ms'])/1000)**2) for t in rates if (m,t) in services)/(2*(1-rho[m]))
            q95=float(np.searchsorted(cdf,.95))*dt*1000
            for i in range(n):worker[f'{m}#{i}']=dict(mean_ms=waitmeans[m]*1000,p95_ms=q95,rho=rho[m],
                queue_tail_grid_seconds=len(cdf)*dt,queue_tail_resolved=resolved,
                family_queue='Separate FIFO per independently thinned replica; all routed tasks share worker')
        def pathcdf(m,t,a):
            cdf,step=distributions[m];s=np.array(services[m,t]['quantiles_ms'])/1000+services[m,t]['transport_ms']/1000
            z=(a-s)/step;i=np.clip(np.floor(z).astype(np.int64),0,len(cdf)-1)
            j=np.minimum(i+1,len(cdf)-1);v=cdf[i]+(z-i)*(cdf[j]-cdf[i])
            v=np.where(z<0,0.,np.where(z>=len(cdf)-1,1.,v))
            return float(np.mean(v))
        def mixedcdf(paths,a):return sum(f*pathcdf(m,t,a) for m,t,f in paths)
        def quantile(paths):
            lo=0.;hi=max(len(distributions[m][0])*dt+max(services[m,t]['quantiles_ms'])/1000 for m,t,f in paths)
            for _ in range(35):
                mid=(lo+hi)/2
                if mixedcdf(paths,mid)>=.95:hi=mid
                else:lo=mid
            return (lo+hi)/2*1000
        pt={};T=0.;total=sum(rates.values());paths_all=[]
        for t,q in rates.items():
            paths=[(m,t,v/q) for m,v in routing[t].items() if v>1e-12];mean=0.;windowloss=0.
            for m,_,f in paths:
                p=services[m,t];mean+=f*(p['mean_ms']/1000+p['transport_ms']/1000+waitmeans[m]);paths_all.append((m,t,f*q/total))
                cdf,_=distributions[m];survival=1-cdf
                for s in np.array(p['quantiles_ms'])/1000+p['transport_ms']/1000:
                    cap=max(0,120-s);index=min(len(cdf),int(cap/dt));integral=float(np.sum(survival[:index])*dt)
                    windowloss+=f/len(p['quantiles_ms'])*min(120,s+integral)/120
            T+=q/total*windowloss;pt[t]=dict(mean_ms=mean*1000,p95_ms=quantile(paths))
        norms=components(pt,rates,refs,T,memory/budget,weights);eligible=memory<=budget and tail_ok
        return dict(**{k:v for k,v in norms.items() if k!='U'},U=norms['U'] if eligible else None,eligible=eligible,
            rho=rho,per_task=pt,pooled_request_mean_ms=sum(rates[t]/total*pt[t]['mean_ms'] for t in rates),
            pooled_request_P95_ms=quantile(paths_all),worker_queues=worker,safe_memory_mib=memory,queue_tail_resolved=tail_ok,
            queue_method='Generalized Module21 M/G/1 empirical residual-life FFT; 1ms grid, adaptive up to262s, no clipped tail accepted',
            finite_window_expected_within_QPS=total*(1-T),
            finite_window_limitation='Stationary latency mixture predicts expected within-window completion; actual empty-start finite-window state and service correlation are not certified')

    def difference(self,d,x,proposed,control,rates,budget,refs,weights,**kwargs):
        a=self.predict(d,x,proposed,rates,budget,refs,weights,**kwargs)
        b=self.predict(d,x,control,rates,budget,refs,weights,**kwargs)
        delta=a['U']-b['U'] if a['U'] is not None and b['U'] is not None else None
        return dict(proposed=a,control=b,delta_U=delta,point_direction='POSITIVE' if delta is not None and delta>0 else 'NEGATIVE' if delta is not None and delta<0 else 'UNKNOWN',
            interval=None,uncertainty='Independent whole-contrast calibration required; support failures never receive a finite fabricated CI',
            support=a.get('support',False) and b.get('support',False))

def original_J(d,x,routing,rates):
    catalog={m:SimpleNamespace(service_time_ms=v['service_time_ms'],supported_tasks=set(v['supported_tasks'])) for m,v in d['models'].items()}
    calibration=SimpleNamespace(capacity_factors={m:{int(n):v for n,v in values.items()} for m,values in d['capacity_factors'].items()})
    return evaluate_surrogate(x,{(t,m):q for t,row in routing.items() for m,q in row.items()},rates,catalog,calibration)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--model',default=str(OUT/'colocation_prediction_model.json'));parser.add_argument('--query',required=True)
    args=parser.parse_args();q=read(args.query);definition,refs=utility_inputs();model=UtilityDifferencePredictor.load(args.model)
    result=model.difference(read(q['catalog']),q['deployment'],q['routing'],q['control_routing'],q['rates'],q['budget_mib'],refs,definition['weights'],arrival_process=q.get('arrival_process','poisson'),hardware_uuid=q.get('hardware_uuid'))
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
