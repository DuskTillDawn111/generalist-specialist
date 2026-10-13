"""Instrument the unchanged periodic replay; calibration is path-specific.

The zero-calibration path is checked against Module20.predict, including T/U.
Empirical service distributions and queue offsets are fitted only to training
sessions. No actual test arrivals, service observations or queues are replayed.
"""
from .common import *
from module20.optimizer import components

def replay(d,x,routing,rates,budget,refs,weights,model=None,horizon=120.,phase=.37,detail=True):
    active={t:q for t,q in rates.items() if q>0}
    params={};rho={};unsupported=[]
    for m,n in x.items():
        demand=0.
        for t in active:
            q=routing[t].get(m,0)
            if q<=1e-9:continue
            p=model.parameter(x,m,n,t) if model else dict(ratio=1.,shape=[1.],queue_offset_ms=0.,scope='ORIGINAL')
            params[m,t]=p
            demand+=q*d['models'][m]['service_time_ms'][t]*p['ratio']/1000
            if p['scope'] not in ['EXACT_COMBINATION','ORIGINAL']:unsupported.append(dict(model=m,task=t,scope=p['scope']))
        rho[m]=demand/d['capacity_factors'][m][str(n)]
    paths={};latencies={t:[] for t in active};offered=within=late=backlog=0
    for m,n in x.items():
        arrival=[];tasks=[];svc=[];replica=[]
        for ti,t in enumerate(active):
            q=routing[t].get(m,0)
            if q<=1e-9:continue
            a=np.arange(-10*q+phase*(ti+1),horizon*q,1.)/q
            a=a[(a>=-10)&(a<horizon)]
            p=params[m,t];shape=np.asarray(p['shape'])
            # A fixed permutation of empirical quantiles; no favorable seed fitting.
            si=(np.arange(len(a))*17+ti*7)%len(shape)
            service=d['models'][m]['service_time_ms'][t]/1000*n/d['capacity_factors'][m][str(n)]*p['ratio']*shape[si]
            arrival.extend(a);tasks.extend([t]*len(a));svc.extend(service);replica.extend(np.arange(len(a))%n)
        if not arrival:continue
        arrival=np.asarray(arrival);tasks=np.asarray(tasks);svc=np.asarray(svc);replica=np.asarray(replica)
        for ri in range(n):
            ix=np.flatnonzero(replica==ri);ix=ix[np.argsort(arrival[ix],kind='stable')]
            a=arrival[ix];s=svc[ix];tt=tasks[ix];c=np.cumsum(s)
            finish=c+np.maximum.accumulate(a-(c-s));queue=(finish-a-s)*1000
            use=a>=0
            for t in set(tt[use]):
                mask=use & (tt==t);qs=np.maximum(0,queue[mask]+params[m,t]['queue_offset_ms'])
                latency=qs+s[mask]*1000
                f=a[mask]+latency/1000
                offered+=len(f);within+=int(sum(f<=horizon));late+=int(sum(f>horizon+60));backlog+=int(sum(f>horizon))
                latencies[t].extend(latency)
                if detail:
                    p=paths.setdefault(identity([m,t]),dict(model=m,task=t,services=[],queues=[],latencies=[]))
                    p['services'].extend(s[mask]*1000);p['queues'].extend(qs);p['latencies'].extend(latency)
    per_task={t:{'mean_ms':float(np.mean(v)),'p95_ms':float(np.percentile(v,95)),'request_count':len(v)} for t,v in latencies.items() if v}
    mem=d['safe_factor']*sum(d['memory_table_mib'][m][str(n)] for m,n in x.items())
    coverage=set(per_task)==set(active)
    admissible=all(m in x and t in d['models'][m]['supported_tasks'] for t,row in routing.items() for m,q in row.items() if q>1e-9)
    conservation=all(abs(sum(routing[t].values())-q)<1e-6 for t,q in active.items())
    T=1-min(1.,(offered-late)/max(1,offered),within/(sum(active.values())*horizon))
    norms=components(per_task,active,refs,T,mem/budget,weights) if coverage else dict(U=None)
    eligible=coverage and admissible and conservation and mem<=budget+1e-6 and max(rho.values(),default=0)<1 and late==0
    merged=[v for vs in latencies.values() for v in vs]
    result={**norms,'U':norms['U'] if eligible else None,'eligible':bool(eligible),
            'completed_subset_diagnostic_U_NOT_sortable':norms.get('U') if not eligible else None,
            'rho':rho,'max_rho':max(rho.values(),default=0),'safe_memory_mib':mem,'per_task':per_task,
            'pooled_request_mean_ms':float(np.mean(merged)) if merged else None,
            'pooled_request_P95_ms':float(np.percentile(merged,95)) if merged else None,
            'predicted_within_window_QPS':within/horizon,'end_backlog_requests':backlog,
            'incomplete_after_60s_drain':late,'unsupported_paths':unsupported,
            'exact_combination_supported':not unsupported,'stability_certified':False,
            'workload_group_observed_in_training':group(x,active) in model.training_groups if model else None,
            'uncertainty':'Archival group-balanced residuals; periodic arrivals remain a surrogate; unseen co-location and hardware are unverified'}
    pathstats={k:dict(model=p['model'],task=p['task'],service=summarize_samples(p['services']),
                      queue=summarize_samples(p['queues']),e2e=summarize_samples(p['latencies'])) for k,p in paths.items()}
    return result,pathstats
