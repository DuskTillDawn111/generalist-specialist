"""Independent real-data development objective; never a production oracle.

The strict frozen applicability rules are preserved. An explicitly requested
numerical extrapolation is exposed separately for finite diagnostic search.
No runtime results, queues, production edits, native inference or GPU access.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/static-spillover-study-20261022'
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
FAMILIES=['generalist_1','generalist_2','generalist_3','text','image','ocr','asr']
SPEC=importlib.util.spec_from_file_location('_development_oracle_exposure',ROOT/'tools/spillover_capacity_20261022/exposure.py')
exposure=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(exposure)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite(v):return type(v) in (int,float) and math.isfinite(v)
def percentile(values,q=.95):
    if not values:return None
    v=sorted(values);i=(len(v)-1)*q;a=math.floor(i);b=math.ceil(i)
    return v[a]+(v[b]-v[a])*(i-a)
def convex_hull(point,points):
    if not points or any(not finite(v) for v in point):return dict(inside=None,status='UNKNOWN_MISSING_CONTEXT')
    a=np.asarray(points,dtype=float);p=np.asarray(point,dtype=float)
    result=linprog(np.zeros(len(a)),A_eq=np.vstack([a.T,np.ones(len(a))]),b_eq=np.r_[p,1.],bounds=(0,None),method='highs')
    if result.status not in (0,2):return dict(inside=None,status='UNKNOWN_SOLVER_STATUS',solver_status=int(result.status))
    return dict(inside=result.status==0,status='INSIDE_FEATURE_HULL' if result.status==0 else 'OUTSIDE_FEATURE_HULL',
        solver_status=int(result.status),point=list(point),training_contexts=len(points))


def fifo(rows,horizon,drain):
    """Unknown service poisons only that alias suffix; no partial full metric."""
    clocks=defaultdict(float);unknown=set();out=[]
    for row in rows:
        r=dict(row);alias=r['alias'];service=r['service_ms']
        if service is None:unknown.add(alias)
        if alias in unknown:r.update(finish_seconds=None,latency_ms=None,queue_ms=None,completed_in_window=None,completed_by_drain=None)
        else:
            start=max(r['arrival_seconds'],clocks[alias]);finish=start+service/1000;clocks[alias]=finish
            r.update(finish_seconds=finish,latency_ms=(finish-r['arrival_seconds'])*1000,queue_ms=(start-r['arrival_seconds'])*1000,
                completed_in_window=finish<=horizon,completed_by_drain=finish<=horizon+drain)
        out.append(r)
    return out


class DevelopmentOracle:
    def __init__(self,*,service_candidate='family_self_other',memory_candidate='M_full',
                 energy_window_candidate='E_full_window',energy_startup_candidate='E_full_startup',
                 packet=BASE/'text185-fine-development-v1/experiment-packet.json'):
        self.inputs={};self._context_cache={};self._resource_cache={};self._cohort_cache={};self._baseline_cache={}
        self.api=exposure.frozen_api()
        service_dir=BASE/'local-context-development-v1';resource_dir=BASE/'context-resource-candidates-dev-v1/fit-v1'
        sm=self._read(service_dir/'manifest.json');rm=self._read(resource_dir/'manifest.json')
        if sm['files']['report.json']!='325615553434de8446753c08b56b47780d52da2adcc3502d162a854a0e8de2fd':raise ValueError('Unexpected service development report')
        if rm['files']['report.json']!='3c4b27b0e48af44ec65117d49c07aec590a71aa347047e47f1082ea33ee70e61':raise ValueError('Unexpected resource development report')
        self.development=self._read(service_dir/'report.json',sm['files']['report.json'])
        self.resources=self._read(resource_dir/'report.json',rm['files']['report.json'])
        self.resource_rows=self._read(resource_dir/'run-fields.json',rm['files']['run-fields.json'])['rows']
        self.inventory=Path(self.development['inventory']['path']);self.inventory_manifest=self._read(self.inventory/'manifest.json',self.development['inventory']['manifest_sha256'])
        if sm['inputs'][str(self.inventory/'manifest.json')]!=self.development['inventory']['manifest_sha256']:raise ValueError('Service inventory binding differs')
        if rm['inputs'][str(self.inventory/'manifest.json')]!=self.development['inventory']['manifest_sha256']:raise ValueError('Resource inventory binding differs')
        self.packet=self._read(packet);self.config=self.packet['config'];self.catalog=self.config['catalog']
        self.samples=Path(self.packet['inputs'])/'runtime-inputs.jsonl'
        self._bind(self.samples,self.catalog['quality']['identity']['input_pool_sha256'])
        self.sample_rows=[json.loads(l) for l in self.samples.read_text().splitlines() if l.strip()]
        self.pool_ids={t:[r['pool_id'] for r in self.sample_rows if r['task_type']==t] for t in self.config['workload']}
        for task,ids in self.pool_ids.items():
            if ids!=self.catalog['quality']['input_order'][task]:raise ValueError('Original quality/input order differs')
        self.model=next(c for c in self.development['candidates'] if c['name']==service_candidate)
        if not self.model['full_development_fit']['success']:raise ValueError('Service candidate fit unknown')
        self.columns=self.model['columns'];self.beta=self.model['full_development_fit']['coefficients']
        self.domain={(d['family'],d['task'],d['n']):d for d in self.development['supported_local_domain'] if d['included']}
        self.memory_name=memory_candidate;self.window_name=energy_window_candidate;self.startup_name=energy_startup_candidate
        for name in [memory_candidate,energy_window_candidate,energy_startup_candidate]:
            if name not in self.resources['models']:raise ValueError('Unknown frozen resource candidate: '+name)
        self._bind(Path(__file__).resolve());self._bind(Path(exposure.__file__))
        for rel in ['gsdeploy/trace.py','gsdeploy/routing.py','gsdeploy/quality.py','gsdeploy/runtime/workload.py','gsdeploy/prediction.py']:
            self._bind(SOURCE/rel)

    def _bind(self,path,expected=None):
        path=Path(path).resolve();actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('Frozen input changed: '+str(path))
        self.inputs[str(path)]=actual
        return actual

    def _read(self,path,expected=None):
        self._bind(path,expected);return json.loads(Path(path).read_text())

    def provenance(self):
        return dict(inputs=dict(self.inputs),service_candidate=self.model['name'],memory_candidate=self.memory_name,
            energy_window_candidate=self.window_name,energy_startup_candidate=self.startup_name,
            status='DEVELOPMENT_ONLY_NOT_VALIDATED',seed_results_already_observed_in_session=[107,108],
            inputs_are_development_only=True,production_activated=False,raw_results_read_by_oracle=False)

    def domain_summary(self):
        return dict(supported_path_states=[dict(family=m,task=t,n=n,input_count=len(d['sample_ids']),
                weak_support=d['weak_support']) for (m,t,n),d in sorted(self.domain.items())],
            workload=self.config['workload'],horizon_seconds=self.config['horizon_seconds'],
            drain_seconds=self.config['drain_seconds'],seeds=[107,108],families=list(self.catalog['models']),
            full_seven_pool_coverage=False,uncovered_families=self.development['uncovered_families'],
            roles_used_as_admission=False,finite_search_domain_declared_by_caller=True,
            strict_service_domain='Exact listed path/n/input and frozen4D context hull',
            diagnostic_service_domain='Included exact path/n with exact original per-input b; input/context extrapolation must remain diagnostic')

    def verify_inputs(self):
        for path,expected in self.inputs.items():
            if sha(path)!=expected:raise ValueError('Input changed during oracle session: '+path)
        return True

    def _baselines(self,family,task,n):
        key=(family,task,n)
        if key in self._baseline_cache:return self._baseline_cache[key]
        profile=self.catalog.get('task_replica_profiles',{}).get(family,{}).get(task,{}).get(str(n))
        if not profile:self._baseline_cache[key]=None;return None
        grouped=defaultdict(list)
        for filename in profile['raw_service_samples']:
            path=str(Path(filename).resolve());expected=self.inventory_manifest['inputs'].get(path)
            if expected is None or not Path(path).is_file():
                self._baseline_cache[key]=None;return None
            self._bind(path,expected)
            for line in Path(path).read_text().splitlines():
                if not line.strip():continue
                r=json.loads(line);v=r.get('service_ms')
                if r.get('success') is True and finite(v) and v>0:grouped[r['sample_id']].append(v)
        values={p:statistics.median(grouped[p]) if grouped[p] else None for p in self.pool_ids[task]}
        self._baseline_cache[key]=values;return values

    def _validate(self,plan,seed):
        if seed not in [107,108]:raise ValueError('Only explicit development seeds107/108 allowed')
        x=plan.get('deployment',{});route=plan.get('routing',{})
        if not x or any(m not in self.catalog['models'] or type(n) is not int or n<=0
            or str(n) not in self.catalog.get('capacity_factors',{}).get(m,{})
            or str(n) not in self.catalog.get('memory_table_mib',{}).get(m,{}) for m,n in x.items()):
            raise ValueError('Invalid or unprofiled resident replica state')
        if set(route)!=set(self.config['workload']):raise ValueError('Route must cover exact configured tasks')
        for task,rate in self.config['workload'].items():
            row=route[task]
            if any(m not in x or not finite(q) or q<0 or task not in self.catalog['models'][m]['supported_tasks'] for m,q in row.items()):raise ValueError('Invalid task route')
            if not math.isclose(math.fsum(row.values()),rate,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('Routing does not conserve task workload')

    def _cohort(self,seed):
        if seed not in self._cohort_cache:
            rates=self.config['workload'];trace=self.api['trace'].make_trace(rates,self.config['horizon_seconds'],seed,self.config['arrival_process'])
            pool=self.api['workload'].SamplePool(self.samples,list(rates),seed)
            self._cohort_cache[seed]=(trace,[(arrival,task,pool.next(task)['pool_id']) for arrival,task in trace])
        return self._cohort_cache[seed]

    def _context(self,plan):
        x=plan['deployment'];route=plan['routing'];A={};missing=[]
        for family,n in x.items():
            terms=[]
            for task,row in route.items():
                q=row.get(family,0)
                if q<=0:continue
                bs=self._baselines(family,task,n);factor=self.catalog.get('capacity_factors',{}).get(family,{}).get(str(n))
                if bs is None or any(v is None for v in bs.values()) or not finite(factor) or factor<=0:
                    missing.append(dict(kind='MISSING_COMPLETE_BASELINE_OR_CAPACITY_FACTOR',family=family,task=task,n=n));terms.append(None)
                else:terms.append(q*statistics.mean(bs.values())/1000/factor)
            A[family]=math.fsum(terms) if all(v is not None for v in terms) else None
        residency=[self.catalog['memory_table_mib'].get(m,{}).get(str(n)) for m,n in x.items()]
        M=math.fsum(residency)/self.config['gpu_memory_mib'] if all(finite(v) for v in residency) else None
        if M is None:missing.append(dict(kind='MISSING_EXACT_RESIDENCY'))
        H={}
        for m in x:
            qq=[r[m] for r in route.values() if r.get(m,0)>0];total=math.fsum(qq)
            H[m]=-math.fsum(q/total*math.log(q/total) for q in qq) if total else 0.
        return A,M,H,missing

    def _service(self,plan,A,M,H,strict_context_checks=True):
        values={};strict={};proofs=[];reasons=[]
        for task,row in plan['routing'].items():
            for family,q in row.items():
                if q<=0:continue
                n=plan['deployment'][family];key=(family,task,n);d=self.domain.get(key);bs=self._baselines(*key)
                others=[v for m,v in A.items() if m!=family];other=math.fsum(others) if all(v is not None for v in others) else None
                target=[A[family],other,M,H[family]]
                cachekey=(key,tuple(target))
                if strict_context_checks and cachekey not in self._context_cache:
                    points=[[v[k] for k in ['A_self','A_other','M_resident','H_self']] for v in (d or {}).get('contexts_by_run',{}).values()]
                    self._context_cache[cachekey]=convex_hull(target,points)
                hull=self._context_cache[cachekey] if strict_context_checks else dict(inside=None,status='NOT_CHECKED_DIAGNOSTIC_ONLY')
                path_reasons=[]
                if d is None:path_reasons.append('UNSUPPORTED_EXACT_PATH_N')
                if hull['inside'] is not True:path_reasons.append('OUTSIDE_OR_UNKNOWN_FROZEN_SERVICE_CONTEXT_HULL')
                if bs is None or any(v is None for v in (bs or {}).values()):path_reasons.append('MISSING_COMPLETE_PER_INPUT_BASELINE')
                allowed=set((d or {}).get('sample_ids',[]));unseen=sorted(set(self.pool_ids[task])-allowed)
                if unseen:path_reasons.append('UNSUPPORTED_POOL_INPUTS')
                proofs.append(dict(family=family,task=task,n=n,context=target,hull=hull,unobserved_pool_inputs=unseen,
                    weak_support=(d or {}).get('weak_support'),reasons=path_reasons))
                reasons.extend(dict(kind=reason,family=family,task=task,n=n) for reason in path_reasons)
                logmean=statistics.mean(math.log(v) for v in bs.values()) if bs and all(v is not None for v in bs.values()) else None
                for sid in self.pool_ids[task]:
                    b=(bs or {}).get(sid);value=None
                    features=[float(f==family) for f in FAMILIES]+[A[family],other,M,H[family],
                        (math.log(b)-logmean)*other if b and logmean is not None and other is not None else None]
                    if d is not None and b is not None and all(finite(features[i]) for i in self.columns):
                        try:value=b*math.exp(math.fsum(features[i]*coef for i,coef in zip(self.columns,self.beta)))
                        except OverflowError:value=None
                        if not finite(value) or value<=0:value=None
                    values[family,task,n,sid]=value
                    strict[family,task,n,sid]=value if not path_reasons or (hull['inside'] is True and d and sid in allowed and b is not None) else None
        return values,strict,proofs,reasons

    def _resource(self,name,features,deployment,strict_context_checks=True):
        model=self.resources['models'][name];names=model['features'];missing=[k for k in names if not finite(features.get(k))]
        point=[features.get(k) for k in names];key=(name,tuple(point))
        if strict_context_checks and key not in self._resource_cache:self._resource_cache[key]=convex_hull(point,[[r['features'][k] for k in names] for r in self.resource_rows])
        hull=self._resource_cache[key] if strict_context_checks else dict(inside=None,status='NOT_CHECKED_DIAGNOSTIC_ONLY')
        seen=any(r['deployment']==deployment for r in self.resource_rows);beta=model['all_run_fit']['coefficients']
        probe=math.fsum(b*v for b,v in zip(beta,point)) if not missing and beta is not None else None
        valid=probe if hull['inside'] is True and seen else None
        return dict(candidate=name,features=names,point=point,coefficients=beta,hull=hull,deployment_observed=seen,
            missing_features=missing,value=valid,diagnostic_value=probe,valid_prediction=False,safety_bound=False)

    def _metrics(self,rows,Q,M,E):
        rates=self.config['workload'];horizon=self.config['horizon_seconds'];drain=self.config['drain_seconds']
        predicted=fifo(rows,horizon,drain);tasks={};all_known=bool(rows) and all(r['latency_ms'] is not None for r in predicted)
        for t in rates:
            rr=[r for r in predicted if r['task']==t];vv=[r['latency_ms'] for r in rr if r['latency_ms'] is not None]
            tasks[t]=dict(offered=len(rr),known=len(vv),mean_ms=statistics.mean(vv) if vv and len(vv)==len(rr) else None,
                p95_ms=percentile(vv) if vv and len(vv)==len(rr) else None)
        complete=all_known and all(v['offered']>0 for v in tasks.values());total=math.fsum(rates.values())
        terms=dict(Q=Q,T=1-sum(r['completed_in_window'] for r in predicted)/len(rows) if complete else None,
            L=math.fsum(rates[t]/total*tasks[t]['mean_ms']/self.config['latency_reference_ms'][t] for t in rates) if complete else None,
            P=math.fsum(rates[t]/total*tasks[t]['p95_ms']/self.config['latency_reference_ms'][t] for t in rates) if complete else None,M=M,E=E)
        score=-math.fsum(self.config['weights'][t]*v for t,v in terms.items()) if all(finite(v) for v in terms.values()) else None
        return dict(terms=terms,U=score,utility_complete=score is not None,per_task=tasks,drain_complete=all(r['completed_by_drain'] for r in predicted) if complete else None,
            complete=complete,offered=len(rows),completed_in_window=sum(r['completed_in_window'] is True for r in predicted),requests=predicted)

    def evaluate(self,plan,seed=107,diagnostic_extrapolation=False,include_requests=False,strict_context_checks=True):
        plan=copy.deepcopy(plan)
        try:self._validate(plan,seed)
        except (ValueError,TypeError,KeyError) as exc:
            return dict(status='INVALID_PLAN',terms={t:None for t in 'QTLPME'},U=None,rho_by_family={},unknown_reasons=[dict(kind='INVALID_PLAN',detail=str(exc))],diagnostic_extrapolation_only=None)
        A,Mresident,H,reasons=self._context(plan);services,strict_services,proofs,path_reasons=self._service(plan,A,Mresident,H,strict_context_checks);reasons+=path_reasons
        demand=math.fsum(A.values()) if all(v is not None for v in A.values()) else None
        features=dict(intercept=1.,raw_resident_over_budget=Mresident,sum_baseline_pressure=demand,workers=sum(plan['deployment'].values()))
        resources={k:self._resource(name,features,plan['deployment'],strict_context_checks) for k,name in [('M',self.memory_name),('E_window',self.window_name),('E_startup',self.startup_name)]}
        for key,r in resources.items():
            if r['value'] is None:reasons.append(dict(kind='UNSUPPORTED_RESOURCE_CONTEXT',target=key))
        safe=self.catalog['safe_factor'];watts=self.config['energy_reference_watts'];amortization=self.config['startup_amortization_seconds']
        m=resources['M']['value'];pm=resources['M']['diagnostic_value'];w=resources['E_window']['value'];pw=resources['E_window']['diagnostic_value'];s=resources['E_startup']['value'];ps=resources['E_startup']['diagnostic_value']
        normm=m*safe if m is not None else None;diagnostic_m=pm*safe if pm is not None else None
        norme=(w+s/amortization)/watts if w is not None and s is not None else None
        diagnostic_e=(pw+ps/amortization)/watts if pw is not None and ps is not None else None
        trace,cohort=self._cohort(seed);aliases,routes=self.api['trace'].expand_routes(plan['deployment'],plan['routing']);router=self.api['routing'].Router(routes)
        rows=[];strict_rows=[]
        for i,(arrival,task,sid) in enumerate(cohort):
            alias=router.choose(task);family=aliases[alias];key=(family,task,plan['deployment'][family],sid)
            common=dict(index=i,arrival_seconds=arrival,task=task,pool_id=sid,alias=alias,family=family)
            rows.append(dict(**common,service_ms=services.get(key)));strict_rows.append(dict(**common,service_ms=strict_services.get(key)))
        Q,qe=self.api['quality'].predict_quality(self.catalog,plan['deployment'],plan['routing'],self.config['workload'],trace,seed)
        if Q is None:reasons.append(dict(kind='HISTORICAL_QUALITY_UNKNOWN',evidence=qe))
        strict=self._metrics(strict_rows,Q,normm,norme);diag=self._metrics(rows,Q,diagnostic_m,diagnostic_e)
        rho={};strict_rho={};mean_services={}
        for family,n in plan['deployment'].items():
            terms=[];sterms=[]
            for task,row in plan['routing'].items():
                q=row.get(family,0)
                if q<=0:continue
                vv=[services.get((family,task,n,sid)) for sid in self.pool_ids[task]]
                sv=[strict_services.get((family,task,n,sid)) for sid in self.pool_ids[task]]
                mean=statistics.mean(vv) if all(v is not None for v in vv) else None
                sm=statistics.mean(sv) if all(v is not None for v in sv) else None
                mean_services[f'{family}/{task}/n{n}']=mean
                terms.append(q*mean/1000/n if mean is not None else None);sterms.append(q*sm/1000/n if sm is not None else None)
            rho[family]=math.fsum(terms) if all(v is not None for v in terms) else None
            strict_rho[family]=math.fsum(sterms) if all(v is not None for v in sterms) else None
        rho_known=all(v is not None for v in rho.values());memory_known=diagnostic_m is not None
        # Same numeric0.8 planning threshold, applied here to arithmetic point
        # service means/n. This is a new diagnostic, never a formal rho gate.
        diagnostic_feasible=(max(rho.values(),default=0)<=.8+1e-8 and diagnostic_m<=1+1e-8 and diag['drain_complete'] is True) if rho_known and memory_known and diag['drain_complete'] is not None else None
        diag.update(status='DIAGNOSTIC_EXTRAPOLATION_ONLY',rho_by_family=rho,max_rho=max(rho.values()) if rho_known else None,
            arithmetic_mean_point_service_ms=mean_services,planning_feasible=diagnostic_feasible,
            resource_feasible=diagnostic_feasible,
            feasibility_scope='Diagnostic arithmetic-point rho<=0.8 plus fitted memory and deterministic FIFO drain. resource_feasible is a compatibility alias for planning_feasible; neither replaces original measured runtime stability or execution admission.',
            memory_budget_pass=diagnostic_m<=1+1e-8 if memory_known else None,
            rho_threshold_pass=max(rho.values(),default=0)<=.8+1e-8 if rho_known else None,rho_threshold=.8,
            valid_prediction=False,certifies_optimum=False,execution_admission=False)
        result=dict(status='DEVELOPMENT_ONLY_UNKNOWN_FORMAL_U',plan=plan,seed=seed,terms=strict['terms'],U=None,
            strict_context_checks_performed=strict_context_checks,
            supported_domain_development_arithmetic_U=strict['U'],rho_by_family=strict_rho,unknown_reasons=reasons,
            service_domain=proofs,resources=resources,context=dict(A_by_family=A,M_resident=Mresident,H_by_family=H),
            per_task=strict['per_task'],trace_sha256=self.api['trace'].trace_digest(trace),quality_evidence=qe,
            diagnostic_extrapolation_only=diag if diagnostic_extrapolation else None,
            candidate_selection_performed=False,predictor_activated=False,full_seven_pool_optimality=False,
            service_scope='Unsmeared b*exp(X beta) geometric/median point services. rho is arithmetic mean of these per-input points, not calibrated arithmetic mean of actual runtime service.',
            seed_scope='Both107/108 are development seeds whose prior outputs were observed in this session; no independent validation claim.',
            physical_memory_bound=False,actual_stability='UNKNOWN',resource_models_are_development_only=True)
        if not include_requests:
            strict.pop('requests',None);diag.pop('requests',None)
        else:result['requests']=strict['requests']
        return result

    evaluate_plan=evaluate


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=107);parser.add_argument('--diagnostic-extrapolation',action='store_true');parser.add_argument('--include-requests',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();oracle=DevelopmentOracle();result=oracle.evaluate(json.loads(args.plan.read_text()),args.seed,args.diagnostic_extrapolation,args.include_requests)
    oracle.verify_inputs();document=dict(result=result,provenance=oracle.provenance())
    if args.output:
        with args.output.open('x') as stream:json.dump(document,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    else:print(json.dumps(document,ensure_ascii=False,allow_nan=False))
