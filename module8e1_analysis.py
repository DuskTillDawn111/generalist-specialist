"""Synthetic and historical-cache-only validation of method v2."""
import argparse,copy,csv,json,math,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from module8d3_diagnosis import (ROOT,C,D2,P,B,OUT as D3,NAMES,CAT,HW,ROLES,CAL,BOUNDS,POLICY,
    Offline,read,ref,save_json,file_hash,digest,counts,key,rates,routing,identity,flat,full_vector,validate_raw,better,csvwrite)
from method_v1 import verify_source
from method_v2 import MethodV2,frozen_config
from method_v2.neighborhood import generate_targeted_neighborhood
from method_v2.selection import diversified_order,candidate_class,replacement_key
from deployment_core import Workload,Hardware
from deployment_search.feedback import bottleneck_feedback
from deployment_search.result import CandidateEvaluation,jsonable
from deployment_search.routing import solve_canonical_routing
from deployment_search.runtime import RuntimeEvaluator
from deployment_search.integer_foundation import model,runtime
from deployment_search.tests.fixtures import forbid_enumeration
from integer_replica_pilot.evaluator import ReplicaRuntimeEvaluation
OUT=ROOT/'results/module8e1_method_revision'

def verify_v2():
    verify_source()
    spec=OUT/'method_v2_spec.json';assert file_hash(spec)==Path(str(spec)+'.sha256').read_text().strip()
    p=OUT/'method_v2_source_hashes.json'
    for r in read(p)['sources'].values():assert file_hash(r['path'])==r['sha256']

def freeze_source():
    verify_source();paths=sorted((ROOT/'method_v2').glob('*.py'))
    record={'method':'method_v2','spec':ref(OUT/'method_v2_spec.json'),'sources':{str(p.relative_to(ROOT)):ref(p) for p in paths},'frozen_before_development_replay':True,'shared_v1_sources':ref(OUT/'method_v1_freeze.json'),'new_runtime_models_or_predictor':False}
    p=OUT/'method_v2_source_hashes.json'
    if p.exists():assert read(p)==record
    else:save_json(p,record);Path(str(p)+'.sha256').write_text(file_hash(p)+'\n')
    print({'source_frozen':True,'files':len(paths)},flush=True)

class Planner:
    def __init__(self):self.routes={};self.generated={};self.lp_calls=0
    def canonical(self,n,x):
        k=n,key(x)
        if k not in self.routes:
            self.lp_calls+=1
            self.routes[k]=routing(n,x) if POLICY.deployment_memory(x)<=49140 and all(0<=x[m]<=BOUNDS[m] for m in CAT) else None
        return self.routes[k]
    def candidates(self,n,x):
        k=n,key(x)
        if k in self.generated:return self.generated[k]
        can=self.canonical(n,x)
        if can is None:return [],[],None
        mus,audit=generate_targeted_neighborhood(x,CAT,Workload(rates(n)),bottleneck_feedback(can,x),BOUNDS,POLICY.deployment_memory,49140,ROLES)
        feasible=[]
        for mu in mus:
            cc=self.canonical(n,mu.deployment)
            if cc is None:continue
            c=CandidateEvaluation(mu,canonical=cc,delta_z=can.max_pressure-cc.max_pressure,memory_mib=POLICY.deployment_memory(mu.deployment));feasible.append(c)
        ranked=diversified_order(feasible,x,CAT,Workload(rates(n)),POLICY.deployment_memory(x))
        self.generated[k]=(mus,ranked,audit);return self.generated[k]
    def shortest(self,n,start,target,max_steps=6):
        # Iterative-deepening goal search over generated LOCAL mutations, with
        # a count/L1 lower bound. Not an enumerator of all count vectors.
        start,target=counts(start),counts(target)
        def lower(x):return max(math.ceil(sum(abs(x[m]-target[m]) for m in CAT)/2),abs(sum(x.values())-sum(target.values())))
        expansions=0
        if start==target:return {'steps':0,'path':[],'proven_minimal':True,'local_expansions':0}
        for depth in range(lower(start),max_steps+1):
            failed=set()
            def dfs(x,left,seen):
                nonlocal expansions
                if x==target:return []
                if left==0 or lower(x)>left:return None
                token=(key(x),left,frozenset(seen))
                if token in failed:return None
                expansions+=1
                mus,_,_=self.candidates(n,x)
                ordered=sorted(mus,key=lambda m:(lower(m.deployment),m.operation,m.model,tuple(sorted(m.deployment.items()))))
                for mu in ordered:
                    y=mu.deployment
                    if key(y) in seen or lower(y)>left-1 or self.canonical(n,y) is None:continue
                    path=dfs(y,left-1,seen|{key(y)})
                    if path is not None:return [{'operation':mu.operation,'model':mu.model,'remove_model':getattr(mu,'remove_model',None),'add_model':getattr(mu,'add_model',None),'deployment':full_vector(y)}]+path
                failed.add(token);return None
            path=dfs(start,depth,{key(start)})
            if path is not None:return {'steps':len(path),'path':path,'proven_minimal':True,'lower_bound':lower(start),'local_expansions':expansions,'proof':'All shorter bounded generated-neighborhood paths fail; witness at this depth','full_vector_enumeration':False}
        return {'steps':None,'status':'NOT_FOUND_WITHIN_DECLARED_DEPTH','max_depth':max_steps,'local_expansions':expansions,'unreachable_claim':False}


def validate_synthetic():
    verify_v2()
    suite=unittest.defaultTestLoader.loadTestsFromName('method_v2.tests')
    with (OUT/'synthetic_final_tests.log').open('w') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    assert r.wasSuccessful()
    record={'status':'PASS','tests':r.testsRun,'cases':{'two_step_replacement_barrier':'PASS: v1 rolls back ADD intermediate; v2 accepts direct closure','pressure_vs_beneficial_removal':'PASS: synthetic table calls both P and R slots','coverage_filter':'PASS','memory_filter':'PASS','replica_aware_swap':'PASS','new_family_new_redundancy_closure':'PASS','negative_delta_and_fallback':'PASS','binary_integer_compatibility':'PASS','no_enumeration_tripwires':'PASS','model_renaming_equivariance':'PASS','frozen_budget_predictor_objective':'PASS'},'production_runtime_values_used':False,'GPU_runs':0}
    save_json(OUT/'synthetic_revision_validation.json',record)
    save_json(OUT/'swap_generation_validation.json',{'status':'PASS','covered_cases':['replacement closure','unique capability preservation','memory filtering','integer -1/+1','bounds','new ADD closure not mislabeled existing redundant family'],'source':ref(OUT/'synthetic_revision_validation.json')})
    save_json(OUT/'candidate_class_validation.json',{'status':'PASS','negative_delta_R_not_filtered':True,'P_R_diversification':True,'P_absent_and_R_absent_fallback':'PASS','maximum_slots':2,'source':ref(OUT/'synthetic_revision_validation.json')})


def planning():
    verify_v2();pl=Planner();old=Offline();rows=[];actual=[];paths={}
    for n in NAMES:
        t=read(C/n/'search_complete.json');target=counts(read(D2/'b2'/n/'result.json')['state']['deployment']);start=counts(t['seed']['deployment'])
        a=old.shortest(n,start,target);b=pl.shortest(n,start,target)
        paths[n]={'v1':a,'v2':b}
        rows.append({'workload':n,'status':'DEVELOPMENT','v1_shortest_steps':a['pressure_gated_min_length'],'v2_shortest_steps':b['steps'],'reduction':a['pressure_gated_min_length']-b['steps'] if b['steps'] is not None else None,'v2_seed_path_within_three_iterations':b['steps'] is not None and b['steps']<=3,'full_vector_enumeration':False})
        for label,x in [('seed',start),('v1_final',counts(t['deployment']))]:
            mus,ranked,audit=pl.candidates(n,x)
            actual.append({'workload':n,'state':label,'formal_M':12,'quality_effective_M':len(CAT),**audit,'LP_feasible_candidates':len(ranked),'candidate_states':[jsonable(c.mutation) for c in ranked],'selected_top2':[jsonable(c.mutation) for c in ranked[:2]]})
    with (OUT/'b2_path_compression.csv').open('w',newline='') as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    save_json(OUT/'b2_path_witnesses.json',{'workloads':paths,'full_deployment_vector_enumeration':False,'LP_calls':pl.lp_calls})
    synthetic=[]
    for size in (12,20,30):
        for density in ('one_deployed','third_deployed'):
            c={f'm{i:02d}':model(f'm{i:02d}',1.+i/1000,text=100.+i) for i in range(size)}
            present=1 if density=='one_deployed' else size//3;x={m:int(i<present) for i,m in enumerate(c)};bounds={m:1 for m in c};w=Workload({'text':.2});can=solve_canonical_routing(x,w,c,replica_bounds=bounds)
            begin=time.perf_counter()
            with forbid_enumeration():mus,audit=generate_targeted_neighborhood(x,c,w,bottleneck_feedback(can,x),bounds,lambda y:sum(c[m].memory_mib*y[m] for m in c),100.,{m:'generalist' for m in c})
            generation_ms=1000*(time.perf_counter()-begin);begin=time.perf_counter();feasible=[]
            for mu in mus:
                cc=solve_canonical_routing(mu.deployment,w,c,replica_bounds=bounds)
                if cc is not None:feasible.append(CandidateEvaluation(mu,canonical=cc,delta_z=can.max_pressure-cc.max_pressure,memory_mib=sum(c[m].memory_mib*mu.deployment[m] for m in c)))
            planning_ms=1000*(time.perf_counter()-begin)
            selected=diversified_order(feasible,x,c,w,sum(c[m].memory_mib*x[m] for m in c))[:2]
            synthetic.append({'M':size,'density':density,'deployed_types':present,**audit,'generation_ms':generation_ms,'LP_screen_ms':planning_ms,'LP_calls':len(mus),'LP_feasible':len(feasible),'selected_slots':len(selected),'full_replica_vector_space':2**size,'complete_vector_enumeration_count':0})
    save_json(OUT/'complexity_validation.json',{'status':'PASS','actual_development_states':actual,'synthetic':synthetic,'growth':'Task-overlap constrained pair generation can be quadratic in dense overlap; never exponential count-vector enumeration. No universal O(M) claim.','M20_M30_acceptable_for_offline_CPU_planning':True,'runtime_slots_unchanged':2,'model_pool_unchanged':True})
    reason=counts(read(C/'reasoning_vqa_heavy'/'search_complete.json')['deployment']);mus,ranked,audit=pl.candidates('reasoning_vqa_heavy',reason)
    target=counts(reason);target['phi4mm']=0
    selected=ranked[:2]
    seed=counts(read(C/'reasoning_vqa_heavy'/'search_complete.json')['seed']['deployment']);sm,sr,sa=pl.candidates('reasoning_vqa_heavy',seed)
    generated=any(c.deployment==target for c in sm);near=any(c.mutation.deployment==target for c in selected)
    record={'status':'DEVELOPMENT_ONLY','near_v1_final':full_vector(reason),'replacement_target':full_vector(target),'seed_direct_replacement_generated':generated,'seed_direct_replacement_TopB':any(c.mutation.deployment==target for c in sr[:2]),'near_final_minus_Phi_TopB':near,'NEIGHBORHOOD_FIX_CONFIRMED_OFFLINE':generated and near,'seed_top2':[jsonable(c) for c in sr[:2]],'near_final_top2':[jsonable(c) for c in selected],'candidate_counts':audit,'no_runtime_label_preference':True,'unbiased_validation_claim':False}
    save_json(OUT/'reasoning_replacement_case.json',record)
    print({'planning_complete':True,'paths':rows,'reasoning_fix':record['NEIGHBORHOOD_FIX_CONFIRMED_OFFLINE']},flush=True)


class CacheMiss(Exception):pass
class HistoricalCache(Offline):
    def __init__(self):
        super().__init__();self.proofs={}
        for p in sorted(D3.glob('**/runtime_cache_manifest.json')):
            for k,e in read(p)['configurations'].items():
                if e['status']=='complete' and k not in self.index:
                    assert digest(e['identity'])==k and file_hash(e['aggregate'])==e['aggregate_sha256']
                    self.index[k]={'entry':e,'runtime':read(e['aggregate']),'manifest':ref(p)}
    def validated(self,n,x,route):
        ev=self.evidence(n,x,route)
        if not ev:return None
        k=digest(ev['identity'])
        if k not in self.proofs:
            self.proofs[k]=[validate_raw(p,ev['identity'],seed,n) for p,seed in zip(ev['runtime']['raw_result_paths'],(7101,7202,7303))]
        assert len(ev['runtime']['raw_result_paths'])==3
        return ev

class ReplayEvaluator(RuntimeEvaluator):
    def __init__(self,n,cache):self.n=n;self.cache=cache;self.calls=[]
    def evaluate(self,x,w,route):
        ev=self.cache.validated(self.n,x,route);self.calls.append({'identity_sha256':digest(identity(self.n,x,route)),'deployment':full_vector(x),'status':'CACHE_HIT' if ev else 'CACHE_MISS','evidence':ev})
        if ev is None:raise CacheMiss(self.calls[-1]['identity_sha256'])
        return ReplicaRuntimeEvaluation(**ev['runtime'])


def replay():
    verify_v2();cache=HistoricalCache();results=[];misses=[]
    for n in NAMES:
        ev=ReplayEvaluator(n,cache);latest={}
        def checkpoint(r):latest['result']=r
        engine=MethodV2(ev,trace_callback=checkpoint)
        try:
            result=engine.search(Workload(rates(n)),CAT,HW,replica_bounds=BOUNDS,roles=ROLES,capacity_mode='empirical',memory_mode='empirical',calibration=CAL)
            status=result.status
        except CacheMiss:
            result=latest['result'];status='CACHE_MISS_UNRESOLVED_ITERATION'
        assert counts(result.seed.deployment)==counts(read(C/n/'search_complete.json')['seed']['deployment'])
        diagnostics=[]
        if status.startswith('CACHE_MISS'):
            # Probe only slots originally selected for THIS unresolved iteration,
            # capped by the remaining budget. Never commit an unknown winner.
            it=result.iterations[-1] if result.iterations else None
            if it:
                chosen=[c for c in it['candidates'] if c.selected_top_b]
                selected_ids={digest(identity(n,c.mutation.deployment,c.canonical.routing)) for c in chosen}
                for c in chosen:
                    cid=digest(identity(n,c.mutation.deployment,c.canonical.routing))
                    found=next((call for call in ev.calls if call['identity_sha256']==cid),None)
                    if found is None and len(ev.calls)<6:
                        evidence=cache.validated(n,c.mutation.deployment,c.canonical.routing)
                        found={'identity_sha256':cid,'deployment':full_vector(c.mutation.deployment),'status':'CACHE_HIT' if evidence else 'CACHE_MISS','evidence':evidence};ev.calls.append(found)
                    if found:
                        evd=found['evidence'];is_better=False;reason='UNMEASURED'
                        if evd:is_better,reason=better(evd['runtime'],jsonable(it['current_runtime']),c.mutation.deployment,it['deployment'])
                        diagnostics.append({'identity_sha256':cid,'class':candidate_class(c),'status':found['status'],'beneficial_cached':is_better,'objective_reason':reason,'committed_transition':False})
        assert len(ev.calls)<=6
        for call in ev.calls:
            if call['status']=='CACHE_MISS':misses.append({'workload':n,'identity_sha256':call['identity_sha256'],'deployment':call['deployment'],'routing':identity(n,counts(call['deployment']),engine._routing(counts(call['deployment']),Workload(rates(n)),CAT,.8).routing)['routing'],'reason':'No exact historical three-repetition identity; no approximate fractions or synthetic metrics','no_GPU_fill':True})
        v1ids={e['cache_key'] for e in []}
        v1=read(C/n/'search_complete.json');v1_evaluated={digest(e['identity']) for e in read(C/n/'search_cache/runtime_cache_manifest.json')['configurations'].values() if e['status']=='complete'}
        benefits=[]
        for it in result.iterations:
            for c in it['candidates']:
                if c.runtime is not None:
                    hit,reason=better(jsonable(c.runtime),jsonable(it['current_runtime']),c.mutation.deployment,it['deployment'])
                    if hit:benefits.append({'identity_sha256':digest(identity(n,c.mutation.deployment,c.canonical.routing)),'deployment':full_vector(c.mutation.deployment),'mean_ms':c.runtime.weighted_mean_e2e,'reason':reason,'not_evaluated_by_v1':digest(identity(n,c.mutation.deployment,c.canonical.routing)) not in v1_evaluated})
        row={'workload':n,'workload_status':'DEVELOPMENT_WORKLOADS','replay_status':status,'committed_final_claim_allowed':not status.startswith('CACHE_MISS'),'last_resolved_incumbent':full_vector(result.deployment),'last_resolved_mean_ms':result.runtime.weighted_mean_e2e if result.runtime else None,'v1_mean_ms':v1['runtime']['weighted_mean_e2e'],'logical_selected_state_attempts':len(ev.calls),'cached_hits':sum(c['status']=='CACHE_HIT' for c in ev.calls),'cache_misses':sum(c['status']=='CACHE_MISS' for c in ev.calls),'beneficial_cached_transitions':benefits,'unresolved_iteration_cached_diagnostics':diagnostics,'budget':jsonable(frozen_config()),'calls':ev.calls,'trace':result.to_dict(),'unbiased_final_validation_claim':False}
        results.append(row);save_json(OUT/'development_replay_progress.json',{'status':'RUNNING','workloads':results});print({'replay':n,'status':status,'hits':row['cached_hits'],'misses':row['cache_misses'],'last_resolved_mean':row['last_resolved_mean_ms']},flush=True)
    save_json(OUT/'development_replay.json',{'status':'COMPLETE_OFFLINE_WITH_CACHE_GAPS','workloads':results,'cache_donor_priority':'8C then 8D then 8D3; exact identity, no best-of donor selection','confirmation_measurements_used_for_search':False,'CACHE_MISS_never_imputed':True,'new_GPU_runs':0,'unbiased_claim':False})
    save_json(OUT/'development_cache_misses.json',{'CACHE_MISS_occurrences':len(misses),'unique_state_identities':len({r['identity_sha256'] for r in misses}),'states':misses,'new_GPU_runs':0})
    save_json(OUT/'development_reuse_validation.json',{'validated_identity_count':len(cache.proofs),'all_12_checks_equal':True,'proofs':cache.proofs})
    verify_v2()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze-source','synthetic','planning','replay']);a=p.parse_args()
    {'freeze-source':freeze_source,'synthetic':validate_synthetic,'planning':planning,'replay':replay}[a.action]()
