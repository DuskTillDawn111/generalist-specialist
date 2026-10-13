"""Freeze inputs and exhaust all pair/triple capacity directions, CPU only."""
import itertools
import math
from .common import *
from .envelope import solve

def enumerate_states(c,roles,cal,budget,background):
    models=sorted(c);states=[]
    for counts in itertools.product(*(range(cal.evidence_bounds[m]+1) for m in models)):
        x=dict(zip(models,counts))
        if safe_memory(x,cal)>budget+1e-7:continue
        if any(not any(x[m] and k in c[m].supported_tasks for m in models) for k,v in background.items() if v):continue
        states.append({'deployment':x,'safe_memory_mib':safe_memory(x,cal),
                       'N_G':sum(x[m] for m in models if roles[m]=='generalist')})
    return states

def missing_screen(c,roles,cal,budget,background):
    required={k for k,v in background.items() if v};records=[]
    for m in sorted(c):
        others=sorted(set(c)-{m});best=None
        missing=cal.evidence_bounds[m]+1
        prefix=cal.memory_table_mib[m][missing-1]+c[m].memory_mib
        for bits in itertools.product([0,1],repeat=len(others)):
            active=[n for n,b in zip(others,bits) if b]
            covered=set(c[m].supported_tasks)
            for n in active:covered.update(c[n].supported_tasks)
            if not required<=covered:continue
            memory=factor()*(prefix+sum(c[n].memory_mib for n in active))
            key=(memory,len(active),tuple(active))
            if best is None or key<best[0]:best=(key,active)
        records.append({'model':m,'role':roles[m],'first_missing_count':missing,
            'minimum_coverage_memory_screen_mib':best[0][0], 'cover':best[1],
            'can_fit_memory_screen':best[0][0]<=budget+1e-7,
            'missing_g_and_memory_not_invented':True})
    return records

def two_g_screen(c,roles,cal,background):
    gs=sorted(m for m in c if roles[m]=='generalist');ss=sorted(m for m in c if roles[m]=='specialist')
    required={k for k,v in background.items() if v};best=None
    for g1,g2 in itertools.combinations_with_replacement(gs,2):
        gc={m:[g1,g2].count(m) for m in gs};missing=[m for m,n in gc.items() if n>cal.evidence_bounds[m]]
        gm=sum(cal.memory_table_mib[m][n] if n in cal.memory_table_mib[m] else n*c[m].memory_mib for m,n in gc.items())
        for bits in itertools.product([0,1],repeat=len(ss)):
            active=[m for m,b in zip(ss,bits) if b];covered=set()
            for m in active+[g1,g2]:covered.update(c[m].supported_tasks)
            if not required<=covered:continue
            mem=factor()*(gm+sum(c[m].memory_mib for m in active));key=(mem,tuple([g1,g2]),tuple(active))
            if best is None or key<best[0]:best=(key,gc,active,missing)
    return {'safe_memory_screen_mib':best[0][0],'G_counts':best[1],'S_cover':best[2],
        'missing_generalist_counts':best[3],
        'scope':'Memory/quality cover screen only. Unmeasured G count uses two unchanged single-replica residencies; not a new M(x) or g(x) table, not a physical measurement.'}

def main():
    assert not (OUT/'input_freeze.json').exists(),'Module11A already frozen'
    c,hw,roles,cal=inputs();bgcontexts=[r for r in contexts() if r['background']=='I1'];bg=bgcontexts[0]['background_lambda']
    eligible=sorted(k for k in bg if any(k in c[m].supported_tasks and roles[m]=='specialist' for m in c)
                    and any(k in c[m].supported_tasks and roles[m]=='generalist' for m in c))
    sets=[list(s) for n in [2,3] for s in itertools.combinations(eligible,n)]
    refs=protected()
    for d in ['module10','results/module10_specialist_saturation','deployment_core','deployment_search','method_v2']:
        refs.extend(ref(p) for p in (ROOT/d).rglob('*') if p.is_file() and p.suffix!='.pyc')
    refs=list({r['path']:r for r in refs}.values());verify(refs)
    freeze={'at_utc':now(),'CPU_only':True,'new_GPU_runs':0,'background':bg,'contexts':bgcontexts,
        'eligible_tasks':eligible,'sets':sets,'rho_limit':.8,'protected_references':refs,
        'count_policy':'All measured counts; unknown g/M counts explicitly uncensored in claims, never extrapolated or called physical limits.',
        'capacity_tie_break':'max alpha, min safe memory, min total replicas, deterministic deployment vector',
        'latency_landmark_rule':['0.80*alpha_S','0.98*alpha_S','alpha_S+0.30*W_1G','alpha_S+0.70*W_1G','0.80*alpha_1G','0.98*alpha_1G'],
        'latency_landmark_scope':'All nonnegative unique landmarks strictly within alpha_1G; determined only by CPU envelopes, before any J comparison.',
        'J_policy':'Original frozen solve_e2e_routing, then separate rho<=.8 certificate. No replacement routes or solver modification.',
        'primary_tie_break':'R2 descending, W2 descending, budget ascending, deterministic task tuple; latency fallback DeltaJ descending then budget and tuple.',
        'calibration_source_hashes':cal.source_hashes,'evidence_bounds':cal.evidence_bounds}
    save('input_freeze.json',freeze)
    old=read(ROOT/'results/module10_specialist_saturation/operational_envelope_audit/capacity_envelope_operational.json')['rows']
    heads_by_budget={};direction=[];ba=[];state_map={};twos=two_g_screen(c,roles,cal,bg)
    for i,con in enumerate(bgcontexts):
        bid=f'B{i+1}';B=con['memory_budget_mib'];heads={}
        for k in eligible:
            prev=next(r for r in old if r['context']==con['id'] and r['target']==k and r['mode']=='S_ONLY')
            h=prev['capacity_rps']-bg[k]
            if h>0:heads[k]=h
            direction.append({'budget':bid,'task':k,'C_Sonly_op':prev['capacity_rps'],'background_rate':bg[k],
                 'h':h,'headroom_positive':h>0,'individual_headroom_replica_audit':prev['replica_audit_status']})
        heads_by_budget[bid]=heads
        states=enumerate_states(c,roles,cal,B,bg);state_map[bid]=states
        missing=missing_screen(c,roles,cal,B,bg)
        known2=[s for s in states if s['N_G']>=2]
        bestknown=min(known2,key=lambda s:(s['safe_memory_mib'],tuple(s['deployment'].values()))) if known2 else None
        status='TWO_G_EMPIRICAL_RESIDENCY_FEASIBLE' if bestknown else 'TWO_G_STRUCTURALLY_IMPOSSIBLE' if twos['safe_memory_screen_mib']>B+1e-7 else 'TWO_G_MEMORY_SCREEN_POSSIBLE_REPLICA_EVIDENCE_MISSING'
        ba.append({'budget':bid,'memory_budget_mib':B,'status':status,
            'physical_impossibility_scope':'frozen residency screen; no unmeasured replica memory asserted',
            'known_count_2G_cover_witness':bestknown,'optimistic_two_G_screen':twos,
            'missing_count_screen':missing,'replica_certificate':not any(r['can_fit_memory_screen'] for r in missing),
            'missing_feasible_models':[r['model'] for r in missing if r['can_fit_memory_screen']],
            'memory_coverage_states':len(states)})
    save('budget_structure_audit.json',{'budgets':ba,'GPU_runs':0})
    save('deployment_space.json',state_map)
    save('normalized_directions.json',{'formula':'lambda_k(alpha)=lambda_bg[k]+alpha*h_k','directions':direction,
         'heads_by_budget':heads_by_budget,'source':ref(ROOT/'results/module10_specialist_saturation/operational_envelope_audit/capacity_envelope_operational.json'),
         'background':bg,'outcome_driven_direction_search':False})
    setrows=[{'set_id':'+'.join(s),'size':len(s),'tasks':s} for s in sets];csvsave('hotspot_sets.csv',setrows)
    fits=[];windows=[]
    for i,con in enumerate(bgcontexts):
        bid=f'B{i+1}';cert=ba[i]['replica_certificate']
        for tasks in sets:
            if any(k not in heads_by_budget[bid] for k in tasks):continue
            sid='+'.join(tasks);heads={k:heads_by_budget[bid][k] for k in tasks};pair={}
            for mode in ['S','1G','2G']:
                f=solve(c,roles,cal,con['memory_budget_mib'],bg,heads,mode,factor());pair[mode]=f
                fits.append({'budget':bid,'set_id':sid,'tasks':tasks,'mode':mode,'headroom':heads,
                    'memory_budget_mib':con['memory_budget_mib'],'replica_certificate':cert,
                    'claim_scope':'certified under frozen residency screen' if cert else 'conditional empirical-count-space only',**f})
            s,a,b=[pair[m]['alpha'] for m in ['S','1G','2G']]
            w1=a-s if a is not None and s is not None else None;w2=b-a if b is not None and a is not None else None
            if w1 is not None and abs(w1)<1e-9:w1=0.
            if w2 is not None and abs(w2)<1e-9:w2=0.
            assert w2 is None or w2>=-1e-7
            windows.append({'budget':bid,'set_id':sid,'tasks':tasks,'alpha_S':s,'alpha_1G':a,'alpha_2G':b,
                'W_1G':w1,'W_2G':w2,'R_1G':w1/s if s and w1 is not None else None,'R_2G':w2/a if a and w2 is not None else None,
                'replica_certificate':cert,'one_G_overflow_in_empirical_space':bool(w1 is not None and w1>0),
                'second_G_necessary_in_empirical_space':bool(w2 is not None and w2>0),
                'SECOND_GENERALIST_CAPACITY_NECESSARY':bool(cert and w2 is not None and w2>0),
                'status':'CERTIFIED_UNDER_FROZEN_RESIDENCY_SCREEN' if cert else 'REPLICA_EVIDENCE_REQUIRED'})
            save('capacity_witnesses.json',{'rows':fits});csvsave('multihotspot_capacity_envelopes.csv',fits)
            save('windows.json',{'rows':windows});csvsave('second_generalist_regions.csv',windows)
            print(json.dumps({'budget':bid,'set':sid,'alpha_S':s,'alpha_1G':a,'alpha_2G':b,'certified':cert}),flush=True)
    save('capacity_stage_status.json',{'status':'COMPLETE','envelopes':len(fits),'contexts':len(windows),'GPU_runs':0})

if __name__=='__main__':main()
