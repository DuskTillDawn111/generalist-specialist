"""Minimal-memory feasible boundary witnesses and local expansion alternatives."""
from .common import *
from .envelope import fixed_lp

def main():
    c,hw,roles,cal=inputs();freeze=read(OUT/'input_freeze.json');bg=freeze['background']
    windows=read(OUT/'windows.json')['rows'];fits=read(OUT/'capacity_witnesses.json')['rows'];states=read(OUT/'deployment_space.json')
    budgetaudit={b['budget']:b for b in read(OUT/'budget_structure_audit.json')['budgets']};rows=[]
    for w in windows:
        f=next(f for f in fits if f['budget']==w['budget'] and f['set_id']==w['set_id'] and f['mode']=='1G')
        pool=sorted(states[w['budget']],key=lambda s:(s['safe_memory_mib'],sum(s['deployment'].values()),tuple(sorted(s['deployment'].items()))))
        def minimum(alpha,ng):
            for state in pool:
                if (ng==1 and state['N_G']>1) or (ng==2 and state['N_G']!=2):continue
                fit=fixed_lp(state['deployment'],bg,f['headroom'],c,roles,cal,'1G' if ng==1 else '2G',alpha=alpha)
                if fit is not None:return fit
            return None
        low=minimum(.8*w['alpha_S'],1);near=minimum(.98*w['alpha_1G'],1);second=minimum(.98*w['alpha_1G'],2)
        assert low and near
        x=near['deployment'];alternatives=[]
        for m in sorted(c):
            if roles[m]!='specialist' or not (set(w['tasks'])&c[m].supported_tasks):continue
            n=x[m]+1
            if n<=cal.evidence_bounds[m]:
                y={**x,m:n};memory=safe_memory(y,cal);fitsmem=memory<=f['memory_budget_mib']+1e-7
                cap=fixed_lp(y,bg,f['headroom'],c,roles,cal,'1G') if fitsmem else None
                alternatives.append({'kind':'SPECIALIST_IDENTITY_ADD' if not x[m] else 'SPECIALIST_REPLICA_ADD',
                    'model':m,'next_count':n,'memory_mib':memory,'memory_feasible':fitsmem,'g_measured':True,
                    'resulting_max_alpha':cap['alpha'] if cap else None})
            else:
                memory=safe_memory(x,cal)+factor()*c[m].memory_mib
                alternatives.append({'kind':'SPECIALIST_REPLICA_ADD_UNMEASURED','model':m,'next_count':n,
                    'memory_screen_mib':memory,'memory_screen_feasible':memory<=f['memory_budget_mib']+1e-7,
                    'g_measured':False,'resulting_max_alpha':None})
        increased={m:[low['deployment'][m],x[m]] for m in c if roles[m]=='specialist' and x[m]>low['deployment'][m]}
        removed={m:[low['deployment'][m],x[m]] for m in c if roles[m]=='specialist' and x[m]<low['deployment'][m]}
        reasons=[];b=budgetaudit[w['budget']]
        if b['status']=='TWO_G_STRUCTURALLY_IMPOSSIBLE':reasons.append('TWO_G_STRUCTURALLY_IMPOSSIBLE')
        elif b['status']=='TWO_G_MEMORY_SCREEN_POSSIBLE_REPLICA_EVIDENCE_MISSING':reasons.append('GENERALIST_REPLICA_EVIDENCE_MISSING')
        if not w['replica_certificate']:reasons.append('SPECIALIST_REPLICA_EVIDENCE_INCOMPLETE_NO_CAUSAL_EXHAUSTION_CLAIM')
        if second and near['safe_memory_mib']<second['safe_memory_mib']:
            reasons.append('ONE_G_PLUS_SPECIALISTS_HAS_LOWER_MINIMUM_FEASIBLE_MEMORY_AT_NEAR_BOUNDARY')
        if w['W_2G']>0:reasons.append('SECOND_G_EXTENDS_EMPIRICAL_COUNT_SPACE_CAPACITY')
        elif second:reasons.append('ONE_G_SUFFICIENT_AT_AUDITED_BOUNDARY_NO_EMPIRICAL_CAPACITY_EXTENSION')
        elif b['known_count_2G_cover_witness']:reasons.append('TWO_G_RESIDENCY_EXISTS_BUT_OPERATIONAL_BOUNDARY_INFEASIBLE')
        row={'budget':w['budget'],'set_id':w['set_id'],'tasks':w['tasks'],'near_alpha':.98*w['alpha_1G'],
            'low_alpha':.8*w['alpha_S'],'low_minimum_memory_witness':low,'near_1G_minimum_memory_witness':near,
            'near_exact_2G_minimum_memory_witness':second,'S_count_increases_from_low':increased,'S_count_decreases_from_low':removed,
            'S_identity_change_from_low':bool(set(low['specialist_replicas'])!=set(near['specialist_replicas'])),
            'local_specialist_expansion_alternatives':alternatives,'structural_reasons':reasons,
            'routing_redistribution_low_to_near':{'low_G_fraction':low['hotspot_G_fraction'],'near_G_fraction':near['hotspot_G_fraction']},
            'candidate_generation_limitation':'NOT_AUDITED_AS_RUNTIME_SEARCH: full frozen evidence-space enumeration; no neighborhood changed',
            'specialists_too_fast_causal_claim':False,'replica_certificate':w['replica_certificate']}
        rows.append(row)
        print(json.dumps({'structural_context':w['budget']+'/'+w['set_id'],'reasons':reasons}),flush=True)
    save('structural_witnesses.json',{'rows':rows});csvsave('structural_failure_reasons.csv',rows)

if __name__=='__main__':main()
