"""Independent enumeration and deterministic-repeat checks, CPU only."""
import itertools
from .common import *
from .envelope import fixed_lp,solve

def main():
    c,hw,roles,cal=inputs();freeze=read(OUT/'input_freeze.json');bg=freeze['background']
    fits=read(OUT/'capacity_witnesses.json')['rows'];states=read(OUT/'deployment_space.json');windows=read(OUT/'windows.json')['rows']
    # B1: every set and mode; all other budgets: the first pair, the ASR/LID
    # shared-specialist pair, and the OCR/Text pair, fixed without GPU labels.
    selected=[w for w in windows if w['budget']=='B1' or w['set_id'] in ['asr+image','asr+lid','ocr+text']]
    checks=[];repeat=[]
    for w in selected:
        for mode in ['S','1G','2G']:
            f=next(f for f in fits if f['budget']==w['budget'] and f['set_id']==w['set_id'] and f['mode']==mode)
            best=None;count=0
            for state in states[w['budget']]:
                if mode!='S' and state['N_G']>int(mode[0]):continue
                fit=fixed_lp(state['deployment'],bg,f['headroom'],c,roles,cal,mode)
                count+=1
                if fit is not None:best=max(best or 0.,fit['alpha'])
            assert best is not None and abs(best-f['alpha'])<1e-7,(w,mode,best,f['alpha'])
            checks.append({'budget':w['budget'],'set_id':w['set_id'],'mode':mode,'enumerated_states':count,
                'enumeration_alpha':best,'MILP_alpha':f['alpha'],'absolute_difference':abs(best-f['alpha'])})
        if w['set_id']=='asr+lid':
            for mode in ['S','1G','2G']:
                f=next(f for f in fits if f['budget']==w['budget'] and f['set_id']==w['set_id'] and f['mode']==mode)
                r=solve(c,roles,cal,f['memory_budget_mib'],bg,f['headroom'],mode,factor())
                assert r['deployment']==f['deployment'] and r['routing']==f['routing'] and abs(r['alpha']-f['alpha'])<1e-10
                repeat.append({'budget':w['budget'],'mode':mode,'deployment_routing_alpha_exact_repeat':True})
        print(json.dumps({'validation_context':w['budget']+'/'+w['set_id']}),flush=True)
    necessity=[]
    for w in windows:
        if not w['second_G_necessary_in_empirical_space']:continue
        f=next(f for f in fits if f['budget']==w['budget'] and f['set_id']==w['set_id'] and f['mode']=='2G')
        alpha=w['alpha_1G']+.5*w['W_2G']
        witness=fixed_lp(f['deployment'],bg,f['headroom'],c,roles,cal,'2G',alpha=alpha)
        assert witness is not None and alpha>w['alpha_1G']
        necessity.append({'budget':w['budget'],'set_id':w['set_id'],'alpha':alpha,
            'two_G_fixed_LP_feasible':True,'one_G_MILP_upper_bound':w['alpha_1G'],
            'replica_certificate':w['replica_certificate'],'physical_necessity_claimed':w['replica_certificate']})
    save('cpu_validation.json',{'status':'PASS','count_space_size':math.prod(cal.evidence_bounds[m]+1 for m in c),
        'envelopes_with_full_constraint_witnesses':len(fits),'independent_enumeration_checks':checks,
        'deterministic_repeat_checks':repeat,'two_G_region_midpoint_checks':necessity,'new_GPU_runs':0})

if __name__=='__main__':main()
