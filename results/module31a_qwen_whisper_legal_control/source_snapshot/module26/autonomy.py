"""Stage C only after actual replication: original fixed-pool optimizer checks."""
from .common import *
from module20.optimizer import predict,Generator
from module25.predictor import UtilityDifferencePredictor
import module24.audit as optimizer

class HighRates(dict):
    def __missing__(self,key):
        if key=='LOW':return self['HIGH']
        raise KeyError(key)

def main():
    result=read(OUT/'replication_result.json');assert result['Stage_C_permitted'],'No CPU search before repeated positive evidence'
    assert not (OUT/'autonomous_optimizer_check.json').exists(),'No completed optimizer rerun'
    d,refs,weights=inputs();manifest=read(OUT/'frozen_replication_manifest.json');rates=manifest['slots'][0]['lambda_rps']
    optimizer.RATES=HighRates(HIGH=rates);optimizer.save=save
    old=optimizer.search(d,refs,weights,label='stage_c_original20A')
    model_path=ROOT/'results/module25_utility_difference/colocation_prediction_model.json'
    model=UtilityDifferencePredictor.load(model_path);digest=ref(model_path)
    def frozen_replay(d,x,r,rates,budget,refs,weights,model=None,detail=True,**kwargs):
        p=model.predict(d,x,r,rates,budget,refs,weights,hardware_uuid=manifest['plan']['GPU']['uuid'],detail=detail)
        p['max_rho']=max(p.get('rho',{}).values(),default=1.)
        return p,p.get('service_paths',{})
    # Swap only the scorer in this process; no historical source files edited.
    optimizer.replay=frozen_replay
    new=optimizer.search(d,refs,weights,model=model,label='stage_c_frozen25')
    # Retain unchanged formal MILP-generated seeds in the new scorer competition too.
    newstage=new['stages']['HIGH'];extras=[]
    for cap in [.8,.95]:
        c=Generator(d,refs).solve(rates,BUDGET,'asr',None,cap,weights)
        if 'deployment' in c:
            c['prediction']=model.predict(d,c['deployment'],c['routing'],rates,BUDGET,refs,weights)
            c['ASR_G_fraction']=c['routing']['asr'].get(G,0)/rates['asr'];extras.append(c)
    winner=optimizer.best([c for c in [newstage['winner'],*extras] if c]);newstage['winner']=winner
    if winner:
        f=winner['ASR_G_fraction'];newstage['choice']='G-only' if f>=1-1e-9 else 'S-only' if f<=1e-9 else 'G/S'
    newstage['unchanged_Module20_Generator_seeds_also_competed']=extras
    proposed=next(s for s in manifest['slots'] if s['arm']=='proposed');control=next(s for s in manifest['slots'] if s['arm']=='control')
    x=proposed['deployment'];rr=proposed['routing'];rows=[];sensitivity=[]
    for label,res in [('A20',old),('FROZEN25',new)]:
        score=lambda x,r:predict(d,x,r,rates,BUDGET,refs,weights,phase=.37) if label=='A20' else model.predict(d,x,r,rates,BUDGET,refs,weights)
        candidates=[('ACTUALLY_REPLICATED_GS18',x,rr),('ACTUALLY_REPLICATED_S_ONLY',x,control['routing'])]
        for record in res['stages']['HIGH']['candidates']:
            if record['winner']:candidates.append((record['candidate_id'],record['deployment'],record['winner']['routing']))
        for name,xx,r in candidates:
            p=score(xx,r);rows.append(dict(algorithm=label,candidate=name,deployment=xx,routing=r,
                predicted_U=p['U'],eligible=p['eligible'],prediction_supported=p.get('support') if label!='A20' else None,
                actual_independent_U=float(np.mean([r['actual_U'] for r in result['runs'] if r['arm']==('proposed' if name=='ACTUALLY_REPLICATED_GS18' else 'control')])) if name.startswith('ACTUALLY_REPLICATED') else None,
                measured_candidate= name.startswith('ACTUALLY_REPLICATED')))
        for f in manifest['Stage_C']['route_sensitivity_grid']:
            r=copy.deepcopy(rr);r['asr']={G:rates['asr']*f,S:rates['asr']*(1-f)}
            a=score(x,r);b=score(x,control['routing'])
            sensitivity.append(dict(algorithm=label,ASR_G_fraction=f,predicted_U=a['U'],
                predicted_delta_U=a['U']-b['U'] if a['U'] is not None and b['U'] is not None else None,
                actual_route_tested=f==.18,nearby_CPU_predictions_not_empirical_evidence=True))
    def exact_replicated_candidate(c):
        if not c or c['deployment']!=x:return False
        return all(abs(c['routing'].get(t,{}).get(m,0)-rr[t].get(m,0))<1e-6 for t in rates for m in set(c['routing'].get(t,{}))|set(rr[t]))
    oldmatch=exact_replicated_candidate(old['stages']['HIGH']['winner']);newmatch=exact_replicated_candidate(new['stages']['HIGH']['winner'])
    doc=dict(at_utc=now(),Stage_C_executed=True,original20A=old,frozen25=new,
        actually_replicated_candidate=dict(deployment=x,routing=rr,original_frozen_ASR_G_fraction=.18),
        original20A_selects_replicated_candidate=oldmatch,frozen25_selects_replicated_candidate=newmatch,
        autonomous_replicated_candidate_selected=oldmatch or newmatch,
        matching_tolerance_absolute_route_rps=1e-6,nearby_split_not_equivalent_to_actually_validated_route=True,
        no_new_models_no_fitted_parameters=True,model25= digest,model25_bytes_unchanged=ref(model_path)==digest,
        original20A_forecast_horizon_seconds=120.,frozen25_forecast_horizon_seconds=120.,GPU_replication_horizon_seconds=2400.,
        horizon_limitation='Original predictor internals retained;120s nominal throughput horizons differ from2400s experimental window. No new predictor is developed to conceal this limitation.',
        sensitivity_scope='CPU only around16-20 percent; only18 percent has independent GPU evidence',
        other_legal_configurations_may_have_higher_predicted_U_but_unmeasured=True,
        no_new_pool_search=True,original_file_modifications=False)
    csvsave('autonomous_candidate_ranking.csv',rows);csvsave('routing_sensitivity.csv',sensitivity);save('autonomous_optimizer_check.json',doc)
    assert ref(model_path)==digest
    print('STAGE_C_MATCH','A20',oldmatch,'frozen25',newmatch,flush=True)

if __name__=='__main__':main()
