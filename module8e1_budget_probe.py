"""Conditional evaluation-slot audit of shortest generated paths, no runtime."""
from module8e1_analysis import OUT,NAMES,C,CAT,POLICY,Planner,read,save_json,counts,key,rates,Workload,verify_v2
from method_v2.selection import diversified_order
verify_v2();pl=Planner();out=[]
for n in NAMES:
    path=read(OUT/'b2_path_witnesses.json')['workloads'][n]['v2']['path']
    x=counts(read(C/n/'search_complete.json')['seed']['deployment']);visited={key(x)};used=1;steps=[];possible=True
    for index,mutation in enumerate(path,1):
        mus,ranked,audit=pl.candidates(n,x)
        remaining=[c for c in ranked if key(c.mutation.deployment) not in visited]
        selected=diversified_order(remaining,x,CAT,Workload(rates(n)),POLICY.deployment_memory(x))[:2]
        slot=next((i for i,c in enumerate(selected,1) if key(c.mutation.deployment)==key(mutation['deployment'])),None)
        slots=min(len(selected),6-used);available=slot is not None and slot<=slots
        steps.append({'step':index,'target_mutation':mutation,'TopB_slot':slot,'runtime_slots_available':slots,'path_step_evaluable':available,'budget_used_before':used,'selected_states':[c.mutation.deployment for c in selected[:slots]]})
        if not available:possible=False;break
        # Hypothetical path acceptance only, not inferred from any new metric.
        used+=slots;visited.update(key(c.mutation.deployment) for c in selected[:slots]);x=counts(mutation['deployment'])
    out.append({'workload':n,'specific_shortest_witness_evaluable_if_each_path_state_were_accepted':possible,'slots_used_or_reserved':used,'steps':steps,'other_generated_paths_not_ruled_out':True})
save_json(OUT/'shortest_path_slot_audit.json',{'status':'COUNTERFACTUAL_PLANNING_ONLY','workloads':out,'runtime_acceptance_assumed_not_measured':True,'budget':{'states_including_seed':6,'iterations':3,'top_b':2,'patience':1},'no_runtime_imputation':True,'warning':'Three generated mutations can require seven states if both slots are used every iteration. Generation-path compression alone never proves budgeted runtime reachability.'})
print([(r['workload'],r['specific_shortest_witness_evaluable_if_each_path_state_were_accepted']) for r in out],flush=True)
