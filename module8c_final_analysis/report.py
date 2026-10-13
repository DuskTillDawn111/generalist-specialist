import csv,json,statistics,math
from pathlib import Path
from module8c_final_real_search.common import OUT,B,P,ROOT,read,ref,verify_freeze
from deployment_search.runtime_cache import save_json,file_hash
from joint_deploy.measurement.metrics import percentile

def csvwrite(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})

def main():
    verify_freeze()
    summary=read(OUT/'module8c_summary.json')
    assert len(summary['workloads'])==6 and read(OUT/'campaign_status.json')['status']=='complete'
    states=read(OUT/'runtime_state_inventory.json');family_rows=[];task_rows=[]
    official={}
    for state in states:
        name=state['workload'];cache=read(OUT/name/'search_cache/runtime_cache_manifest.json')
        identity=cache['configurations'][state['cache_key']]['identity']
        rates=identity['workload'];total=sum(rates.values())
        runtime=state['runtime'];official[(name,state['cache_key'])]=runtime['weighted_mean_e2e']
        for task,lam in rates.items():
            if lam<=0:continue
            mu=runtime['per_task_mean'][task]['mean'];p95=runtime['per_task_p95'][task]['mean']
            task_rows.append({'workload':name,'state_cache_key':state['cache_key'],'task':task,'lambda_rps':lam,
                'request_share':lam/total,'task_mean_e2e_ms':mu,'task_p95_ms':p95,
                'weighted_mean_contribution_ms':lam/total*mu,'weighted_p95_contribution_ms':lam/total*p95,
                'deployment':state['deployment']})
        if rates['visual_qa']<=0:continue
        for model in ('qwen','phi4mm','minicpm'):
            reps=[]
            for rep in runtime['repetitions']:
                assert file_hash(rep['raw_result_path'])==rep['raw_sha256']
                raw=read(rep['raw_result_path'])
                requests=[r for r in raw['request_records'] if r['task']=='visual_qa' and r['model_family']==model]
                completed=[r for r in requests if r['status']=='completed']
                values=[r['e2e_latency_ms'] for r in completed]
                workers=[r['worker_latency_ms'] for r in completed]
                reps.append({'repetition':rep['repetition'],'requests':len(requests),'completed':len(completed),
                    'mean_e2e_ms':statistics.mean(values) if values else None,'p95_e2e_ms':percentile(values,95),
                    'mean_worker_latency_ms':statistics.mean(workers) if workers else None,
                    'source':ref(rep['raw_result_path'])})
            available=[r for r in reps if r['mean_e2e_ms'] is not None]
            family_rows.append({'workload':name,'state_cache_key':state['cache_key'],'model':model,
                'model_count':state['deployment'][model],'planned_vqa_routing_rps':identity['routing']['visual_qa'][model],
                'observed_vqa_requests':sum(r['requests'] for r in reps),'observed_completed_vqa_requests':sum(r['completed'] for r in reps),
                'vqa_actually_served':bool(available),'family_vqa_mean_e2e_ms':statistics.mean(r['mean_e2e_ms'] for r in available) if available else None,
                'family_vqa_mean_of_rep_p95_ms':statistics.mean(r['p95_e2e_ms'] for r in available) if available else None,
                'valid_observation_repetitions':len(available),'system_mean_e2e_ms':runtime['weighted_mean_e2e'],
                'system_weighted_task_p95_ms':runtime['weighted_p95_e2e'],'system_stability':runtime['stability_state_v2'],
                'deployment':state['deployment'],'repetitions':reps})
    for (name,key),mu in official.items():
        component=sum(r['weighted_mean_contribution_ms'] for r in task_rows if r['workload']==name and r['state_cache_key']==key)
        assert math.isclose(mu,component,rel_tol=1e-10,abs_tol=1e-9)
    csvwrite(OUT/'generalist_vqa_runtime_breakdown.csv',family_rows)
    save_json(OUT/'generalist_vqa_runtime_breakdown.json',{'rows':family_rows,
       'aggregation':'Equal average of observed per-repetition family-VQA means/p95s; diagnostic only, no objective replacement',
       'unserved_models_have_null_metrics':True,'search_acceptance_modified':False})
    csvwrite(OUT/'runtime_task_contributions.csv',task_rows)
    reasoning=[r for r in family_rows if r['workload']=='reasoning_vqa_heavy' and r['vqa_actually_served']]
    no=next(r for r in summary['workloads'] if r['workload']=='no_vqa_phase')
    lines=['# Final deployment diversity and Generalist runtime evidence','',
       f"Six complete workloads; distinct final vectors={summary['final_deployment_diversity']['distinct_final_deployments']}.",
       'Actual served VQA requests are checked separately from model presence. Family timings below are diagnostics over only requests assigned to that family; input subsets may differ under routing, and co-deployment is not a single-Generalist replacement trial. Official acceptance still uses the unchanged whole-workload objective.','',
       '## Reasoning/VQA-heavy actual serving','',
       '|State|Model|Count|VQA λ|Observed requests (3 reps)|Family VQA mean ms|Family mean-of-P95 ms|Whole-workload mean ms|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in reasoning:lines.append(f"|{r['state_cache_key'][:12]}|{r['model']}|{r['model_count']}|{r['planned_vqa_routing_rps']:.6f}|{r['observed_vqa_requests']}|{r['family_vqa_mean_e2e_ms']:.3f}|{r['family_vqa_mean_of_rep_p95_ms']:.3f}|{r['system_mean_e2e_ms']:.3f}|")
    models=sorted({r['model'] for r in reasoning})
    no_generalist_decisions=[r for r in read(OUT/'no_vqa_phase/integer_replica_decisions.json')
                            if r['candidate'] in ('qwen','phi4mm','minicpm') and r['runtime_evaluated']]
    lines+=['',f'Actually observed VQA-serving Generalists in the visited reasoning phase: {models}. No unvisited singleton superiority is inferred.',
       '', '## No-VQA phase','',f"Final N_G={no['N_G']}, N_S={no['N_S']}; search stability={no['final_stability']}, independent confirmation stability={no['confirmation_final_stability']}. Generalists remained available to the unchanged candidate generator; no N_G=0 constraint was imposed.",
       '', '|Actually evaluated Generalist mutation|Mean E2E ms|Weighted task P95 ms|Stability|Decision|Reason|',
       '|---|---:|---:|---|---|---|']
    for r in no_generalist_decisions:
        lines.append(f"|{r['candidate']} x={r['old_count']}→{r['new_count']}|{r['real_mean_e2e_ms']:.3f}|{r['real_weighted_task_p95_ms']:.3f}|{r['stability']}|{r['decision']}|{r['reason']}|")
    lines+=['', 'These observations cover only visited states; no universal Generalist disadvantage or unvisited Phi/MiniCPM result is inferred.',
       '', '## Interpretation','',
       'The frozen workload suite may lead to repeated final deployments. Those repetitions and objective rollbacks are experimental results, not a reason to retune λ. Per-task weighted contribution reconstruction matches every official state mean. Full phase/model observations, null values for unserved models, routing rates and raw hashes are retained in the accompanying JSON/CSV.',
       'VQA/LID numeric quality thresholds remain UNSPECIFIED under the authorized semantic-only policy. This analysis adds neither quality measurements nor thresholds and does not affect admission or decisions.']
    (OUT/'final_deployment_analysis.md').write_text('\n'.join(lines)+'\n')
    save_json(OUT/'post_search_analysis_validation.json',{'status':'PASS','source':ref(Path(__file__)),
       'runtime_states_reconstructed':len(states),'mean_contributions_match':True,
       'reasoning_vqa_actually_serving_generalists':models,'search_decisions_modified':False,
       'no_VQA_actual_generalist_mutations_analyzed':len(no_generalist_decisions),
       'new_gpu_runs':0,'new_quality_measurements':0,'baseline_or_ablation_runs':0})
    print('Post-search analysis PASS; actual reasoning VQA-serving models:',models)
if __name__=='__main__':main()
