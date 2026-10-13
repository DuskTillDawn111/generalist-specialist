"""CPU-only descriptive attribution, executed after the frozen campaign ends.

Request matching decomposes a run's mean; requests never become deployment samples.
This file does not change the frozen experiment implementation or acceptance rules.
"""
from pathlib import Path
import json
import csv
import statistics

OUT = Path(__file__).resolve().parent
G = 'qwen'
S = 'facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a'

def read(p):return json.loads(Path(p).read_text())

def main():
    status=read(OUT/'campaign_status.json')
    assert not status['pending_runs'], 'Do not attribute a still-running campaign'
    data=read(OUT/'real_utility_analysis.json')
    stats=read(OUT/'paired_utility_statistics.json')['stages']
    tasks={(r['stage'],r['repetition'],r['arm'],r['task']):r for r in data['tasks']}
    rows=[]
    for stage in ['MID','HIGH']:
        for rep in [1,2,3]:
            pair=stats[stage]['pairs'][rep-1]
            for task in ['text','image','ocr','asr','lid','visual_qa']:
                a=tasks.get((stage,rep,'proposed',task));b=tasks.get((stage,rep,'control',task))
                if not a or not b:continue
                row={'stage':stage,'repetition':rep,'task':task,'normal_utility_comparison_valid':pair['comparison_valid']}
                for key in ['offered','accepted_submitted','completed','within_window_completed','mean_E2E_ms','P95_E2E_ms',
                            'mean_worker_service_ms','P95_worker_service_ms','mean_queue_ms','P95_queue_ms',
                            'L_norm_contribution','P_norm_contribution','L_utility_contribution','P_utility_contribution']:
                    row['proposed_'+key]=a[key];row['control_'+key]=b[key]
                    row['delta_'+key]=a[key]-b[key] if a[key] is not None and b[key] is not None else None
                if pair['comparison_valid']:
                    row['delta_U_latency']=row['delta_L_utility_contribution']+row['delta_P_utility_contribution']
                else:row['delta_U_latency']=None
                rows.append(row)
    with (OUT/'per_task_paired_differences.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)
    splits=[]
    for rep in [1,2,3]:
        pa,pb=OUT/f'raw_runs/high_proposed_rep{rep}.json',OUT/f'raw_runs/high_control_rep{rep}.json'
        if not pa.exists() or not pb.exists():
            splits.append({'repetition':rep,'reason':'Missing run; no fabricated mean attribution'});continue
        a,b=read(pa),read(pb)
        if not a['phases'].get('measurement') or not b['phases'].get('measurement'):
            splits.append({'repetition':rep,'reason':'Aborted phase; see raw safety records'});continue
        aa=[r for r in a['phases']['measurement']['request_records'] if r['task']=='lid']
        bb={r['request_id']:r for r in b['phases']['measurement']['request_records'] if r['task']=='lid'}
        if any(r['status']!='completed' for r in aa+list(bb.values())):
            splits.append({'repetition':rep,'reason':'Incomplete LID cohort; no uncensored attribution'});continue
        assert len(aa)==len(bb)
        groups={'G_routed':[],'S_shared':[]}
        for ar in aa:
            br=bb[ar['request_id']]
            assert ar['sample_id']==br['sample_id'] and ar['frozen_offset_seconds']==br['frozen_offset_seconds']
            assert b['replica_to_family'][br['model']]==S
            group='G_routed' if a['replica_to_family'][ar['model']]==G else 'S_shared'
            groups[group].append((ar,br))
        taskrow=tasks['HIGH',rep,'proposed','lid']
        coefficient=-.3*taskrow['offered_weight']/taskrow['reference_ms']
        attribution={'repetition':rep,'full_completed_LID_n':len(aa),'normal_utility_comparison_valid':stats['HIGH']['pairs'][rep-1]['comparison_valid']}
        for name,pairs in groups.items():
            attribution[name]={'n':len(pairs),'fraction':len(pairs)/len(aa),
                'mean_E2E_change_ms':statistics.mean(ar['e2e_latency_ms']-br['e2e_latency_ms'] for ar,br in pairs),
                'mean_service_change_ms':statistics.mean(ar['worker_latency_ms']-br['worker_latency_ms'] for ar,br in pairs),
                'mean_queue_change_ms':statistics.mean(ar['worker_queue_wait_ms']-br['worker_queue_wait_ms'] for ar,br in pairs),
                'exact_contribution_to_LID_U_mean_delta':coefficient*sum(ar['e2e_latency_ms']-br['e2e_latency_ms'] for ar,br in pairs)/len(aa)}
        attribution['LID_U_mean_subgroup_sum']=sum(attribution[k]['exact_contribution_to_LID_U_mean_delta'] for k in groups)
        full_delta=tasks['HIGH',rep,'proposed','lid']['L_utility_contribution']-tasks['HIGH',rep,'control','lid']['L_utility_contribution']
        assert abs(attribution['LID_U_mean_subgroup_sum']-full_delta)<1e-12
        splits.append(attribution)
    output={'statistical_unit':'Independent GPU pair, n<=3; matched requests only give within-run additive mean attribution, never extra deployment sample size',
            'not_a_transferable_fit_or_causal_isolation_model':True,
            'P95_attribution':'Quantiles are not additive. Use whole-task paired P95 contribution, no sum of subgroup P95s.',
            'HIGH_mean_split_attribution':splits,
            'per_task_average_delta_U_mean_plus_P95':{stage:{task:statistics.mean(r['delta_U_latency'] for r in rows if r['stage']==stage and r['task']==task and r['delta_U_latency'] is not None)
                for task in ['text','image','ocr','asr','lid','visual_qa'] if any(r['stage']==stage and r['task']==task and r['delta_U_latency'] is not None for r in rows)} for stage in ['MID','HIGH']}}
    (OUT/'task_and_split_diagnosis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
