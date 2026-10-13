"""Offline descriptive sub-bin and episode analysis of normalized event exports."""
import sys
sys.dont_write_bytecode=True
import csv,json,collections
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
def save(name,x):(P/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvsave(name,rows):
    with (P/name).open('w',newline='') as s:
        w=csv.DictWriter(s,fieldnames=list(rows[0]) if rows else ['no_observations']);w.writeheader();w.writerows(rows)
def avg(values):return float(np.mean(values)) if values else None
def main():
    records=list(csv.DictReader((P/'normalized_events_and_latencies.csv').open()))
    for r in records:
        for k in ['scheduled_s','submitted_s','completion_s','native_start_s','native_end_s','handler_service_ms','worker_wait_transport_ms']:r[k]=float(r[k])
    f=json.loads((Path('/root/autodl-tmp/generalist-specialist/results/module28_dynamic_split')/'router_parameters.json').read_text());S=f['S_worker'];G=f['G_worker']
    micro=[];episodes=[];detail={}
    curves=json.loads((P/'pending_curves.json').read_text())
    for name in ['S_only','Dynamic']:
        rs=[r for r in records if r['run']==name]
        for left in range(0,600,5):
            right=left+5;sent=[r for r in rs if left<=r['submitted_s']<right];done=[r for r in rs if left<=r['completion_s']<right]
            busy=lambda model,task=None:sum(max(0.,min(r['native_end_s'],right)-max(r['native_start_s'],left)) for r in rs if r['worker']==model and (task is None or r['task']==task))
            micro.append(dict(run=name,left_s=left,right_s=right,sent=len(sent),responses=len(done),net_increase=len(sent)-len(done),
                S_sent=sum(r['worker']==S for r in sent),S_responses=sum(r['worker']==S for r in done),
                LID_offered=sum(left<=r['scheduled_s']<right and r['task']=='lid' for r in rs),ASR_offered=sum(left<=r['scheduled_s']<right and r['task']=='asr' for r in rs),
                S_native_busy_seconds=busy(S),S_ASR_native_busy_seconds=busy(S,'asr'),G_native_busy_seconds=busy(G),G_target_native_busy_seconds=busy(G,'lid')))
        for threshold in [10,20]:
            points=curves[name]['S']['B'];begin=None;max_b=0
            for t,b in points:
                if t>600:break
                if b>=threshold:
                    if begin is None:begin=t;max_b=b
                    max_b=max(max_b,b)
                elif begin is not None:
                    episodes.append(dict(run=name,threshold=threshold,left_s=begin,right_s=t,duration_s=t-begin,peak_submitted_S_pending=max_b,truncated_at_600=False));begin=None
            if begin is not None:episodes.append(dict(run=name,threshold=threshold,left_s=begin,right_s=600.,duration_s=600.-begin,peak_submitted_S_pending=max_b,truncated_at_600=True))
        detail[name]={}
        for left,right in [(450,480),(480,510),(510,540),(540,570),(570,600)]:
            rr=[r for r in rs if r['worker']==S]
            ss=sum(left<=r['submitted_s']<right for r in rr);dd=sum(left<=r['completion_s']<right for r in rr)
            occ=sum(max(0.,min(r['native_end_s'],right)-max(r['native_start_s'],left)) for r in rr)
            detail[name][f'{left}-{right}']=dict(S_sent=ss,S_response=dd,S_offered_rps=ss/30.,S_response_rps=dd/30.,S_busy_fraction=occ/30.,
                S_ASR_busy_s=sum(max(0.,min(r['native_end_s'],right)-max(r['native_start_s'],left)) for r in rr if r['task']=='asr'))
    csvsave('microburst_5s_bins.csv',micro);csvsave('high_S_pending_episodes.csv',episodes);save('tail_native_pressure.json',detail)
    source=Path('/root/autodl-tmp/generalist-specialist/results/module29_offline_opportunity/stage2_20261009T153923Z')
    bg=list(csv.DictReader((source/'G_background_queue_context.csv').open()))
    at=[r for r in bg if json.loads(r['G_target_executing_at_submit_ids'])];not_at=[r for r in bg if not json.loads(r['G_target_executing_at_submit_ids'])]
    save('background_context_summary.json',dict(n=len(bg),G_target_executing_at_submit_n=len(at),
        queue_mean_at_ms=avg([float(r['worker_queue_ms']) for r in at]),queue_mean_not_at_ms=avg([float(r['worker_queue_ms']) for r in not_at]),
        scope='Retrospective serial-order association; not causal added background delay'))
    # Cluster sensitivity interval accounts for repeated source samples, but still assumes exchangeable selected clusters.
    raw=json.loads((source/'gpu_validation_raw.json').read_text());g=[r for r in raw['phases']['measurement']['request_records'] if r['task']=='lid' and r['model']==G]
    grouped=collections.defaultdict(list)
    for r in g:grouped[r['sample_id']].append(r['worker_response']['correct'])
    values=np.array([[sum(v),len(v)] for _,v in sorted(grouped.items())]);rng=np.random.default_rng(300030)
    draws=values[rng.integers(0,len(values),size=(10000,len(values)))].sum(axis=1);rates=draws[:,0]/draws[:,1]
    save('G_quality_cluster_sensitivity.json',dict(request_correct=61,request_n=67,unique_source_samples=len(values),
        nominal_Wilson95=json.loads((P/'audit_findings.json').read_text())['G_LID_Wilson95'],
        exploratory_sample_cluster_bootstrap_percentile95=[float(np.percentile(rates,2.5)),float(np.percentile(rates,97.5))],
        replicates=10000,analysis_RNG_seed=300030,workload_seeds_untouched=True,
        caveat='27 selected source-sample clusters are not independent GPU replications. Exchangeability/representativeness unverified; neither interval certifies general task quality.'))
    print(json.dumps(dict(tail_pressure=detail,background=json.loads((P/'background_context_summary.json').read_text()),
        cluster_quality=json.loads((P/'G_quality_cluster_sensitivity.json').read_text()),longest_tail_episodes=sorted([r for r in episodes if r['threshold']==10 and r['right_s']>=450],key=lambda x:x['duration_s'],reverse=True)[:8])))
if __name__=='__main__':main()
