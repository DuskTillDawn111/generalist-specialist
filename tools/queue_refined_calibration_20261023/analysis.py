"""Whole-run comparison of declared native policies, retaining unknown outcomes.

Repeated requests are never treated as independent experimental replicates.
Exact finite-table ranking is distinct from global physical optimality.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.queue_refined_calibration_20261023.protocol import digest

TERMS='QTLPME'


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def describe(values):
    if not values:return None
    n=len(values)
    return dict(n_whole_run_pairs=n,mean=statistics.mean(values),median=statistics.median(values),
        minimum=min(values),maximum=max(values),sample_sd=statistics.stdev(values) if n>1 else None,
        positive=sum(v>0 for v in values),negative=sum(v<0 for v in values),zero=sum(v==0 for v in values),
        leave_one_out_means=[statistics.mean(values[:i]+values[i+1:]) for i in range(n)] if n>1 else [],
        uncertainty_scope='Whole paired runs only; different seeds do not establish independence of hardware/service drift.')


def read_run(path):
    path=Path(path).resolve();raw=json.loads(path.read_text());utility=raw.get('evaluation',{}).get('utility',{})
    contract=raw['policy_contract'];seed=raw['transport_config']['random_seed']
    symbolic=dict(deployment=contract['deployment'],strategy=contract['strategy'],policy_contract_sha256=digest(contract))
    candidate_id=digest(symbolic)
    requests=raw.get('request_records',[]);window=raw.get('measurement_window') or {}
    cohort=[dict(request_id=r['request_id'],task=r['task'],pool_id=r['pool_id'],
        scheduled_offset_ns=r['scheduled_time_ns']-window['start_time_ns']) for r in requests]
    counts=Counter((r['task'],raw['replica_to_family'].get(r.get('model'),'UNKNOWN')) for r in requests)
    totals=Counter(r['task'] for r in requests)
    terms={t:utility.get(t+'_norm') for t in TERMS};weights=raw['config']['weights']
    contributions={t:-weights[t]*value if finite(value) else None for t,value in terms.items()}
    if utility.get('eligible') is True:
        verified=['quality_identity_verified','cohort_verified','constraints_verified','memory_verified','energy_verified']
        if utility.get('utility_complete') is not True or any(utility.get(k) is not True for k in verified):
            raise ValueError('Eligible record lacks verified six-term evidence: '+str(path))
        if not all(finite(v) for v in terms.values()) or not finite(utility.get('U')):
            raise ValueError('Eligible record has incomplete utility: '+str(path))
        if not math.isclose(sum(contributions.values()),utility['U'],abs_tol=1e-10):
            raise ValueError('Six-term arithmetic mismatch: '+str(path))
    alias_summaries={}
    for alias in raw['replica_to_family']:
        rows=[r for r in requests if r.get('model')==alias]
        services=[r['worker_latency_ms'] for r in rows if finite(r.get('worker_latency_ms'))]
        queues=[r['worker_queue_wait_ms'] for r in rows if finite(r.get('worker_queue_wait_ms'))]
        alias_summaries[alias]=dict(requests=len(rows),mean_service_ms=statistics.mean(services) if services else None,
            sum_service_seconds=sum(services)/1000 if services else None,
            mean_worker_queue_wait_ms=statistics.mean(queues) if queues else None,
            max_worker_queue_wait_ms=max(queues) if queues else None)
    return dict(candidate_id=candidate_id,candidate=symbolic,seed=seed,
        raw=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        status=raw.get('status'),errors=raw.get('errors'),utility=utility,terms=terms,
        contributions=contributions,cohort_sha256=digest(cohort),request_count=len(requests),
        routing_counts={t:{m:n for (tt,m),n in counts.items() if tt==t} for t in totals},
        routing_fractions={t:{m:n/totals[t] for (tt,m),n in counts.items() if tt==t} for t in totals},
        alias_summaries=alias_summaries,stability=raw.get('evaluation',{}).get('stability'),
        memory_evidence_summary={k:v for k,v in raw.get('memory_evidence',{}).items() if k not in {'samples','before','after'}},
        energy_evidence_summary={k:v for k,v in raw.get('energy_evidence',{}).items() if k not in {'samples'}},
        scope='Original six-term native evaluation; finite declared policy domain only')


def analyze(raw_paths,seeds,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    observations=[read_run(p) for p in raw_paths];groups=defaultdict(dict);cohorts={}
    for row in observations:
        if row['seed'] not in seeds:raise ValueError('Undeclared seed')
        if row['seed'] in groups[row['candidate_id']]:raise ValueError('Duplicate candidate/seed')
        groups[row['candidate_id']][row['seed']]=row
        old=cohorts.setdefault(row['seed'],row['cohort_sha256'])
        if old!=row['cohort_sha256']:raise ValueError('Candidate comparisons have different arrival/input cohorts')
    ranking=[]
    for cid,rows in groups.items():
        known=all(s in rows and rows[s]['utility'].get('eligible') is True and finite(rows[s]['utility'].get('U')) for s in seeds)
        ranking.append(dict(candidate_id=cid,candidate=next(iter(rows.values()))['candidate'],
            complete_eligible_all_declared_seeds=known,
            mean_U=statistics.mean(rows[s]['utility']['U'] for s in seeds) if known else None,
            per_seed_U={str(s):rows[s]['utility'].get('U') if s in rows else None for s in seeds}))
    ranking.sort(key=lambda r:(r['mean_U'] is None,-r['mean_U'] if r['mean_U'] is not None else 0,r['candidate_id']))
    if ranking and ranking[0]['mean_U'] is not None:
        best=ranking[0]['mean_U'];ties=[r for r in ranking if r['mean_U'] is not None and abs(r['mean_U']-best)<=1e-12]
        rest=[r for r in ranking if r not in ties];ranking=sorted(ties,key=lambda r:r['candidate_id'])+rest
    pairs=[]
    for a,b in itertools.combinations(sorted(groups),2):
        delta=[];per=[]
        for seed in seeds:
            ra=groups[a].get(seed);rb=groups[b].get(seed)
            known=bool(ra and rb and ra['utility'].get('eligible') is True and rb['utility'].get('eligible') is True)
            value=rb['utility']['U']-ra['utility']['U'] if known else None
            if value is not None:delta.append(value)
            per.append(dict(seed=seed,delta_U_B_minus_A=value,
                six_term_delta_contributions={t:rb['contributions'][t]-ra['contributions'][t] for t in TERMS} if known else None))
        pairs.append(dict(A=a,B=b,all_declared_pairs_eligible=len(delta)==len(seeds),per_seed=per,
            descriptive_whole_pair_summary=describe(delta),
            selection_warning='Development comparisons are selection evidence; validation must freeze candidates first.'))
    configs=[json.loads(Path(p).read_text())['config']['weights'] for p in raw_paths]
    if configs and any(w!=configs[0] for w in configs):raise ValueError('Different utility weights across comparisons')
    table=dict(candidates=[r['candidate'] for r in ranking],seeds=list(seeds),weights=configs[0] if configs else {},
        records=[dict(candidate_id=r['candidate_id'],seed=r['seed'],utility=r['utility'],raw=r['raw']) for r in observations],
        scope='Only these declared measured candidates and seeds; unavailable leaves remain UNKNOWN')
    report=dict(observations=observations,ranking=ranking,comparisons=pairs,seeds=list(seeds),
        winner=ranking[0] if ranking and ranking[0]['mean_U'] is not None else None,
        full_declared_table_complete=bool(ranking) and all(r['complete_eligible_all_declared_seeds'] for r in ranking),
        shared_cohorts_verified=True,physical_full_pool_optimality=False)
    write(output/'report.json',report);write(output/'measured-search-table.json',table)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw',type=Path,nargs='+',required=True)
    parser.add_argument('--seeds',type=int,nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.raw,args.seeds,args.output)
    print(json.dumps(dict(winner=result['winner'],complete=result['full_declared_table_complete']),ensure_ascii=False))
