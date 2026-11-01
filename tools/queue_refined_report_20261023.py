"""Append-only closure of the refined queue study; no GPU execution."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.queue_refined_20261023.protocol import receipt,digest,write
from tools.queue_refined_20261023.analysis import read_run
from tools.queue_refined_20261023.search import measured_table_search


def read(path):return json.loads(Path(path).read_text())
def mean(v):return statistics.mean(v) if v else None
def fmt(value,digits=6):return f'{value:.{digits}f}' if isinstance(value,(int,float)) else 'UNKNOWN'
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def equal(a,b):return a==b if a is None or b is None else finite(a) and finite(b) and math.isclose(a,b,rel_tol=0,abs_tol=1e-12)


def verify_receipts(value,issues,label):
    """Validate receipts referenced by a completed independent closure."""
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
            try:valid=receipt(value['path'])['sha256']==value['sha256']
            except (OSError,ValueError):valid=False
            if not valid:issues.append(label+':'+value['path'])
        else:
            for child in value.values():verify_receipts(child,issues,label)
    elif isinstance(value,list):
        for child in value:verify_receipts(child,issues,label)


def native_dfs_audit(stage,packet,supervisor,rows,issues,partial_output=None):
    """Recompute the finite measured ranking from verified whole-run records.

    DFS here only looks up already collected values. No native run is started.
    """
    label=stage['label'];analysis=Path(stage['progress']).parent/'analysis'
    table_path=analysis/'measured-search-table.json';comparison_path=analysis/'measured-dfs-comparison.json'
    report_path=Path(stage.get('report',{}).get('path',analysis/'report.json'))
    missing=[str(p) for p in (table_path,comparison_path,report_path) if not p.exists()]
    partial=partial_output is not None
    if missing and not partial:
        issues.append('NATIVE_DFS_FILES_MISSING:'+label)
        return dict(stage=label,verified=False,missing=missing)
    if partial:
        # Preserve expected candidates AND all declared seeds. Missing outcomes
        # stay UNKNOWN; never average only the successfully observed subset.
        candidates={digest(dict(deployment=j['policy_contract']['deployment'],strategy=j['policy_contract']['strategy'],policy_contract_sha256=digest(j['policy_contract']))):dict(deployment=j['policy_contract']['deployment'],strategy=j['policy_contract']['strategy'],policy_contract_sha256=digest(j['policy_contract'])) for j in packet['jobs']}
        table=dict(candidates=list(candidates.values()),seeds=packet['seeds'],weights={t:1/6 for t in 'QTLPME'},
            records=[dict(candidate_id=r['summary']['candidate_id'],seed=r['summary']['seed'],utility=r['summary']['utility'],raw=r['summary']['raw']) for r in rows],
            scope='Interrupted registered outcomes only; absent candidate/seed records remain UNKNOWN, no retries.')
        target=Path(partial_output);target.mkdir(parents=True,exist_ok=False)
        table_path=target/'measured-search-table.json';comparison_path=target/'measured-dfs-comparison.json';report_path=target/'partial-ranking.json'
    else:table=read(table_path)
    observed=defaultdict(dict);cohorts=defaultdict(set)
    for row in rows:
        summary=row['summary'];cid=summary['candidate_id'];seed=summary['seed']
        if seed in observed[cid]:issues.append('NATIVE_DUPLICATE_CANDIDATE_SEED:'+label)
        observed[cid][seed]=summary;cohorts[seed].add(summary['cohort_sha256'])
    seeds=packet['seeds'];declared={digest(dict(deployment=j['policy_contract']['deployment'],strategy=j['policy_contract']['strategy'],policy_contract_sha256=digest(j['policy_contract']))) for j in packet['jobs']}
    if (not set(observed)<=declared or (not partial and set(observed)!=declared) or not set(cohorts)<=set(seeds)
        or (not partial and set(cohorts)!=set(seeds)) or any(len(v)!=1 for v in cohorts.values())):issues.append('NATIVE_COHORT_OR_DOMAIN_MISMATCH:'+label)
    table_records={(r['candidate_id'],r['seed']):r for r in table['records']}
    if (len(table_records)!=len(table['records']) or set(table_records)!={(cid,s) for cid,group in observed.items() for s in group}
        or table.get('seeds')!=seeds or table.get('weights')!={t:1/6 for t in 'QTLPME'}
        or {digest(c) for c in table['candidates']}!=declared):issues.append('NATIVE_MEASURED_TABLE_DOMAIN_MISMATCH:'+label)
    means={}
    for cid,group in observed.items():
        if set(group)==set(seeds) and all(r['utility'].get('eligible') is True and finite(r['utility'].get('U')) for r in group.values()):means[cid]=mean([group[s]['utility']['U'] for s in seeds])
        for seed,r in group.items():
            entry=table_records.get((cid,seed),{})
            if entry.get('raw')!=r['raw'] or entry.get('utility')!=r['utility']:issues.append('NATIVE_MEASURED_TABLE_VALUE_MISMATCH:'+label)
    maximum=max(means.values(),default=None);ties=sorted(cid for cid,u in means.items() if maximum-u<=1e-12) if maximum is not None else []
    winner=ties[0] if ties else None;objective=means.get(winner)
    computed={name:measured_table_search(table,name) for name in ('plain_dfs','bounded')}
    expected=dict(winner_id=winner,objective=objective,exact_max_objective=maximum,tie_ids=ties)
    if partial:
        saved=dict(computed,same_winner_and_value=True,partial=True)
        report=dict(ranking=[dict(candidate_id=cid,mean_U=means.get(cid)) for cid in sorted(declared)],winner=dict(candidate_id=winner,mean_U=objective) if winner else None,exact_max_objective=maximum)
        write(table_path,table);write(comparison_path,saved);write(report_path,report)
    else:saved=read(comparison_path);report=read(report_path)
    for algorithm in ('plain_dfs','bounded'):
        for key,value in expected.items():
            for result in (computed[algorithm],saved.get(algorithm,{})):
                valid=equal(result.get(key),value) if key in ('objective','exact_max_objective') else result.get(key)==value
                if not valid:issues.append('NATIVE_DFS_DIFFERENT_'+key+':'+label+':'+algorithm)
    ranking={r['candidate_id']:r for r in report.get('ranking',[])}
    if set(ranking)!=declared or any(not equal(ranking.get(cid,{}).get('mean_U'),means.get(cid)) for cid in declared):issues.append('NATIVE_RANKING_MEAN_MISMATCH:'+label)
    if (report.get('winner') or {}).get('candidate_id')!=winner or not equal((report.get('winner') or {}).get('mean_U'),objective) or not equal(report.get('exact_max_objective'),maximum):issues.append('NATIVE_RANKING_WINNER_MISMATCH:'+label)
    roles={r['summary']['candidate_id']:r['role'] for r in rows}
    winner_row=next((r for r in rows if r['summary']['candidate_id']==winner),None)
    return dict(stage=label,seeds=seeds,candidates=len(declared),independent_means=means,**expected,partial=partial,
        complete_declared_domain=False if partial else not computed['plain_dfs']['unknown_ids'],unknown_candidate_ids=computed['plain_dfs']['unknown_ids'],
        winner_role=roles.get(winner),winner_deployment=winner_row['contract']['deployment'] if winner_row else None,
        tie_roles=[roles.get(cid) for cid in ties],cohort_by_seed={str(s):next(iter(v)) if len(v)==1 else None for s,v in cohorts.items()},
        same_winner_value_exact_max_and_ties=not any(i.startswith(('NATIVE_DFS_','NATIVE_RANKING_','NATIVE_MEASURED_','NATIVE_COHORT_','NATIVE_DUPLICATE_')) and label in i for i in issues),
        table=receipt(table_path),saved_comparison=receipt(comparison_path),report=receipt(report_path),recomputed=computed,
        scope='Exact finite measured-table ranking only; stored U lookups execute no GPU inference and establish no physical full-pool optimum.')


def cpu_efficiency(report,report_path,issues):
    a=report['plain_dfs'];b=report['bounded'];log_path=Path(report_path).parent/'bounded-evaluations.jsonl'
    if log_path.exists():
        rows=[json.loads(line) for line in log_path.read_text().splitlines() if line]
        representatives=[r for r in rows if r.get('oracle_evaluated_for_this_identity')]
        calls_from_log=sum(len(r.get('runs',[])) for r in representatives)
        if calls_from_log!=b['oracle_calls']:issues.append('CPU_LOG_CALL_COUNT_MISMATCH:'+str(report['load']['text']))
        late_prunes=sum(r.get('status')=='SEED_BOUND_PRUNED' and len(r.get('runs',[]))==2 for r in representatives)
    else:
        calls_from_log=None;late_prunes=None;issues.append('CPU_EVALUATION_LOG_MISSING:'+str(report['load']['text']))
    seed_count=a['oracle_calls']//a['candidates'] if a['candidates'] and a['oracle_calls']%a['candidates']==0 else None
    structural=(a['candidates']-b['representatives'])*seed_count if seed_count else None
    bound_leaf=(b.get('bound_pruned') or 0)*seed_count if seed_count else None
    bound_seed=b['representatives']*seed_count-b['oracle_calls']-bound_leaf if seed_count else None
    if (a['oracle_calls']!=102 or b['oracle_calls']!=78 or structural!=24 or bound_leaf!=0 or bound_seed!=0):issues.append('CPU_EFFICIENCY_ACCOUNTING_CHANGED:'+str(report['load']['text']))
    return dict(load=report['load']['text'],dfs_calls=a['oracle_calls'],optimized_calls=b['oracle_calls'],
        call_reduction_percent=100*(a['oracle_calls']-b['oracle_calls'])/a['oracle_calls'],
        dfs_seconds=a['elapsed_seconds'],optimized_seconds=b['elapsed_seconds'],
        observed_cpu_speedup=a['elapsed_seconds']/b['elapsed_seconds'] if b['elapsed_seconds']>0 else None,
        structure_saved_calls=structural,bound_leaf_saved_calls=bound_leaf,bound_seed_saved_calls=bound_seed,
        representative_calls_verified_from_log=calls_from_log,late_seed_pruned_after_all_seeds=late_prunes,
        evaluation_log=receipt(log_path) if log_path.exists() else None,cpu_timing_exclusive=report.get('cpu_timing_exclusive'),
        scope='Process-local nonexclusive CPU conditional-mean proxy; structure grouping saves actual simulator calls, not GPU measurements.')


def independent_closure(study,execution,native,native_dfs,total_runs,issues,closure_path=None,interrupted=False):
    default=study/'native-independent-terminal-audit-v1/final-closure.json'
    path=Path(closure_path).resolve() if closure_path is not None else default
    if not path.exists():
        if closure_path is not None:issues.append('INDEPENDENT_CLOSURE_EXPLICIT_PATH_MISSING:'+str(path))
        return dict(present=False,path=str(path),verified=False,status='NOT_YET_PRESENT',explicit_override=closure_path is not None)
    doc=read(path);verify_receipts(doc,issues,'INDEPENDENT_CLOSURE_RECEIPT_CHANGED')
    recovery_path=path.parent/'recovery-receipt.json';recovery=None
    # The supplemental interrupted closure can reference the resumed auditor's
    # summary rather than duplicate its preserved recovery receipt.
    if not recovery_path.exists() and isinstance(doc.get('audit'),dict) and doc['audit'].get('path'):
        candidate=Path(doc['audit']['path']).parent/'recovery-receipt.json'
        if candidate.exists():recovery_path=candidate
    if recovery_path.exists():
        recovery_doc=read(recovery_path);before=len(issues)
        verify_receipts(recovery_doc,issues,'INDEPENDENT_CLOSURE_RECOVERY_RECEIPT_CHANGED')
        for name in ('prior_progress','prior_watch_script','prior_finalize_script'):
            bound=recovery_doc.get(name)
            if not isinstance(bound,dict) or not bound.get('path') or not bound.get('sha256'):issues.append('INDEPENDENT_CLOSURE_RECOVERY_PROVENANCE_MISSING:'+name)
        protected=[v for v in recovery_doc.values() if isinstance(v,dict) and isinstance(v.get('path'),str) and isinstance(v.get('sha256'),str)]
        recovery=dict(receipt=receipt(recovery_path),document=recovery_doc,protected_receipts=protected,verified=len(issues)==before,
            summary_preservation_scope='Verify every recovery receipt as written. Prior summary is protected if the recovery receipt records it; do not claim a pre-existing summary receipt when none was recorded.')
    elif path.parent.name=='native-independent-terminal-audit-resume-v1':
        issues.append('INDEPENDENT_CLOSURE_RECOVERY_RECEIPT_MISSING:'+str(recovery_path))
    allowed_status=('COMPLETE_INDEPENDENT_INTERRUPTED_CLOSURE_PASS','COMPLETE_INTERRUPTED_INDEPENDENT_CLOSURE_PASS') if interrupted else ('COMPLETE_INDEPENDENT_FULL_CLOSURE_PASS',)
    if doc.get('valid') is not True or doc.get('status') not in allowed_status or doc.get('issues'):issues.append('INDEPENDENT_CLOSURE_FAILED')
    energy_source=None
    if interrupted:
        if doc.get('interrupted') is not True or doc.get('expected_formal_jobs')!=26 or len(doc.get('unexecuted_jobs',[]))!=1:issues.append('INDEPENDENT_CLOSURE_INTERRUPTION_DOMAIN_MISMATCH')
        bound=doc.get('energy_failure')
        if isinstance(bound,dict) and bound.get('path'):
            energy_doc=read(bound['path']);verify_receipts(energy_doc,issues,'INDEPENDENT_CLOSURE_ENERGY_SOURCE_CHANGED')
            if energy_doc.get('valid') is not True or energy_doc.get('audit_consistency_valid') is not True or energy_doc.get('E') is not None or energy_doc.get('U') is not None or energy_doc.get('no_energy_imputation') is not True:issues.append('INDEPENDENT_CLOSURE_ENERGY_SOURCE_INVALID')
            energy_source=dict(receipt=bound,document=energy_doc)
        else:issues.append('INDEPENDENT_CLOSURE_ENERGY_SOURCE_MISSING')
    if (doc.get('workflow')!=receipt(execution/'progress.json') or doc.get('total_jobs')!=total_runs
        or doc.get('formal_jobs')!=len(native) or doc.get('calibration_jobs',total_runs-len(native))!=total_runs-len(native)):issues.append('INDEPENDENT_CLOSURE_SCOPE_MISMATCH')
    stages={r['label']:r for r in doc.get('stages',[])}
    if set(stages)!={r['stage'] for r in native_dfs}:issues.append('INDEPENDENT_CLOSURE_STAGE_SET_MISMATCH')
    for row in native_dfs:
        other=stages.get(row['stage'],{})
        if (other.get('winner')!=row.get('winner_id') or other.get('ties')!=row.get('tie_ids')
            or not equal(other.get('exact_max_objective'),row.get('exact_max_objective'))
            or set(other.get('independent_means',{}))!=set(row.get('independent_means',{}))
            or any(not equal(other['independent_means'][cid],v) for cid,v in row.get('independent_means',{}).items() if cid in other.get('independent_means',{}))):issues.append('INDEPENDENT_CLOSURE_RANKING_MISMATCH:'+row['stage'])
    expected={}
    for row in native:
        if not row['stage'].startswith('development'):expected.setdefault(row['summary']['seed'],{})[row['role']]=row
    pairs={p['seed']:p for p in doc.get('reserved_pairs',[])}
    if set(pairs)!=set(expected):issues.append('INDEPENDENT_CLOSURE_PAIR_SET_MISMATCH')
    for seed,group in expected.items():
        if set(group)!={'calibrated_spill','risk_matched_no_spill'} and not interrupted:issues.append('RESERVED_PAIR_ROLE_MISMATCH:'+str(seed));continue
        a=(group.get('calibrated_spill') or {}).get('summary',{});b=(group.get('risk_matched_no_spill') or {}).get('summary',{});known=a.get('utility',{}).get('eligible') is True and b.get('utility',{}).get('eligible') is True
        delta=a['utility']['U']-b['utility']['U'] if known else None;pair=pairs.get(seed,{})
        if pair.get('eligible_pair')!=known or not equal(pair.get('delta_U_spill_minus_matched_no_spill'),delta):issues.append('INDEPENDENT_CLOSURE_PAIR_VALUE_MISMATCH:'+str(seed))
        if known and any(not equal((pair.get('six_term_delta_contributions') or {}).get(t),a['contributions'][t]-b['contributions'][t]) for t in 'QTLPME'):issues.append('INDEPENDENT_CLOSURE_PAIR_TERMS_MISMATCH:'+str(seed))
    return dict(present=True,receipt=receipt(path),verified=not any(i.startswith('INDEPENDENT_CLOSURE_') for i in issues),document=doc,
        explicit_override=closure_path is not None,recovery=recovery,energy_failure_source=energy_source)


def plots(native,cpu,output,selected_load=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    seeds=[];deltas=[];spill_seeds=[];spills=[];unknown=[]
    for seed in (205,206,207,208,209):
        rows={r['role']:r for r in native if not r['stage'].startswith('development') and r['summary']['seed']==seed}
        if 'calibrated_spill' in rows:
            spill_seeds.append(str(seed));spills.append(rows['calibrated_spill']['mechanism']['spilled_text'])
        if set(rows)!={'calibrated_spill','risk_matched_no_spill'}:unknown.append(str(seed));continue
        a=rows['calibrated_spill'];b=rows['risk_matched_no_spill']
        if a['summary']['utility'].get('eligible') is not True or b['summary']['utility'].get('eligible') is not True:unknown.append(str(seed));continue
        seeds.append(str(seed));deltas.append(a['summary']['utility']['U']-b['summary']['utility']['U'])
    # Numeric positions preserve all five registered seeds, including UNKNOWN.
    positions0=[int(s)-205 for s in seeds]
    axes[0].bar(positions0,deltas,color=['#168b71' if v>0 else '#b94646' for v in deltas]);axes[0].axhline(0,color='black',linewidth=.7)
    axes[0].set_xticks(range(5),[str(s) for s in (205,206,207,208,209)]);axes[0].set(title='Reserved paired total utility',xlabel='Seed',ylabel='U(spill) - U(no spill)')
    for seed in unknown:axes[0].text(int(seed)-205,0,'UNKNOWN',rotation=90,ha='center',va='bottom',color='#777777')
    axes[1].bar([int(s)-205 for s in spill_seeds],spills,color='#446fa8');axes[1].set_xticks(range(5),[str(s) for s in (205,206,207,208,209)])
    axes[1].set(title='Autonomous generalist routing',xlabel='Seed',ylabel='Text requests sent to G2')
    width=.33;positions=list(range(len(cpu)));labels=[str(r['load']['text']) for r in cpu]
    axes[2].bar([x-width/2 for x in positions],[r['plain_dfs']['oracle_calls'] for r in cpu],width,label='DFS',color='#858b94')
    axes[2].bar([x+width/2 for x in positions],[r['bounded']['oracle_calls'] for r in cpu],width,label='Optimized',color='#446fa8')
    axes[2].set_xticks(positions,labels);axes[2].set(title='Finite CPU proxy search',xlabel='Text rate (requests/s)',ylabel='Simulator calls');axes[2].legend()
    fig.suptitle(f'Frozen text load: {selected_load}; evaluable reserved pairs: {len(deltas)}/5')
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(output/'results.png',dpi=160);plt.close(fig)


def mechanism(raw):
    aliases=raw['replica_to_family'];counts=Counter();spills=[];prediction=[]
    for row in raw.get('request_records',[]):
        family=aliases.get(row.get('model'),'UNKNOWN');counts[(row['task'],family)]+=1
        rd=row.get('routing_decision') or {};decision=rd.get('decision') or {}
        if row['task']!='text':continue
        audit=decision.get('audit',[]);chosen=next((a for a in audit if a.get('alias')==row.get('model')),None)
        own=(chosen or {}).get('predicted_own_service_ms')
        # Actual worker service is distinct from host end-to-end response time.
        actual=row.get('worker_response',{}).get('service_ms')
        if not finite(actual):actual=row.get('worker_latency_ms')
        if finite(own) and own>0 and finite(actual):
            prediction.append(dict(family=family,predicted_own_ms=own,actual_service_ms=actual,ratio=actual/own))
        if family=='generalist_2':
            snapshot=rd.get('snapshot',{});general=snapshot.get(row['model'],{}).get('outstanding',[])
            specs={a:len(v.get('outstanding',[])) for a,v in snapshot.items() if aliases.get(a)=='text'}
            spills.append(dict(request_id=row['request_id'],pool_id=row['pool_id'],generalist_visible_pending=len(general),specialist_pending=specs,
                both_specialists_pending=len(specs)==2 and all(v>0 for v in specs.values()),actual_service_ms=actual,
                predicted_own_service_ms=own,predicted_completion_ms=(chosen or {}).get('predicted_completion_ms'),
                best_specialist_predicted_completion_ms=min((a['predicted_completion_ms'] for a in audit if a.get('family')=='text' and isinstance(a.get('predicted_completion_ms'),(int,float))),default=None),
                delta_proxy_cost_vs_best_specialist=(chosen or {}).get('delta_proxy_cost_vs_best_specialist'),
                token_guard=decision.get('rate_guard')))
    byfamily={}
    for family in sorted({r['family'] for r in prediction}):
        rows=[r for r in prediction if r['family']==family]
        byfamily[family]=dict(n=len(rows),mean_actual_ms=mean([r['actual_service_ms'] for r in rows]),mean_predicted_ms=mean([r['predicted_own_ms'] for r in rows]),
            mean_ratio=mean([r['ratio'] for r in rows]),median_ratio=statistics.median(r['ratio'] for r in rows),
            mean_absolute_error_ms=mean([abs(r['actual_service_ms']-r['predicted_own_ms']) for r in rows]),
            mean_signed_error_actual_minus_predicted_ms=mean([r['actual_service_ms']-r['predicted_own_ms'] for r in rows]))
    return dict(counts={t:{m:n for (tt,m),n in counts.items() if tt==t} for t in ('text','image','ocr','asr')},
        spilled_text=len(spills),unique_spilled_inputs=len({r['pool_id'] for r in spills}),
        spill_both_specialists_pending=sum(r['both_specialists_pending'] for r in spills),
        spill_generalist_visible_empty=sum(r['generalist_visible_pending']==0 for r in spills),
        prediction_diagnostics=byfamily,spills=spills,
        interpretation='Actual service=worker_response.service_ms, falling back to worker_latency_ms. Both-specialists-pending requires exactly two text aliases with host-visible outstanding work; G2-visible-empty is not an observed GPU-idle claim. Unchosen-path service/outcomes remain unobserved; request ratios are correlated descriptive diagnostics.')


def close(study,execution,screening,output,independent_closure_path=None,allow_interrupted=False):
    study=Path(study).resolve();execution=Path(execution).resolve();output=Path(output).resolve()
    state=read(execution/'progress.json')
    interrupted=state.get('status')=='STOPPED_FOR_DIAGNOSIS_NO_RETRY'
    if not state['status'].startswith('COMPLETE') and not (allow_interrupted and interrupted and state.get('no_retries') is True):raise ValueError('Workflow not complete; interrupted closure requires explicit --allow-interrupted and no-retry terminal')
    # Reject unfinished stages before creating append-only output or reading raw.
    for stage in state['stages']:
        if read(stage['progress']).get('status') not in (('COMPLETE','COMPLETE_AUDITED','STOPPED_FOR_DIAGNOSIS_NO_RETRY') if interrupted else ('COMPLETE','COMPLETE_AUDITED')):raise ValueError('Stage not complete: '+stage['label'])
    if read(study/'calibration-execution-v1/progress.json').get('status')!='COMPLETE':raise ValueError('Calibration not complete')
    output.mkdir(parents=True,exist_ok=False)
    issues=[];bindings=[];native=[];native_dfs=[];arithmetic=[];declared_issues=[];unexecuted=[]
    freeze=read(study/'main-code-freeze-v1.json')
    for path,expected in freeze['sources'].items():
        if receipt(path)['sha256']!=expected:issues.append('MAIN_FREEZE_CHANGED:'+path)
    for bound in state['bindings']:
        bindings.append(bound)
        if receipt(bound['path'])!=bound:issues.append('BOUND_CHANGED:'+bound['path'])
    for stage in state['stages']:
        p=read(stage['progress'])
        for name in ('packet','report','terminal_progress'):
            if isinstance(stage.get(name),dict):
                bindings.append(stage[name]);verify_receipts(stage[name],issues,'STAGE_RECEIPT_CHANGED')
        packet=read(stage['packet']['path'])
        observed_ids={j['job_id'] for j in p['jobs']};expected_ids={j['id'] for j in packet['jobs']};partial=p.get('status')=='STOPPED_FOR_DIAGNOSIS_NO_RETRY'
        if (len(observed_ids)!=len(p['jobs']) or not observed_ids<=expected_ids or (not partial and observed_ids!=expected_ids)):issues.append('STAGE_JOB_SET_MISMATCH:'+stage['label'])
        for job in packet['jobs']:
            if job['id'] not in observed_ids:unexecuted.append(dict(stage=stage['label'],role=job['role'],seed=job['seed'],job_id=job['id'],status='UNEXECUTED_NO_RETRY'))
        stage_rows=[]
        for registered in p['jobs']:
            bound=registered['raw'];bindings.append(bound)
            if receipt(bound['path'])!=bound:issues.append('RAW_CHANGED:'+bound['path']);continue
            raw=read(bound['path']);job=next((j for j in packet['jobs'] if j['id']==raw.get('experiment_job_id')),None)
            if job is None:issues.append('RAW_JOB_NOT_IN_PACKET:'+bound['path']);continue
            if raw.get('executed_measurement_identity')!=job['identity']:issues.append('RAW_IDENTITY_CHANGED:'+bound['path'])
            if (raw.get('executed_measurement_id')!=job['id'] or raw.get('policy_contract')!=job['policy_contract']
                or raw.get('transport_config',{}).get('random_seed')!=job['seed']):issues.append('RAW_CONTRACT_OR_SEED_CHANGED:'+bound['path'])
            transport=raw.get('transport_config',{});config=raw.get('config',{})
            if (config.get('weights')!={t:1/6 for t in 'QTLPME'} or config.get('gpu_memory_mib')!=49140
                or config.get('catalog',{}).get('safe_factor')!=1.110854 or transport.get('measurement_seconds')!=600
                or transport.get('warmup_seconds')!=30 or transport.get('drain_timeout_seconds')!=300
                or raw.get('evaluation',{}).get('utility',{}).get('absolute_quality_gate') is not False):
                issues.append('ORIGINAL_PROTOCOL_CHANGED:'+bound['path'])
            if raw.get('preflight',{}).get('gpu',{}).get('uuid')!='GPU-8a262fe0-598c-46b7-c953-41d9da84a47e':
                issues.append('GPU_UUID_CHANGED:'+bound['path'])
            if not raw.get('evaluation',{}).get('dynamic_audit',{}).get('verified'):issues.append('CAUSAL_REPLAY_FAILED:'+bound['path'])
            memory=raw.get('memory_evidence',{});energy=raw.get('energy_evidence',{});utility=raw.get('evaluation',{}).get('utility',{})
            energy_unknown=(interrupted and partial and job['seed']==209 and job['role']=='calibrated_spill'
                and raw.get('status')=='completed' and memory.get('resource_valid') is True and energy.get('coverage_valid') is False
                and energy.get('run_joules') is None and utility.get('E_norm') is None and utility.get('U') is None and utility.get('eligible') is not True)
            if energy_unknown:
                declared_issues.append(dict(code='DECLARED_ENERGY_COVERAGE_FAILURE',stage=stage['label'],role=job['role'],seed=job['seed'],raw=bound,
                    missing_terms=['E'],reason='Registered terminal has invalid energy coverage and no run_joules; E and total U remain UNKNOWN. No imputation, retry or remeasurement.',
                    energy_evidence={k:v for k,v in energy.items() if k!='samples'},workflow_error=state.get('error'),stage_error=p.get('error')))
            elif not memory.get('resource_valid') or not energy.get('coverage_valid'):issues.append('RESOURCES_FAILED:'+bound['path'])
            summary=read_run(bound['path']);u=summary['utility'];terms=summary['terms'];known=all(finite(terms.get(t)) for t in 'QTLPME')
            reconstructed=-math.fsum(config['weights'][t]*terms[t] for t in 'QTLPME') if known else None
            arithmetic.append(dict(stage=stage['label'],seed=summary['seed'],candidate_id=summary['candidate_id'],raw=bound,
                eligible=u.get('eligible'),reconstructed_U=reconstructed,reported_U=u.get('U'),
                six_term_sum_verified=known and finite(u.get('U')) and math.isclose(reconstructed,u['U'],rel_tol=0,abs_tol=1e-10),
                cohort_sha256=summary['cohort_sha256'],cohort_verified=u.get('cohort_verified') is True))
            row=dict(stage=stage['label'],role=job['role'],load=job['policy_contract']['text_rate'],contract=job['policy_contract'],summary=summary,mechanism=mechanism(raw))
            native.append(row);stage_rows.append(row)
        native_dfs.append(native_dfs_audit(stage,packet,p,stage_rows,issues,output/'partial-measured-tables'/stage['label'] if partial else None))
    old=read(study/'design-review-v1/old-evidence-ledger.json')
    for b in old['protected']:
        if receipt(b['path'])!=b:issues.append('OLD_PROTECTED_CHANGED:'+b['path'])
    cal=read(study/'calibration-execution-v1/progress.json')
    if cal['status']!='COMPLETE':issues.append('CALIBRATION_NOT_COMPLETE')
    calibration=[]
    for job in cal['jobs']:
        b=job['raw'];bindings.append(b)
        if receipt(b['path'])!=b:issues.append('CAL_RAW_CHANGED:'+b['path'])
        raw=read(b['path']);calibration.append(dict(label=job['label'],seed=raw['transport_config']['random_seed'],
            probability=raw['policy_contract']['probe']['probability'],utility=raw['evaluation']['utility'],
            mechanism=mechanism(raw),raw=b))
    selection=read(execution/'selection.json');cpu=[];efficiency=[]
    for item in read(Path(screening)/'report.json')['per_load']:
        report=read(item['report_path']);cpu.append(report)
        a,b=report['plain_dfs'],report['bounded']
        for key in ('winner_id','objective','exact_max_objective','tie_ids'):
            if a.get(key)!=b.get(key):issues.append('CPU_DFS_DIFFERENT_'+key)
        if (a['candidates']!=51 or b.get('representatives')!=39 or b.get('proven_equivalent_skips')!=12
            or len(b.get('groups',[]))!=39 or sum(len(g['member_ids']) for g in b.get('groups',[]))!=51):
            issues.append('CPU_QUOTIENT_COUNTS_DIFFER')
        if report.get('bound_violations')!=0 or any(c['violation']>1e-12 or any(r['violation']>1e-12 for r in c['per_seed']) for c in report.get('bound_checks',[])):
            issues.append('CPU_BOUND_VIOLATION')
        if len(report.get('equivalence_audit',[]))!=39 or not all(g.get('independent_DFS_all_seed_U_feasibility_and_decision_hash_equal') is True for g in report.get('equivalence_audit',[])):
            issues.append('CPU_EQUIVALENCE_AUDIT_FAILED')
        efficiency.append(cpu_efficiency(report,item['report_path'],issues))
    expected_formal=26 if selection.get('status')=='FROZEN' else 16
    if len(native)+len(unexecuted)!=expected_formal or len(cal['jobs'])!=4 or (not interrupted and unexecuted):issues.append('EXPECTED_COMPLETE_RUN_COUNT_MISMATCH')
    if interrupted and (len(native)!=25 or len(unexecuted)!=1 or unexecuted[0]['seed']!=209 or unexecuted[0]['role']!='risk_matched_no_spill' or len(declared_issues)!=1):issues.append('INTERRUPTED_TERMINAL_SHAPE_MISMATCH')
    pair_checks=[]
    for seed in (205,206,207,208,209):
        group={r['role']:r for r in native if not r['stage'].startswith('development') and r['summary']['seed']==seed}
        if not group and selection.get('status')!='FROZEN':continue
        missing_roles=sorted({'calibrated_spill','risk_matched_no_spill'}-set(group))
        if missing_roles:
            if not interrupted:issues.append('RESERVED_PAIR_ROLE_MISMATCH:'+str(seed))
            pair_checks.append(dict(seed=seed,eligible_pair=False,delta_U=None,delta_contributions=None,six_term_delta_sum_verified=False,
                missing_roles=missing_roles,measurement_outcomes={role:dict(status='MEASURED',eligible=r['summary']['utility'].get('eligible'),U=r['summary']['utility'].get('U')) for role,r in group.items()}))
            continue
        a=group['calibrated_spill'];b=group['risk_matched_no_spill'];sa=a['summary'];sb=b['summary'];ca=a['contract'];cb=b['contract']
        same=ca['deployment']==cb['deployment'] and sa['cohort_sha256']==sb['cohort_sha256']
        if not same:issues.append('RESERVED_RESIDENCY_OR_COHORT_MISMATCH:'+str(seed))
        # Preserve policy settings, allowing only the preregistered path switch.
        ra=ca.get('refinement') or {};rb=cb.get('refinement') or {}
        same_parameters=all(ra.get(k)==rb.get(k) for k in ('mode','risk_quantile','concurrency_context')) and ra.get('allow_spill') is True and rb.get('allow_spill') is False
        if not same_parameters:issues.append('RESERVED_POLICY_PARAMETERS_MISMATCH:'+str(seed))
        known=sa['utility'].get('eligible') is True and sb['utility'].get('eligible') is True
        delta=sa['utility']['U']-sb['utility']['U'] if known else None
        terms={t:sa['contributions'][t]-sb['contributions'][t] for t in 'QTLPME'} if known else None
        arithmetic_ok=known and math.isclose(math.fsum(terms.values()),delta,rel_tol=0,abs_tol=1e-10)
        if known and not arithmetic_ok:issues.append('RESERVED_PAIR_ARITHMETIC_MISMATCH:'+str(seed))
        pair_checks.append(dict(seed=seed,eligible_pair=known,same_resident_deployment_and_cohort=same,same_mode_risk_context=same_parameters,
            spill_margin_ms=ra.get('margin_ms'),control_margin_ms=rb.get('margin_ms'),control_margin_inactive_when_no_spill=True,
            delta_U=delta,delta_contributions=terms,six_term_delta_sum_verified=arithmetic_ok,cohort_sha256=sa['cohort_sha256'],missing_roles=[],
            measurement_outcomes={role:dict(status='MEASURED',eligible=r['summary']['utility'].get('eligible'),U=r['summary']['utility'].get('U')) for role,r in group.items()}))
    closure=independent_closure(study,execution,native,native_dfs,len(native)+len(cal['jobs']),issues,independent_closure_path,interrupted)
    complete=not issues
    observed_deltas=[r['delta_U'] for r in pair_checks if r['eligible_pair']]
    result=dict(status=('COMPLETE_INTERRUPTED_AUDITED' if interrupted else 'COMPLETE_AUDITED') if complete else 'AUDIT_FAILED',issues=issues,
        interrupted=interrupted,expected_formal_runs=expected_formal,unexecuted_jobs=unexecuted,declared_issues=declared_issues,
        all_five_pair_conclusion='UNKNOWN' if interrupted else 'AVAILABLE',no_retry_preserved=state.get('no_retries') is True,
        reserved_pair_summary=dict(declared_pairs=5,evaluable_pairs=len(observed_deltas),observed_subset_mean_delta_U=mean(observed_deltas),
            all_five_mean_delta_U=mean(observed_deltas) if len(observed_deltas)==5 else None,positive=sum(v>0 for v in observed_deltas),negative=sum(v<0 for v in observed_deltas),zero=sum(v==0 for v in observed_deltas)),
        generator=receipt(__file__),main_code_freeze=receipt(study/'main-code-freeze-v1.json'),
        finished_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),workflow_status=state['status'],
        selection=selection,native=native,calibration=calibration,cpu=cpu,cpu_efficiency=efficiency,native_measured_dfs=native_dfs,
        six_term_arithmetic_and_cohorts=arithmetic,reserved_pair_checks=pair_checks,independent_final_closure=closure,old_protected_files=len(old['protected']),
        new_native_runs=len(native),calibration_runs=len(cal['jobs']),total_new_runs=len(native)+len(cal['jobs']),
        old_outputs_unchanged=not any(i.startswith('OLD_') for i in issues),bindings=bindings,
        original_weights_preserved=not any(i.startswith('ORIGINAL_PROTOCOL_CHANGED:') for i in issues),absolute_quality_gate=False,physical_full_seven_pool_optimality=False,
        limitations=['Three exact resident deployments only; missing deployments/models are UNKNOWN, not infeasible.',
         'CPU environment is a conditional mean FIFO proxy. Measured total U includes Q/T/L/P/M/E.',
         'Online global_incremental scores Q/L/P and optional known E only; T is unknown online. Offline screening scores QTLP. A broad contract local_objective string does not add T to the actual route scorer.',
         'Reserved efficacy pair isolates routing within primary residency; alternative deployment remains development evidence.',
         'Shared historical input pool does not test unseen-input generalization.',
         'Measured table DFS lookup equality is not a GPU efficiency benchmark.'])
    write(output/'audit-and-data.json',result)
    table=[]
    for row in native:
        c=row['contract'];s=row['summary'];ref=c.get('refinement') or {};u=s['utility'];mech=row['mechanism']
        table.append(dict(stage=row['stage'],load_text_rps=row['load'],role=row['role'],seed=s['seed'],
            deployment=json.dumps(c['deployment'],sort_keys=True),mode=ref.get('mode','legacy_latency'),
            risk_quantile=ref.get('risk_quantile'),margin_ms=ref.get('margin_ms'),eligible=u.get('eligible'),U=u.get('U'),
            stability_state=(s.get('stability') or {}).get('stability_state_v2'),request_count=s['request_count'],
            **{t+'_norm':u.get(t+'_norm') for t in 'QTLPME'},
            **{'contribution_'+t:s['contributions'][t] for t in 'QTLPME'},
            text_to_g2=mech['spilled_text'],unique_text_to_g2=mech['unique_spilled_inputs'],
            spill_both_specialists_pending=mech['spill_both_specialists_pending'],spill_generalist_visible_empty=mech['spill_generalist_visible_empty'],
            g2_mean_actual_service_ms=mech['prediction_diagnostics'].get('generalist_2',{}).get('mean_actual_ms'),
            g2_mean_predicted_service_ms=mech['prediction_diagnostics'].get('generalist_2',{}).get('mean_predicted_ms'),
            g2_service_MAE_ms=mech['prediction_diagnostics'].get('generalist_2',{}).get('mean_absolute_error_ms'),
            cohort_sha256=s['cohort_sha256'],
            actual_text_routes=json.dumps(mech['counts']['text'],sort_keys=True),
            per_replica=json.dumps(s['alias_summaries'],sort_keys=True),raw_path=s['raw']['path'],raw_sha256=s['raw']['sha256']))
    if table:
        with (output/'native-runs.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    plots(native,cpu,output,selected_load=(selection.get('selected') or {}).get('load'))
    lines=['# 校准后队列分流：本轮完整实验记录','',f'记录时间（北京时间）：{result["finished_beijing"]}',
      f'审计状态：{result["status"]}；新 GPU 运行 {result["total_new_runs"]} 条（校准 {result["calibration_runs"]}、正式 {result["new_native_runs"]}）。','',
      '总效用继续使用原 Q/T/L/P/M/E，六项等权；质量由实际输出评分，不设绝对精度准入。600 秒测量、30 秒预热、300 秒排空及原稳定性、rho≤0.8、显存安全系数均保持原定义。','',
      'U=−(Q+T+L+P+M+E)/6，越大越好。Q 按任务到达率加权，分别为文本 1−EM、图像 1−accuracy、OCR CER、ASR WER；CER/WER 不截断。独立阶段的质量来自实际生成输出。','',
      '![独立效能与有限域搜索结果](results.png)','',
      '## 模型池、负载与比较范围','',
      '冻结的七模型池保持原定义。本轮具备同驻留服务证据的有限搜索域包括：ChatQA×2+G2×1+OCR×1，ChatQA×2+SigLIP2×1+OCR×1+Granite×1，以及主混合部署增加 SigLIP2×1。其余部署为 UNKNOWN，不据此宣称全七模型池物理全局最优。',
      '','文本分别为 1.85、1.89 请求/秒；图像、OCR、ASR 各 0.01 请求/秒，泊松到达。两个负载各自比较，不以跨负载绝对 U 排名推断部署优势。','',
      '主混合部署的图像/ASR 固定交给 G2，OCR 交给 PP-OCRv6；文本在两个 ChatQA 副本间按可见队列分配，并由分流策略自主决定是否交给 G2。增加 SigLIP2 的部署把图像交给 SigLIP2，纯专部署把 ASR 交给 Granite。','',
      '## 开发阶段原六项总效用','',
      '|文本负载|方案角色|部署|模式/风险分位/余量|seed203 U|seed204 U|均值 U|实际文本→G2|',
      '|---|---|---|---|---|---|---|---|']
    if interrupted:
        lines[5:5]=['原流程终态为 STOPPED_FOR_DIAGNOSIS_NO_RETRY：正式25/26次已登记、校准4次，共29次已测；最后seed209零分流控制未执行。seed209分流运行请求已完成，但能耗覆盖无效、run_joules/E_norm/U均未知。遵守无重试协议，未补测、未补能耗、未删失败项；COMPLETE_INTERRUPTED_AUDITED仅表示已存在证据的中断闭合核验，不表示原实验完整完成。','']
    for rate in (1.85,1.89):
        for role in ('calibrated_spill','risk_matched_no_spill','legacy_latency','strong_other_deployment'):
            rows=[r for r in native if r['stage'].startswith('development') and r['load']==rate and r['role']==role]
            if not rows:continue
            c=rows[0]['contract'];v={r['summary']['seed']:r['summary']['utility'] for r in rows};eligible=len(rows)==2 and all(u.get('eligible') for u in v.values())
            refinement=c.get('refinement') or {};mode='/'.join(str(refinement.get(k,'旧版')) for k in ('mode','risk_quantile','margin_ms'))
            lines.append(f'|{rate}|{role}|{json.dumps(c["deployment"],ensure_ascii=False)}|{mode}|{fmt(v.get(203,{}).get("U"))}|{fmt(v.get(204,{}).get("U"))}|{fmt(mean([u["U"] for u in v.values()]) if eligible else None)}|'+','.join(str(r['mechanism']['spilled_text']) for r in sorted(rows,key=lambda r:r['summary']['seed']))+'|')
    lines+=['','开发数据只用于选择。校准分流与零分流控制使用同驻留、同模式和同风险分位；两者仅允许的文本路由路径不同。旧版延迟路由和另一部署是明确的额外替代方案。','',
      '## 输入无关随机校准','',
      '|种子|零探测 U|随机探测 U|随机−零 ΔU|实际探测文本|覆盖输入数|','|---|---|---|---|---|---|']
    for seed in (201,202):
        pair={r['probability']:r for r in calibration if r['seed']==seed}
        if len(pair)!=2:continue
        a=pair[1/16];b=pair[0.];ua=a['utility'];ub=b['utility']
        delta=ua.get('U')-ub.get('U') if ua.get('eligible') is True and ub.get('eligible') is True else None
        lines.append('|'+ '|'.join([str(seed),fmt(ub.get('U')),fmt(ua.get('U')),fmt(delta),str(a['mechanism']['spilled_text']),str(a['mechanism']['unique_spilled_inputs'])])+'|')
    lines+=['','随机探测用于拓宽服务训练覆盖，不按收益挑请求；此处的效用差属于校准干预描述，正式结论依据后续自主策略的独立配对。','',
      '## 冻结选择与独立配对','']
    if selection['status']=='FROZEN':
        selected=selection['selected'];lines += [f'按开发阶段配对 ΔU 选定文本负载 {selected["load"]} 请求/秒，开发平均 ΔU={fmt(selected["mean_delta_U"])}；之后冻结两个候选和全部独立种子，未重选、未重拟合。','',
          '|种子|分流 U|同模式同风险零分流 U|ΔU|实际文本→G2|双专用均有待处理|G2可见队列空|同输入到达cohort|ΔQ贡献|ΔT贡献|ΔL贡献|ΔP贡献|ΔM贡献|ΔE贡献|',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        delta=[];actual_spill_counts=[];backlog_spill_counts=[]
        template=next(r for r in native if r['load']==selected['load'] and r['role']=='calibrated_spill')
        ref=template['contract'].get('refinement') or {}
        lines.insert(len(lines)-2,'冻结分流参数：'+json.dumps({k:ref.get(k) for k in ('mode','risk_quantile','margin_ms','concurrency_context')},ensure_ascii=False)+'。驻留仍为 ChatQA×2+G2×1+OCR×1。')
        lines.insert(len(lines)-2,'')
        for seed in (205,206,207,208,209):
            pairs={r['role']:r for r in native if not r['stage'].startswith('development') and r['summary']['seed']==seed}
            a=pairs.get('calibrated_spill');b=pairs.get('risk_matched_no_spill');ua=a['summary']['utility'] if a else {};ub=b['summary']['utility'] if b else {};ok=ua.get('eligible') is True and ub.get('eligible') is True
            d=ua['U']-ub['U'] if ok else None
            if d is not None:delta.append(d)
            if a:actual_spill_counts.append(a['mechanism']['spilled_text']);backlog_spill_counts.append(a['mechanism']['spill_both_specialists_pending'])
            contrib=[a['summary']['contributions'][t]-b['summary']['contributions'][t] if ok else None for t in 'QTLPME']
            mech=a['mechanism'] if a else {};same_cohort=a['summary']['cohort_sha256']==b['summary']['cohort_sha256'] if a and b else None
            lines.append('|'+ '|'.join([str(seed),fmt(ua.get('U')) if a else 'UNEXECUTED',fmt(ub.get('U')) if b else 'UNEXECUTED',fmt(d),str(mech.get('spilled_text','UNKNOWN')),str(mech.get('spill_both_specialists_pending','UNKNOWN')),str(mech.get('spill_generalist_visible_empty','UNKNOWN')),str(same_cohort) if same_cohort is not None else 'UNKNOWN']+[fmt(v) for v in contrib])+'|')
        positive=sum(d>0 for d in delta)
        lines+=['',f'完整可评价配对 {len(delta)}/5，平均 ΔU={fmt(mean(delta))}，其中正收益 {positive}/{len(delta)}。',
          'Δ贡献=−w×(分流方案的归一项−控制方案的归一项)，正值表示改善；六项Δ贡献之和等于ΔU。',
          '收益判断使用整次配对运行；不能把每个请求当作独立复现实验。']
        if interrupted:lines += ['', '这里的平均ΔU只描述4个完整可评价配对；预注册仍是5个配对，第五个保持UNKNOWN，因此全部5次配对平均ΔU、5次稳定优势及完整最终确认均为UNKNOWN。seed208若效用严重下降但证据合格，仍原样进入这4个配对及均值，不作为异常值删除。']
        lines += ['', '|种子|G2文本服务样本|预测均值ms|实际均值ms|逐请求实际/预测比值均值|服务MAE ms|实际−预测均值ms|', '|---|---|---|---|---|---|---|']
        for seed in (205,206,207,208,209):
            row=next((r for r in native if not r['stage'].startswith('development') and r['summary']['seed']==seed and r['role']=='calibrated_spill'),None)
            if row is None:continue
            p=row['mechanism']['prediction_diagnostics'].get('generalist_2',{})
            lines.append('|'+ '|'.join([str(seed),str(p.get('n',0)),fmt(p.get('mean_predicted_ms'),2),fmt(p.get('mean_actual_ms'),2),fmt(p.get('mean_ratio'),3),fmt(p.get('mean_absolute_error_ms'),2),fmt(p.get('mean_signed_error_actual_minus_predicted_ms'),2)])+'|')
        lines += ['', '“双专用均有待处理”与“G2可见队列空”均为分流决策时主机可见 outstanding 状态；可见队列空不能证明物理 GPU 空闲。实际服务优先读取 worker_response.service_ms，预测仅使用决策时冻结模型；未选路径的真实服务和答案没有观测。']
        if len(delta)==5:
            mechanism_seen=len(actual_spill_counts)==5 and all(n>0 for n in actual_spill_counts) and all(n>0 for n in backlog_spill_counts)
            if positive==5 and mechanism_seen:
                verdict='这5次冻结配对均观察到实际分流、双专用待处理时分流及总效用提升，支持此条件下的重复性；仍不代表对其他负载、输入池或硬件已证明泛化。'
            elif positive==5:
                verdict='该允许分流策略在5次配对中的整体U均较高，但实际拥堵分流证据不足，尚不能将优势归因为分流机制。'
            else:verdict='本轮尚未获得每个独立种子均为正的稳定分流收益；应保留该负结果，不能据此声称已证明稳定优势。'
            lines += ['',verdict]
    else:lines += ['没有在两个开发种子上均完整可评价的主部署配对，独立效能阶段未启动；结果记为未解决，没有替换未知方案。']
    lines+=['','## 算法与 DFS','',
      '|文本负载|原候选/代表|结构等价省叶|上界剪枝省叶|DFS调用数|优化调用数|调用下降|DFS秒|优化秒|CPU实测加速倍数|同一候选/值/全部并列|','|---|---|---|---|---|---|---|---|---|---|---|']
    for report,e in zip(cpu,efficiency):
        a=report['plain_dfs'];b=report['bounded'];lines.append('|'+ '|'.join([str(report['load']['text']),str(a['candidates'])+'/'+str(b.get('representatives')),str(b.get('proven_equivalent_skips')),str(b.get('bound_pruned')),str(a['oracle_calls']),str(b['oracle_calls']),fmt(e['call_reduction_percent'],2)+'%',fmt(a['elapsed_seconds'],3),fmt(b['elapsed_seconds'],3),fmt(e['observed_cpu_speedup'],3)+'×',str(report['same_winner'] and a['tie_ids']==b['tie_ids'])])+'|')
    lines += ['', '每个负载原51候选×2个CPU种子=102次模拟；结构等价合并后39代表×2=78次，实际减少24次、23.53%。这24次全部来自预先证明的纯专部署策略等价；上界在叶级节省0次，种子级也节省0次。有些 SEED_BOUND_PRUNED 标记出现在最后一个种子已算完后，只抑制劣解进入排名，不节省模拟调用。结构等价展开保留全部51身份及并列最优集。',
      'CPU时间包含结构归组、上界计算、代表模拟和原域展开，模型对象在计时前构造、每次搜索清空模型缓存、没有预存目标值缓存。时间是同机单进程且CPU非独占的一次观察，不能据此声称GPU推理或整体GPU补测同倍率提速；额外全DFS等价审计属于验证成本。']
    lines+=['','|文本负载|51候选代理域最佳部署|模式|风险分位|余量ms|代理QTLP目标|','|---|---|---|---|---|---|']
    for report in cpu:
        winner=report['comparison']['finite_domain_winner']
        if winner is None:continue
        c=winner['candidate'];lines.append('|'+ '|'.join([str(report['load']['text']),json.dumps(c['deployment'],ensure_ascii=False),c['mode'],str(c['risk_quantile']),str(c['margin_ms']),fmt(winner['objective'])])+'|')
    lines+=['','上述为51候选代理域的整体最佳方案。主部署分流候选是预注册的同驻留机制比较方案，可能与整体最佳不同。CPU按部分QTLP代理排序；GPU开发阶段在每个负载明确的4个候选内按完整六项U排序，不能把同驻留配对收益直接改写成全域部署最优。',
      '', '|实测阶段|候选数/种子数|完整六项U最佳角色|最佳部署|均值U|精确最大均值U|并列最优角色|DFS/优化/独立重算一致|', '|---|---|---|---|---|---|---|---|']
    for row in native_dfs:
        lines.append('|'+ '|'.join([row['stage'],str(row.get('candidates','UNKNOWN'))+'/'+str(len(row.get('seeds',[]))),str(row.get('winner_role')),json.dumps(row.get('winner_deployment'),ensure_ascii=False),fmt(row.get('objective')),fmt(row.get('exact_max_objective')),','.join(str(v) for v in row.get('tie_roles',[])),str(row.get('same_winner_value_exact_max_and_ties',False))])+'|')
    lines += ['', '实测DFS核验逐项比较最佳候选身份、所选候选均值、精确最大值及全部并列身份，并从原始运行重新汇总均值、核对六项效用与相同输入到达cohort；已保存比较表的每条效用及raw SHA也必须与原始汇总一致。独立阶段只验证冻结的主部署二策略；其他部署的名次仍是开发证据。']
    lines+=['','这些 CPU 数值针对声明的条件均值代理和 QTLP 部分目标。在线 global_incremental 路由实际使用固定 N/K 的 Q/L/P 增量代理；T 在线保持未知，本轮未提供能耗率，E 在线亦未知。合同 local_objective 的笼统 QTLP 文案不改变实际评分字段。','',
      '边界是否提速按实测报告，不把缺失能耗/显存成本当作零，更不把它解释为物理全局最优。正式 GPU 比较表另经 DFS 与剪枝搜索复核；查表耗时不代表节省了 GPU 推理。','',
      '## 结果边界与数据','',
      '模型校准按完整驻留部署隔离。随机探测用于改善 G2 输入覆盖；残余时间和并发条件预测仍是经验模型，并非可保证的置信上界。路由只读取到达时可见队列、历史模型和任务身份，未读取真实未来服务、答案或评测标签。',
      '','各运行汇总表见 [native-runs.csv](native-runs.csv)。完整六项分解、逐副本路由、预测误差、所有分流事件、原始记录收据和不可变性核验见 [audit-and-data.json](audit-and-data.json)。']
    if closure['present']:
        lines += ['',f'独立审计最终闭合：{closure["document"].get("status")}；本报告核验其全部引用收据、运行数量、各阶段赢家/值/并列集及独立配对六项分解，与本次重算一致={closure["verified"]}。文件：[final-closure.json]({closure["receipt"]["path"]})；SHA256：{closure["receipt"]["sha256"]}。']
        if closure.get('recovery'):
            recovery=closure['recovery'];doc=recovery['document']
            lines += ['',f'审计恢复来源：[recovery-receipt.json]({recovery["receipt"]["path"]})，SHA256：{recovery["receipt"]["sha256"]}。恢复时旧审计已完成{doc.get("old_formal_count","UNKNOWN")}次，已逐项核验恢复收据列出的旧进度、日志、审计源码及其他保护文件，核验通过={recovery["verified"]}。恢复只承接审计，未重试GPU测量；原审计进程停止原因保持原收据描述，不推断为OOM或GPU故障。']
    else:lines += ['', '独立审计的 final-closure.json 尚未生成；本报告记录为未引用，不能替代该独立闭合。']
    if interrupted:
        lines += ['', '## 中断与缺失证据', '', '最后阶段的原analysis未生成。本报告只从原监督进程已登记raw构造附带的partial measured table，保留两个候选及预注册207/208/209三个种子；任何候选缺一个完整有效总效用，其三种子均值、赢家及并列最优集均不可评价。原workflow/stage状态和raw收据没有重写。', '']
        for failure in declared_issues:
            lines += [f'- seed{failure["seed"]} {failure["role"]}：{failure["reason"]} 原始收据：{failure["raw"]["path"]}；SHA256={failure["raw"]["sha256"]}。能耗证据：'+json.dumps(failure['energy_evidence'],ensure_ascii=False)+'。']
        if closure.get('energy_failure_source'):
            source=closure['energy_failure_source'];doc=source['document'];gaps=[g for g in doc.get('excess_sample_gaps',[]) if g.get('overlaps_measurement')]
            lines += ['', f'独立缺能耗一致性复核：[seed209-missing-energy-consistency.json]({source["receipt"]["path"]})；SHA256={source["receipt"]["sha256"]}。测量窗口重叠的过长功率采样间隔：'+json.dumps(gaps,ensure_ascii=False)+f'，超过原阈值{doc.get("original_max_gap_seconds","UNKNOWN")}秒，因此原积分规则返回UNKNOWN；这属于资源测量未知，不是数学不可行。采样间隔变长的根因仍未知，不能归因为GPU或算法。原审计对None积分的数值比较问题保留原记录，由此补充复核说明；五项已知贡献之和也不替代完整六项U。']
    (output/'实验报告.md').write_text('\n'.join(lines)+'\n')
    (output/'README.md').write_text('[实验报告](实验报告.md)\n\n完整数据：[audit-and-data.json](audit-and-data.json)\n')
    if issues:raise ValueError('Final audit failed: '+str(issues))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--execution',type=Path,required=True);p.add_argument('--screening',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--independent-closure',type=Path,help='Explicit final independent closure, including resumed auditor directory; defaults to original audit directory');p.add_argument('--allow-interrupted',action='store_true',help='Explicitly close a registered STOPPED_FOR_DIAGNOSIS_NO_RETRY run, retaining failed/missing outcomes as UNKNOWN');a=p.parse_args()
    result=close(a.study,a.execution,a.screening,a.output,a.independent_closure,a.allow_interrupted);print(json.dumps({k:result[k] for k in ('status','total_new_runs','new_native_runs','calibration_runs','issues')},ensure_ascii=False))
