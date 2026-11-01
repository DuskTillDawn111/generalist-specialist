"""Independent raw-event and probability-preservation check of frozen v15 prior."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import sqlite3
from unittest.mock import patch

from .context_initial_prior_v14 import InitialContextPrior
from .context_initial_prior_v15 import JointInitialContextPrior,realize_joint_initial_scenario
from .multi_native_composite_v5 import checked
from .native_collect import receipt,save
from .profiling import digest


def verify(model_path,output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    ref=receipt(model_path);model=checked(ref);old=checked(model['original_prior'])
    new_provider=JointInitialContextPrior(model);old_provider=InitialContextPrior(old)
    queries=[];seen=set()
    for row in old['scenarios']:
        key=digest([row['first_pending_frontier'],row['public_setup']])
        if key not in seen:seen.add(key);queries.append((row['first_pending_frontier'],row['public_setup']))
    first=old['scenarios'][0];pool=first['first_pending_frontier'][0][2];setup=first['public_setup']
    queries.extend([([['text#0','text',pool,['prefill']]],setup),
                    ([['text#0','text',pool,['outside_language_forwards']],['text#1','text',pool,['outside_language_forwards']]],setup)])
    probabilities=[]
    # Any source-event lookup or file load in prospective predict/realize fails.
    with patch('sqlite3.connect',side_effect=AssertionError('prospective source event query')),patch('pathlib.Path.read_bytes',side_effect=AssertionError('prospective file read')):
        for frontier,setup in queries:
            args=dict(first_pending_frontier=frontier,public_execution_setup=setup,
                      allow_unvalidated_frontier_pooling=True,allow_unvalidated_observer_transfer=True)
            before=old_provider.predict(**args);after=new_provider.predict(**args)
            lhs=[(r['scenario_id'],r['probability']) for r in before['scenarios']]
            rhs=[(r['scenario_id'],r['probability']) for r in after['scenarios']]
            if lhs!=rhs:raise AssertionError('Original joint scenario probability/order changed')
            if abs(before['unknown_holder_mass']-after['in_transition_mass']-after['unknown_state_mass'])>1e-12:
                raise AssertionError('Unknown mass dropped or renormalized')
            probabilities.append(dict(frontier=frontier,conditioning_tier=after['conditioning_tier'],
                scenarios=len(rhs),original_unknown_holder_mass=before['unknown_holder_mass'],
                transition_mass=after['in_transition_mass'],unresolved_unknown_state_mass=after['unknown_state_mass'],
                probabilities_exactly_preserved=True))
            for scenario in after['scenarios']:
                if scenario['state_kind']=='IN_TRANSITION':
                    realized=realize_joint_initial_scenario(scenario,query_clock_ns=1_000_000_000)
                    if realized['service_ready_ns'] is not None:raise AssertionError('Missing readiness silently made a point')
    source=checked(model['source_supervision']);raw_checks=[];refs=[receipt(__file__),ref,model['original_prior'],model['source_supervision']]
    for row in model['scenarios']:
        if row['state_kind']!='IN_TRANSITION':continue
        audit=next(a for a in source['training_audits'] if a['raw']['sha256']==row['capture'])
        report=checked(audit['report']);sql=next(r for r in report['sources'] if r['path'].endswith('.sqlite'))
        refs.extend([audit['report'],sql]);c=sqlite3.connect('file:'+sql['path']+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
        tags={r['id']:r['name'] for r in c.execute('select * from ENUM_GPU_CTX_SWITCH')}
        mapping={tuple(r['context']):r['alias'] for r in report['controlled_temporal_attribution']['creation_windows']}
        clock=row['state_evidence']['query_clock_ns'];prior=[];future=[];last={}
        for event in c.execute('select * from GPU_CONTEXT_SWITCH_EVENTS order by timestamp,seqNo'):
            key=(event['vmId'],event['gpuId'],event['contextId'],event['globalPid']);tag=tags[event['tag']]
            value=dict(event,tag_name=tag,owner=mapping.get(key))
            if event['timestamp']<=clock:
                last[key]=tag
                if tag=='SAVE_END':prior.append(value)
            elif tag=='RESTORE_START':future.append(value)
        if any(v=='RESTORE_START' for v in last.values()):raise AssertionError('Direct raw events show a held recorded context')
        previous=max(prior,key=lambda r:r['timestamp']);following=min(future,key=lambda r:r['timestamp'])
        if (previous['owner']!=row['previous_held_owner'] or following['owner']!=row['next_owner']
                or clock-previous['timestamp']!=row['elapsed_gap_ns']
                or following['timestamp']-clock!=row['residual_restore_gap_ns']):
            raise AssertionError('Joint transition label differs from direct raw source events')
        lo=following['timestamp'];hi=row['state_evidence']['next_SAVE_ns']
        kernels=c.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<?',(lo,hi)).fetchone()[0]
        c.close()
        if kernels==0 and row['service_ready_delay_after_restore_bounds_ns']['upper_ns'] is not None:
            raise AssertionError('Service upper witness invented in an empty kernel envelope')
        raw_checks.append(dict(scenario_id=row['scenario_id'],previous_SAVE=previous,next_RESTORE=following,
            elapsed_gap_ns=row['elapsed_gap_ns'],residual_restore_gap_ns=row['residual_restore_gap_ns'],
            observed_kernels_in_next_envelope=kernels,service_upper_preserved_unknown=row['service_ready_delay_after_restore_bounds_ns']['upper_ns'] is None,
            raw_event_initial_gap_verified=True))
    for reference in refs:
        if receipt(reference['path'])!=reference:raise ValueError('Independent verification source changed')
    result=dict(status='INDEPENDENT_RAW_EVENT_AND_ORIGINAL_PROBABILITY_CHECK_PASS',sources=refs,
        original_rows=len(old['scenarios']),new_rows=len(model['scenarios']),raw_event_checks=raw_checks,
        probability_queries=probabilities,predict_and_realize_prohibited_all_source_event_and_file_reads=True,
        all_original_joint_scenario_probabilities_exactly_preserved=True,unknown_rows_deleted=False,
        missing_readiness_imputed=False,validation_labels_used=False,GPU_started=False)
    save(out/'report.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=verify(a.model,a.output);print(json.dumps(dict(status=r['status'],queries=len(r['probability_queries']),raw_gaps=len(r['raw_event_checks']))))
