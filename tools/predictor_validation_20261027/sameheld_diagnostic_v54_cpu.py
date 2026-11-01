"""Bounded read-only audit of frozen v54 and its original physical labels."""
import argparse
from collections import Counter, deque
import gzip
import hashlib
import json
from pathlib import Path

from .implementation_source_closure_v27 import implementation_sources, strict_references
from .native_collect import receipt
from .owner_lifecycle_mark_v21 import OwnerLifecycle
from .projected_handler_bridge_v44 import ObservableReferenceMark, feature_projection
from .projected_handler_context_v42 import ObservableContextCandidate, OWNERS
from .unknown_preflight_archive_v1 import save_synced


def selected(path):
    # ijson is an external read-only parser, not a model dependency.
    import ijson
    from ijson.common import ObjectBuilder
    objects = {'configuration', 'plan', 'reason',
        'scheduler_diagnostic.oracle.current_lifecycle',
        'scheduler_diagnostic.oracle.current_lifecycle_clock_ns',
        'scheduler_diagnostic.oracle.latest_cumulative_hazard',
        'scheduler_diagnostic.held_owner', 'scheduler_diagnostic.restore_start_ns',
        'scheduler_diagnostic.review_deadline_ns', 'scheduler_diagnostic.pending_switch'}
    tails = {key:deque(maxlen=n) for key,n in {
        'scheduler_diagnostic.oracle.decisions.item':2,
        'scheduler_diagnostic.transitions.item':12,
        'scheduler_diagnostic.queries.item':4}.items()}
    result, counts = {}, Counter()
    builder = None; depth = 0; target = None
    with gzip.open(path, 'rb') as stream:
        for prefix,event,value in ijson.parse(stream, use_float=True):
            if builder is not None:
                builder.event(event,value)
                depth += (event in ('start_map','start_array')) - (event in ('end_map','end_array'))
                if depth == 0:
                    if target in tails: tails[target].append(builder.value)
                    else: result[target] = builder.value
                    builder = None
            elif prefix in objects or prefix in tails:
                if prefix in tails: counts[prefix] += 1
                if event in ('start_map','start_array'):
                    builder = ObjectBuilder(); builder.event(event,value); depth=1; target=prefix
                elif event not in ('end_map','end_array','map_key'): result[prefix]=value
    result.update({key:list(value) for key,value in tails.items()})
    result['retained_list_counts'] = dict(counts)
    return result


def run(output):
    import ijson
    root=Path(__file__).resolve().parents[2];out=Path(output).absolute()
    forecast=root/'artifacts/predictor-projected-handler-v54-20261028-preflight01/shared_spill_high-forecast.json.gz'
    ref=receipt(forecast)
    if ref['sha256']!='1c4352f0d5d86958277cc61f4461fc573d9bfe1334530f544c5d4f478ddf3654':
        raise ValueError('Exact frozen v54 forecast required')
    data=selected(forecast);sha=hashlib.sha256();size=0
    with gzip.open(forecast,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):sha.update(block);size+=len(block)
    if size!=527827782 or sha.hexdigest()!='4875f922fe6de7e5e41bf772952becfa962b6fe2b33258fcc73c467791e42bb0':
        raise ValueError('Full gzip CRC/hash/length mismatch')
    hazard=data['scheduler_diagnostic.oracle.latest_cumulative_hazard'];now=data['scheduler_diagnostic.oracle.current_lifecycle_clock_ns']
    held=data['scheduler_diagnostic.held_owner'];previous=data['scheduler_diagnostic.oracle.decisions.item'][-1]['held_owner']
    states={o:OwnerLifecycle(**s) for o,s in data['scheduler_diagnostic.oracle.current_lifecycle'].items()}
    model_ref=data['configuration']['observable_context_model'];model=json.loads(Path(model_ref['path']).read_text())
    candidate=ObservableContextCandidate(model)
    features=feature_projection(held,previous,hazard['restore_age_ns'],states)
    physical=candidate.next_owner(features)
    mark=object.__new__(ObservableReferenceMark);mark.candidate=candidate;mark.states=states;mark.clock=now
    bridged=mark.predict_given_save_now(held_owner=held,previous_held_owner=previous,context_age_ns=hazard['restore_age_ns'],pending_frontier=hazard['state']['frontier'])
    if not (data['reason']=='UNSUPPORTED_SAME_CONTEXT_REACQUISITION_PREDICTION'
            and now==hazard['absolute_save_candidate_ns']==31485677331
            and physical['most_likely_physical_owner']=='text#0'
            and bridged['most_likely_next_owner']==held
            and hazard['consumed_hazard']==hazard['hazard_budget'] and hazard['budget_redrawn'] is False):
        raise ValueError('Frozen exact-time bridge mechanism changed')
    table={(o,format(bits,'04b')):dict(held_owner=o,mask=format(bits,'04b'),rows=0,exposure_ns=0,events=0,
        unresolved_rows=0,unresolved_exposure_ns=0,unresolved_events=0,age_min_ns=None,age_max_ns=None,by_capture={})
        for o in OWNERS for bits in range(16)}
    marks=[];refs=[ref,model_ref];native_unresolved=[]
    for capture,entry in model['captures'].items():
        report=json.loads(Path(entry['report']['path']).read_text());native=json.loads(Path(report['source_adapter']['path']).read_text())
        refs.extend([entry['report'],report['risk_chronology'],report['mark_records'],report['source_adapter'],native['native_transition_ledger']])
        with gzip.open(report['risk_chronology']['path'],'rt') as stream:
            for line in stream:
                row=json.loads(line);mask=''.join('1' if row['shared_left_features']['handler_active'][o] else '0' for o in OWNERS)
                v=table[row['held_owner'],mask];width=row['end_ns']-row['start_ns'];event=int(row['physical_SAVE_END_observed'])
                if not row['exposure_qualified']:
                    v['unresolved_rows']+=1;v['unresolved_exposure_ns']+=width;v['unresolved_events']+=event;continue
                v['rows']+=1;v['exposure_ns']+=width;v['events']+=event
                v['age_min_ns']=min(v['age_min_ns'],row['age_entry_ns']) if v['age_min_ns'] is not None else row['age_entry_ns']
                v['age_max_ns']=max(v['age_max_ns'],row['age_exit_ns']) if v['age_max_ns'] is not None else row['age_exit_ns']
                b=v['by_capture'].setdefault(entry['route'],dict(rows=0,exposure_ns=0,events=0));b['rows']+=1;b['exposure_ns']+=width;b['events']+=event
        count=Counter()
        with gzip.open(report['mark_records']['path'],'rt') as stream:
            for line in stream:
                row=json.loads(line);count['qualified_marks']+=1;count['same_owner']+=int(row['same_owner']);count['same_driver_context']+=int(row['same_driver_context'])
                mask=row['features']['handler_active'];count['target_handler_inactive']+=int(not mask[row['next_owner']])
                count['only_held_active']+=int(mask[row['features']['held_owner']] and sum(mask.values())==1)
        marks.append(dict(route=entry['route'],capture=capture,**count))
        count=Counter()
        with Path(native['native_transition_ledger']['path']).open('rb') as stream:
            for row in ijson.items(stream,'unresolved.item'):
                count['unresolved']+=1;count['same_owner']+=int(row['same_owner']);count['same_driver_context']+=int(row['same_driver_context'])
                count['same_owner_different_context']+=int(row['same_owner'] and not row['same_driver_context'])
        native_unresolved.append(dict(route=entry['route'],**count))
    refs=strict_references([refs,implementation_sources(__file__)])
    for source in refs:
        if receipt(source['path'])!=source:raise ValueError('Frozen source changed')
    values=list(table.values());total={key:sum(v[key] for v in values) for key in ('rows','exposure_ns','events','unresolved_rows','unresolved_exposure_ns','unresolved_events')}
    if (total['rows'],total['exposure_ns'],total['events'])!=(107685,130426645735,105141):raise ValueError('Original population changed')
    out.mkdir(parents=True,exist_ok=False);save_synced(out/'selected-diagnostic.json',data);save_synced(out/'all64-statistics.json',values)
    result=dict(schema='gsdeploy.v54_sameheld_bounded_readonly_diagnostic.v1',status='MECHANISM_IDENTIFIED_NOT_VALIDATED',
        original_forecast=ref,uncompressed_bytes=size,uncompressed_sha256=sha.hexdigest(),full_gzip_CRC_hash_length_verified=True,
        selected_diagnostic=receipt(out/'selected-diagnostic.json'),all64_statistics=receipt(out/'all64-statistics.json'),
        reason=data['reason'],clock_ns=now,restore_start_ns=data['scheduler_diagnostic.restore_start_ns'],restore_age_ns=hazard['restore_age_ns'],late_by_ns=0,
        physical_mark=physical,active_conditioned_mark=bridged,features=features,
        mechanism='Raw physical mark selects inactive text#0; conditioning on the sole active generalist maps it to the held owner and triggers the original same-held UNKNOWN guard.',
        skipped_SAVE=False,raw_same_holder_prediction=False,observed_same_owner_different_context_evidence=False,
        qualified_risk_totals=total,observed_strata=sum(v['exposure_ns']>0 for v in values),unseen_strata=sum(v['exposure_ns']==0 for v in values),
        positive_event_strata=sum(v['events']>0 for v in values),zero_event_exposed_strata=sum(v['exposure_ns']>0 and v['events']==0 for v in values),
        qualified_mark_support=marks,native_unresolved_adjacency=native_unresolved,sources=refs,
        limitations=['Predicted physical events and projected handler clocks are unvalidated hypotheses.',
            'No qualified mark targets an inactive handler; raw inactive-target restoration is unsupported by these labels.',
            'Unresolved same-owner adjacency is not qualified physical transition evidence.',
            'Age minima/maxima do not imply continuous support; v57 records the union of exposure intervals.',
            'Jeffreys zero-event means can be large under tiny exposure; no monotone hazard reduction is claimed.'],
        GPU_started=False,validation_labels_read=False,fit_performed=False,frozen_sources_modified=False)
    save_synced(out/'report.json',result);return receipt(out/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(json.dumps(run(parser.parse_args().output)))
