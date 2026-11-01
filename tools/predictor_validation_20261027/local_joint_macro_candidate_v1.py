"""Isolated empirical local-macro joint atoms for training diagnostics only.

Joint atoms retain response, internal host END/query and notification delays.
No cohort/request lookup, interpolation, physical service claim or runtime hook.
This empirical law has no qualified cross-macro transition or loss model.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

SCHEMA='gsdeploy.local_joint_macro_training_candidate.v1'
FEATURES={'owner','public_input_id','semantic_phase','prompt_sequence_length','input_batch_size','forward_index',
    'terminal_kind','peer_macro_count_at_actual_BEGIN','peer_same_phase_count_at_actual_BEGIN'}
CLOCKS=('occurrence_ns','reader_receipt_ns','dequeued_ns','processed_ns')


def receipt(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(ref):
    if receipt(ref['path'])!=ref:raise ValueError('Changed training source: '+ref['path'])
    return json.loads(Path(ref['path']).read_text())


def canonical(features):
    if type(features) is not dict or set(features)!=FEATURES:
        raise ValueError('Exact local features required; request/cohort identities are forbidden')
    if (features['semantic_phase'] not in ('prefill','decode')
            or features['terminal_kind'] not in ('next_forward_BEGIN','native_END')
            or type(features['owner']) is not str
            or type(features['public_input_id']) is not str or not features['public_input_id']
            or any(type(features[k]) is not int or features[k]<0 for k in FEATURES-
                {'owner','public_input_id','semantic_phase','terminal_kind'})):
        raise ValueError('Explicit local phase/shape/template/covariate features required')
    return json.dumps(features,sort_keys=True,separators=(',',':'))


def mark(event,origin,ident,semantic,source_macro_id,occurrence_macro_id):
    names=('perf_ns','received_time_ns','collector_dequeued_time_ns','collector_processing_time_ns')
    times=[event[k] for k in names]
    if any(type(x) is not int for x in times) or times!=sorted(times) or times[0]<origin:
        raise ValueError('Common monotone occurrence/reader/dequeue/processing clocks required')
    payload=dict(semantic_kind=semantic,observed_phase_id=event.get('phase_id',event.get('segment_id')),
        source_macro_id=source_macro_id,occurrence_macro_id=occurrence_macro_id)
    if event.get('kind')=='device_event_query':
        payload.update(end_event_complete=event['end_event_complete'],
            evidence_scope=event['evidence_scope'],all_request_GPU_work_complete=False,
            physical_SM_service_ns=None,
            host_query_begin_offset_ns=event['host_query_begin_ns']-origin,
            host_query_end_offset_ns=event['host_query_end_ns']-origin,
            source_stream_identity_is_runtime_stream_identity=False)
    return dict(mark_id=ident,occurrence_offset_ns=times[0]-origin,
        reader_delay_ns=times[1]-times[0],dequeue_extra_ns=times[2]-times[1],
        processing_extra_ns=times[3]-times[2],payload=payload)


def _fit_verified_document(macros,closure):
    """Empirical joint measure within observed local feature cells only.

    Peer features at actual BEGIN and future terminal kind are *training label*
    conditioning variables. Their prospective qualifications are unresolved.
    """
    from .stage_response_features_v1 import training_roles
    if (macros.get('status')!='DISJOINT_TRAINING_RESPONSE_MACROS_NOT_FITTED_PROVIDER'
            or macros.get('mode')!='natural' or macros.get('main_queue_provider_eligible') is not False):
        raise ValueError('Reviewed natural training macro03 diagnostics required')
    review=checked(macros['sources'][0]);raw_ref=review['inputs']['raw.json'];raw=checked(raw_ref)
    training_roles(raw) # includes exact complete training source/review checks
    rows={r['request_id']:r for r in raw['request_records']}
    cells={};provenance=[]
    for request in macros['requests']:
        row=rows[request['request_id']]
        forwards=[m for m in request['macros'] if m['kind']=='language_forward_macro']
        if not forwards:continue
        prompt=forwards[0]['forward_shape']['input_sequence_length']
        for index,macro in enumerate(forwards):
            terminal_kind='native_END' if index==len(forwards)-1 else 'next_forward_BEGIN'
            state=macro['causal_covariate_path'][0]['state']
            features=dict(owner=macro['owner'],public_input_id=macro['pool_id'],semantic_phase=macro['semantic_phase'],
                prompt_sequence_length=prompt,input_batch_size=macro['forward_shape']['input_batch_size'],
                forward_index=macro['forward_shape']['forward_index'],terminal_kind=terminal_kind,
                peer_macro_count_at_actual_BEGIN=state['peer_visible_macro_owner_count'],
                peer_same_phase_count_at_actual_BEGIN=state['peer_same_visible_macro_phase_owner_count'])
            key=canonical(features);marks=[];query_count=0
            tail=next((m for m in request['macros'] if m['kind']=='host_ready_response'),None)
            milestones=list(macro['internal_training_milestones'])
            if terminal_kind=='native_END':
                if tail is None:raise ValueError('Final macro must retain complete native END to READY tail')
                milestones.extend(tail['internal_training_milestones'])
            for ordinal,milestone in enumerate(milestones):
                e=milestone['raw_event']
                occurrence_macro=macro['macro_id'] if e['perf_ns']<macro['right_ns'] else 'native_end_to_worker_ready'
                if e.get('event')=='start' and e.get('segment_id')==macro['macro_id']:
                    ident,semantic='BEGIN','forward_BEGIN'
                elif e.get('event')=='end' and e.get('segment_id')==macro['macro_id']:
                    ident,semantic='HOST_END','internal_host_END'
                elif e.get('kind')=='device_event_query':
                    ident,semantic='QUERY:'+str(query_count),'recorded_stream_query';query_count+=1
                elif occurrence_macro=='native_end_to_worker_ready' and e['perf_ns']>macro['right_ns']:
                    ident,semantic='TAIL:'+str(ordinal),'ready_tail_host_milestone'
                else:continue
                marks.append(mark(e,macro['left_ns'],ident,semantic,
                    e.get('phase_id',e.get('segment_id')),occurrence_macro))
            terminal=[e for e in row['execution_events'] if e['perf_ns']==macro['right_ns']
                and ((terminal_kind=='native_END' and e.get('event')=='end' and e.get('segment_id')=='native_inference')
                    or (terminal_kind=='next_forward_BEGIN' and e.get('event')=='start'
                        and e.get('semantic_phase') in ('prefill','decode')))]
            if len(terminal)!=1:raise ValueError('Unique macro terminal event required')
            marks.append(mark(terminal[0],macro['left_ns'],'TERMINAL',terminal_kind,
                macro['macro_id'],macro['macro_id']))
            tail_ns=0
            if terminal_kind=='native_END':
                tail_ns=tail['response_wall_ns']
                result_event=dict(perf_ns=row['worker_response']['end_perf_ns'],
                    received_time_ns=row['response_reader_event_enqueue_time_ns'],
                    collector_dequeued_time_ns=row['response_collector_dequeued_time_ns'],
                    collector_processing_time_ns=row['response_collector_processing_time_ns'])
                marks.append(mark(result_event,macro['left_ns'],'RESULT','RESULT',
                    'native_end_to_worker_ready','native_end_to_worker_ready'))
                marks.append(mark(row['previous_handler_next_request_ready'],macro['left_ns'],
                    'READY','WORKER_READY','native_end_to_worker_ready','native_end_to_worker_ready'))
            marks.sort(key=lambda m:m['occurrence_offset_ns'])
            if sum(m['mark_id']=='BEGIN' for m in marks)!=1 or sum(m['mark_id']=='HOST_END' for m in marks)!=1:
                raise ValueError('One paired local BEGIN and host END required')
            for m in marks:
                tail_mark=m['payload']['occurrence_macro_id']=='native_end_to_worker_ready'
                origin=macro['response_wall_ns'] if tail_mark else 0
                denominator=tail_ns if tail_mark else macro['response_wall_ns']
                m['occurrence_response_clock_coordinate']=dict(
                    macro_role='ready_tail' if tail_mark else 'forward',
                    numerator=m['occurrence_offset_ns']-origin,denominator=denominator)
                if m['payload']['semantic_kind']=='recorded_stream_query':
                    m['payload']['policy_site']=('after_host_END_poll' if m['mark_id']=='QUERY:0' else
                        'observer_context_exit_poll' if tail_mark else 'next_forward_pre_hook_poll')
            atom=dict(response_ns=macro['response_wall_ns'],following_ready_tail_ns=tail_ns,marks=marks,
                response_domain='natural_forward_and_async_tail_wall',
                loss_model_qualified=False,response_is_effective_SM_service=False)
            # Atom identity hashes local outcome, never request/cohort provenance.
            atom['atom_id']=hashlib.sha256(json.dumps(atom,sort_keys=True).encode()).hexdigest()
            cell=cells.setdefault(key,dict(features=features,atoms=[]));cell['atoms'].append(atom)
            provenance.append(dict(atom_id=atom['atom_id'],capture_sha256=raw_ref['sha256'],
                request_id=row['request_id'],macro_id=macro['macro_id']))
    return dict(schema=SCHEMA,status='TRAINING_EMPIRICAL_JOINT_ATOMS_ONLY',
        cells=list(cells.values()),provenance_not_used_for_generation=provenance,
        sources=closure,
        clock='host_perf_counter_ns',visibility_endpoint='collector_processing_time_ns',
        fitted_empirical_measure=True,probability_calibrated=False,
        prospective_conditioning_features_qualified=False,
        dynamic_peer_path_conditioned=False,cross_macro_transition_qualified=False,
        producer_queue_or_loss_mechanism_identified=False,native_adapter_eligible=False,
        main_queue_provider_eligible=False,validation_labels_read=False,GPU_started=False)


def fit_training_atoms(macro_paths):
    """Rederive complete training closure; pool exact-content local cells.

    Capture IDs remain provenance/leaveout units, never lookup features.
    """
    from .macro_joint_response_fit_v1 import verified_document
    paths=[macro_paths] if isinstance(macro_paths,(str,Path)) else list(macro_paths)
    if not paths:raise ValueError('At least one whole reviewed training capture required')
    documents=[];fingerprints=[]
    for path in paths:
        doc,closure=verified_document(path)
        review=checked(doc['sources'][0]);context=checked(review['inputs']['context.json'])
        fingerprints.append(sorted({x['frozen']['sha256'] for x in context['execution_sources']}))
        documents.append(_fit_verified_document(doc,closure))
    if any(f!=fingerprints[0] for f in fingerprints):
        raise ValueError('Cross-capture execution source identities differ')
    if not documents:raise ValueError('At least one whole reviewed training capture required')
    combined=copy.deepcopy(documents[0]);merged={}
    provenance=[];sources=[]
    for doc in documents:
        for cell in doc['cells']:
            merged.setdefault(canonical(cell['features']),dict(features=cell['features'],atoms=[]))['atoms'].extend(cell['atoms'])
        provenance.extend(doc['provenance_not_used_for_generation']);sources.extend(doc['sources'])
    captures={x['capture_sha256'] for x in provenance}
    if len(captures)!=len(documents):raise ValueError('Distinct whole captures required; duplicate inputs forbidden')
    combined.update(cells=list(merged.values()),provenance_not_used_for_generation=provenance,
        sources=list({r['path']:r for r in [*sources,receipt(__file__)]}.values()),
        capture_count=len(captures),training_captures=sorted(captures),
        execution_source_sha256_set=fingerprints[0],
        exact_public_input_identity_conditioned=True,final_forward_and_READY_tail_jointly_retained=True,
        independent_macro_draws_are_unvalidated_assumption=True,
        arbitrary_response_rescaling_implemented=False)
    return combined


def select_atoms(law,features,*,response_ns=None):
    key=canonical(features)
    if law.get('schema')!=SCHEMA:raise ValueError('Training candidate schema required')
    found=[c for c in law['cells'] if canonical(c['features'])==key]
    atoms=[] if not found else found[0]['atoms']
    if response_ns is not None:
        if type(response_ns) is not int or response_ns<0:raise ValueError('Exact discrete response required')
        atoms=[a for a in atoms if a['response_ns']==response_ns]
    return copy.deepcopy(atoms)


def realize(atom,begin_ns):
    if type(begin_ns) is not int or begin_ns<0:raise ValueError('Integer BEGIN clock required')
    notices=[]
    for m in atom['marks']:
        occurrence=begin_ns+m['occurrence_offset_ns'];reader=occurrence+m['reader_delay_ns']
        dequeue=reader+m['dequeue_extra_ns'];processed=dequeue+m['processing_extra_ns']
        notice=dict(mark_id=m['mark_id'],occurrence_ns=occurrence,reader_receipt_ns=reader,
            dequeued_ns=dequeue,processed_ns=processed,payload=copy.deepcopy(m['payload']))
        if 'occurrence_response_clock_coordinate' in m:
            notice['occurrence_response_clock_coordinate']=copy.deepcopy(m['occurrence_response_clock_coordinate'])
        notices.append(notice)
    if any(any(a[c]>b[c] for c in CLOCKS) for a,b in zip(notices,notices[1:])):
        raise ValueError('Atom violates observed local producer FIFO')
    return dict(response_END_ns=begin_ns+atom['response_ns'],
        following_ready_tail_ns=atom.get('following_ready_tail_ns',0),
        ready_tail_END_ns=None if not atom.get('following_ready_tail_ns') else
            begin_ns+atom['response_ns']+atom['following_ready_tail_ns'],notifications=notices,
        latent_response_completion_is_observed=False,query_true_ends_macro=False,
        all_request_GPU_work_complete=None,physical_SM_service_ns=None)


def draw(law,features,*,begin_ns,quantile,response_ns=None,prior_notification_watermarks=None):
    if type(quantile) not in (int,float) or not math.isfinite(quantile) or not 0<=quantile<1:
        raise ValueError('Explicit finite quantile in [0,1) required')
    atoms=select_atoms(law,features,response_ns=response_ns)
    if not atoms:return dict(status='UNKNOWN_NO_EMPIRICAL_LOCAL_SUPPORT',main_queue_provider_eligible=False)
    atom=atoms[min(len(atoms)-1,int(quantile*len(atoms)))];result=realize(atom,begin_ns)
    if prior_notification_watermarks is not None:
        if set(prior_notification_watermarks)!=set(CLOCKS[1:]):raise ValueError('Three explicit prior FIFO clocks required')
        first=result['notifications'][0]
        if any(first[k]<prior_notification_watermarks[k] for k in CLOCKS[1:]):
            return dict(status='UNKNOWN_CROSS_MACRO_FIFO_CONFLICT',main_queue_provider_eligible=False)
    return dict(result,status='TRAINING_EMPIRICAL_LOCAL_DRAW_ONLY',atom_id=atom['atom_id'],
        response_conditioning='unconditional_local_cell' if response_ns is None else 'exact_discrete_response_atom',
        cross_macro_transition_qualified=False,main_queue_provider_eligible=False)


def condition_atoms(atoms,observations,*,begin_ns,cutoff_ns,registry_complete_through_ns):
    """Finite exact reference arithmetic; no jitter/kernel bandwidth is invented."""
    if (type(cutoff_ns) is not int or cutoff_ns<0 or type(registry_complete_through_ns) is not int
            or registry_complete_through_ns<cutoff_ns):
        return dict(status='UNKNOWN_INCOMPLETE_PROCESSING_REGISTRY',main_queue_provider_eligible=False)
    if any(e['processed_ns']>cutoff_ns for e in observations):raise ValueError('Only already-processed observations allowed')
    if len({e['mark_id'] for e in observations})!=len(observations):raise ValueError('Unique visible marks required')
    observations=sorted(observations,key=lambda e:e['mark_id']);selected=[]
    for atom in atoms:
        world=realize(atom,begin_ns)
        visible=sorted([e for e in world['notifications'] if e['processed_ns']<=cutoff_ns],key=lambda e:e['mark_id'])
        if visible!=observations:continue
        selected.append(dict(atom_id=atom['atom_id'],response_completed=world['response_END_ns']<=cutoff_ns,
            remaining_response_ns=max(0,world['response_END_ns']-cutoff_ns),
            pending_notifications=[e for e in world['notifications'] if e['processed_ns']>cutoff_ns]))
    if not selected:return dict(status='UNKNOWN_ZERO_EXACT_JOINT_SUPPORT',main_queue_provider_eligible=False)
    for world in selected:world['conditional_mass']=1/len(selected)
    return dict(status='TRAINING_FINITE_JOINT_POSTERIOR_ONLY',worlds=selected,
        completed_response_mass=sum(w['conditional_mass'] for w in selected if w['response_completed']),
        notification_absence_implies_unfinished=False,probability_calibrated=False,
        main_queue_provider_eligible=False,physical_SM_service_ns=None,all_request_GPU_work_complete=None)
