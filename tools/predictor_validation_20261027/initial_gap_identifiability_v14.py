"""Read-only original initial-gap evidence; no relabeling or fitted repair."""
from bisect import bisect_right
import argparse
import json
from pathlib import Path
import sqlite3

from .context_envelope_diagnostic_v8 import context_intervals
from .multi_native_composite_v5 import checked
from .native_collect import receipt,save


def audit(supervision_path,output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    supervision_ref=receipt(supervision_path);supervision=checked(supervision_ref)
    sources=[receipt(__file__),supervision_ref];rows=[]
    for initial in supervision['initial_observations']:
        if initial['held_owner'] is not None:continue
        training=next(r for r in supervision['training_audits'] if r['raw']['sha256']==initial['capture'])
        report=checked(training['report']);raw=checked(training['raw'])
        sql=next(r for r in report['sources'] if r['path'].endswith('.sqlite'))
        if receipt(sql['path'])!=sql:raise ValueError('Original SQL changed')
        sources.extend([training['report'],training['raw'],sql,report['kernel_ready_observations']])
        c=sqlite3.connect('file:'+sql['path']+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
        events,brackets,issues=context_intervals(c)
        owners={tuple(r['context']):r['alias'] for r in report['controlled_temporal_attribution']['creation_windows']}
        intervals=sorted((a,b,owners.get(k),k) for k,pieces in brackets.items() for a,b in pieces)
        clock=initial['clock_ns'];at=bisect_right([a for a,b,o,k in intervals],clock)
        previous=intervals[at-1] if at else None;following=intervals[at] if at<len(intervals) else None
        if previous is None or following is None:raise ValueError('Two-sided source envelope witnesses required')
        pa,pb,po,pk=previous;na,nb,no,nk=following
        covering=[list(k) for a,b,o,k in intervals if a<=clock<b]
        kernels_at=[dict(r) for r in c.execute('select globalPid,start,end,contextId,streamId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where start<=? and end>=?',(clock,clock))]
        next_kernels=[dict(r) for r in c.execute('select globalPid,start,end,contextId,streamId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<?',(na,nb))]
        c.close()
        source_initial=next(r for r in checked(report['kernel_ready_observations'])['rows'] if r['certified_ready_bound_ns']==clock)
        rows.append(dict(original_initial_observation=initial,source_context_count=len(events),
            controlled_mapped_context_count=len(owners),covering_paired_contexts=covering,
            previous_envelope=dict(restore_ns=pa,save_ns=pb,owner=po,opaque_context=list(pk)),
            next_envelope=dict(restore_ns=na,save_ns=nb,owner=no,opaque_context=list(nk)),
            elapsed_since_previous_SAVE_ns=clock-pb,residual_to_next_RESTORE_ns=na-clock,
            complete_SAVE_to_RESTORE_gap_ns=na-pb,
            unresolved_between_previous_restore_and_next_save=[r for r in issues if pa<=r['timestamp']<=nb],
            unresolved_within_10ms=[r for r in issues if abs(r['timestamp']-clock)<=10_000_000],
            all_observed_kernels_covering_query=kernels_at,
            all_observed_kernels_starting_in_next_envelope=next_kernels,
            earliest_certified_ready_row=source_initial,
            scheduler_state_partial_identification='OBSERVED_PAIRED_ENVELOPE_GAP_NOT_A_KNOWN_HELD_OWNER',
            physical_holder_or_service_identified=False,
            repair_recommendation='Retain original row and probability mass; add explicit scheduler-gap joint scenario with previous owner and gap age, remaining restore delay/target and interval-censored readiness. Future endpoints are training targets only, never current predictor features.'))
    for ref in sources:
        if receipt(ref['path'])!=ref:raise ValueError('Read-only gap evidence source changed')
    result=dict(status='INITIAL_UNKNOWN_GAP_PARTIAL_IDENTIFIABILITY_AUDIT_COMPLETE',sources=sources,
        unknown_initial_rows=len(rows),all_initial_rows=len(supervision['initial_observations']),rows=rows,
        original_rows_edited=False,unknown_mass_dropped=False,known_mass_renormalized=False,
        models_fitted=False,main_oracle_modified=False,validation_labels_used=False,GPU_started=False)
    save(out/'report.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--supervision',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=audit(a.supervision,a.output);print(json.dumps(dict(status=r['status'],unknown_initial_rows=r['unknown_initial_rows'])))
