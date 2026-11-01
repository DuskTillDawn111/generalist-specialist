"""Handoff labels with a prequeued-kernel witness, never a pure SM cost.

RESTORE_START is a lower boundary. A mapped, already-certified-ready kernel
starting in that envelope supplies an upper witness without waiting for future
host supply. The interval still contains dispatch uncertainty and is not an
exact physical service-ready timestamp or a counterfactual bound.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path

from .context_initial_transition_v13 import extract
from .multi_native_composite_v5 import checked
from .native_collect import receipt, save


def prequeued_witness(rows, starts, restore_ns, save_ns):
    if (type(restore_ns) is not int or type(save_ns) is not int or restore_ns >= save_ns
            or len(rows) != len(starts)):
        raise ValueError('Ordered envelope and complete sorted kernel index required')
    index = bisect_left(starts, restore_ns)
    while index < len(rows) and starts[index] < save_ns:
        row = rows[index]; ready = row['certified_ready_bound_ns']
        if (ready is not None and ready <= restore_ns
                and row.get('recorded_dependencies_known') is True):
            return row
        index += 1
    return None


def derive(supervision_path):
    source = receipt(supervision_path); supervision = checked(source)
    rebuilt = extract([r['report'] for r in supervision['training_audits']])
    if rebuilt != supervision:
        raise ValueError('Handoff supervision differs from complete source rederivation')
    labels = []; counts = Counter(); sources = [source, receipt(__file__)]
    for audit in supervision['training_audits']:
        report = checked(audit['report']); raw = checked(audit['raw'])
        request_rows = {r['request_id']:r for r in raw['request_records']}
        kernels = checked(report['kernel_ready_observations'])['rows']
        by = defaultdict(list)
        for row in kernels:
            owner = request_rows[row['request_id']]['model']
            by[owner].append(row)
        for rows in by.values():
            rows.sort(key=lambda r:(r['observed_start_ns'],r['observed_end_ns'],r['node_id']))
        starts = {owner:[r['observed_start_ns'] for r in rows] for owner,rows in by.items()}
        for transition in supervision['transition_observations']:
            if transition['capture'] != audit['raw']['sha256']:
                continue
            target = transition['next_owner']; restore = transition['next_restore_start_ns']
            witness = prequeued_witness(by.get(target,[]),starts.get(target,[]),restore,
                                        transition['next_envelope_save_end_ns'])
            # A witness may belong to any complete measured request of this
            # owner, but it must remain in the same declared job cohort.
            if witness is not None and request_rows[witness['request_id']]['profile_job_id'] != transition['job_id']:
                raise ValueError('A foreign job cannot identify this handoff witness')
            labels.append(dict(capture=transition['capture'],job_id=transition['job_id'],
                held_owner=transition['held_owner'],next_owner=target,
                context_age_at_save_ns=transition['context_age_at_save_ns'],
                pending_frontier=transition['pending_frontier'],
                save_end_ns=transition['save_end_ns'],next_restore_start_ns=restore,
                save_to_restore_gap_ns=transition['save_to_next_restore_ns'],
                service_after_restore_interval_ns=dict(lower_ns=0,
                    upper_ns=None if witness is None else witness['observed_start_ns']-restore),
                prequeued_kernel_witness=witness,
                missing_reason='NO_CERTIFIED_PREQUEUED_KERNEL_START_IN_NEXT_ENVELOPE' if witness is None else None,
                exact_service_ready_observed=False,physical_SM_cost_identified=False))
            counts['total_transitions'] += 1
            counts['prequeued_witness' if witness is not None else 'missing_prequeued_witness'] += 1
        sources.extend((report['kernel_ready_observations'],audit['raw'],audit['report']))
    if len(labels) != len(supervision['transition_observations']):
        raise ValueError('All original transitions, including missing witnesses, must be retained')
    for ref in sources:
        if receipt(ref['path']) != ref:
            raise ValueError('Handoff witness source changed during extraction')
    return dict(schema='gsdeploy.prequeued_context_handoff_witness.v14',sources=sources,
        labels=labels,counts=dict(counts),training_only=True,GPU_started=False,
        observations_are_exogenous_counterfactual_bounds=False,
        restore_is_exact_service_ready=False,host_supply_wait_fitted_as_switch_cost=False,
        independent_prediction_validation=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervision',required=True);parser.add_argument('--output',required=True)
    args = parser.parse_args();out = Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    result = derive(args.supervision);save(out/'labels.json',result)
    save(out/'report.json',dict(counts=result['counts'],labels=receipt(out/'labels.json'),
                              training_only=True,GPU_started=False))
    print(json.dumps(dict(counts=result['counts'],labels=receipt(out/'labels.json'))))


if __name__ == '__main__':main()
