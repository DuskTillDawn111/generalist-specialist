"""Exact-source CPU join of audited occupancy groups to canonical kernels."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .native_collect import receipt
from .kernel_sequence_audit_v2 import verify
from .occupancy_declaration_v2 import read_sqlite_launches
from .trace_export import linux_pid


def signature(row):
    return dict(kernel={k:v for k,v in row['kernel'].items() if k!='static_evidence'},
                mode=row['function_attributes']['shmem_limit_config'],
                effective_shared_bytes=row['effective_shared_bytes'],
                completed_launch=bool(row.get('completed_launch_evidence')))


def key(source, launch):
    return (source['sha256'],launch['start'],launch['end'],linux_pid(launch['global_pid']),
            launch['context_id'],launch['correlation_id'])


class AuditedOccupancyResolver:
    def __init__(self,audit_path):
        audit_path=Path(audit_path).resolve(); audit=json.loads(audit_path.read_text())
        if audit.get('schema')!='gsdeploy.arch_aware_occupancy_declaration_audit.v2':
            raise ValueError('Architecture declaration audit required')
        verify(audit['hardware_source'])
        for name in ('header','binary'):
            verify(audit['build'][name])
        for source in audit['implementation_sources']:
            verify(source)
        audit_ref=receipt(audit_path)
        self.hardware=copy.deepcopy(audit['hardware'])
        self.source_receipts=[audit_ref,audit['hardware_source'],*audit['implementation_sources']]
        self.index={}
        for trace in audit['traces']:
            if not trace.get('unchanged') or trace.get('source')!=trace.get('source_after'):
                raise ValueError('Only unchanged audited exact SQLite sources supported')
            verify(trace['source'])
            read=read_sqlite_launches(trace['source']['path'],self.hardware)
            if not read['known'] or read['source']!=trace['source'] or read['uuid_binding']!='UNIQUE_EXACT_UUID':
                raise ValueError('Occupancy audit is not bound to unique actual hardware/source')
            rows={r['launch_identity']['sqlite_rowid']:r for r in read['rows']}; seen=set()
            for group in trace['groups']:
                representative=rows.get(group['representative_launch']['sqlite_rowid'])
                if representative is None or representative['launch_identity']!=group['representative_launch']:
                    raise ValueError('Occupancy representative launch mismatch')
                resolution=group['resolution']
                if resolution.get('known'):
                    calculation=resolution['calculation']
                    if (calculation.get('assumed_fields') or calculation['declaration']['kernel']!=representative['kernel']
                            or calculation['declaration']['hardware']!=self.hardware):
                        raise ValueError('Occupancy declaration has changed kernel/hardware or unobserved assumptions')
                for rowid in group['member_sqlite_rowids']:
                    row=rows.get(rowid)
                    if row is None or rowid in seen or signature(row)!=signature(representative):
                        raise ValueError('Occupancy group membership/signature is not exact')
                    seen.add(rowid)
                    launch=row['launch_identity']; lookup=key(read['source'],launch)
                    if lookup in self.index:
                        raise ValueError('Ambiguous canonical occupancy launch join')
                    patched=None
                    if resolution.get('known'):
                        patched=copy.deepcopy(row['kernel']); patched.update(calculation['kernel_patch'])
                        patched['allocation_evidence']=dict(audit=audit_ref,
                            calculation=calculation['kernel_patch']['allocation_evidence'],
                            exact_member_launch=launch,representative_launch=group['representative_launch'],
                            signature_equivalence_checked=True,source=read['source'])
                    self.index[lookup]=patched
            if seen!=set(rows):
                raise ValueError('Occupancy groups omit launches from denominator')
            if receipt(read['source']['path'])!=read['source']:
                raise ValueError('SQLite changed during occupancy join')

    def __call__(self,kernel):
        lookup=(kernel['source_sqlite']['sha256'],kernel['nsys_start_ns'],kernel['nsys_end_ns'],
                kernel['pid'],kernel['context_id'],kernel['correlation_id'])
        if lookup not in self.index:
            raise ValueError('Exact canonical launch absent from occupancy source audit')
        return copy.deepcopy(self.index[lookup])
