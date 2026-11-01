"""Audit closed native receipts sequentially; no writes to experiment evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools import queue_five_term_audit_20261023 as audit


def write_once(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','execution','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prior-audit',type=Path)
    p.add_argument('--poll-seconds',type=float,default=5.)
    p.add_argument('--maximum-seconds',type=float,default=14400.)
    a=p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite '+str(a.output))
    if not 1 <= a.poll_seconds <= 60 or a.maximum_seconds <= 0:
        raise SystemExit('Poll interval must be1..60 seconds and timeout positive')
    packet=json.loads(a.packet.read_bytes());packet_ref=audit.sha(a.packet)
    source=audit.sha(audit.__file__);watcher=audit.sha(__file__)
    binding=audit.bindings(packet)
    if not binding['verified']:
        raise SystemExit('Source/input bindings changed: '+repr(binding['errors']))
    prior_refs=[];observations={}
    if a.prior_audit:
        prior=json.loads(a.prior_audit.read_bytes())
        if prior['packet']!=packet_ref or prior['audit_source']!=source or not prior['source_input_bindings']['verified']:
            raise SystemExit('Prior audit packet/source/binding identity mismatch')
        prior_refs.append(audit.sha(a.prior_audit))
        for row in prior['observations']:
            if row.get('status')=='AUDITED' and row.get('verified') and audit.sha(row['raw']['path'])==row['raw']:
                observations[row['label']]=row
    a.output.mkdir(parents=True)
    started=time.monotonic();snapshots=[]

    def snapshot(status):
        rows=[observations[j['label']] for j in packet['jobs'] if j['label'] in observations]
        summary=audit.aggregate(packet,rows)
        report=dict(schema='gsdeploy.five_term_external_stream_audit.v1',status=status,
                    packet=packet_ref,audit_source=source,watcher_source=watcher,
                    prior_audit_receipts=prior_refs,source_input_bindings=binding,
                    observations=rows,summary=summary,complete=summary['all_expected_jobs_terminal'],
                    verified=summary['all_expected_jobs_terminal'] and all(r.get('verified') for r in rows),
                    GPU_started=False,frozen_sources_modified=False)
        index=len(snapshots)+1;name=f'audit-snapshot-{index:02}.json'
        write_once(a.output/name,report);snapshots.append(audit.sha(a.output/name))
        progress=dict(status=status,completed_audits=len(rows),expected_jobs=len(packet['jobs']),
                      latest_snapshot=snapshots[-1],snapshots=snapshots,elapsed_seconds=time.monotonic()-started)
        temp=a.output/'progress.json.tmp';temp.write_text(json.dumps(progress,indent=2,ensure_ascii=False)+'\n')
        temp.replace(a.output/'progress.json')
        print(json.dumps(dict(status=status,audited=len(rows),summary=summary),ensure_ascii=False),flush=True)
        return report

    snapshot('WATCHING')
    while True:
        if audit.sha(a.packet)!=packet_ref or audit.sha(audit.__file__)!=source or audit.sha(__file__)!=watcher:
            snapshot('AUDIT_SOURCE_OR_PACKET_CHANGED_STOPPED');return
        for job in packet['jobs']:
            if job['label'] in observations:
                continue
            marker=a.execution/(job['label']+'-interim-analysis.json')
            if not marker.exists():
                continue
            result=audit.audit_run(packet,job,a.execution/job['label']/'run.json',terminal_marker=marker)
            if result['status']=='PENDING':
                continue
            observations[job['label']]=result
            write_once(a.output/(job['label']+'-audit.json'),dict(packet=packet_ref,audit_source=source,observation=result))
            snapshot('WATCHING')
        if len(observations)==len(packet['jobs']):
            final=snapshot('COMPLETED')
            write_once(a.output/'terminal.json',final)
            return
        terminal=a.execution/'terminal.json'
        if terminal.exists():
            final=snapshot('EXECUTION_TERMINAL_WITH_PARTIAL_EVIDENCE')
            write_once(a.output/'terminal.json',final)
            return
        if time.monotonic()-started>a.maximum_seconds:
            final=snapshot('WATCH_TIMEOUT_PARTIAL_EVIDENCE')
            write_once(a.output/'terminal.json',final)
            return
        time.sleep(a.poll_seconds)


if __name__=='__main__':
    main()
