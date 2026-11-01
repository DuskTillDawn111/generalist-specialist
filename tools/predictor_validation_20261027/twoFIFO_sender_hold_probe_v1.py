"""Explicit host-timing intervention for future mechanism probes, never native fit.

No runtime is started here. A future owned collector may wrap an already primed
transport; send/flush and worker FIFO code are delegated unchanged after the
declared hold. Tests inject a clock and an in-memory transport only.
"""
import argparse
import json
from pathlib import Path

from .twoFIFO_training_manifest_v1 import digest,require
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .stage_response_features_v1 import receipt


class SenderHoldProbe:
    def __init__(self,transport,owner,holds_by_owner_ordinal,*,clock_ns,wait_until_ns,events):
        require(all(type(k) is int and k>=0 and type(v) is int and v>=0 for k,v in holds_by_owner_ordinal.items()),
                'Explicit nonnegative measured-owner ordinal hold schedule required')
        self.transport=transport;self.owner=owner;self.holds=dict(holds_by_owner_ordinal)
        self.clock=clock_ns;self.wait=wait_until_ns;self.events=events;self.ordinal=0

    def __getattr__(self,key):return getattr(self.transport,key)

    def send(self,payload):
        require(isinstance(payload,dict) and payload.get('request_id') and payload.get('task_type')!='__profile_start__',
                'Install probe after priming/control; complete measured payload required')
        ordinal=self.ordinal;self.ordinal+=1;hold=self.holds.get(ordinal,0)
        begin=self.clock();self.wait(begin+hold);end=self.clock()
        self.events.append(dict(owner=self.owner,request_id=payload['request_id'],owner_ordinal=ordinal,
            intervention='explicit_sender_pre_serialization_hold',requested_hold_ns=hold,hold_begin_ns=begin,hold_end_ns=end,
            natural_transport_fit_eligible=False))
        return self.transport.send(payload)


def prepare(base_manifest_path,output):
    base_ref=receipt(base_manifest_path);base=json.loads(Path(base_manifest_path).read_text())
    short=base['inputs']['text']['short']['pool_id'];cases=[]
    for block in base['capture_blocks']:
        for owner in ('text#0','generalist_2#0'):
            rows=[dict(request_id=block['capture_id']+':sender-idle-worker-probe:'+owner+':'+str(i),
                owner=owner,task='text',pool_id=short,arrival_ns=i*1_000_000,arrival_index=i) for i in range(3)]
            cases.append(dict(capture_id=block['capture_id'],case_id='sender-held-worker-idle-'+owner,
                arrivals=rows,offered=len(rows),public_arrivals_sha256=digest(rows),owner=owner,
                sender_hold_ns_by_measured_owner_ordinal={'0':20_000_000},
                intended_stratum='sender_senderFIFO_busy_while_worker_waits_for_first_input',
                initial_state_required='Original owner loaded/primed, previous actual READY and flush/read markers bound; no earlier pending requests.',
                achieved_stratum_requires='stdin_read_enter <= hold_begin < hold_end <= stdin_line_read_return; otherwise UNKNOWN',
                natural_response_or_transport_fit_eligible=False))
    sources=strict_references([base['sources'],[base_ref],implementation_sources(__file__)])
    require(all(receipt(r['path'])==r for r in sources),'Base design source changed')
    result=dict(schema='gsdeploy.original_twoFIFO_controlled_sender_probe_design.v1',sources=sources,
        base_manifest=base_ref,measurement_ns=base['measurement_ns'],drain_ns=base['drain_ns'],cases=cases,
        mechanism_probe_only=True,natural_training_rows=0,planned_probe_offered=sum(c['offered'] for c in cases),
        pre_serialization_hold_changes_timing_law=True,source_FIFO_topology_preserved=True,
        worker_or_collector_source_modified=False,actual_stratum_achieved='UNKNOWN_NOT_COLLECTED',
        required_new_worker_markers=['stdin_read_enter_ns','stdin_line_read_return_ns','READY_flush_return_ns'],
        sample_retention='Every offered probe row retained; no substitution into natural training fit.',
        hardware_counters_required=False,validation_labels_read=False,GPU_started=False,main_queue_provider_eligible=False)
    with Path(output).open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    return receipt(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base-manifest',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.base_manifest,args.output)))
