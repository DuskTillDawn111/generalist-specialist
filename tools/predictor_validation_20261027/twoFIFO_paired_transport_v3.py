"""Typed observed sender and paired feasible worker transport for v5_2."""
from dataclasses import dataclass
from fractions import Fraction

from gsdeploy.forecast_errors import UnknownEvidence
from . import twoFIFO_transport_partial_identification_v2 as partial


@dataclass(frozen=True)
class PairedTransportConstraints:
    enqueue_after_arrival_ns: Fraction
    sender_turn_extra_ns: Fraction
    submit_to_write_record_ns: Fraction
    observed_submit_to_START_ns: Fraction
    previous_READY_minus_submit_ns: Fraction


class UnknownTransportInterval(UnknownEvidence):
    def __init__(self,evidence):
        self.evidence=evidence
        super().__init__('UNKNOWN_PAIRED_TRANSPORT_START_INTERVAL:'+repr(evidence['START_ns_bounds']))


def source_constraints(draw):
    if type(draw) is not PairedTransportConstraints:
        raise ValueError('Exact typed paired transport constraints required')
    for key in ('enqueue_after_arrival_ns','sender_turn_extra_ns','submit_to_write_record_ns','observed_submit_to_START_ns'):
        partial.ns(getattr(draw,key))
    q=draw.previous_READY_minus_submit_ns
    partial.require(type(q) in (int,Fraction),'Exact signed READY offset required')
    anchor=max(Fraction(0),-q)
    return partial.identify_source(anchor,anchor+draw.submit_to_write_record_ns,
        anchor+draw.observed_submit_to_START_ns,anchor+q)


def sender_schedule(arrival_ns,previous_write_record_ns,draw):
    source=source_constraints(draw);arrival=partial.ns(arrival_ns);previous=partial.ns(previous_write_record_ns)
    enqueue=arrival+draw.enqueue_after_arrival_ns;submit=max(enqueue,previous)+draw.sender_turn_extra_ns
    return dict(enqueued_ns=enqueue,parent_submit_ns=submit,parent_write_record_ns=submit+draw.submit_to_write_record_ns,
        stdin_line_available_ns=None,stdin_line_availability_known=False,
        paired_source_constraints=source,latent_line_or_continuation_point_selected=False)


def worker_START(previous_READY_ns,send,*,request_id,public_key,public_key_ordinal,offered_request_ids):
    result=partial.project_target(send['paired_source_constraints'],send['parent_submit_ns'],previous_READY_ns)
    result.update(request_id=request_id,public_key=list(public_key),public_key_arrival_ordinal=public_key_ordinal,
        offered_request_ids=list(offered_request_ids),offered_requests=len(offered_request_ids),
        missing_predictions_retained_as_UNKNOWN=True)
    if result['identified_START_ns'] is None:raise UnknownTransportInterval(result)
    return result['identified_START_ns'],result
