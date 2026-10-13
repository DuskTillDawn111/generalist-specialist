"""Evaluation groups that keep identical planning inputs together across protocols."""
from .common import identity


def evaluation_configuration_id(data):
    return 'cfg_'+identity(dict(deployment={m:n for m,n in data['deployment'].items() if n},
        workload_rps={k:float(v) for k,v in data['workload_rps'].items() if v},
        routing_rps={k:{m:float(v) for m,v in row.items() if v} for k,row in data['routing_rps'].items() if any(row.values())}))


def protocol_id(data):
    return 'protocol_'+identity({k:data.get(k) for k in ('warmup_seconds','measurement_seconds',
        'drain_timeout_seconds','max_pending_per_model')} |
        {'arrival_process':'deterministic_periodic','source':data.get('source',{})})
