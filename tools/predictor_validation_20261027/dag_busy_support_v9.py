"""Reconstruct joint host/reader scenarios from reviewed TRAINING native raw.

Loading reruns the strict capture review and the complete deterministic fit.
Changing emissions, start supply, priors or timing radius and resigning a JSON
manifest cannot qualify the change. CPU fixtures have an explicit separate gate.
These are empirical host/reader scenarios, not calibrated DAG error bounds.
"""
from __future__ import annotations

from collections import defaultdict
import copy
from fractions import Fraction
import json
from pathlib import Path
import tempfile

from gsdeploy.forecasting.observed_state import (
    feedback_from_runtime_records,new_observed_state,advance_observed_state,
)
from gsdeploy.resources import identity
from .dag_checkpoint_v9 import receipt

SCHEMA='gsdeploy.joint_dag_reader_scenario_support.v8'
FIXTURE_REVIEW='gsdeploy.cpu_fixture.queue_training_review.v9'
FIT_VERSION='source_rederived_joint_host_reader_whole_run_occurrences_v9'


def _key(row):
    return json.dumps([row['model'],row['task'],row['pool_id']],separators=(',',':'))


def _ns(value):
    if type(value) is not int or value<0:
        raise ValueError('Nonnegative native training timestamp or residual required')
    return value


def _read(ref):
    if receipt(ref['path'])!=ref:
        raise ValueError('Changed reviewed training source')
    return json.loads(Path(ref['path']).read_text())


def _reviewed(ref,allow_cpu_fixture):
    report=_read(ref)
    if report.get('status')!='PASS' or report.get('issues'):
        raise ValueError('Strict reviewed complete queue-training capture required')
    if report.get('schema')==FIXTURE_REVIEW:
        if not allow_cpu_fixture:
            raise ValueError('CPU fixture cannot qualify measured training support')
        fixture=True
    else:
        from .native_queue_training_review_v8 import SCHEMA as REVIEW,review
        if report.get('schema')!=REVIEW or report.get('training_only') is not True:
            raise ValueError('Independent native queue strict review required')
        plan_path=report['evidence']['plan']['path']
        run=Path(report['evidence']['raw']['path']).parent.parent
        with tempfile.TemporaryDirectory(prefix='dag-busy-review-') as temporary:
            recomputed=review(plan_path,run,Path(temporary)/'review')
        if recomputed!=report:
            raise ValueError('Training review differs from recomputed native capture evidence')
        fixture=False
    plan=_read(report['evidence']['plan']);raw=_read(report['evidence']['raw'])
    if (raw.get('split')!='training' or raw.get('training_only') is not True
            or raw.get('not_for_training',False) or plan.get('split')!='training'
            or plan.get('training_only') is not True or raw.get('kind')!=('cpu_fixture' if fixture else 'measured')
            or raw.get('execution_context_sha256')!=plan.get('execution_context_sha256')
            or raw.get('status')!='completed'):
        raise ValueError('Validation, foreign context or incomplete raw cannot enter fitting')
    if raw.get('plan')!=report['evidence']['plan']:
        raise ValueError('Native raw is not bound to the reviewed declared plan')
    for row in raw['request_records']:
        if row.get('split')!='training' or row.get('not_for_training',False):
            raise ValueError('Heldout request cannot enter host/reader fitting')
    return plan,raw,fixture


def extract_run(review_ref,*,allow_cpu_fixture=False):
    """Select timing only; output text/tokens/quality never leave extraction."""
    plan,raw,fixture=_reviewed(review_ref,allow_cpu_fixture)
    start=_ns(raw['phase_start_ns']);context=raw['execution_context_sha256']
    rows=raw['request_records'];trace=plan['trace']
    if (len(rows)!=len(trace) or len({r['request_id'] for r in rows})!=len(rows)
            or raw.get('run_id')!=plan.get('run_id') or not raw.get('run_id')):
        raise ValueError('Distinct original run and complete declared registry required')
    aliases=plan['aliases'];previous={a:start for a in aliases};counts=defaultdict(int)
    events=feedback_from_runtime_records(rows,context,max(
        _ns(row['previous_handler_next_request_ready']['received_time_ns']) for row in rows))
    now=max(e['received_time_ns'] for e in events)
    advance_observed_state(new_observed_state(aliases,context),now,feedback=events)
    records=[]
    for row,public in zip(rows,trace):
        alias=row['model'];pid=raw['ready'][alias]['pid']
        if (row['request_id']!=public['request_id'] or alias!=public['alias']
                or row['task']!=public['task'] or row['pool_id']!=public['pool_id']
                or row['scheduled_time_ns']!=start+public['offset_ns']):
            raise ValueError('Native training registry changed public route/input/arrival')
        begins=[e for e in row['execution_events'] if 'event' not in e]
        if len(begins)!=1 or begins[0].get('pid')!=pid:
            raise ValueError('Exactly one source-owned START required')
        begin=_ns(begins[0]['perf_ns']);end=_ns(row['worker_response']['end_perf_ns'])
        submit=_ns(row['actual_submit_time_ns']);arrival=_ns(row['scheduled_time_ns'])
        result=_ns(row['response_reader_event_enqueue_time_ns'])
        ready=row['previous_handler_next_request_ready'];worker_ready=_ns(ready['perf_ns'])
        ready_reader=_ns(ready['received_time_ns']);start_reader=_ns(begins[0]['received_time_ns'])
        eligible=max(arrival,previous[alias])
        if (not arrival<=submit<=begin<=end<=result<=ready_reader or begin<eligible
                or ready.get('pid')!=pid or ready['request_id']!=row['request_id']
                or not end<=worker_ready<=ready_reader or not begin<=start_reader<=result):
            raise ValueError('Invalid native queue/START/END/RESULT/READY chronology')
        phases=[]
        for e in row['execution_events']:
            if 'event' not in e:continue
            perf=_ns(e['perf_ns']);reader=_ns(e['received_time_ns'])
            if not begin<=perf<=end or not perf<=reader<=result:
                raise ValueError('Phase outside its actual host handler/reader lifecycle')
            phases.append(dict(segment_id=e['segment_id'],kind=e['kind'],event=e['event'],
                perf_offset_ns=perf-begin,reader_delay_ns=reader-perf))
        k=_key(row);occurrence=counts[k];counts[k]+=1
        records.append(dict(input_key=k,occurrence=occurrence,request_id=row['request_id'],
            owner_eligible_to_start_ns=begin-eligible,
            submit_after_public_arrival_ns=submit-arrival,
            observed_handler_wall_ns=end-begin,
            lifecycle=dict(parent_receipt_ms=(result-end)/1e6,worker_next_ready_ms=(worker_ready-end)/1e6),
            emissions=dict(start_reader_delay_ns=start_reader-begin,ready_reader_delay_ns=ready_reader-worker_ready,phases=phases)))
        previous[alias]=worker_ready
    shape=[{k:r[k] for k in ('offset_ns','alias','task','pool_id')} for r in trace]
    return dict(run_id=raw['run_id'],raw=copy.deepcopy(_read(review_ref)['evidence']['raw']),
        review=copy.deepcopy(review_ref),execution_context_sha256=context,fixture_only=fixture,
        trace_signature=identity(shape),public_trace_shape=shape,records=records)


def _channels(record):
    values={'start_lag':record['owner_eligible_to_start_ns'],
        'handler_wall_diagnostic':record['observed_handler_wall_ns'],
        'result_after_end':round(record['lifecycle']['parent_receipt_ms']*1e6),
        'ready_after_end':round(record['lifecycle']['worker_next_ready_ms']*1e6),
        'start_reader':record['emissions']['start_reader_delay_ns'],
        'ready_reader':record['emissions']['ready_reader_delay_ns']}
    for phase in record['emissions']['phases']:
        k=json.dumps([phase['segment_id'],phase['kind'],phase['event']],separators=(',',':'))
        values['phase_worker:'+k]=phase['perf_offset_ns']
        values['phase_reader:'+k]=phase['reader_delay_ns']
    return values


def fit_support(review_refs,*,allow_cpu_fixture=False,minimum_independent_runs=2):
    """Leave one entire run out when deriving empirical timing support errors.

    Node durations/generation work are not inferred from these mixed wall
    vectors. A separate trained DAG provider and context qualification remain
    necessary. The support radius is a training diagnostic, not a risk bound.
    """
    if type(minimum_independent_runs) is not int or minimum_independent_runs<2 or not review_refs:
        raise ValueError('At least two independent training runs are required')
    runs=[extract_run(ref,allow_cpu_fixture=allow_cpu_fixture) for ref in review_refs]
    if (len({r['run_id'] for r in runs})!=len(runs) or len({r['raw']['sha256'] for r in runs})!=len(runs)
            or len({r['execution_context_sha256'] for r in runs})!=1 or len({r['fixture_only'] for r in runs})!=1):
        raise ValueError('Distinct independent raw runs in one exact context/domain required')
    population=defaultdict(list)
    for run in runs:
        for record in run['records']:
            population[record['input_key'],record['occurrence']].append((run['run_id'],record))
    errors=[];missing=[]
    for run in runs:
        for target in run['records']:
            peers=[row for other,row in population[target['input_key'],target['occurrence']] if other!=run['run_id']]
            actual=_channels(target);compatible=[_channels(r) for r in peers if set(_channels(r))==set(actual)]
            if not compatible:
                missing.append(dict(run_id=run['run_id'],input_key=target['input_key'],occurrence=target['occurrence']));continue
            channel_errors={k:abs(actual[k]-round(sum(Fraction(p[k]) for p in compatible)/len(compatible))) for k in actual}
            errors.append(dict(heldout_run_id=run['run_id'],input_key=target['input_key'],occurrence=target['occurrence'],
                training_run_ids=sorted(other for other,row in population[target['input_key'],target['occurrence']]
                    if other!=run['run_id'] and set(_channels(row))==set(actual)),absolute_channel_errors_ns=channel_errors))
    scenarios=[]
    for run in runs:
        emissions=defaultdict(list);starts=defaultdict(list);lifecycle=defaultdict(list)
        for row in run['records']:
            k=row['input_key'];emissions[k].append(row['emissions']);starts[k].append(row['owner_eligible_to_start_ns']);lifecycle[k].append(row['lifecycle'])
        scenarios.append(dict(id=run['run_id'],training_run_ids=[run['run_id']],prior_mass=1,
            emissions=dict(emissions),request_start_delay_ns_by_input=dict(starts),lifecycle_delays_by_input=dict(lifecycle)))
    complete=len(runs)>=minimum_independent_runs and not missing
    result=dict(schema=SCHEMA,split='training',execution_context_sha256=runs[0]['execution_context_sha256'],
        sources=[r['raw'] for r in runs],generation_work_scope='training_only_public_input_conditioned',
        joint_scope='whole_run_node_host_lifecycle_reader_emission',timing_radius_basis='cross_fitted_training_run_support',
        observation_tolerance_ns=max((max(e['absolute_channel_errors_ns'].values()) for e in errors),default=0),
        minimum_independent_survivor_runs=minimum_independent_runs,scenarios=scenarios,
        fit_contract=dict(version=FIT_VERSION,review_refs=copy.deepcopy(review_refs),minimum_independent_runs=minimum_independent_runs,
            fitter_source=receipt(__file__),fixture_only=runs[0]['fixture_only']),
        fit_status='FITTED_EMPIRICAL_HOST_READER_JOINT_SCENARIOS' if complete else 'UNKNOWN_INDEPENDENT_RUN_OR_LOO_SUPPORT_INSUFFICIENT',
        fixture_only=runs[0]['fixture_only'],reviewed_training_runs=len(runs),
        supported_trace_signatures=sorted({r['trace_signature'] for r in runs}),
        training_timing_records=runs,leave_one_run_out_errors=errors,missing_leave_one_run_out_support=missing,
        request_start_definition='START - max(original public arrival, preceding same-owner worker READY); queue waiting is not an exogenous dispatch lag',
        occurrence_definition='Each run is one joint draw; same public alias/task/input occurrences retain original whole-run order',
        timing_radius_scope='Empirical LOO host/reader/phase-relative and handler-wall diagnostic error maximum; not calibrated absolute DAG rollout error, tail probability or risk bound',
        dag_provider_scope='Separate source-bound trained node/host/generation providers and exact-context transfer qualification required; no whole-request wall multiplier fitted here',
        posterior_calibrated=False,risk_bound_calibrated=False,physical_accuracy_validated=False,deployment_qualified=False,GPU_started=False)
    result['model_sha256']=identity(result)
    return result


def verify_support(document,*,allow_cpu_fixture=False):
    binding=document.get('fit_contract',{})
    if binding.get('version')!=FIT_VERSION or binding.get('fitter_source')!=receipt(__file__):
        raise ValueError('Reconstructed training fit and unchanged fitter source required')
    if document.get('fixture_only') and not allow_cpu_fixture:
        raise ValueError('CPU fixture support cannot qualify measured native prediction')
    reconstructed=fit_support(binding['review_refs'],allow_cpu_fixture=allow_cpu_fixture,
                              minimum_independent_runs=binding['minimum_independent_runs'])
    if reconstructed!=document:
        raise ValueError('Support differs from recomputed raw-derived emissions/start supply/radius/prior/joint scenarios')
    return reconstructed


def load_support(path,*,allow_cpu_fixture=False):
    return verify_support(json.loads(Path(path).read_text()),allow_cpu_fixture=allow_cpu_fixture)


class JointLifecycleFromSupport:
    """Use the same whole-run occurrence draw for END/RESULT/worker READY."""
    def __init__(self,support,scenario_id,public_cohort,*,allow_cpu_fixture=False):
        from .dag_busy_posterior_v8 import occurrence_indices
        verify_support(support,allow_cpu_fixture=allow_cpu_fixture)
        self.support_sha256=support['model_sha256'];self.cohort=copy.deepcopy(public_cohort)
        self.indices=occurrence_indices(self.cohort)
        self.scenario=copy.deepcopy(next(s for s in support['scenarios'] if s['id']==scenario_id))

    def delays(self,row,aliases,scenario):
        from .dag_busy_posterior_v8 import input_draw
        if self.cohort.get(row['request_id'])!=row:
            raise ValueError('Joint lifecycle changed original public row')
        value=input_draw(self.scenario['lifecycle_delays_by_input'],row,self.indices)
        return dict(copy.deepcopy(value),known=True,scope='reviewed_training_whole_run_joint_lifecycle_draw')
