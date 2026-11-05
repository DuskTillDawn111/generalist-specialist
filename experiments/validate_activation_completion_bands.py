"""Completion-boundary candidate replay with unchanged shared FIFO/energy gates.

Raw coarse PHASE records remain authoritative. Derived host observations are
used only for work-error scoring; forecast inputs contain no observed bands,
completion clocks, forward lists or eventual token counts.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.activation_completion_provider import CompletionBandContextProvider as BandContextProvider
from experiments.activation_completion_bands import completion_band_layout as band_layout, SCHEMA as BOUNDARY_SCHEMA
from experiments.fit_activation_native import load_bundle
from experiments.analyze_activation_path_accuracy import path_accuracy
from gsdeploy.energy import integrate_power
from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.lifecycle import LifecycleProvider
from gsdeploy.forecasting.native_context import counts
from gsdeploy.forecasting.native_order import request_orders
from gsdeploy.forecasting.native_training import digest, stage_layout, validate_capture
from gsdeploy.forecasting.native_validation import DEFAULT_CRITERIA, metrics
from gsdeploy.forecasting.simulator import FullHorizonModel


def score_band_fit(fit,bundles,*,training_replay):
    template=fit['profiles'];context=template['execution_context_sha256']
    provider=BandContextProvider(template,context,allow_cpu_fixture=True)
    lifecycle=LifecycleProvider(fit['lifecycle'],context,allow_cpu_fixture=True)
    trained={doc['run_id'] for doc in fit['normalized_runs']}
    ids=[bundle['raw']['run_id'] for bundle in bundles]
    receipts={bundle['source'] for bundle in bundles}
    if len(set(ids))!=len(ids):
        raise ValueError('Distinct independent capture runs required')
    if training_replay:
        if not set(ids)<=trained or not receipts<=set(fit['source_receipts']):
            raise ValueError('Training replay requires original fitted receipts')
    elif set(ids)&trained or receipts&set(fit['source_receipts']):
        raise ValueError('Independent validation cannot reuse training runs or receipts')
    criteria=DEFAULT_CRITERIA; reports=[]; reasons=[]; all_latency=[]; all_work=[]
    for bundle in bundles:
        if validate_capture(bundle,allowed_splits=('training',) if training_replay else ('development_validation',))!=context:
            raise ValueError('Frozen band predictor and capture execution differ')
        raw=bundle['raw'];orders=request_orders(raw)
        if raw['kind']!=template['provenance']['kind']:
            raise ValueError('Physical and fixture evidence cannot be mixed')
        aliases={alias:dict(family=family,supported_tasks=sorted({p['task'] for p in provider.profiles.values() if p['family']==family}))
                 for alias,family in raw['resident_aliases'].items()}
        window=[raw['observation_start_ns'],raw['observation_end_ns']]
        spans=[];baselines={};changes=defaultdict(list);unsupported=[]
        for row in raw['request_records']:
            plan=provider.plan(dict(alias=row['model'],task=row['task'],pool_id=row['pool_id'],
                                    native_order=orders[row['request_id']]),aliases,'mean')
            observed=band_layout(row,aliases[row['model']]['family'])
            if [(s['id'],s['kind']) for s in observed]!=[(s['id'],s['kind']) for s in plan['segments']]:
                raise ValueError('Observed derived layout differs from frozen predicted layout')
            for event in row['execution_events']:
                if (event.get('request_id')!=row['request_id'] or type(event.get('received_time_ns')) is not int
                        or event['received_time_ns']<event['perf_ns']):
                    raise ValueError('Band source event identity or common clock visibility differs')
            response=row['worker_response'];ready=row['previous_handler_next_request_ready']
            if (row['completion_time_ns']!=row['response_reader_event_enqueue_time_ns']
                    or row['completion_time_ns']<response['end_perf_ns']
                    or ready.get('request_id')!=row['request_id'] or ready.get('clock')!='host_perf_counter_ns'
                    or ready.get('boundary')!='previous_handler_next_request_ready'
                    or ready.get('scope')!='after_response_flush_before_next_read'
                    or not response['end_perf_ns']<=ready['perf_ns']<=ready['received_time_ns']):
                raise ValueError('Original handler/result/next-ready lifecycle mismatch')
            for actual,predicted in zip(observed,plan['segments']):
                if not window[0]<=actual['start_ns']<actual['end_ns']<=window[1]:
                    raise ValueError('Band outside declared observation window')
                span=dict(alias=row['model'],profile_id=plan['profile_id'],segment_id=actual['id'],kind=actual['kind'],
                    task=row['task'],pool_id=row['pool_id'],request_id=row['request_id'],
                    start_ns=actual['start_ns'],end_ns=actual['end_ns'])
                spans.append(span);baselines[row['request_id'],actual['id']]=predicted['duration_ms']
                changes[span['start_ns']].append(('start',span));changes[span['end_ns']].append(('end',span))
        active={}; work=defaultdict(float);bad_work=set();windows=[]
        cuts=sorted({*window,*changes})
        for index,left in enumerate(cuts):
            for kind,span in sorted(changes[left],key=lambda pair:pair[0]):
                if kind=='end':
                    if active.pop(span['alias']) is not span:
                        raise ValueError('Unmatched observed band boundary')
                else:
                    if span['alias'] in active:
                        raise ValueError('Observed same-owner overlap')
                    active[span['alias']]=span
            if index+1==len(cuts):
                continue
            right=cuts[index+1]
            windows.append((left,right,counts(active,provider.profiles,provider.names)))
            try:
                context_values,_=provider._context(active,aliases,'mean')
                for alias,span in active.items():
                    if span['kind']=='gpu':
                        key=span['request_id'],span['segment_id']
                        work[key]+=(right-left)/1e6/baselines[key]*context_values['progress'][alias]
            except UnknownEvidence as exc:
                unsupported.append(str(exc))
                bad_work.update((span['request_id'],span['segment_id']) for span in active.values() if span['kind']=='gpu')
        if active:
            raise ValueError('Incomplete observed band roster')
        work_errors=[abs(value-1.) for key,value in work.items() if key not in bad_work]
        prefix=[s['end_ns'] for row in raw.get('priming_request_records',[]) for s in stage_layout(row) if s['kind']=='gpu']
        initial_state=max(prefix) if prefix and max(prefix)<=window[0] else None
        measured=integrate_power(raw['board_samples'],*window,max_gap_seconds=.25)
        if measured is None or measured<=0:
            raise ValueError('Complete positive measured board energy required')
        observed_energy=None
        try:
            observed_energy,_=provider.energy.integrate(windows,initial_state)
        except UnknownEvidence as exc:
            unsupported.append(str(exc))
        model=FullHorizonModel(aliases,provider,lifecycle_provider=lifecycle)
        by_id={row['request_id']:row for row in raw['request_records']}; seen=set()
        batches=[];latency=[];fifo_joules=0.;fifo_supported=True
        clock=window[0];state=initial_state;previous_actual_end=window[0]
        for batch in raw['batches']:
            rows=[by_id[rid] for rid in batch['request_ids']]
            if seen&set(batch['request_ids']):
                raise ValueError('Duplicate batch request ownership')
            seen.update(batch['request_ids']); now=batch['start_ns']
            end=batch.get('end_ns') or max([now+1,*[max(row['completion_time_ns'],row['previous_handler_next_request_ready']['perf_ns']) for row in rows]])
            if not window[0]<=now<end<=window[1] or now<previous_actual_end:
                raise ValueError('Complete nonoverlapping controlled batches required')
            try:
                gap=now-previous_actual_end
                if gap:
                    joules,state=provider.energy_interval({},clock,clock+gap,state)
                    fifo_joules+=joules;clock+=gap
                batch_state=now-(clock-state) if state is not None else None
                queues=defaultdict(list);future=[]
                for row in rows:
                    causal=dict(request_id=row['request_id'],task=row['task'],pool_id=row['pool_id'],alias=row['model'],
                                arrival_ns=row['scheduled_time_ns'],native_order=orders[row['request_id']])
                    if causal['arrival_ns']<now:
                        raise ValueError('Arrival before its declared batch')
                    if causal['arrival_ns']==now:
                        queues[causal['alias']].append(causal)
                    else:
                        future.append(causal)
                deadline=max([now,batch.get('arrival_end_ns') or now,*[row['arrival_ns'] for row in future]])+1
                forecast=model.forecast(dict(queues),now,deadline,future_arrivals=future,
                    route_future=lambda row,pending,at_ns:row['alias'],drain_deadline_ns=deadline+120_000_000_000,
                    require_energy=False,require_memory=False,require_episode_energy=True,energy_state=batch_state)
                if forecast['incomplete_ids']:
                    raise UnknownEvidence('INCOMPLETE_DERIVED_BAND_FIFO_REPLAY')
                errors=[]
                for row in rows:
                    actual=(row['completion_time_ns']-row['scheduled_time_ns'])/1e6
                    predicted=(forecast['finishes'][row['request_id']]['finish_ns']-row['scheduled_time_ns'])/1e6
                    error=abs(predicted-actual)/actual;latency.append(error)
                    errors.append(dict(request_id=row['request_id'],observed_e2e_ms=actual,predicted_e2e_ms=predicted,relative_error=error))
                joules=forecast['through_drain_board_energy_joules']
                if joules is None:
                    raise UnknownEvidence('UNKNOWN_PREDICTED_BAND_FIFO_ENERGY')
                fifo_joules+=joules;clock=forecast['drain_finish_ns'];state=forecast['final_energy_state']
                batches.append(dict(id=batch['id'],status='predicted',requests=errors,predicted_batch_energy_joules=joules,
                                    predicted_batch_duration_ms=(clock-now)/1e6))
            except UnknownEvidence as exc:
                fifo_supported=False;unsupported.append(str(exc))
                batches.append(dict(id=batch['id'],status='UNKNOWN',reason=str(exc)))
            previous_actual_end=end
        if seen!=set(by_id):
            raise ValueError('Full observed cohort not covered by predicted batches')
        if fifo_supported:
            try:
                tail=window[1]-previous_actual_end
                if tail:
                    joules,_=provider.energy_interval({},clock,clock+tail,state);fifo_joules+=joules
            except UnknownEvidence as exc:
                fifo_supported=False;unsupported.append(str(exc))
        fifo_energy=fifo_joules if fifo_supported else None
        exposure_error=abs(observed_energy-measured)/measured if observed_energy is not None else None
        fifo_error=abs(fifo_energy-measured)/measured if fifo_energy is not None else None
        if unsupported:reasons.append(raw['run_id']+':UNSUPPORTED_CONTEXT_OR_REPLAY')
        if exposure_error is None or exposure_error>criteria['maximum_integrated_energy_relative_error']:
            reasons.append(raw['run_id']+':INTEGRATED_ENERGY_ERROR_OR_UNKNOWN')
        if fifo_error is None or fifo_error>criteria['maximum_integrated_energy_relative_error']:
            reasons.append(raw['run_id']+':FIFO_INTEGRATED_ENERGY_ERROR_OR_UNKNOWN')
        reports.append(dict(run_id=raw['run_id'],source=bundle['source'],batches=batches,
            latency_errors=metrics(latency),native_work_errors=metrics(work_errors),unsupported=sorted(set(unsupported)),
            energy=dict(observed_joules=measured,predicted_joules=observed_energy,relative_error=exposure_error),
            fifo_energy=dict(observed_joules=measured,predicted_joules=fifo_energy,relative_error=fifo_error,
                scope='autonomous conditional FIFO with continuous predicted energy history and fixed controller gaps; observed priming prefix only')))
        all_latency.extend(latency);all_work.extend(work_errors)
    latency=metrics(all_latency);work=metrics(all_work)
    if len(bundles)<criteria['minimum_independent_runs']:reasons.append('INSUFFICIENT_INDEPENDENT_VALIDATION_RUNS')
    if not latency['n'] or latency['mean']>.15 or latency['p95']>.30:reasons.append('FIFO_E2E_ERROR_OR_UNKNOWN')
    if not work['n'] or work['p95']>.20:reasons.append('NATIVE_WORK_ERROR_OR_UNKNOWN')
    report=dict(schema='gsdeploy.completion_band_training_replay.v1' if training_replay else 'gsdeploy.completion_band_development_validation.v1',
        boundary_policy=BOUNDARY_SCHEMA,
        fit_sha256=digest(fit),criteria=criteria,training_replay=training_replay,development_accuracy_passed=False,
        admission_enabled=False,formal_utility_qualified=False,independent_run_ids=ids,latency_errors=latency,
        native_work_errors=work,reasons=reasons,runs=reports)
    paths=path_accuracy(report)
    if not paths['all_paths_timing_passed']:reasons.append('PER_PATH_FIFO_E2E_ERROR_OR_UNKNOWN')
    report['additional_path_gate']=paths
    if training_replay:
        report['training_replay_accuracy_within_threshold']=not reasons
    else:
        report['development_accuracy_passed']=not reasons
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    fit=json.loads(args.fit.read_text())
    if not fit['training_fit_gates_passed']:
        raise ValueError('All phase-progress and energy fit groups must pass before training replay')
    bundles=[load_bundle(Path(source.rsplit(':sha256:',1)[0]).parent) for source in fit['source_receipts']]
    if [bundle['source'] for bundle in bundles]!=fit['source_receipts']:
        raise ValueError('Original training receipts changed')
    paths=sorted((ROOT/'gsdeploy/forecasting').glob('*.py'))+[
        ROOT/'experiments/activation_semantic_bands.py',ROOT/'experiments/activation_completion_bands.py',
        ROOT/'experiments/activation_completion_provider.py',ROOT/'experiments/fit_activation_completion_bands.py',
        ROOT/'experiments/fit_activation_declared_idle_energy.py',
        ROOT/'experiments/analyze_activation_path_accuracy.py',Path(__file__)]
    sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report=score_band_fit(fit,bundles,training_replay=True)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha for path,sha in sources.items()):
        raise ValueError('Band replay source changed during execution')
    report['prediction_sources']=sources
    with args.output.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ('runs','prediction_sources','additional_path_gate')}))


if __name__=='__main__':
    main()
