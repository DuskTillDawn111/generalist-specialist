"""CPU independent test and causal-cutoff replay evidence for ledger v1."""
import argparse
import io
import json
from pathlib import Path
import unittest

from .native_collect import receipt, save
from .visible_device_event_ledger_v1 import advance,new_state


def review(raw_path,output):
    raw_ref=receipt(raw_path);raw=json.loads(Path(raw_path).read_text());context_ref=raw['context']
    if receipt(context_ref['path'])!=context_ref:raise ValueError('Context changed')
    context=json.loads(Path(context_ref['path']).read_text())
    if raw['performance_scope']!='unprofiled_with_phase_markers' or context['paired_profile_mode']!='natural':raise ValueError('Natural raw required')
    rows=raw['request_records'];cutoffs=set();queries=[]
    for row in rows:
        cutoffs.update(e['received_time_ns'] for e in row['execution_events'])
        for key in ('response_reader_event_enqueue_time_ns',):
            if key in row:cutoffs.add(row[key])
        cutoffs.add(row['previous_handler_next_request_ready']['received_time_ns'])
        queries.extend(dict(alias=row['model'],**e) for e in row['execution_events'] if e.get('event')=='observation')
    state=new_state(context['residents'],raw['execution_context_sha256']);replay=[]
    for cutoff in sorted(cutoffs):
        state=advance(state,rows,cutoff)
        expected=sorted((q for q in queries if q['received_time_ns']<=cutoff),key=lambda q:(q['received_time_ns'],q['request_id'],q['observation_sequence']))
        if state['visible_device_events']!=expected or state['kernel_cursor'] is not None or state['main_lifecycle_resume_eligible'] is not False:raise ValueError('Causal prefix or scope failure')
        if any(r['kernel_cursor'] is not None for r in state['host']['requests'].values()):raise ValueError('Host cursor inferred')
        replay.append(dict(cutoff_ns=cutoff,queries=len(expected),requests=len(state['host']['requests']),state_sha256=state['state_sha256']))
    modules=['tests.test_visible_device_event_ledger_v1_independent_review','tests.test_visible_device_event_ledger_v1'];stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    if not result.wasSuccessful() or result.testsRun!=18:raise ValueError('Independent regressions failed')
    root=Path(__file__).resolve().parents[2]
    sources=[receipt(__file__),receipt(Path(__file__).with_name('visible_device_event_ledger_v1.py')),receipt(root/'gsdeploy/forecasting/observed_state.py')]
    sources.extend(receipt(root/(module.replace('.','/')+'.py')) for module in modules)
    if receipt(raw_path)!=raw_ref:raise ValueError('Raw changed')
    out=Path(output);out.mkdir(exist_ok=False)
    report=dict(schema='gsdeploy.visible_device_event_ledger_independent_CPU_review.v1',status='PASS_LIMITED_CAUSAL_LEDGER',issues=[],raw=raw_ref,context=context_ref,sources=sources,
        tests=dict(run=result.testsRun,independent=11,author=7,failures=len(result.failures),errors=len(result.errors)),
        replay=dict(reader_receipt_cutoffs=len(cutoffs),measured_requests=len(rows),visible_queries=len(queries),final_state_sha256=state['state_sha256']),
        kernel_cursor=None,main_lifecycle_resume_eligible=False,GPU_started=False,validation_results_read=False,
        limitations=['The persistent registry must retain all consumed requests and events; this is not a sliding-window adapter.',
            'State hashes detect accidental mutation, not a malicious writer who can reseal modified checkpoints.',
            'Source/raw bindings and exact visible prefixes do not independently authenticate native CUDA calls; natural raw has no native CUDA API trace.',
            'Per-request recorded-stream completion does not identify global GPU completion, progress fraction, SM service, or executable continuation.'])
    save(out/'report.json',report);save(out/'cutoff-replay.json',replay);(out/'tests.txt').write_text(stream.getvalue());return receipt(out/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(review(args.raw,args.output)))
