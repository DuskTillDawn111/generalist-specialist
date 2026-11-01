"""Opt-in same-owner ChatQA batched worker. --help never loads a model.

Wire input: {op: batch, batch_id: str, requests: [{request_id, sample, ...}]}.
No implicit batching window or routing; the CPU scheduler supplies each batch.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback


def emit(kind, payload):
    print(kind+'|'+json.dumps(payload, ensure_ascii=False), flush=True)


def serve(backend, input_lines, *, sink=emit):
    """Injectable CPU-tested wire loop; failures preserve identities and propagate.

    Stops on a batch failure, because a generate error can invalidate owner/KV
    state. It never retries, substitutes single-request execution or skips rows.
    """
    sink('READY', dict(backend='same_owner_chatqa_hf_batch_v2', owner_identity=backend.identity,
                       model_owner_count=1, tasks=['text'], batch_wire=True,
                       final_gain_eligible=False))
    for line in input_lines:
        message = None
        try:
            message = json.loads(line)
            if message.get('op') == 'shutdown':
                sink('BYE', dict(backend='same_owner_chatqa_hf_batch_v2'))
                return
            if message.get('op') != 'batch' or not isinstance(message.get('batch_id'), str) or not message['batch_id']:
                raise ValueError('Explicit batch operation and nonempty batch_id required')
            requests = message.get('requests')
            if not isinstance(requests, list) or not requests:
                raise ValueError('Nonempty requests list required')
            ids = [r.get('request_id') for r in requests]
            if any(not isinstance(rid, str) or not rid for rid in ids) or len(set(ids)) != len(ids):
                raise ValueError('Distinct nonempty request IDs required')
            if any(r.get('task_type', 'text') != 'text' for r in requests):
                raise ValueError('Text-only worker')
            start_unix = time.time_ns()
            start_perf = time.perf_counter_ns()
            prepared = [backend.prepare(r['sample']) for r in requests]
            sink('START', dict(batch_id=message['batch_id'], request_ids=ids,
                               perf_ns=start_perf, unix_ns=start_unix))
            execution = backend.execute_prepared(prepared, evaluate_quality=False)
            end_unix = time.time_ns()
            end_perf = time.perf_counter_ns()
            if len(execution.results) != len(requests):
                raise ValueError('Backend result count changed')
            # All results become visible only after the whole batch completes.
            rows = []
            for request, result in zip(requests, execution.results):
                row = dict(result)
                row.update(request_id=request['request_id'], task_type='text',
                           start_ns=start_unix, end_ns=end_unix,
                           start_perf_ns=start_perf, end_perf_ns=end_perf,
                           worker_latency_ms=(end_perf-start_perf)/1e6,
                           batch_id=message['batch_id'])
                for key in ('submit_ns','pool_id','arrival_index'):
                    if key in request:
                        row[key] = request[key]
                if 'submit_ns' in request:
                    row['queue_wait_ms'] = (start_unix-request['submit_ns'])/1e6
                rows.append(row)
            sink('BATCH_RESULT', dict(batch_id=message['batch_id'], request_ids=ids,
                                      results=rows, metrics=execution.metrics))
        except Exception as exc:
            sink('ERROR', dict(batch_id=message.get('batch_id') if isinstance(message, dict) else None,
                request_ids=[r.get('request_id') for r in message.get('requests', []) if isinstance(r, dict)]
                    if isinstance(message, dict) else [],
                error=repr(exc), traceback=traceback.format_exc(), retry_permitted=False))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--spec-sha256')
    parser.add_argument('--execute', action='store_true', help='Explicitly load the real retained GPU model')
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error('Model loading is disabled; explicit --execute is required')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gsdeploy.backends.shared_text_batch import load_retained
    backend = load_retained(args.spec, args.source_root, execute=True,
                            expected_spec_sha256=args.spec_sha256)
    serve(backend, sys.stdin)


if __name__ == '__main__':
    main()
