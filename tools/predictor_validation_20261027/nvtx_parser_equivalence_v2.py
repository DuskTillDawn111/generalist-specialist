"""Compare integrated indexed parser output with immutable pre-index outputs.

Reuses explicit anchors preserved in the prior phase output. Full payloads,
including kernels, NVTX ranges, phase unions and clock mappings, must match.
This CPU check neither regenerates nor rewrites the historical observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .nvtx_interval_audit_v2 import _receipt
from .trace_export import parse_sqlite


def _first_difference(left, right, path='$'):
    if type(left) is not type(right):
        return path+': type'
    if isinstance(left, dict):
        if set(left) != set(right):
            return path+': keys'
        for key in left:
            found = _first_difference(left[key], right[key], path+'.'+key)
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return path+': length'
        for index, (a, b) in enumerate(zip(left, right)):
            found = _first_difference(a, b, path+'['+str(index)+']')
            if found:
                return found
        return None
    return None if left == right else path+': value'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-parsed', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for path in args.legacy_parsed:
        original_receipt = _receipt(path)
        original = json.loads(path.read_text())
        if len(original['sources']) != 1 or original['sources'][0] != _receipt(original['sources'][0]['path']):
            raise ValueError('Legacy SQLite identity missing or changed')
        if original['unmatched_anchor_count']:
            raise ValueError('Cannot recover unmatched original anchors from phase output')
        anchors = [{k: phase[k] for k in ('pid', 'request_id', 'kind', 'segment_id', 'start_ns', 'end_ns')}
                   for phase in original['phases']]
        begun = time.perf_counter_ns()
        current = parse_sqlite(original['sources'][0]['path'], anchors)
        elapsed_ns = time.perf_counter_ns()-begun
        # Historical evidence is JSON. NVTX stage tuples become JSON arrays;
        # compare identical serialized contracts rather than Python container
        # representations that JSON cannot preserve.
        current = json.loads(json.dumps(current, allow_nan=False))
        difference = _first_difference(original, current)
        if _receipt(path) != original_receipt:
            raise ValueError('Historical parse output changed during read-only comparison')
        parsed_path = args.output/('parsed-indexed-'+str(len(results)+1)+'.json')
        with parsed_path.open('x') as stream:
            json.dump(current, stream, allow_nan=False)
        result = dict(status='PASS' if difference is None else 'FAIL', legacy_parsed=original_receipt,
            indexed_parsed=_receipt(parsed_path), sqlite=original['sources'][0], first_difference=difference,
            phase_count=len(current['phases']), kernel_count=sum(p['kernel_count'] for p in current['phases'])+len(current['unattributed_kernels']),
            full_output_exactly_equal=difference is None, indexed_parse_elapsed_ns=elapsed_ns, historical_sources_unchanged=True)
        results.append(result)
        print(json.dumps(dict(path=str(path), status=result['status'], first_difference=difference)), flush=True)
    report = dict(schema='gsdeploy.indexed_nsys_parser_equivalence.v2',
        status='PASS' if all(r['status'] == 'PASS' for r in results) else 'FAIL', traces=results,
        sources=[_receipt(__file__), _receipt(Path(__file__).with_name('trace_export.py')),
                 _receipt(Path(__file__).with_name('nvtx_interval_index_v2.py'))], GPU_started=False,
        scope='Full parser structural/numeric equivalence, not GPU prediction or throughput qualification')
    with (args.output/'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
