"""Read-only SQLite equivalence audit: interval index versus a separate scan.

Every CUPTI kernel is checked; repeated launches share identical query results
on both paths. Source SQLite is opened read-only and never indexed or modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from .nvtx_interval_index_v2 import NVTXRangeIndex
from .trace_export import _stage_label, linux_pid


def _receipt(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest())


def brute_owner(ranges, tid, start, end):
    """Intentionally independent O(R) oracle, matching the original exporter."""
    matching = [r for r in ranges if r['global_tid'] == tid
                and r['nsys_start_ns'] <= start and end <= r['nsys_end_ns']]
    stages = [r for r in matching if r['stage']]
    return min(stages or matching, key=lambda r: r['nsys_end_ns']-r['nsys_start_ns']) if matching else None


def audit_sqlite(path):
    source = _receipt(path)
    connection = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        strings = {r['id']: r['value'] for r in connection.execute('select * from StringIds')}
        ranges = []
        for row in connection.execute('select * from NVTX_EVENTS where end is not null order by start,end'):
            label = row['text'] or strings.get(row['textId'], '')
            ranges.append(dict(global_tid=row['globalTid'], nsys_start_ns=row['start'], nsys_end_ns=row['end'],
                               stage=_stage_label(label), label=label, source_ordinal=len(ranges)))
        launches = {}
        for row in connection.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
            if row['correlationId'] is not None:
                launches[linux_pid(row['globalTid']), row['correlationId']] = (row['globalTid'], row['start'], row['end'])
        kernels, unique_queries = [], {}
        for row in connection.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start,end'):
            key = (linux_pid(row['globalPid']), row['correlationId'])
            query = launches.get(key)
            kernels.append((key, query))
            if query is not None:
                unique_queries.setdefault(query, None)
    finally:
        connection.close()
    begun = time.perf_counter_ns()
    index = NVTXRangeIndex(ranges)
    build_ns = time.perf_counter_ns()-begun
    indexed, brute = {}, {}
    visits, matches, maximum_visits = 0, 0, 0
    begun = time.perf_counter_ns()
    for query in unique_queries:
        stats = {}
        owner = index.owner_for_launch(*query, stats=stats)
        indexed[query] = owner['source_ordinal'] if owner is not None else None
        visits += stats['visited_nodes']
        matches += stats['matched_ranges']
        maximum_visits = max(maximum_visits, stats['visited_nodes'])
    indexed_ns = time.perf_counter_ns()-begun
    begun = time.perf_counter_ns()
    for query in unique_queries:
        owner = brute_owner(ranges, *query)
        brute[query] = owner['source_ordinal'] if owner is not None else None
    brute_ns = time.perf_counter_ns()-begun
    mismatches, mismatch_count, assigned = [], 0, 0
    index_hash, brute_hash = hashlib.sha256(), hashlib.sha256()
    for ordinal, (key, query) in enumerate(kernels):
        a, b = indexed.get(query), brute.get(query)
        assigned += int(a is not None)
        for digest, owner in ((index_hash, a), (brute_hash, b)):
            digest.update(json.dumps([ordinal, list(key), owner], separators=(',', ':')).encode()+b'\n')
        if a != b:
            mismatch_count += 1
            if len(mismatches) < 20:
                mismatches.append(dict(kernel_ordinal=ordinal, process_correlation=list(key), launch=query,
                                       indexed_owner=a, brute_owner=b))
    after = _receipt(path)
    if after != source:
        raise ValueError('Source SQLite changed during read-only audit')
    baseline_checks = len(unique_queries)*len(ranges)
    return dict(status='PASS' if not mismatch_count else 'FAIL', source=source,
        range_count=len(ranges), thread_count=len(index.roots), kernel_count=len(kernels),
        process_count=len({key[0] for key, _ in kernels}), unique_launch_intervals=len(unique_queries),
        kernels_without_runtime_launch=sum(q is None for _, q in kernels), attributed_kernel_count=assigned,
        unmatched_kernel_count=len(kernels)-assigned, mismatch_count=mismatch_count, mismatch_examples=mismatches,
        indexed_kernel_ownership_sha256=index_hash.hexdigest(), brute_kernel_ownership_sha256=brute_hash.hexdigest(),
        work=dict(legacy_per_kernel_full_scan_checks=len(kernels)*len(ranges),
                  cached_oracle_range_predicates=baseline_checks, indexed_visited_nodes=visits,
                  indexed_matched_ranges=matches, maximum_query_node_visits=maximum_visits,
                  visited_to_cached_full_scan_ratio=visits/baseline_checks if baseline_checks else None),
        elapsed_ns=dict(build_index=build_ns, indexed_queries=indexed_ns, brute_queries=brute_ns),
        source_unchanged=True, exact_kernel_owner_equivalence=mismatch_count == 0,
        scope='Every kernel owner checked; clocks, phase aggregation, GPU accuracy and physical qualification are unchanged/not tested')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sqlite', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for path in args.sqlite:
        result = audit_sqlite(path)
        results.append(result)
        # Individual reports survive an interrupted multi-file audit.
        with (args.output/('trace-'+str(len(results))+'.json')).open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps(dict(path=str(path), status=result['status'], kernels=result['kernel_count'],
                              mismatches=result['mismatch_count'])), flush=True)
    report = dict(schema='gsdeploy.nvtx_interval_index_equivalence_audit.v2',
        status='PASS' if all(r['status'] == 'PASS' for r in results) else 'FAIL', GPU_started=False,
        index_source=_receipt(Path(__file__).with_name('nvtx_interval_index_v2.py')),
        oracle_source=_receipt(__file__), legacy_exporter_source=_receipt(Path(__file__).with_name('trace_export.py')),
        traces=results, existing_artifacts_modified=False)
    with (args.output/'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
