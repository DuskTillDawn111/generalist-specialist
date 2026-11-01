"""Read-only Nsight session-clock and causal worker-boundary audit.

The worker pushes NVTX before its perf start, and timestamps perf end before
emitting/flushing PHASE and popping NVTX. These are brackets, not simultaneous
clock samples. No new clock/profiler/GPU experiment is performed here.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sqlite3

from .occupancy_calculator_v2 import receipt


def causal_offset_interval(phases, pid):
    """Conditional constant-offset interval from strictly nested host anchors."""
    selected = [row for row in phases if row.get('pid') == pid]
    if not selected:
        return dict(known=False, reason='NO_SOURCE_ORDERED_MATCHED_ANCHORS')
    for row in selected:
        values = [row.get(name) for name in ('start_ns','end_ns','nsys_start_ns','nsys_end_ns')]
        if any(type(value) is not int for value in values) or values[1] < values[0] or values[3] < values[2]:
            raise ValueError('Explicit valid integer paired phase boundaries required')
    lower = max(row['end_ns']-row['nsys_end_ns'] for row in selected)
    upper = min(row['start_ns']-row['nsys_start_ns'] for row in selected)
    if lower > upper:
        return dict(known=False, reason='NO_COMMON_CONSTANT_OFFSET_FITS_CAUSAL_BRACKETS',
                    lower_ns=lower, upper_ns=upper, phase_count=len(selected))
    return dict(known=True, lower_ns=lower, upper_ns=upper, width_ns=upper-lower,
                phase_count=len(selected), constant_offset_assumed=True,
                drift_bound_ns=None, device_host_clock_calibration_error_bound_ns=None)


def audit_one(sqlite_path, parsed_path):
    source, parsed_source = receipt(sqlite_path), receipt(parsed_path)
    parsed = json.loads(Path(parsed_path).read_text())
    if source not in parsed.get('sources', []):
        raise ValueError('Parsed phases do not bind the exact source SQLite bytes')
    connection = sqlite3.connect('file:'+str(Path(sqlite_path).resolve())+'?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        sessions = [dict(row) for row in connection.execute('select * from TARGET_INFO_SESSION_START_TIME')]
        if len(sessions) != 1 or any(type(sessions[0].get(name)) is not int for name in ('utcEpochNs','systemClockNs')):
            raise ValueError('One exact integer Nsight session clock origin required')
        capture = [dict(row) for row in connection.execute('select * from META_DATA_CAPTURE')]
        exports = [dict(row) for row in connection.execute('select * from META_DATA_EXPORT')]
        session = sessions[0]
    finally:
        connection.close()
    records = []
    for pid_text, mapping in parsed.get('clock_mappings', {}).items():
        interval = causal_offset_interval(parsed['phases'], int(pid_text))
        session_offset = session['systemClockNs']
        median = mapping.get('offset_ns')
        if type(median) is not int:
            raise ValueError('Explicit integer retained median map required')
        records.append(dict(pid=int(pid_text), causal_interval=interval,
            median_mapping=mapping, session_system_clock_offset_ns=session_offset,
            median_minus_session_offset_ns=median-session_offset,
            session_origin_fits_causal_brackets=(interval['lower_ns'] <= session_offset <= interval['upper_ns'])
                if interval['known'] else None,
            median_origin_fits_causal_brackets=(interval['lower_ns'] <= median <= interval['upper_ns'])
                if interval['known'] else None,
            offset_error_relative_to_session_interval_ns=[interval['lower_ns']-session_offset,
                                                        interval['upper_ns']-session_offset]
                if interval['known'] else None,
            calibrated_precision_bound_ns=None,
            one_microsecond_cross_process_alignment_qualified=False))
    if receipt(sqlite_path) != source or receipt(parsed_path) != parsed_source:
        raise ValueError('Frozen input changed during read-only audit')
    return dict(sqlite_source=source, parsed_source=parsed_source, session=session,
        utc_minus_system_clock_ns=session['utcEpochNs']-session['systemClockNs'],
        recorded_target_display_names=[row['value'] for row in capture if row['name']=='DEVICE_DISPLAY_NAME'],
        export_metadata=exports, process_mappings=records, source_unchanged=True)


def audit(pairs, output):
    traces = [audit_one(sqlite_path, parsed_path) for sqlite_path,parsed_path in pairs]
    offsets = [trace['utc_minus_system_clock_ns'] for trace in traces]
    relative_intervals = []
    processes = [(trace['sqlite_source']['path'],mapping) for trace in traces for mapping in trace['process_mappings']]
    for (left_source,left),(right_source,right) in itertools.combinations(processes,2):
        a,b = left['offset_error_relative_to_session_interval_ns'],right['offset_error_relative_to_session_interval_ns']
        relative_intervals.append(dict(left_source=left_source,left_pid=left['pid'],
            right_source=right_source,right_pid=right['pid'],
            relative_offset_error_interval_ns=[a[0]-b[1],a[1]-b[0]] if a and b else None,
            constant_independent_offset_brackets_assumed=True,
            calibrated_precision_bound_ns=None, one_microsecond_alignment_qualified=False))
    report = dict(schema='gsdeploy.nsys_session_clock_audit.v2', traces=traces,
        utc_minus_system_clock_spread_ns=max(offsets)-min(offsets) if offsets else None,
        relative_intervals=relative_intervals,
        implementation=receipt(__file__),
        source_ordering_receipts=[receipt(Path(__file__).with_name(name)) for name in ('semantic_worker.py','semantic_observer.py')],
        official_documentation=[receipt('/opt/nvidia/nsight-systems/2025.6.3/documentation/'+name+'/index.html')
                                for name in ('AnalysisGuide','UserGuide')],
        official_basis=dict(sqlite_schema='systemClockNs: Target system clock timestamp at start of capture',
            synchronization='Same-target TSC synchronization can align reports with nanoseconds-level precision; UTC/NTP alignment has millisecond-scale limitations',
            missing='SQLite session origin alone does not expose alignment source, cross-clock calibration error, or a measured drift bound'),
        recommended_mapping='Preserve session systemClockNs as the metadata-based candidate origin; retain causal intervals and calibration limitations, not the median residual as an accuracy bound',
        target_one_microsecond_alignment_qualified=False,
        GPU_workload_executed=False, profiling_started=False,
        natural_timing_validated=False, deployment_qualified=False)
    output = Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    path = output/'audit.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'output-manifest.json').write_text(json.dumps(dict(outputs=[receipt(path)]),indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair',nargs=2,action='append',required=True,metavar=('SQLITE','PARSED'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = audit(args.pair,args.output)
    print(json.dumps(dict(utc_minus_system_clock_spread_ns=report['utc_minus_system_clock_spread_ns'],
                         one_microsecond_alignment_qualified=False,GPU_workload_executed=False)))


if __name__ == '__main__':
    main()
