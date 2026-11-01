"""Bounded execution of an already sealed original main prediction candidate.

Preparation never starts a GPU. Execution requires an explicit subcommand and
an artifact-bound v48 production receipt. The original v11 child owns the GPU
lock, READY checks, 12 priming requests, 57 offered requests and 30+120s window.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .implementation_source_closure_v27 import implementation_sources
from .known_input_observed_v5 import _bindings
from .learned_campaign_seal_v15 import _preserved_plan
from .native_collect import receipt
from .unified_corun_collect_v8_1 import cleanup_group, group_members, wait_bounded

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'gsdeploy.main_prediction_gate_supervision.v1'
CANDIDATES = ('shared_balanced', 'shared_skewed', 'shared_spill_low', 'shared_spill_high')
BUDGET = dict(free_disk_reserve_bytes=12*2**30, disk_abort_margin_bytes=2**30,
              incremental_storage_scenario_bytes=6*2**30, total_timeout_seconds=2400.,
              watchdog_interval_seconds=.2, cleanup_grace_seconds=8.)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def checked(ref):
    if set(ref) != {'path', 'sha256'} or receipt(ref['path']) != ref:
        raise ValueError('Immutable execution reference changed')
    return read(ref['path'])


def complete_cells(predictions, plan):
    cells = predictions['cells']
    if (len(cells) != 4 or {c['candidate_id'] for c in cells} != set(CANDIDATES)
            or len(plan['trace']) != 57):
        raise ValueError('All four complete original 57-request predictions required')
    expected = {r['request_id'] for r in plan['trace']}
    def numeric(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    for cell in cells:
        if (cell['run_id'] != plan['run_id'] or len(cell['requests']) != 57
                or {r['request_id'] for r in cell['requests']} != expected
                or not isinstance(cell['terms'], dict) or set(cell['terms']) != set('QTLPME')
                or not all(numeric(v) for v in cell['terms'].values())
                or any(not numeric(r[k]) for r in cell['requests'] for k in ('service_ms', 'queue_ms', 'e2e_ms'))
                or [s['offset_ns'] for s in cell.get('queue_depth_samples', [])] != plan['queue_sample_offsets_ns']
                or any(not numeric(s['depth']) for s in cell.get('queue_depth_samples', []))):
            raise ValueError('UNKNOWN or incomplete numeric prediction prohibits GPU collection')


def prerequisites(plan_path, protocol_path, predictions_path, production_path):
    protocol, predictions, plan = _bindings(protocol_path, predictions_path, plan_path, read(plan_path)['run_id'])
    _preserved_plan(plan)
    complete_cells(predictions, plan)
    production = read(production_path)
    if (production.get('schema') != 'gsdeploy.projected_handler_artifact_binding_campaign.v48'
            or production.get('status') != 'ARTIFACT_BOUND_PREDICTIONS_SEALED_REQUIRES_POSTSEAL_GPU_VALIDATION'
            or production.get('protocol') != receipt(protocol_path)
            or production.get('predictions') != receipt(predictions_path)
            or production.get('original_v5_precollection_bindings_passed') is not True
            or production.get('all_GPU_must_start_after_seal') is not True
            or production.get('GPU_started') is not False):
        raise ValueError('Successful exact v48 post-review seal receipt required')
    preparation = checked(production['preparation'])
    review = checked(production['independent_review'])
    if (preparation.get('plan') != receipt(plan_path)
            or review.get('schema') != 'gsdeploy.projected_handler_artifact_binding_independent_review.v48'
            or review.get('status') != 'PASS' or review.get('issues') != []
            or review.get('preparation') != production['preparation']
            or review.get('complete_four_forecasts_reprojected') is not True):
        raise ValueError('Exact independent four-artifact review required')
    sources = implementation_sources(__file__)
    if any(ref not in protocol['sources'] for ref in sources):
        raise ValueError('Supervisor and its complete local implementation closure must be frozen in protocol')
    return plan, sources


def prepare(plan, protocol, predictions, production, output):
    original, sources = prerequisites(plan, protocol, predictions, production)
    result = dict(schema=SCHEMA, status='CPU_PREPARED_REQUIRES_EXPLICIT_EXECUTE',
                  plan=receipt(plan), protocol=receipt(protocol), predictions=receipt(predictions),
                  production=receipt(production), sources=sources, candidates=list(CANDIDATES),
                  run_id=original['run_id'], budget=BUDGET,
                  preflight_free_bytes_required=18*2**30, disk_abort_floor_bytes=13*2**30,
                  timeout_scope='2400 seconds per candidate; includes GPU lock wait, load, priming, 30s offer, 120s drain and shutdown; no campaign-wide 2400s cap',
                  GPU_started=False, NSYS_started=False, no_retries=True,
                  lock_owner='v11 child only; supervisor must not acquire child GPU lock')
    save(output, result)
    return result


def execute(manifest_path, candidate, output):
    manifest_ref = receipt(manifest_path); manifest = checked(manifest_ref)
    if (manifest.get('schema') != SCHEMA or manifest.get('status') != 'CPU_PREPARED_REQUIRES_EXPLICIT_EXECUTE'
            or manifest.get('budget') != BUDGET or candidate not in CANDIDATES
            or manifest.get('candidates') != list(CANDIDATES)
            or manifest.get('preflight_free_bytes_required') != 18*2**30
            or manifest.get('disk_abort_floor_bytes') != 13*2**30):
        raise ValueError('Exact prepared bounded execution manifest required')
    for key in ('plan', 'protocol', 'predictions', 'production'):
        checked(manifest[key])
    plan, sources = prerequisites(*(manifest[k]['path'] for k in ('plan', 'protocol', 'predictions', 'production')))
    if sources != manifest['sources'] or plan['run_id'] != manifest['run_id']:
        raise ValueError('Execution source or run identity drift')
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('Exclusive candidate execution output already exists; no retry')
    if shutil.disk_usage(out.parent).free < 18*2**30:
        raise ValueError('18GiB preflight reserve required; threshold is never reduced')
    out.mkdir(exist_ok=False)
    command = [sys.executable, '-m', 'tools.predictor_validation_20261027.known_input_native_flow_v11',
               'execute', '--plan', manifest['plan']['path'], '--candidate', candidate,
               '--output', str(out/'collection'), '--protocol', manifest['protocol']['path'],
               '--predictions', manifest['predictions']['path'], '--run-id', plan['run_id']]
    result = dict(schema=SCHEMA, status='FAILED_NO_RETRY', manifest=manifest_ref,
                  candidate_id=candidate, command=command, budget=BUDGET,
                  source_before=sources, started_unix_ns=time.time_ns(),
                  collection_subprocess_launched=False, NSYS_started=False,
                  physical_prediction_accuracy_validated=False)
    save(out/'execution-start.json', result)
    process = None
    try:
        with (out/'stdout.log').open('x') as stdout, (out/'stderr.log').open('x') as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       start_new_session=True,
                                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONUNBUFFERED':'1'})
            result.update(collection_subprocess_launched=True, process_group=process.pid)
            result['watchdog'] = wait_bounded(process, out, BUDGET)
        raw_path = out/'collection/raw.json'
        raw = read(raw_path) if raw_path.is_file() else None
        scored = out/'collection/scored-raw.json'
        precollection = out/'collection.precollection.json'
        result.update(raw=receipt(raw_path) if raw is not None else None,
                      scored_raw=receipt(scored) if scored.is_file() else None,
                      precollection=receipt(precollection) if precollection.is_file() else None)
        if (result['watchdog'].get('status') == 'EXITED'
                and result['watchdog'].get('return_code') == 0 and raw is not None
                and raw.get('status') == 'completed' and len(raw.get('request_records', [])) == 57
                and scored.is_file() and precollection.is_file()):
            result['status'] = 'COLLECTED_REQUIRES_ORIGINAL_OBSERVED_ADAPTER_AND_GATE'
    except BaseException as exc:
        result['error'] = repr(exc)
    finally:
        try:
            if process is not None and group_members(process.pid):
                result['exception_cleanup'] = cleanup_group(process, BUDGET['cleanup_grace_seconds'])
                result['status'] = 'FAILED_NO_RETRY'
        except BaseException as exc:
            result.update(status='FAILED_NO_RETRY', cleanup_error=repr(exc))
        try:
            checked(manifest_ref)
            for key in ('plan', 'protocol', 'predictions', 'production'):
                checked(manifest[key])
            _, after = prerequisites(*(manifest[k]['path'] for k in ('plan', 'protocol', 'predictions', 'production')))
            if after != sources:
                raise ValueError('Supervisor implementation drift')
            result.update(source_after=after, final_source_audit='PASS')
        except BaseException as exc:
            result.update(status='FAILED_NO_RETRY', final_source_audit='FAIL', source_error=repr(exc))
        result.update(ended_unix_ns=time.time_ns(), free_bytes_at_end=shutil.disk_usage(out).free)
        save(out/'execution.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='op', required=True)
    command = sub.add_parser('prepare')
    for key in ('plan', 'protocol', 'predictions', 'production', 'output'):
        command.add_argument('--'+key, required=True)
    command = sub.add_parser('execute')
    command.add_argument('--manifest', required=True)
    command.add_argument('--candidate', choices=CANDIDATES, required=True)
    command.add_argument('--output', required=True)
    args = parser.parse_args()
    result = (prepare(args.plan, args.protocol, args.predictions, args.production, args.output)
              if args.op == 'prepare' else execute(args.manifest, args.candidate, args.output))
    print(json.dumps(dict(status=result['status'])))
    return 0 if result['status'] != 'FAILED_NO_RETRY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
