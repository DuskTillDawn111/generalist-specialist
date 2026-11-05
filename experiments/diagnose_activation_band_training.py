"""Training-only repeat and asynchronous-boundary diagnostics; no qualification."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.activation_semantic_bands import virtual_profiles
from experiments.analyze_activation_semantic import output_identity
from experiments.fit_activation_native import load_bundle
from experiments.fit_activation_semantic_bands import export_band_observations
from gsdeploy.forecasting.native_context import categories, counts
from gsdeploy.forecasting.native_training import digest


def diagnose(fit, clocks):
    template = fit['profiles']
    profiles = {p['id']: p for p in template['profiles']}
    virtual, mapping = virtual_profiles(template)
    vprofiles = {p['id']: p for p in virtual}
    names = categories(vprofiles)
    vertices = defaultdict(lambda: defaultdict(set))
    clock_times = [c['time_ns'] for c in clocks]
    rows = []
    for doc, source in zip(fit['normalized_runs'], fit['source_receipts']):
        bundle = load_bundle(Path(source.rsplit(':sha256:', 1)[0]).parent)
        if bundle['source'] != source or bundle['plan']['split'] != 'training':
            raise ValueError('Exact original training-only receipts required')
        raw = {r['request_id']: r for r in bundle['raw']['request_records']}
        for observation in export_band_observations(doc, template):
            target = mapping.get((observation['profile_id'], observation['segment_id']))
            if target is None:
                continue
            profile = vprofiles[target]
            key = profile['family'] + '/' + profile['task']
            for interval in observation['intervals']:
                members = {a: dict(profile_id=mapping[m['profile_id'], m['segment_id']],
                                   segment_id='native_band')
                           for a, m in interval['members'].items()
                           if (m['profile_id'], m['segment_id']) in mapping}
                vertex = tuple(counts(members, vprofiles, names, omit=observation['target_alias']))
                vertices[key][vertex].add(doc['run_id'])
        for request in doc['requests']:
            original = raw[request['request_id']]
            profile = profiles[request['profile_id']]
            response = original['worker_response']
            bands = []
            for span in request['segments']:
                if (profile['id'], span['segment_id']) not in mapping:
                    continue
                contained = [e for e in response['semantic_phase_events']
                             if span['start_ns'] <= e['host_start_ns'] < e['host_end_ns'] <= span['end_ns']]
                clock_index = bisect_right(clock_times, span['start_ns']) - 1
                before = clocks[clock_index] if clock_index >= 0 else None
                bands.append(dict(id=span['segment_id'], duration_ms=(span['end_ns']-span['start_ns'])/1e6,
                    events=[{k: e.get(k) for k in ('component', 'ordinal', 'host_start_ns', 'host_end_ns',
                        'device_elapsed_ms', 'device_start_from_first_event_ms', 'device_end_from_first_event_ms')}
                            for e in contained],
                    preceding_external_clock=before))
            rows.append(dict(run_id=doc['run_id'], mode=bundle['plan']['mode'], request_id=request['request_id'],
                family=profile['family'], task=profile['task'], pool_id=profile['pool_id'], bands=bands,
                output_identity_sha256=digest(output_identity(response)),
                generated_tokens=response.get('generated_tokens'),
                diagnostic_only_future_fields_not_prediction_features=True))
    solo = defaultdict(list)
    outputs = defaultdict(set)
    for row in rows:
        outputs[row['family'], row['task'], row['pool_id']].add(row['output_identity_sha256'])
        if row['mode'] == 'solo':
            for band in row['bands']:
                solo[row['family'], row['task'], row['pool_id'], band['id']].append(dict(
                    run_id=row['run_id'], request_id=row['request_id'], duration_ms=band['duration_ms']))
    contrasts = []
    for key, values in solo.items():
        duration = [v['duration_ms'] for v in values]
        contrasts.append(dict(family=key[0], task=key[1], pool_id=key[2], segment_id=key[3],
            maximum_to_minimum_duration=max(duration)/min(duration), observations=values))
    return dict(schema='gsdeploy.band_training_diagnostic.v1', training_fit_sha256=digest(fit),
        source_receipts=fit['source_receipts'], categories=names,
        under_repeated_vertices={key: [dict(vertex=list(v), independent_run_ids=sorted(runs))
            for v, runs in sorted(group.items()) if len(runs) < 2] for key, group in sorted(vertices.items())},
        solo_band_contrasts=sorted(contrasts, key=lambda r: r['maximum_to_minimum_duration'], reverse=True),
        changed_output_inputs=[dict(family=k[0], task=k[1], pool_id=k[2], identities=len(v))
                               for k, v in outputs.items() if len(v) > 1],
        request_diagnostics=rows, physical_predictor_qualified=False, formal_utility_qualified=False,
        limitations=['host-call boundaries and CUDA event intervals use different clocks; no invented absolute GPU phase boundary',
                     'external 1Hz clocks cannot resolve subsecond causes',
                     'training-only postcompletion diagnosis; eventual output/forward counts are forbidden online features'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--clocks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    clocks = [json.loads(line) for line in args.clocks.open()]
    clocks = [row for row in clocks if 'time_ns' in row and 'sm_mhz' in row]
    report = diagnose(json.loads(args.fit.read_text()), clocks)
    report['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), changed_output_inputs=report['changed_output_inputs'],
        under_repeated_vertices={k: len(v) for k, v in report['under_repeated_vertices'].items()},
        worst_solo_contrasts=report['solo_band_contrasts'][:5])))


if __name__ == '__main__':
    main()
