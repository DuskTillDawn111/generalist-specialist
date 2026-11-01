"""Descriptive host service interval analysis; never imports a GPU API."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import gc
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/queue-five-term-study-20261023'


def receipt(path):
    path = Path(path).resolve()
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def avg(values):
    return statistics.fmean(values) if values else None


def load_compact(path):
    """Read one pretty-JSON request at a time, discarding large forecasts."""
    path=Path(path)
    before=path.stat()
    result={}
    keep = ('request_id', 'task', 'model', 'pool_id', 'sample_id', 'scheduled_time_ns',
            'scheduled_offset_seconds', 'actual_submit_time_ns', 'actual_submit_unix_ns',
            'completion_time_ns', 'worker_queue_wait_ms', 'e2e_latency_ms')
    result['request_records'] = []
    section=None
    buf=[]
    with path.open() as stream:
        for line in stream:
            if section in ('replica_to_family','measurement_window'):
                buf.append(line)
                if line.startswith('  },'):
                    result[section]=json.loads(''.join(buf).rstrip().rstrip(','))
                    section=None;buf=[]
                continue
            if section=='request_records':
                if line.startswith('  ],'):
                    section=None
                    continue
                if line.startswith('    {'):
                    buf=[line]
                elif buf:
                    buf.append(line)
                if buf and (line.startswith('    },') or line.rstrip()=='    }'):
                    row=json.loads(''.join(buf).rstrip().rstrip(','))
                    small = {k:row[k] for k in keep}
                    small['routing_decision'] = {'decision_time_ns':row['routing_decision']['decision_time_ns']}
                    small['worker_response'] = {k:row['worker_response'].get(k) for k in
                                                ('end_ns','start_ns','service_ms','input_tokens','generated_tokens')}
                    result['request_records'].append(small)
                    buf=[]
                continue
            for key in ('replica_to_family','measurement_window'):
                if line.startswith('  "'+key+'": {'):
                    section=key;buf=['{\n'];break
            if line.startswith('  "request_records": ['):
                section='request_records'
    after=path.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    assert result['request_records'] and result.get('measurement_window') and result.get('replica_to_family')
    return result


def intervals(raw):
    result = []
    for r in raw['request_records']:
        worker = r['worker_response']
        # end_ns is worker wall time. service_ms is the handler's measured wall
        # service duration; start_ns is wrapper start and is not used here.
        end = worker['end_ns']
        start = end - round(worker['service_ms'] * 1e6)
        result.append(dict(request_id=r['request_id'], alias=r['model'],
                           family=raw['replica_to_family'][r['model']], task=r['task'],
                           pool_id=r['pool_id'], sample_id=r['sample_id'],
                           start_ns=start, end_ns=end, service_ms=worker['service_ms'],
                           wrapper_start_ns=worker['start_ns'],
                           input_tokens=worker.get('input_tokens'), generated_tokens=worker.get('generated_tokens')))
    return result


def intersection(a, b):
    return max(0, min(a['end_ns'], b['end_ns']) - max(a['start_ns'], b['start_ns']))


def partition(target, all_intervals):
    overlapping = [x for x in all_intervals
                   if x['request_id'] != target['request_id'] and intersection(target, x) > 0]
    cuts = sorted({target['start_ns'], target['end_ns'],
                   *[max(target['start_ns'], x['start_ns']) for x in overlapping],
                   *[min(target['end_ns'], x['end_ns']) for x in overlapping]})
    result = []
    for start, end in zip(cuts, cuts[1:]):
        active = [x for x in overlapping if x['start_ns'] < end and x['end_ns'] > start]
        text = [x for x in active if x['family'] == 'text']
        ocr = [x for x in active if x['family'] == 'ocr']
        own_g = [x for x in active if x['alias'] == target['alias']]
        assert 0 <= len(text) <= 2
        result.append(dict(start_ns=start, end_ns=end, duration_ms=(end - start) / 1e6,
                           specialist_active_count=len(text), specialist_aliases=sorted(x['alias'] for x in text),
                           specialist_request_ids=sorted(x['request_id'] for x in text),
                           ocr_active=bool(ocr), ocr_request_ids=sorted(x['request_id'] for x in ocr),
                           same_G2_worker_active_tasks=sorted(x['task'] for x in own_g),
                           all_other_active={x['alias']: x['request_id'] for x in active}))
    assert abs(sum(s['duration_ms'] for s in result) - target['service_ms']) < 1e-6
    return result


def grouped_spill_summary(rows):
    result = dict(n=len(rows))
    if not rows:
        return result
    for key in ('on_service_ms', 'off_matched_service_ms', 'on_e2e_ms', 'off_e2e_ms',
                'e2e_change_ms', 'on_worker_queue_ms', 'off_worker_queue_ms', 'input_tokens',
                'generated_tokens', 'later_arrival_count', 'later_specialist_service_start_count'):
        result[key + '_mean'] = avg([x[key] for x in rows if x[key] is not None])
    total = sum(x['on_service_ms'] for x in rows)
    result['G2_service_total_ms'] = total
    result['time_by_specialist_active_count'] = {
        str(n): dict(milliseconds=sum(s['duration_ms'] for x in rows for s in x['segments']
                                     if s['specialist_active_count'] == n)) for n in range(3)}
    for value in result['time_by_specialist_active_count'].values():
        value['fraction'] = value['milliseconds'] / total
    ocr = sum(s['duration_ms'] for x in rows for s in x['segments'] if s['ocr_active'])
    own = sum(s['duration_ms'] for x in rows for s in x['segments'] if s['same_G2_worker_active_tasks'])
    result['OCR_overlap_ms'] = ocr
    result['OCR_overlap_fraction'] = ocr / total
    result['same_G2_worker_overlap_ms'] = own
    result['joint_context_time_ms'] = dict(Counter())
    joint = defaultdict(float)
    for x in rows:
        for s in x['segments']:
            key = f"text_active={s['specialist_active_count']};ocr={int(s['ocr_active'])};G2own={','.join(s['same_G2_worker_active_tasks']) or 'none'}"
            joint[key] += s['duration_ms']
    result['joint_context_time_ms'] = dict(joint)
    result['pool_composition'] = dict(Counter(x['pool_id'] for x in rows))
    return result


def comparison_summary(rows):
    result = dict(n=len(rows))
    for key in ('on_service_ms', 'off_service_ms', 'service_change_ms', 'on_G2_spill_overlap_fraction',
                'off_generalist_background_overlap_fraction', 'on_worker_queue_ms', 'off_worker_queue_ms',
                'on_e2e_ms', 'off_e2e_ms', 'e2e_change_ms'):
        result[key + '_mean'] = avg([x[key] for x in rows])
    return result


def timeline(spill, rowmap, intmap, all_intervals, raw):
    row = rowmap[spill['request_id']]
    start, end = spill['start_ns'], spill['end_ns']
    origin = raw['measurement_window']['start_unix_ns']
    # Scheduled wall time uses each request's paired Unix/perf submit stamps.
    scheduled_wall = lambda r: r['actual_submit_unix_ns'] - (r['actual_submit_time_ns'] - r['scheduled_time_ns'])
    decision_wall = row['actual_submit_unix_ns'] - (row['actual_submit_time_ns'] - row['routing_decision']['decision_time_ns'])
    events = [dict(event='spill_decision', request_id=row['request_id'], alias=row['model'], task=row['task'], wall_ns=decision_wall),
              dict(event='spill_service_start', request_id=row['request_id'], alias=row['model'], task=row['task'], wall_ns=start),
              dict(event='spill_service_end', request_id=row['request_id'], alias=row['model'], task=row['task'], wall_ns=end)]
    related_ids = set()
    for x in all_intervals:
        if x['request_id'] == spill['request_id']:
            continue
        if intersection(spill, x) > 0:
            related_ids.add(x['request_id'])
            if x['start_ns'] < start:
                events.append(dict(event='already_active_at_spill_start', request_id=x['request_id'], alias=x['alias'], task=x['task'], wall_ns=start,
                                   actual_service_start_ns=x['start_ns']))
            else:
                events.append(dict(event='other_service_start', request_id=x['request_id'], alias=x['alias'], task=x['task'], wall_ns=x['start_ns']))
            events.append(dict(event='other_service_end', request_id=x['request_id'], alias=x['alias'], task=x['task'], wall_ns=x['end_ns']))
    for r in rowmap.values():
        t = scheduled_wall(r)
        if decision_wall < t < end:
            related_ids.add(r['request_id'])
            events.append(dict(event='later_scheduled_arrival', request_id=r['request_id'], alias=r['model'], task=r['task'], wall_ns=t))
    for e in events:
        e['measurement_offset_seconds'] = (e['wall_ns'] - origin) / 1e9
        e['since_spill_service_start_ms'] = (e['wall_ns'] - start) / 1e6
    events.sort(key=lambda e: (e['wall_ns'], e['event'], e['request_id']))
    records = [intmap[i] for i in related_ids | {row['request_id']}]
    return dict(request_id=spill['request_id'], pool_id=spill['pool_id'],
                scope='Handler wall intervals and scheduled/host timestamps only; not CUDA kernel or stream events.',
                spill_service_start_ns=start, spill_service_end_ns=end, events=events, intervals=records)


def svg_timeline(case, path):
    width, label_w, height = 1200, 180, 250
    origin = case['spill_service_start_ns']
    limit = case['spill_service_end_ns']
    # Show the complete spill plus related service tails, capped at a5s tail.
    lo = min([origin - 100_000_000] + [x['start_ns'] for x in case['intervals']])
    hi = max(limit, min(max(x['end_ns'] for x in case['intervals']), limit + 5_000_000_000))
    aliases = sorted({x['alias'] for x in case['intervals']})
    height = 95 + len(aliases) * 42
    scale = (width - label_w - 30) / (hi - lo)
    xcoord = lambda t: label_w + (t - lo) * scale
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="16" y="23" font-family="sans-serif" font-size="16">Host handler service intervals; GPU kernel overlap UNKNOWN</text>',
             f'<text x="16" y="44" font-family="sans-serif" font-size="13">{case["request_id"]}; zero = G2 spill service start; duration ms</text>']
    for k, alias in enumerate(aliases):
        y = 67 + k * 42
        parts.append(f'<text x="10" y="{y + 18}" font-family="sans-serif" font-size="13">{alias}</text>')
        for r in sorted((r for r in case['intervals'] if r['alias'] == alias), key=lambda r:r['start_ns']):
            start, end = max(lo, r['start_ns']), min(hi, r['end_ns'])
            if end <= start:
                continue
            color = '#dc6a36' if r['request_id'] == case['request_id'] else '#497ca5'
            parts.append(f'<rect x="{xcoord(start):.2f}" y="{y}" width="{max(.6,(end-start)*scale):.2f}" height="24" fill="{color}"><title>{r["request_id"]} {r["task"]}: {(r["start_ns"]-origin)/1e6:.2f}..{(r["end_ns"]-origin)/1e6:.2f} ms</title></rect>')
    for t, label in ((origin, '0'), (limit, f'{(limit-origin)/1e6:.1f}')):
        parts.append(f'<line x1="{xcoord(t):.2f}" x2="{xcoord(t):.2f}" y1="55" y2="{height-25}" stroke="#555" stroke-dasharray="3 3"/>')
        parts.append(f'<text x="{xcoord(t)+3:.2f}" y="{height-8}" font-family="sans-serif" font-size="12">{label}ms</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts) + '\n')


def main():
    out = BASE / 'gpu-contention-forensics-v1'
    if out.exists():
        raise SystemExit('Refusing to overwrite ' + str(out))
    terminal_path = BASE / 'independent-runtime-audit-stream-v1/terminal.json'
    terminal = json.loads(terminal_path.read_bytes())
    assert terminal['verified']
    observed = {o['label']: o for o in terminal['observations']}
    results, segment_csv, spill_csv, specialist_csv, stratum_csv, pool_csv = [], [], [], [], [], []
    cases = []
    for seed in (401, 402):
        receipts, raws = [], []
        for role in ('coupled_spill', 'matched_no_spill'):
            ref = observed[f'{role}-seed{seed}']['raw']
            assert receipt(ref['path']) == ref
            receipts.append(ref)
            raws.append(load_compact(ref['path']))
        onraw, offraw = raws
        onrows, offrows = [{r['request_id']: r for r in raw['request_records']} for raw in raws]
        oni, offi = [intervals(raw) for raw in raws]
        onint, offint = [{r['request_id']: r for r in rows} for rows in (oni, offi)]
        assert set(onrows) == set(offrows)
        for rid in onrows:
            assert all(onrows[rid][k] == offrows[rid][k]
                       for k in ('task', 'pool_id', 'sample_id', 'scheduled_offset_seconds'))
        spills = [r for r in oni if r['task'] == 'text' and r['family'] == 'generalist_2']
        spill_ids = {r['request_id'] for r in spills}
        g_bg_off = [r for r in offi if r['family'].startswith('generalist')]
        analysis = []
        for r in spills:
            rid = r['request_id']
            a, b = onrows[rid], offrows[rid]
            segments = partition(r, oni)
            later = [x for x in onrows.values() if a['routing_decision']['decision_time_ns'] < x['scheduled_time_ns'] < a['completion_time_ns']]
            future_starts = [x for x in oni if x['family'] == 'text' and r['start_ns'] < x['start_ns'] < r['end_ns']]
            successors = [x for x in oni if x['family'] == 'text' and x['start_ns'] >= r['end_ns'] and x['start_ns'] < r['end_ns'] + 5_000_000_000]
            item = dict(request_id=rid, pool_id=r['pool_id'], sample_id=r['sample_id'],
                        outcome_group='faster' if a['e2e_latency_ms'] < b['e2e_latency_ms'] else 'slower_or_equal',
                        on_alias=a['model'], off_alias=b['model'], on_service_ms=r['service_ms'],
                        off_matched_service_ms=offint[rid]['service_ms'],
                        on_e2e_ms=a['e2e_latency_ms'], off_e2e_ms=b['e2e_latency_ms'],
                        e2e_change_ms=a['e2e_latency_ms']-b['e2e_latency_ms'],
                        on_worker_queue_ms=a['worker_queue_wait_ms'], off_worker_queue_ms=b['worker_queue_wait_ms'],
                        start_ns=r['start_ns'], end_ns=r['end_ns'], input_tokens=r['input_tokens'],
                        generated_tokens=r['generated_tokens'], off_input_tokens=offint[rid]['input_tokens'],
                        off_generated_tokens=offint[rid]['generated_tokens'],
                        later_arrival_count=len(later), later_text_arrival_count=sum(x['task']=='text' for x in later),
                        later_specialist_service_start_count=len(future_starts),
                        specialist_starts_within5s_after_spill_end=len(successors),
                        later_specialist_starts_mean_service_change_ms=avg([x['service_ms']-offint[x['request_id']]['service_ms'] for x in future_starts]),
                        post5s_specialist_mean_service_change_ms=avg([x['service_ms']-offint[x['request_id']]['service_ms'] for x in successors]),
                        segments=segments)
            analysis.append(item)
            times = {n:sum(s['duration_ms'] for s in segments if s['specialist_active_count']==n) for n in range(3)}
            spill_csv.append(dict(seed=seed, **{k:v for k,v in item.items() if k!='segments'},
                                  **{f'text_active{n}_ms':times[n] for n in range(3)},
                                  **{f'text_active{n}_fraction':times[n]/r['service_ms'] for n in range(3)}))
            for s in segments:
                segment_csv.append(dict(seed=seed, spill_request_id=rid, outcome_group=item['outcome_group'],
                                        start_ns=s['start_ns'], end_ns=s['end_ns'], duration_ms=s['duration_ms'],
                                        specialist_active_count=s['specialist_active_count'], ocr_active=s['ocr_active'],
                                        specialist_request_ids=';'.join(s['specialist_request_ids']),
                                        same_G2_worker_active_tasks=';'.join(s['same_G2_worker_active_tasks'])))
        specialist_rows = []
        for r in oni:
            if r['family'] != 'text' or r['request_id'] in spill_ids:
                continue
            rid = r['request_id']
            other = offint[rid]
            overlap_ns = sum(intersection(r, x) for x in spills)
            bg_off_ns = sum(intersection(other, x) for x in g_bg_off)
            # G2 serial execution makes its intervals disjoint, hence no union double counting.
            assert overlap_ns <= r['end_ns']-r['start_ns']+1
            v = dict(seed=seed, request_id=rid, pool_id=r['pool_id'], sample_id=r['sample_id'],
                     on_alias=r['alias'], off_alias=other['alias'], same_alias=r['alias']==other['alias'],
                     overlaps_on_G2_spill=overlap_ns>0, on_G2_spill_overlap_ms=overlap_ns/1e6,
                     on_G2_spill_overlap_fraction=overlap_ns/1e6/r['service_ms'],
                     off_generalist_background_overlap_fraction=bg_off_ns/1e6/other['service_ms'],
                     on_service_ms=r['service_ms'], off_service_ms=other['service_ms'],
                     service_change_ms=r['service_ms']-other['service_ms'],
                     on_worker_queue_ms=onrows[rid]['worker_queue_wait_ms'], off_worker_queue_ms=offrows[rid]['worker_queue_wait_ms'],
                     on_e2e_ms=onrows[rid]['e2e_latency_ms'], off_e2e_ms=offrows[rid]['e2e_latency_ms'],
                     e2e_change_ms=onrows[rid]['e2e_latency_ms']-offrows[rid]['e2e_latency_ms'])
            specialist_rows.append(v)
        specialist_csv.extend(specialist_rows)
        specialist_summary = {}
        for same_alias in (False, True):
            for overlap in (False, True):
                group=[r for r in specialist_rows if (not same_alias or r['same_alias']) and r['overlaps_on_G2_spill']==overlap]
                specialist_summary[f'{"same_alias_only" if same_alias else "all_aliases"}_overlap_{int(overlap)}']=comparison_summary(group)
        strata = defaultdict(lambda: {True:[],False:[]})
        for r in specialist_rows:
            if r['same_alias']:
                strata[(r['pool_id'],r['on_alias'])][r['overlaps_on_G2_spill']].append(r)
        strata_results=[]
        for (pool, alias), groups in sorted(strata.items()):
            yes, no = groups[True],groups[False]
            if not yes or not no:
                continue
            item=dict(seed=seed,pool_id=pool,alias=alias,overlap_n=len(yes),no_overlap_n=len(no),
                      overlap_on_service_mean_ms=avg([r['on_service_ms'] for r in yes]),
                      no_overlap_on_service_mean_ms=avg([r['on_service_ms'] for r in no]),
                      overlap_matched_delta_ms=avg([r['service_change_ms'] for r in yes]),
                      no_overlap_matched_delta_ms=avg([r['service_change_ms'] for r in no]),
                      weight=min(len(yes),len(no)))
            item['difference_of_matched_changes_ms']=item['overlap_matched_delta_ms']-item['no_overlap_matched_delta_ms']
            item['on_context_service_difference_ms']=item['overlap_on_service_mean_ms']-item['no_overlap_on_service_mean_ms']
            strata_results.append(item)
        stratum_csv.extend(strata_results)
        weight=sum(x['weight'] for x in strata_results)
        controlled=dict(common_pool_alias_strata=len(strata_results),sum_min_group_count_weights=weight,
                        overlap_requests_covered=sum(x['overlap_n'] for x in strata_results),
                        no_overlap_requests_covered=sum(x['no_overlap_n'] for x in strata_results),
                        weighted_difference_of_matched_service_changes_ms=sum(x['weight']*x['difference_of_matched_changes_ms'] for x in strata_results)/weight if weight else None,
                        weighted_on_context_service_difference_ms=sum(x['weight']*x['on_context_service_difference_ms'] for x in strata_results)/weight if weight else None,
                        scope='Same pool_id and same alias in BOTH arms; observed overlap vs no-overlap standardized with min subgroup counts. Descriptive difference-of-changes, not causal difference-in-differences identification.')
        pool_rows=[]
        for pool in sorted({r['pool_id'] for r in analysis}):
            group=[r for r in analysis if r['pool_id']==pool]
            for outcome in ('faster','slower_or_equal'):
                rows=[r for r in group if r['outcome_group']==outcome]
                if rows:
                    pool_rows.append(dict(seed=seed,pool_id=pool,outcome_group=outcome, n=len(rows),
                                          on_G2_service_ms=avg([r['on_service_ms'] for r in rows]),
                                          off_same_input_specialist_service_ms=avg([r['off_matched_service_ms'] for r in rows]),
                                          input_tokens=avg([r['input_tokens'] for r in rows]),generated_tokens=avg([r['generated_tokens'] for r in rows]),
                                          e2e_change_ms=avg([r['e2e_change_ms'] for r in rows])))
        pool_csv.extend(pool_rows)
        for group_name, choose in [('good',min),('bad',max)]:
            chosen=choose(analysis,key=lambda r:r['e2e_change_ms'])
            case=timeline(chosen,onrows,onint,oni,onraw)
            case.update(seed=seed,outcome=group_name,paired_outcome={k:chosen[k] for k in ('on_service_ms','off_matched_service_ms','on_e2e_ms','off_e2e_ms','e2e_change_ms','on_worker_queue_ms','off_worker_queue_ms')})
            cases.append(case)
        results.append(dict(seed=seed,raw_receipts=receipts,
                            interval_checks=dict(measurement_rows_on=len(oni),measurement_rows_off=len(offi),
                                                 nonnegative_handler_wrapper_gap_on=all(r['start_ns']>=r['wrapper_start_ns'] for r in oni),
                                                 segment_duration_closed=True,
                                                 same_G2_worker_overlap_total_ms=sum(s['duration_ms'] for x in analysis for s in x['segments'] if s['same_G2_worker_active_tasks'])),
                            spill_context_summaries={g:grouped_spill_summary([r for r in analysis if g=='all' or r['outcome_group']==g]) for g in ('all','faster','slower_or_equal')},
                            spill_requests=analysis,specialist_overlap_summaries=specialist_summary,
                            alias_change_confounding=dict(remaining_specialist_requests=len(specialist_rows),
                                                         on_off_alias_changed=sum(not r['same_alias'] for r in specialist_rows),
                                                         on_off_alias_same=sum(r['same_alias'] for r in specialist_rows)),
                            pool_alias_controlled_overlap_association=controlled,
                            pool_outcome_summary=pool_rows))
    out.mkdir()
    for name, rows in [('spill-requests',spill_csv),('spill-context-segments',segment_csv),
                       ('specialist-matched-requests',specialist_csv),('pool-alias-controlled-strata',stratum_csv),
                       ('spill-pool-outcomes',pool_csv)]:
        with (out/(name+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for case in cases:
        name=f"seed{case['seed']}-{case['outcome']}-{case['request_id'].replace(':','-')}"
        (out/(name+'-timeline.json')).write_text(json.dumps(case,indent=2,ensure_ascii=False)+'\n')
    result=dict(schema='gsdeploy.gpu_contention_host_forensics.v1',source=receipt(__file__),audit_terminal=receipt(terminal_path),
                results=results,case_timelines=[dict(seed=c['seed'],outcome=c['outcome'],request_id=c['request_id'],paired_outcome=c['paired_outcome']) for c in cases],
                interpretation=dict(interval='Unix-wall interval [worker end_ns - round(service_ms*1e6), worker end_ns), per user requirement. Wrapper start_ns excluded. Handler may do post-timer work, so inferred bounds remain a proxy unless handler timing boundaries are independently instrumented.',
                                    overlap='Simultaneous host handler service intervals of separate processes, not measured CUDA kernel/stream overlap. No GPU resource arbitration, SM occupancy, bandwidth contention or kernel causality is identified.',
                                    groups='Good/bad based on matched cross-run E2E on-off; slow requests have more opportunity for arrivals and overlap. Input, output-token counts, alias changes, run order, system history and selection remain potential confounders.',
                                    controls='Pool+alias strata reduce observed input/alias confounding, not unobserved load/temperature/state or reverse duration selection.',
                                    aftereffects='Subsequent starts within5s counted per spill can duplicate requests across spills; post-period changes are descriptive only. Nonoverlap does not mean no earlier spill influence.'),
                GPU_started=False,frozen_sources_modified=False)
    (out/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out),results=[{k:v for k,v in r.items() if k not in ('spill_requests','pool_outcome_summary')} for r in results]),ensure_ascii=False))


if __name__=='__main__':
    main()
