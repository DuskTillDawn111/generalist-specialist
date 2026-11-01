"""Snapshot current sources and reuse only the old public validation contract.

No validation outcomes are read, no model/protocol is sealed and no GPU runs.
Four routes, offsets, inputs, normalizers and thresholds remain unchanged;
request/run identities and the actual collector/context bindings are fresh.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import shutil

from . import known_input_native_flow_v11 as flow
from .known_input_campaign_seal_v5 import public_inputs
from .native_collect import context_identity, receipt, save
from .native_execution_projection_v5 import native_execution_projection
from .profiling import digest
from .prospective_flow_contract_v11 import bind_public


def prepare(public_plan_path, output, *, run_id):
    if not isinstance(run_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',run_id):
        raise ValueError('Explicit simple fresh run identity required')
    source_plan_ref = receipt(public_plan_path)
    original = json.loads(Path(source_plan_ref['path']).read_text())
    flow.validate_plan(original)
    context = json.loads(flow.checked(original['context']).read_text())
    original_native = native_execution_projection(context)
    sources = {Path(s['original_path']).resolve() for s in context['execution_sources']}
    for source in context['execution_sources']:
        flow.checked(source['frozen'])
        if receipt(source['original_path'])['sha256'] != source['frozen']['sha256']:
            raise ValueError('Current execution source differs from original trained context')
    sources.update(flow.collection_source_paths())
    out = Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    snapshots = []
    for i,path in enumerate(sorted(sources)):
        destination = out/'source-snapshot'/str(i)/path.name
        destination.parent.mkdir(parents=True)
        before = receipt(path)
        shutil.copyfile(path,destination)
        if receipt(path) != before or receipt(destination)['sha256'] != before['sha256']:
            raise ValueError('Execution source changed during fresh snapshot')
        snapshots.append(dict(original_path=str(path),frozen=receipt(destination)))
    context.update(execution_sources=snapshots,sources=[s['frozen'] for s in snapshots],
        training_only=False,collection_schema='gsdeploy.fresh_known_input_validation_context.v11',
        parent_context=original['context'],validation_scope=original['validation_scope'],
        actual_collector=receipt(flow.__file__))
    save(out/'context.json',context)
    native = native_execution_projection(context)
    if native['projection_sha256'] != original_native['projection_sha256']:
        raise ValueError('Fresh collector snapshot changed original native execution projection')
    full = context_identity(context)
    plan = copy.deepcopy(original)
    for key in ('plan_sha256','base_model_bundle','primary_host_supply_scenario',
                'sensitivity_host_supply_scenario','host_scenario_selected_after_observations',
                'protocol_generator'):
        plan.pop(key,None)
    ids = {r['request_id']:run_id+':req'+str(i).zfill(3) for i,r in enumerate(plan['trace'])}
    if len(ids) != len(plan['trace']) or set(ids)&set(ids.values()):
        raise ValueError('Fresh unique request IDs required')
    for row in plan['trace']:
        row['request_id'] = ids[row['request_id']]
    for candidate in plan['candidates']:
        candidate['routes'] = {ids[rid]:alias for rid,alias in candidate['routes'].items()}
        candidate['execution_context_sha256'] = full
    plan.update(run_id=run_id,context=receipt(out/'context.json'),execution_context_sha256=full,
        runner_source=receipt(flow.__file__),collector_sources=[receipt(p) for p in flow.collection_source_paths()],
        public_contract_source=source_plan_ref,preparation_source=receipt(__file__),
        trace_sha256=digest(plan['trace']),GPU_started=False,
        scope='Prepared fresh complete public cohort; not a frozen model, protocol, prediction or accuracy result')
    plan['plan_sha256'] = digest(plan)
    flow.validate_plan(plan)
    save(out/'plan.json',plan)
    defaults = {k:plan[k] for k in ('aliases','latency_reference_ms','memory_budget_mib',
                                  'energy_reference_watts','memory_safe_factor')}
    defaults['route_initial_with_continuation_policy'] = False
    jobs = []
    for candidate in plan['candidates']:
        cid = candidate['id'];public,routes = public_inputs(plan,candidate)
        bind_public(plan,cid,public['forecast_inputs'],routes,defaults,plan['weights'])
        save(out/('public-'+cid+'.json'),public);save(out/('routes-'+cid+'.json'),routes)
        jobs.append(dict(run_id=run_id,candidate_id=cid,
                         public_inputs_path=str(out/('public-'+cid+'.json'))))
    save(out/'jobs.json',jobs)
    for source in snapshots:
        if receipt(source['original_path'])['sha256'] != source['frozen']['sha256']:
            raise ValueError('Current execution source changed after preparation')
    if receipt(source_plan_ref['path']) != source_plan_ref:
        raise ValueError('Original public contract changed')
    report = dict(schema='gsdeploy.fresh_public_validation_preparation.v11',
        status='PREPARED_PUBLIC_ONLY_NOT_SEALED',plan=receipt(out/'plan.json'),jobs=receipt(out/'jobs.json'),
        source_public_plan=source_plan_ref,original_native=original_native,fresh_native=native,
        full_context_equality_claimed=False,run_id=run_id,request_ids=list(ids.values()),
        candidate_count=len(jobs),requests_per_candidate=len(plan['trace']),
        public_trace_and_routes_preserved_up_to_ID_mapping=True,original_thresholds_preserved=True,
        complete_training_index_disjointness_not_yet_verified=True,
        model_sealed=False,protocol_sealed=False,predictions_sealed=False,
        validation_outcomes_read=False,GPU_started=False)
    save(out/'preparation-report.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('public-plan','output','run-id'):
        parser.add_argument('--'+field,required=True)
    args = parser.parse_args()
    report = prepare(args.public_plan,args.output,run_id=args.run_id)
    print(json.dumps({k:report[k] for k in ('status','candidate_count','requests_per_candidate','GPU_started')}))


if __name__ == '__main__':
    main()
