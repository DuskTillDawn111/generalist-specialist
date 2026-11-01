"""CPU-only paired real-model benchmark for fixed-state cumulative hazard."""
from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import time

from . import cumulative_save_hazard_v14 as old_module
from . import cumulative_save_hazard_v15 as new_module
from .native_collect import receipt
from .profiling import digest

OMIT={'schema','controller_identity','state_sha256','empty_intervals','empty_audit_summary',
      'audit_scope_id','complete_empty_audit_external'}


def numerical(value):
    if isinstance(value,dict):return {k:numerical(v) for k,v in value.items() if k not in OMIT}
    if isinstance(value,list):return [numerical(v) for v in value]
    return value


def run(model_path,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source_refs=[receipt(old_module.__file__),receipt(new_module.__file__),receipt(__file__),receipt(model_path)]
    reference=receipt(model_path)
    frontier=[['text#0','text','tydiqa-new-native-02ea54b33b0492f20de11e0f',['prefill']]]
    old_class=old_module.CumulativeSaveHazardController
    new_class=new_module.CumulativeSaveHazardController
    kwargs=dict(allow_unvalidated_approximation=True,empty_frontier_policy='NO_QUERY_PAUSE')
    origin=0;observation=1_000_000;results=[]
    started=time.perf_counter()
    for n in (100,300,1000):
        before=time.perf_counter();old=old_class(reference,**kwargs);old_init=time.perf_counter()-before
        before=time.perf_counter();new=new_class(reference,**kwargs);new_init=time.perf_counter()-before
        cursor=new.begin_audit_scope('real-model-paired-'+str(n))
        left=old.start_restore('text#0',origin,frontier,'restore',observation_ns=observation)
        right=new.start_restore('text#0',origin,frontier,'restore',observation_ns=observation)
        assert numerical(left)==numerical(right)
        old_seconds=new_seconds=0.;shapes=set();comparisons=1
        for i in range(n):
            for clock,visible in ((observation+2*i+1,[]),(observation+2*i+2,frontier)):
                before=time.perf_counter();left=old.update(left['state'],clock,visible);old_seconds+=time.perf_counter()-before
                before=time.perf_counter();right=new.update(right['state'],clock,visible);new_seconds+=time.perf_counter()-before
                assert numerical(left)==numerical(right)
                shapes.add((len(right['state']),len(right['state']['empty_audit_summary'])))
                comparisons+=1
        before=time.perf_counter();audit=new.export_audit_since(cursor);export_seconds=time.perf_counter()-before
        intervals=[{k:r[k] for k in ('start_ns','end_ns','policy')} for r in audit['events']]
        assert intervals==left['state']['empty_intervals']
        assert len(intervals)==n and len(shapes)==1
        assert right['state']['empty_audit_summary']['count']==n
        assert audit['total_empty_interval_ns']==n
        assert left['state']['terminal_save_candidate_ns'] is None
        assert right['budget_redrawn'] is False
        previous_hash=audit['ledger_hash_before']
        for event in audit['events']:
            assert event['ledger_previous_hash']==previous_hash
            assert event['ledger_entry_hash']==digest({k:v for k,v in event.items() if k!='ledger_entry_hash'})
            previous_hash=event['ledger_entry_hash']
        assert previous_hash==audit['ledger_hash_after']
        (output/f'audit-{n}.json').write_text(json.dumps(audit,sort_keys=True,indent=2)+'\n')
        results.append(dict(empty_intervals=n,update_calls=2*n,stepwise_exact_comparisons=comparisons,
            v14_update_seconds=old_seconds,v15_update_seconds=new_seconds,speedup=old_seconds/new_seconds,
            v14_construction_seconds=old_init,v15_construction_seconds=new_init,
            v15_audit_export_seconds=export_seconds,
            v14_state_json_bytes=len(json.dumps(left['state'],sort_keys=True).encode()),
            v15_state_json_bytes=len(json.dumps(right['state'],sort_keys=True).encode()),
            v15_state_field_shapes=sorted(shapes),numeric_results_identical=True,
            complete_unmerged_intervals_identical=True,audit_hashchain_verified=True,
            audit=receipt(output/f'audit-{n}.json'),final_status=right['status'],
            final_origin_restore_ns=right['origin_restore_ns'],budget_redrawn=right['budget_redrawn']))
    profiles=[]
    for version,cls in (('v14',old_class),('v15',new_class)):
        controller=cls(reference,**kwargs)
        if version=='v15':controller.begin_audit_scope('profile-1000')
        state=controller.start_restore('text#0',origin,frontier,'restore',observation_ns=observation)
        profiler=cProfile.Profile();profiler.enable()
        for i in range(1000):
            state=controller.update(state['state'],observation+2*i+1,[])
            state=controller.update(state['state'],observation+2*i+2,frontier)
        profiler.disable();profiler.dump_stats(str(output/f'{version}-1000.prof'))
        stats=pstats.Stats(profiler)
        deepcopy_seconds=sum(value[3] for key,value in stats.stats.items() if key[2]=='deepcopy')
        profiles.append(dict(version=version,empty_intervals=1000,update_calls=2000,
            total_calls=stats.total_calls,primitive_calls=stats.prim_calls,
            seconds=stats.total_tt,deepcopy_cumulative_seconds=deepcopy_seconds,
            raw_profile=receipt(output/f'{version}-1000.prof')))
    assert all(receipt(r['path'])==r for r in source_refs)
    report=dict(schema='gsdeploy.cumulative_hazard_history_performance.v15',
        status='STEPWISE_IDENTICAL_FIXED_STATE_WITH_COMPLETE_EXTERNAL_AUDIT',sources=source_refs,
        case=dict(holder='text#0',restore_ns=origin,observation_ns=observation,frontier=frontier,
            update_sequence='For i=0..n-1: explicit empty at observation+2*i+1, original frontier at observation+2*i+2'),
        results=results,profiles=profiles,total_seconds=time.perf_counter()-started,
        update_timings_exclude_construction_comparison_and_export=True,
        timing_method='perf_counter around each controller.update; interleaved v14/v15; no profiler during paired timings',
        source_closure_unchanged=True,all_stepwise_numeric_results_identical=True,
        complete_append_only_audit_preserved=True,sealed_prediction_state_history_list_removed=True,
        audit_export_cost_is_linear_in_exported_events=True,
        independent_prediction_accuracy_established=False,GPU_started=False)
    (output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(report=receipt(output/'report.json'),results=results,profiles=profiles),indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.model,args.output)
