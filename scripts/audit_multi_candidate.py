"""Streaming verification of candidate exclusions, identities and ledgers."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gsdeploy import SystemConfig
from gsdeploy.continuous.contracts import digest
from gsdeploy.continuous.predictor_adapter import normalize_prediction
from experiments.joint_scalability import source_manifest
from experiments.diagnose_continuous_phase1 import assignment


def audit(sources):
    total=dict(runs=0,predictions=0,milp=0,routes=0,candidates=0,refined=0,positive_stages=0,
               beating_initial=0,unknown_events=0,uncertified_candidates=0,assignment_pairs_checked=0)
    contracts={}
    for source in sources:
        source=Path(source)
        assert json.loads((source/'source-manifest.json').read_text())==source_manifest()
        for summary in json.loads((source/'summary.json').read_text())['rows']:
            filename=f"{summary['case']}-{summary['algorithm']}-s{summary['seed']}.json"
            row=json.loads((source/filename).read_text());raw=row['raw']
            c=SystemConfig.load(source/f"{row['case']}-config.json")
            assert row['config_sha256']==digest(c.as_dict())
            contracts.setdefault(row['case'],set()).add(digest([c.as_dict(),row['states']]))
            assert row['predictions']==raw['budget']['full_prediction_calls']
            assert row['predictions']<=row['budget_limits']['predictions']
            assert row['route_evaluations']<=row['budget_limits']['routes']
            assert raw['budget']['predictor_leaf_calls']==row['predictions']*3
            pool=raw.get('candidates',[]);stages=raw.get('stages',[])
            assert len(pool)==len({digest(candidate['deployment']) for candidate in pool})
            for candidate in pool:
                assert normalize_prediction(candidate['grid_prediction'],c.search_policy)==('FEASIBLE',candidate['grid_utility'])
                i=candidate['generation_index'];solve=raw['joint_solves'][i]
                assert solve['excluded_deployments']==[x['deployment'] for x in pool[:i]]
                assert candidate['restricted_grid_certified']==solve['optimality_certified']
                assert candidate['grid_prediction']['safe_memory_mib']<=c.gpu_memory_mib+1e-6
                total['uncertified_candidates']+=not candidate['restricted_grid_certified']
            if pool:
                assert raw['winner']['utility']>=pool[0]['grid_utility']-1e-12
                if 'selected_indices' in raw:assert 0 in raw['selected_indices']
                total['beating_initial']+=raw['winner']['utility']>pool[0]['grid_utility']+1e-12
            if 'generation_counts' in raw:
                assert raw['budget']['full_prediction_calls']==raw['generation_counts']['predictions']+sum(s['counts']['predictions'] for s in stages)
                assert raw['budget']['milp_solve_calls']==raw['generation_counts']['milp']
                assert raw['budget']['outer_evaluations']==sum(s['counts']['outer'] for s in stages)
            for stage in stages:
                assert stage['counts']['milp']==0
                assert stage['continuous_utility']>=stage['grid_utility']-1e-12
                total['positive_stages']+=stage['improvement']>1e-12
                for h in stage['history']:
                    assert h['deployment']==stage['deployment'] if h.get('deployment') else True
                    for task,shares in h['routing_shares'].items():
                        assert abs(sum(shares.values())-1)<2e-15
                        assert set(shares)<=set(stage['deployment'])
                    assert h['milp_solve_calls']==0
            assert sum(not e['cached'] for e in raw['events'])==row['predictions']
            for event in raw['events']:
                state,u=normalize_prediction(event['prediction'],c.search_policy)
                assert (state,u)==(event['state'],event['utility'])
                total['unknown_events']+=state=='UNKNOWN'
                if state=='FEASIBLE':
                    assert abs(u-sum(event['prediction'][k+'_contribution'] for k in 'QTLPME'))<1e-9
            # Rebuild every unique exact route/deployment FIFO mapping. Source
            # and config identities bind the input-feature sequence as well.
            audit=row['routing_audit']
            inputs={digest([e['deployment'],e['routing']]):(e['deployment'],e['routing']) for e in raw['events']}
            inputs.update({digest([s['deployment'],h['routing_qps']]):(s['deployment'],h['routing_qps'])
                           for s in stages for h in s['history']})
            for key,saved in audit['assignments'].items():
                x,r=inputs[key];rebuilt=assignment(c,x,r)
                assert rebuilt==saved
                total['assignment_pairs_checked']+=1
            total['runs']+=1;total['predictions']+=row['predictions'];total['milp']+=row['milp_calls']
            total['routes']+=row['route_evaluations'];total['candidates']+=len(pool);total['refined']+=len(stages)
    assert all(len(cs)==1 for cs in contracts.values())
    total.update(original_sources_unchanged=True,same_contract_per_case=True,exact_whole_deployment_exclusions=True,
        no_deployment_milp_during_refinement=True,initial_incumbent_preserved=True,
        continuous_global_optimality_claimed=False,global_top_k_claimed=False)
    return total


def audit_face(source):
    source=Path(source)
    assert json.loads((source/'source-manifest.json').read_text())==source_manifest()
    c=SystemConfig.load(source/'config.json')
    result=dict(runs=0,predictions=0,milp=0,routes=0,beating_initial=0,positive_stages=0,assignments=0)
    for seed in (11,29,47):
        row=json.loads((source/f'seed{seed}.json').read_text())
        assert row['continuous_U']>=row['grid_U']
        assert row['exceeds_first']==(row['continuous_U']>row['global_first_U']+1e-12)
        assert row['budget']['full_prediction_calls']==sum(s['wrapper_counts']['predictions'] for s in row['joint_solves'])+sum(h['full_prediction_calls'] for h in row['history'])
        assert all(h['milp_solve_calls']==0 for h in row['history'])
        assert row['budget']['milp_solve_calls']==sum(s['wrapper_counts']['milp'] for s in row['joint_solves'])
        known={digest([e['deployment'],e['routing']]):(e['deployment'],e['routing']) for e in row['events']}
        known.update({digest([row['candidate']['deployment'],h['routing_qps']]):(row['candidate']['deployment'],h['routing_qps']) for h in row['history']})
        for key,saved in row['routing_audit']['assignments'].items():
            x,r=known[key];assert assignment(c,x,r)==saved;result['assignments']+=1
        for h in row['history']:
            for task,route in h['routing_qps'].items():
                if task!='text':assert route==row['candidate']['grid_routing'][task]
        result['runs']+=1;result['predictions']+=row['budget']['full_prediction_calls']
        result['milp']+=row['budget']['milp_solve_calls'];result['routes']+=row['budget']['outer_evaluations']
        result['beating_initial']+=row['exceeds_first'];result['positive_stages']+=row['improvement']>1e-12
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--face')
    args=p.parse_args();result=audit(args.sources)
    if args.face:result['independent_one_task_face_check']=audit_face(args.face)
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
