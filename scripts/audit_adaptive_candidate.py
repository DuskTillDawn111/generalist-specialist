"""Streaming verification of budgets, certified screening and actual FIFO maps."""
import argparse
import gc
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gsdeploy import SystemConfig
from gsdeploy.continuous import ImmutableContinuousProblem,SessionPredictorOracle
from gsdeploy.continuous.candidate_bounds import ContinuousDeploymentBounds
from gsdeploy.continuous.contracts import digest
from gsdeploy.continuous.predictor_adapter import normalize_prediction
from experiments.analyze_candidate_potential import load_compact
from experiments.diagnose_continuous_phase1 import assignment
from experiments.joint_scalability import source_manifest


def mapping_entries(path):
    """Decode one mapping at a time from the preserved indented JSON."""
    active=False;block=[]
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith('    "assignments": {'):
                active=True;continue
            if not active:continue
            if line.startswith('    }'):break
            if line.startswith('      "') and not block:block=[line]
            elif block:block.append(line)
            if block and line.startswith('      }'):
                parsed=json.loads('{'+''.join(block).strip().rstrip(',')+'}')
                yield next(iter(parsed.items()))
                block=[]


def audit(sources):
    current=source_manifest();total=dict(runs=0,predictions=0,milp=0,routes=0,candidates=0,
        safe_pruned=0,heuristic_paused=0,assignments=0,checked_utility_events=0,
        higher_than_joint=0,unknown_events=0,uncertified_candidates=0)
    bound_cache={};contracts={}
    for source in sources:
        source=Path(source);manifest=json.loads((source/'source-manifest.json').read_text())
        assert current==manifest,'Run optimizer code changed'
        for meta in json.loads((source/'summary.json').read_text())['rows']:
            path=source/meta['file'];run=load_compact(path);raw=run['raw']
            c=SystemConfig.load(source/f"{run['case']}-config.json")
            assert digest(c.as_dict())==run['config_sha256']
            contracts.setdefault(run['case'],set()).add(digest([c.as_dict(),run['states']]))
            assert run['predictions']==raw['budget']['full_prediction_calls']<=run['budget_limits']['predictions']
            assert run['routes']==raw['budget']['outer_evaluations']<=run['budget_limits']['routes']
            assert run['milp']==raw['budget']['milp_solve_calls']
            assert raw['budget']['predictor_leaf_calls']==3*run['predictions']
            candidates=raw.get('candidates',[]);stages=raw.get('stages',[])
            assert len(candidates)==len({digest(c['deployment']) for c in candidates})
            for i,candidate in enumerate(candidates):
                assert raw['joint_solves'][i]['excluded_deployments']==[c['deployment'] for c in candidates[:i]]
                assert normalize_prediction(candidate['grid_prediction'],c.search_policy)==('FEASIBLE',candidate['grid_utility'])
                total['uncertified_candidates']+=not candidate['restricted_grid_certified']
            if candidates:
                assert raw['winner']['utility']>=candidates[0]['grid_utility']-1e-12
                total['higher_than_joint']+=raw['winner']['utility']>candidates[0]['grid_utility']+1e-12
            if 'generation_counts' in raw:
                assert raw['generation_counts']['predictions']+sum(s['counts']['predictions'] for s in stages)==run['predictions']
                assert raw['generation_counts']['milp']==run['milp']
                assert sum(s['counts']['outer'] for s in stages)==run['routes']
            proofs=bound_cache.setdefault(run['config_sha256'],{})
            new={digest(e['deployment']):e['deployment'] for e in raw['events'] if digest(e['deployment']) not in proofs}
            if new:
                p=ImmutableContinuousProblem(c,replica_options=run['states'])
                with SessionPredictorOracle(p) as oracle:
                    b=ContinuousDeploymentBounds(p,oracle)
                    for key,x in new.items():proofs[key]=b.evaluate(x)
            inputs={digest([e['deployment'],e['routing']]):(e['deployment'],e['routing']) for e in raw['events']}
            for stage in stages:
                assert stage['counts']['milp']==0
                assert stage['continuous_utility']>=stage['grid_utility']-1e-12
                total['safe_pruned']+=stage.get('safe_pruned',False)
                total['heuristic_paused']+=stage.get('state')=='PAUSED_STAGNATION'
                if stage.get('safe_pruned'):
                    proof=stage['bound'];assert proof['safe_pruned']
                    assert stage['counts']['predictions']==0
                    assert stage['counts']['outer']==0
                    assert proof['reason'] in {'UPPER_BOUND_DOMINATED','RESIDENT_MEMORY','UNCOVERED_TASK','ILLEGAL_REPLICA_STATE','INEVITABLE_DRAIN_FAILURE'}
                for h in stage['history']:
                    assert h['milp_solve_calls']==0
                    assert h['deployment']==stage['deployment'] if h.get('deployment') else True
                    for task,shares in h['routing_shares'].items():
                        assert abs(sum(shares.values())-1)<2e-15
                        assert set(shares)<=set(stage['deployment'])
                    inputs[digest([stage['deployment'],h['routing_qps']])]=(stage['deployment'],h['routing_qps'])
            for d in raw.get('decisions',[]):
                if d['action']=='safe_prune' and d['upper_bound'] is not None:
                    assert d['upper_bound']<=d['incumbent']+1e-12
                if d['action']=='pause':assert 'HEURISTIC' in d['reason'] or d['reason']=='NO_PROGRESS_WITHIN_CHUNK_TIME'
            assert sum(not e['cached'] for e in raw['events'])==run['predictions']
            for e in raw['events']:
                state,u=normalize_prediction(e['prediction'],c.search_policy)
                assert (state,u)==(e['state'],e['utility'])
                total['unknown_events']+=state=='UNKNOWN'
                if u is not None:
                    assert abs(u-sum(e['prediction'][k+'_contribution'] for k in 'QTLPME'))<1e-9
                    proof=proofs[digest(e['deployment'])]
                    if proof['upper_bound'] is not None:assert u<=proof['upper_bound']+1e-9
                    total['checked_utility_events']+=1
            for key,saved in mapping_entries(path):
                x,r=inputs[key];assert assignment(c,x,r)==saved;total['assignments']+=1
            total['runs']+=1;total['predictions']+=run['predictions'];total['routes']+=run['routes']
            total['milp']+=run['milp'];total['candidates']+=len(candidates)
            print(path.name,'verified',flush=True)
            del run,raw,c
            gc.collect()
    assert all(len(v)==1 for v in contracts.values())
    total.update(original_optimizer_sources_unchanged=True,shared_input_contract=True,
        exact_deployment_exclusions=True,no_cross_route_cuts=True,no_deployment_milp_during_refinement=True,
        continuous_optimality_claimed=False,physical_dynamic_interference_validated=False)
    return total


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',nargs='+',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();result=audit(args.sources)
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
