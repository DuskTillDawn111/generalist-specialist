"""Explicit one-task face diagnostic; not the all-task multi-candidate method."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gsdeploy import SystemConfig
from gsdeploy.continuous import Budget, ImmutableContinuousProblem, SessionPredictorOracle, DERoutingOptimizer
from gsdeploy.continuous.refinement import FixedDeploymentRoutingEvaluator, project_shares
from gsdeploy.continuous.multi_candidate import JointDeploymentCandidates
from experiments.joint_scalability import ROOT,DEFAULT_SOURCE,source_manifest,subset_config,save
from experiments.multi_candidate_refinement import audit_routes


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    manifest=source_manifest();save(out/'source-manifest.json',manifest)
    base=SystemConfig.load(DEFAULT_SOURCE)
    c=subset_config(base,['generalist_2','text','image','ocr','asr'],1,40.,seed=42)
    states={m:[1] for m in c.catalog['models']}
    save(out/'config.json',c.as_dict())
    save(out/'plan.json',dict(task='text',candidate='first generated deployment with G2 and Text',
        seeds=[11,29,47],route_budget=128,prediction_budget=240,seconds=40,
        frozen_other_rows='original candidate grid route',
        purpose='check whether all-task DE allocation misses a useful existing support face',
        independent_ablation_not_proposed_algorithm=True))
    summary=[]
    for seed in (11,29,47):
        b=Budget(40.,240,128)
        p=ImmutableContinuousProblem(c,replica_options=states)
        with SessionPredictorOracle(p) as oracle:
            ex=JointDeploymentCandidates(p,oracle,b)
            candidates=ex.generate(5)
            candidate=next(v for v in candidates if 'generalist_2' in v['deployment'] and 'text' in v['deployment'])
            e=FixedDeploymentRoutingEvaluator(p,oracle,candidate['deployment'],b)
            e.prediction_cache=ex.cache
            for task in e.domain.edges:
                if task!='text':e.domain.edges[task]=tuple(candidate['grid_routing'][task])
            e.domain.dimension=sum(len(ms)-1 for ms in e.domain.edges.values())
            original=oracle.predict
            ex.stage='text_face_diagnostic'
            def observed(problem,deployment,routing,trace,budget,deadline=None):
                prediction=original(problem,deployment,routing,trace,budget,deadline)
                ex.observe(deployment,routing,prediction)
                return prediction
            with patch.object(oracle,'predict',observed):
                shares={t:{m:q/c.workload[t] for m,q in row.items() if m in e.domain.edges[t]}
                        for t,row in candidate['grid_routing'].items()}
                raw=DERoutingOptimizer(e,seed=seed,popsize=3,maxiter=100,
                    initial_shares=shares).solve()
            best=raw['winner'];u=max(candidate['grid_utility'],best['best_feasible_utility'] if best else -float('inf'))
            all_raw=dict(events=ex.events,stages=[dict(deployment=candidate['deployment'],history=e.history)])
            result=dict(seed=seed,candidates=candidates,candidate=candidate,grid_U=candidate['grid_utility'],
                continuous_U=u,improvement=u-candidate['grid_utility'],global_first_U=candidates[0]['grid_utility'],
                exceeds_first=u>candidates[0]['grid_utility']+1e-12,seconds=time.perf_counter()-b.start,
                budget=b.report(),joint_solves=ex.solves,raw=raw,history=e.history,events=ex.events,
                routing_audit=audit_routes(c,all_raw),continuous_optimality=False,
                scope='fixed Text deployment, only text may vary; other routes frozen; generation cost included')
            save(out/f'seed{seed}.json',result)
            summary.append({k:result[k] for k in ('seed','grid_U','continuous_U','improvement','global_first_U','exceeds_first','seconds','budget')})
            save(out/'summary.json',summary)
            print(seed,u,result['improvement'],result['exceeds_first'],flush=True)
    assert manifest==source_manifest()
    save(out/'complete.json',dict(runs=3,original_sources_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    main(p.parse_args())
