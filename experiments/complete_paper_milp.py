"""Bounded CPU demonstration of training, context laws and MILP reuse.

All observations here are explicit analytical fixtures from the test suite.
No GPU is touched and no timings are evidence of real-workload speedup.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from gsdeploy.continuous import Budget, ImmutableContinuousProblem, SessionPredictorOracle
from gsdeploy.continuous.contracts import TraceSpec
from gsdeploy.continuous.candidate_bounds import ContinuousDeploymentBounds
from gsdeploy.continuous.multi_candidate import JointDeploymentCandidates
from gsdeploy.continuous.adaptive_candidate import AdaptiveCandidateRefinementOptimizer
from gsdeploy.continuous.predictor_adapter import source_identity
from gsdeploy.paper_forecast import MODE, canonical
from gsdeploy.paper_training import fit_config
from tests.test_paper_completion import config, training, interpolation_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    c=config();entry=c.catalog[MODE]['deployments'][0]
    observations={canonical(entry['deployment']):training(entry)}
    save('training-observations-CPU-FIXTURE.json',observations)
    fitted,fit_report=fit_config(c,observations,interpolate_contexts=True)
    save('fitted-system-config.json',fitted);save('fit-report.json',fit_report)
    p=ImmutableContinuousProblem(c)
    column_rows=[]
    for enabled in (False,True):
        with SessionPredictorOracle(p) as oracle:
            budget=Budget(6,64,64)
            ex=JointDeploymentCandidates(p,oracle,budget,cache_columns=enabled)
            candidates=ex.generate(3)
            save(f'columns-{enabled}.json',dict(candidates=candidates,solves=ex.solves,budget=budget.report()))
            column_rows.append(dict(cache_enabled=enabled,seconds=budget.report()['elapsed_seconds'],
                columns_constructed=ex.column_cache.misses if enabled else len(ex.solves),
                cache_hits=ex.column_cache.hits if enabled else 0,
                deployments=[c['deployment'] for c in candidates],utilities=[c['grid_utility'] for c in candidates],
                predictions=budget.counts['predictions'],milp=budget.counts['milp']))
            column_rows[-1]['column_seconds']=sum(r['column_seconds'] for r in ex.solves)
    identical=all(column_rows[0][k]==column_rows[1][k] for k in ('deployments','utilities','predictions','milp'))
    adaptive=[]
    for enabled in (False,True):
        raw=AdaptiveCandidateRefinementOptimizer(c,k=3,safe_filter=enabled,
            seconds=8,max_predictions=48,max_routes=16).solve()
        save(f'adaptive-filter-{enabled}.json',raw)
        adaptive.append(dict(safe_filter=enabled,utility=raw['winner']['utility'] if raw['winner'] else None,
            budget=raw['budget'],decisions=raw['decisions'],status=raw['status']))
    interpolated=[]
    for enabled in (False,True):
        c2=interpolation_config()
        if not enabled:
            c2.catalog[MODE]['deployments'][-1]['profiles'].pop('context_interpolation')
        p2=ImmutableContinuousProblem(c2,traces=[TraceSpec('burst',11,((0.,'text'),(0.,'text')))])
        with SessionPredictorOracle(p2) as oracle:
            prediction=oracle.predict(p2,{'g':1,'s':1},{'text':{'g':1.,'s':1.}},p2.train_traces[0],Budget(3,4,4))
            proof=ContinuousDeploymentBounds(p2,oracle).evaluate({'g':1,'s':1})
            save(f'interpolation-{enabled}.json',dict(prediction=prediction,bound=proof))
            interpolated.append(dict(enabled=enabled,utility=prediction['U'],eligible=prediction['eligible'],
                reason=prediction.get('reason'),upper_bound=proof['upper_bound'],
                request_mapping=prediction.get('request_mapping')))
    report=dict(schema='gsdeploy.paper_milp_completion_smoke.v1',CPU_FIXTURE_ONLY=True,GPU_started=False,
        physical_accuracy_or_gain_claim=False,source_identity=source_identity(),
        columns=column_rows,exact_column_ablation_identical=identical,
        adaptive=adaptive,context_interpolation=interpolated,
        timing_scope='one tiny deterministic correctness run; no speedup/scalability inference')
    save('report.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8,3.2))
    axes[0].bar(['No reuse','Exact reuse'],[r['columns_constructed'] for r in column_rows],color=['#708090','#2b8cbe'])
    axes[0].set_ylabel('Task-column constructions');axes[0].set_ylim(0,4)
    axes[1].bar(['Filter off','Filter on'],[r['budget']['full_prediction_calls'] for r in adaptive],color=['#708090','#2b8cbe'])
    axes[1].set_ylabel('Full prediction requests')
    fig.suptitle('CPU fixture only: structural counters, no physical gain claim',fontsize=10)
    fig.tight_layout();fig.savefig(out/'counters.png',dpi=160);plt.close(fig)
    files=sorted({*ROOT.glob('gsdeploy/continuous/*.py'),*ROOT.glob('gsdeploy/forecasting/*.py'),
                  ROOT/'gsdeploy/paper_training.py',ROOT/'gsdeploy/paper_forecast.py',
                  ROOT/'tests/test_paper_completion.py',Path(__file__),ROOT/'experiments/fit_paper_predictor.py',
                  ROOT/'configs/paper_fifo_cpu_fixture.json'})
    save('source-manifest.json',[dict(path=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files])
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if identical and all(r['utility'] is not None for r in adaptive) and interpolated[1]['eligible'] else 1


if __name__=='__main__':raise SystemExit(main())
