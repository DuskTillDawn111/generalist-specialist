"""Reuse historical candidates without new full predictions or changed gates."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gsdeploy import SystemConfig
from gsdeploy.continuous import ImmutableContinuousProblem,SessionPredictorOracle
from gsdeploy.continuous.candidate_bounds import ContinuousDeploymentBounds
from gsdeploy.continuous.contracts import digest
from experiments.joint_scalability import source_manifest
from scripts.report_multi_candidate import write,save


def load_compact(path):
    # Full request mappings dominate file size; candidate analysis needs only
    # the preceding optimization ledger. Keep the raw files untouched.
    chunks=[]
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith('  "routing_audit":'):
                return json.loads(''.join(chunks).rstrip().rstrip(',')+'\n}')
            chunks.append(line)
    return json.loads(''.join(chunks))


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rows=[];checked=0;bound_cache={};source=source_manifest()
    for src in args.sources:
        src=Path(src);old=json.loads((src/'source-manifest.json').read_text())
        assert all(source[k]==v for k,v in old.items()),'Historical optimizer sources changed'
        summary=json.loads((src/'summary.json').read_text())['rows']
        for meta in summary:
            path=src/f"{meta['case']}-{meta['algorithm']}-s{meta['seed']}.json"
            run=load_compact(path);raw=run['raw']
            c=SystemConfig.load(src/f"{meta['case']}-config.json")
            config_id=digest(c.as_dict())
            if config_id not in bound_cache:
                p=ImmutableContinuousProblem(c,replica_options=run['states'])
                with SessionPredictorOracle(p) as oracle:
                    b=ContinuousDeploymentBounds(p,oracle)
                    # Any deployment seen in this case may be evaluated later
                    # using a fresh exact oracle; no route cuts are transferred.
                    proofs={}
                    deployments={digest(e['deployment']):e['deployment'] for e in raw['events']}
                    for key,x in deployments.items():proofs[key]=b.evaluate(x)
                bound_cache[config_id]=proofs
            proofs=bound_cache[config_id]
            missing={digest(e['deployment']):e['deployment'] for e in raw['events'] if digest(e['deployment']) not in proofs}
            if missing:
                p=ImmutableContinuousProblem(c,replica_options=run['states'])
                with SessionPredictorOracle(p) as oracle:
                    b=ContinuousDeploymentBounds(p,oracle)
                    for key,x in missing.items():proofs[key]=b.evaluate(x)
            for e in raw['events']:
                proof=proofs[digest(e['deployment'])]
                if e['utility'] is not None and proof['upper_bound'] is not None:
                    assert e['utility']<=proof['upper_bound']+1e-9
                    checked+=1
            best=run['utility']
            for candidate in raw.get('candidates',[]):
                i=candidate['generation_index'];x=candidate['deployment'];pred=candidate['grid_prediction']
                stage=next((s for s in raw.get('stages',[]) if s['generation_index']==i),None)
                calls=0;first=None;gains=0;previous=candidate['grid_utility']
                for h in stage['history'] if stage else []:
                    calls+=h['full_prediction_calls']
                    u=h['best_feasible_utility']
                    if u is not None and u>previous+1e-12:
                        previous=u;gains+=1
                        if first is None:first=calls
                refined=stage['continuous_utility'] if stage else None
                rows.append(dict(case=run['case'],algorithm=run['algorithm'],seed=run['seed'],index=i,
                    deployment=x,activation=sorted(x),grid_U=candidate['grid_utility'],refined_U=refined,
                    observed_gain=refined-candidate['grid_utility'] if refined is not None else None,
                    incumbent_U=best,gap_to_incumbent=best-candidate['grid_utility'],
                    residual_gap=best-refined if refined is not None else None,
                    dimension=sum(sum(t in c.catalog['models'][m]['supported_tasks'] for m in x)-1 for t in c.workload),
                    memory_mib=pred['safe_memory_mib'],M_contribution=pred['M_contribution'],E_contribution=pred['E_contribution'],
                    predictions=stage['counts']['predictions'] if stage else 0,
                    routes=stage['counts']['outer'] if stage else 0,seconds=stage['seconds'] if stage else 0,
                    first_gain_predictions=first,improving_evaluations=gains,
                    cost_per_gain=stage['counts']['predictions']/gains if stage and gains else None,
                    bound=proofs[digest(x)]['upper_bound'],certified=candidate['restricted_grid_certified'],source=str(path)))
            print(path.name,'analyzed',flush=True)
            del run,raw,c
            gc.collect()
    write(out/'candidate-potential.csv',rows)
    groups=[]
    for case,algorithm,key in sorted({(r['case'],r['algorithm'],digest(r['deployment'])) for r in rows}):
        rr=[r for r in rows if r['case']==case and r['algorithm']==algorithm and digest(r['deployment'])==key and r['refined_U'] is not None]
        if not rr:continue
        groups.append(dict(case=case,algorithm=algorithm,deployment=rr[0]['deployment'],seeds=len(rr),
            gain_min=min(r['observed_gain'] for r in rr),gain_max=max(r['observed_gain'] for r in rr),
            U_range=max(r['refined_U'] for r in rr)-min(r['refined_U'] for r in rr),
            median_predictions=statistics.median(r['predictions'] for r in rr),
            zero_gain_seeds=sum(r['observed_gain']<=1e-12 for r in rr)))
    write(out/'seed-stability.csv',groups)
    save(out/'audit.json',dict(candidate_rows=len(rows),historical_prediction_events_bound_checked=checked,
        original_historical_sources_unchanged=True,new_full_predictions=0,bounds=bound_cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in rows if r['algorithm']=='utility-k5' and r['seed']==11]
    fig,axs=plt.subplots(1,3,figsize=(15,4.8))
    for r in selected:
        if r['refined_U'] is None:continue
        label=r['case'].split('trace')[-1]+':D'+str(r['index'])
        axs[0].scatter(r['gap_to_incumbent'],r['observed_gain'])
        axs[0].annotate(label,(r['gap_to_incumbent'],r['observed_gain']),fontsize=7)
        axs[1].scatter(r['dimension'],r['predictions'])
        axs[2].scatter(r['predictions'],r['observed_gain'])
    axs[0].plot([0,.1],[0,.1],'--',color='gray',label='gain needed to match incumbent')
    axs[0].set(xlabel='Grid gap to incumbent',ylabel='Observed continuous gain',ylim=(-.002,.04));axs[0].legend(fontsize=7)
    axs[1].set(xlabel='Fixed-deployment dimension',ylabel='Full predictions')
    axs[2].set(xlabel='Full predictions',ylabel='Observed continuous gain')
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'candidate-potential.png',dpi=170)
    print('Historical candidate rows',len(rows),'bound-checked events',checked,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',nargs='+',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
