"""Bounded full development enumeration, then predeclared independent pairs."""
from .common import *
import traceback,fcntl

def verify(f):
    for r in f['native_sources']+f['sources']+f['inputs']:
        assert ref(r['path'])['sha256']==r['sha256'],r['path']

def execute(slot,f):
    from .gpu_native import run
    from .gpu_assess import assess
    raw=run(slot,f);o=assess(raw);save('gpu_observations/'+slot['run_id']+'.json',o)
    print('GPU_FINISH',slot['run_id'],'valid',o['technical_valid'],'stable',o['stable'],'U',o['U'],'checks',o['checks'],flush=True)
    if not o['technical_valid']:raise RuntimeError(('GPU_TECHNICAL_INVALID_NO_RETRY',slot['run_id'],o['checks'],raw['errors']))
    return o

def main():
    # Confinement is inherited by every native child before CUDA initialization.
    from .isolation import install
    install('supervisor')
    lock=(OUT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    f=read(OUT/'frozen_gpu_manifest_v4.json');dev=read(OUT/'development_observations_reconciled.json');pairs=[];failure=None
    # The sole earlier raw is an initialization failure before any measurement.
    # Its distinct run ID and NULL U are permanently retained. No measured result
    # can authorize this infrastructure-only relaunch.
    previous=list((OUT/'raw_runs').glob('*.json'))
    assert len(previous)==2 and len(dev)==1 and dev[0]['technical_valid']
    try:
        for slot in f['slots'][1:]:
            verify(f);save('campaign_status.json',dict(status='DEVELOPMENT',run_id=slot['run_id'],completed=len(dev),expected=8,pid=os.getpid()))
            print('GPU_START',slot['run_id'],flush=True);dev.append(execute(slot,f));save('development_observations.json',dev)
        eligible=[o for o in dev if o['eligible']]
        if not eligible:raise RuntimeError('NO_VALID_GPU_DEVELOPMENT_CANDIDATE')
        best=max(eligible,key=lambda o:o['U']);candidates=read(OUT/'gpu_candidates.json');by={c['candidate_id']:c for c in candidates}
        decisions=read(OUT/'optimizer_decisions.json');selected=[]
        for label in ['20A_FINITE_DOMAIN_ADAPTER','20A_NATIVE_CONTINUOUS','FORMAL_V3_CPU_BACKEND']:
            dd=next(r for r in decisions if r['pool']=='SMALL6' and r['load']=='ASR32B' and r['algorithm']==label)
            cc=next((c for c in candidates if identity(c['deployment'],c['routing'])==identity(dd['deployment'],dd['routing'])),None)
            if cc and cc['candidate_id'] not in selected:selected.append(cc['candidate_id'])
        arms=list(dict.fromkeys([best['candidate_id'],*selected]));frozen_pairs=[]
        fm=read(OUT/'frozen_problem_manifest.json');gp=fm['GPU_protocol']
        for rep,seed in enumerate(gp['independent_seeds'],1):
            order=arms if rep%2 else list(reversed(arms))
            for aid in order:
                c=by[aid];frozen_pairs.append(dict(run_id=f'PAIR{rep}_'+aid,candidate_id=aid,stage='INDEPENDENT_PAIR',repetition=rep,
                    deployment=c['deployment'],routing=c['routing'],lambda_rps=f['slots'][0]['lambda_rps'],seed=seed,
                    measurement_seconds=gp['independent_pair_seconds'],safe_memory_budget_mib=f['slots'][0]['safe_memory_budget_mib'],
                    trajectories=f['pair_trajectories'][str(seed)],predicted_U=c['predicted_U']))
        save('independent_pair_selection.json',dict(selected_before_independent_measurements=True,development_best=best['candidate_id'],
            optimizer_selected=selected,arms=arms,slots=frozen_pairs,development_all_measured=True,
            identical_candidate_rule='If observed best and optimizer are identical, measure it once per independent trace; regret against itself is exactly zero, no fabricated independent A/B contrast.'))
        for slot in frozen_pairs:
            verify(f);save('campaign_status.json',dict(status='INDEPENDENT_PAIR',run_id=slot['run_id'],completed=len(pairs),expected=len(frozen_pairs),pid=os.getpid()))
            print('GPU_START',slot['run_id'],flush=True);pairs.append(execute(slot,f));save('pair_observations.json',pairs)
    except BaseException as exc:
        failure=dict(error=repr(exc),traceback=traceback.format_exc());save('gpu_campaign_failure.json',failure);print('GPU_STOP',repr(exc),flush=True)
    finally:
        save('development_observations.json',dev);save('pair_observations.json',pairs)
        save('campaign_status.json',dict(status='STOPPED_ON_FAILURE' if failure else 'COMPLETED',development_runs=len(dev),pair_runs=len(pairs),failure=failure))
    if failure:raise SystemExit(1)
if __name__=='__main__':main()
