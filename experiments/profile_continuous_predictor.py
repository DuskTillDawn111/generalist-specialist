"""Existing-session cache sensitivity and cProfile of one identical real candidate.

No output caching or model changes: every invocation still runs the FIFO/scorer.
"""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gsdeploy import SystemConfig,DeploymentSystem
from gsdeploy.continuous import Budget,ContinuousProblem,ContinuousRoutingEvaluator,LegacyPredictorOracle
from gsdeploy.search_session import SearchSession
from gsdeploy.prediction import predict_scenarios


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    case=json.loads((source/'pool5-config.json').read_text())
    config=SystemConfig(**case['system'])
    old=json.loads((source/'pool5-joint_milp-r0.json').read_text())
    x,r=old['winner']['deployment'],old['winner']['routing']
    e=ContinuousRoutingEvaluator(ContinuousProblem(config,replica_options=case['replica_options']),LegacyPredictorOracle(),
                                budget=Budget(120,1000),cache=False)
    raw=[];wrapped=[];cached=[]
    predictions=[]
    for i in range(3):
        start=time.perf_counter()
        p=predict_scenarios(config.catalog,x,r,config.workload,config.gpu_memory_mib,
            config.latency_reference_ms,config.weights,**DeploymentSystem(config).context())
        raw.append(time.perf_counter()-start)
        start=time.perf_counter();q=e.candidate(x,r);wrapped.append(time.perf_counter()-start)
        assert p==q['traces'][0]['prediction']
        predictions.append(p)
    start=time.perf_counter()
    with SearchSession(Path(__file__).resolve().parents[1],config.as_dict()) as session:
        setup=time.perf_counter()-start
        for i in range(3):
            start=time.perf_counter();p=session.predict(x,r);cached.append(time.perf_counter()-start)
            assert p==predictions[i]
        stats=session.stats()
    profile=cProfile.Profile()
    profile.runcall(e.candidate,x,r)
    profile.dump_stats(str(output/'candidate.pstats'))
    with (output/'candidate-profile.txt').open('w') as stream:
        pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(45)
    result=dict(source=str(source),deployment=x,routing=r,prediction_json_exactly_equal=True,
        raw_core_seconds=raw,continuous_candidate_seconds=wrapped,existing_frozen_session_seconds=cached,
        raw_core_median=statistics.median(raw),continuous_candidate_median=statistics.median(wrapped),
        existing_frozen_session_median=statistics.median(cached),session_setup_seconds=setup,session_stats=stats,
        cache_scope='existing SearchSession digest cache only; no prediction result reuse; setup reported separately',
        evidence='validated real-data replay; not GPU measurement')
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();run(a.source,a.output)
