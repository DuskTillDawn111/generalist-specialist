"""Predeclare native discrimination of the actual complete-search top two.

Rank1 remains the algorithmic proposal; rank2 remains its strongest evaluated
competitor. This development calibration is not independent confirmation or a
claim that the runner-up was selected. No GPU is started here.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
sys.path.insert(0,str(SOURCE))
from gsdeploy import static_study
from gsdeploy.config import SystemConfig
from gsdeploy.quality import quality_digest


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def prepare(aggregate,output):
    aggregate=Path(aggregate).resolve();output=Path(output).resolve()
    inputs=[aggregate/'evaluations.jsonl',aggregate/'top-controls.json']
    rows=[json.loads(line) for line in inputs[0].read_text().splitlines() if line.strip()]
    if len(rows)!=7508 or len({r['identity'] for r in rows})!=7508:
        raise ValueError('Exactly7508 unique full-domain identities required')
    for row in rows:
        rr=row['runs']
        if len(rr)!=2 or {r['seed'] for r in rr}!={107,108} or row['formal_U'] is not None:
            raise ValueError('Both development seeds and explicit formal UNKNOWN required')
        values=[r['diagnostic_U'] for r in rr]
        mean=sum(values)/2 if all(v is not None for v in values) else None
        if (mean is None)!=(row['diagnostic_objective'] is None) or (
                mean is not None and not math.isclose(mean,row['diagnostic_objective'],abs_tol=1e-12)):
            raise ValueError('Aggregate mean changed')
        if row['diagnostic_feasible']!=all(r['resource_feasible'] is True for r in rr):
            raise ValueError('Aggregate planning gate changed')
    ranked=sorted((r for r in rows if r['diagnostic_feasible'] and r['diagnostic_objective'] is not None),
                  key=lambda r:(-r['diagnostic_objective'],r['identity']))
    top=json.loads(inputs[1].read_text())['top10']
    if len(ranked)<2 or [r['identity'] for r in ranked[:2]]!=[r['identity'] for r in top[:2]]:
        raise ValueError('Top-two selection differs from complete independent enumeration')
    chosen={f'rank{i}':row for i,row in enumerate(ranked[:2],1)}
    basepath=ROOT/'artifacts/static-spillover-study-20261022/text185-fine-development-v1/experiment-packet.json'
    base=json.loads(basepath.read_text());raw=copy.deepcopy(base['config']);config=SystemConfig(**raw)
    checks={name:config.validate_plan(**row['plan']) for name,row in chosen.items()}
    if not all(r['constraints_valid'] for r in checks.values()):
        raise ValueError('Search candidate fails unchanged native proxy admission: '+str(checks))
    output.mkdir(parents=True,exist_ok=False)
    design=dict(schema='gsdeploy.prequential_search_top_two_discrimination.v1',
        purpose='Native calibration and discrimination of complete-search rank1 versus rank2; not confirmation of a desired spillover winner.',
        selection='Rank all7508 common identities by mean diagnostic U107/108, requiring both development planning gates; take first two without a role/route filter.',
        candidates=chosen,unchanged_native_proxy_admission=checks,
        workload=raw['workload'],horizon_seconds=600,warmup_seconds=30,arrival_process=raw['arrival_process'],
        development_seeds=[109,110,111,112],
        order=[dict(seed=109,conditions=['rank1','rank2']),dict(seed=110,conditions=['rank2','rank1']),
               dict(seed=111,conditions=['rank2','rank1']),dict(seed=112,conditions=['rank1','rank2'])],
        statistics=dict(primary='Per-seed matched ΔU=rank2−rank1 from unchanged native six-term evaluator.',
            unit='Complete run pair, not requests.',retain_all=True,
            incomplete_or_ineligible='Retain status and original diagnostic terms; never impute utility or drop a pair to improve mean.',
            descriptive='Report all four paired effects, mean/range and uncertainty with assumptions; no outcome-driven additional seeds.',
            inference='No calibrated deployment-global optimum or stable benefit claim from a positive point average alone.'),
        protocol='Original input pools, trace, SamplePool, expanded SWRR, native protocols and v5 stability unchanged; fresh workers every condition.',
        runtime_source=str(SOURCE),quality='Score actual native outputs; historical replay Q is not substituted.',
        reserved_seeds=dict(independent_validation=[81,82],final_confirmation=[71,72,73]),
        no_hidden_retry=True,all_conditions_declared_before_new_results=True,
        physical_or_full_seven_pool_optimality=False,predictor_activated=False,GPU_started=False)
    write(output/'predeclared-protocol.json',design)
    bindings=base['bindings']+[static_study._file(p) for p in [*inputs,basepath,Path(__file__).resolve(),output/'predeclared-protocol.json']]
    bindings=sorted({b['path']:b for b in bindings}.values(),key=lambda b:b['path'])
    jobs=[]
    for row in design['order']:
        for condition in row['conditions']:
            job=static_study._job(config,base['workers'],chosen[condition]['plan'],
                f'search_{condition}_seed{row["seed"]}','calibration',row['seed'],bindings,30)
            job.update(purpose='prequential_search_top_two_discrimination',proxy_constraints=checks[condition],search_rank=condition)
            jobs.append(job)
    packet=dict(schema='gsdeploy.static_experiments.v1',status='PREDECLARED_SEARCH_TOP_TWO_DISCRIMINATION',
        inputs=base['inputs'],config=raw,workers=base['workers'],bindings=bindings,
        frozen_files_sha256=quality_digest(bindings),jobs=jobs,gpu_started=False,
        discovery_protocol=str(output/'predeclared-protocol.json'),
        execution=dict(policy='Serial fresh-worker native conditions. Retain all outcomes, no hidden retries. Stop and preserve infrastructure/resource failures.',
            runtime_source=str(SOURCE),scheduled=False))
    validation=static_study.validate_packet(packet)
    if not validation['valid']:raise ValueError(str(validation))
    write(output/'experiment-packet.json',packet)
    receipt=dict(status='PREDECLARED_CPU_VALID_PACKET',packet=static_study._file(output/'experiment-packet.json'),
        validation=validation,jobs=len(jobs),selection=chosen,source=static_study._file(Path(__file__).resolve()),GPU_started=False)
    write(output/'preparation-receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],packet=receipt['packet'],jobs=len(jobs))))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--aggregate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.aggregate,a.output)
