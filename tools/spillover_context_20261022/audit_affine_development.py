"""Post-search full-leaf checks of development affine envelopes and rho.

Reads completed DFS scores only after search, never as optimizer input.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.spillover_context_20261022.development_oracle import DevelopmentOracle
from tools.spillover_context_20261022.development_affine_bounds import DevelopmentAffineBounds

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=[];inputs={};oracle=DevelopmentOracle()
    for seed in [107,108]:
        path=args.base/f'plain-dfs-seed{seed}'/'evaluations.jsonl';inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        bounds=DevelopmentAffineBounds(oracle,[seed]);n=0;known=0;feasible=0;min_margin=math.inf;max_resource_error=0.;max_rho_excess=-math.inf;violations=[]
        for line in path.read_text().splitlines():
            row=json.loads(line);n+=1;p=row['plan'];x=p['deployment'];route=p['routing'];r=row['runs'][0]
            if row['diagnostic_objective'] is None:continue
            known+=1;margin=-row['diagnostic_objective']+bounds.upper(x,route,0);min_margin=min(min_margin,margin)
            if margin< -1e-9:violations.append(dict(kind='COST_FLOOR',identity=row['identity'],margin=margin))
            pressure=sum(q*statistics.mean(oracle._baselines(m,t,x[m]).values())/1000/oracle.catalog['capacity_factors'][m][str(x[m])]
                for t,rr in route.items() for m,q in rr.items())
            resources=bounds.state_cost(x)+bounds.weighted['intercept']+bounds.weighted['sum_baseline_pressure']*pressure
            actual=oracle.config['weights']['M']*r['terms']['M']+oracle.config['weights']['E']*r['terms']['E']
            error=abs(resources-actual);max_resource_error=max(max_resource_error,error)
            if error>1e-10:violations.append(dict(kind='AFFINE_RESOURCE_DECOMPOSITION',identity=row['identity'],error=error))
            if row['diagnostic_feasible']:
                feasible+=1
                for m,nrep in x.items():
                    lower=sum(rr.get(m,0)*bounds.pressure_service(m,t,nrep)/1000/oracle.catalog['capacity_factors'][m][str(nrep)] for t,rr in route.items())
                    excess=lower-r['rho_by_family'][m];max_rho_excess=max(max_rho_excess,excess)
                    if excess>1e-10 or lower>.8+1e-9:violations.append(dict(kind='NECESSARY_RHO',identity=row['identity'],family=m,lower=lower,actual=r['rho_by_family'][m]))
        if n!=7508:raise ValueError('Requires all7508 declared DFS leaves')
        out.append(dict(seed=seed,complete_leaves=n,known=known,planning_feasible=feasible,min_cost_minus_affine_envelope=min_margin,
            maximum_affine_resource_decomposition_error=max_resource_error,max_necessary_rho_minus_actual=max_rho_excess,
            violations=violations,contract=bounds.contract()))
    oracle.verify_inputs();args.output.mkdir(parents=True,exist_ok=False)
    result=dict(whole_leaf_checks=out,performed_after_search=True,used_as_solver_score_table=False,formal_U=None,
        formal_optimality_certified=False,physical_bound=False,inputs=inputs,oracle_provenance=oracle.provenance())
    dest=args.output/'report.json';dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs=inputs,
        files={'report.json':hashlib.sha256(dest.read_bytes()).hexdigest()}),indent=2)+'\n')
    if any(r['violations'] for r in out):raise ValueError('Affine envelope audit failed; retained report contains violations')
    print(json.dumps(dict(checked_leaves=sum(r['complete_leaves'] for r in out),violations=0)))

if __name__=='__main__':main()
