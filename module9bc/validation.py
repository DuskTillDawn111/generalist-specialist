"""Analytic optima and frozen-stack integration checks, CPU only."""
from types import SimpleNamespace
from copy import deepcopy
import math
from deployment_core import Workload
from deployment_core.models import ModelProfile
from deployment_core.routing.e2e_router import solve_e2e_routing
from deployment_core.routing.e2e_surrogate import evaluate_surrogate
from .common import *

def main():
    catalog={m:ModelProfile(m,frozenset(['text']),1.,{'text':s}) for m,s in [('a',100.),('b',100.)]}
    cal=SimpleNamespace(evidence_bounds={'a':1,'b':1},capacity_factors={'a':{0:0.,1:1.},'b':{0:0.,1:1.}})
    # Use a real validated ReplicaCalibration for synthetic canonical routing.
    from deployment_core.replica_calibration import ReplicaCalibration
    cal=ReplicaCalibration(cal.capacity_factors,{m:{0:0.,1:1.} for m in catalog},cal.evidence_bounds,{m:{} for m in catalog},{})
    w=Workload({'text':10.});x={'a':1,'b':1}
    fit=solve_e2e_routing(x,w,catalog,cal)
    assert fit['converged'] and all(abs(fit['routing']['text',m]-5.)<1e-6 for m in catalog)
    assert abs(fit['evaluation']['J_seconds']-.2)<1e-10
    # Interior KKT optimum for unequal service times: equal marginal cost.
    catalog['b']=ModelProfile('b',frozenset(['text']),1.,{'text':200.})
    fit=solve_e2e_routing(x,w,catalog,cal);r=fit['routing'];a=r['text','a'];b=r['text','b']
    assert fit['converged'] and abs(.1/(1.-.1*a)**2-.2/(1.-.2*b)**2)<1e-5
    heavy=solve_e2e_routing({'a':1,'b':0},Workload({'text':11.}),catalog,cal)
    assert heavy['status']=='CAPACITY_INFEASIBLE'
    zero=solve_e2e_routing(x,Workload({'text':0.}),catalog,cal);assert zero['evaluation']['J_seconds']==0.
    try:evaluate_surrogate({'a':1,'b':0},{('text','b'):10.},w,catalog,cal)
    except ValueError:pass
    else:raise AssertionError('Absent path accepted')
    # Thin integration must generate the same exact first neighborhood under
    # a dry-run, preserving MILP, canonical feedback, and all pruning semantics.
    from .integration import create_search
    from method_v2 import frozen_config
    from dataclasses import replace
    from deployment_core import Hardware
    c,h,roles,realcal=inputs();cell=cells()['C07'];config=replace(frozen_config(),dry_run=True)
    args=(Workload(cell['lambda_rps']),c,Hardware(cell['planning_budget_input_mib']))
    before={p.name:file_hash(p) for p in (ROOT/'method_v2').glob('*.py')}
    old=create_search(None,c,realcal,ranking='delta_z').search(*args,config,roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=realcal)
    new=create_search(None,c,realcal,ranking='delta_J').search(*args,config,roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=realcal)
    assert old.seed.deployment==new.seed.deployment
    a=old.to_dict()['iterations'][0];b=new.to_dict()['iterations'][0]
    assert [r['mutation'] for r in a['candidates']]==[r['mutation'] for r in b['candidates']]
    assert a['candidate_complexity']==b['candidate_complexity'] and a['canonical']==b['canonical']
    assert before=={p.name:file_hash(p) for p in (ROOT/'method_v2').glob('*.py')}
    save('analytic_and_integration_validation.json',{'status':'PASS','identical_worker_convex_optimum':True,'unequal_service_KKT':True,'capacity_domain_zero_load_absent_path':True,'private_v2_stack_seed_generation_feedback_identical':True,'frozen_v2_files_unchanged':True,'GPU_runs':0})
    print('Analytic and integration validation PASS',flush=True)
if __name__=='__main__':main()
