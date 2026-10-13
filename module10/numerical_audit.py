"""Diagnostic solves of the old H0, with fixed objective and rho <= .8.

No routing is adopted, no frozen solver or scientific tolerance is modified.
"""
from decimal import Decimal, localcontext
import math
import numpy as np
from scipy.optimize import minimize, linprog, brentq
from deployment_core import Workload
from deployment_search.routing import solve_pressure_relaxation
from deployment_core.routing.e2e_router import solve_e2e_routing, CERTIFICATE_TOLERANCE_SECONDS
from .operational import ARCHIVE, AUDIT, save, inputs, read, ref, check_archive

def main():
    c,h,roles,cal=inputs()
    old=read(ARCHIVE/'planning/H0_V3_MINIMAL.json')
    point=read(ARCHIVE/'hotspot_points.json')['points'][0]
    x=old['partial_search_trace']['seed']['deployment']; w=Workload(point['lambda_rps'])
    models=sorted(m for m in x if x[m]);tasks=sorted(w.active_tasks)
    pairs=[(k,m) for k in tasks for m in models if k in c[m].supported_tasks]
    eq=np.array([[float(k==t) for k,m in pairs] for t in tasks])
    S=np.array([[w.arrival_rate[k]*c[m].service_time_ms[k]/1000 if m==model else 0 for k,m in pairs] for model in models])
    g=np.array([cal.capacity_factors[m][x[m]] for m in models]);L=sum(w.arrival_rate.values())
    initial=solve_pressure_relaxation(x,w,c,replica_bounds=cal.evidence_bounds,capacity_mode='empirical',calibration=cal)
    u0=np.array([initial.routing.get((k,m),0)/w.arrival_rate[k] for k,m in pairs])
    def obj(u):
        d=S@u;return float(np.sum(d/np.maximum(1-d/g,1e-9))/L)
    def grad(u):return S.T@(1/np.maximum(1-S@u/g,1e-9)**2)/L
    def certificate(u,ceiling=.8):
        v=grad(u)
        lp=linprog(v,A_eq=eq,b_eq=np.ones(len(tasks)),A_ub=S,b_ub=ceiling*g,
                   bounds=[(0,1)]*len(pairs),method='highs',
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        assert lp.success
        return max(0.,float(v@(u-lp.x)))
    constraints=[{'type':'eq','fun':lambda u:eq@u-1,'jac':lambda u:eq},
                 {'type':'ineq','fun':lambda u:.8*g-S@u,'jac':lambda u:-S}]
    assert np.max(S@u0/g)<=.8+1e-9
    runs=[]
    # Three unit representations, identical absolute objective stopping precision
    # in seconds and identical scientific certificate threshold (10 ns).
    for scale,name in [(1.,'seconds'),(1000.,'milliseconds'),(1e6,'microseconds')]:
        repeats=[]
        for repeat in range(2):
            r=minimize(lambda u:scale*obj(u),u0.copy(),jac=lambda u:scale*grad(u),
                       bounds=[(0,1)]*len(pairs),constraints=constraints,method='SLSQP',
                       options={'ftol':scale*1e-15,'maxiter':1000})
            u=r.x;gap=certificate(u);valid=bool(np.max(abs(eq@u-1))<=1e-7 and np.min(u)>=-1e-9 and np.max(S@u/g)<=.8+1e-9)
            repeats.append({'repeat':repeat+1,'solver_success':bool(r.success),'message':r.message,
                'iterations':int(r.nit),'J_seconds':obj(u),'optimality_gap_seconds':gap,
                'gap_ns':gap*1e9,'certificate_pass':gap<=CERTIFICATE_TOLERANCE_SECONDS,
                'constraints_valid':valid,'max_rho':float(np.max(S@u/g)),
                'routing_rps':{f'{k}::{m}':float(v*w.arrival_rate[k]) for (k,m),v in zip(pairs,u)}})
        assert repeats[0]['routing_rps']==repeats[1]['routing_rps']
        runs.append({'objective_units':name,'scale':scale,'ftol_objective_units':scale*1e-15,
                     'ftol_seconds':1e-15,'deterministic_repeat_exact_match':True,'repeats':repeats})
    # Primary deployment has exactly one free routing direction: ASR G/S split.
    flexible=[k for k in tasks if sum(t==k for t,m in pairs)>1]
    assert flexible==['asr'] and x['phi4mm']==x['whisper-small']==1
    assert cal.capacity_factors['phi4mm'][1]==cal.capacity_factors['whisper-small'][1]==1.
    lp=w.arrival_rate['asr'];sg=c['phi4mm'].service_time_ms['asr']/1000;ss=c['whisper-small'].service_time_ms['asr']/1000
    bg=w.arrival_rate['visual_qa']*c['phi4mm'].service_time_ms['visual_qa']/1000
    bs=w.arrival_rate['lid']*c['whisper-small'].service_time_ms['lid']/1000
    lower=max(0.,(bs+ss*lp-.8)/ss);upper=min(lp,(.8-bg)/sg)
    def derivative(t):return (sg/(1-bg-sg*t)**2-ss/(1-bs-ss*(lp-t))**2)/L
    root=brentq(derivative,lower,upper,xtol=1e-14,rtol=1e-14)
    def vector(t):
        return np.array([(t/lp if m=='phi4mm' else 1-t/lp) if k=='asr' else 1. for k,m in pairs])
    old_t=(old['frozen_J_solver_diagnostic']['evaluation']['D']['phi4mm']-bg)/sg
    with localcontext() as ctx:
        ctx.prec=60
        d=lambda v:Decimal(str(v))
        SG,SS,BG,BS,LP,LL=map(d,[sg,ss,bg,bs,lp,L])
        A=SG.sqrt();B=SS.sqrt()
        exact=(B*(1-BG)-A*(1-BS-SS*LP))/(A*SS+B*SG)
        def derivative_dec(t):return (SG/(1-BG-SG*t)**2-SS/(1-BS-SS*(LP-t))**2)/LL
        def gap_dec(t,ceiling):
            lo=max(Decimal(0),(BS+SS*LP-ceiling)/SS)
            hi=min(LP,(ceiling-BG)/SG)
            v=derivative_dec(t); endpoint=lo if v>=0 else hi
            return max(Decimal(0),v*(t-endpoint))
        t=d(old_t);dp=derivative_dec(t)
        high={'precision_digits':60,'exact_optimal_G_asr_rps':str(exact),
              'old_G_asr_rps_reconstructed_from_saved_D':str(t),'old_derivative_seconds_per_rps':str(dp),
              'old_operational_gap_seconds':str(gap_dec(t,d(.8))),
              'old_frozen_domain_gap_seconds':str(gap_dec(t,d(1.-1e-9))),
              'old_split_distance_to_optimum_rps':str(abs(t-exact)),
              'old_actual_objective_excess_seconds':str((BG+SG*t)/(1-BG-SG*t)/LL+(BS+SS*(LP-t))/(1-BS-SS*(LP-t))/LL-(BG+SG*exact)/(1-BG-SG*exact)/LL-(BS+SS*(LP-exact))/(1-BS-SS*(LP-exact))/LL)}
    exact_u=vector(float(exact));root_u=vector(root)
    assert max(S@exact_u/g)<.8 and abs(root-float(exact))<1e-12
    assert certificate(exact_u)<=CERTIFICATE_TOLERANCE_SECONDS
    rerun=solve_e2e_routing(x,w,c,cal)
    oldgap=old['frozen_J_solver_diagnostic']['optimality_gap_seconds']
    decgap=float(high['old_frozen_domain_gap_seconds'])
    assert abs(decgap-oldgap)<1e-13
    classification='SMALL_OPTIMIZATION_RESIDUAL_CONFIRMED_NOT_FALSE_FLOATING_POINT_CERTIFICATE'
    out={'status':'CPU_NUMERICAL_AUDIT_COMPLETE','classification':classification,
        'physical_routing_infeasible':False,'operational_feasible':True,
        'scientific_certificate_tolerance_seconds':CERTIFICATE_TOLERANCE_SECONDS,
        'scientific_threshold_changed':False,'objective':'sum D/(1-D/g) / Lambda',
        'requested_cross_check_capacity_ceiling_rho':.8,'frozen_Module9_modified':False,
        'runtime_modified':False,'new_GPU_runs':0,'diagnostic_routing_adopted':False,
        'implementation_semantics':{'frozen_seed_and_candidate_gate_rho_max':.8,
            'frozen_J_solver_internal_ceiling':1.-1e-9,
            'audit_variants_all_use_rho_max':.8,
            'old_rejected_iterate_and_independent_optimum_both_below_point8':True},
        'old_failure':old['frozen_J_solver_diagnostic'],'frozen_solver_CPU_reproduction':rerun,
        'units_cross_check':runs,'decimal_cross_check':high,
        'independent_brent_root_G_asr_rps':root,
        'independent_optimum':{'J_seconds':obj(exact_u),'gap_seconds':certificate(exact_u),
            'max_rho':float(max(S@exact_u/g)),'constraints_valid':True,
            'G_asr_rps':float(exact),'S_asr_rps':lp-float(exact)},
        'old_iterate_operational_certificate_gap_seconds':certificate(vector(old_t)),
        'old_iterate_operational_certificate_pass':certificate(vector(old_t))<=CERTIFICATE_TOLERANCE_SECONDS,
        'old_iterate_frozen_domain_certificate_gap_seconds':certificate(vector(old_t),1.-1e-9),
        'conclusion':'The saved 12.24 ns gap agrees with a 60-digit calculation. It is a tiny solver stopping residual, not a physical or operational infeasibility and not merely a float64 subtraction artifact. Deterministic independent solves certify the same objective under rho<=0.8. No fallback was installed or selected.',
        'sources':[ref(ARCHIVE/'planning/H0_V3_MINIMAL.json'),ref(__file__),ref('deployment_core/routing/e2e_router.py')],
        'archive_hashes_verified':check_archive()}
    save('numerical_certificate_audit.json',out)
    print(__import__('json').dumps({'classification':classification,'old_gap_ns':oldgap*1e9,
        'old_operational_gap_ns':out['old_iterate_operational_certificate_gap_seconds']*1e9,
        'independent_optimum':out['independent_optimum'],
        'unit_gaps_ns':{r['objective_units']:r['repeats'][0]['gap_ns'] for r in runs}}),flush=True)

if __name__=='__main__':main()
