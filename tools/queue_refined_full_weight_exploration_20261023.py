"""CPU-only posthoc six-weight exploration of fixed measured conditions.

Business preferences are not inferred. Fifth-pair UNKNOWN is never imputed.
HiGHS is a numerical optimizer; exact rational witnesses verify feasibility
and actual margins, independently of the solver's success flag.
"""
from __future__ import annotations
import argparse
import csv
from fractions import Fraction
import itertools
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.queue_refined_weight_sensitivity_20261023 import (
    TERMS,ROLES,load_completed,pair_data,receipt,write,sha,finite)


def f(value):return Fraction.from_float(value) if type(value) is float else Fraction(value)
def display(value):return dict(exact=str(value),display=float(value))
def gains(a,b):
    """Return control-minus-spill costs: positive gains improve U."""
    if not all(finite(r.get(t)) for r in (a,b) for t in TERMS):return None
    exact={t:f(b[t])-f(a[t]) for t in TERMS}
    return dict({t:float(v) for t,v in exact.items()},_exact={t:str(v) for t,v in exact.items()})
def coefficient(g,t):return Fraction(g['_exact'][t]) if '_exact' in g else f(g[t])
def dot(g,w):return sum(coefficient(g,t)*w[t] for t in TERMS)
def mean_gain(rows):
    if not rows:return None
    exact={t:sum(coefficient(r,t) for r in rows)/len(rows) for t in TERMS}
    return dict({t:float(v) for t,v in exact.items()},_exact={t:str(v) for t,v in exact.items()})


def mean_preference_thresholds(rows):
    if not rows:return dict(status='UNKNOWN')
    m={t:sum(coefficient(r,t) for r in rows)/len(rows) for t in TERMS}
    energy=m['E'];others=sum(m[t] for t in TERMS if t!='E')/5
    slope=energy-others;largest=max(m.values());denominator=6*largest-sum(m.values())
    return dict(mean_gains_exact={t:str(v) for t,v in m.items()},mean_gains={t:float(v) for t,v in m.items()},
        energy_only_family=dict(formula='wE=e; other five weights=(1-e)/5; mean gain=other_mean+e*(E_gain-other_mean)',
            other_mean=display(others),slope=display(slope),break_even=display(-others/slope) if slope else None,
            strict_positive_side='e greater than threshold' if slope>0 else 'e below threshold' if slope<0 else 'constant'),
        equal_floor_cap1=dict(formula='max mean with every weight>=f and cap1: max_gain+f*(sum_gains-6*max_gain), for0<=f<=1/6',
            best_terms=[t for t in TERMS if m[t]==largest],best_gain=display(largest),slope=display(-denominator),
            strict_positive_floor_threshold=display(largest/denominator) if denominator else None,
            strict_positive_side='f below threshold' if denominator>0 else 'constant',
            interpretation='An algebraic preference requirement, not an endorsed business weight'))


def two_pair_obstruction(a,b):
    """Find an exact convex combination <=0 in every gain coordinate."""
    lo,hi=Fraction(0),Fraction(1)
    for t in TERMS:
        intercept=coefficient(b,t);slope=coefficient(a,t)-intercept
        if slope==0:
            if intercept>0:return None
        elif slope>0:hi=min(hi,-intercept/slope)
        else:lo=max(lo,-intercept/slope)
    if lo>hi:return None
    alpha=(lo+hi)/2;combination={t:alpha*coefficient(a,t)+(1-alpha)*coefficient(b,t) for t in TERMS}
    if not all(v<=0 for v in combination.values()):return None
    return dict(alpha=display(alpha),beta=display(1-alpha),combined_gains={t:display(v) for t,v in combination.items()},
        nonnegative_combination_verified=True,all_coordinate_gains_nonpositive=True,
        proof='For any nonnegative weights, alpha*delta_A+beta*delta_B<=0. If alpha,beta>0, both deltas cannot be strictly positive.',
        both_multipliers_strictly_positive=0<alpha<1)


def pareto_certificate(g):
    if g is None:return dict(status='UNKNOWN')
    exact={t:coefficient(g,t) for t in TERMS}
    control_dominates=all(v<=0 for v in exact.values()) and any(v<0 for v in exact.values())
    spill_dominates=all(v>=0 for v in exact.values()) and any(v>0 for v in exact.values())
    return dict(status='CONTROL_WEAKLY_DOMINATES_SPILL' if control_dominates else 'SPILL_WEAKLY_DOMINATES_CONTROL' if spill_dominates else 'EQUAL' if all(v==0 for v in exact.values()) else 'TRADEOFF',
        gains=g,negative_terms=[t for t,v in exact.items() if v<0],zero_terms=[t for t,v in exact.items() if v==0],
        positive_terms=[t for t,v in exact.items() if v>0],
        strict_positive_spill_gain_impossible_for_all_nonnegative_weights=control_dominates or all(v==0 for v in exact.values()),
        proof='If every gain coordinate <=0, any nonnegative weighted sum <=0; positive spill benefit is impossible, independent of the LP solver.')


def exact_witness(raw,floor,cap):
    """Tiny explicit projection into an exact bounded rational simplex."""
    lo,hi=f(floor),f(cap);w={t:f(float(v)) for t,v in zip(TERMS,raw)}
    if 6*lo>1 or 6*hi<1:return None
    w={t:min(hi,max(lo,v)) for t,v in w.items()}
    remainder=Fraction(1)-sum(w.values())
    for t in TERMS:
        change=min(remainder,hi-w[t]) if remainder>0 else max(remainder,lo-w[t])
        w[t]+=change;remainder-=change
        if remainder==0:break
    if remainder:return None
    displacement=max(abs(w[t]-f(float(v))) for t,v in zip(TERMS,raw))
    if displacement>f(1e-7):return None
    return dict(weights=w,max_projection_displacement=displacement)


def solve_lp(rows,*,floor=0.,cap=1.,objective='maximin',target=0.):
    """Numerical optimum plus exact feasible-weight and margin verification."""
    from scipy.optimize import linprog
    if not rows or any(r is None for r in rows):return dict(status='UNKNOWN',objective=objective)
    if not all(finite(x) and x>=0 for x in (floor,cap)) or floor>cap:raise ValueError('Invalid symmetric bounds')
    if 6*f(floor)>1 or 6*f(cap)<1:return dict(status='INFEASIBLE_BOX',floor=floor,cap=cap,objective=objective)
    m=mean_gain(rows);n=6;A=[];b=[];eq=[[1.]*6];rhs=[1.];bounds=[(floor,cap)]*6
    if objective=='maximin':
        c=[0.]*6+[-1.];bounds+=[(None,None)];eq=[eq[0]+[0.]]
        A=[[-r[t] for t in TERMS]+[1.] for r in rows];b=[0.]*len(rows)
    elif objective=='mean':c=[-m[t] for t in TERMS]
    elif objective=='nearest_mean':
        c=[0.]*6+[1.]*6;bounds += [(0,None)]*6;eq=[eq[0]+[0.]*6]
        A=[[-m[t] for t in TERMS]+[0.]*6];b=[-target]
        for i in range(6):
            upper=[0.]*12;upper[i]=1.;upper[6+i]=-1.;A.append(upper);b.append(1/6)
            lower=[0.]*12;lower[i]=-1.;lower[6+i]=-1.;A.append(lower);b.append(-1/6)
    else:raise ValueError('Unknown LP objective')
    result=linprog(c,A_ub=A or None,b_ub=b or None,A_eq=eq,b_eq=rhs,bounds=bounds,method='highs')
    report=dict(status='SOLVER_OPTIMAL' if result.success else 'SOLVER_INFEASIBLE' if result.status==2 else 'SOLVER_OTHER',
        objective=objective,target=target,floor=floor,cap=cap,solver_status=int(result.status),solver_message=result.message,
        mathematical_scope='Numerical HiGHS optimum; exact rational witness below certifies its feasibility and realized gains, not exact optimality')
    if not result.success:return report
    raw=list(map(float,result.x[:6]));report.update(raw_solver_weights=dict(zip(TERMS,raw)),solver_objective=float(result.fun),
        solver_equality_residuals=list(map(float,result.eqlin.residual)),solver_inequality_residuals=list(map(float,result.ineqlin.residual)))
    witness=exact_witness(raw,floor,cap)
    if witness is None:report['exact_witness_status']='NO_SMALL_EXACT_SIMPLEX_PROJECTION';return report
    w=witness['weights'];values=[dot(g,w) for g in rows];mean_value=sum(values)/len(values)
    if objective=='nearest_mean' and mean_value<f(target):
        # A solver boundary solution may undershoot by a rounding bit. Move
        # exactly along a feasible segment to the exact mean halfspace;
        # accept this only when the total correction remains <=1e-7.
        exact_mean={t:sum(coefficient(r,t) for r in rows)/len(rows) for t in TERMS}
        best={t:f(floor) for t in TERMS};remaining=1-sum(best.values())
        for t in sorted(TERMS,key=lambda t:exact_mean[t],reverse=True):
            change=min(remaining,f(cap)-best[t]);best[t]+=change;remaining-=change
        best_value=sum(exact_mean[t]*best[t] for t in TERMS)
        if best_value>=f(target) and best_value>mean_value:
            step=(f(target)-mean_value)/(best_value-mean_value)
            adjusted={t:(1-step)*w[t]+step*best[t] for t in TERMS}
            displacement=max(abs(adjusted[t]-f(v)) for t,v in zip(TERMS,raw))
            if displacement<=f(1e-7):
                w=adjusted;witness['max_projection_displacement']=displacement
                values=[dot(g,w) for g in rows];mean_value=sum(values)/len(values)
    bound_slacks={t:dict(lower=display(w[t]-f(floor)),upper=display(f(cap)-w[t])) for t in TERMS}
    feasible=sum(w.values())==1 and all(f(floor)<=w[t]<=f(cap) for t in TERMS)
    target_ok=mean_value>=f(target) if objective=='nearest_mean' else True
    report.update(exact_witness_status='EXACT_FEASIBLE' if feasible and target_ok else 'EXACT_TARGET_NOT_MET',
        weights={t:float(w[t]) for t in TERMS},weights_exact={t:str(w[t]) for t in TERMS},
        exact_simplex_sum=display(sum(w.values())),bound_slacks=bound_slacks,
        projection_max_change=display(witness['max_projection_displacement']),per_pair_gains=[display(v) for v in values],
        minimum_gain=display(min(values)),mean_gain=display(mean_value),
        all_known_pairs_strictly_positive=feasible and all(v>0 for v in values),mean_strictly_positive=feasible and mean_value>0,
        positive_pairs=sum(v>0 for v in values),known_pairs=len(values),
        L1_distance_to_original_equal=display(sum(abs(v-Fraction(1,6)) for v in w.values())),
        target_constraint_slack=display(mean_value-f(target)) if objective=='nearest_mean' else None)
    if objective=='maximin':report['achieved_margin_minus_solver_margin']=display(min(values)-f(float(result.x[6])))
    return report


def family_analysis(rows,floor,cap,positive_margin):
    return dict(floor=floor,cap=cap,preference_design='Symmetric diagnostic bounds only; not a business definition of reasonable weights',
        maximin=solve_lp(rows,floor=floor,cap=cap),maximum_mean=solve_lp(rows,floor=floor,cap=cap,objective='mean'),
        nearest_equal_mean_nonnegative=solve_lp(rows,floor=floor,cap=cap,objective='nearest_mean',target=0.),
        nearest_equal_mean_positive_margin=solve_lp(rows,floor=floor,cap=cap,objective='nearest_mean',target=positive_margin))


def development_analysis(native,families,positive_margin):
    result=[]
    for load in sorted({r['load'] for r in native if r['stage'].startswith('development')}):
        bykey={(r['role'],r['summary']['seed']):r for r in native if r['stage'].startswith('development') and r['load']==load}
        if set(bykey)!=set(itertools.product(ROLES,[203,204])):raise ValueError('Incomplete development four-condition domain')
        for role in ROLES:
            comparisons=[];labels=[]
            for other,seed in itertools.product([r for r in ROLES if r!=role],[203,204]):
                a,b=bykey[role,seed]['summary']['utility'],bykey[other,seed]['summary']['utility']
                known=a.get('eligible') is True and b.get('eligible') is True
                g=gains({t:a.get(t+'_norm') for t in TERMS},{t:b.get(t+'_norm') for t in TERMS}) if known else None
                comparisons.append(g);labels.append(dict(candidate=role,alternative=other,seed=seed,gains=g))
            result.append(dict(load=load,candidate_role=role,deployment=bykey[role,203]['contract']['deployment'],
                contrasts=labels,known_contrasts=sum(g is not None for g in comparisons),
                families=[dict(floor=lo,cap=hi,maximin=solve_lp(comparisons,floor=lo,cap=hi)) for lo,hi in families],
                scope='Exploratory two-development-seed dominance over three measured alternatives; not independent confirmation or new deployment search'))
    return result


def csv_write(path,rows):
    if not rows:return
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def number(r,key):return r.get(key,{}).get('display')
def fmt(v):return f'{v:.6g}' if finite(v) else 'UNKNOWN'


def analyze(execution,data,plan,output,*,allow_interrupted=False,positive_margin=1e-6):
    if not finite(positive_margin) or positive_margin<=0:raise ValueError('Explicit positive diagnostic margin required')
    native,_,sources=load_completed(execution,data,plan,allow_interrupted)
    pairs=pair_data(native,allow_interrupted);known=[];pair_records=[]
    for pair in pairs:
        g=gains(pair['spill_terms'],pair['control_terms']) if pair['original_eligible_pair'] else None
        if g is not None:known.append(g)
        pair_records.append(dict(seed=pair['seed'],phase=pair['phase'],gains=g,pareto=pareto_certificate(g),
            original_equal_gain=display(dot(g,{t:Fraction(1,6) for t in TERMS})) if g is not None else None,
            spill_raw=pair['spill_raw'],control_raw=pair['control_raw']))
    families=list(itertools.product((0.,.02,.05,.10),(1.,.5,.35)))
    whole=family_analysis(known,0.,1.,positive_margin)
    constrained=[family_analysis(known,lo,hi,positive_margin) for lo,hi in families if (lo,hi)!=(0.,1.)]
    leave_one_out=[]
    for excluded in [r for r in pair_records if r['gains'] is not None]:
        retained=[r for r in pair_records if r['gains'] is not None and r['seed']!=excluded['seed']]
        retained_gains=[r['gains'] for r in retained];average=mean_gain(retained_gains)
        leave_one_out.append(dict(excluded_seed=excluded['seed'],retained_seeds=[r['seed'] for r in retained],
            mean_gains=average,mean_pareto=pareto_certificate(average),maximum_mean=solve_lp(retained_gains,objective='mean'),
            full_fit_weights_evaluated={name:dict(mean_gain=display(sum(dot(r,{t:Fraction(v) for t,v in fit['weights_exact'].items()}) for r in retained_gains)/len(retained_gains)),
                positive_pairs=sum(dot(r,{t:Fraction(v) for t,v in fit['weights_exact'].items()})>0 for r in retained_gains),known_pairs=len(retained_gains))
                for name,fit in whole.items() if isinstance(fit,dict) and 'weights_exact' in fit},
            interpretation='Whole-pair leave-one-out descriptive sensitivity, no request-level replication or Gaussian confidence interval'))
    result=dict(status='COMPLETE_EXPLORATORY_FULL_WEIGHT_ANALYSIS',sources=sources+[receipt(__file__)],
        scipy_version=__import__('scipy').__version__,known_seed_order=[r['seed'] for r in pair_records if r['gains'] is not None],
        declared_seed_order=[r['seed'] for r in pair_records],all_five_stable_positive='UNKNOWN' if len(known)<5 else None,
        all_five_reason='Missing/ineligible pairs remain UNKNOWN; known4 subset never replaces registered5 endpoint',
        pairs=pair_records,leave_one_pair_out=leave_one_out,mean_preference_thresholds=mean_preference_thresholds(known),
        two_pair_obstructions=[dict(seed_A=a['seed'],seed_B=b['seed'],certificate=cert)
            for a,b in itertools.combinations([p for p in pair_records if p['gains'] is not None],2)
            if (cert:=two_pair_obstruction(a['gains'],b['gains'])) is not None],
        full_simplex=whole,restricted_symmetric_boxes=constrained,
        diagnostic_positive_margin=positive_margin,positive_margin_design='Explicit exploratory utility margin; not a business tolerance or a statistical significance threshold',
        development=development_analysis(native,families,positive_margin),
        interpretation='Fixed measured outcomes under posthoc preference exploration. No GPU, new policy optimization, physical performance improvement, or original-protocol success claim.',
        future_requirement='Business weights require external preferences. A selected objective and reoptimized policy require a separately frozen new experiment.',GPU_used=False)
    if len(known)==5:result['all_five_stable_positive']=whole['maximin'].get('all_known_pairs_strictly_positive')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);write(output/'analysis.json',result)
    csv_write(output/'pair-gains.csv',[dict(seed=p['seed'],phase=p['phase'],pareto=p['pareto']['status'],original_equal_gain=p['original_equal_gain']['display'] if p['original_equal_gain'] else None,**{t:(p['gains'][t] if p['gains'] else None) for t in TERMS}) for p in pair_records])
    csv_write(output/'weight-solutions.csv',[dict(floor=fam['floor'],cap=fam['cap'],objective=name,status=r['status'],exact_status=r.get('exact_witness_status'),minimum_gain=number(r,'minimum_gain'),mean_gain=number(r,'mean_gain'),positive_pairs=r.get('positive_pairs'),all_known_positive=r.get('all_known_pairs_strictly_positive'),**{t:r.get('weights',{}).get(t) for t in TERMS}) for fam in [whole,*constrained] for name,r in fam.items() if isinstance(r,dict) and 'objective' in r])
    csv_write(output/'development-robust-feasibility.csv',[dict(load=d['load'],role=d['candidate_role'],floor=fam['floor'],cap=fam['cap'],status=fam['maximin']['status'],exact_status=fam['maximin'].get('exact_witness_status'),minimum_gain=number(fam['maximin'],'minimum_gain'),all_six_contrasts_positive=fam['maximin'].get('all_known_pairs_strictly_positive'),**{t:fam['maximin'].get('weights',{}).get(t) for t in TERMS}) for d in result['development'] for fam in d['families']])
    lines=['# 六项权重探索性分析','',
        '本分析只调整对固定实测结果的偏好，不改变性能、精度、稳定性或路由。没有预设业务上何种权重合理。全单纯形与对称floor/cap诊断分开；后者是设计敏感性，不是业务偏好。','',
        f'已知配对种子：{result["known_seed_order"]}。全部5对稳定正收益：{result["all_five_stable_positive"]}。缺失第5对不能通过调权补齐。','',
        'gain=控制归一成本−分流归一成本，ΔU=Σw×gain；w≥0且Σw=1。','',
        '|种子|Pareto关系|等权ΔU|','|---|---|---|']
    for pair in pair_records:lines.append('|'+ '|'.join([str(pair['seed']),pair['pareto']['status'],fmt(pair['original_equal_gain']['display'] if pair['original_equal_gain'] else None)])+'|')
    impossible=[p['seed'] for p in pair_records if p['pareto'].get('strict_positive_spill_gain_impossible_for_all_nonnegative_weights')]
    if impossible:lines+=['',f'种子{impossible}的六项gain均非正，因此任何非负权重都不能使这些配对严格获益；这个结论来自逐项支配证书，不依赖求解器，也不能靠重新分配权重绕过。']
    thresholds=result['mean_preference_thresholds'];energy=thresholds['energy_only_family'];floor=thresholds['equal_floor_cap1']
    lines+=['',f'均值偏好解析：只改变E且其余五项等权时，E权重盈亏平衡点={fmt(energy["break_even"]["display"] if energy["break_even"] else None)}，严格正收益方向为{energy["strict_positive_side"]}。',
        f'每项均至少为f、单项上限1时，最大均值收益的floor盈亏平衡点={fmt(floor["strict_positive_floor_threshold"]["display"] if floor["strict_positive_floor_threshold"] else None)}，方向为{floor["strict_positive_side"]}。精确分数和代数公式见JSON；极端偏好不等于合理业务权重。']
    for obstruction in result['two_pair_obstructions']:
        if obstruction['certificate']['both_multipliers_strictly_positive']:
            lines+=['',f'种子{obstruction["seed_A"]}/{obstruction["seed_B"]}存在非负组合支配证书：两者收益的正系数组合在全部指标上非正，因此同一非负权重不能使两对都严格获益。']
    for omitted in leave_one_out:
        if omitted['mean_pareto'].get('strict_positive_spill_gain_impossible_for_all_nonnegative_weights'):
            lines+=['',f'删去整次配对种子{omitted["excluded_seed"]}后，余下种子{omitted["retained_seeds"]}的均值gain在六项上均非正，任何非负权重都不能使这个剩余均值严格获益。因此原子集均值正收益依赖被删配对，不能据此称稳定。']
    lines+=['','|权重域floor/cap|目标|最差已知配对ΔU|已知均值ΔU|正收益配对数|全部已知配对严格正|精确见证|','|---|---|---|---|---|---|---|']
    for fam in [whole,*constrained]:
        for name,r in fam.items():
            if not isinstance(r,dict) or 'objective' not in r:continue
            lines.append('|'+ '|'.join([str(fam['floor'])+'/'+str(fam['cap']),name,fmt(number(r,'minimum_gain')),fmt(number(r,'mean_gain')),str(r.get('positive_pairs'))+'/'+str(r.get('known_pairs')),str(r.get('all_known_pairs_strictly_positive')),r.get('exact_witness_status',r['status'])])+'|')
    lines+=['','HiGHS成功只说明数值优化状态。JSON同时保存原始solver权重/残差、最多1e−7的显式有界单纯形投影、精确分数权重、约束余量及每个配对的精确收益；正收益判断来自见证重算。数值最优不等于精确最优证明。','',
        f'最近等权解使用L1距离，分别要求已知均值≥0和≥{positive_margin}；这个正裕量是探索性数值设置，不是统计显著性，也不代表逐对稳定收益。','',
        '开发阶段另对两个负载的四个实际方案逐个检查：同一权重能否在203/204两个种子上同时优于其他三个方案。结果仅覆盖六个已测对比，不证明全模型池最优。见development-robust-feasibility.csv。','',
        '数据：analysis.json含全部SHA、权重、残差、精确约束证书与逐配对结果；pair-gains.csv、weight-solutions.csv、development-robust-feasibility.csv供直接比较。若采用新权重，需先确认业务偏好并另行冻结目标与后续实验。']
    (output/'六项权重探索报告.md').write_text('\n'.join(lines)+'\n')
    write(output/'manifest.json',dict(outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()},sources=result['sources']))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-interrupted',action='store_true');p.add_argument('--positive-margin',type=float,default=1e-6)
    a=p.parse_args();r=analyze(a.execution,a.data,a.plan,a.output,allow_interrupted=a.allow_interrupted,positive_margin=a.positive_margin)
    print(json.dumps(dict(status=r['status'],known_seeds=r['known_seed_order'],all_five=r['all_five_stable_positive']),ensure_ascii=False))
