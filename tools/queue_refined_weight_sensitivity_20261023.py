"""Posthoc preference sensitivity after the entire frozen native workflow finishes.

No GPU, policy refitting, workload reselection, or missing-term imputation.
Measured service, quality, stability and routing remain unchanged.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from fractions import Fraction

TERMS='QTLPME'
OTHER='QTME'
SEEDS=(205,206,207,208,209)
ROLES=('calibrated_spill','risk_matched_no_spill','legacy_latency','strong_other_deployment')
PHASES={'validation':(205,206),'confirmation':(207,208,209),'all5':SEEDS}


def finite(x):return type(x) in (int,float) and math.isfinite(x)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def receipt(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def average(values):return statistics.mean(values) if values and all(finite(x) for x in values) else None
def weights(l,p):
    if not all(finite(x) and 0<=x<=1 for x in (l,p)):raise ValueError('L/P scales must be finite in [0,1]')
    denominator=4+l+p
    return {t:(l if t=='L' else p if t=='P' else 1.)/denominator for t in TERMS}
def utility(terms,l,p,eligible=True):
    # Even a term with zero preference weight remains unknown if unmeasured.
    if eligible is not True or not all(finite(terms.get(t)) for t in TERMS):return None
    w=weights(l,p);return -math.fsum(w[t]*terms[t] for t in TERMS)
def coefficients(spill,control):
    if not all(finite(v.get(t)) for v in (spill,control) for t in TERMS):return None
    d={t:spill[t]-control[t] for t in TERMS}
    return dict(A=-math.fsum(d[t] for t in OTHER),B=-d['L'],C=-d['P'],delta_norm=d)
def delta_value(coef,l,p):
    weights(l,p)
    return None if coef is None else (coef['A']+coef['B']*l+coef['C']*p)/(4+l+p)

def mean_coefficients(coefs):
    if not coefs or any(c is None for c in coefs):return None
    return {k:statistics.mean(c[k] for c in coefs) for k in ('A','B','C')}


def rational(value):
    # Preserve the exact binary coefficient supplied by the measured float.
    return Fraction.from_float(value) if type(value) is float else Fraction(value)

def exact_coefficients(coefs):
    return [{k:rational(c[k]) for k in ('A','B','C')} for c in coefs]

def axis_interval(coefs,axis):
    """Strict linear intervals computed in exact rational arithmetic."""
    if any(c is None for c in coefs):return dict(status='UNKNOWN')
    lo=Fraction(0);hi=Fraction(1);li=hi_i=True;roots=[]
    for c in exact_coefficients(coefs):
        if axis=='L_only':a,b=c['A']+c['C'],c['B']
        elif axis=='P_only':a,b=c['A']+c['B'],c['C']
        elif axis=='both':a,b=c['A'],c['B']+c['C']
        else:raise ValueError('Unknown axis')
        if b==0:
            roots.append(dict(root=None,constant_numerator=float(a),constant_numerator_exact=str(a)))
            if a<=0:return dict(status='EMPTY',reason='A constant strict inequality is nonpositive',roots=roots)
            continue
        root=-a/b;roots.append(dict(root=float(root),root_exact=str(root),positive_side='above' if b>0 else 'below'))
        if b>0:
            if root>lo:lo=root;li=False
            elif root==lo:li=False
        else:
            if root<hi:hi=root;hi_i=False
            elif root==hi:hi_i=False
    if lo>hi or (lo==hi and not(li and hi_i)):return dict(status='EMPTY',roots=roots)
    return dict(status='NONEMPTY',lower=float(lo),upper=float(hi),lower_exact=str(lo),upper_exact=str(hi),
        lower_included=li,upper_included=hi_i,roots=roots,
        arithmetic='Exact rationals of supplied binary-float coefficients; display floats may collapse a narrow interval')


def clip_polygon(poly,a,b,c):
    """Exact rational clipping by a+b*x+c*y >= 0 (closed geometry)."""
    out=[]
    if not poly:return out
    for u,v in zip(poly,poly[1:]+poly[:1]):
        fu=a+b*u[0]+c*u[1];fv=a+b*v[0]+c*v[1]
        if fu>=0:out.append(u)
        if (fu>=0)!=(fv>=0):
            ratio=fu/(fu-fv);out.append((u[0]+ratio*(v[0]-u[0]),u[1]+ratio*(v[1]-u[1])))
    return out

def region(coefs):
    if not coefs or any(c is None for c in coefs):return dict(status='UNKNOWN',closure_polygon=None)
    exact=exact_coefficients(coefs)
    poly=[(Fraction(0),Fraction(0)),(Fraction(1),Fraction(0)),(Fraction(1),Fraction(1)),(Fraction(0),Fraction(1))]
    for c in exact:poly=clip_polygon(poly,c['A'],c['B'],c['C'])
    if not poly:return dict(status='EMPTY',closure_polygon=[],closure_polygon_exact=[])
    witness=tuple(sum(v[i] for v in poly)/len(poly) for i in (0,1))
    margins=[c['A']+c['B']*witness[0]+c['C']*witness[1] for c in exact]
    # A rational relative-interior point avoids losing strict narrow regions
    # when a midpoint rounds onto one of their boundaries as a float.
    exists=all(v>0 for v in margins)
    return dict(status='NONEMPTY' if exists else 'EMPTY_STRICT_INTERIOR',
        closure_polygon=[[float(x),float(y)] for x,y in poly],
        closure_polygon_exact=[[str(x),str(y)] for x,y in poly],
        strict_witness=[float(v) for v in witness] if exists else None,
        strict_witness_exact=[str(v) for v in witness] if exists else None,
        witness_numerator_margins=[float(v) for v in margins],
        witness_numerator_margins_exact=[str(v) for v in margins],
        constraint='A+B*L_scale+C*P_scale > 0 for each whole-run pair; square edges0/1 allowed, break-even lines excluded',
        geometry_scope='Exact rational feasibility for supplied binary-float coefficients; float display is approximate and no statistical confidence region is claimed')


def pair_data(native,allow_interrupted=False):
    reserved=[r for r in native if not r['stage'].startswith('development')]
    pairs=[]
    for seed in SEEDS:
        rows=[r for r in reserved if r['summary']['seed']==seed]
        byrole={r['role']:r for r in rows}
        if len(rows)!=len(byrole):raise ValueError('Duplicate registered reserved pair')
        if set(byrole)!={'calibrated_spill','risk_matched_no_spill'} and not (allow_interrupted and seed==209 and set(byrole)=={'calibrated_spill'}):raise ValueError('Incomplete registered reserved pair')
        a=byrole['calibrated_spill'];b=byrole.get('risk_matched_no_spill')
        if b and a['load']!=b['load']:raise ValueError('Pair workload mismatch')
        ua,ub=a['summary']['utility'],b['summary']['utility'] if b else {}
        ta={t:ua.get(t+'_norm') for t in TERMS};tb={t:ub.get(t+'_norm') for t in TERMS}
        known=ua.get('eligible') is True and ub.get('eligible') is True
        pairs.append(dict(seed=seed,phase='validation' if seed in (205,206) else 'confirmation',load=a['load'],
            spill_terms=ta,control_terms=tb,coefficients=coefficients(ta,tb) if known else None,
            original_eligible_pair=known,spill_raw=a['summary']['raw'],control_raw=b['summary']['raw'] if b else None,
            pair_status='COMPLETE_EVALUABLE' if known else ('CONTROL_NOT_EXECUTED' if b is None else 'UNKNOWN_EVIDENCE')))
    return pairs

def point_summary(pairs,l,p):
    values={str(r['seed']):delta_value(r['coefficients'],l,p) for r in pairs}
    groups={}
    for name,seeds in PHASES.items():
        v=[values[str(s)] for s in seeds];known=all(finite(x) for x in v)
        groups[name]=dict(mean_delta_U=average(v),known_pairs=sum(finite(x) for x in v),expected_pairs=len(seeds),
            positive_pairs=sum(x>0 for x in v if finite(x)),all_positive=all(x>0 for x in v) if known else None,
            minimum_delta_U=min(v) if known else None,maximum_delta_U=max(v) if known else None)
    available=[v for v in values.values() if finite(v)]
    groups['evaluable_only']=dict(mean_delta_U=average(available),known_pairs=len(available),expected_pairs=len(available),
        positive_pairs=sum(x>0 for x in available),all_positive=all(x>0 for x in available) if available else None,
        minimum_delta_U=min(available) if available else None,maximum_delta_U=max(available) if available else None,
        scope='Available complete pairs only; cannot replace the registered5-pair endpoint')
    return dict(L_scale=l,P_scale=p,weights=weights(l,p),per_seed_delta_U=values,phases=groups)

def rank_development(native,l,p):
    result=[]
    for load in sorted({r['load'] for r in native if r['stage'].startswith('development')}):
        entries=[]
        for role in ROLES:
            rows=[r for r in native if r['stage'].startswith('development') and r['load']==load and r['role']==role]
            seeds=[r['summary']['seed'] for r in rows]
            if sorted(seeds)!=[203,204]:raise ValueError('Development role must have exactly seeds203/204')
            values={str(r['summary']['seed']):utility({t:r['summary']['utility'].get(t+'_norm') for t in TERMS},l,p,r['summary']['utility'].get('eligible')) for r in rows}
            entries.append(dict(role=role,deployment=rows[0]['contract']['deployment'],mean_U=average(list(values.values())),per_seed_U=values,rank=None))
        known=[r for r in entries if r['mean_U'] is not None]
        known.sort(key=lambda r:(-r['mean_U'],r['role']))
        # Exact maximum anchors each absolute1e-12 tie band.
        rank=1;remaining=list(known)
        while remaining:
            maximum=max(r['mean_U'] for r in remaining);band=[r for r in remaining if maximum-r['mean_U']<=1e-12]
            for r in band:r['rank']=rank
            remaining=[r for r in remaining if r not in band];rank+=len(band)
        result.append(dict(load=load,L_scale=l,P_scale=p,weights=weights(l,p),ranking=sorted(entries,key=lambda r:(r['rank'] is None,r['rank'] or 0,r['role'])),
            whole_four_condition_domain_known=len(known)==4,scope='Reweighting four actually measured conditions only; no policy reoptimization or full-pool optimum'))
    return result


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def checked_bound(bound,sources,*,parse=True):
    if set(bound)!={'path','sha256'} or receipt(bound['path'])!=bound:raise ValueError('Workflow receipt SHA changed')
    sources.append(bound);return read(bound['path']) if parse else bound

def assert_matched_pair(a,b):
    if a['load']!=b['load']:raise ValueError('Pair workload mismatch')
    ca,cb=a['contract'],b['contract'];ra,rb=ca.get('refinement'),cb.get('refinement')
    if not ra or not rb or ra.get('allow_spill') is not True or rb.get('allow_spill') is not False:
        raise ValueError('Pair allow_spill identity mismatch')
    if {k:v for k,v in ra.items() if k!='allow_spill'}!={k:v for k,v in rb.items() if k!='allow_spill'}:
        raise ValueError('Pair mode/risk/model/margin mismatch')
    if {k:v for k,v in ca.items() if k not in {'strategy','guard','refinement'}}!={k:v for k,v in cb.items() if k not in {'strategy','guard','refinement'}}:
        raise ValueError('Pair deployment/background/contract mismatch')
    guard=ca.get('guard')
    if (cb.get('guard') is not None or not guard or guard.get('task')!='text' or guard.get('family')!='generalist_2'
        or guard.get('rate_rps')!=a['load']/8 or guard.get('burst')!=1):raise ValueError('Pair guard mismatch')
    if ca.get('strategy')!='queue_quality' or cb.get('strategy')!='queue_specialist_only':raise ValueError('Pair strategy mismatch')


def load_completed(execution,data,plan_path,allow_interrupted=False):
    """Bind every row to a unique registered raw and frozen packet job."""
    state=read(Path(execution)/'progress.json')
    interrupted=state.get('status')=='STOPPED_FOR_DIAGNOSIS_NO_RETRY'
    if state.get('status')!='COMPLETE' and not (allow_interrupted and interrupted):raise ValueError('Entire original execution must be COMPLETE before posthoc analysis, or explicitly allow a stopped terminal')
    doc=read(data)
    expected_status='COMPLETE_INTERRUPTED_AUDITED' if interrupted else 'COMPLETE_AUDITED'
    if doc.get('status')!=expected_status or doc.get('issues'):raise ValueError('Completed original final audit required')
    plan=read(plan_path)
    if plan.get('run_after_current_native_completion') is not True or not plan.get('current_experiment_weights_unchanged'):raise ValueError('Invalid post-experiment plan')
    native=doc['native']
    expected_count=25 if interrupted else 26
    if len(native)!=expected_count:raise ValueError('Unexpected registered native result count')
    sources=[receipt(Path(execution)/'progress.json'),receipt(data),receipt(plan_path),receipt(__file__)]
    expected_stages={'development-text1.85':('development',1.85,(203,204),ROLES),
        'development-text1.89':('development',1.89,(203,204),ROLES),
        'independent_validation':('independent_validation',None,(205,206),ROLES[:2]),
        'final_confirmation':('final_confirmation',None,(207,208,209),ROLES[:2])}
    stages=state.get('stages',[])
    if len(stages)!=4 or {z['label'] for z in stages}!=set(expected_stages):raise ValueError('Missing/extra/duplicate workflow stages')
    for bound in state.get('bindings',[]):checked_bound(bound,sources,parse=False)
    registered={};job_ids=set();reserved_contracts={};unexecuted=[];packet_rows=[]
    for stage in stages:
        packet=checked_bound(stage['packet'],sources)
        progress_receipt=receipt(stage['progress'])
        stopped_stage=interrupted and stage['label']=='final_confirmation'
        if stage.get('terminal_progress')!=progress_receipt and not (stopped_stage and 'terminal_progress' not in stage):raise ValueError('Terminal stage progress receipt mismatch')
        progress=checked_bound(progress_receipt,sources)
        if progress.get('status')!=('STOPPED_FOR_DIAGNOSIS_NO_RETRY' if stopped_stage else 'COMPLETE') or progress.get('packet')!=stage['packet']:raise ValueError('Stage progress/packet mismatch')
        phase,load,seeds,roles=expected_stages[stage['label']]
        jobs=packet['jobs'];expected=set(itertools.product(seeds,roles))
        if (packet.get('phase')!=phase or len(jobs)!=len(expected)
            or {(j['seed'],j['role']) for j in jobs}!=expected):raise ValueError('Missing/extra/duplicate packet job')
        byjob={j['id']:j for j in jobs}
        if len(byjob)!=len(jobs) or any(jid in job_ids for jid in byjob):raise ValueError('Duplicate workflow job identity')
        job_ids.update(byjob)
        for job in jobs:
            contract=job['policy_contract'];ident=job['identity'];rate=contract['text_rate']
            if (job['id']!=identity(ident) or ident.get('role')!=job['role'] or ident.get('seed')!=job['seed']
                or ident.get('phase')!=phase or job['phase']!=phase or ident.get('policy_contract_sha256')!=identity(contract)
                or (load is not None and rate!=load)):raise ValueError('Frozen packet job identity mismatch')
            packet_rows.append(dict(stage=stage['label'],role=job['role'],load=rate,contract=contract,seed=job['seed']))
            if phase!='development':
                previous=reserved_contracts.setdefault(job['role'],contract)
                if previous!=contract:raise ValueError('Reserved policy changed across seeds/phases')
        regs=progress['jobs']
        registered_ids={j['job_id'] for j in regs}
        missing=set(byjob)-registered_ids
        if len(registered_ids)!=len(regs) or registered_ids-set(byjob):raise ValueError('Missing/extra/duplicate registered job')
        if missing:
            if not stopped_stage or len(missing)!=1 or any(byjob[j]['seed']!=209 or byjob[j]['role']!='risk_matched_no_spill' for j in missing):raise ValueError('Unexpected unexecuted jobs')
            unexecuted.extend(dict(stage=stage['label'],role=byjob[j]['role'],seed=byjob[j]['seed'],job_id=j,status='UNEXECUTED_NO_RETRY') for j in missing)
        elif stopped_stage:raise ValueError('Interrupted dataset must retain the unexecuted registered endpoint')
        for reg in regs:
            job=byjob[reg['job_id']];raw=checked_bound(reg['raw'],sources);key=reg['raw']['path']
            if key in registered:raise ValueError('Duplicate registered raw')
            contract=job['policy_contract'];ident=job['identity'];rate=contract['text_rate'];seed=job['seed']
            if (job['id']!=identity(ident) or ident.get('role')!=job['role'] or ident.get('seed')!=seed
                or ident.get('phase')!=phase or job['phase']!=phase or ident.get('policy_contract_sha256')!=identity(contract)
                or (load is not None and rate!=load)):raise ValueError('Frozen packet job identity mismatch')
            workload=ident['workload']
            if (workload!={'text':rate,'image':.01,'ocr':.01,'asr':.01}
                or raw.get('experiment_job_id')!=job['id'] or raw.get('executed_measurement_identity')!=ident
                or raw.get('executed_measurement_id')!=identity(ident) or raw.get('experiment_phase')!=phase
                or raw.get('policy_contract')!=contract or raw.get('source_packet')!=job.get('base_packet',packet.get('base_packet'))
                or raw.get('transport_config',{}).get('random_seed')!=seed
                or raw.get('transport_config',{}).get('workload_rps')!=workload
                or raw.get('config',{}).get('random_seed')!=seed or raw.get('config',{}).get('workload')!=workload):
                raise ValueError('Raw execution identity/seed/workload/contract mismatch')
            if phase!='development':
                previous=reserved_contracts.setdefault(job['role'],contract)
                if previous!=contract:raise ValueError('Reserved policy changed across seeds/phases')
            registered[key]=dict(raw=raw,bound=reg['raw'],job=job,stage=stage['label'])
    if len(registered)!=expected_count:raise ValueError('Unexpected uniquely registered raw count')
    if interrupted:
        if doc.get('interrupted') is not True or doc.get('expected_formal_runs')!=26 or doc.get('unexecuted_jobs')!=unexecuted:raise ValueError('Interrupted audit unexecuted identity mismatch')
    seen=set()
    for row in native:
        bound=row['summary']['raw'];key=bound['path']
        if key in seen or key not in registered:raise ValueError('Duplicate/unregistered report raw')
        seen.add(key);entry=registered[key];raw,job=entry['raw'],entry['job'];contract=job['policy_contract'];summary=row['summary']
        symbolic=dict(deployment=contract['deployment'],strategy=contract['strategy'],policy_contract_sha256=identity(contract))
        if (bound!=entry['bound'] or row.get('stage')!=entry['stage'] or row.get('role')!=job['role']
            or row.get('load')!=contract['text_rate'] or row.get('contract')!=contract or summary.get('seed')!=job['seed']
            or summary.get('candidate')!=symbolic or summary.get('candidate_id')!=identity(symbolic)):
            raise ValueError('Report role/stage/load/seed/contract identity mismatch')
        u=raw['evaluation']['utility'];su=summary['utility']
        if raw['config']['weights']!={t:1/6 for t in TERMS}:raise ValueError('Original weights were changed')
        if any(u.get(t+'_norm')!=su.get(t+'_norm') for t in TERMS) or u.get('eligible')!=su.get('eligible') or u.get('U')!=su.get('U'):
            raise ValueError('Raw/summary normalized terms mismatch')
    if seen!=set(registered):raise ValueError('Missing report raw results')
    if interrupted:
        spill209=next(r for r in native if r['role']=='calibrated_spill' and r['summary']['seed']==209)
        entry=registered[spill209['summary']['raw']['path']];u=entry['raw']['evaluation']['utility']
        if u.get('eligible') is not False or u.get('E_norm') is not None or u.get('U') is not None:
            raise ValueError('Interrupted seed209 must preserve unknown energy and utility')
        if entry['raw'].get('energy_evidence',{}).get('coverage_valid') is not False:
            raise ValueError('Interrupted seed209 energy evidence mismatch')
    for stage in expected_stages:
        seeds=expected_stages[stage][2]
        for seed in seeds:
            rows={r['role']:r for r in packet_rows if r['stage']==stage and r['seed']==seed}
            assert_matched_pair(rows['calibrated_spill'],rows['risk_matched_no_spill'])
    if len({r['load'] for r in native if not r['stage'].startswith('development')})!=1:raise ValueError('Reserved workload changed')
    return native,plan,list({s['path']:s for s in sources}.values())


def csv_write(path,rows):
    if not rows:return
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def flat_point(p):
    return dict(L_scale=p['L_scale'],P_scale=p['P_scale'],**{'weight_'+t:p['weights'][t] for t in TERMS},
        **{'delta_seed'+s:v for s,v in p['per_seed_delta_U'].items()},
        **{name+'_'+k:v for name,z in p['phases'].items() for k,v in z.items()})
def fmt(x):return f'{x:.6g}' if finite(x) else 'UNKNOWN'


def plots(fine,pairs,geometry,output,scope='all5',suffix=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    n=round(math.sqrt(len(fine)));fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    for ax,field,title in zip(axes,['mean_delta_U','minimum_delta_U','all_positive'],[f'Mean paired delta U ({scope})','Worst seed delta U',f'Strict positivity ({scope})']):
        values=np.array([p['phases'][scope][field] if p['phases'][scope][field] is not None else float('nan') for p in fine],dtype=float).reshape(n,n)
        cmap=plt.get_cmap('RdYlGn' if field!='all_positive' else 'Greys').copy();cmap.set_bad('#b3b3b3')
        kwargs={} if field=='all_positive' else dict(vmin=-max(abs(float(np.nanmin(values))),abs(float(np.nanmax(values))),1e-12),vmax=max(abs(float(np.nanmin(values))),abs(float(np.nanmax(values))),1e-12)) if np.isfinite(values).any() else {}
        im=ax.imshow(values,origin='lower',extent=(0,1,0,1),aspect='equal',cmap=cmap,**kwargs)
        known_grid=np.isfinite(values).any()
        if known_grid:fig.colorbar(im,ax=ax,shrink=.8)
        else:ax.text(.5,.5,'UNKNOWN: incomplete registered pairs',ha='center',va='center',transform=ax.transAxes,fontsize=9)
        ax.set(xlabel='P weight scale',ylabel='L weight scale',title=title)
        for pair in pairs:
            c=pair['coefficients']
            if c is None or not known_grid:continue
            # Draw exact break-even equation with x=P and y=L.
            x=np.linspace(0,1,201)
            if c['B']!=0:
                y=-(c['A']+c['C']*x)/c['B'];mask=(y>=0)&(y<=1);ax.plot(x[mask],y[mask],linewidth=.8,label=str(pair['seed']))
            elif c['C']!=0:
                px=-c['A']/c['C']
                if 0<=px<=1:ax.axvline(px,linewidth=.8)
        ax.plot([1],[1],'bx',markersize=8)
    fig.suptitle('Posthoc preferences: '+scope+'; fixed measurements/policies; gray=UNKNOWN; blue x=original weights')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(output/('weight-regions'+suffix+'.png'),dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for axis,label in [('L_only','Reduce L only'),('P_only','Reduce P only'),('both','Reduce L and P equally')]:
        x=np.linspace(0,1,101);ys=[]
        for z in x:
            l,p=(float(z),1.) if axis=='L_only' else ((1.,float(z)) if axis=='P_only' else (float(z),float(z)))
            ys.append(point_summary(pairs,l,p)['phases'][scope]['mean_delta_U'])
        ax.plot(x,[y if y is not None else float('nan') for y in ys],label=label)
    ax.axhline(0,color='black',linewidth=.7);ax.set(xlabel='Remaining weight scale',ylabel='Mean delta U (spill - control)',title='Posthoc preference sensitivity: '+scope);ax.legend();fig.tight_layout();fig.savefig(output/('axis-sensitivity'+suffix+'.png'),dpi=170);plt.close(fig)


def analyze(execution,data,plan_path,output,fine_steps=101,allow_interrupted=False):
    native,plan,sources=load_completed(execution,data,plan_path,allow_interrupted)
    if fine_steps<101:raise ValueError('Fine grid must have at least101 points per axis')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    pairs=pair_data(native,allow_interrupted);coefs=[r['coefficients'] for r in pairs]
    evaluable=[r for r in pairs if r['coefficients'] is not None];available=[r['coefficients'] for r in evaluable]
    evaluable_geometry=region(available)
    evaluable_analysis=dict(seeds=[r['seed'] for r in evaluable],expected_registered_pairs=5,
        scope='Only complete evaluable pairs; cannot replace the full5-pair or full confirmation endpoint',
        strict_positive_region=evaluable_geometry,mean_coefficients=mean_coefficients(available),
        mean_positive_region=region([mean_coefficients(available)]),
        axes={axis:dict(all_evaluable_positive=axis_interval(available,axis),mean_positive=axis_interval([mean_coefficients(available)],axis)) for axis in ('L_only','P_only','both')})
    lvals=plan['axes']['L_scale'];pvals=plan['axes']['P_scale']
    if len(set(lvals))<6 or len(set(pvals))<6:raise ValueError('At least6x6 declared grid required')
    coarse=[point_summary(pairs,l,p) for l,p in itertools.product(lvals,pvals)]
    fine=[point_summary(pairs,i/(fine_steps-1),j/(fine_steps-1)) for i,j in itertools.product(range(fine_steps),repeat=2)]
    geometry=region(coefs);phase_geometry={name:region([r['coefficients'] for r in pairs if r['seed'] in seeds]) for name,seeds in PHASES.items()}
    phase_means={name:mean_coefficients([r['coefficients'] for r in pairs if r['seed'] in seeds]) for name,seeds in PHASES.items()}
    axes={axis:dict(all5=axis_interval(coefs,axis),per_seed={str(r['seed']):axis_interval([r['coefficients']],axis) for r in pairs},
        phases={name:axis_interval([r['coefficients'] for r in pairs if r['seed'] in seeds],axis) for name,seeds in PHASES.items()},
        phase_mean_positive_intervals={name:axis_interval([c],axis) for name,c in phase_means.items()}) for axis in ('L_only','P_only','both')}
    ranks=[r for l,p in itertools.product(lvals,pvals) for r in rank_development(native,l,p)]
    original=point_summary(pairs,1.,1.)
    result=dict(status='COMPLETE_INTERRUPTED_POSTHOC_SENSITIVITY' if len(native)<26 else 'COMPLETE_POSTHOC_SENSITIVITY',
        interrupted=len(native)<26,registered_native_runs=len(native),sources=sources,original_weights={t:1/6 for t in TERMS},
        weight_formula='wQ=wT=wM=wE=1/(4+l+p), wL=l/(4+l+p), wP=p/(4+l+p)',
        delta_formula='delta U=(A+B*l+C*p)/(4+l+p); A=-sum(delta norm Q,T,M,E), B=-delta L, C=-delta P',
        original=original,pairs=pairs,strict_positive_all5_region=geometry,phase_regions=phase_geometry,
        mean_delta_coefficients=phase_means,phase_mean_positive_regions={name:region([c]) for name,c in phase_means.items()},axis_critical_points=axes,
        evaluable_pairs_only=evaluable_analysis,coarse=coarse,fine_grid=dict(steps=fine_steps,points=fine),development_reranking=ranks,
        interpretation='事后偏好敏感性；测量、策略、路由、质量、稳定性均未改变，不代表物理性能提高或原协议实验成功。使用本轮独立数据调权后，它们对新权重已属探索数据。',
        unknown_policy='Any missing normalized term or ineligible original run keeps utility UNKNOWN even if its new weight is zero. Full5 positivity UNKNOWN if any pair unknown.',
        future_requirement='Any selected new objective and reoptimized policy need a separately frozen subsequent experiment.',GPU_used=False,original_outputs_modified=False)
    write(output/'analysis.json',result)
    csv_write(output/'coarse-grid.csv',[flat_point(p) for p in coarse]);csv_write(output/'fine-grid.csv',[flat_point(p) for p in fine])
    csv_write(output/'development-reranking.csv',[dict(load=r['load'],L_scale=r['L_scale'],P_scale=r['P_scale'],role=v['role'],deployment=json.dumps(v['deployment'],sort_keys=True),rank=v['rank'],mean_U=v['mean_U'],seed203_U=v['per_seed_U']['203'],seed204_U=v['per_seed_U']['204'],**{'weight_'+t:r['weights'][t] for t in TERMS}) for r in ranks for v in r['ranking']])
    csv_write(output/'measured-pair-tradeoffs.csv',[dict(seed=r['seed'],phase=r['phase'],term=t,spill_norm=r['spill_terms'][t],control_norm=r['control_terms'][t],delta_norm=r['coefficients']['delta_norm'][t] if r['coefficients'] else None,measurement_changed=False) for r in pairs for t in TERMS])
    plots(fine,pairs,geometry,output)
    if len(evaluable)<5:plots(fine,pairs,evaluable_geometry,output,scope='evaluable_only',suffix='-evaluable-only')
    lines=['# L/P 权重事后敏感性分析','',
        f'本分析在原流程终态后执行；已登记正式运行{len(native)}/26。'+('原流程中断、未重试；seed209缺控制且分流E未知，全部5对结论保持UNKNOWN。' if len(native)<26 else '26次正式实验均完成。')+'没有重新运行GPU、改变策略或替换不利结果。权重改变只改变偏好评分，实际延迟、精度、能耗、显存、路由及稳定性均未改善。','',
        '令 l、p 为原L/P权重保留比例。最终 wL=l/(4+l+p)，wP=p/(4+l+p)，其余四项均为1/(4+l+p)，六项权重显式归一化为1。原实验位于(l,p)=(1,1)。','',
        f'原权重全部5对平均ΔU={fmt(original["phases"]["all5"]["mean_delta_U"])}；已知配对{original["phases"]["all5"]["known_pairs"]}/5。',
        f'整个[0,1]²内全部5对严格正收益区域：{geometry["status"]}。边界ΔU=0不计正收益。UNKNOWN绝不补零，也不靠把其权重调零消除证据缺口。','',
        '![均值、最差种子与全部正收益区域](weight-regions.png)','', '![单降和双降](axis-sensitivity.png)','',
        '|调整路径|全部5对正收益的比例区间|','|---|---|']
    for axis in axes:lines.append('|'+axis+'|'+json.dumps(axes[axis]['all5'],ensure_ascii=False)+'|')
    if len(evaluable)<5:
        examples=[('原权重',1.,1.),('仅L减半',.5,1.),('仅P减半',1.,.5),('L/P均减半',.5,.5),
            ('仅L归零',0.,1.),('仅P归零',1.,0.),('L/P均归零',0.,0.)]
        mean_coef=evaluable_analysis['mean_coefficients'];subset_mean_region=evaluable_analysis['mean_positive_region']
        lines+=['',f'仅完整可评价的{len(evaluable)}对种子{evaluable_analysis["seeds"]}另行描述：原权重均值ΔU={fmt(original["phases"]["evaluable_only"]["mean_delta_U"])}，严格正收益区域={evaluable_geometry["status"]}。此子集不能替代全部5对或确认阶段3对；缺失项未补零。','',
            '|偏好设置|L保留比例|P保留比例|仅完整配对均值ΔU|正收益配对数|','|---|---|---|---|---|']
        for label,l,p in examples:
            subset=point_summary(pairs,l,p)['phases']['evaluable_only']
            lines.append('|'+ '|'.join([label,fmt(l),fmt(p),fmt(subset['mean_delta_U']),str(subset['positive_pairs'])+'/'+str(subset['known_pairs'])])+'|')
        if mean_coef is not None:
            lines+=['',f'该完整配对子集的均值解析式：ΔU均值=({fmt(mean_coef["A"])} + ({fmt(mean_coef["B"])})×l + ({fmt(mean_coef["C"])})×p)/(4+l+p)。',
                '子集均值正收益区域：'+subset_mean_region['status']+'；子集每一对均正收益区域：'+evaluable_geometry['status']+'。']
            lines+=['对该子集，'+('至少存在一组非负L/P保留比例使均值ΔU严格为正。' if subset_mean_region['status']=='NONEMPTY' else '整个[0,1]²内均值ΔU没有严格为正的可行比例。' if subset_mean_region['status'] in {'EMPTY','EMPTY_STRICT_INTERIOR'} else '均值正收益区域因未知项无法判定。')+
                ('但这不能恢复缺失的第5对，全部5对均值与正收益区域继续为UNKNOWN。')]
        lines+=['',
            '![仅可评价配对子集，非全部5对](weight-regions-evaluable-only.png)','', '![仅可评价配对单轴敏感性](axis-sensitivity-evaluable-only.png)','',
            '|调整路径|仅完整配对全部正收益|仅完整配对均值正收益|','|---|---|---|']
        for axis in evaluable_analysis['axes']:
            v=evaluable_analysis['axes'][axis];lines.append('|'+axis+'|'+json.dumps(v['all_evaluable_positive'],ensure_ascii=False)+'|'+json.dumps(v['mean_positive'],ensure_ascii=False)+'|')
    lines+=['','解析式为ΔU=(A+B l+C p)/(4+l+p)。由于分母始终为正，全部5对正收益等价于5个严格半平面与单位正方形的交集。JSON保存所有逐种子系数、临界点、交集闭包顶点和严格可行见证；闭包不意味着零收益边界可接受。','',
        '|L保留|P保留|验证205–206均值ΔU|确认207–209均值ΔU|全部5均值ΔU|全部5正收益|','|---|---|---|---|---|---|']
    for p in coarse:lines.append('|'+ '|'.join([fmt(p['L_scale']),fmt(p['P_scale'])]+[fmt(p['phases'][name]['mean_delta_U']) for name in ('validation','confirmation','all5')]+[str(p['phases']['all5']['all_positive'])])+'|')
    lines+=['','两负载均对四个已测条件按新权重重排，逐种子与均值保存在 development-reranking.csv。某分流配对获益不说明它超过另一个部署；重排范围只有四个已测条件，未重新搜索策略，也不是全七模型池最优。','',
        '此处各归一化指标是同一组实测值，权重转换没有改变Pareto支配关系；原始逐项差在 measured-pair-tradeoffs.csv。','',
        '这是事后偏好分析，不能写成原协议成功或独立验证了调后的目标。若选用新权重并重新优化，必须另行冻结新目标、策略和后续实验；本轮保留种子对新目标已成为探索材料。','',
        '数据：analysis.json（全部实际输入SHA与解析结果）、coarse-grid.csv（6×6及逐种子结果）、fine-grid.csv（细网格）、development-reranking.csv、measured-pair-tradeoffs.csv。']
    (output/'权重敏感性报告.md').write_text('\n'.join(lines)+'\n')
    write(output/'manifest.json',dict(outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()},sources=sources))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execution',required=True,type=Path);parser.add_argument('--data',required=True,type=Path,help='Completed original final audit-and-data.json');parser.add_argument('--plan',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--fine-steps',type=int,default=101)
    parser.add_argument('--allow-interrupted',action='store_true',help='Explicitly allow an audited STOPPED terminal, preserve missing pairs as UNKNOWN')
    args=parser.parse_args();result=analyze(args.execution,args.data,args.plan,args.output,args.fine_steps,args.allow_interrupted)
    print(json.dumps(dict(status=result['status'],region=result['strict_positive_all5_region']['status']),ensure_ascii=False))
