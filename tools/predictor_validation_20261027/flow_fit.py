"""One-time training-only continuous-flow fits and leave-one-flow-out models.

The unchanged numerical engine gives each target request/phase one regression
row. Long flows therefore change observation weights and may change genuine
solo baselines. Diagnostics expose both effects and the relative-work loss's
size bias; they are not independent validation or physical identification.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
from pathlib import Path
import statistics
import time

from .background_profiles import background_semantic_traces, checked_receipt, receipt
from .background_statistical import BackgroundStatisticalProvider, family_task, fit_traces
from .flow_forensics import reconstruct_work
from .flow_profiles import map_flow_collection
from .profiling_analysis import digest
from .statistical_profiles import DEFINITION as NUMERICS, full_count, phase_count

PROTOCOL_SCHEMA='gsdeploy.continuous_flow_fit_protocol.v1'
EVIDENCE_SCHEMA='gsdeploy.verified_continuous_training_evidence.v1'
TREE_SCHEMA='gsdeploy.training_evidence_tree.v1'
TOOLS=('flow_fit.py','flow_profiles.py','background_profiles.py','background_statistical.py',
       'statistical_profiles.py','semantic_profiles.py','profiling_analysis.py','flow_forensics.py',
       'flow_validation.py','flow_metrics.py','probe_validation.py','profiling.py','validation.py','mechanism_audit.py')
POLICY={
 'regression_unit':'One equal-weight row per target handler and per target phase; not one row per collection',
 'objective':'mean((duration/baseline * speed - 1)^2) plus unchanged bounded ridge numerical engine',
 'solo_baseline':'Arithmetic mean of all same-target handlers completely nonoverlapping with any other handler, including true continuous-flow solos; >=2 observations',
 'cohort':'Each continuous flow remains one intact trace; never split it into artificial solo requests',
 'crossfit':'Leave exactly one entire continuous flow out, retaining both old training collections and the other continuous flows; refit baseline and coefficients independently',
 'seed_crossfit':'Additionally leave one whole seed and all four route flows out; retain old profiles and the other seed. Use seed folds for model-development comparisons, flow folds for diagnostics only',
 'ensemble':'Disabled; no crossfit model ensembling is enabled',
 'loss_bias':'Unregularized single-context relative-work least squares implies duration E[d^2]/E[d], not E[d]; ridge and multi-context coupling further change this quantity',
 'validation':'Training and retrospective training cross-validation only; all prior independent validation raws excluded',
 'numerical_engine':NUMERICS}


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def _checked(ref):return checked_receipt(ref)


def _unique_refs(refs):
    found={}
    for ref in refs:
        _checked(ref)
        if ref['path'] in found and found[ref['path']]!=ref:raise ValueError('Conflicting source receipts')
        found[ref['path']]=ref
    return list(found.values())


def _tools():return [receipt(Path(__file__).with_name(name)) for name in TOOLS]


def _training_trace(trace):
    if (trace.get('split')!='training' or trace.get('not_for_training') is True or not trace.get('requests')
            or any(row.get('split','training')!='training' for row in trace['requests'])):
        raise ValueError('Every fitted observation must be strictly training')


def prepare_protocol(old_raw_paths,suite_path,output,*,expected_flow_count=8):
    """Freeze source and weighting rules without reading future flow outcomes."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    suite=read(suite_path)
    if (suite.get('schema')!='gsdeploy.continuous_training_suite.v1' or suite.get('no_validation_data') is not True
            or len(suite.get('jobs',[]))!=expected_flow_count or expected_flow_count<2):
        raise ValueError('Complete predeclared continuous-training suite required')
    _checked(suite['source']);refs=[receipt(suite_path),suite['source'],*_tools()]
    old=[]
    for path in old_raw_paths:
        raw=read(path)
        if (raw.get('schema')!='gsdeploy.native_phase_collection.v1' or raw.get('status')!='completed'
                or not raw.get('request_records') or any(r.get('split')!='training' for r in raw['request_records'])):
            raise ValueError('Old collections must be complete and explicitly training-only')
        old.append(receipt(path));refs.append(receipt(path))
    if not old or len({r['path'] for r in old})!=len(old):raise ValueError('Distinct nonempty old training sources required')
    jobs=[];ids=set();outputs=set()
    for job in suite['jobs']:
        plan=read(_checked(job['plan']))
        if (plan.get('phase')!='predictor_training' or plan.get('split')!='training' or type(plan.get('created_ns')) is not int
                or plan.get('plan_sha256')!=digest({k:v for k,v in plan.items() if k!='plan_sha256'})):
            raise ValueError('Prospective training-only flow plans required')
        candidate=next((c for c in plan['candidates'] if c['id']==job['candidate_id']),None)
        if candidate is None:raise ValueError('Suite candidate absent from frozen plan')
        fid='seed-'+str(plan.get('seed'))+'__'+job['candidate_id']
        raw_path=str((Path(job['output'])/'raw.json').resolve())
        if fid in ids or raw_path in outputs:raise ValueError('Unique whole-flow units required')
        ids.add(fid);outputs.add(raw_path);refs.append(job['plan'])
        jobs.append({'flow_id':fid,'seed':plan.get('seed'),'plan':job['plan'],'candidate_id':job['candidate_id'],'raw_path':raw_path})
    value={'schema':PROTOCOL_SCHEMA,'created_ns':time.perf_counter_ns(),'suite':receipt(suite_path),
        'old_training_raws':old,'flow_jobs':jobs,'expected_flow_count':expected_flow_count,
        'policy':POLICY,'policy_sha256':digest(POLICY),'source_receipts':_unique_refs(refs),
        'GPU_started':False,'new_flow_outcomes_read_by_prepare':False,'validation_sources_used':[]}
    value['protocol_sha256']=digest(value);save(output,value);return value


def load_training(protocol_path):
    """Audit every original source and preserve complete trace boundaries."""
    protocol=read(protocol_path)
    if (protocol.get('schema')!=PROTOCOL_SCHEMA or protocol.get('policy')!=POLICY
            or protocol.get('protocol_sha256')!=digest({k:v for k,v in protocol.items() if k!='protocol_sha256'})):
        raise ValueError('Frozen fit protocol/source policy mismatch')
    _unique_refs(protocol['source_receipts'])
    # Check every required run exists before mapping or creating output files.
    if any(not Path(j['raw_path']).is_file() for j in protocol['flow_jobs']):
        raise ValueError('All declared continuous training flows must finish before fitting')
    groups=[];ids=set();identities=set()
    for index,ref in enumerate(protocol['old_training_raws']):
        path=_checked(ref);raw=read(path)
        if any(r.get('split')!='training' for r in raw['request_records']):raise ValueError('Old training split changed')
        traces,mapping=background_semantic_traces(path)
        refs=_unique_refs([ref,*[r for t in traces for r in t.get('sources',[])]])
        groups.append({'group_id':'old:'+str(index),'role':'old_profile_training','raw':ref,'traces':traces,
            'mapping_report':mapping,'transitive_sources':refs})
    for job in protocol['flow_jobs']:
        mapped=map_flow_collection(job['raw_path']);raw=read(job['raw_path'])
        if raw['plan']!=job['plan'] or raw['candidate_id']!=job['candidate_id']:
            raise ValueError('Actual continuous flow differs from prospective suite unit')
        groups.append({'group_id':job['flow_id'],'role':'continuous_training','raw':receipt(job['raw_path']),
            'plan':job['plan'],'seed':job['seed'],'candidate_id':job['candidate_id'],'traces':mapped['traces'],
            'mapping_report':mapped['mapping_report'],'resource_summary':{k:v for k,v in mapped['resource_evidence'].items()
                if k not in ('sources','board_samples','stage_history')},
            'transitive_sources':_unique_refs(mapped['mapping_report']['sources'])})
    for group in groups:
        for trace in group['traces']:
            _training_trace(trace)
            if trace['id'] in ids:raise ValueError('Duplicate trace identities across training sources')
            ids.add(trace['id']);identities.add(trace['execution_context_sha256'])
    if len(identities)!=1:raise ValueError('All old and continuous training execution contexts must match')
    return {'protocol':protocol,'protocol_receipt':receipt(protocol_path),'groups':groups,
        'execution_context_sha256':next(iter(identities)),'source_receipts':_unique_refs([
            *protocol['source_receipts'],*[r for g in groups for r in g['transitive_sources']]])}


def _materialize_evidence(bundle,folder):
    folder.mkdir(parents=True,exist_ok=False)
    for group in bundle['groups']:
        if group['role']=='old_profile_training':group['direct_source']=group['raw'];continue
        trace=group['traces'][0]
        wrapper={'schema':EVIDENCE_SCHEMA,'kind':'derived_training_evidence','split':'training',
            'flow_id':group['group_id'],'execution_context_sha256':bundle['execution_context_sha256'],
            'raw_source':group['raw'],'plan_source':group['plan'],'source_receipts':group['transitive_sources'],
            'mapping_report':group['mapping_report'],'trace':trace,'trace_sha256':digest(trace),
            'resource_summary':group['resource_summary'],
            'split_basis':'Verified prospective plan.phase=predictor_training and created_ns<startup_begin_ns; original raw is unchanged',
            'GPU_started':False}
        path=folder/(group['group_id']+'.json');save(path,wrapper);group['direct_source']=receipt(path)


def _fit_groups(bundle,groups,*,omitted_flow_id=None,omitted_flow_ids=(),omitted_seed=None):
    traces=[t for g in groups for t in g['traces']]
    for trace in traces:_training_trace(trace)
    sources=_unique_refs([bundle['protocol_receipt'],*_tools(),*[g['direct_source'] for g in groups]])
    document=fit_traces(traces,sources=sources)
    omitted=sorted(set(omitted_flow_ids)|({omitted_flow_id} if omitted_flow_id else set()))
    tree={'schema':TREE_SCHEMA,'protocol':bundle['protocol_receipt'],'omitted_flow_id':omitted_flow_id,
        'omitted_flow_ids':omitted,'omitted_seed':omitted_seed,
        'groups':[{k:copy.deepcopy(v) for k,v in g.items() if k in
                   ('group_id','role','raw','plan','seed','candidate_id','direct_source','transitive_sources')} for g in groups],
        'transitive_source_receipts':_unique_refs([r for g in groups for r in g['transitive_sources']]),
        'continuous_training_flow_count':sum(g['role']=='continuous_training' for g in groups),
        'validation_sources_used':[]}
    document['training_evidence_tree']=tree
    document['training_policy']=POLICY
    document['mapping_reports']=[{'group_id':g['group_id'],'report':g['mapping_report']} for g in groups]
    document.pop('model_sha256');document['model_sha256']=digest(document)
    return document


def fit_leave_one_flow_out(bundle,heldout_flow_id):
    """Refit all baselines/coefficients with exactly one whole flow excluded."""
    matches=[g for g in bundle['groups'] if g['group_id']==heldout_flow_id and g['role']=='continuous_training']
    if len(matches)!=1:raise ValueError('Exactly one declared continuous flow must be held out')
    return _fit_groups(bundle,[g for g in bundle['groups'] if g['group_id']!=heldout_flow_id],omitted_flow_id=heldout_flow_id)


def fit_leave_one_seed_out(bundle,heldout_seed):
    omitted=[g['group_id'] for g in bundle['groups'] if g['role']=='continuous_training' and g['seed']==heldout_seed]
    if not omitted or not any(g['role']=='continuous_training' and g['seed']!=heldout_seed for g in bundle['groups']):
        raise ValueError('Held-out seed and another training seed required')
    return _fit_groups(bundle,[g for g in bundle['groups'] if g['group_id'] not in omitted],
        omitted_flow_ids=omitted,omitted_seed=heldout_seed)


def verify_training_model(path):
    """Independently re-walk direct and transitive original-raw evidence roles."""
    document=read(path);context=document['execution_context_sha256']
    BackgroundStatisticalProvider(document,context)
    tree=document.get('training_evidence_tree',{})
    if tree.get('schema')!=TREE_SCHEMA or tree.get('validation_sources_used')!=[]:
        raise ValueError('Explicit validation-free training evidence tree required')
    protocol=read(_checked(tree['protocol']))
    if (protocol.get('schema')!=PROTOCOL_SCHEMA or protocol.get('policy')!=document.get('training_policy')
            or protocol.get('protocol_sha256')!=digest({k:v for k,v in protocol.items() if k!='protocol_sha256'})):
        raise ValueError('Model training protocol changed')
    _unique_refs(protocol['source_receipts']);_unique_refs(document['source_receipts'])
    _unique_refs(tree['transitive_source_receipts'])
    direct={r['path']:r for r in document['source_receipts']}
    declared={j['flow_id']:j for j in protocol['flow_jobs']};groups=tree['groups'];seen=set();flows=set();raws=[];trace_ids=[];expected_transitive=[]
    expected_direct=_unique_refs([tree['protocol'],*_tools(),*[g['direct_source'] for g in groups]])
    if direct!={r['path']:r for r in expected_direct} or document['base_model']['source_receipts']!=document['source_receipts']:
        raise ValueError('Model has hidden, missing or inconsistent direct fit sources')
    for group in groups:
        gid=group['group_id']
        if gid in seen:raise ValueError('Duplicate evidence group')
        seen.add(gid);_unique_refs(group['transitive_sources']);_checked(group['raw']);raws.append(group['raw'])
        ref=group['direct_source']
        if direct.get(ref['path'])!=ref:raise ValueError('Evidence group missing from model direct sources')
        if group['role']=='old_profile_training':
            if ref!=group['raw'] or group['raw'] not in protocol['old_training_raws']:
                raise ValueError('Old profile source is not a declared original raw')
            raw=read(group['raw']['path'])
            if not raw.get('request_records') or any(r.get('split')!='training' for r in raw['request_records']):
                raise ValueError('Old original raw contains nontraining observations')
            traces,_=background_semantic_traces(group['raw']['path'])
            expected_refs=_unique_refs([group['raw'],*[r for t in traces for r in t.get('sources',[])]])
        elif group['role']=='continuous_training':
            if gid not in declared or group['raw']['path']!=declared[gid]['raw_path']:
                raise ValueError('Continuous original raw not in prospective suite')
            wrapper=read(_checked(ref));_unique_refs(wrapper['source_receipts'])
            if (wrapper.get('schema')!=EVIDENCE_SCHEMA or wrapper.get('kind')!='derived_training_evidence'
                    or wrapper.get('split')!='training' or wrapper.get('raw_source')!=group['raw']
                    or wrapper.get('plan_source')!=declared[gid]['plan'] or wrapper.get('flow_id')!=gid):
                raise ValueError('Derived training wrapper/source role mismatch')
            mapped=map_flow_collection(group['raw']['path']);traces=mapped['traces']
            if (read(group['raw']['path'])['candidate_id']!=declared[gid]['candidate_id']
                    or wrapper.get('trace_sha256')!=digest(wrapper['trace'])
                    or wrapper['trace_sha256']!=digest(traces[0])
                    or wrapper.get('mapping_report')!=mapped['mapping_report']):
                raise ValueError('Derived training trace does not match strictly remapped original raw')
            expected_refs=_unique_refs(mapped['mapping_report']['sources'])
            if wrapper['source_receipts']!=expected_refs:raise ValueError('Wrapper hides transitive original source receipts')
            flows.add(gid)
        else:raise ValueError('Unknown training evidence group role')
        if group['transitive_sources']!=expected_refs:raise ValueError('Evidence group omits original transitive receipts')
        expected_transitive.extend(expected_refs)
        for trace in traces:
            _training_trace(trace)
            if trace['execution_context_sha256']!=context:raise ValueError('Evidence/model execution context mismatch')
            trace_ids.append(trace['id'])
    if tree['transitive_source_receipts']!=_unique_refs(expected_transitive):raise ValueError('Incomplete global transitive training evidence')
    omitted=tree.get('omitted_flow_id');omitted_ids=tree.get('omitted_flow_ids',[]);omitted_seed=tree.get('omitted_seed');expected=set(declared)
    if len(omitted_ids)!=len(set(omitted_ids)) or not set(omitted_ids)<=expected:raise ValueError('Unknown or duplicate omitted flows')
    if omitted is not None and omitted_ids!=[omitted]:raise ValueError('Single-flow exclusion metadata mismatch')
    if omitted_seed is not None and set(omitted_ids)!={fid for fid,j in declared.items() if j['seed']==omitted_seed}:
        raise ValueError('Whole-seed exclusion metadata mismatch')
    if omitted is None and omitted_seed is None and omitted_ids:raise ValueError('Unspecified exclusion unit')
    expected.difference_update(omitted_ids)
    if flows!=expected or tree.get('continuous_training_flow_count')!=len(flows):
        raise ValueError('Wrong complete/crossfit flow evidence coverage')
    if {g['raw']['path'] for g in groups if g['role']=='old_profile_training'}!={r['path'] for r in protocol['old_training_raws']}:
        raise ValueError('Old training collection omitted or substituted')
    return {'verified':True,'model':receipt(path),'execution_context_sha256':context,
        'training_flow_count':len(flows),'training_flow_ids':sorted(flows),'training_trace_ids':trace_ids,
        'source_group_count':len(groups),'omitted_flow_id':omitted,'omitted_flow_ids':omitted_ids,'omitted_seed':omitted_seed,
        'training_raw_receipts':raws,'direct_and_transitive_sources_verified':True,'validation_sources_used':[],
        'GPU_started':False}


def _moments(values):
    values=list(values)
    if not values:return {'count':0,'mean':None,'min':None,'max':None,'p05':None,'p50':None,'p95':None}
    values.sort()
    def quantile(q):
        pos=(len(values)-1)*q;lo=math.floor(pos);hi=math.ceil(pos)
        return values[lo]+(values[hi]-values[lo])*(pos-lo)
    return {'count':len(values),'mean':statistics.mean(values),'min':min(values),'max':max(values),
        'p05':quantile(.05),'p50':quantile(.5),'p95':quantile(.95)}


def weight_audit(groups):
    counts=collections.defaultdict(lambda:collections.Counter());solos=collections.defaultdict(lambda:collections.defaultdict(list));orders=collections.defaultdict(set)
    for group in groups:
        role='continuous' if group['role']=='continuous_training' else 'old'
        for trace in group['traces']:
            rows=trace['requests']
            for row in rows:
                target=(row['family'],row['task'],row['pool_id']);counts[target][role]+=1
                orders[target].add(tuple(s['phase'] for s in row['segments']))
                if not any(o is not row and o['handler_start_ns']<row['handler_end_ns'] and row['handler_start_ns']<o['handler_end_ns'] for o in rows):
                    solos[target][role].append((row['handler_end_ns']-row['handler_start_ns'])/1e6)
    report=[]
    for target,c in sorted(counts.items()):
        old=solos[target]['old'];new=solos[target]['continuous'];all_solo=old+new
        before=statistics.mean(old) if old else None;after=statistics.mean(all_solo) if all_solo else None
        report.append({'target':list(target),'old_handler_rows':c['old'],'continuous_handler_rows':c['continuous'],
            'continuous_share_within_target':c['continuous']/sum(c.values()),'old_solo_count':len(old),'continuous_solo_count':len(new),
            'old_solo_handler_mean_ms':before,'combined_solo_handler_mean_ms':after,
            'relative_solo_baseline_shift':after/before-1 if before else None,'phase_orders':[list(v) for v in sorted(orders[target])],
            'phase_order_conflict':len(orders[target])!=1})
    return {'policy':POLICY,'targets':report,'note':'Equal observation weights operate separately within each target fit; continuous-run size is not normalized per collection'}


def single_context_bias(traces):
    """Descriptive analytic toy moments, not the multi-context fitted estimator."""
    durations=collections.defaultdict(list)
    for trace in traces:
        segments=[(r,s) for r in trace['requests'] for s in r['segments']]
        cuts=sorted({s[k] for _,s in segments for k in ('start_ns','end_ns')});intervals=[]
        for left,right in zip(cuts,cuts[1:]):
            active=[(r,s) for r,s in segments if s['start_ns']<=left and right<=s['end_ns']]
            if not active:continue
            encoded=[{'family':family_task(r['family'],r['task'])} for r,_ in active]
            fk,_=full_count(encoded)
            pk,_=phase_count([({'family':family_task(r['family'],r['task'])},s) for r,s in active])
            intervals.append((left,right,fk,pk))
        for row in trace['requests']:
            target=(row['family'],row['task'],row['pool_id'])
            contexts={fk for a,b,fk,pk in intervals if row['handler_start_ns']<=a and b<=row['handler_end_ns']}
            if len(contexts)==1:durations[('handler',*target,'whole',next(iter(contexts)))].append((row['handler_end_ns']-row['handler_start_ns'])/1e6)
            for segment in row['segments']:
                contexts={pk for a,b,fk,pk in intervals if segment['start_ns']<=a and b<=segment['end_ns']}
                if len(contexts)==1:durations[('phase',*target,segment['phase'],next(iter(contexts)))].append((segment['end_ns']-segment['start_ns'])/1e6)
    rows=[]
    for target,values in sorted(durations.items()):
        if len(values)<2:continue
        mean=statistics.mean(values);second=statistics.mean(v*v for v in values);implied=second/mean
        rows.append({'level':target[0],'target':list(target[1:4]),'phase':target[4],'context_id':target[5],
            'samples':len(values),'arithmetic_mean_duration_ms':mean,'second_moment_duration_ms2':second,
            'unregularized_single_context_implied_duration_ms':implied,'relative_size_bias':implied/mean-1})
    return {'groups':rows,'scope':'Subsets whose complete handler/phase has only one observed context; analytic unregularized one-context toy, not actual multi-context ridge predictions',
        'derivation':'x=d/b; argmin_s E[(x*s-1)^2]=b*E[d]/E[d^2]; implied b/s=E[d^2]/E[d]. Numerical bounds/ridge and multiple contexts alter this solution.',
        'mean_is_unbiased_conditional_prediction':False}


def reconstruction_summary(vectors):
    values=collections.defaultdict(list);unknown=collections.Counter()
    for row in vectors['requests']:
        for phase,work in row['work_by_phase'].items():
            key=(row['family'],row['task'],row['pool_id'],phase)
            if work is None:unknown[key]+=1
            else:values[key].append(work)
    keys=sorted(set(values)|set(unknown))
    return {'phase_work':[{'target':list(k),'W':_moments(values[k]),'unknown_phases':unknown[k],
        'mean_absolute_work_residual':statistics.mean(abs(v-1) for v in values[k]) if values[k] else None} for k in keys],
        'request_vectors':len(vectors['requests']),'known_joint_vectors':sum(r['complete_known_vector'] for r in vectors['requests']),
        'scope':'Reconstruction of fitted/held-out training mixed semantic wall work; no independent validation or physical progress observation'}


def _diagnostics(document,groups):
    provider=BackgroundStatisticalProvider(document,document['execution_context_sha256'])
    traces=[t for g in groups for t in g['traces']];vectors=reconstruct_work(traces,provider)
    base=document['base_model']
    return {'schema':'gsdeploy.continuous_training_fit_diagnostics.v1','model_sha256':document['model_sha256'],
        'training_only':True,'independently_validated':False,'physically_identified':False,'weight_and_baseline_audit':weight_audit(groups),
        'single_context_loss_bias':single_context_bias(traces),'phase_work_distribution':reconstruction_summary(vectors),
        'phase_work_distribution_by_source_group':[{'group_id':g['group_id'],**reconstruction_summary({
            'requests':[r for r in vectors['requests'] if r['trace_id'] in {t['id'] for t in g['traces']}]})} for g in groups],
        'handler_fit_diagnostics':[{'target':m['target'],'diagnostics':m['fit_diagnostics']} for m in base['handler_models']],
        'semantic_fit_diagnostics':[{'target':m['target'],'diagnostics':m['fit_diagnostics']} for m in base['semantic_models']],
        'coverage':document['coverage'],'GPU_started':False}


def _crossfits(bundle,folder):
    folder.mkdir(parents=True,exist_ok=False);folds=[]
    continuous=[g for g in bundle['groups'] if g['role']=='continuous_training']
    for heldout in continuous:
        fid=heldout['group_id'];dest=folder/fid;dest.mkdir()
        model=fit_leave_one_flow_out(bundle,fid);model_path=dest/'model.json';save(model_path,model)
        verified=verify_training_model(model_path)
        provider=BackgroundStatisticalProvider(model,bundle['execution_context_sha256'])
        vectors=reconstruct_work(heldout['traces'],provider)
        vectors.update(retrospective_training_cross_validation=True,independent_validation=False,
            crossfit_model=receipt(model_path),heldout_source=heldout['raw'],heldout_excluded_from_fit=True)
        work_path=dest/'heldout-work.json';save(work_path,vectors)
        diag={'schema':'gsdeploy.leave_one_continuous_flow_out_diagnostics.v1','fold_id':fid,
            'verification':verified,'heldout_work':reconstruction_summary(vectors),'training_coverage':model['coverage'],
            'retrospective_training_cross_validation':True,'independent_validation':False,'GPU_started':False}
        diag_path=dest/'diagnostics.json';save(diag_path,diag)
        folds.append({'fold_id':fid,'model':receipt(model_path),'heldout_raw':heldout['raw'],'heldout_plan':heldout['plan'],
            'unit':'flow',
            'candidate_id':heldout['candidate_id'],'heldout_trace_id':heldout['traces'][0]['id'],
            'training_flow_ids':[g['group_id'] for g in continuous if g is not heldout],
            'work_vectors':receipt(work_path),'diagnostics':receipt(diag_path)})
        print(json.dumps({'event':'crossfit_complete','fold_id':fid,'known_joint_vectors':diag['heldout_work']['known_joint_vectors']}),flush=True)
    return {'schema':'gsdeploy.continuous_flow_crossfit_manifest.v1','protocol':bundle['protocol_receipt'],
        'folds':folds,'source_receipts':bundle['source_receipts'],'retrospective_training_cross_validation':True,
        'no_independent_validation':True,'new_GPU_experiments_started':False,'GPU_started':False}


def _seed_crossfits(bundle,folder):
    folder.mkdir(parents=True,exist_ok=False);folds=[]
    continuous=[g for g in bundle['groups'] if g['role']=='continuous_training']
    for seed in sorted({g['seed'] for g in continuous}):
        fid='seed-'+str(seed);dest=folder/fid;dest.mkdir()
        model=fit_leave_one_seed_out(bundle,seed);model_path=dest/'model.json';save(model_path,model)
        verified=verify_training_model(model_path);provider=BackgroundStatisticalProvider(model,bundle['execution_context_sha256'])
        heldout=[g for g in continuous if g['seed']==seed];heldout_rows=[];all_vectors=[]
        for group in heldout:
            vectors=reconstruct_work(group['traces'],provider);all_vectors.extend(vectors['requests'])
            vectors.update(retrospective_training_cross_validation=True,independent_validation=False,
                crossfit_model=receipt(model_path),heldout_source=group['raw'],heldout_excluded_from_fit=True,whole_seed_excluded=True)
            work_path=dest/(group['group_id']+'-heldout-work.json');save(work_path,vectors)
            heldout_rows.append({'flow_id':group['group_id'],'heldout_raw':group['raw'],'heldout_plan':group['plan'],
                'candidate_id':group['candidate_id'],'heldout_trace_id':group['traces'][0]['id'],'work_vectors':receipt(work_path)})
        diag={'schema':'gsdeploy.leave_one_continuous_seed_out_diagnostics.v1','fold_id':fid,'verification':verified,
            'heldout_work':reconstruction_summary({'requests':all_vectors}),'training_coverage':model['coverage'],
            'retrospective_training_cross_validation':True,'independent_validation':False,'GPU_started':False}
        diag_path=dest/'diagnostics.json';save(diag_path,diag)
        folds.append({'fold_id':fid,'unit':'seed','heldout_seed':seed,'model':receipt(model_path),
            'heldout_flows':heldout_rows,'training_flow_ids':[g['group_id'] for g in continuous if g['seed']!=seed],
            'diagnostics':receipt(diag_path)})
        print(json.dumps({'event':'seed_crossfit_complete','fold_id':fid,'known_joint_vectors':diag['heldout_work']['known_joint_vectors']}),flush=True)
    return {'schema':'gsdeploy.continuous_seed_crossfit_manifest.v1','protocol':bundle['protocol_receipt'],
        'folds':folds,'source_receipts':bundle['source_receipts'],'retrospective_training_cross_validation':True,
        'no_independent_validation':True,'model_development_selection_unit':'whole seed',
        'ensemble_enabled':False,'GPU_started':False}


def fit_protocol(protocol_path,output,*,with_crossfit=False,crossfit_only=False):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    bundle=load_training(protocol_path)
    output.mkdir(parents=True,exist_ok=False)
    save(output/'started.json',{'protocol':bundle['protocol_receipt'],'created_ns':time.perf_counter_ns(),
        'crossfit_only':crossfit_only,'with_crossfit':with_crossfit,'GPU_started':False})
    _materialize_evidence(bundle,output/'training-evidence')
    outputs=[]
    if not crossfit_only:
        model=_fit_groups(bundle,bundle['groups']);model_path=output/'training-v11-model.json';save(model_path,model)
        verification=verify_training_model(model_path);save(output/'model-source-verification.json',verification)
        diagnostics=_diagnostics(model,bundle['groups']);diag_path=output/'training-v11-diagnostics.json';save(diag_path,diagnostics)
        outputs.extend([receipt(model_path),receipt(diag_path),receipt(output/'model-source-verification.json')])
    if with_crossfit or crossfit_only:
        crossfit=_crossfits(bundle,output/'crossfit');save(output/'crossfit-manifest.json',crossfit);outputs.append(receipt(output/'crossfit-manifest.json'))
        seed_crossfit=_seed_crossfits(bundle,output/'seed-crossfit');save(output/'seed-crossfit-manifest.json',seed_crossfit)
        outputs.append(receipt(output/'seed-crossfit-manifest.json'))
    _unique_refs(bundle['source_receipts'])
    manifest={'schema':'gsdeploy.continuous_flow_training_fit_manifest.v1','protocol':bundle['protocol_receipt'],
        'outputs':outputs,'transitive_source_receipts':bundle['source_receipts'],
        'old_training_raws':bundle['protocol']['old_training_raws'],
        'continuous_training_raws':[g['raw'] for g in bundle['groups'] if g['role']=='continuous_training'],
        'training_only':True,'validation_sources_used':[],'GPU_started':False,
        'once_only_output_policy':'Exclusive new output directory; started marker retained on any failure; no overwrite/retry in place'}
    save(output/'manifest.json',manifest);return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    a=sub.add_parser('prepare');a.add_argument('--suite',required=True,type=Path);a.add_argument('--old-raw',action='append',required=True,type=Path);a.add_argument('--output',required=True,type=Path)
    for name in ('fit','crossfit'):
        a=sub.add_parser(name);a.add_argument('--protocol',required=True,type=Path);a.add_argument('--output',required=True,type=Path)
        if name=='fit':a.add_argument('--with-crossfit',action='store_true')
    a=sub.add_parser('verify');a.add_argument('--model',required=True,type=Path)
    a=p.parse_args()
    if a.op=='prepare':result=prepare_protocol(a.old_raw,a.suite,a.output)
    elif a.op=='verify':result=verify_training_model(a.model)
    else:result=fit_protocol(a.protocol,a.output,with_crossfit=getattr(a,'with_crossfit',False),crossfit_only=a.op=='crossfit')
    print(json.dumps({'status':'VERIFIED' if a.op=='verify' else 'COMPLETE','schema':result.get('schema'),'GPU_started':False}))


if __name__=='__main__':main()
