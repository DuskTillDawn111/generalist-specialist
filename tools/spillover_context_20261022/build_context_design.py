"""Explicit terminal development inventory and 12-column design audit; no fitting."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
S1=ROOT/'artifacts/static-study-gpu-20261022'
S2=ROOT/'artifacts/static-spillover-study-20261022'
SOURCE=S1/'execution-source-v5'
sys.path.insert(0,str(SOURCE))
from gsdeploy.calibration import _import_source
from gsdeploy.quality import quality_digest

FAMILIES=['generalist_1','generalist_2','generalist_3','text','image','ocr','asr']
FEATURES=['alpha_'+m for m in FAMILIES]+['A_self','A_other','M_resident','H_self','centered_log_b_x_A_other']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def positive(v):return type(v) in (float,int) and math.isfinite(v) and v>0

def design_rank(rows,width):
    a=np.asarray(rows,dtype=float).reshape((-1,width))
    if not len(a):return dict(rows=0,columns=width,rank=0,singular_values=[],condition_number=None,nullity=width)
    s=np.linalg.svd(a,compute_uv=False);tol=max(a.shape)*np.finfo(float).eps*s[0]
    rank=int(sum(s>tol));scales=np.linalg.norm(a,axis=0);scaled=a/np.where(scales>0,scales,1)
    ss=np.linalg.svd(scaled,compute_uv=False)
    return dict(rows=len(a),columns=width,rank=rank,nullity=width-rank,singular_values=s.tolist(),
        numerical_tolerance=float(tol),condition_number=float(s[0]/s[-1]) if rank==width else None,
        scaled_singular_values=ss.tolist(),scaled_condition_number=float(ss[0]/ss[-1]) if rank==width else None,
        zero_columns=[i for i in range(width) if scales[i]==0],
        rank_is_not_independent_run_count=True)

def identity_match(response,baseline):
    keys=['request_sha256','protocol_sha256','effective_generation']
    checks={k:bool(response.get(k) is not None and baseline.get(k) and any(response[k]==v for v in baseline[k])) for k in keys}
    # Concrete historical request fingerprints are required; missing fingerprints stay UNKNOWN.
    return all(checks.values()),checks

def identity_with_receipt(response,baseline,entry,scope):
    matched,checks=identity_match(response,baseline)
    if matched:return True,checks,'EXACT_THREE_OBSERVED_FIELDS'
    if not entry:return False,checks,'UNKNOWN_NO_SCOPED_EQUIVALENCE_RECEIPT'
    fields=['raw_sha256','request_id','family','task','n','sample_id']
    if any(entry.get(k)!=scope.get(k) for k in fields):return False,checks,'RECEIPT_SCOPE_MISMATCH'
    if entry.get('response_digest')!=digest(response) or entry.get('baseline_digest')!=digest(baseline):
        return False,checks,'RECEIPT_OBSERVATION_OR_BASELINE_MISMATCH'
    allowed={'generalist_2':'PROTOCOL_ONLY_EXACT_NATIVE_REQUEST_EQUIVALENCE',
        'image':'SIGLIP_EXACT_REQUEST_NON_GENERATIVE','ocr':'OCR_IDENTICAL_ADAPTER_INPUT_EXECUTION_CHECKPOINT_CONTRACT'}
    if allowed.get(scope['family'])!=entry.get('proof_kind'):return False,checks,'RECEIPT_KIND_NOT_VALID_FOR_FAMILY'
    return True,checks,entry['proof_kind']

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--identity-receipt',type=Path)
    parser.add_argument('--identity-receipt-sha256')
    parser.add_argument('--include-split107',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    fingerprints={};opened=[]
    def read(path,role):
        path=Path(path).resolve();fingerprints[str(path)]=sha(path);opened.append(dict(path=str(path),role=role))
        return json.loads(path.read_text())
    approvals={};identity_receipt=None
    if args.identity_receipt:
        assert args.identity_receipt_sha256 and sha(args.identity_receipt)==args.identity_receipt_sha256
        identity_receipt=read(args.identity_receipt,'exact_cpu_identity_equivalence_receipt')
        assert identity_receipt['schema']=='gsdeploy.context_identity_equivalence.v1'
        assert identity_receipt['scope']=='EXACT_TERMINAL_DEVELOPMENT_RECORDS_ONLY'
        for path,expected in identity_receipt['inputs'].items():assert sha(path)==expected
        approvals={(r['raw_sha256'],r['request_id']):r for r in identity_receipt['approvals']}
    config=read(S2/'text185-fine-development-v1/system-config.json','frozen_config');c=config['catalog']
    protocol=read(S1/'replay-v2-prepared-final/protocol.json','machine_role_reclassification_receipt')
    assert protocol['old_holdout_policy']=='Original eight conditions, including viewed v1 holdouts, are development only'
    sources=read(S1/'surrogate-v2/sources.json','source_metadata_only')
    ledger=[];excluded=[]
    for s in sources:
        if s['purpose']=='legacy_v1_development' and s['id'] in protocol['legacy_development_ids']:
            ledger.append(dict(**s,source_role='study01_legacy_calibration',role_receipt=str(S1/'replay-v2-prepared-final/protocol.json')))
        else:excluded.append(dict(path=s['path'],reason='EXCLUDED_INDEPENDENT_HOLDOUT_NO_RAW_READ'))
    assert len(ledger)==8
    for name,kind in [('five-development-point-audit-v1','five'),('asr19-replication-development-audit-v3','asr'),('text26-development-audit-v2','text')]:
        audit_path=S2/name/'audit.json';audit=read(audit_path,'completed_development_audit')
        for row in audit['conditions']+audit.get('excluded_user_paused_attempts',[]):
            source=dict(row['source']);source.setdefault('packet_path',row.get('packet_path'))
            if not source.get('packet_path'):source['packet_path']=audit['packet']['path']
            source.update(source_role='study02_'+kind+'_development',role_receipt=str(audit_path),
                declared_id=row['id'],audited_identity_verified=row.get('identity_verified',row.get('identity',{}).get('verified')))
            ledger.append(source)
    # Snapshot exact completed nosplit107 only. Never discover or open a running split.
    queue_path=S2/'text185-fine-development-v1/gpu-runs-v1/queue-progress.json'
    queue_bytes=queue_path.read_bytes();queue=json.loads(queue_bytes)
    queue_digest=hashlib.sha256(queue_bytes).hexdigest()
    save(out/'fine-queue-snapshot.json',dict(source=str(queue_path),sha256=queue_digest,snapshot=queue))
    opened.append(dict(path=str(queue_path),role='terminal_no107_receipt_only_mutable_metadata_snapshot',sha256=queue_digest))
    no107=[j for j in queue['jobs'] if j['label']=='text185_g2share_0of1_seed107']
    assert len(no107)==1 and no107[0]['status'] in {'completed','failed'}
    ledger.append(dict(**no107[0]['raw'],id=no107[0]['job_id'],source_role='study02_text185_no107_development',
        packet_path=str(S2/'text185-fine-development-v1/experiment-packet.json'),role_receipt=str(queue_path)))
    if args.include_split107:
        ledger.append(dict(path=str(S2/'text185-fine-development-v1/gpu-runs-v1/text185_g2share_1of32_seed107-attempt1/run.json'),
            sha256='05a1aa8f6b8aee93c909a8f065dabf706470f2e51254c0a77442dfb044fcee19',
            id='aa454b73a148202a018b3fd00cf082c9769e9f8e323ef84a03ee3ffa5788f41f',
            source_role='study02_text185_split107_development',packet_path=str(S2/'text185-fine-development-v1/experiment-packet.json'),
            role_receipt='Explicit parent authorization of exact terminal seed107 split SHA; not queue discovery'))
    excluded.append(dict(path=queue.get('current',{}).get('output'),reason='RUNNING_OR_UNDECLARED_FINE_JOBS_NOT_OPENED'))
    unique={};duplicates=[]
    for s in ledger:
        if s['sha256'] in unique:duplicates.append(dict(path=s['path'],same_raw_as=unique[s['sha256']]['path']));continue
        unique[s['sha256']]=s
    ledger=list(unique.values())
    for s in ledger:
        p=Path(s['path']).resolve()
        if any(any(word in part.lower() for word in ('holdout','validation','confirmation')) for part in p.parts):
            raise ValueError('Result-path holdout/validation/confirmation forbidden')
        assert p.is_relative_to(S1/'calibration') or p.is_relative_to(S2)
        assert p.name=='run.json' and sha(p)==s['sha256']

    # Standalone closed-loop records define b only, never become open-load regressors.
    baselines={};baseline_inventory=[]
    for m,ts in c['task_replica_profiles'].items():
        for t,states in ts.items():
            for sn,profile in states.items():
                grouped=defaultdict(list)
                for filename in profile['raw_service_samples']:
                    p=Path(filename).resolve()
                    assert p.is_relative_to(ROOT/'artifacts/retained-replica-profiles-20261021') and p.suffix=='.jsonl'
                    fingerprints[str(p)]=sha(p);opened.append(dict(path=str(p),role='closed_loop_baseline_only'))
                    for line in p.read_text().splitlines():
                        if not line.strip():continue
                        r=json.loads(line)
                        if r.get('success') is True and positive(r.get('service_ms')):grouped[r['sample_id']].append(r)
                samples={}
                for sample,rr in grouped.items():
                    samples[sample]=dict(ms=statistics.median(r['service_ms'] for r in rr),
                        **{k:[r[k] for r in rr if r.get(k) is not None] for k in ['request_sha256','protocol_sha256','effective_generation']})
                ids=c['quality']['input_order'][t];missing=sorted(set(ids)-set(samples))
                mean=statistics.mean(samples[i]['ms'] for i in ids) if not missing else None
                logmean=statistics.mean(math.log(samples[i]['ms']) for i in ids) if not missing else None
                baselines[m,t,int(sn)]=dict(samples=samples,mean=mean,logmean=logmean)
                baseline_inventory.append(dict(family=m,task=t,n=int(sn),input_count=len(samples),required=len(ids),missing=missing,
                    source=profile['source'],profile_mean_ms=profile['mean_service_ms'],input_median_mean_ms=mean,
                    role='CLOSED_LOOP_BASELINE_ONLY_NOT_OPEN_LOAD_TRAINING'))

    runs=[];requests=[];paths=[]
    for index,s in enumerate(ledger):
        raw=read(s['path'],s['source_role']);assert raw['status'] in {'completed','failed','aborted'}
        seed=raw['transport_config']['random_seed'];assert seed not in {61,62,71,72,73,81,82,108}
        packet_path=Path(s['packet_path']);packet=read(packet_path,'bound_development_packet')
        if s.get('packet_sha256'):assert sha(packet_path)==s['packet_sha256']
        jobid=raw['experiment_job_id'];job=next(j for j in packet['jobs'] if j['id']==jobid)
        assert job['phase']=='calibration'
        assert raw.get('experiment_phase')=='calibration' or (raw.get('experiment_phase') is None and s['source_role']=='study01_legacy_calibration')
        identity=raw.get('executed_measurement_identity',{})
        identity_ok=(raw.get('executed_measurement_id')==quality_digest(identity)
            and raw.get('plan')==job['plan'] and raw['plan']['routing']==job['identity']['routing']
            and raw['measurement_window']['trace_sha256']==job['identity']['trace_sha256'])
        quality_identity_ok=all(raw['config']['catalog']['quality']['identity'].get(k)==c['quality']['identity'].get(k)
            for k in ('input_pool_sha256','reference_sha256','protocol_sha256'))
        baseline_catalog_ok=all(raw['config']['catalog'].get(k)==c.get(k)
            for k in ('task_replica_profiles','capacity_factors','memory_table_mib','safe_factor','models'))
        strict_error=None;strict=False
        try:
            _,_,receipt=_import_source(dict(id=jobid,path=s['path'],sha256=s['sha256'],
                packet_path=str(packet_path),packet_sha256=sha(packet_path)),{identity['runtime_code_sha256']})
            strict=bool(receipt)
        except (ValueError,KeyError,TypeError) as exc:strict_error=str(exc)
        plan=raw['plan'];x=plan['deployment'];routing=plan['routing'];rates=raw['config']['workload']
        required=[(m,t,x[m]) for t,row in routing.items() for m,q in row.items() if q>0]
        missing=[list(k) for k in required if k not in baselines or baselines[k]['mean'] is None]
        A={m:0. for m in x};entropy={m:0. for m in x}
        if not missing:
            for t,row in routing.items():
                for m,q in row.items():
                    if q:A[m]+=q*baselines[m,t,x[m]]['mean']/1000/c['capacity_factors'][m][str(x[m])]
            for m in x:
                shares=[row.get(m,0) for row in routing.values() if row.get(m,0)>0];total=sum(shares)
                entropy[m]=-sum(q/total*math.log(q/total) for q in shares) if total else 0.
        resident=sum(c['memory_table_mib'][m][str(n)] for m,n in x.items())/49140
        memory=raw.get('memory_evidence',{});energy=raw.get('energy_evidence',{});utility=raw.get('evaluation',{}).get('utility',{})
        state=raw.get('evaluation',{}).get('stability',{}).get('stability_state_v2','UNKNOWN')
        run_valid=bool(identity_ok and quality_identity_ok and baseline_catalog_ok and strict and raw['status']=='completed' and memory.get('resource_valid') is True
            and utility.get('eligible') is True and state in {'steady','recoverable_congested'} and not missing)
        runid=f'run{index:02d}'
        rr=dict(run_id=runid,**s,seed=seed,status=raw['status'],deployment=x,routing=routing,workload=rates,
            horizon_seconds=raw['transport_config']['measurement_seconds'],identity_verified=identity_ok,strict_import_passed=strict,
            strict_import_error=strict_error,run_valid_for_fit=run_valid,stability=state,
            quality_pool_identity_verified=quality_identity_ok,baseline_catalog_identity_verified=baseline_catalog_ok,
            offered=len(raw['request_records']),status_counts=dict(Counter(r['status'] for r in raw['request_records'])),
            missing_context_baselines=missing,A_by_family=A if not missing else None,M_resident=resident,H_by_family=entropy,
            memory_resource_valid=memory.get('resource_valid'),safe_memory_mib=memory.get('safe_memory_mib'),
            energy_coverage_valid=energy.get('coverage_valid'),run_joules=energy.get('run_joules'),startup_joules=energy.get('startup_joules'),
            M_features=[1.,resident,sum(x.values())],E_features=[1.,sum(A.values()),sum(x.values())] if not missing else None)
        runs.append(rr);groups=defaultdict(list)
        for rec in raw['request_records']:
            response=rec.get('worker_response',{});m=raw['replica_to_family'][rec['model']];t=rec['task'];n=x[m]
            sample=rec.get('pool_id');b=baselines.get((m,t,n),{}).get('samples',{}).get(sample)
            scope=dict(raw_sha256=s['sha256'],request_id=rec['request_id'],family=m,task=t,n=n,sample_id=sample)
            identity_pass,checks,identity_basis=identity_with_receipt(response,b or {},approvals.get((s['sha256'],rec['request_id'])),scope)
            identity_status={k:('MATCH' if checks[k] else 'UNKNOWN_MISSING_HASH_OR_CONTROL' if response.get(k) is None or not (b or {}).get(k) else 'DIFFERENT_REQUIRES_EQUIVALENCE_RECEIPT') for k in checks}
            service=response.get('service_ms');success=rec.get('status')=='completed' and response.get('success') is True and positive(service)
            features=None;ratio=None
            if b and not missing:
                other=sum(v for k,v in A.items() if k!=m)
                features=[float(f==m) for f in FAMILIES]+[A[m],other,resident,entropy[m],(math.log(b['ms'])-baselines[m,t,n]['logmean'])*other]
                if success:ratio=service/b['ms']
            row=dict(run_id=runid,request_id=rec['request_id'],family=m,task=t,n=n,sample_id=sample,
                alias=rec['model'],status=rec.get('status'),success=success,
                baseline_ms=b['ms'] if b else None,service_ms=service if positive(service) else None,
                handler_ms=rec.get('worker_latency_ms'),queue_wait_ms=rec.get('worker_queue_wait_ms'),
                e2e_ms=rec.get('e2e_latency_ms'),baseline_identity_verified=identity_pass,identity_checks=checks,
                identity_status=identity_status,
                identity_basis=identity_basis,
                service_ratio=ratio,log_service_ratio=math.log(ratio) if ratio else None,
                features=features,valid_for_fit=bool(run_valid and identity_pass and success and features is not None),
                diagnostic_generated_tokens=response.get('generated_tokens'),diagnostic_input_tokens=response.get('input_tokens'))
            requests.append(row);groups[m,t,n].append(row)
        for (m,t,n),rows in groups.items():
            good=[r for r in rows if r['success'] and r['baseline_ms'] is not None]
            ids=set(r['sample_id'] for r in good);counts=Counter(r['sample_id'] for r in good)
            paths.append(dict(run_id=runid,family=m,task=t,n=n,offered=len(rows),successful=len(good),
                identity_verified_success=sum(r['baseline_identity_verified'] for r in good),fit_records=sum(r['valid_for_fit'] for r in rows),
                input_coverage=len(ids),required_inputs=len(c['quality']['input_order'][t]),min_repetitions=min(counts.values()) if counts else 0,
                ratio_of_service_sums=sum(r['service_ms'] for r in good)/sum(r['baseline_ms'] for r in good) if good else None,
                median_per_request_ratio=statistics.median(r['service_ratio'] for r in good) if good else None,
                run_valid_for_fit=run_valid,run_status=raw['status'],memory_resource_valid=memory.get('resource_valid')))

    valid=[r for r in requests if r['valid_for_fit']]
    contexts=defaultdict(list)
    for r in valid:contexts[r['run_id'],r['family'],r['task'],r['n']].append(r['features'])
    blocks=[np.mean(v,axis=0).tolist() for v in contexts.values()]
    # Diagnostic feature support includes failed/identity-unverified observations, never fit eligible.
    diag=[r['features'] for r in requests if r['success'] and r['features'] is not None]
    rank=dict(features=FEATURES,strict_request_design=design_rank([r['features'] for r in valid],12),
        strict_run_path_block_design=design_rank(blocks,12),diagnostic_all_success_design=design_rank(diag,12),
        independent_valid_runs=len({r['run_id'] for r in valid}),fitted=False)
    covered_columns=[i for i in range(7) if any(r['features'][i] for r in valid)]
    minimal_columns=covered_columns+[7,8];extended_columns=covered_columns+list(range(7,12))
    local_identifiability=dict(
        minimal_features=[FEATURES[i] for i in minimal_columns],
        minimal_request_design=design_rank([[r['features'][i] for i in minimal_columns] for r in valid],len(minimal_columns)),
        minimal_run_path_design=design_rank([[r[i] for i in minimal_columns] for r in blocks],len(minimal_columns)),
        extended_features=[FEATURES[i] for i in extended_columns],
        extended_run_path_design=design_rank([[r[i] for i in extended_columns] for r in blocks],len(extended_columns)),
        full_seven_family_rank_is_not_a_prerequisite=True,local_development_fitting_possible=True,
        eligibility_warning='Identifiability permits development fitting, not accuracy approval. Per-path input/n/context and independent-run coverage must bound the local domain; ASR n3 single-run and image n1 sparse coverage are weak support.')
    # Proposed open-load anchors are synthetic feature designs only, not observations.
    anchor_candidates=[]
    base_matrix=[r['features'] for r in valid]
    base_rank=rank['strict_request_design']['rank']
    for m in FAMILIES:
        t='text' if m.startswith('generalist') else m
        b=baselines[m,t,1]
        for pressure in (.15,.35):
            q=pressure*1000*c['capacity_factors'][m]['1']/b['mean']
            mem=c['safe_factor']*c['memory_table_mib'][m]['1']
            form_rho=q*c['task_replica_profiles'][m][t]['1']['mean_service_ms']/1000/c['capacity_factors'][m]['1']
            resident=c['memory_table_mib'][m]['1']/49140
            fs=[float(k==m) for k in FAMILIES]+[pressure,0.,resident,0.,0.]
            extra=design_rank(base_matrix+[fs],12)['rank']-base_rank
            anchor_candidates.append(dict(id=f'{m}_n1_open_anchor_{pressure}',deployment={m:1},
                calibration_traffic={t:{m:q}},workload={t:q},baseline_pressure=pressure,formal_profile_rho=form_rho,
                safe_residency_proxy_mib=mem,proxy_constraints_pass=form_rho<=.8 and mem<=49140,
                synthetic_features=fs,rank_gain_vs_current_strict=extra,scheduled=False,
                input_requirement='all original task inputs >=3 successful observations in >=2 independent execution blocks; determine horizon before preparing',
                other_tasks='Explicit calibration subprotocol only; cannot masquerade as full-task target workload'))
    missing_identity=Counter((r['family'],r['task'],k,v) for r in requests if r['success']
        for k,v in r['identity_status'].items() if v!='MATCH')
    fitruns=[r for r in runs if r['run_valid_for_fit']]
    resources=dict(M_features=['intercept','raw_resident_over_budget','workers'],
        E_features=['intercept','sum_baseline_pressure','workers'],
        M_design=design_rank([r['M_features'] for r in fitruns if positive(r['safe_memory_mib'])],3),
        E_design=design_rank([r['E_features'] for r in fitruns if r['energy_coverage_valid'] is True and positive(r['run_joules'])],3),
        fitted=False,limitations='Rank only; no M domain bound or E coefficients inferred. Actual peak, steady board integral and startup remain distinct.')
    coverage=[]
    for m in FAMILIES:
        for sn in c['capacity_factors'][m]:
            n=int(sn)
            if not n:continue
            for t in c['models'][m]['executable_tasks']:
                rs=[r for r in valid if (r['family'],r['task'],r['n'])==(m,t,n)]
                counts=Counter(r['sample_id'] for r in rs);blocks_seen={r['run_id'] for r in rs}
                coverage.append(dict(family=m,task=t,n=n,baseline_available=(m,t,n) in baselines,
                    fit_records=len(rs),independent_runs=len(blocks_seen),covered_inputs=len(counts),
                    missing_inputs=sorted(set(c['quality']['input_order'][t])-set(counts)),
                    min_repetitions_across_full_pool=min((counts[i] for i in c['quality']['input_order'][t]),default=0)))
    recommendations=[dict(priority=0,condition='CPU audit concrete shared-worker/request protocol equivalence for G2, missing OCR fingerprints and absent image/ocr generation-control semantics',
        why='Strict identity feature rank excludes unverified rows conservatively; missing or different hashes are not proof payloads differ. Use existing CPU equivalence receipts if exact bindings match; do not silently waive.',gpu_scheduled=False),
        dict(priority=1,condition='G1 n1 text-only pressure .15 and .35; G3 n1 text-only pressure .15 and .35',
        why='Both family intercept columns have no permitted open-load coverage; repeats of text/G2 cannot recover them',gpu_scheduled=False),
        dict(priority=2,condition='Fixed resident text2+G2n1+OCRn1, hold text pressure fixed and vary G2 activity0/.15/.35; matched single-family/idle-resident block',
            why='Separate active A_other from resident context and own pressure; rank/condition selection after complete input coverage',gpu_scheduled=False),
        dict(priority=3,condition='Generalist representative task mixtures and same n replacement selected by rank/coverage; full input coverage >=3 per input in >=2 independent blocks',
            why='Entropy/path/n applicability cannot come from sparse .01/H600 backgrounds; nine original profile gaps remain UNKNOWN',gpu_scheduled=False)]
    minimal_seven_family=[]
    for first in [a for a in anchor_candidates if a['id'].startswith('generalist_1_')]:
        for second in [a for a in anchor_candidates if a['id'].startswith('generalist_3_')]:
            diagnosis=design_rank(blocks+[first['synthetic_features'],second['synthetic_features']],12)
            minimal_seven_family.append(dict(candidate_ids=[first['id'],second['id']],design=diagnosis,
                scaled_smallest_singular_value=diagnosis['scaled_singular_values'][-1]))
    minimal_seven_family.sort(key=lambda v:v['scaled_smallest_singular_value'],reverse=True)
    if identity_receipt:
        recommendations=[dict(priority=0,condition='Fit only an identifiable observed local domain; compare baseline/family+self+other with limited extensions using run-held-out development diagnostics',
            why='Exact CPU identity closure makes observed five-family local forms identifiable; seven-family full rank is not a prerequisite.',gpu_scheduled=False),
            dict(priority=1,condition='If seven-pool coverage is required next: one G1 n1 text-only anchor and one G3 n1 text-only anchor, chosen by block design conditioning, each repeated in two independent execution blocks',
                chosen_candidate_ids=minimal_seven_family[0]['candidate_ids'],
                why='Two new family directions suffice for rank; this does not prove broad context generalization. No mandatory four low/high anchor identities or saturated reruns.',gpu_scheduled=False),
            dict(priority=2,condition='Use local-model grouped diagnostics to choose one discriminating matched-context followup only if a material error remains',
                why='Avoid filling every missing profile/state before testing the actual decision problem. Sparse image inputs and single-run ASR n3 stay scoped or excluded.',gpu_scheduled=False)]
    for p,v in fingerprints.items():assert sha(p)==v
    report=dict(schema='gsdeploy.context_design_inventory.v1',status='INVENTORIED_NOT_FITTED',model='log S=log b+7 family intercepts+5 static context terms',
        feature_definition='A uses mean per-input median baseline / capacity_factor; M raw residency/49140; H route-rps shares natural-log entropy; shape centers log b over full input pool',
        runs=len(runs),run_valid_count=sum(r['run_valid_for_fit'] for r in runs),requests=len(requests),valid_fit_records=len(valid),
        rank=rank,local_identifiability=local_identifiability,resources=resources,coverage=coverage,recommendations=recommendations,
        minimal_seven_family_synthetic_designs=minimal_seven_family,
        proposed_anchor_candidates=anchor_candidates,
        baseline_identity_gaps=[dict(family=m,task=t,field=k,status=v,requests=count) for (m,t,k,v),count in missing_identity.items()],
        excluded=excluded,duplicate_evidence=duplicates,
        closed_loop_role='Baselines and separate provenance only; not open-load regressors',
        leakage_policy='Only explicit8 legacy role-receipted calibration, audited ASR/text development, exact terminal no107 and optionally pinned split107. No result-path validation/confirmation/holdout reads; reserved seeds including108 excluded.',
        identity_equivalence_receipt=dict(path=str(args.identity_receipt.resolve()),sha256=args.identity_receipt_sha256) if args.identity_receipt else None,
        model_fitted=False,predictor_activated=False,gpu_started=False,core_modified=False,
        all_failures_preserved=True,production_U=None)
    for filename,data in [('report.json',report),('source-ledger.json',runs),('baseline-inventory.json',baseline_inventory),('run-path-ratios.json',paths),('requests.json',requests),('opened-inputs.json',opened),('proposed-anchor-candidates.json',anchor_candidates)]:save(out/filename,data)
    with (out/'context-features.csv').open('x',newline='') as f:
        fields=['run_id','request_id','family','task','n','sample_id','status','success','baseline_identity_verified','valid_for_fit','baseline_ms','service_ms','service_ratio',*FEATURES]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in requests:
            writer.writerow({**{k:row[k] for k in fields if k in row},**dict(zip(FEATURES,row['features'] or [None]*12))})
    text=['# 新上下文预测器：数据清单与12参数设计诊断','',
        f"状态INVENTORIED_NOT_FITTED；终态开发{len(runs)}运行，原run资格有效{sum(r['run_valid_for_fit'] for r in runs)}，严格输入身份/上下文匹配拟合可用{len(valid)}记录。未拟合/激活/启动GPU。",'',
        '逐run/path服务比保留failed与资源不合法记录作诊断；这些记录的valid_for_fit=False。逐请求服务用service_ms，handler/queue/e2e另列；实际生成token只作诊断不进特征。',
        '',f'请求设计：{rank["strict_request_design"]}', '',f'run/path块设计：{rank["strict_run_path_block_design"]}',
        '',f'全成功记录诊断设计（含不合格，不能冒充训练秩）：{rank["diagnostic_all_success_design"]}',
        '',f'M/E秩诊断：{resources}',
        '', '模型特征为7族截距+自身压力+其他压力+原驻留比+自身任务份额熵+中心化log基线×其他压力。b按同family/task/n/sample原始闭环中位数，n状态已包含副本影响，不额外乘n。',
        '',f'可辨识局部模型：{local_identifiability}',
        '', '下一步先在已观测、可辨识局部域开发正值模型，以run留一诊断决定是否需要补点。七池覆盖与局部拟合分开；七池最小设计仅需G1/G3各一个独立锚点身份（各至少两执行块），合成秩/条件选择见minimal_seven_family_synthetic_designs。缺失画像和稀疏输入不外推，完整列表见coverage；不能以请求很多或秩满替代独立运行覆盖与留组验证。任何追加点只为建议，不建立GPU队列。',
        '', 'source-ledger/requests/run-path-ratios保存身份、失败、M/E与服务比；opened-inputs给出完整读取清单。旧v1两条曾holdout的来源仅因原机器protocol显式legacy_development_ids及old_holdout_policy才允许读取；新旧独立验证/确认结果未读。旧failedtransfer未使用。','']
    (out/'README.zh.md').write_text('\n'.join(text))
    save(out/'manifest.json',dict(script={'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},inputs=fingerprints,
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()},all_inputs_unchanged=True,gpu_started=False))
    print(json.dumps({k:report[k] for k in ['status','runs','run_valid_count','requests','valid_fit_records','rank']},ensure_ascii=False))

if __name__=='__main__':main()
