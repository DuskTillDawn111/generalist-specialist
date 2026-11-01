"""CPU-only exact scoped identity receipts for terminal development records.

No missing hash is invented. G2 requests must differ ONLY by the shared policy
digest; OCR uses an identical non-generative adapter/input/checkpoint contract;
SigLIP still requires exact observed request and protocol hashes.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/retained-replica-profiles-20261021'
STUDY=ROOT/'artifacts/static-spillover-study-20261022'
FROZEN=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
sys.path.insert(0,str(FROZEN))
from gsdeploy.profiling.static_inputs import frozen_compact_adapter

SPLIT107=dict(path=str(STUDY/'text185-fine-development-v1/gpu-runs-v1/text185_g2share_1of32_seed107-attempt1/run.json'),
    sha256='05a1aa8f6b8aee93c909a8f065dabf706470f2e51254c0a77442dfb044fcee19',
    id='aa454b73a148202a018b3fd00cf082c9769e9f8e323ef84a03ee3ffa5788f41f')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def protocol_only_equivalent(old,new):
    a=dict(old);b=dict(new)
    if not a.pop('protocol_sha256',None) or not b.pop('protocol_sha256',None):return False
    return a==b

def contract_equal(original,runtime,task,*,protocol_changes=False):
    """Only output location and explicitly proven protocol digest may differ."""
    if original['candidate']!=runtime['task_candidates'].get(task):
        return False
    if original['execution']!=runtime['execution'] or original['adapter']!=runtime['adapter']:return False
    a=dict(original['resources']);b=dict(runtime['resources'])
    a.pop('request_output',None);b.pop('request_output',None)
    if protocol_changes:a.pop('protocol',None);b.pop('protocol',None)
    return a==b

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    inputs={};opened=[]
    def bind(path,role):
        p=Path(path).resolve();inputs[str(p)]=sha(p);opened.append(dict(path=str(p),role=role));return p
    def read(path,role):return json.loads(bind(path,role).read_text())
    def lines(path,role):return [json.loads(v) for v in bind(path,role).read_text().splitlines() if v.strip()]
    config=read(STUDY/'text185-fine-development-v1/system-config.json','frozen_config')['catalog']
    ledger=read(args.inventory/'source-ledger.json','explicit_terminal_development_allowlist')
    raw_sources=[dict(path=s['path'],sha256=s['sha256'],id=s.get('id')) for s in ledger]
    if SPLIT107['sha256'] not in {s['sha256'] for s in raw_sources}:raw_sources.append(SPLIT107)
    runtime_root=ROOT/'artifacts/static-study-v6-20261021/inputs'
    pool=lines(runtime_root/'runtime-inputs.jsonl','frozen_runtime_input_payloads')
    samples={r['pool_id']:r['sample'] for r in pool}
    compact_source=BASE/'source-qa'
    for p in (compact_source/'gsdeploy/backends').glob('compact*.py'):bind(p,'frozen_compact_cpu_request_source')
    bind(FROZEN/'gsdeploy/profiling/static_inputs.py','frozen_cpu_loader_source')
    Compact=frozen_compact_adapter(compact_source)
    # Never import a native executor: inspect only the fixed SigLIP AST/source.
    native_path=compact_source/'gsdeploy/backends/compact_native.py'
    native=bind(native_path,'non_generative_siglip_source_static_only').read_text()
    tree=ast.parse(native)
    sig=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_siglip')
    sig_source=ast.get_source_segment(native,sig)
    assert "'finish_reason': 'classified'" in sig_source and "'generated_tokens': 0" in sig_source
    assert 'effective_generation' not in sig_source and '.generate(' not in sig_source
    assert "if self.family == 'siglip':\n                raw, observation, extras = self._siglip(request, prepared['media'])" in native
    adapter_paths=[ROOT/'scripts/retained_ppocrv6_profile_adapter.py',
        BASE/'source-qa/gsdeploy/profiling/retained_ppocrv6.py',BASE/'source-granite/gsdeploy/profiling/retained_ppocrv6.py']
    adapter_hashes=[sha(bind(p,'exact_non_generative_ocr_adapter_source')) for p in adapter_paths]
    assert len(set(adapter_hashes))==1
    ocr_code=adapter_paths[0].read_text()
    assert 'post_process_text_recognition' in ocr_code and "finish_reason='ctc_complete'" in ocr_code
    assert all(k not in ocr_code for k in ['effective_generation','request_sha256','protocol_sha256'])
    # Bind actual loader paths as well as the adapter: no claim of changed code equivalence.
    for p in (BASE/'source-qa/gsdeploy/profiling/runner.py',FROZEN/'scripts/static_study_worker.py'):
        bind(p,'frozen_adapter_loader_source_static_only')
    contracts={};baseline_by_key={};profile_proofs=[];request_proofs=[]
    for family in ['generalist_2','image','ocr']:
        runtime_path=runtime_root/'workers'/family/'worker.json';runtime=read(runtime_path,'runtime_worker_contract')
        for task,states in config['task_replica_profiles'][family].items():
            original_path=BASE/'packets'/f'{family}-{task}'/'worker.json'
            original=read(original_path,'original_profile_worker_contract')
            assert contract_equal(original,runtime,task,protocol_changes=family!='ocr')
            assert original['execution']==runtime['execution'] and original['adapter']==runtime['adapter']
            assert runtime['task_candidates'][task]==original['candidate']
            ar=dict(original['resources']);br=dict(runtime['resources'])
            for r in (ar,br):r.pop('request_output',None)
            if family!='ocr':
                old_policy=read(ar.pop('protocol'),'frozen_input_protocol_contract_only')
                new_policy=read(br.pop('protocol'),'shared_runtime_input_protocol_contract')
                classes=read(ar['classmap'],'frozen_classmap_input')
                assert br['classmap']==original['resources']['classmap']
                old_adapter=Compact(old_policy,classes);new_adapter=Compact(new_policy,classes)
            else:
                assert runtime['adapter_source_sha256']==adapter_hashes[0]
                assert Path(runtime['adapter_source']).resolve()==adapter_paths[0]
                assert sha(bind(ar['inference_source'],'frozen_ocr_inference_contract_source'))==ar['inference_source_sha256']
                for kind in ['det','rec']:
                    read(Path(ar['checkpoint'])/f'PaddlePaddle--PP-OCRv6_medium_{kind}_safetensors/download-receipt.json','ocr_recorded_checkpoint_receipt_no_weights_read')
            assert ar==br
            job=read(BASE/'packets'/f'{family}-{task}'/'job.json','original_profile_job_input_contract')
            original_samples={r['id']:r for r in job['samples']}
            manifest=hashlib.sha256(json.dumps(job['samples'],sort_keys=True).encode()).hexdigest()
            for n,profile_ref in states.items():
                profile=read(profile_ref['source'],'closed_loop_profile_contract_only')
                assert profile['status']=='COMPLETE' and profile['replicas']==int(n)
                assert profile['manifest_sha256']==manifest and profile['model']==original
                grouped=defaultdict(list)
                for path in profile_ref['raw_service_samples']:
                    for row in lines(path,'closed_loop_service_identity_only'):
                        if row.get('success') is True and isinstance(row.get('service_ms'),(float,int)) and row['service_ms']>0:grouped[row['sample_id']].append(row)
                profile_proofs.append(dict(family=family,task=task,n=int(n),profile=profile_ref['source'],
                    manifest_sha256=manifest,worker_sha256=sha(original_path),runtime_worker_sha256=sha(runtime_path),samples=len(grouped)))
                for sid,baseline_rows in grouped.items():
                    assert sid in samples and sid in original_samples
                    if family=='ocr':
                        assert samples[sid]==original_samples[sid]
                        assert sha(bind(samples[sid]['input'],'exact_ocr_media_input'))==samples[sid]['input_sha256']
                        assert all(r.get('finish_reason')=='ctc_complete' and all(r.get(k) is None for k in ['request_sha256','protocol_sha256','effective_generation']) for r in baseline_rows)
                        proof=dict(kind='OCR_IDENTICAL_ADAPTER_INPUT_EXECUTION_CHECKPOINT_CONTRACT',sample_payload_sha256=digest(samples[sid]),
                            original_adapter_sha256=adapter_hashes[0],runtime_adapter_sha256=adapter_hashes[0],non_generative=True)
                    else:
                        old=old_adapter.build(original['candidate'],original_samples[sid])
                        old_runtime_payload=old_adapter.build(original['candidate'],samples[sid])
                        new=new_adapter.build(runtime['task_candidates'][task],samples[sid])
                        assert old==old_runtime_payload and protocol_only_equivalent(old,new)
                        old_fp=digest(old);new_fp=digest(new)
                        assert all(r.get('request_sha256')==old_fp and r.get('protocol_sha256')==old['protocol_sha256'] for r in baseline_rows)
                        if family=='image':
                            assert old==new and old_fp==new_fp
                            assert all(r.get('finish_reason')=='classified' and r.get('generated_tokens')==0 and r.get('effective_generation') is None for r in baseline_rows)
                        proof=dict(kind='PROTOCOL_ONLY_EXACT_NATIVE_REQUEST_EQUIVALENCE' if family=='generalist_2' else 'SIGLIP_EXACT_REQUEST_NON_GENERATIVE',
                            old_request_sha256=old_fp,new_request_sha256=new_fp,old_protocol_sha256=old['protocol_sha256'],new_protocol_sha256=new['protocol_sha256'],
                            native_request_without_protocol_sha256=digest({k:v for k,v in old.items() if k!='protocol_sha256'}),
                            non_generative=family=='image')
                    baseline=dict(ms=statistics.median(r['service_ms'] for r in baseline_rows),
                        **{k:[r[k] for r in baseline_rows if r.get(k) is not None] for k in ['request_sha256','protocol_sha256','effective_generation']})
                    key=(family,task,int(n),sid);contracts[key]=proof;baseline_by_key[key]=baseline
                    request_proofs.append(dict(family=family,task=task,n=int(n),sample_id=sid,**proof,baseline_digest=digest(baseline)))
    approvals=[];unknown=[];counts=Counter()
    for source in raw_sources:
        path=Path(source['path']).resolve()
        assert path.name=='run.json' and (path.is_relative_to(STUDY) or path.is_relative_to(ROOT/'artifacts/static-study-gpu-20261022/calibration'))
        assert not any(any(v in part.lower() for v in ['holdout','validation','confirmation']) for part in path.parts)
        assert sha(path)==source['sha256']
        raw=read(path,'explicit_terminal_development_raw')
        assert raw['status'] in ['completed','failed','aborted'] and raw['transport_config']['random_seed'] not in [61,62,71,72,73,81,82,108]
        if source['sha256']==SPLIT107['sha256']:assert raw['experiment_job_id']==SPLIT107['id'] and raw['status']=='completed'
        workers=raw['executed_measurement_identity']['workers']
        for family in set(raw['replica_to_family'].values()) & {'generalist_2','image','ocr'}:
            w=workers[family];p=Path(w['spec'])
            assert p.resolve()==runtime_root/'workers'/family/'worker.json' and w['spec_sha256']==sha(p)
            assert w['execution']==json.loads(p.read_text())['execution']
        for rec in raw['request_records']:
            family=raw['replica_to_family'][rec['model']]
            if family not in ['generalist_2','image','ocr']:continue
            response=rec.get('worker_response',{});key=(family,rec['task'],raw['plan']['deployment'][family],rec['pool_id'])
            proof=contracts.get(key);baseline=baseline_by_key.get(key);reason=None
            if proof is None:reason='NO_EXACT_PROFILE_SAMPLE_CONTRACT'
            elif response.get('sample_id')!=rec['pool_id'] or response.get('task')!=rec['task']:reason='RESPONSE_SAMPLE_OR_TASK_UNKNOWN'
            elif family=='generalist_2':
                if response.get('request_sha256')!=proof['new_request_sha256'] or response.get('protocol_sha256')!=proof['new_protocol_sha256']:reason='RUNTIME_NATIVE_REQUEST_HASH_NOT_EXACT'
                elif response.get('effective_generation') is None or response['effective_generation'] not in baseline['effective_generation']:reason='GENERATION_CONTROL_NOT_EXACT'
            elif family=='image':
                if response.get('request_sha256')!=proof['new_request_sha256'] or response.get('protocol_sha256')!=proof['new_protocol_sha256']:reason='RUNTIME_NATIVE_REQUEST_HASH_NOT_EXACT'
                elif response.get('finish_reason')!='classified' or response.get('generated_tokens')!=0 or response.get('effective_generation') is not None:reason='NON_GENERATIVE_SIGLIP_CONTRACT_NOT_EXACT'
            elif family=='ocr':
                if any(response.get(k) is not None for k in ['request_sha256','protocol_sha256','effective_generation']):reason='OCR_UNEXPECTED_ADAPTER_FIELDS'
                elif response.get('finish_reason')!='ctc_complete' or response.get('output_sha256')!=hashlib.sha256(response.get('raw_output','').encode()).hexdigest():reason='OCR_OUTPUT_ADAPTER_CONTRACT_NOT_EXACT'
            if reason:
                counts[family+':UNKNOWN']+=1;unknown.append(dict(raw_sha256=source['sha256'],request_id=rec['request_id'],family=family,reason=reason));continue
            counts[family+':PROVEN']+=1
            approvals.append(dict(raw_sha256=source['sha256'],request_id=rec['request_id'],family=family,task=rec['task'],n=key[2],sample_id=key[3],
                response_digest=digest(response),baseline_digest=digest(baseline),proof_kind=proof['kind']))
    for path,expected in inputs.items():assert sha(path)==expected
    receipt=dict(schema='gsdeploy.context_identity_equivalence.v1',scope='EXACT_TERMINAL_DEVELOPMENT_RECORDS_ONLY',
        inputs=inputs,source_raws=raw_sources,profile_proofs=profile_proofs,request_contract_proofs=request_proofs,approvals=approvals,unknown=unknown,
        counts=dict(counts),policy='No missing hash fabricated; non-generative controls explicitly inapplicable only for exact proven image/OCR contract; run/resource validity unchanged.',
        excluded_result_seeds=[61,62,71,72,73,81,82,108],fitted=False,predictor_activated=False,gpu_started=False)
    save(out/'receipt.json',receipt);save(out/'opened-inputs.json',opened)
    save(out/'manifest.json',dict(script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),files={p.name:sha(p) for p in out.iterdir() if p.is_file()},inputs=inputs,all_inputs_unchanged=True))
    (out/'README.zh.md').write_text('# 精确 CPU 请求身份等价收据\n\n'+json.dumps(dict(counts),ensure_ascii=False)+'\n\nG2 逐输入原/共享请求仅协议摘要不同，原/新指纹都由冻结 CPU 构造器重算并和实测逐条匹配；generation 仍严格匹配。OCR 原画像/运行适配器逐字节相同，完整输入、执行和检查点收据契约相同，三字段原本均未输出；image 的请求/协议实测严格相同，SigLIP 静态分支证明不生成。缺字段没有补造；逐记录授权只对 raw SHA、request_id、完整 response 摘要和原基线摘要有效。失败和资源资格仍由原导入器决定。未读 108/验证/确认/留出结果；共同冻结输入契约即使路径带 confirmation 也仅作为输入读取。未拟合或激活。\n')
    print(json.dumps(dict(counts)))

if __name__=='__main__':main()
