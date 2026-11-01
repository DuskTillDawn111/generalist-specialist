"""CPU public ChatQA tokenizer, observed affine shape rules and hull checks.

Loads tokenizer JSON only. Never loads model weights, executes CUDA, consumes
heldout timings, or infers generated output length from a training response.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import linprog

from gsdeploy.forecasting.kernel_baseline import FEATURES
from .kernel_baseline_intake_v3 import checked, _verify, WEIGHT_SELECTION
from .kernel_domain_bridge_review_v3 import load_report, request_kernels
from .native_collect import receipt, save, context_identity
from .profiling import digest
from .smoke_verifier_v3 import load_cpu_adapter


TOKENIZER_HELPER = '''import hashlib,json,sys
from transformers import AutoTokenizer
import transformers,tokenizers
doc=json.load(sys.stdin)
t=AutoTokenizer.from_pretrained(doc['checkpoint'],local_files_only=True,trust_remote_code=False)
if not t.bos_token or type(t.bos_token_id) is not int:raise ValueError('Explicit BOS required')
rows=[]
for r in doc['requests']:
 opts=dict(r['tokenizer_options']);opts.pop('return_tensors',None)
 ids=t(t.bos_token+r['text'],**opts)['input_ids']
 if not ids or ids[0]!=t.bos_token_id:raise ValueError('Explicit BOS not preserved')
 rows.append(dict(key=r['key'],input_tokens=len(ids),input_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),bos_token_id=t.bos_token_id))
print(json.dumps(dict(rows=rows,transformers=transformers.__version__,tokenizers=tokenizers.__version__,tokenizer_class=type(t).__name__,model_weights_loaded=False,GPU_workload_executed=False)))
'''


def public_content_sha(sample):
    # IDs and reference answers do not change the public model input.
    return digest({key:sample[key] for key in ('task','dataset','context','question')})


def learn_module_rule(records):
    """Require one actual module's prefill tensor relation at distinct lengths."""
    signatures=[]; lengths=set(); modes=set()
    for record,semantic in records:
        length=semantic['input_sequence_length'];batch=semantic['input_batch_size']
        if semantic['semantic_phase']!='prefill' or batch!=1 or type(length) is not int or length<2:
            raise ValueError('Distinct batch-one prefill shape evidence required')
        name=WEIGHT_SELECTION.get(record['implementation'])
        weights=[t for t in record['direct_tensors'] if name is not None and t['name']==name]
        if len(weights)!=1:raise ValueError('Explicit named weight required')
        i,o=record['input'],record['output'];nf,no=record['in_features'],record['out_features']
        if len(i['shape'])!=3 or len(o['shape'])!=3 or i['shape'][0]!=1 or o['shape'][0]!=1:
            raise ValueError('Unsupported observed tensor rank/batch')
        positions=i['shape'][1]
        if i['shape'][2]!=nf or o['shape']!=[1,positions,no]:raise ValueError('Unsupported declared affine shape')
        mode='all_input_positions' if positions==length else 'last_position_only' if positions==1 else None
        if mode is None:raise ValueError('Unknown public length to operator positions relation')
        if i['logical_bytes']!=positions*nf*i['element_size_bytes'] or o['logical_bytes']!=positions*no*o['element_size_bytes']:
            raise ValueError('Tensor byte metadata does not equal declared shape')
        if record['dense_gemm_flops']!=2*positions*nf*no:raise ValueError('Declared GEMM feature mismatch')
        modes.add(mode);lengths.add(length)
        signatures.append(dict(module=record['module'],implementation=record['implementation'],in_features=nf,out_features=no,
            input_dtype=i['dtype'],output_dtype=o['dtype'],input_element_bytes=i['element_size_bytes'],
            output_element_bytes=o['element_size_bytes'],weight_tensor_name=name,
            weight_dtype=weights[0]['dtype'],weight_bytes=weights[0]['logical_bytes']))
    if not signatures or len(lengths)<2 or len(modes)!=1 or any(s!=signatures[0] for s in signatures[1:]):
        raise ValueError('Consistent module relation at two distinct training lengths required')
    return dict(**signatures[0],positions_rule=next(iter(modes)),observed_prefill_lengths=sorted(lengths),
                rule_evidence_kind='measured_training_shape_relation',
                unseen_shape_execution_validated=False)


def project_features(rule, input_tokens, batch_size=1):
    if batch_size!=1 or type(input_tokens) is not int or input_tokens<1:
        raise ValueError('Positive public token count and observed batch one required')
    positions=input_tokens if rule['positions_rule']=='all_input_positions' else 1
    return dict(batch_size=1,weight_bytes=rule['weight_bytes'],
        input_tensor_bytes=positions*rule['in_features']*rule['input_element_bytes'],
        output_tensor_bytes=positions*rule['out_features']*rule['output_element_bytes'],
        flops=2*positions*rule['in_features']*rule['out_features'])


def hull_support(features, points):
    values=np.array(points,dtype=float)
    if not len(values):return dict(known=False,reason='NO_TRAINING_FEATURE_POINTS')
    scale=np.ptp(values,axis=0);scale[scale==0]=1;center=values.mean(axis=0)
    design=np.column_stack((np.ones(len(values)),(values-center)/scale))
    query=np.r_[1.,(np.array([features[f] for f in FEATURES])-center)/scale]
    weights=np.linalg.lstsq(design.T,query,rcond=None)[0]
    if np.max(np.abs(design.T@weights-query))>1e-8:
        return dict(known=False,reason='OUTSIDE_TRAINING_AFFINE_SUBSPACE')
    hull=linprog(np.zeros(len(values)),A_eq=design.T,b_eq=query,bounds=(0,None),method='highs')
    valid=bool(hull.success and np.max(np.abs(design.T@hull.x-query))<=1e-7)
    return dict(known=valid,reason='INSIDE_EXACT_GROUP_FEATURE_CONVEX_HULL' if valid else 'OUTSIDE_TRAINING_CONVEX_HULL')


def tokenize_public(context, output, samples):
    binding=context['worker_bindings']['text'];spec=json.loads(Path(binding['spec']).read_text())
    if receipt(binding['spec'])['sha256']!=binding['spec_sha256']:
        raise ValueError('Actual formatter worker configuration changed')
    command=binding['command'];root=Path(command[command.index('--source-root')+1]).resolve()
    formatter_sources=[]
    required={str(Path(spec['resources'][k]).resolve()) for k in ('protocol','classmap')}
    for source in context['execution_sources']:
        original=Path(source['original_path']).resolve()
        if (original.is_relative_to(root/'gsdeploy') and original.suffix=='.py') or str(original) in required:
            actual=receipt(original);_verify(source['frozen'])
            if actual['sha256']!=source['frozen']['sha256']:raise ValueError('Actual frozen formatter source changed')
            formatter_sources.append(actual);required.discard(str(original))
    if required or not any(s['path'].endswith('/gsdeploy/backends/compact.py') for s in formatter_sources):
        raise ValueError('Exact formatter/protocol/classmap source binding required')
    adapter=load_cpu_adapter(binding,spec);checkpoint=Path(spec['resources']['checkpoint']['path'])
    sources=[]
    for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','config.json'):
        path=checkpoint/name
        if not path.exists():continue
        actual=receipt(path)
        frozen=next((s['frozen'] for s in context['execution_sources'] if s['original_path']==str(path)),None)
        if frozen is None or actual['sha256']!=frozen['sha256']:
            raise ValueError('Tokenizer/config asset not bound to frozen training source: '+name)
        _verify(frozen);sources.append(actual)
    requests=[]
    for sample in samples:
        request=adapter.build(spec['task_candidates']['text'],sample);native=request['native']
        if native['prepend_bos'] is not True or native['tokenizer_options'].get('add_special_tokens') is not False:
            raise ValueError('Frozen ChatQA tokenizer options required')
        requests.append(dict(key=public_content_sha(sample),text=native['text'],tokenizer_options=native['tokenizer_options']))
    payload=dict(checkpoint=str(checkpoint),requests=requests)
    helper=output/'tokenizer-helper.py';helper.write_text(TOKENIZER_HELPER)
    env=dict(os.environ,USE_TORCH='0',USE_TF='0',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1',
             CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    proc=subprocess.run([binding['tasks']['text']['python'],str(helper)],input=json.dumps(payload),text=True,
                        capture_output=True,env=env,check=True)
    result=json.loads(proc.stdout);save(output/'tokenizer-results.json',result)
    return {r['key']:r for r in result['rows']},dict(assets=sources,helper=receipt(helper),
        result=receipt(output/'tokenizer-results.json'),python=binding['tasks']['text']['python'],
        actual_verified_formatter_sources=formatter_sources)


def audit(bridge_paths,intake_path,support_path,output,queries=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    intake=checked(receipt(intake_path));support=checked(receipt(support_path))
    for source in intake['sources']:_verify(source)
    train=[];contexts=[];sources=[];samples={};module_records=defaultdict(list);template_evidence=defaultdict(list)
    for bridge_path in bridge_paths:
        bridge=checked(receipt(bridge_path));sources.append(receipt(bridge_path))
        if bridge.get('status')!='PASS':raise ValueError('Passing real training bridge required')
        for domain,key in [('metadata','metadata_review'),('light-kernel','light_review'),('natural','natural_review')]:
            checked(bridge[key]);report,plan,raw=load_report(bridge[key]['path'],domain);context=checked(report['context'])
            if domain=='light-kernel':
                if context_identity(context)!=intake['execution_context_sha256']:
                    raise ValueError('Public shape mapper cannot pool different canonical execution contexts')
                contexts.append(context)
            path=next(s['frozen']['path'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl'))
            pool={r['pool_id']:r for r in map(json.loads,Path(path).read_text().splitlines())}
            parsed={}
            for row in raw['request_records']:
                if context['residents'][row['model']]!='text':continue
                sample=pool[row['pool_id']]['sample'];sha=public_content_sha(sample);samples[sha]=sample
                train.append(dict(domain=domain,raw=report['raw'],pool_id=row['pool_id'],alias=row['model'],
                    request_id=row['request_id'],public_content_sha256=sha,actual_input_tokens=row['worker_response']['input_tokens']))
                if domain=='light-kernel':
                    owner=report['owners'][row['model']]
                    if row['model'] not in parsed:parsed[row['model']]=checked(owner['parsed'])
                    items=request_kernels(parsed[row['model']],owner['sqlite']['path'],row)
                    sequence=[item['signature'] for item in items if item['group'].startswith('prefill_')]
                    if not sequence:raise ValueError('Actual prefill sequence required')
                    template_evidence[row['worker_response']['input_tokens']].append(dict(raw=report['raw'],
                        parsed=owner['parsed'],sqlite=owner['sqlite'],request_id=row['request_id'],
                        alias=row['model'],sequence_sha256=digest(sequence),sequence=sequence))
                if domain=='metadata':
                    obs=row['worker_response']['semantic_forward_observations']
                    forwards=[s for s in obs['records'] if s['semantic_phase']=='prefill']
                    if len(forwards)!=1:raise ValueError('Unique actual prefill required')
                    sem=forwards[0]
                    for record in obs['affine_operator_observations']['records']:
                        if sem['start_ns']<=record['start_ns']<record['end_ns']<=sem['end_ns']:
                            module_records[record['module']].append((record,sem))
    rules={};rejected=[]
    for module,records in module_records.items():
        try:rules[module]=learn_module_rule(records)
        except ValueError as exc:rejected.append(dict(module=module,reason=str(exc)))
    query_items=[] if queries is None else json.loads(Path(queries).read_text())
    query_samples=[q['sample'] for q in query_items]
    tokenized,token_sources=tokenize_public(contexts[0],output,list(samples.values())+query_samples)
    for row in train:
        row['public_input_tokens']=tokenized[row['public_content_sha256']]['input_tokens']
        row['matched']=row['public_input_tokens']==row['actual_input_tokens']
    if not all(r['matched'] for r in train):raise ValueError('Public tokenizer length differs from actual training input tokens')
    shape_templates={}
    for length,evidence in template_evidence.items():
        signatures={e['sequence_sha256'] for e in evidence}
        shape_templates[str(length)]=dict(input_tokens=length,
            status='EXACT_OBSERVED_TRAINING_SHAPE_SIGNATURE_ORDER' if len(signatures)==1 else 'UNKNOWN_DIFFERENT_TRAINING_CAPTURE_SEQUENCE',
            observed_sequence_variant_count=len(signatures),observed_kernel_count=len(evidence[0]['sequence']),
            ordered_launch_signatures=evidence[0]['sequence'] if len(signatures)==1 else None,
            evidence=[{k:v for k,v in e.items() if k!='sequence'} for e in evidence],
            unseen_input_transfer_validated=False,host_gap_prediction_known=False)
    # Verify all metadata prefill shape projections, not just two example rows.
    shape_checks=0
    for module,records in module_records.items():
        if module not in rules:continue
        for record,sem in records:
            f=project_features(rules[module],sem['input_sequence_length'])
            if (f['input_tensor_bytes'],f['output_tensor_bytes'],f['flops'])!=(record['input']['logical_bytes'],record['output']['logical_bytes'],record['dense_gemm_flops']):
                raise ValueError('Public affine rule does not reproduce actual training metadata')
            shape_checks+=1
    bindings=[]
    for c in intake['coverage']:
        if c['alias'] in ('text#0','text#1'):bindings.append(checked(c['bindings']))
    prefill_configs={}
    observed_sequences=defaultdict(set)
    for binding in bindings:
        for launch in binding['launches']:
            config=binding['configurations'][launch['config_id']]
            if not config['semantic_record_id'].startswith('prefill_'):continue
            module=config['explicit_execution']['module']
            identity=(config['kernel_id'],config['precision'],config['execution_key'])
            prefill_configs[identity]=module
            observed_sequences[module].add(config['explicit_execution']['operator_local_kernel_ordinal'])
    group_lookup=defaultdict(list)
    for g in support['groups']:
        group=g['group'];identity=(group['kernel_id'],group['precision'],group['execution_key'])
        if identity in prefill_configs:group_lookup[identity].append(g)
    queries_out=[]
    for q in query_items:
        queries_out.append(query_support(q['sample'],q['pool_id'],samples,train,tokenized,rules,prefill_configs,group_lookup,shape_templates))
    training_queries=[]
    for sha,sample in samples.items():
        row=next(r for r in train if r['public_content_sha256']==sha)
        training_queries.append(query_support(sample,row['pool_id'],samples,train,tokenized,rules,prefill_configs,group_lookup,shape_templates))
    result=dict(schema='gsdeploy.chatqa_public_shape_audit.v3',status='PUBLIC_TOKEN_AND_TRAINING_AFFINE_SHAPE_MATCH',
        bridges=sources,intake=receipt(intake_path),support=receipt(support_path),tokenizer=token_sources,
        training_token_checks=train,training_token_check_count=len(train),unique_public_training_inputs=len(samples),
        shape_rules=rules,shape_rule_count=len(rules),training_prefill_shape_check_count=shape_checks,rejected_rules=rejected,
        observed_prefill_templates=shape_templates,
        dispatch_thresholds=dict(status='UNKNOWN_BETWEEN_OBSERVED_SHAPES',
            observed_lengths=sorted(template_evidence),
            inference='Exact geometry/resources/order retained at measured lengths; no inferred dispatch boundary or interpolated grid.'),
        training_support=training_queries,queries=queries_out,
        full_request_forecast_known=False,generated_output_length_known=False,
        gap_template_transfer_validated=False,natural_timing_validated=False,deployment_qualified=False,
        GPU_workload_executed=False,model_weights_loaded=False,heldout_timing_consumed=False,
        implementation_sources=[receipt(__file__)],
        interpretation='Public tokenizer and prefill affine covariates reproduce training; unknown shape dispatch, non-affine templates, decode count and natural gaps remain explicit.')
    save(output/'report.json',result)
    return result


def query_support(sample,pool_id,training_samples,training_rows,tokenized,rules,prefill_configs,group_lookup,shape_templates):
    sha=public_content_sha(sample);tokens=tokenized[sha]['input_tokens'];same_content=sha in training_samples
    same_pool=pool_id in {r['pool_id'] for r in training_rows}
    same_work=any(r['pool_id']==pool_id and r['public_content_sha256']==sha for r in training_rows)
    template=shape_templates.get(str(tokens))
    template_candidate=bool(template and template['status']=='EXACT_OBSERVED_TRAINING_SHAPE_SIGNATURE_ORDER')
    records=[]
    for identity,module in prefill_configs.items():
        if module not in rules:continue
        features=project_features(rules[module],tokens)
        groups=[]
        for group in group_lookup[identity]:
            hull=hull_support(features,group['unique_feature_vectors'])
            groups.append(dict(execution_context_sha256=group['group']['execution_context_sha256'],
                feature_hull_support=hull,duration_model_known=group['duration_model_known'],
                duration_unknown_reasons=group['duration_unknown_reasons']))
        records.append(dict(module=module,kernel_id=identity[0],execution_key=identity[2],features=features,
            matched_training_groups=groups,feature_support_known=any(g['feature_hull_support']['known'] for g in groups)))
    return dict(pool_id=pool_id,public_content_sha256=sha,input_tokens=tokens,
        content_overlaps_training=same_content,pool_id_overlaps_training=same_pool,
        independent_heldout_input=not same_content and not same_pool,
        prefill_kernel_configuration_count=len(records),feature_supported_configuration_count=sum(r['feature_support_known'] for r in records),
        configurations=records,
        exact_training_workload_template_available=same_work and template_candidate,
        exact_observed_prefill_shape_template_available=template_candidate,
        unseen_input_kernel_sequence_supported=False,unseen_input_launch_grid_supported=False,
        gaps_supported=False,decode_forward_count_known=False,full_request_forecast_known=False,
        reason='Only observed exact training work has a replay template; new content retains UNKNOWN sequence/grid/gaps/decode even if affine features lie in hull.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bridge',action='append',required=True)
    p.add_argument('--intake',required=True);p.add_argument('--support',required=True);p.add_argument('--output',required=True);p.add_argument('--queries')
    args=p.parse_args();r=audit(args.bridge,args.intake,args.support,args.output,args.queries)
    print(json.dumps({k:r[k] for k in ('status','training_token_check_count','unique_public_training_inputs','shape_rule_count','training_prefill_shape_check_count')}))


if __name__=='__main__':main()
