"""Source-bound nominal decoder projection work, without models or device APIs.

Declared dense-equivalent arithmetic and logical operand bytes are structural
features. They do not measure DRAM, GPU allocations, cache hits or SM service.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct

from .native_collect import receipt, save

SCHEMA = 'gsdeploy.static_language_projection_work.v1'
PROJECTIONS = ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
DTYPE_BYTES = {'F16':2,'BF16':2,'F32':4,'F64':8,'I8':1,'U8':1,'I16':2,'I32':4,'I64':8,'BOOL':1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value, name):
    require(type(value) is int and value > 0, name+' requires a positive integer')
    return value


def read(path):
    return json.loads(Path(path).read_text())


def binding_digest(binding):
    return hashlib.sha256(json.dumps({k:v for k,v in binding.items() if k!='binding_sha256'},
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_binding_sources(binding):
    require(binding_digest(binding)==binding.get('binding_sha256'),'Derived binding fields changed')
    for ref in binding['sources']:
        require(receipt(ref['path'])==ref,'Static topology source changed')
    for row in binding['model_header_files']:
        require(file_stat(row['path'])==row['stat'],'Model/header stat binding changed')


def file_stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, device=value.st_dev, inode=value.st_ino,
        mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def safetensors_headers(directory):
    """Read bounded JSON headers only; never read/hash full tensor payloads."""
    tensors, files = {}, []
    for path in sorted(Path(directory).glob('*.safetensors')):
        before = file_stat(path)
        with path.open('rb') as stream:
            length_bytes = stream.read(8)
            require(len(length_bytes)==8, 'Truncated safetensors length')
            length = struct.unpack('<Q', length_bytes)[0]
            require(0 < length <= 32*2**20 and 8+length<=before['size'], 'Bounded safetensors header required')
            data = stream.read(length)
        require(len(data)==length, 'Truncated safetensors header')
        header = json.loads(data)
        intervals=[]
        for name, entry in header.items():
            if name=='__metadata__':
                continue
            require(name not in tensors, 'Duplicate safetensors tensor')
            shape, dtype = entry['shape'], entry['dtype']
            require(isinstance(shape,list) and all(type(x) is int and x>=0 for x in shape), 'Nonnegative integer tensor shape required')
            require(dtype in DTYPE_BYTES, 'Unsupported serialized tensor dtype')
            offsets = entry['data_offsets']
            require(isinstance(offsets,list) and len(offsets)==2 and all(type(x) is int for x in offsets), 'Integer tensor offsets required')
            start,end=offsets
            require(0<=start<=end<=before['size']-8-length and end-start==math.prod(shape)*DTYPE_BYTES[dtype], 'Tensor bounds/shape/dtype byte count mismatch')
            if start!=end:
                intervals.append((start,end))
            tensors[name]=dict(shape=shape,dtype=dtype,serialized_logical_bytes=end-start,file=str(path.resolve()))
        intervals.sort()
        require(all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])), 'Overlapping tensor payload offsets')
        require(file_stat(path)==before, 'Weight file changed during header inspection')
        files.append(dict(path=str(path.resolve()),stat=before,header_bytes=length,
            header_sha256=hashlib.sha256(data).hexdigest(),entire_weight_file_rehashed=False))
    require(bool(tensors), 'Local safetensors headers required')
    return tensors,files


def projection_dimensions(config):
    h=positive(config.get('hidden_size'),'hidden_size')
    m=positive(config.get('intermediate_size'),'intermediate_size')
    heads=positive(config.get('num_attention_heads'),'attention heads')
    kvheads=positive(config.get('num_key_value_heads'),'KV heads')
    head_dim=config.get('head_dim')
    if head_dim is None:
        require(h%heads==0, 'Integral default head dimension required')
        head_dim=h//heads
    positive(head_dim,'head_dim')
    q,kv=heads*head_dim,kvheads*head_dim
    return dict(q_proj=(h,q),k_proj=(h,kv),v_proj=(h,kv),o_proj=(q,h),
        gate_proj=(h,m),up_proj=(h,m),down_proj=(m,h))


def linear_source_declarations(path, architecture):
    """Verify seven nn.Linear constructor declarations in the named classes.

    This validates declared Python topology, not loaded quantized kernel work.
    Config dimensions are independently checked against every layer header.
    """
    tree=ast.parse(Path(path).read_text())
    expected={architecture+'Attention':set(PROJECTIONS[:4]),architecture+'MLP':set(PROJECTIONS[4:])}
    arguments=dict(q_proj=('config.hidden_size','config.num_attention_heads * self.head_dim'),
        k_proj=('config.hidden_size','config.num_key_value_heads * self.head_dim'),
        v_proj=('config.hidden_size','config.num_key_value_heads * self.head_dim'),
        o_proj=('config.num_attention_heads * self.head_dim','config.hidden_size'),
        gate_proj=('self.hidden_size','self.intermediate_size'),
        up_proj=('self.hidden_size','self.intermediate_size'),
        down_proj=('self.intermediate_size','self.hidden_size'))
    declarations={}
    for node in tree.body:
        if not isinstance(node,ast.ClassDef) or node.name not in expected:
            continue
        fields={}
        for item in ast.walk(node):
            if not isinstance(item,ast.Assign) or not isinstance(item.value,ast.Call):
                continue
            call=item.value
            if ast.unparse(call.func)!='nn.Linear':
                continue
            for target in item.targets:
                name=ast.unparse(target)
                if name.startswith('self.') and name[5:] in expected[node.name]:
                    require(name[5:] not in fields,'Duplicate linear declaration')
                    require(tuple(ast.unparse(arg) for arg in call.args)==arguments[name[5:]],'Projection constructor dimension expression mismatch')
                    fields[name[5:]]=dict(line=item.lineno,constructor=ast.unparse(call))
        require(set(fields)==expected[node.name], 'All named decoder projection declarations required')
        declarations[node.name]=fields
    require(set(declarations)==set(expected),'Actual installed decoder classes required')
    return declarations


def validate_projection_headers(tensors, config, family, adapter=None, adapter_tensors=None):
    dims=projection_dimensions(config)
    layers=positive(config.get('num_hidden_layers'),'decoder layers')
    prefix='model.layers.' if family=='text' else 'llm.model.layers.'
    result=[]
    for layer in range(layers):
        for name,(incoming,outgoing) in dims.items():
            location='self_attn.' if name in PROJECTIONS[:4] else 'mlp.'
            stem=f'{prefix}{layer}.{location}{name}'
            if family=='text':
                entry=tensors.get(stem+'.weight',{})
                require(entry.get('shape')==[outgoing,incoming] and entry.get('dtype')=='F16','Dense checkpoint projection does not match config')
                names=[stem+'.weight']
            else:
                require(incoming%128==0 and outgoing%32==0,'Pinned GPTQ grouping/alignment required')
                expected={'qweight':([incoming//8,outgoing],'I32'),
                    'qzeros':([incoming//128,outgoing//8],'I32'),
                    'scales':([incoming//128,outgoing],'F16'),'g_idx':([incoming],'I32')}
                names=[]
                for suffix,(shape,dtype) in expected.items():
                    tensor_name=stem+'.'+suffix
                    entry=tensors.get(tensor_name,{})
                    require(entry.get('shape')==shape and entry.get('dtype')==dtype,'Packed GPTQ header does not match pinned four-bit declaration')
                    names.append(tensor_name)
            lora_names=[]
            if adapter is not None:
                rank=positive(adapter['r'],'LoRA rank')
                astem=f'base_model.model.model.layers.{layer}.{location}{name}'
                for suffix,shape in [('lora_A.weight',[rank,incoming]),('lora_B.weight',[outgoing,rank])]:
                    aname=astem+'.'+suffix
                    entry=adapter_tensors.get(aname,{})
                    require(entry.get('shape')==shape and entry.get('dtype')=='F32','Every LoRA A/B shape and F32 header required')
                    lora_names.append(aname)
            result.append(dict(layer=layer,projection=name,in_features=incoming,out_features=outgoing,
                checkpoint_tensor_names=names,lora_tensor_names=lora_names))
    return result


def build_binding(registry_path, family, *, language_source, quantization_source,
                  loader_source, lora_source=None):
    require(family in ('text','generalist_2'),'Only actual text and G2 decoder declarations supported')
    registry=read(registry_path)
    worker=registry[family]
    spec_path=Path(worker['spec'])
    require(receipt(spec_path)['sha256']==worker['spec_sha256'],'Registry/spec binding drift')
    spec=read(spec_path)
    require(spec['execution']==worker['execution'],'Registry/spec execution mismatch')
    require(spec['candidate']['model']==worker['model'],'Registry/spec model mismatch')
    checkpoint=Path(spec['resources']['checkpoint']['path'])
    require(checkpoint==Path(worker['checkpoint']['path'])==Path(worker['env']['GS_MODEL_PATH']),'Actual checkpoint path mismatch')
    config_path=checkpoint/'config.json'
    config=read(config_path)
    require(config.get('attention_bias') is False and not config.get('mlp_bias',False),'Bias-free decoder projection declaration required')
    execution=spec['execution']
    sources=[receipt(p) for p in (registry_path,spec_path,config_path,language_source,quantization_source,loader_source)]
    tensors,files=safetensors_headers(checkpoint)
    architecture='Llama' if family=='text' else 'Qwen3'
    declarations=linear_source_declarations(language_source,architecture)
    adapter,adapter_tensors=None,None
    if family=='text':
        require(config.get('model_type')=='llama' and execution.get('family')=='chatqa' and execution.get('dtype')=='bfloat16','Actual ChatQA/BF16 declaration required')
        quant=execution['quantization']
        require(quant.get('method')=='bitsandbytes' and quant.get('bits')==8 and quant.get('llm_int8_threshold')==0.,'Pinned INT8 declaration required')
        require('lm_head' in quant.get('skip_modules',[]),'Unquantized LM head identity required')
        manifest_path=Path(execution['peft_adapter']['manifest'])
        require(receipt(manifest_path)['sha256']==execution['peft_adapter']['sha256'],'Pinned LoRA manifest drift')
        manifest=read(manifest_path)
        require(manifest['model']==worker['model'] and manifest['revision']==spec['candidate']['revision'],'LoRA base identity mismatch')
        adapter_path=Path(manifest['path'])/'adapter_config.json'
        adapter=read(adapter_path)
        require(adapter.get('peft_type')=='LORA' and set(adapter.get('target_modules',[]))==set(PROJECTIONS)
            and adapter.get('bias')=='none' and not adapter.get('use_dora') and not adapter.get('use_rslora')
            and not adapter.get('rank_pattern') and not adapter.get('alpha_pattern')
            and not adapter.get('layer_replication') and not adapter.get('layers_to_transform')
            and not adapter.get('fan_in_fan_out') and adapter.get('modules_to_save') is None,'Simple unmerged seven-projection LoRA contract required')
        require(lora_source is not None,'Actual PEFT bnb implementation required')
        sources.extend(receipt(p) for p in (manifest_path,adapter_path,lora_source))
        adapter_tensors,adapter_files=safetensors_headers(manifest['path'])
        require(len(adapter_tensors)==2*7*config['num_hidden_layers'],'Exact complete LoRA header cohort required')
        files.extend(adapter_files)
        activation_dtype='bfloat16'
    else:
        require(config.get('model_type')=='minicpmo' and execution.get('family')=='minicpm45' and execution.get('dtype')=='float16','Actual MiniCPMO/FP16 declaration required')
        quant=execution['quantization']
        require((quant.get('method'),quant.get('bits'),quant.get('group_size'),quant.get('desc_act'),quant.get('kernel'))==('gptq',4,128,False,'exllama_v1'),'Pinned GPTQ declaration required')
        qc=config.get('quantization_config',{})
        require((qc.get('bits'),qc.get('group_size'),qc.get('desc_act'),qc.get('sym'))==(4,128,False,True),'Checkpoint GPTQ configuration mismatch')
        model_source=checkpoint/'modeling_minicpmo.py'
        tree=ast.parse(model_source.read_text())
        require(any(isinstance(n,ast.Call) and ast.unparse(n.func)=='Qwen3ForCausalLM' for n in ast.walk(tree)),'Actual MiniCPMO Qwen3 language construction required')
        sources.extend(receipt(p) for p in (model_source,checkpoint/'quantize_config.json'))
        activation_dtype='float16'
    projections=validate_projection_headers(tensors,config,family,adapter,adapter_tensors)
    for file in files:
        require(file_stat(file['path'])==file['stat'],'Model changed while deriving static declaration')
    result=dict(schema='gsdeploy.source_bound_decoder_projection_binding.v1',
        status='DECLARED_TOPOLOGY_AND_HEADER_VERIFIED_NOT_RUNTIME_AFFINE_MEASUREMENT',
        family=family,model=worker['model'],revision=spec['candidate']['revision'],
        execution=execution,spec=receipt(spec_path),configuration=receipt(config_path),
        activation_dtype=activation_dtype,activation_dtype_bytes=2,decoder_layers=config['num_hidden_layers'],
        projections=projections,source_declarations=declarations,sources=sources,model_header_files=files,
        lora=(dict(rank=adapter['r'],alpha=adapter['lora_alpha'],checkpoint_dtype='F32',checkpoint_dtype_bytes=4,
            execution_dtype_evidence='PEFT casts to lora_A dtype outside autocast; runtime autocast/dtype not directly measured',
            merged=False) if adapter is not None else None),
        runtime_quantized_weight_storage_bytes=None,runtime_quantization_scales=None,
        hardware_matched_precision_peak=None,classifier='unknown',GPU_started=False,
        weight_payloads_read=False,full_weight_files_rehashed=False,
        installed_source_historical_runtime_identity_independently_certified=False)
    result['binding_sha256']=binding_digest(result)
    return result


def shape_work(binding, batch_size, sequence_length, phase):
    require(binding.get('schema')=='gsdeploy.source_bound_decoder_projection_binding.v1','Source-bound decoder binding required')
    require(binding_digest(binding)==binding.get('binding_sha256'),'Derived binding fields changed')
    require(phase in ('prefill','decode'),'Actual semantic phase required')
    rows=positive(batch_size,'batch_size')*positive(sequence_length,'sequence_length')
    activation_bytes=positive(binding['activation_dtype_bytes'],'activation dtype bytes')
    params=sum(p['in_features']*p['out_features'] for p in binding['projections'])
    io=sum(rows*(p['in_features']+p['out_features'])*activation_bytes for p in binding['projections'])
    dense_weights=params*activation_bytes
    lora=None
    if binding['lora'] is not None:
        rank=binding['lora']['rank']
        elements=sum(rank*(p['in_features']+p['out_features']) for p in binding['projections'])
        logical_io=sum(rows*(p['in_features']+p['out_features']+2*rank)*4 for p in binding['projections'])
        lora=dict(projection_families=list(PROJECTIONS),rank=rank,
            nominal_two_gemm_flops=2*rows*elements,parameter_elements=elements,
            checkpoint_f32_logical_parameter_bytes=elements*4,
            f32_equivalent_logical_input_output_operand_bytes_sum=logical_io,
            scale_and_add_flops=None,actual_execution_dtype=None,merged_into_base=False)
    return dict(schema=SCHEMA,binding_sha256=binding['binding_sha256'],family=binding['family'],
        phase=phase,batch_size=batch_size,sequence_length=sequence_length,token_rows=rows,
        decoder_projection_count=len(binding['projections']),
        base_nominal_dense_gemm_flops=2*rows*params,base_dense_parameter_elements=params,
        base_dense_equivalent_parameter_bytes_at_activation_dtype=dense_weights,
        base_logical_input_output_operand_bytes_sum=io,
        base_dense_equivalent_structural_intensity_flops_per_logical_byte=(2*rows*params)/(dense_weights+io),
        lora=lora,attention_kv_length=None,attention_matmul_flops=None,attention_kv_cache_bytes=None,
        measured_dram_bytes=None,additional_gpu_memory_bytes=None,hardware_cache_hit_rate=None,
        runtime_quantized_weight_storage_bytes=None,runtime_quantization_scales=None,
        classification='unknown',classification_reason='MATCHED_ACTUAL_QUANTIZED_PRECISION_PEAK_AND_DRAM_DEMAND_NOT_ESTABLISHED',
        precision_signature=dict(base_weight=binding['execution']['quantization'],
            activation_dtype=binding['activation_dtype'],lora_checkpoint_dtype='F32' if lora else None),
        scope='Nominal seven decoder projection families; excludes attention products, norms/RoPE, activation functions, LM head, embedding, quantization/packing/kernel overhead and LoRA casts/scale/add.',
        logical_bytes_basis='Per-affine operand-size sum and hypothetical dense base weights at activation dtype; repeated/aliased operands included, not unique storage, live peak, DRAM or cache traffic.',
        runtime_features_available=False,pruning_allowed=False,interference_reduction_proven=False)


def features_from_visible_begin(binding, event, *, parent_now_ns):
    """Only a BEGIN already processed by the parent exposes this actual shape."""
    require(type(parent_now_ns) is int and parent_now_ns>=0,'Parent decision clock required')
    require(event.get('event')=='start' and event.get('kind')=='gpu'
        and event.get('semantic_phase') in ('prefill','decode'),'Actual semantic BEGIN required')
    keys=('perf_ns','received_time_ns','collector_dequeued_time_ns','collector_processing_time_ns')
    values=[event.get(k) for k in keys]
    require(all(type(v) is int and v>=0 for v in values) and values==sorted(values)
        and values[-1]<=parent_now_ns,'BEGIN shape not yet causally visible to parent')
    phase=event['semantic_phase']
    require(event.get('segment_id','').startswith(phase+'_'),'BEGIN identity/phase mismatch')
    result=shape_work(binding,event.get('input_batch_size'),event.get('input_sequence_length'),phase)
    result.update(runtime_features_available=True,available_at_parent_processing_ns=values[-1],
        request_id=event.get('request_id'),phase_id=event['segment_id'],
        availability_basis='Actual BEGIN shape after parent processing; no response END/query/elapsed/future generated-token count read.')
    return result


def complementary_structural_candidates(items):
    """Prioritize intensity contrast, retaining every input pair and all unknowns."""
    require(isinstance(items,list) and all(isinstance(x.get('id'),str) and x['id'] for x in items),'Explicit candidate identities required')
    require(len({x['id'] for x in items})==len(items),'Unique candidate identities required')
    pairs=[]
    for i,left in enumerate(items):
        for right in items[i+1:]:
            values=[]
            for item in (left,right):
                feature=item.get('features') or {}
                value=feature.get('base_dense_equivalent_structural_intensity_flops_per_logical_byte')
                if value is not None:
                    require(type(value) in (int,float) and math.isfinite(value) and value>0,'Positive finite structural intensity required')
                values.append(value)
            score=abs(math.log(values[0]/values[1])) if None not in values else None
            pairs.append(dict(candidates=[left['id'],right['id']],relative_structural_intensity_log_contrast=score,
                classifications=['unknown','unknown'],priority_basis='Shape/phase dense-equivalent arithmetic contrast only; no hardware resource type or interference benefit identified.',
                pruning_allowed=False,physical_interference_reduction_proven=False))
    return sorted(pairs,key=lambda row:(row['relative_structural_intensity_log_contrast'] is None,
        -(row['relative_structural_intensity_log_contrast'] or 0),row['candidates']))


def training_begin_features(full_review_path, bindings):
    """Training-only actual BEGIN feature extraction; no outcome/timing labels.

    Returns every measured request, including explicit unsupported-owner rows.
    Priming is excluded. Shape visibility is per forward after BEGIN processing;
    later decode shapes are not exposed to an earlier routing decision.
    """
    review=read(full_review_path)
    require(review.get('schema')=='gsdeploy.profile_training_evidence_review.v1'
        and review.get('status')=='PASS_COMPLETE_TRAINING_COLLECTION_NOT_PREDICTOR_QUALIFICATION'
        and review.get('role',{}).get('split')=='training','Complete training-only collection review required')
    for ref in review['sources']:
        require(receipt(ref['path'])==ref,'Training source/evidence changed')
    for binding in bindings.values():
        validate_binding_sources(binding)
    refs=review['inputs']
    for name in ('plan.json','context.json','raw.json'):
        require(receipt(refs[name]['path'])==refs[name],'Training raw/plan/context changed')
    plan,context,raw=(read(refs[name]['path']) for name in ('plan.json','context.json','raw.json'))
    require(plan.get('split')=='training' and raw.get('plan')==refs['plan.json']
        and raw.get('context')==refs['context.json'],'Exact training raw bindings required')
    rows=[]
    for row in raw['request_records']:
        require(row.get('split')=='training' and row.get('validation_view')=='training','Nontraining request forbidden')
        family=context['residents'][row['model']]
        record=dict(request_id=row['request_id'],profile_job_id=row['profile_job_id'],owner=row['model'],
            family=family,pool_id=row['pool_id'],task=row['task'],features=[],outcome_labels_read=False)
        if family not in bindings:
            record.update(status='UNSUPPORTED_OWNER_STATIC_TOPOLOGY_UNKNOWN')
        else:
            binding=bindings[family]
            require(context['worker_bindings'][family]['spec_sha256']==binding['spec']['sha256']
                and raw['ready'][row['model']]['spec_sha256']==binding['spec']['sha256'],'Actual capture worker spec mismatch')
            for event in row['execution_events']:
                if event.get('event')=='start' and event.get('semantic_phase') in ('prefill','decode'):
                    require(event.get('request_id')==row['request_id'],'Request/BEGIN identity mismatch')
                    record['features'].append(features_from_visible_begin(binding,event,
                        parent_now_ns=event['collector_processing_time_ns']))
            require(bool(record['features']),'Declared language owner lacks actual observed BEGIN shapes')
            record['status']='NOMINAL_FEATURES_AT_EACH_OBSERVED_PARENT_VISIBLE_BEGIN'
        rows.append(record)
    return dict(schema='gsdeploy.training_actual_begin_static_projection_features.v1',
        full_training_review=receipt(full_review_path),timing_domain=plan['mode'],requests=rows,
        measured_requests=len(rows),semantic_forward_features=sum(len(r['features']) for r in rows),
        priming_included=False,outcome_labels_read=False,validation_labels_read=False,
        routing_before_shape_available_qualified=False,fitted_model_qualified=False,GPU_started=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',required=True);p.add_argument('--family',required=True,choices=('text','generalist_2'))
    p.add_argument('--language-source',required=True);p.add_argument('--quantization-source',required=True)
    p.add_argument('--loader-source',required=True);p.add_argument('--lora-source');p.add_argument('--output',required=True)
    args=p.parse_args()
    require(not Path(args.output).exists(),'Exclusive static binding output required')
    result=build_binding(args.registry,args.family,language_source=args.language_source,
        quantization_source=args.quantization_source,loader_source=args.loader_source,lora_source=args.lora_source)
    save(args.output,result)
    print(json.dumps(receipt(args.output)))


if __name__=='__main__':
    main()
