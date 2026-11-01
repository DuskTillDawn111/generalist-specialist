"""CPU-only observed template patterns and testable public-shape grid models.

Separates source launch arithmetic, observed workload binding, and unverified
dispatch. Produces numeric candidate sequences for new M without claiming
those candidates are measured, validated, natural timing, or deployment proof.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import copy
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
import re
import shutil

from .kernel_baseline_intake_v3 import checked, _verify
from .kernel_domain_bridge_review_v3 import request_kernels
from .native_collect import receipt,save
from .profiling import digest


SITE=Path('/root/autodl-tmp/gsdeploy-assets/envs/compact-standard/lib/python3.10/site-packages')


def ceildiv(a,b):
    return (a+b-1)//b


def evaluate_axis(rule,m):
    if type(m) is not int or m<1:raise ValueError('Positive integer public input tokens required')
    if rule['kind']=='constant':return rule['value']
    if rule['kind']=='linear_integer':return m*rule['coefficient']
    if rule['kind']=='ceil_ratio':return rule['multiple']*ceildiv(m*rule['numerator'],rule['denominator'])
    raise ValueError('Unsupported grid rule')


@lru_cache(maxsize=None)
def axis_candidates(m0,g0,m1,g1):
    if g0==g1:return (dict(kind='constant',value=g0),)
    candidates={}
    for numerator in range(1,65):
        for denominator in (1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192):
            fraction=Fraction(numerator,denominator)
            for multiple in (1,2,4,8,16,32,64,128):
                if fraction.denominator==1:
                    rule=dict(kind='linear_integer',coefficient=fraction.numerator*multiple)
                else:
                    rule=dict(kind='ceil_ratio',numerator=fraction.numerator,denominator=fraction.denominator,multiple=multiple)
                if evaluate_axis(rule,m0)==g0 and evaluate_axis(rule,m1)==g1:
                    candidates[json.dumps(rule,sort_keys=True)]=rule
    return tuple(sorted(candidates.values(),key=lambda r:(r['kind']!='linear_integer',
        r.get('numerator',1)!=1,r.get('multiple',1)!=1,r.get('denominator',1),r.get('coefficient',1))))


def elementwise_work(name,block):
    legacy=re.search(r'::elementwise_kernel<\(int\)(\d+), \(int\)(\d+),',name)
    if legacy:
        if block!=[int(legacy[1]),1,1]:raise ValueError('Observed block differs from source kernel nt')
        return int(legacy[1])*int(legacy[2])
    if 'vectorized_elementwise_kernel<(int)4,' in name or 'unrolled_elementwise_kernel<' in name:
        if block!=[128,1,1]:raise ValueError('Unsupported elementwise launch block')
        # Actual observed specializations use 4 values/thread. Header io-size
        # dispatch remains a condition, not an assertion about every kernel.
        return 512
    return None


def infer_integer_width(m0,g0,m1,g1,work):
    lower=max((g0-1)*work//m0+1,(g1-1)*work//m1+1)
    upper=min(g0*work//m0,g1*work//m1)
    return list(range(max(1,lower),upper+1)) if upper-lower<128 else []


def cuda_sources(output,upstream_packet):
    root=output/'source-snapshot';root.mkdir()
    paths=[SITE/'torch/include/ATen/native/cuda'/name for name in
           ('CUDALoops.cuh','Loops.cuh','thread_constants.h','Reduce.cuh')]
    paths += [SITE/'bitsandbytes/functional.py',SITE/'bitsandbytes-0.45.5.dist-info/METADATA',SITE/'torch/version.py']
    sources=[]
    for index,path in enumerate(paths):
        target=root/(str(index)+'-'+path.name);shutil.copyfile(path,target)
        sources.append(dict(original=receipt(path),frozen=receipt(target),
            scope='Installed wheel source/header; source-to-loaded-binary build equivalence not independently attested'))
    packet=checked(receipt(upstream_packet))
    for index,source in enumerate(packet['sources']):
        if 'source' not in source:continue
        _verify(source['source']);path=Path(source['source']['path']);target=root/('upstream-'+path.name);shutil.copyfile(path,target)
        sources.append(dict(url=source['url'],frozen=receipt(target),scope=source['scope']))
    return sources


def source_citations(sources):
    """Check exact arithmetic snippets and bind their actual frozen line numbers."""
    by_name={Path(s['frozen']['path']).name.split('-',1)[-1]:s['frozen'] for s in sources}
    requirements={
        'torch_elementwise':[
            ('CUDALoops.cuh','return num_threads() * elems_per_thread<io_sizes>();'),
            ('CUDALoops.cuh','return input_size + output_size;'),
            ('CUDALoops.cuh','} else if constexpr (io_sizes < 4) {'),
            ('CUDALoops.cuh','int64_t grid = (N + io_block_work_size<io_size>() - 1) / io_block_work_size<io_size>();'),
            ('CUDALoops.cuh','int64_t grid = (N + block_work_size() - 1) / block_work_size();'),
            ('CUDALoops.cuh','dim3 grid((N + block.x * vt - 1) / (block.x * vt));'),
            ('thread_constants.h','return C10_WARP_SIZE * 4;'),
            ('thread_constants.h','constexpr int thread_work_size() { return 4; }')],
        'torch_bool_reduction':[
            ('Reduce.cuh','static constexpr int input_vec_size = 4;'),
            ('Reduce.cuh','static constexpr int MAX_NUM_THREADS = 512;'),
            ('Reduce.cuh','int dim0_pow2 = dim0 < max_num_threads ? static_cast<int>(last_pow2(dim0)) : max_num_threads;'),
            ('Reduce.cuh','dim0 /= config.input_vec_size;'),
            ('Reduce.cuh','return element_size_bytes * num_threads * output_vec_size;')],
        'bnb_vector_quant':[
            ('functional.py','rows = prod(A.shape[:-1])'),
            ('ops.cu','kInt8VectorQuant<half, 1024, 0><<<rows, 1024, 0, stream>>>(A, out, rowStats, threshold, rows, cols);')],
        'bnb_dequant':[
            ('ops.cu','const int threads = 512;'),
            ('ops.cu','const int num_per_thread = 4;'),
            ('ops.cu','const int n = numRows*numCols;'),
            ('ops.cu','const int num_blocks = (n + num_per_block - 1) / num_per_block;'),
            ('kernels.cu','template __global__ void kdequant_mm_int32_fp16<4, 512>')],
    }
    result={}
    for group,items in requirements.items():
        citations=[]
        for name,needle in items:
            source=by_name[name];_verify(source)
            matches=[(i+1,line) for i,line in enumerate(Path(source['path']).read_text().splitlines()) if needle in line]
            if not matches:raise ValueError('Required CUDA source arithmetic changed: '+needle)
            citations.extend(dict(source=source,line=line,exact_source_line=snippet) for line,snippet in matches)
        result[group]=citations
    return result


def affine_slots(public_report,length):
    template=public_report['observed_prefill_templates'][str(length)];evidence=template['evidence'][0]
    raw=checked(evidence['raw']);parsed=checked(evidence['parsed']);_verify(evidence['sqlite'])
    row=next(r for r in raw['request_records'] if r['request_id']==evidence['request_id'] and r['model']==evidence['alias'])
    sequence=[item for item in request_kernels(parsed,evidence['sqlite']['path'],row) if item['group'].startswith('prefill_')]
    if [i['signature'] for i in sequence]!=template['ordered_launch_signatures']:raise ValueError('Changed template sequence')
    lookup={(item['kernel']['correlation_id'],item['kernel']['nsys_start_ns']):i for i,item in enumerate(sequence)}
    intake=checked(public_report['intake']);coverage=next(c for c in intake['coverage'] if c['alias']==row['model'] and c['request_id']==row['request_id'])
    bindings=checked(coverage['bindings']);result={}
    for launch in bindings['launches']:
        if not launch['semantic_group'].startswith('prefill_'):continue
        actual=launch['launch'];position=lookup[(actual['correlation_id'],actual['nsys_start_ns'])]
        config=bindings['configurations'][launch['config_id']]
        result[position]=public_report['shape_rules'][config['explicit_execution']['module']]
    return result


def fit_pattern(left,right,m0,m1,source_receipts,affine_rules):
    changed=[k for k in left if left[k]!=right[k]]
    static_same=all(left[k]==right[k] for k in left if k!='grid')
    result=dict(changed_fields=changed,anchor_signatures={str(m0):left,str(m1):right},
                measured_new_shape_validation=False)
    if static_same:
        axes=[]
        for g0,g1 in zip(left['grid'],right['grid']):
            candidates=axis_candidates(m0,g0,m1,g1)
            if not candidates:raise ValueError('No integer/ceil basis matches an observed stable grid axis')
            axes.append(dict(selected=candidates[0],candidate_count=len(candidates),candidates=list(candidates)))
        result.update(kind='SAME_KERNEL_STATIC_GRID_MODEL',grid_axes=axes,
            formula_evidence='TWO_SHAPE_INTEGER_CEIL_HYPOTHESIS',
            source_formula_verified=False,shape_binding_verified=False)
        work=elementwise_work(left['name'],left['block'])
        if work is not None and left['grid'][1:]==[1,1] and right['grid'][1:]==[1,1]:
            widths=infer_integer_width(m0,left['grid'][0],m1,right['grid'][0],work)
            if len(widths)==1:
                width=widths[0]
                rule=dict(kind='ceil_ratio',numerator=width,denominator=work,multiple=1)
                axes[0]['selected']=rule
                result.update(formula_evidence='INSTALLED_TORCH_HEADER_GRID_EQUALS_CEIL_NUMEL_OVER_BLOCK_WORK',
                    source_formula_verified=True,source_reference='CUDALoops.cuh launch_vectorized/launch_unrolled/launch_legacy',
                    assumed_elements_per_public_token=width,block_work_elements=work,
                    shape_binding_verified=bool(affine_rules and all(width in (r['in_features'],r['out_features']) for r in affine_rules)),
                    source_citation_key='torch_elementwise',
                    condition='Same kernel specialization and contiguous/alignment/dispatch path; vectorized io_size>=4 or explicit unroll4; N=M*width derived from observed shape relation, not solely the kernel name')
        elif 'kInt8VectorQuant<' in left['name']:
            if left['grid']!=[m0,1,1] or right['grid']!=[m1,1,1]:raise ValueError('Quantization row launch mismatch')
            result.update(formula_evidence='BNB_0_45_5_UPSTREAM_VECTOR_QUANT_GRID_EQUALS_ROWS',
                source_formula_verified=True,source_reference='ops.cu int8VectorQuant',
                source_citation_key='bnb_vector_quant',
                shape_binding_verified=bool(affine_rules),
                condition='Installed Python passes rows=prod(input.shape[:-1]); upstream tag launch matches rows; exact compiled binary/source equivalence unproved')
        elif 'kdequant_mm_int32_fp16<' in left['name'] and affine_rules:
            widths={r['out_features'] for r in affine_rules}
            if len(widths)==1:
                width=next(iter(widths));rule=dict(kind='ceil_ratio',numerator=width,denominator=2048,multiple=1)
                if evaluate_axis(rule,m0)==left['grid'][0] and evaluate_axis(rule,m1)==right['grid'][0]:
                    axes[0]['selected']=rule
                    result.update(formula_evidence='BNB_0_45_5_UPSTREAM_DEQUANT_512_THREADS_4_ITEMS',source_formula_verified=True,
                        source_reference='ops.cu dequant_mm_int32_fp16; kernels.cu kdequant_mm_int32_fp16',
                        source_citation_key='bnb_dequant',
                        shape_binding_verified=True,assumed_elements_per_public_token=width,block_work_elements=2048,
                        condition='Out tensor has M*out_features elements; upstream tag launch matches observed specialization, binary/source equivalence unproved')
        result['required_source_receipts']=source_receipts
    elif left['name']==right['name'] and 'ReduceOp<bool,' in left['name']:
        for m,signature in ((m0,left),(m1,right)):
            width=min(512,2**((max(1,m//4)).bit_length()-1))
            if signature['block']!=[width,1,1] or signature['dynamic_shared_bytes']!=width:
                raise ValueError('Observed bool reduction does not follow pinned vector4/block/shared relation')
        result.update(kind='REDUCTION_BLOCK_RESOURCE_MODEL',
            formula_evidence='INSTALLED_REDUCE_HEADER_FLOOR_POWER_OF_TWO_AFTER_INPUT_VECTOR4',
            source_formula_verified=True,
            source_citation_key='torch_bool_reduction',
            condition='Fast contiguous bool reduction, input vector width4, one output; driver sharedMemoryExecuted remains a dispatch hypothesis',
            required_source_receipts=source_receipts)
    else:
        for m,signature in ((m0,left),(m1,right)):
            if variant_grid(signature,m,m)[0]!=signature['grid']:
                raise ValueError('Endpoint-domain tile hypothesis does not reproduce its own observed grid')
        result.update(kind='SEPARATE_DISPATCH_DOMAINS',formula_evidence='OBSERVED_ENDPOINT_ALGORITHMS_ONLY',
            source_formula_verified=False,dispatch_threshold_known=False,
            condition='cuBLAS/cuBLASLt algorithm and split-K heuristics are not reconstructed from kernel names; explicit endpoint-domain hypotheses only')
    return result


def variant_grid(signature,m,anchor):
    name=signature['name'];grid=list(signature['grid'])
    # Numeric hypotheses are kept separate per algorithm; no name, register,
    # shared-memory or block interpolation occurs across dispatch variants.
    cutlass=re.search(r'gemm_(?:s8_)?(\d+)x(\d+)_',name)
    if cutlass and 'cutlass::Kernel2<' in name:
        tile_m=int(cutlass[2]);grid[0]=8*ceildiv(m,tile_m)
        return grid,'PACKED_8_CUTLASS_TILE_HYPOTHESIS'
    ampere=re.search(r'ampere_sgemm_(\d+)x(\d+)',name)
    if ampere:
        grid[1]=ceildiv(m,int(ampere[2]))
        return grid,'AMPERE_TILE_Y_AND_ANCHOR_SPLIT_K_HYPOTHESIS'
    return grid,'ANCHOR_GEOMETRY_HYPOTHESIS_NOT_A_SHAPE_FORMULA'


def instantiate_pattern(pattern,m,policy='nearest_anchor'):
    if type(m) is not int or m<1:raise ValueError('Positive integer public input tokens required')
    if policy not in ('lower_domain','upper_domain','nearest_anchor'):raise ValueError('Unknown explicit dispatch hypothesis')
    anchors=sorted(map(int,pattern['anchor_signatures']));m0,m1=anchors
    if not m0<=m<=m1:raise ValueError('Template candidates restricted to observed public-length bracket')
    if m in anchors:return copy.deepcopy(pattern['anchor_signatures'][str(m)]),'EXACT_OBSERVED_ANCHOR'
    if pattern['kind']=='SAME_KERNEL_STATIC_GRID_MODEL':
        out=copy.deepcopy(pattern['anchor_signatures'][str(m0)])
        out['grid']=[evaluate_axis(axis['selected'],m) for axis in pattern['grid_axes']]
        return out,'SOURCE_CONDITIONAL_GRID_CANDIDATE' if pattern['source_formula_verified'] else 'GRID_HYPOTHESIS_CANDIDATE'
    anchor=m0 if policy=='lower_domain' or policy=='nearest_anchor' and m-m0<=m1-m else m1
    out=copy.deepcopy(pattern['anchor_signatures'][str(anchor)])
    if pattern['kind']=='REDUCTION_BLOCK_RESOURCE_MODEL':
        width=min(512,2**((max(1,m//4)).bit_length()-1))
        out['block']=[width,1,1];out['dynamic_shared_bytes']=width
        return out,'SOURCE_CONDITIONAL_REDUCTION_DRIVER_CARVEOUT_HYPOTHESIS'
    out['grid'],status=variant_grid(out,m,anchor)
    return out,status


def instantiate_model(report,m,policy='nearest_anchor'):
    """Return a complete ordered numeric candidate, never substitute timings."""
    sequence=[];counts=Counter()
    for index in report['slot_pattern_indices']:
        value,status=instantiate_pattern(report['patterns'][index],m,policy)
        sequence.append(value);counts[status]+=1
    return dict(schema='gsdeploy.kernel_template_shape_candidate.v3',input_tokens=m,
        evidence_kind='exact_observed_training_anchor' if m in report['training_lengths'] else 'unmeasured_shape_hypothesis',
        dispatch_policy=policy,ordered_launch_signatures=sequence,status_counts=dict(counts),
        source_formula_is_conditional=True,source_model_training_lengths=report['training_lengths'],
        new_shape_validated=False,natural_timing_validated=False,deployment_qualified=False)


def build(public_report_path,upstream_packet,output,query_lengths,validation_reports=(),training_selection=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    public_source=receipt(public_report_path);public=checked(public_source)
    for source in public['implementation_sources']:_verify(source)
    templates=public['observed_prefill_templates'];lengths=sorted(map(int,templates))
    if len(lengths)!=2:raise ValueError('Explicit two-anchor initial model required')
    m0,m1=lengths;a,b=[templates[str(m)]['ordered_launch_signatures'] for m in lengths]
    if len(a)!=len(b):raise ValueError('Different template lengths require a different sequence model')
    sources=cuda_sources(output,upstream_packet);source_receipts=[s['frozen'] for s in sources]
    citations=source_citations(sources)
    affine0,affine1=affine_slots(public,m0),affine_slots(public,m1)
    buckets={}
    for index,(left,right) in enumerate(zip(a,b)):
        key=digest([left,right])
        if key not in buckets:buckets[key]=dict(left=left,right=right,positions=[],affine_rules=[])
        bucket=buckets[key];bucket['positions'].append(index)
        for rules in (affine0,affine1):
            if index in rules:bucket['affine_rules'].append(rules[index])
    patterns=[];slots={}
    for index,bucket in enumerate(buckets.values()):
        pattern=fit_pattern(bucket['left'],bucket['right'],m0,m1,source_receipts,bucket['affine_rules'])
        pattern.update(id='pattern_'+str(index),member_positions=bucket['positions'],member_count=len(bucket['positions']))
        for position in bucket['positions']:slots[position]=index
        # Mandatory exact endpoint reproduction, including changed algorithm resources.
        for m,expected in ((m0,bucket['left']),(m1,bucket['right'])):
            if instantiate_pattern(pattern,m)[0]!=expected:raise ValueError('Model does not reproduce its real anchor')
        patterns.append(pattern)
    instances=[]
    for m in query_lengths:
        for policy in ('nearest_anchor','lower_domain','upper_domain'):
            path=output/f'candidate-M{m}-{policy}.json'
            save(path,instantiate_model(dict(slot_pattern_indices=[slots[p] for p in range(len(a))],
                patterns=patterns,training_lengths=lengths),m,policy))
            instances.append(receipt(path))
    validation=[]
    training_raw={e['raw']['sha256'] for m in templates.values() for e in m['evidence']}
    for path in validation_reports:
        source=receipt(path);doc=checked(source)
        for implementation in doc['implementation_sources']:_verify(implementation)
        if checked(doc['intake'])['execution_context_sha256']!=checked(public['intake'])['execution_context_sha256']:
            raise ValueError('Independent template check must retain exact canonical execution context')
        for key,template in doc['observed_prefill_templates'].items():
            m=int(key);raws={e['raw']['sha256'] for e in template['evidence']}
            for evidence in template['evidence']:
                for name in ('raw','parsed','sqlite'):_verify(evidence[name])
            if raws&training_raw:raise ValueError('Independent template validation must use different raw captures')
            if not m0<=m<=m1:continue
            for policy in ('nearest_anchor','lower_domain','upper_domain'):
                predicted=[instantiate_pattern(patterns[slots[p]],m,policy)[0] for p in range(len(a))]
                actual=template['ordered_launch_signatures'];matches=[x==y for x,y in zip(predicted,actual)]
                validation.append(dict(source=source,input_tokens=m,policy=policy,
                    unseen_training_shape=m not in lengths,actual_kernel_count=len(actual),predicted_kernel_count=len(predicted),
                    matched_position_count=sum(matches),all_positions_match=len(actual)==len(predicted) and all(matches),timing_used=False))
    counts=Counter()
    for pattern in patterns:counts[pattern['kind']]+=pattern['member_count']
    selected=None
    if training_selection:
        selection_source=receipt(training_selection);selection=checked(selection_source)
        _verify(selection['selected_input'])
        selected=dict(source=selection_source,selected=selection['selected'],
            purpose='ADDITIONAL_TRAINING_NOT_HELDOUT_VALIDATION')
    first_odd=selected['selected']['input_tokens'] if selected else 1025
    ambiguous_slots=sum(p['member_count'] for p in patterns if p['kind']=='SAME_KERNEL_STATIC_GRID_MODEL' and
        any(len({evaluate_axis(r,first_odd) for r in axis['candidates']})>1 for axis in p['grid_axes']))
    report=dict(schema='gsdeploy.kernel_template_grid_model.v3',public_shape_source=public_source,
        source_packet=receipt(upstream_packet),cuda_sources=sources,source_formula_citations=citations,
        source_arithmetic_scope='Verified source snippets, conditional workload bindings; compiled binary/source equivalence unproved',
        training_lengths=lengths,kernel_count=len(a),
        distinct_pattern_count=len(patterns),slot_kind_counts=dict(counts),patterns=patterns,
        slot_pattern_indices=[slots[p] for p in range(len(a))],candidate_instances=instances,independent_validation=validation,
        new_shape_validation_count=len({(v['source']['sha256'],v['input_tokens']) for v in validation if v['unseen_training_shape']}),
        legal_support=dict(public_token_bracket=lengths,batch_size=1,
            context=checked(public['intake'])['execution_context_sha256'],
            exact_measured_template_lengths=lengths,new_length_candidates_are_conditional=True,
            ordered_topology_transfer_known=False,
            outside_bracket_rejected=True,full_request_decode_and_gap_model_known=False),
        validation_plan=dict(proposed_training_public_token_lengths=[415,512,513,1024,1025,2048,2049,4096,4513],
            first_minimal_discriminating_shape=first_odd,additional_real_training_selection=selected,
            odd_shape_discriminates_two_even_anchor_grid_bases=True,
            ambiguous_slots_at_first_odd_shape=ambiguous_slots,
            entire_dispatch_domain_identifiable_from_one_extra_shape=False,
            motivation='Ceil boundaries, reduction powers, and intermediate algorithm dispatch; authentic public training inputs, never renaming or heldout fitting',
            dispatch_policy_is_hypothesis=True,algorithm_threshold_monotonicity_assumed=False),
        GPU_workload_executed=False,model_weights_loaded=False,new_shape_execution_validated=False,
        natural_timing_validated=False,deployment_qualified=False,
        implementation_sources=[receipt(__file__)],
        interpretation='Numeric, testable template candidates; source launch arithmetic does not itself prove public-shape-to-numel/algorithm dispatch or binary build equivalence.')
    save(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--public-report',required=True);p.add_argument('--upstream-packet',required=True)
    p.add_argument('--output',required=True);p.add_argument('--input-tokens',type=int,action='append',required=True)
    p.add_argument('--validation-report',action='append',default=[])
    p.add_argument('--training-selection')
    args=p.parse_args();r=build(args.public_report,args.upstream_packet,args.output,args.input_tokens,args.validation_report,args.training_selection)
    print(json.dumps({k:r[k] for k in ('kernel_count','distinct_pattern_count','slot_kind_counts','new_shape_validation_count')}))


if __name__=='__main__':main()
