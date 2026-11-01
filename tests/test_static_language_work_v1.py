"""CPU-only nominal work and causal availability checks; no torch/device imports."""
import copy
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027.static_language_work_v1 import (
    PROJECTIONS, binding_digest, complementary_structural_candidates, features_from_visible_begin,
    linear_source_declarations, projection_dimensions, safetensors_headers,
    shape_work, training_begin_features, validate_projection_headers)


def config():
    return dict(hidden_size=4,intermediate_size=8,num_attention_heads=2,num_key_value_heads=1,num_hidden_layers=1)


def binding(lora=False):
    value=dict(schema='gsdeploy.source_bound_decoder_projection_binding.v1',
        family='text',activation_dtype='bfloat16',activation_dtype_bytes=2,
        execution=dict(quantization=dict(method='bitsandbytes',bits=8)),
        lora=dict(rank=2) if lora else None,
        projections=[dict(in_features=a,out_features=b) for a,b in projection_dimensions(config()).values()])
    value['binding_sha256']=binding_digest(value)
    return value


def begin():
    return dict(event='start',kind='gpu',semantic_phase='prefill',segment_id='prefill_0000',
        request_id='r',perf_ns=10,received_time_ns=12,collector_dequeued_time_ns=15,
        collector_processing_time_ns=20,input_batch_size=2,input_sequence_length=3)


def text_headers(cfg):
    headers={}
    for layer in range(cfg['num_hidden_layers']):
        for name,(a,b) in projection_dimensions(cfg).items():
            location='self_attn.' if name in PROJECTIONS[:4] else 'mlp.'
            headers[f'model.layers.{layer}.{location}{name}.weight']=dict(shape=[b,a],dtype='F16')
    return headers


class StaticLanguageWorkTests(unittest.TestCase):
    def test_modified_derived_topology_rejected(self):
        value=binding();value['projections'][0]['out_features']=8
        with self.assertRaisesRegex(ValueError,'Derived binding fields changed'):
            shape_work(value,1,1,'decode')

    def test_validation_review_rejected_before_any_input_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'validation.json'
            path.write_text(json.dumps(dict(schema='gsdeploy.profile_training_evidence_review.v1',
                status='PASS_COMPLETE_TRAINING_COLLECTION_NOT_PREDICTOR_QUALIFICATION',role=dict(split='validation'))))
            with self.assertRaisesRegex(ValueError,'training-only'):
                training_begin_features(path,{})

    def test_analytic_gqa_shapes_and_dense_arithmetic(self):
        value=shape_work(binding(),2,3,'prefill')
        self.assertEqual(value['base_dense_parameter_elements'],144)
        self.assertEqual(value['base_nominal_dense_gemm_flops'],1728)
        self.assertEqual(value['base_dense_equivalent_parameter_bytes_at_activation_dtype'],288)
        self.assertEqual(value['base_logical_input_output_operand_bytes_sum'],768)
        self.assertAlmostEqual(value['base_dense_equivalent_structural_intensity_flops_per_logical_byte'],1728/1056)
        self.assertEqual(projection_dimensions(config())['k_proj'],(4,2))
        self.assertEqual(projection_dimensions(config())['o_proj'],(4,4))

    def test_lora_separate_two_gemms_at_declared_f32_equivalence(self):
        value=shape_work(binding(True),2,3,'prefill')
        self.assertEqual(value['base_nominal_dense_gemm_flops'],1728)
        lora=value['lora']
        self.assertEqual(lora['nominal_two_gemm_flops'],1536)
        self.assertEqual(lora['parameter_elements'],128)
        self.assertEqual(lora['checkpoint_f32_logical_parameter_bytes'],512)
        self.assertEqual(lora['f32_equivalent_logical_input_output_operand_bytes_sum'],2208)
        self.assertIsNone(lora['actual_execution_dtype'])
        self.assertFalse(lora['merged_into_base'])

    def test_no_attention_or_memory_invention(self):
        event=begin();event['past_key_values_present']=True
        value=features_from_visible_begin(binding(),event,parent_now_ns=20)
        for key in ('attention_kv_length','attention_matmul_flops','attention_kv_cache_bytes',
                    'measured_dram_bytes','additional_gpu_memory_bytes','hardware_cache_hit_rate',
                    'runtime_quantized_weight_storage_bytes','runtime_quantization_scales'):
            self.assertIsNone(value[key])
        self.assertEqual(value['classification'],'unknown')

    def test_actual_begin_available_only_after_parent_processing(self):
        with self.assertRaisesRegex(ValueError,'not yet causally visible'):
            features_from_visible_begin(binding(),begin(),parent_now_ns=19)
        value=features_from_visible_begin(binding(),begin(),parent_now_ns=20)
        self.assertTrue(value['runtime_features_available'])
        self.assertEqual(value['available_at_parent_processing_ns'],20)
        self.assertFalse(shape_work(binding(),2,3,'prefill')['runtime_features_available'])

    def test_reject_end_future_and_invalid_chronology(self):
        for field,value in [('event','end'),('collector_processing_time_ns',11),('segment_id','decode_0001')]:
            event=begin();event[field]=value
            with self.assertRaises(ValueError):
                features_from_visible_begin(binding(),event,parent_now_ns=25)

    def test_unknown_or_boolean_shape_never_filled_with_one(self):
        for bad in (None,True,0,-1,1.5):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                shape_work(binding(),1,bad,'decode')

    def test_distinct_head_dimension_not_assumed_hidden_div_heads(self):
        cfg=config();cfg['head_dim']=3
        dims=projection_dimensions(cfg)
        self.assertEqual(dims['q_proj'],(4,6));self.assertEqual(dims['k_proj'],(4,3));self.assertEqual(dims['o_proj'],(6,4))

    def test_header_requires_every_projection_every_layer(self):
        cfg=config();cfg['num_hidden_layers']=2;headers=text_headers(cfg)
        self.assertEqual(len(validate_projection_headers(headers,cfg,'text')),14)
        del headers['model.layers.1.mlp.up_proj.weight']
        with self.assertRaises(ValueError):validate_projection_headers(headers,cfg,'text')

    def test_gptq_packed_header_uses_input_packing_not_dense_transpose(self):
        cfg=dict(hidden_size=128,intermediate_size=256,num_attention_heads=4,num_key_value_heads=1,num_hidden_layers=1)
        headers={}
        for name,(a,b) in projection_dimensions(cfg).items():
            location='self_attn.' if name in PROJECTIONS[:4] else 'mlp.'
            stem=f'llm.model.layers.0.{location}{name}'
            for suffix,shape,dtype in [('qweight',[a//8,b],'I32'),('qzeros',[a//128,b//8],'I32'),('scales',[a//128,b],'F16'),('g_idx',[a],'I32')]:
                headers[stem+'.'+suffix]=dict(shape=shape,dtype=dtype)
        self.assertEqual(len(validate_projection_headers(headers,cfg,'generalist_2')),7)
        headers['llm.model.layers.0.self_attn.k_proj.qweight']['shape']=[32,16]
        with self.assertRaises(ValueError):validate_projection_headers(headers,cfg,'generalist_2')

    def test_lora_missing_rank_or_dtype_rejected(self):
        cfg=config();headers=text_headers(cfg);adapters={}
        for name,(a,b) in projection_dimensions(cfg).items():
            location='self_attn.' if name in PROJECTIONS[:4] else 'mlp.'
            stem=f'base_model.model.model.layers.0.{location}{name}'
            adapters[stem+'.lora_A.weight']=dict(shape=[2,a],dtype='F32')
            adapters[stem+'.lora_B.weight']=dict(shape=[b,2],dtype='F32')
        self.assertEqual(len(validate_projection_headers(headers,cfg,'text',dict(r=2),adapters)),7)
        adapters['base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight']['dtype']='F16'
        with self.assertRaises(ValueError):validate_projection_headers(headers,cfg,'text',dict(r=2),adapters)

    def test_contrast_priority_preserves_all_candidates_including_unknown(self):
        small=shape_work(binding(),1,1,'decode');large=shape_work(binding(),1,100,'prefill')
        candidates=[dict(id='a',features=small),dict(id='b',features=large),dict(id='ocr',features=None)]
        pairs=complementary_structural_candidates(candidates)
        self.assertEqual(len(pairs),3);self.assertEqual(pairs[0]['candidates'],['a','b'])
        self.assertTrue(all(not row['pruning_allowed'] for row in pairs))
        self.assertTrue(all(row['classifications']==['unknown','unknown'] for row in pairs))
        self.assertEqual(sum(row['relative_structural_intensity_log_contrast'] is None for row in pairs),2)

    def test_source_rejects_missing_or_wrong_dimension_declarations(self):
        source='class LlamaAttention:\n def __init__(self):\n  self.q_proj = nn.Linear(1, 2)\nclass LlamaMLP:\n pass\n'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.py';path.write_text(source)
            with self.assertRaisesRegex(ValueError,'dimension expression mismatch'):
                linear_source_declarations(path,'Llama')

    def test_safetensors_header_only_and_payload_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.safetensors'
            header=json.dumps(dict(w=dict(dtype='F16',shape=[2,3],data_offsets=[0,12]))).encode()
            path.write_bytes(struct.pack('<Q',len(header))+header+b'not-inspected')
            real_open=Path.open;reads=[]
            class Spy(io.BufferedReader):
                def read(self,n=-1):reads.append(n);return super().read(n)
            def open_spy(p,mode='r',*args,**kwargs):
                if mode=='rb':return Spy(io.FileIO(p,mode))
                return real_open(p,mode,*args,**kwargs)
            with patch.object(Path,'open',open_spy):tensors,files=safetensors_headers(tmp)
            self.assertEqual(reads,[8,len(header)])
            self.assertEqual(tensors['w']['serialized_logical_bytes'],12)
            self.assertFalse(files[0]['entire_weight_file_rehashed'])
            path.write_bytes(struct.pack('<Q',len(header))+header+b'x')
            with self.assertRaises(ValueError):safetensors_headers(tmp)

    def test_safetensors_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.safetensors'
            header=json.dumps({name:dict(dtype='F16',shape=[2],data_offsets=[0,4]) for name in ('a','b')}).encode()
            path.write_bytes(struct.pack('<Q',len(header))+header+b'abcd')
            with self.assertRaisesRegex(ValueError,'Overlapping'):safetensors_headers(tmp)


if __name__=='__main__':unittest.main()
