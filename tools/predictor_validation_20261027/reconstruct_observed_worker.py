"""Separate ExLlama v1 reconstruction runtime, not a change to old workers.

Set threshold one AFTER model loading (AutoGPTQ initialization sets eight).
All positive row counts then take ExLlama's dequantize/cuBLAS branch. The
quantized weights, generation policy and float16 activations remain intact.
New accuracy, memory, output repeatability and episode evidence are required.
"""
import os
import sys


def configure_model(adapter,kernel_api=None):
    if adapter.executor.family!='minicpm45':
        return None
    layers=[layer for layer in adapter.model.modules() if getattr(layer,'QUANT_TYPE',None)=='exllama']
    if not layers or any(layer.bits!=4 or layer.group_size!=128 or getattr(layer,'_use_act_order',None) is not False
                         for layer in layers):
        raise ValueError('Pinned four-bit, group128, no-act-order ExLlama v1 layers required')
    buffers=getattr(adapter.model,'device_to_buffers',None)
    required=max(layer.infeatures*layer.outfeatures for layer in layers)
    if not buffers or any(buffer.get('max_dq_buffer_size',0)<required for buffer in buffers.values()):
        raise ValueError('Existing ExLlama weight-reconstruction scratch must cover every layer')
    if kernel_api is None:
        import exllama_kernels as kernel_api
    kernel_api.set_tuning_params(1,False,False)
    evidence=dict(schema='gsdeploy.exllama_reconstruct_runtime.v1',matmul_recons_thd=1,
        matmul_fused_remap=False,matmul_no_half2=False,configured_after_model_load=True,
        quantized_layers=len(layers),existing_maximum_reconstruction_elements=required,
        quantized_weights_changed=False,output_equivalence_to_old_runtime_claimed=False,
        determinism_qualified=False)
    adapter.executor.memory_audit['exllama_reconstruct_runtime']=evidence
    return evidence


def main():
    source=sys.argv[sys.argv.index('--source-root')+1]
    sys.path.insert(0,source)
    import torch
    from deterministic_observed_worker import configure
    configure(torch)
    from gsdeploy.backends import compact_native
    original=compact_native.ProfileAdapter
    class ReconstructAdapter(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            configure_model(self)
    compact_native.ProfileAdapter=ReconstructAdapter
    mode=os.environ.get('GS_RECONSTRUCT_OBSERVER')
    if mode=='complete':
        import complete_observed_worker as worker
    elif mode=='semantic':
        # Also retain the separately qualified PEFT base-forward coverage.
        import base_forward_observed_worker as worker
    else:
        raise ValueError('Explicit complete/semantic reconstruction observer required')
    worker.main()


if __name__=='__main__':main()
