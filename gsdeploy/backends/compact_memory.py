"""Explicit inference memory policies. Importing this module never imports CUDA."""
from contextlib import contextmanager
import math


def minicpm_native_model(model):
    """Resolve the module that owns llm, rather than GPTQ's attribute proxy.

    BaseGPTQForCausalLM forwards reads through __getattr__, but deletion and
    assignment still affect the wrapper. Mutations must target its real model.
    """
    for _ in range(2):
        owned = vars(model)
        if 'llm' in owned or 'llm' in owned.get('_modules', {}):
            return model
        model = getattr(model, 'model', None)
        if model is None:
            break
    raise ValueError('Cannot resolve the native MiniCPM module owning llm')


def actual_generation_scratch_rows(kwargs, input_limit):
    """Rows for the actual batch-one native prefill and its real beam expansion."""
    inputs = kwargs.get('inputs_embeds')
    dimensions = 3
    if inputs is None:
        inputs, dimensions = kwargs.get('input_ids'), 2
    if inputs is None or len(inputs.shape) != dimensions or int(inputs.shape[0]) != 1:
        raise ValueError('Actual scratch sizing requires one complete native generation input')
    length = int(inputs.shape[-2] if dimensions == 3 else inputs.shape[-1])
    if type(input_limit) is not int or input_limit < 1 or not 1 <= length <= input_limit:
        raise ValueError('Complete native prefill exceeds the declared input bound; no truncation')
    beams = kwargs.get('num_beams',1)
    if type(beams) is not int or not 1 <= beams <= 8 or kwargs.get('num_return_sequences',1) != 1:
        raise ValueError('Unexpected native beam expansion for actual scratch sizing')
    if inputs is kwargs.get('input_ids') and kwargs.get('use_cache') is not True:
        raise ValueError('ID-only actual prefill scratch requires explicit cached decoding')
    # Pinned HF always caches inputs_embeds. Prefill has length*beams rows;
    # every subsequent decode has beams rows, so no input/kernel rows are split.
    return length * beams


@contextmanager
def exllama_request_scratch(model, enabled, active_rows, torch, *, audit=None, resize=None):
    """Keep original full-batch kernels; truly release unused idle scratch.

    Resizing and synchronization are inside the request's measured interval.
    This does not change context length, weights or projection batch shapes.
    """
    if not enabled:
        yield
        return
    if type(active_rows) is not int or active_rows < 1:
        raise ValueError('Active scratch rows must be a positive integer')
    if resize is None:
        from auto_gptq.utils.exllama_utils import exllama_set_max_input_length
        resize = exllama_set_max_input_length
    if audit is None:
        audit = {'active_rows': active_rows, 'idle_rows': 1, 'requests': 0, 'releases': 0}
    audit['active_rows'] = active_rows
    audit['peak_active_rows'] = max(audit.get('peak_active_rows', 0), active_rows)
    torch.cuda.synchronize()
    resize(model, active_rows)
    audit['requests'] += 1
    try:
        yield
    finally:
        torch.cuda.synchronize()
        resize(model, 1)
        torch.cuda.synchronize()
        audit['releases'] += 1


@contextmanager
def last_position_logits(owner, enabled):
    """Slice only the language output projection, preserving the full prefill/KV.

    Limited to the study's batch-one greedy inference. Never attach this hook
    during training, loss computation, beam search or audio generation.
    """
    if not enabled:
        yield
        return
    if owner.training:
        raise ValueError('Last-position logits are inference-only')

    def last(module, args):
        if not args or args[0].ndim != 3 or args[0].shape[0] != 1:
            raise ValueError('Expected batch-one language hidden states')
        return (args[0][:, -1:, :], *args[1:])

    handle = owner.lm_head.register_forward_pre_hook(last)
    try:
        yield
    finally:
        handle.remove()


def install_phi_vision_microbatch(vision, batch_size, torch):
    """Pinned Phi get_img_features operates independently on each crop.

    Its selected-layer/compression logic stays inside the original method.
    All crops and masks are retained in their original order.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Vision microbatch must be a positive integer')
    original = vision.get_img_features
    audit = {'microbatch_size': batch_size, 'calls': 0, 'crops': 0, 'maximum_batch_seen': 0}

    def features(images, attention_mask=None):
        if vision.training:
            raise ValueError('Crop microbatching is inference-only')
        count = len(images)
        if not count or (attention_mask is not None and len(attention_mask) != count):
            raise ValueError('Image crop/mask batch mismatch')
        parts = []
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            mask = attention_mask[start:end] if attention_mask is not None else None
            parts.append(original(images[start:end], attention_mask=mask))
            audit['calls'] += 1
            audit['crops'] += end - start
            audit['maximum_batch_seen'] = max(audit['maximum_batch_seen'], end - start)
        return torch.cat(parts, dim=0)

    vision.get_img_features = features
    return audit


def configure_minicpm_language_sdpa(inner):
    """Switch only pinned Qwen2 eager attention to its native SDPA subclass.

    AutoGPTQ's pinned loader does not forward attention options to AutoConfig.
    The native subclasses share the same parameters and initialization; no
    quantized projection is reconstructed or dequantized by this operation.
    The model-level mask path must be switched along with the layer backend.
    """
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2SdpaAttention
    backbone = inner.llm.model
    layers = [layer.self_attn for layer in backbone.layers]
    if not layers or any(type(layer) not in (Qwen2Attention, Qwen2SdpaAttention) for layer in layers):
        raise ValueError('Unsupported pinned MiniCPM language attention class')
    for layer in layers:
        layer.__class__ = Qwen2SdpaAttention
        layer.config._attn_implementation = 'sdpa'
    inner.llm.config._attn_implementation = 'sdpa'
    backbone._attn_implementation = 'sdpa'
    return {'language_attention': 'sdpa', 'language_layers': len(layers),
            'actual_attention_classes': sorted({type(layer).__name__ for layer in layers})}


def install_linear_row_microbatch(layer, row_limit, torch, *, audit=None):
    """Bound independent linear rows without chunking attention or its KV.

    The original quantized projection, scales, zero points, bias and input
    order are retained. This helper is inference-only; it does not implement
    autograd through its output assembly.
    """
    if type(row_limit) is not int or row_limit < 1:
        raise ValueError('Linear row limit must be a positive integer')
    original = layer.forward
    if audit is None:
        audit = {'split_calls': 0, 'maximum_rows_per_kernel': 0, 'row_limit': row_limit}

    def forward(x):
        if layer.training:
            raise ValueError('Linear row microbatching is inference-only')
        if x.ndim < 2 or x.shape[-1] != layer.infeatures:
            raise ValueError('Invalid linear input shape')
        flat = x.reshape(-1, x.shape[-1])
        rows = len(flat)
        if rows <= row_limit:
            audit['maximum_rows_per_kernel'] = max(audit['maximum_rows_per_kernel'], rows)
            return original(x)
        output = None
        for start in range(0, rows, row_limit):
            part = original(flat[start:start + row_limit])
            if output is None:
                output = part.new_empty((rows, part.shape[-1]))
            output[start:start + len(part)] = part
            audit['maximum_rows_per_kernel'] = max(audit['maximum_rows_per_kernel'], len(part))
        audit['split_calls'] += 1
        return output.reshape(*x.shape[:-1], output.shape[-1])

    layer.forward = forward
    return audit


def configure_exllama_row_microbatch(inner, row_limit, torch):
    """Every ExLlama v1 projection must respect its shared scratch row bound."""
    from auto_gptq.nn_modules.qlinear.qlinear_exllama import QuantLinear
    layers = [(name, module) for name, module in inner.named_modules()
              if getattr(module, 'QUANT_TYPE', None) == 'exllama']
    if not layers or any(type(module) is not QuantLinear for _, module in layers):
        raise ValueError('Expected pinned ExLlama v1 quantized projections')
    audit = {'split_calls': 0, 'maximum_rows_per_kernel': 0, 'row_limit': row_limit}
    for _, module in layers:
        install_linear_row_microbatch(module, row_limit, torch, audit=audit)
    return {'quantized_linear_microbatch_rows': row_limit, 'quantized_projection_count': len(layers),
            'quantized_projection_microbatch': audit}


def retire_minicpm_audio_output(inner):
    """Remove only unused output speech modules after the official GPTQ load."""
    required = {name: getattr(inner, name) for name in ('llm', 'vpm', 'apm', 'audio_projection_layer', 'resampler')}
    removed, parameter_bytes = [], 0
    for name in ('tts', 'vocos', 'tts_processor'):
        if hasattr(inner, name):
            value = getattr(inner, name)
            if hasattr(value, 'parameters'):
                parameter_bytes += sum(p.numel() * p.element_size() for p in value.parameters())
            delattr(inner, name)
            removed.append(name)
    inner.config.init_tts = False
    if any(getattr(inner, name) is not value for name, value in required.items()):
        raise RuntimeError('Input understanding module changed during audio-output retirement')
    return {'retired_audio_output_modules': removed, 'retired_parameter_bytes': parameter_bytes}


def reclaim_unused_cache(torch, threshold_mib):
    """Real release, included in service timing; never reset peak statistics."""
    if threshold_mib is None:
        return {'performed': False}
    if type(threshold_mib) not in (int, float) or not math.isfinite(threshold_mib) or threshold_mib < 0:
        raise ValueError('Cache release threshold must be nonnegative')
    before = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    performed = before > threshold_mib * 1024**2
    if performed:
        torch.cuda.empty_cache()
    return {'performed': performed, 'unused_reserved_before_mib': before / 1024**2,
            'unused_reserved_after_mib': (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024**2}


def allocator_environment(execution):
    """Must be applied by the controller before the worker imports PyTorch."""
    value = execution.get('allocator_conf')
    return {'PYTORCH_CUDA_ALLOC_CONF': value} if value else {}
