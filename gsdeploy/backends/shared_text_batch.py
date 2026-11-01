"""Opt-in ChatQA owner with a real single batched Hugging Face generate call.

Import/CPU planning never imports torch or loads a model. The explicit retained
factory loads one frozen native model only with execute=True. This is a new
execution identity, not a modification of the frozen single-request backend.
Only padding-safe deterministic greedy configs are accepted. Each batch starts
with fresh per-row HF cache state; no prefix/KV reuse or row-early delivery.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import types


SCHEMA = 'gsdeploy.shared_text_batch.v2'
_ALLOWED_GENERATION = {'do_sample', 'num_beams', 'max_new_tokens', 'num_return_sequences',
    'repetition_penalty', 'no_repeat_ngram_size', 'top_p', 'top_k', 'temperature',
    'typical_p', 'use_cache', 'min_new_tokens', 'renormalize_logits', 'logits_to_keep'}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _config_identity(config):
    value = config.to_dict() if callable(getattr(config, 'to_dict', None)) else vars(config)
    return _digest(value)


def _rows(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    value = value.tolist() if hasattr(value, 'tolist') else value
    if not isinstance(value, list) or not value or any(not isinstance(row, list) for row in value):
        raise ValueError('Expected nonempty 2D token rows')
    if any(type(token) is not int or token < 0 for row in value for token in row):
        raise ValueError('Expected nonnegative integer token IDs/masks')
    return value


@dataclass(frozen=True)
class PreparedText:
    sample: dict
    request: dict
    input_ids: tuple[int, ...]
    compatibility_key: str
    preparation_start_ns: int
    preparation_end_ns: int
    binding_sha256: str
    owner_token: object

    @property
    def input_tokens(self):
        return len(self.input_ids)


@dataclass(frozen=True)
class BatchExecution:
    results: tuple[dict, ...]
    metrics: dict


class SharedTextBatchExecutor:
    """One model owner; scheduler groups prepared items by compatibility_key.

    executor is the frozen NativeExecutor or an injected CPU double exposing
    adapter.build/finish/generation_policy, prepare, model.generate, tokenizer,
    candidate, family, and _context_limit. tensor_factory(rows) returns long
    IDs/masks on the owner's device. CPU tests inject NumPy, not fake GPU data.
    """
    def __init__(self, executor, *, tensor_factory=None, synchronize=None,
                 inference_context=None, reclaim_cache=None, memory=None,
                 clock=time.perf_counter_ns, identity=None):
        if executor.family != 'chatqa' or executor.candidate.get('model') != 'nvidia/Llama3-ChatQA-1.5-8B':
            raise ValueError('This backend supports the pinned ChatQA text family only')
        if getattr(executor.model.config, 'is_encoder_decoder', False) or getattr(executor.model.config, 'model_type', None) != 'llama':
            raise ValueError('Requires decoder-only Llama with HF mask-derived row positions')
        unsupported = {k:v for k,v in getattr(executor, 'memory_options', {}).items()
                       if v not in (False, None, 0)}
        if unsupported:
            raise ValueError('Native per-request memory hooks require a separate batch implementation')
        self.executor = executor
        self.identity = copy.deepcopy(identity or dict(candidate=executor.candidate, family=executor.family))
        self._candidate = copy.deepcopy(executor.candidate)
        self._model = executor.model
        self._generation_config_identity = _config_identity(executor.model.generation_config)
        self._owner_token = object()
        self._lock = threading.Lock()
        self._clock = clock
        self._tensor = tensor_factory or (lambda rows: executor.torch.tensor(
            rows, dtype=executor.torch.long, device=executor.device))
        self._sync = synchronize or (lambda: None)
        self._context = inference_context or nullcontext
        self._reclaim = reclaim_cache or (lambda: None)
        self._memory = memory

    @contextmanager
    def _owned(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Shared text owner is already active; scheduler must serialize calls')
        try:
            if self.executor.model is not self._model or self.executor.candidate != self._candidate:
                raise ValueError('Model owner/candidate changed')
            if _config_identity(self._model.generation_config) != self._generation_config_identity:
                raise ValueError('Inherited model generation config changed')
            yield
        finally:
            self._lock.release()

    def _kwargs(self, request):
        generation = copy.deepcopy(request['generation'])
        native = request['native']
        overrides = native.get('generate_options', {})
        if not isinstance(overrides, dict) or any(k in generation and generation[k] != v for k,v in overrides.items()):
            raise ValueError('Native kwargs cannot change the declared quality policy')
        generation.update(overrides)
        if set(generation) - _ALLOWED_GENERATION:
            raise ValueError('Unsupported batch generation controls: '+str(sorted(set(generation)-_ALLOWED_GENERATION)))
        cfg = self.executor.model.generation_config
        effective = lambda name, default: generation.get(name, getattr(cfg, name, default))
        if effective('do_sample', False) is not False or effective('num_beams', 1) != 1 or effective('num_return_sequences', 1) != 1:
            raise ValueError('Batch backend preserves only greedy beam1/return1; no policy override')
        if effective('repetition_penalty', 1.0) != 1.0 or effective('no_repeat_ngram_size', 0) != 0:
            raise ValueError('History-dependent penalties are not padding-safe in this backend')
        if effective('renormalize_logits', False) not in (None, False, True):
            raise ValueError('Invalid unchanged renormalize_logits')
        # Absolute prompt length, history processors, assisted generation and
        # stopping-time controls can change individual greedy quality semantics.
        neutral = {'min_length': (0, None), 'encoder_repetition_penalty': (1.0,),
            'encoder_no_repeat_ngram_size': (0,), 'forced_bos_token_id': (None,),
            'forced_eos_token_id': (None,), 'bad_words_ids': (None, []),
            'force_words_ids': (None, []), 'constraints': (None, []),
            'sequence_bias': (None, {}), 'begin_suppress_tokens': (None, []),
            'suppress_tokens': (None, []), 'stop_strings': (None, []),
            'max_time': (None,), 'penalty_alpha': (None,), 'dola_layers': (None,),
            'forced_decoder_ids': (None, []), 'exponential_decay_length_penalty': (None,),
            'watermarking_config': (None,), 'prompt_lookup_num_tokens': (None, 0),
            'token_healing': (False,), 'is_assistant': (False,), 'guidance_scale': (None, 1.0),
            'cache_implementation': (None,), 'num_beam_groups': (1,)}
        for key, allowed in neutral.items():
            if getattr(cfg, key, allowed[0]) not in allowed:
                raise ValueError(f'Unsupported inherited generation config: {key}')
        budget = generation.get('max_new_tokens')
        if type(budget) is not int or budget < 1:
            raise ValueError('An unchanged positive max_new_tokens budget is required')
        minimum = effective('min_new_tokens', 0)
        if minimum is not None and (type(minimum) is not int or not 0 <= minimum <= budget):
            raise ValueError('Invalid unchanged min_new_tokens')
        tokenizer = self.executor.tokenizer
        names = native.get('terminator_tokens')
        if names != ['<|end_of_text|>', '<|eot_id|>']:
            raise ValueError('Unsupported ChatQA terminator contract')
        eos = [tokenizer.convert_tokens_to_ids(name) for name in names]
        if (any(type(token) is not int or token < 0 or token == getattr(tokenizer, 'unk_token_id', None) for token in eos)
                or [tokenizer.convert_ids_to_tokens(token) for token in eos] != names or len(set(eos)) != len(eos)):
            raise ValueError('Pinned ChatQA terminators are missing/ambiguous')
        # Matches HF's single-request EOS fallback when no model pad is set;
        # never resize embeddings or borrow a different tokenizer pad token.
        pad = getattr(cfg, 'pad_token_id', None)
        if pad is None:
            pad = eos[0]
        if type(pad) is not int or pad < 0:
            raise ValueError('A valid existing model pad/EOS token is required')
        generation.update(eos_token_id=eos, pad_token_id=pad)
        return generation

    def _compatibility(self, request, kwargs):
        return _digest(dict(owner=self.identity,
            request={k:request.get(k) for k in ('model','revision','precision','protocol_sha256','adapter_sha256')},
            generation=kwargs, native={k:v for k,v in request['native'].items() if k != 'text'}))

    def prepare(self, sample):
        """Reuse native build/prepare without forward/generation; returns exact length."""
        with self._owned():
            if sample.get('task') != 'text':
                raise ValueError('Shared ChatQA owner accepts text requests only')
            start = self._clock()
            sample = copy.deepcopy(sample)
            request = self.executor.adapter.build(self._candidate, sample)
            if request.get('task') != 'text' or request.get('media') is not None or request.get('input_truncation') is not False:
                raise ValueError('Requires complete text-only native requests without truncation')
            kwargs = self._kwargs(request)
            prepared = self.executor.prepare(request)
            inputs = prepared['inputs']
            if set(inputs) - {'input_ids', 'attention_mask'}:
                raise ValueError('Additional native model inputs need an explicit batch adapter')
            ids = _rows(inputs['input_ids'])
            mask = _rows(inputs['attention_mask']) if 'attention_mask' in inputs else [[1]*len(ids[0])]
            if len(ids) != 1 or not ids[0] or mask != [[1]*len(ids[0])] or prepared['input_tokens'] != len(ids[0]):
                raise ValueError('Expected one complete unpadded native input')
            if ids[0][0] != self.executor.tokenizer.bos_token_id:
                raise ValueError('Native ChatQA BOS token changed')
            if len(ids[0])+kwargs['max_new_tokens'] > self.executor._context_limit():
                raise ValueError('Complete input plus unchanged output budget exceeds context')
            key = self._compatibility(request, kwargs)
            binding = _digest(dict(sample=sample, request=request, ids=ids[0], compatibility=key))
            return PreparedText(sample, request, tuple(ids[0]), key, start, self._clock(), binding, self._owner_token)

    def execute_prepared(self, items, *, evaluate_quality=False):
        """Scheduler entry: same compatibility key; complete batch delivered together."""
        items = tuple(items)
        if not items:
            raise ValueError('A nonempty batch is required')
        if type(evaluate_quality) is not bool:
            raise ValueError('evaluate_quality must be Boolean')
        with self._owned():
            admit = self._clock()
            kwargs = None
            for item in items:
                if not isinstance(item, PreparedText) or item.owner_token is not self._owner_token:
                    raise ValueError('Prepared item belongs to another model owner')
                if item.compatibility_key != items[0].compatibility_key:
                    raise ValueError('Incompatible batch; scheduler must group exact policies')
                current = self._kwargs(item.request)
                if (self._compatibility(item.request, current) != item.compatibility_key
                        or _digest(dict(sample=item.sample, request=item.request, ids=list(item.input_ids),
                                        compatibility=item.compatibility_key)) != item.binding_sha256
                        or self.executor.adapter.generation_policy('text', self._candidate['model']) != item.request['generation']):
                    raise ValueError('Prepared input or generation policy changed')
                kwargs = current
            width = max(item.input_tokens for item in items)
            if width+kwargs['max_new_tokens'] > self.executor._context_limit():
                raise ValueError('Padded complete batch exceeds model context')
            pad = kwargs['pad_token_id']
            rows = [[pad]*(width-item.input_tokens)+list(item.input_ids) for item in items]
            masks = [[0]*(width-item.input_tokens)+[1]*item.input_tokens for item in items]
            # Fresh HF generate call: no externally supplied/shared cache,
            # embeddings or stateful stopping criteria. Llama derives per-row
            # position IDs from this mask on every generation step.
            self._sync()
            service_start = self._clock()
            ids_tensor, mask_tensor = self._tensor(rows), self._tensor(masks)
            generate_start = self._clock()
            with self._context():
                output = self._model.generate(input_ids=ids_tensor, attention_mask=mask_tensor, **kwargs)
            self._sync()
            generate_finish = self._clock()
            tokens = _rows(output.sequences if hasattr(output, 'sequences') else output)
            if len(tokens) != len(items) or len({len(row) for row in tokens}) != 1:
                raise ValueError('HF generate returned an invalid batch shape')
            results = []
            for item, row, prefix in zip(items, tokens, rows):
                if row[:width] != prefix:
                    raise ValueError('HF generate changed the padded prompt prefix')
                generated = row[width:]
                budget, eos = kwargs['max_new_tokens'], kwargs['eos_token_id']
                if not generated or len(generated) > budget:
                    raise ValueError('HF generate returned empty/over-budget completion')
                first_eos = next((i for i,t in enumerate(generated) if t in eos), None)
                if first_eos is not None:
                    if any(t != pad for t in generated[first_eos+1:]):
                        raise ValueError('Tokens after row EOS must be HF fill padding')
                    generated = generated[:first_eos+1]
                    reason = 'eos'
                elif len(generated) == budget:
                    reason = 'length'
                else:
                    raise ValueError('Short completion without EOS violates the original stopping contract')
                observation = dict(finish_reason=reason, generated_tokens=len(generated),
                                   terminal_token_id=generated[-1], eos_token_ids=eos)
                raw = self.executor.tokenizer.decode(generated, skip_special_tokens=True)
                result = self.executor.adapter.finish(item.request, item.sample, raw,
                    evaluate_quality=evaluate_quality, **observation)
                audit_keys = ('do_sample','num_beams','repetition_penalty','no_repeat_ngram_size',
                              'max_new_tokens','num_return_sequences','eos_token_id','use_cache','logits_to_keep')
                effective = {k:copy.deepcopy(kwargs[k]) for k in audit_keys if k in kwargs}
                effective['class_constraint'] = False
                result.update(input_tokens=item.input_tokens, effective_generation=effective,
                    reference=item.sample.get('label'), answers=item.sample.get('answers'),
                    memory_optimizations=copy.deepcopy(getattr(self.executor, 'memory_audit', {})),
                    batch_execution=dict(schema=SCHEMA, batch_size=len(items),
                        padded_input_tokens=width, padding_tokens=width-item.input_tokens,
                        delivery='whole_batch_after_generate_and_cleanup',
                        service_scope='shared_batch_wall_time_excludes_recorded_preparation_not_additive_per_request',
                        KV_scope='fresh_generate_per_batch; independent masked rows; no cross-request KV reuse',
                        preparation_start_ns=item.preparation_start_ns, preparation_end_ns=item.preparation_end_ns))
                results.append(result)
            cleanup = self._reclaim()
            self._sync()
            service_finish = self._clock()
            for result in results:
                result.update(service_ms=(service_finish-service_start)/1e6,
                              cache_reclamation=copy.deepcopy(cleanup))
            finish = self._clock()
            metrics = dict(schema=SCHEMA, backend='same_owner_chatqa_hf_batch_v2',
                owner_identity=copy.deepcopy(self.identity), model_owner_count=1,
                generate_calls=1, batch_size=len(items), compatibility_key=items[0].compatibility_key,
                batch_admit_ns=admit, service_start_ns=service_start, generate_start_ns=generate_start,
                generate_finish_ns=generate_finish, service_finish_ns=service_finish, results_ready_ns=finish,
                useful_input_tokens=sum(i.input_tokens for i in items), padded_input_tokens=len(items)*width,
                padding_tokens=sum(width-i.input_tokens for i in items),
                preparation_wall_ns=sum(i.preparation_end_ns-i.preparation_start_ns for i in items),
                tensor_payload_bytes=len(items)*width*2*8,
                shared_weight_owner=True, common_submatrix_merge=False, prefix_cache_reuse=False,
                row_early_delivery=False, final_gain_eligible=False)
            if self._memory:
                metrics['owner_memory'] = self._memory()
            return BatchExecution(tuple(results), metrics)

    def run_batch(self, samples, *, evaluate_quality=False):
        return self.execute_prepared([self.prepare(sample) for sample in samples], evaluate_quality=evaluate_quality)


def load_retained(spec_path, source_root, *, execute=False, expected_spec_sha256=None):
    """Explicit GPU-loading factory; no fallback/model substitution on failure."""
    if execute is not True:
        raise RuntimeError('Model loading requires explicit execute=True')
    spec_path, source_root = Path(spec_path).resolve(), Path(source_root).resolve()
    spec_bytes = spec_path.read_bytes()
    spec_hash = hashlib.sha256(spec_bytes).hexdigest()
    if expected_spec_sha256 is not None and spec_hash != expected_spec_sha256:
        raise ValueError('Worker spec SHA256 changed')
    spec = json.loads(spec_bytes)
    if spec['execution']['family'] != 'chatqa' or spec['candidate']['slot'] != 'text':
        raise ValueError('Only the explicit retained text worker is supported')
    if spec.get('task_candidates', {}).get('text', spec['candidate']) != spec['candidate']:
        raise ValueError('Multiple candidate identities require another backend')
    # A private namespace keeps frozen relative imports frozen even when the
    # current gsdeploy package is already loaded; no sys.path/module replacement.
    package_root = source_root/'gsdeploy'
    sources = {str(p.relative_to(package_root)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(package_root.rglob('*.py'))}
    if not sources or 'backends/compact_native.py' not in sources:
        raise ValueError('Frozen native source package is incomplete')
    namespace = '_gsdeploy_batch_retained_'+_digest(sources)
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(package_root)]
        package.__package__ = namespace
        sys.modules[namespace] = package
    memory_module = importlib.import_module(namespace+'.backends.compact_memory')
    expected = memory_module.allocator_environment(spec['execution'])
    if any(os.environ.get(k) != v for k,v in expected.items()):
        raise ValueError('Frozen allocator environment must be applied before model loading')
    native = importlib.import_module(namespace+'.backends.compact_native')
    if sources != {str(p.relative_to(package_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(package_root.rglob('*.py'))} or spec_path.read_bytes() != spec_bytes:
        raise ValueError('Frozen source/spec changed during imports; model loading stopped')
    adapter = native.ProfileAdapter(spec, spec['resources'])
    if sources != {str(p.relative_to(package_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(package_root.rglob('*.py'))} or spec_path.read_bytes() != spec_bytes:
        raise ValueError('Frozen source/spec changed during model loading; owner rejected')
    return SharedTextBatchExecutor(adapter.executor,
        synchronize=adapter.torch.cuda.synchronize,
        inference_context=adapter.torch.inference_mode,
        reclaim_cache=lambda:memory_module.reclaim_unused_cache(adapter.torch, adapter.cache_release_threshold_mib),
        memory=adapter.memory,
        identity=dict(spec_path=str(spec_path), spec_sha256=spec_hash,
                      source_root=str(source_root), source_tree_sha256=_digest(sources),
                      candidate=spec['candidate'], execution=spec['execution']))
