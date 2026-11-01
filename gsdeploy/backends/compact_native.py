"""Native compact inference with explicit injection; imports/loads nothing at import.

CPU preparation uses real processors and injected CPU test doubles. Loading the
candidate checkpoints is a separate, explicitly enabled GPU-only operation.
"""

from contextlib import contextmanager, nullcontext
import copy
import hashlib
import io
from pathlib import Path
import tempfile

from .compact import CompactAdapter, FAMILIES, CLASS_NAME_DECODING, CLASS_NAME_DECODING_V2
from .compact_adapters import verified_qa_adapter
from .compact_megrez import complete_megrez_inputs, native_output_budget
from .compact_memory import (last_position_logits, install_phi_vision_microbatch,
                             configure_minicpm_language_sdpa, retire_minicpm_audio_output,
                             minicpm_native_model, configure_exllama_row_microbatch,
                             exllama_request_scratch, actual_generation_scratch_rows)


def resolve_dtype(torch, name):
    if name not in {'float16', 'bfloat16', 'float32'}:
        raise ValueError('Unsupported explicit inference dtype: ' + str(name))
    return getattr(torch, name)


def sequence_ids(output):
    value = output.sequences if hasattr(output, 'sequences') else output
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    rows = value.tolist() if hasattr(value, 'tolist') else value
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list):
        raise ValueError('Only batch-one token sequences are supported')
    return [int(i) for i in rows[0]]


def eos_ids(owner, tokenizer, kwargs):
    value = kwargs.get('eos_token_id')
    if value is None:
        value = getattr(getattr(owner, 'generation_config', None), 'eos_token_id', None)
    if value is None:
        value = getattr(tokenizer, 'eos_token_id', None)
    return list(value) if isinstance(value, (list, tuple)) else [value] if type(value) is int else []


def observe(ids, prefix, eos, budget):
    if prefix < 0 or prefix > len(ids):
        raise ValueError('Invalid decoder/input prefix')
    generated = ids[prefix:]
    # Batch-one generation has no post-EOS padding. Never guess termination from text.
    terminal = generated[-1] if generated else None
    reason = 'eos' if terminal in eos else 'length' if len(generated) >= budget else 'unknown'
    return {'finish_reason': reason, 'generated_tokens': len(generated),
            'terminal_token_id': terminal, 'eos_token_ids': eos}


def generation_audit(kwargs):
    """Record explicit effective native kwargs, without tensors or model objects."""
    keys = ['do_sample', 'num_beams', 'repetition_penalty', 'no_repeat_ngram_size',
            'max_new_tokens', 'num_return_sequences', 'eos_token_id', 'use_cache']
    result = {key: kwargs[key] for key in keys if key in kwargs}
    for key in keys:
        if 'thinker_' + key in kwargs:
            result[key] = kwargs['thinker_' + key]
    result['class_constraint'] = 'logits_processor' in kwargs
    return result


def label_token_trie(tokenizer, labels, eos, budget, *, answer_prefix=''):
    """All legal class IDs, without sample references or candidate pruning."""
    if not eos:
        raise ValueError('Constrained classification requires known EOS tokens')
    root, sequences = {}, set()
    for label in labels:
        ids = tuple(tokenizer.encode(label, add_special_tokens=False))
        if answer_prefix:
            prefix = tuple(tokenizer.encode(answer_prefix, add_special_tokens=False))
            if tuple(tokenizer.encode(answer_prefix + label, add_special_tokens=False)) != prefix + ids:
                raise ValueError('Answer prefix changes canonical label token boundaries')
        if not ids or len(ids) + 1 > budget or ids in sequences:
            raise ValueError('Class IDs must have unique tokenizations within the output budget')
        sequences.add(ids)
        node = root
        for token in ids:
            node = node.setdefault(token, {})
        node[None] = tuple(eos)
    return root


def label_prefix_constraint(trie):
    # Each request has one greedy sequence. For native inputs_embeds calls the
    # initial sequence may be empty; detect its prefix on the first callback.
    prefix = None

    def allowed(batch_id, input_ids):
        nonlocal prefix
        if batch_id != 0:
            raise ValueError('Only batch-one constrained classification is supported')
        if prefix is None:
            prefix = len(input_ids)
        node = trie
        for token in input_ids[prefix:].tolist():
            if token not in node:
                raise ValueError('Generation left the complete class-ID token trie')
            node = node[token]
        return sorted([token for token in node if token is not None] + list(node.get(None, ())))

    return allowed


class ClassLabelLogitsProcessor:
    """Works with the empty input_ids used by pinned MiniCPM/Transformers 4.44.

    That version's built-in PrefixConstrainedLogitsProcessor reshapes an empty
    sequence with an ambiguous -1 dimension before invoking the user callback.
    """
    def __init__(self, trie, num_beams=1):
        self.allowed = label_prefix_constraint(trie)
        if type(num_beams) is not int or not 1 <= num_beams <= 8:
            raise ValueError('Invalid classification beam count')
        self.num_beams = num_beams
        self.discarded_beam_rows = 0

    def __call__(self, input_ids, scores):
        if len(input_ids) != self.num_beams or len(scores) != self.num_beams:
            raise ValueError('Classification rows must match the declared beam count')
        mask = scores.new_full(scores.shape, float('-inf'))
        for row in range(self.num_beams):
            try:
                allowed = self.allowed(0, input_ids[row])
            except ValueError:
                if self.num_beams == 1:
                    raise
                # HF may retain impossible/finished hypotheses while other
                # beams continue. Eliminate that row; never relax the class
                # vocabulary or substitute an arbitrary token/class.
                self.discarded_beam_rows += 1
                continue
            mask[row, allowed] = 0
        return scores + mask


@contextmanager
def capture_generation(owner, tokenizer, generation, *, encoder_decoder=False, prefix_override=None, around=None):
    """Observe one native generate call without changing its return type.

    The patched method is restored on exceptions. Each worker owns its model and
    processes one request at a time; this capture is deliberately not thread-safe.
    """
    original = owner.generate
    had_instance = 'generate' in vars(owner)
    previous = vars(owner).get('generate')
    calls = []

    def wrapped(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.update(generation)
        kwargs.pop('temperature', None)
        kwargs.pop('top_p', None)
        kwargs.pop('top_k', None)
        with around(kwargs) if around is not None else nullcontext():
            output = original(*args, **kwargs)
        ids = sequence_ids(output)
        inputs = kwargs.get('input_ids', args[0] if args else None)
        if prefix_override is not None:
            prefix = prefix_override
        elif encoder_decoder:
            decoder = kwargs.get('decoder_input_ids')
            prefix = int(decoder.shape[-1]) if decoder is not None else 1
        else:
            prefix = int(inputs.shape[-1]) if inputs is not None else 0
        calls.append({'ids': ids, 'prefix': prefix,
                      'effective_generation': generation_audit(kwargs),
                      'observation': observe(ids, prefix, eos_ids(owner, tokenizer, kwargs), generation['max_new_tokens'])})
        return output
    owner.generate = wrapped
    try:
        yield calls
        if len(calls) != 1:
            raise ValueError('Expected exactly one native generation call per request')
    finally:
        if had_instance:
            owner.generate = previous
        else:
            delattr(owner, 'generate')


class CompleteMiniProcessor:
    """Turn off MiniCPM's hidden list slicing, then reject overflow explicitly."""
    def __init__(self, processor, limit, assistant_prefix=''):
        self.processor, self.limit = processor, limit
        self.assistant_prefix = assistant_prefix
        self.last_input_tokens = None

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, *args, **kwargs):
        if self.assistant_prefix:
            def append(text):
                if isinstance(text, str):
                    return text + self.assistant_prefix
                if isinstance(text, list) and all(isinstance(t, str) for t in text):
                    return [t + self.assistant_prefix for t in text]
                raise ValueError('Expected native text prompts before MiniCPM tokenization')
            if args:
                args = (append(args[0]), *args[1:])
            elif 'text' in kwargs:
                kwargs['text'] = append(kwargs['text'])
            else:
                raise ValueError('Missing MiniCPM native prompt for answer prefix')
        kwargs['max_length'] = None
        kwargs['truncation'] = False
        result = self.processor(*args, **kwargs)
        self.last_input_tokens = int(result['input_ids'].shape[-1])
        if self.last_input_tokens > self.limit:
            raise ValueError('MiniCPM full multimodal input exceeds the frozen budget')
        return result


class NativeExecutor:
    def __init__(self, adapter, candidate, model, processor, *, torch, device, dtype,
                 tokenizer=None, image_loader=None, audio_loader=None, output_root=None, input_limit=None,
                 memory_options=None, memory_audit=None):
        self.adapter, self.candidate, self.model, self.processor = adapter, candidate, model, processor
        self.family = FAMILIES[candidate['model']]
        self.torch, self.device, self.dtype = torch, device, dtype
        self.tokenizer = tokenizer or getattr(processor, 'tokenizer', processor)
        self.image_loader, self.audio_loader = image_loader, audio_loader
        self.output_root = output_root
        self.input_limit = input_limit
        self.text_features = None
        self.text_cache_key = None
        self.image_token_trie = None
        self.memory_options = memory_options or {}
        self.memory_audit = memory_audit or {}
        self.last_media_binding = None
        self.last_media_bytes = None

    def media(self, request):
        media = request['media']
        self.last_media_binding = None
        self.last_media_bytes = None
        if not media:
            return None
        if media['kind'] == 'image':
            if self.image_loader:
                return self.image_loader(media['path'])
            from PIL import Image, ImageOps
            raw = Path(media['path']).read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            if actual != media['sha256']:
                raise ValueError('Image bytes changed after input freezing')
            self.last_media_bytes = raw
            with Image.open(io.BytesIO(raw)) as image:
                prepared = ImageOps.exif_transpose(image).convert('RGB')
            self.last_media_binding = {'kind': 'image', 'path': media['path'], 'sha256': actual,
                                       'size': list(prepared.size), 'mode': prepared.mode}
            return prepared
        if self.audio_loader:
            return self.audio_loader(media['path'])
        import soundfile as sf
        raw = Path(media['path']).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != media['sha256']:
            raise ValueError('Audio bytes changed after input freezing')
        self.last_media_bytes = raw
        audio, rate = sf.read(io.BytesIO(raw), dtype='float32')
        if rate != 16000 or audio.ndim != 1:
            raise ValueError('Frozen ASR input must be mono 16kHz, without downmix or resampling')
        if abs(len(audio) / rate - media['duration_s']) > 1 / rate:
            raise ValueError('Audio duration differs from frozen metadata')
        self.last_media_binding = {'kind': 'audio', 'path': media['path'], 'sha256': actual,
                                   'samples': len(audio), 'sampling_rate': rate}
        return audio

    def move(self, inputs):
        def moved(value, cast_floating=True):
            if hasattr(value, 'to'):
                # Integer IDs/masks keep their dtype; only floating tensors are cast.
                if cast_floating and getattr(value, 'is_floating_point', lambda: False)():
                    return value.to(device=self.device, dtype=self.dtype)
                return value.to(device=self.device)
            if isinstance(value, list):
                return [moved(v, cast_floating) for v in value]
            if isinstance(value, tuple):
                return tuple(moved(v, cast_floating) for v in value)
            return value
        # Native Phi counts visual tokens by summing a float32 0/1 mask.
        # BF16 rounds large integer sums, breaking processor/model agreement.
        return {key: moved(value, cast_floating=not key.endswith('_mask')) for key, value in inputs.items()}

    def check_input(self, inputs, budget, *, max_output=0):
        length = int(inputs['input_ids'].shape[-1])
        if length + max_output > budget:
            raise ValueError('Complete input plus output budget exceeds model context; no truncation allowed')
        return length

    def _context_limit(self):
        config = self.model.config
        config = getattr(config, 'thinker_config', config)
        config = getattr(config, 'text_config', config)
        return int(getattr(config, 'max_position_embeddings', 32768))

    def prepare(self, request):
        """CPU-capable processor binding; no model forward or token generation."""
        native, task, media = request['native'], request['task'], self.media(request)
        if self.family == 'qwen':
            text = self.processor.apply_chat_template(native['messages'], tokenize=False, add_generation_prompt=True)
            # The pinned Omni preview always batchifies apply_chat_template.
            if isinstance(text, str):
                text = [text]
            if not isinstance(text, list) or len(text) != 1 or not isinstance(text[0], str):
                raise ValueError('Expected one rendered Qwen conversation')
            text = [value + native.get('assistant_prefix', '') for value in text]
            kwargs = {'text': text, **copy.deepcopy(native['processor_options'])}
            if request['media']:
                if task == 'asr':
                    kwargs.update(audio=[media], sampling_rate=16000)
                else:
                    kwargs['images'] = [media]
            inputs = self.processor(**kwargs)
            length = self.check_input(inputs, self._context_limit(), max_output=request['generation']['max_new_tokens'])
        elif self.family == 'megrez':
            messages = copy.deepcopy(native['messages'])
            if request['media']:
                kind = request['media']['kind']
                bound = self.last_media_bytes if kind == 'audio' else media
                if bound is None:
                    raise ValueError('Megrez requires verified complete media bytes')
                messages[0]['content'][kind] = [bound]
            inputs = complete_megrez_inputs(self.processor, messages, native.get('assistant_prefix', ''))
            length = int(inputs['input_ids'].shape[-1])
            effective = native_output_budget(length, self._context_limit(),
                                            request['generation']['max_new_tokens'], native['context_budget'])
            return {'media': media, 'input_tokens': length, 'inputs': inputs,
                    'effective_output_budget': effective}
        elif self.family == 'phi':
            kwargs = {'text': native['prompt'], **copy.deepcopy(native['processor_options'])}
            if task == 'asr':
                kwargs['audios'] = [(media, 16000)]
            elif media is not None:
                kwargs['images'] = [media]
            inputs = self.processor(**kwargs)
            length = self.check_input(inputs, self._context_limit(), max_output=request['generation']['max_new_tokens'])
        elif self.family == 'chatqa':
            if not self.tokenizer.bos_token or type(self.tokenizer.bos_token_id) is not int:
                raise ValueError('Pinned ChatQA requires a known BOS token')
            inputs = self.processor(self.tokenizer.bos_token + native['text'], **native['tokenizer_options'])
            if int(inputs['input_ids'][0, 0]) != self.tokenizer.bos_token_id:
                raise ValueError('ChatQA tokenizer did not preserve the explicit BOS token')
            length = self.check_input(inputs, self._context_limit(), max_output=request['generation']['max_new_tokens'])
        elif self.family == 'unifiedqa':
            inputs = self.processor(native['text'], **native['tokenizer_options'])
            length = self.check_input(inputs, native['max_input_tokens'])
        elif self.family == 'whisper':
            inputs = self.processor(media, **native['processor_options'])
            length = 0
        else:
            return {'media': media, 'input_tokens': None, 'inputs': None}
        return {'media': media, 'input_tokens': length, 'inputs': inputs}

    def _siglip(self, request, image):
        torch = self.torch
        native = request['native']
        if self.text_cache_key != native['cache_key']:
            features = []
            for start in range(0, 1000, 32):
                batch = self.processor(text=native['class_texts'][start:start + 32], **native['text_processor_options'])
                if int(batch['input_ids'].shape[-1]) > native['text_processor_options']['max_length']:
                    raise ValueError('A label exceeds the frozen SigLIP text length')
                features.append(self.model.get_text_features(**self.move(batch)))
            self.text_features = torch.nn.functional.normalize(torch.cat(features), dim=-1)
            self.text_cache_key = native['cache_key']
        inputs = self.processor(images=image, return_tensors='pt')
        features = torch.nn.functional.normalize(self.model.get_image_features(**self.move(inputs)), dim=-1)
        scores = (features @ self.text_features.T)[0].detach().cpu().tolist()
        import math
        if any(not math.isfinite(s) for s in scores):
            raise ValueError('Non-finite classification scores')
        if len(scores) != 1000:
            raise ValueError('Missing class features')
        index = max(range(1000), key=lambda i: scores[i])
        return native['class_ids'][index], {'finish_reason': 'classified', 'generated_tokens': 0,
                                         'class_scores': scores}, {'class_scores': scores}

    def infer(self, sample, *, evaluate_quality=True):
        if self.family in {'minicpm', 'megrez'}:
            inner = minicpm_native_model(self.model) if self.family == 'minicpm' else self.model
            owner = inner.llm
        else:
            owner = getattr(self.model, 'thinker', self.model) if self.family == 'qwen' else self.model
        enabled = self.memory_options.get('last_position_logits', False)
        beams = self.adapter.generation_policy(sample['task'], self.candidate['model'])['num_beams']
        if enabled and beams != 1:
            raise ValueError('Last-position projection hook requires one beam; disable it for native beam search')
        scratch = (self.family == 'minicpm' and self.memory_options.get('defer_exllama_scratch', False)
                   and not self.memory_options.get('actual_prefill_scratch_rows',False))
        active_rows = self.input_limit * beams if self.input_limit is not None else None
        with exllama_request_scratch(self.model, scratch, active_rows, self.torch,
                                     audit=self.memory_audit.get('deferred_exllama_scratch')):
            with last_position_logits(owner, enabled):
                return self._infer(sample, evaluate_quality=evaluate_quality)

    def _infer(self, sample, *, evaluate_quality=True):
        request = self.adapter.build(self.candidate, sample)
        with self.torch.inference_mode():
            prepared = self.prepare(request)
            native = request['native']
            generation = dict(request['generation'])
            if self.memory_options.get('disable_language_hidden_states', False):
                generation.update(output_hidden_states=False, output_attentions=False)
            if self.family == 'megrez':
                # Pinned HF forces caching for decoder-only inputs_embeds.
                # Declare the actual behavior; do not report a cache-off control
                # that the underlying generation implementation overrides.
                if self.memory_options.get('native_kv_cache', True) is not True:
                    raise ValueError('Megrez inputs_embeds generation requires native KV caching')
                generation['use_cache'] = True
            if self.family == 'phi' and self.memory_options.get('native_last_logits', False):
                generation['num_logits_to_keep'] = 1
            class_processor = None
            if (request['task'] == 'image' and self.family != 'siglip'
                    and self.adapter.policy['tasks']['image'].get('generative_decoding') in {
                        'complete_label_token_trie_greedy_v1', CLASS_NAME_DECODING, CLASS_NAME_DECODING_V2}):
                if self.image_token_trie is None:
                    owner = self.model.thinker if self.family == 'qwen' and hasattr(self.model, 'thinker') else self.model
                    if self.family in {'minicpm', 'megrez'}:
                        owner = (minicpm_native_model(self.model) if self.family == 'minicpm' else self.model).llm
                    self.image_token_trie = label_token_trie(self.tokenizer, self.adapter.image_output_labels,
                                                           eos_ids(owner, self.tokenizer, {}), generation['max_new_tokens'],
                                                           answer_prefix=native.get('assistant_prefix', ''))
                class_processor = ClassLabelLogitsProcessor(self.image_token_trie, generation['num_beams'])
                generation['logits_processor'] = [class_processor]
            extras = {}
            if self.family == 'siglip':
                raw, observation, extras = self._siglip(request, prepared['media'])
            elif self.family == 'megrez':
                effective = prepared['effective_output_budget']
                generation['max_new_tokens'] = effective
                with capture_generation(self.model.llm, self.tokenizer, generation, prefix_override=0) as calls:
                    output = self.model.generate(**prepared['inputs'], **generation)
                ids = sequence_ids(output)
                observation = calls[0]['observation']
                raw = self.tokenizer.decode(ids, skip_special_tokens=True)
                extras.update(input_tokens=prepared['input_tokens'], effective_generation=calls[0]['effective_generation'],
                              native_context_budget={'policy':native['context_budget'],
                                  'context_tokens': self._context_limit(),
                                  'requested_max_new_tokens': request['generation']['max_new_tokens'],
                                  'effective_max_new_tokens': effective, 'input_truncated': False})
            elif self.family == 'minicpm':
                inner = minicpm_native_model(self.model)
                processor = CompleteMiniProcessor(self.processor, min(32768 - generation['max_new_tokens'],
                                                                       self.input_limit or 32768),
                                                  native.get('assistant_prefix', ''))
                msgs = copy.deepcopy(native['msgs'])
                if request['media']:
                    msgs[0]['content'][0] = prepared['media']
                around = None
                if self.memory_options.get('actual_prefill_scratch_rows',False):
                    around = lambda kwargs: exllama_request_scratch(self.model,True,
                        actual_generation_scratch_rows(kwargs,self.input_limit),self.torch,
                        audit=self.memory_audit.get('deferred_exllama_scratch'))
                with capture_generation(inner.llm, self.tokenizer, generation, around=around) as calls:
                    raw = inner.chat(msgs=msgs, tokenizer=self.tokenizer, processor=processor,
                                     max_new_tokens=generation['max_new_tokens'], **native['options'])
                raw = raw if isinstance(raw, str) else raw.text
                observation = calls[0]['observation']
                extras['input_tokens'] = processor.last_input_tokens
                extras['effective_generation'] = calls[0]['effective_generation']
            elif self.family in {'deepseek', 'deepseek2'}:
                if self.output_root is None:
                    raise ValueError('An isolated request output root is required')
                Path(self.output_root).mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix='ocr-', dir=self.output_root) as output:
                    options = copy.deepcopy(native['options'])
                    if self.last_media_bytes is not None:
                        # The native file-based reader must consume the bytes
                        # we verified, even if the original path changes later.
                        verified_input = Path(output) / ('verified-input' + (Path(options['image_file']).suffix or '.png'))
                        verified_input.write_bytes(self.last_media_bytes)
                        options['image_file'] = str(verified_input)
                        extras['native_input_binding'] = {'source_path': request['media']['path'],
                            'sha256': hashlib.sha256(self.last_media_bytes).hexdigest(), 'kind': 'owned_verified_bytes'}
                    with capture_generation(self.model, self.tokenizer, generation) as calls:
                        raw = self.model.infer(self.tokenizer, output_path=output, **options)
                    if self.last_media_bytes is not None and verified_input.read_bytes() != self.last_media_bytes:
                        raise ValueError('Native OCR changed its verified input file')
                observation = calls[0]['observation']
                extras['input_tokens'] = calls[0]['prefix']
                extras['effective_generation'] = calls[0]['effective_generation']
            else:
                inputs = self.move(prepared['inputs'])
                kwargs = dict(generation)
                kwargs.update(native.get('generate_options', {}))
                if self.family == 'chatqa':
                    terminators = [self.tokenizer.convert_tokens_to_ids(token) for token in native['terminator_tokens']]
                    if (any(type(token) is not int or token == getattr(self.tokenizer, 'unk_token_id', None)
                            for token in terminators)
                            or [self.tokenizer.convert_ids_to_tokens(token) for token in terminators] != native['terminator_tokens']):
                        raise ValueError('Pinned ChatQA terminator tokens are missing or ambiguous')
                    kwargs['eos_token_id'] = terminators
                if self.family == 'qwen':
                    kwargs['thinker_max_new_tokens'] = kwargs.pop('max_new_tokens')
                    kwargs['thinker_do_sample'] = kwargs.pop('do_sample')
                    kwargs['thinker_num_beams'] = kwargs.pop('num_beams')
                if self.family == 'whisper':
                    # Force and retain a known decoder prefix to count exactly generated tokens.
                    forced = self.processor.get_decoder_prompt_ids(language='english', task='transcribe', no_timestamps=True)
                    prefix_ids = [self.model.config.decoder_start_token_id] + [token for _, token in sorted(forced)]
                    kwargs['decoder_input_ids'] = self.torch.tensor([prefix_ids], device=self.device)
                    if len(prefix_ids) + generation['max_new_tokens'] > self.model.config.max_target_positions:
                        raise ValueError('Whisper decoder prefix plus output exceeds target context')
                    prefix = len(prefix_ids)
                    # Whisper retrieves its language/task prefix before checking
                    # decoder_input_ids. Retain the frozen English transcription
                    # options so it does not run automatic language detection.
                    kwargs['forced_decoder_ids'] = None
                elif self.family == 'unifiedqa':
                    prefix = 1
                else:
                    prefix = prepared['input_tokens']
                output = self.model.generate(**inputs, **kwargs)
                ids = sequence_ids(output)
                observation = observe(ids, prefix, eos_ids(self.model, self.tokenizer, kwargs), generation['max_new_tokens'])
                raw = self.tokenizer.decode(ids[prefix:], skip_special_tokens=True)
                extras['input_tokens'] = prepared['input_tokens']
                extras['effective_generation'] = generation_audit(kwargs)
            result = self.adapter.finish(request, sample, raw, evaluate_quality=evaluate_quality, **observation)
            result.update(extras)
            if class_processor is not None:
                result['class_constraint_audit'] = {'num_beams': class_processor.num_beams,
                    'discarded_impossible_rows': class_processor.discarded_beam_rows,
                    'semantic_candidate_classes': 1000}
            if self.last_media_binding is not None:
                result['media_binding'] = copy.deepcopy(self.last_media_binding)
            if sample['task'] == 'image' and evaluate_quality and result.get('quality_eligible', True):
                result['correct'] = result['score']['correct']
            return result


def load_native(adapter, candidate, profile, resources, *, allow_gpu=False, output_root=None):
    """Only explicit GPU collection may call this; CPU preflight never does."""
    if not allow_gpu:
        raise RuntimeError('GPU model loading is disabled; use CPU readiness/preprocessing checks')
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer
    from transformers import BitsAndBytesConfig, T5ForConditionalGeneration, T5Tokenizer

    directory = Path(resources['path'])
    if (resources.get('model') != candidate['model'] or resources['revision'] != candidate['revision']
            or not resources['files'] or any(f['status'] != 'VERIFIED' for f in resources['files'])):
        raise ValueError('Only the complete verified pinned checkpoint can be loaded')
    for entry in resources['files']:
        stat = (directory / entry['name']).stat()
        current = {'inode': stat.st_ino, 'device': stat.st_dev, 'size': stat.st_size,
                   'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}
        if stat.st_size != entry['size'] or entry.get('stat') != current:
            raise ValueError('Checkpoint file changed after CPU verification')
    trained_adapter = verified_qa_adapter(profile, candidate)
    device = 'cuda:0'
    dtype = resolve_dtype(torch, profile['dtype'])
    kwargs = {'local_files_only': True, 'torch_dtype': dtype, 'device_map': device}
    quant = profile.get('quantization')
    memory_options = profile.get('memory_optimizations', {})
    row_limit = memory_options.get('quantized_linear_microbatch_rows')
    deferred_scratch = memory_options.get('defer_exllama_scratch', False)
    if type(deferred_scratch) is not bool or (row_limit and deferred_scratch):
        raise ValueError('Deferred scratch and linear row microbatching are separate configurations')
    if row_limit is not None and (type(row_limit) is not int or row_limit < 1):
        raise ValueError('Quantized linear microbatch rows must be positive')
    if quant and quant['method'] == 'bitsandbytes':
        options = {'llm_int8_skip_modules': quant['skip_modules']}
        if quant['bits'] == 8:
            options['load_in_8bit'] = True
        else:
            options.update(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kwargs['quantization_config'] = BitsAndBytesConfig(**options)
    family = FAMILIES[candidate['model']]
    if profile['family'] != family:
        raise ValueError('Execution family does not match the pinned candidate')
    actual_rows = memory_options.get('actual_prefill_scratch_rows',False)
    if type(actual_rows) is not bool or actual_rows and (family != 'minicpm' or not deferred_scratch):
        raise ValueError('Actual prefill scratch requires the explicit MiniCPM deferred-scratch configuration')
    max_beams = max(adapter.generation_policy(task, candidate['model'])['num_beams']
                    for task in adapter.supported_tasks(candidate))
    if family == 'qwen':
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        processor = Qwen2_5OmniProcessor.from_pretrained(directory, local_files_only=True)
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            directory, attn_implementation=profile.get('attention_implementation', 'sdpa'),
            **({'enable_audio_output': False} if profile.get('load_audio_output') is False else {}), **kwargs)
        model.disable_talker()
    elif family == 'minicpm':
        from auto_gptq import AutoGPTQForCausalLM
        processor = AutoProcessor.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
        model = AutoGPTQForCausalLM.from_quantized(str(directory), device=device, torch_dtype=dtype,
                                                 trust_remote_code=True, local_files_only=True,
                                                 disable_exllamav2=True, low_cpu_mem_usage=True)
        # The official act-order checkpoint selects ExLlama v1. Its default
        # scratch length is insufficient for the full 1000-class prompt.
        if model.quantize_config.desc_act:
            from auto_gptq.utils.exllama_utils import exllama_set_max_input_length
            # This buffer bounds a linear kernel's rows, not the context length.
            # With row microbatching, all original input tokens still reach the
            # attention layers and their complete KV cache.
            model = exllama_set_max_input_length(model, 1 if deferred_scratch else row_limit or profile['max_input_length'] * max_beams)
    elif family == 'phi':
        processor = AutoProcessor.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(directory, trust_remote_code=True,
                                                     _attn_implementation='flash_attention_2', **kwargs)
    elif family == 'megrez':
        processor = AutoProcessor.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(directory, trust_remote_code=True,
                        attn_implementation=profile.get('attention_implementation', 'flash_attention_2'), **kwargs)
        # The pinned generate calls _get_or_init_processor internally. Bind the
        # same verified local processor and prohibit a second implicit hub load.
        model.processor = processor
    elif family == 'chatqa':
        processor = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            directory, attn_implementation=profile.get('attention_implementation', 'sdpa'), **kwargs)
    elif family == 'unifiedqa':
        processor = T5Tokenizer.from_pretrained(directory, local_files_only=True, legacy=True)
        model = T5ForConditionalGeneration.from_pretrained(directory, use_safetensors=False, **kwargs)
    elif family == 'siglip':
        processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
        model = AutoModel.from_pretrained(directory, **kwargs)
    elif family in {'deepseek', 'deepseek2'}:
        processor = AutoTokenizer.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
        model = AutoModel.from_pretrained(directory, trust_remote_code=True,
                                          _attn_implementation='flash_attention_2', use_safetensors=True, **kwargs)
    else:
        processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(directory, **kwargs)
    if trained_adapter is not None:
        from peft import PeftModel, get_peft_model_state_dict
        from safetensors import safe_open
        model = PeftModel.from_pretrained(model, trained_adapter['path'], adapter_name='qa_train_only',
                                           is_trainable=False, local_files_only=True)
        model.set_adapter('qa_train_only')
        loaded = get_peft_model_state_dict(model, adapter_name='qa_train_only')
        with safe_open(str(Path(trained_adapter['path']) / 'adapter_model.safetensors'), framework='pt', device='cpu') as source:
            if not loaded or set(loaded) != set(source.keys()):
                raise ValueError('Loaded QA adapter tensors do not match the pinned checkpoint')
            if any(list(value.shape) != source.get_slice(name).get_shape() for name, value in loaded.items()):
                raise ValueError('Loaded QA adapter tensor shapes changed')
        if 'qa_train_only' not in model.active_adapters:
            raise ValueError('The pinned QA adapter is not active')
        trained_adapter.update(loaded_tensor_count=len(loaded), active_adapters=list(model.active_adapters))
    model.eval()
    memory_audit = {'requested': copy.deepcopy(memory_options)}
    if trained_adapter is not None:
        memory_audit['trained_qa_adapter'] = trained_adapter
    if family == 'minicpm':
        inner = minicpm_native_model(model)
        if memory_options.get('retire_audio_output', False):
            memory_audit.update(retire_minicpm_audio_output(inner))
        if deferred_scratch:
            if not model.quantize_config.desc_act:
                raise ValueError('Deferred scratch requires the pinned act-order ExLlama checkpoint')
            memory_audit['deferred_exllama_scratch'] = {
                'active_rows': 1 if actual_rows else profile['max_input_length'] * max_beams,
                'row_bound_mode': 'actual_native_prefill_times_beams' if actual_rows else 'declared_input_budget_times_beams',
                'maximum_allowed_rows':profile['max_input_length'] * max_beams,
                'idle_rows': 1, 'requests': 0, 'releases': 0}
        if row_limit:
            memory_audit.update(configure_exllama_row_microbatch(inner, row_limit, torch))
            memory_audit['exllama_scratch_bytes'] = {
                str(device): {name: value.numel() * value.element_size()
                              for name, value in buffers.items() if hasattr(value, 'numel')}
                for device, buffers in model.device_to_buffers.items()}
        if memory_options.get('language_attention') == 'sdpa':
            memory_audit.update(configure_minicpm_language_sdpa(inner))
        if memory_options.get('vision_microbatch_size'):
            inner.config.vision_batch_size = memory_options['vision_microbatch_size']
            memory_audit['vision_microbatch_size'] = inner.config.vision_batch_size
    if family == 'phi' and memory_options.get('vision_microbatch_size'):
        memory_audit['vision_microbatch'] = install_phi_vision_microbatch(
            model.model.embed_tokens_extend.image_embed, memory_options['vision_microbatch_size'], torch)
    if any(p.device.type != 'cuda' for p in model.parameters()):
        raise RuntimeError('CPU/disk offload is outside the candidate contract')
    # Release loading/quantization staging allocations and the disabled Talker
    # cache before priming. Live parameters, caches and scratch buffers remain.
    torch.cuda.empty_cache()
    return NativeExecutor(adapter, candidate, model, processor, torch=torch, device=device, dtype=dtype,
                          output_root=output_root, input_limit=profile.get('max_input_length'),
                          memory_options=memory_options, memory_audit=memory_audit)


class ProfileAdapter:
    """Adapter for the existing measurement worker, using the same native executor."""
    def __init__(self, spec, resources):
        import json
        import torch
        import time

        self.torch = torch
        self.kind = 'compact'
        adapter = CompactAdapter(json.loads(Path(resources['protocol']).read_text()),
                                 json.loads(Path(resources['classmap']).read_text()))
        self.executor = load_native(adapter, spec['candidate'], spec['execution'], resources['checkpoint'],
                                    allow_gpu=True, output_root=resources['request_output'])
        self.model = self.executor.model
        self.cache_release_threshold_mib = spec['execution'].get('cache_release_threshold_mib')
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.parameter_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        torch.cuda.synchronize()

    def memory(self):
        t = self.torch
        return {'allocated_mib': t.cuda.memory_allocated() / 1024**2,
                'reserved_mib': t.cuda.memory_reserved() / 1024**2,
                'peak_allocated_mib': t.cuda.max_memory_allocated() / 1024**2,
                'peak_reserved_mib': t.cuda.max_memory_reserved() / 1024**2}

    def run(self, sample):
        import time
        self.torch.cuda.synchronize()
        start = time.perf_counter_ns()
        # Gold-reference edit distance is offline evaluation, not serving work.
        result = self.executor.infer(sample, evaluate_quality=False)
        self.torch.cuda.synchronize()
        from .compact_memory import reclaim_unused_cache
        result['cache_reclamation'] = reclaim_unused_cache(self.torch, self.cache_release_threshold_mib)
        self.torch.cuda.synchronize()
        result.update(service_ms=(time.perf_counter_ns() - start) / 1e6,
                      reference=sample.get('label'), answers=sample.get('answers'))
        result['memory_optimizations'] = copy.deepcopy(getattr(self.executor, 'memory_audit', {}))
        return result
