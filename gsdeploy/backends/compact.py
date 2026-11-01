"""CPU-safe request/response adapters for the compact four-task study.

Builds native call blueprints without importing model libraries or loading any
weights. An executor must bind media objects, check token budgets and preserve
native generation IDs before these blueprints can become measured inference.
"""

import copy
from collections import Counter
import hashlib
import json
import math
import re

from ..tasks import parse_result, score_result


FAMILIES = {
    'Qwen/Qwen2.5-Omni-3B': 'qwen',
    'Qwen/Qwen2.5-Omni-7B': 'qwen',
    'Infinigence/Megrez-3B-Omni': 'megrez',
    'openbmb/MiniCPM-o-2_6-int4': 'minicpm',
    'microsoft/Phi-4-multimodal-instruct': 'phi',
    'allenai/unifiedqa-v2-t5-3b-1251000': 'unifiedqa',
    'nvidia/Llama3-ChatQA-1.5-8B': 'chatqa',
    'google/siglip2-giant-opt-patch16-384': 'siglip',
    'deepseek-ai/DeepSeek-OCR': 'deepseek',
    'deepseek-ai/DeepSeek-OCR-2': 'deepseek2',
    'openai/whisper-large-v3': 'whisper',
}
SPECIALIST_TASKS = {'unifiedqa': 'text', 'chatqa': 'text', 'siglip': 'image', 'deepseek': 'ocr', 'deepseek2': 'ocr', 'whisper': 'asr'}
TASKS = ('text', 'image', 'ocr', 'asr')
CLASS_NAME_DECODING = 'complete_class_name_token_trie_greedy_v1'
CLASS_NAME_DECODING_V2 = 'complete_class_name_token_trie_v2'
NATIVE_IMAGE_OBSERVATION = 'native_image_text_observation_v1'
NATIVE_DECODING_PRESETS = {
    'minicpm_deterministic_chat_v1': {'family': 'minicpm', 'num_beams': 3, 'repetition_penalty': 1.2},
    'megrez_deterministic_chat_v1': {'family': 'megrez', 'num_beams': 3, 'repetition_penalty': 1.2},
}
CHATQA_NATIVE_SYSTEM = ('This is a chat between a user and an artificial intelligence assistant. '
                        "The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. "
                        'The assistant should also indicate when the answer cannot be found in the context.')
DEEPSEEK_IMAGE_PRESETS = {
    'native_large_v1': {'base_size': 1280, 'image_size': 1280, 'crop_mode': False},
    'native_gundam_v1': {'base_size': 1024, 'image_size': 640, 'crop_mode': True},
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


class CompactAdapter:
    def __init__(self, policy, classmap):
        self.policy = copy.deepcopy(policy)
        self.classmap = copy.deepcopy(classmap)
        if policy.get('schema') != 'gsdeploy.compact_task_protocol.v1' or set(policy['tasks']) != set(TASKS):
            raise ValueError('Expected the compact four-task protocol')
        if policy.get('state') not in {'DRAFT_CPU_VALIDATION_ONLY_NOT_ADMITTED', 'CPU_PROTOCOL_FROZEN_GPU_UNVALIDATED'}:
            raise ValueError('CPU adapters cannot grant model admission')
        if policy['decoding'] != {'do_sample': False, 'num_beams': 1}:
            raise ValueError('The draft requires deterministic greedy decoding')
        if type(policy.get('diagnostic_only', False)) is not bool:
            raise ValueError('diagnostic_only must be a boolean')
        self.diagnostic_only = policy.get('diagnostic_only', False)
        self.phi_prompt_format = policy.get('phi_prompt_format', 'legacy_newlines_v1')
        if self.phi_prompt_format not in {'legacy_newlines_v1', 'native_compact_v1'}:
            raise ValueError('Unknown pinned Phi prompt format')
        image_decoding = policy['tasks']['image'].get('generative_decoding')
        if image_decoding not in {None, NATIVE_IMAGE_OBSERVATION, CLASS_NAME_DECODING, CLASS_NAME_DECODING_V2, 'complete_label_token_trie_greedy_v1'}:
            raise ValueError('Unknown image decoding mode')
        self.native_image_observation = image_decoding == NATIVE_IMAGE_OBSERVATION
        if self.native_image_observation and not self.diagnostic_only:
            raise ValueError('Native image observations require a diagnostic-only protocol')
        self.native_presets = policy.get('native_decoding_presets', {})
        if not isinstance(self.native_presets, dict):
            raise ValueError('Invalid native decoding presets')
        for model, tasks in self.native_presets.items():
            if model not in FAMILIES or not isinstance(tasks, dict) or set(tasks) - set(TASKS):
                raise ValueError('Unknown native decoding model/task')
            for preset in tasks.values():
                if not isinstance(preset, str) or preset not in NATIVE_DECODING_PRESETS or NATIVE_DECODING_PRESETS[preset]['family'] != FAMILIES[model]:
                    raise ValueError('Native decoding preset does not match the pinned model family')
        self.model_controls = policy.get('model_generation_controls', {})
        if not isinstance(self.model_controls, dict) or (self.model_controls and not self.diagnostic_only):
            raise ValueError('Model-specific decoding is allowed only in explicit diagnostic protocols')
        for model, tasks in self.model_controls.items():
            if model not in FAMILIES or not isinstance(tasks, dict) or set(tasks) - set(TASKS):
                raise ValueError('Unknown model or task in diagnostic decoding controls')
            for controls in tasks.values():
                self._validate_controls(controls, allow_beams=True)
        self.assistant_prefix = policy['tasks']['image'].get('assistant_prefix', '')
        if not isinstance(self.assistant_prefix, str) or len(self.assistant_prefix) > 128:
            raise ValueError('Invalid image answer prefix')
        if any('assistant_prefix' in spec for task, spec in policy['tasks'].items() if task != 'image'):
            raise ValueError('Answer-prefix binding is supported only for image classification')
        if set(classmap) != {str(i) for i in range(1000)}:
            raise ValueError('The complete 1000-label catalog is required')
        self.class_ids = []
        for i in range(1000):
            entry = classmap[str(i)]
            if len(entry) != 2 or not re.fullmatch(r'n[0-9]{8}', entry[0]) or not entry[1].strip():
                raise ValueError('Invalid image class catalog entry')
            self.class_ids.append(entry[0])
        if len(set(self.class_ids)) != 1000:
            raise ValueError('Duplicate image class IDs')
        self.policy_sha256 = fingerprint(self.policy)
        self.classmap_sha256 = fingerprint(self.classmap)
        names = [' '.join(self.classmap[str(i)][1].replace('_', ' ').split()) for i in range(1000)]
        counts = Counter(name.casefold() for name in names)
        aliases = [f'{name} [{class_id}]' if counts[name.casefold()] > 1 else name
                   for name, class_id in zip(names, self.class_ids)]
        self.class_name_decoding = image_decoding in {CLASS_NAME_DECODING, CLASS_NAME_DECODING_V2}
        self.class_label_surface = policy['tasks']['image'].get('class_label_surface', 'bare_v1')
        if self.class_label_surface not in {'bare_v1', 'leading_space_v1'}:
            raise ValueError('Unknown canonical class label token surface')
        if self.class_label_surface != 'bare_v1' and not self.class_name_decoding:
            raise ValueError('Leading-space labels require complete canonical class-name decoding')
        self.catalog_presentation = policy['tasks']['image'].get('catalog_presentation', 'prompt_and_decoder')
        if self.catalog_presentation not in {'prompt_and_decoder', 'decoder_only'}:
            raise ValueError('Unknown class catalog presentation')
        if self.catalog_presentation == 'decoder_only' and not (self.class_name_decoding or self.native_image_observation):
            raise ValueError('Decoder-only catalog requires complete canonical class-name decoding')
        if self.native_image_observation and self.catalog_presentation != 'decoder_only':
            raise ValueError('Native image observations must omit the classification catalog prompt')
        self.image_output_labels = ([((' ' if self.class_label_surface == 'leading_space_v1' else '') + name)
                                     for name in aliases] if self.class_name_decoding else self.class_ids)
        self.class_name_to_id = dict(zip(aliases, self.class_ids))
        self.catalog = '\n'.join(f'{class_id}: {alias if self.class_name_decoding else name}'
                                 for class_id, alias, name in zip(self.class_ids, aliases, names))
        for task in TASKS:
            self.generation_policy(task)

    @staticmethod
    def _validate_controls(controls, *, allow_beams=False):
        allowed = {'no_repeat_ngram_size', 'repetition_penalty'} | ({'num_beams'} if allow_beams else set())
        if not isinstance(controls, dict) or set(controls) - allowed:
            raise ValueError('Unsupported declared task generation controls')
        if 'no_repeat_ngram_size' in controls:
            value = controls['no_repeat_ngram_size']
            if type(value) is not int or not 0 <= value <= 128:
                raise ValueError('Invalid declared no-repeat ngram size')
        if 'repetition_penalty' in controls:
            value = controls['repetition_penalty']
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 2:
                raise ValueError('Invalid declared repetition penalty')
        if 'num_beams' in controls and (type(controls['num_beams']) is not int or not 1 <= controls['num_beams'] <= 8):
            raise ValueError('Invalid diagnostic beam count')

    def generation_policy(self, task, model=None):
        spec = self.policy['tasks'][task]
        controls = spec.get('generation_controls', {})
        self._validate_controls(controls, allow_beams=task == 'image' and spec.get('generative_decoding') == CLASS_NAME_DECODING_V2)
        overrides = self.model_controls.get(model, {}).get(task, {})
        preset_name = self.native_presets.get(model, {}).get(task)
        preset = {k: v for k, v in NATIVE_DECODING_PRESETS.get(preset_name, {}).items() if k != 'family'}
        return {**self.policy['decoding'], **preset, 'max_new_tokens': spec['max_new_tokens'], **controls, **overrides}

    def supported_tasks(self, candidate):
        family = FAMILIES.get(candidate['model'])
        if family is None or not re.fullmatch(r'[0-9a-f]{40}', candidate.get('revision', '')):
            raise ValueError('Unknown candidate or unpinned model revision')
        if 'adapter_sha256' in candidate and (family != 'chatqa' or not isinstance(candidate['adapter_sha256'], str)
                or re.fullmatch(r'[0-9a-f]{64}', candidate['adapter_sha256']) is None):
            raise ValueError('Only an explicitly pinned derived QA adapter candidate is supported')
        if family in SPECIALIST_TASKS:
            return (SPECIALIST_TASKS[family],)
        return TASKS

    def build(self, candidate, sample):
        task = sample['task']
        if task not in self.supported_tasks(candidate):
            raise ValueError('Candidate does not support this study task')
        spec = self.policy['tasks'][task]
        datasets = spec.get('datasets', [spec.get('dataset')])
        if sample.get('dataset') not in datasets:
            raise ValueError('Wrong task dataset; historical classifier inputs are incompatible')
        if not isinstance(sample.get('id'), str) or not sample['id']:
            raise ValueError('A sample ID is required')
        media = None
        if task == 'text':
            for key in ['context', 'question']:
                if not isinstance(sample.get(key), str) or not sample[key].strip():
                    raise ValueError('QA requires the full context and question')
            instruction = (spec['instruction'] + '\n\nCONTEXT:\n' + sample['context']
                           + '\n\nQUESTION:\n' + sample['question'])
        else:
            if not isinstance(sample.get('input'), str) or not sample['input']:
                raise ValueError('A local media path is required')
            if not re.fullmatch(r'[0-9a-f]{64}', sample.get('input_sha256', '')):
                raise ValueError('A frozen media hash is required')
            media = {'path': sample['input'], 'sha256': sample['input_sha256'],
                     'kind': 'audio' if task == 'asr' else 'image'}
            instruction = spec['instruction']
            if task == 'image' and self.catalog_presentation == 'prompt_and_decoder':
                instruction += '\n\nLABEL CATALOG:\n' + self.catalog
            elif task == 'asr':
                duration = sample.get('duration_s')
                low, high = spec['duration_range_seconds']
                if not isinstance(duration, (int, float)) or not math.isfinite(duration) or not low <= duration < high:
                    raise ValueError('ASR draft accepts complete 4-to-30-second recordings only')
                media.update(sampling_rate=spec['sampling_rate'], duration_s=duration, resample=False)
        # References, entities, answerability and correctness never enter this dict.
        request = {'schema': 'gsdeploy.compact_request.v1', 'sample_id': sample['id'], 'task': task,
                   'model': candidate['model'], 'revision': candidate['revision'],
                   'precision': candidate['precision'], 'protocol_sha256': self.policy_sha256,
                   'instruction': instruction, 'media': media,
                   'generation': self.generation_policy(task, candidate['model']),
                   'input_truncation': False}
        if 'adapter_sha256' in candidate:
            request['adapter_sha256'] = candidate['adapter_sha256']
        if task == 'image':
            request['classmap_sha256'] = self.classmap_sha256
            if (self.policy['tasks']['image'].get('generative_decoding') in {
                    CLASS_NAME_DECODING, 'complete_label_token_trie_greedy_v1'}
                    and FAMILIES[candidate['model']] != 'siglip' and request['generation']['num_beams'] != 1):
                raise ValueError('Legacy greedy class mode requires one beam; use the versioned v2 class mode')
        request['native'] = self._native(FAMILIES[candidate['model']], task, sample, request)
        if task == 'image' and self.assistant_prefix and FAMILIES[candidate['model']] in {'qwen', 'phi', 'minicpm', 'megrez'}:
            request['native']['assistant_prefix'] = self.assistant_prefix
            if FAMILIES[candidate['model']] == 'phi':
                request['native']['prompt'] += self.assistant_prefix
        return request

    def _native(self, family, task, sample, request):
        instruction, media = request['instruction'], request['media']
        if family == 'qwen':
            limits = self.policy.get('native_input_limits', {})
            content = [{'type': 'text', 'text': instruction}]
            if media:
                content.insert(0, {'type': media['kind'], media['kind']: media['path']})
            return {'api': 'processor.apply_chat_template + processor + model.generate',
                    'messages': [{'role': 'system', 'content': [{'type': 'text', 'text': 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'}]},
                                 {'role': 'user', 'content': content}],
                    'processor_options': {'padding': True, 'truncation': False, 'return_tensors': 'pt',
                                          **({'min_pixels': limits['qwen_image_min_pixels'],
                                              'max_pixels': limits['qwen_image_max_pixels']} if media and task != 'asr' and limits else {}),
                                          **({'audio_kwargs': {'max_length': limits['qwen_audio_max_samples']}} if task == 'asr' and 'qwen_audio_max_samples' in limits else {})},
                    'generate_options': {'return_audio': False, 'use_audio_in_video': False},
                    'bindings': 'Load local RGB image or mono 16kHz array; bind images/audio to processor.',
                    'completion_source': 'generation IDs after the input prefix', 'disable_talker': True}
        if family == 'megrez':
            budget = self.policy.get('megrez_context_budget', 'strict_output_reservation_v1')
            if budget not in {'strict_output_reservation_v1', 'full_input_native_context_v1'}:
                raise ValueError('Unknown pinned Megrez context budget policy')
            content = {'text': instruction}
            if media:
                content[media['kind']] = {'$media': media['kind']}
            return {'api': 'native processor + model.generate',
                    'messages': [{'role': 'user', 'content': content}],
                    'context_budget': budget,
                    'bindings': 'Bind verified RGB PIL image or original verified audio bytes; disable native hidden text/audio truncation.',
                    'completion_source': 'native llm.generate IDs from inputs_embeds, excluding zero input-ID prefix',
                    'runtime_checks': ['Complete native multimodal input must fit context',
                                       'Natural context exhaustion is an invalid output; only observed EOS is complete']}
        if family == 'phi':
            tag = '<|audio_1|>' if task == 'asr' else '<|image_1|>' if media else ''
            if self.phi_prompt_format == 'native_compact_v1':
                prompt = '<|user|>' + tag + instruction + '<|end|><|assistant|>'
            else:
                prefix = tag + '\n' if tag else ''
                prompt = '<|user|>\n' + prefix + instruction + '<|end|>\n<|assistant|>\n'
            return {'api': 'processor + model.generate',
                    'prompt': prompt, 'prompt_format': self.phi_prompt_format,
                    'processor_options': {'truncation': False, 'return_tensors': 'pt'},
                    'bindings': 'Bind RGB images or audios=[(mono_array, 16000)].',
                    'completion_source': 'generation IDs after the input prefix'}
        if family == 'minicpm':
            content = [instruction] if media is None else [{'$media': media['kind']}, instruction]
            return {'api': 'model.chat', 'msgs': [{'role': 'user', 'content': content}],
                    'options': {'sampling': False, 'num_beams': request['generation']['num_beams'],
                                'repetition_penalty': request['generation'].get('repetition_penalty', 1.0),
                                'use_tts_template': task == 'asr', 'generate_audio': False,
                                'stream': False, 'max_inp_length': 32768},
                    'bindings': 'Replace $media with local PIL RGB image or mono 16kHz ndarray; pass pinned tokenizer/processor.',
                    'completion_source': 'capture llm.generate sequences; chat string/OmniOutput does not retain EOS',
                    'runtime_checks': ['No native processor input truncation', 'Effective do_sample=False',
                                       'Audio generation disabled; loaded TTS parameters count toward residency']}
        if family == 'chatqa':
            # The pinned ChatQA model uses plain role headers, not Llama's
            # generic chat template. Keep the common task instruction and the
            # complete, case-preserving context; references never enter it.
            spec = self.policy['tasks']['text']
            formatting = spec.get('chatqa_prompt_format', 'task_system_v1')
            if formatting not in {'task_system_v1', 'native_system_v2'}:
                raise ValueError('Unknown pinned ChatQA prompt format')
            system = CHATQA_NATIVE_SYSTEM if formatting == 'native_system_v2' else spec['instruction']
            user = (spec['instruction'] + ' ' if formatting == 'native_system_v2' else '') + sample['question']
            return {'api': 'AutoTokenizer + AutoModelForCausalLM.generate',
                    'format': formatting,
                    'text': 'System: ' + system + '\n\n' + sample['context']
                            + '\n\nUser: ' + user + '\n\nAssistant:',
                    'tokenizer_options': {'truncation': False, 'add_special_tokens': False, 'return_tensors': 'pt'},
                    'prepend_bos': True, 'terminator_tokens': ['<|end_of_text|>', '<|eot_id|>'],
                    'completion_source': 'generation IDs after the complete input prefix'}
        if family == 'unifiedqa':
            return {'api': 'T5Tokenizer + T5ForConditionalGeneration.generate',
                    'text': sample['question'].lower() + ' \\n ' + sample['context'].lower(),
                    'tokenizer_options': {'truncation': False, 'return_tensors': 'pt'},
                    'max_input_tokens': self.policy['tasks']['text']['t5_max_input_tokens'],
                    'overflow': 'reject_and_record_failure; never silently remove context',
                    'completion_source': 'decoder generation IDs excluding decoder-start prefix'}
        if family == 'siglip':
            spec = self.policy['tasks']['image']
            texts = [spec['class_prompt'].format(name=self.classmap[str(i)][1].replace('_', ' ')) for i in range(1000)]
            return {'api': 'AutoModel.get_text_features + get_image_features',
                    'class_texts': texts, 'class_ids': self.class_ids,
                    'text_processor_options': {'padding': 'max_length', 'max_length': spec['siglip_text_max_tokens'],
                                               'truncation': False, 'return_tensors': 'pt'},
                    'image_resolution': spec['image_resolution'],
                    'cache_key': fingerprint({'model': request['model'], 'revision': request['revision'],
                                              'precision': request['precision'], 'protocol': self.policy_sha256,
                                              'classmap': self.classmap_sha256, 'texts': texts}),
                    'scoring': 'L2-normalize features, argmax over all 1000 text features',
                    'completion_source': 'classified only after all 1000 logits are finite',
                    'runtime_checks': ['AutoConfig uses stored model_type=siglip; do not force Siglip2Model',
                                       'Text features include every label without truncation']}
        if family in {'deepseek', 'deepseek2'}:
            spec = self.policy['tasks']['ocr']
            if family == 'deepseek2':
                if spec.get('deepseek2_image_preset') != 'native_dynamic_v1':
                    raise ValueError('DeepSeek OCR2 requires its explicit pinned native optical preset')
                imaging = {'base_size': 1024, 'image_size': 768, 'crop_mode': True}
            else:
                preset = spec.get('deepseek_image_preset')
                if preset is not None and preset not in DEEPSEEK_IMAGE_PRESETS:
                    raise ValueError('Unknown pinned DeepSeek image preset')
                imaging = (DEEPSEEK_IMAGE_PRESETS[preset] if preset else
                           {key: spec[key] for key in ['base_size', 'image_size', 'crop_mode']})
            if not self.diagnostic_only and spec['deepseek_prompt'] != '<image>\nFree OCR.':
                raise ValueError('Unvalidated DeepSeek prompt is diagnostic-only; formal execution uses the pinned native Free OCR prompt')
            if request['generation']['max_new_tokens'] != 8192:
                raise ValueError('Pinned DeepSeek infer hardcodes 8192; requires a native change for another budget')
            return {'api': 'model.infer',
                    'options': {'prompt': spec['deepseek_prompt'], 'image_file': media['path'],
                                **imaging, 'eval_mode': True,
                                'save_results': False, 'test_compress': False},
                    'bindings': 'Use a fresh per-request output_path outside the source/checkpoint trees.',
                    'completion_source': 'capture model.generate IDs; eval_mode strips EOS from returned text',
                    'runtime_checks': ['Record native no_repeat_ngram_size=35', 'Effective deterministic greedy generation',
                                       'No hidden stopping or lost EOS metadata']}
        return {'api': 'processor + WhisperForConditionalGeneration.generate',
                'processor_options': {'sampling_rate': 16000, 'truncation': False, 'padding': 'max_length',
                                      'max_length': 480000,
                                      'return_attention_mask': True, 'return_tensors': 'pt'},
                'generate_options': {'language': 'en', 'task': 'transcribe', 'return_timestamps': False,
                                     'return_dict_in_generate': True, 'force_unique_generate_call': True},
                'completion_source': 'decoder generation IDs excluding observed forced-language/task prefix',
                'runtime_checks': ['Decoder prefix plus new tokens fits max_target_positions',
                                   'No forced translation or cropped audio']}

    def finish(self, request, sample, raw_output, *, evaluate_quality=True, **observation):
        if request['sample_id'] != sample['id'] or request['task'] != sample['task']:
            raise ValueError('Request/sample identity mismatch')
        if request['protocol_sha256'] != self.policy_sha256:
            raise ValueError('Request protocol changed')
        if request['generation'] != self.generation_policy(sample['task'], request['model']):
            raise ValueError('Request generation policy changed')
        classified = observation.get('finish_reason') == 'classified'
        if classified != (FAMILIES[request['model']] == 'siglip'):
            raise ValueError('Completion kind does not match native model family')
        parsed_output = raw_output
        if sample['task'] == 'image' and self.class_name_decoding and not classified:
            # Exact reversible catalog lookup; never fuzzy-match a prediction
            # or use the sample's reference class to resolve ambiguity.
            parsed_output = self.class_name_to_id.get(raw_output.strip(), '')
        result = parse_result(sample['task'], parsed_output,
                              max_new_tokens=request['generation']['max_new_tokens'],
                              class_ids=self.class_ids,
                              diagnostic_only=sample['task'] == 'image' and self.native_image_observation and not classified,
                              **observation)
        result['raw_output'] = raw_output
        if (sample['task'] == 'text' and self.policy['tasks']['text'].get('no_answer_normalization')
                == 'optional_brackets_terminal_punctuation_v2'
                and re.fullmatch(r'<?\s*no\s+answer\s*>?[.!]?', raw_output.strip(), flags=re.I)):
            result['prediction'] = ''
        result.update(sample_id=sample['id'], protocol_sha256=self.policy_sha256,
                      request_sha256=fingerprint(request))
        if evaluate_quality and result.get('quality_eligible', True):
            result['score'] = score_result(sample, result)
        return result
