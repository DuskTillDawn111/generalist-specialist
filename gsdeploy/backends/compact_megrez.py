"""Pinned Megrez native preprocessing with complete inputs and explicit bounds."""


class CompleteTokenizer:
    def __init__(self, tokenizer, answer_prefix):
        self.tokenizer = tokenizer
        self.answer_prefix = answer_prefix
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, text, **kwargs):
        if not isinstance(text, str) or kwargs.get('max_length') is not None:
            raise ValueError('Megrez requires one complete native prompt without a length limit')
        kwargs['truncation'] = False
        self.calls += 1
        return self.tokenizer(text + self.answer_prefix, **kwargs)


class CompleteAudioExtractor:
    def __init__(self, extractor):
        self.extractor = extractor

    def __getattr__(self, name):
        return getattr(self.extractor, name)

    def __call__(self, *args, **kwargs):
        kwargs['truncation'] = False
        return self.extractor(*args, **kwargs)


def complete_megrez_inputs(processor, messages, answer_prefix=''):
    # Change only the hidden truncation flags. All native optical slices,
    # audio features, role templates and multimodal placeholder positions remain.
    tokenizer, extractor = processor.tokenizer, processor.audio_feature_extractor
    proxy = CompleteTokenizer(tokenizer, answer_prefix)
    processor.tokenizer = proxy
    processor.audio_feature_extractor = CompleteAudioExtractor(extractor)
    try:
        # Native data_collator silently slices at its default 4096, separately
        # from tokenizer truncation. Collate the full merged example explicitly;
        # the executor then rejects overflow before any model inference.
        merged = processor(messages, apply_data_collator=False)
        if proxy.calls != 1:
            raise ValueError('Expected exactly one complete native prompt tokenization')
        full_length = int(merged['input_ids'].shape[-1])
        inputs = processor.data_collator([merged], max_length=full_length)
        if int(inputs['input_ids'].shape[-1]) != full_length:
            raise ValueError('Native Megrez collator changed the complete prompt length')
        return inputs
    finally:
        processor.tokenizer, processor.audio_feature_extractor = tokenizer, extractor


def native_output_budget(input_tokens, context_tokens, requested, policy):
    if any(type(x) is not int or x <= 0 for x in [input_tokens, context_tokens, requested]):
        raise ValueError('Invalid complete input or native output bounds')
    remaining = context_tokens - input_tokens
    if remaining <= 0:
        raise ValueError('Complete Megrez input exceeds native context; truncation prohibited')
    if policy == 'strict_output_reservation_v1':
        if requested > remaining:
            raise ValueError('Complete input plus output reservation exceeds native context')
        return requested
    if policy == 'full_input_native_context_v1':
        # A native context limit never converts exhaustion into a valid answer.
        # Observation must report real EOS; a length stop remains a failure.
        return min(requested, remaining)
    raise ValueError('Unknown native context policy')
