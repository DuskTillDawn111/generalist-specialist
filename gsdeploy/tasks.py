"""Shared CPU task scoring for the compact study, independent of model loading.

These transformations do not repair missing source annotations and grant no
quality admission. Raw predictions and original references must be retained.
"""

from html.parser import HTMLParser
import re
import unicodedata


OCR_PROTOCOL = "page_ocr_text_v4_cpu_frozen"


def parse_sroie_annotation(value):
    """Read native Task 1/2 lines; only the first eight commas delimit geometry."""
    rows = []
    for number, line in enumerate(value.lstrip("\ufeff").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(",", 8)
        if len(fields) != 9:
            raise ValueError(f"SROIE line {number}: expected eight coordinates and text")
        try:
            coordinates = [int(field.strip()) for field in fields[:8]]
        except ValueError as error:
            raise ValueError(f"SROIE line {number}: invalid coordinates") from error
        transcript = fields[8].strip()
        if not transcript:
            raise ValueError(f"SROIE line {number}: empty transcript")
        rows.append({"polygon": coordinates, "text": transcript,
                     "source_line": number, "ignored": transcript == "###"})
    if not rows:
        raise ValueError("Empty SROIE annotation")
    return rows


def restore_sroie_reference(source_row, annotation):
    """Require source geometry and all surviving text prefixes to agree."""
    native = parse_sroie_annotation(annotation)
    boxes = [[row["polygon"][i] for i in [0, 1, 4, 5]] for row in native]
    if boxes != source_row["bboxes"]:
        raise ValueError("Native/source geometry mismatch")
    if len(native) != len(source_row["words"]) or any(
            row["text"].split(",", 1)[0] != old.strip()
            for row, old in zip(native, source_row["words"])):
        raise ValueError("Native/source text prefix mismatch")
    if any(row["ignored"] for row in native):
        raise ValueError("Ignored-region page requires a masking policy")
    return " ".join(row["text"] for row in native)


class _VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"td", "th", "tr", "br", "p", "div", "table", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        if tag in {"td", "th", "tr", "p", "div", "table", "li"}:
            self.parts.append(" ")

    def handle_data(self, value):
        if not self.hidden:
            self.parts.append(value)


def page_ocr_text(value):
    """Frozen visible-text CER view; preserve punctuation, casefold and NFKC.

    HTML cell boundaries and explicit Markdown table syntax become whitespace.
    Text, including commas, currency, decimal points and negative signs, stays.
    YAML front matter and explanatory prose are deliberately not guessed away.
    """
    if re.search(r"</?(?:table|tr|td|th|p|div|br|html|body)\b", value, re.I):
        parser = _VisibleHTML()
        parser.feed(value)
        parser.close()
        value = "".join(parser.parts)
    lines = value.splitlines()
    def table_separator(line):
        if '|' not in line:
            return False
        cells = line.strip().strip('|').split('|')
        return bool(cells) and all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells)
    separators = {i for i, line in enumerate(lines) if table_separator(line)}
    table_lines = set()
    for i in separators:
        if i:
            table_lines.add(i - 1)
        j = i + 1
        while j < len(lines) and "|" in lines[j] and lines[j].strip():
            table_lines.add(j)
            j += 1
    value = " ".join(line.replace("|", " ") if i in table_lines else line
                     for i, line in enumerate(lines) if i not in separators)
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def qa_normalize(value):
    """SQuAD-style lowercase, ASCII punctuation/articles and whitespace rules."""
    import string

    value = value.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', value).split())


def asr_normalize(value):
    """English lexical WER: retain apostrophes, split other punctuation, no number expansion."""
    value = unicodedata.normalize('NFKC', value).casefold().replace('’', "'")
    value = re.sub(r"[^a-z0-9'\s]", ' ', value)
    value = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", ' ', value)
    return ' '.join(value.split())


def distance(left, right):
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        pass
    else:
        return int(Levenshtein.distance(left, right))
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def parse_result(task, raw_output, *, finish_reason, generated_tokens, max_new_tokens,
                 terminal_token_id=None, eos_token_ids=(), class_ids=(), class_scores=None,
                 diagnostic_only=False):
    """Require observed termination; never infer EOS from a plausible string.

    Native wrappers must obtain EOS metadata from generation IDs. A native chat
    string without this metadata is unknown, even when it matches the reference.
    """
    if task not in {'text', 'image', 'ocr', 'asr'} or not isinstance(raw_output, str):
        raise ValueError('Invalid task or non-text output')
    if type(diagnostic_only) is not bool or (diagnostic_only and task != 'image'):
        raise ValueError('Free-text observations are image diagnostics only')
    if diagnostic_only and finish_reason == 'classified':
        raise ValueError('A native text observation is not feature classification')
    if finish_reason not in {'eos', 'length', 'unknown', 'error', 'classified'}:
        raise ValueError('Unknown finish reason')
    if generated_tokens is not None and (type(generated_tokens) is not int or generated_tokens < 0):
        raise ValueError('Invalid generated token count')
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError('Invalid output budget')
    reason = None
    if finish_reason == 'classified':
        if task != 'image' or generated_tokens != 0:
            raise ValueError('Feature classification is only valid for image outputs')
        import math
        if (len(class_ids) != 1000 or class_scores is None or len(class_scores) != 1000
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in class_scores)):
            reason = 'MISSING_OR_INVALID_1000_CLASS_SCORES'
        elif raw_output.strip() != class_ids[max(range(1000), key=lambda i: class_scores[i])]:
            reason = 'CLASS_ARGMAX_MISMATCH'
    elif finish_reason == 'eos':
        if (generated_tokens is None or generated_tokens < 1
                or type(terminal_token_id) is not int or terminal_token_id < 0
                or not eos_token_ids or any(type(t) is not int or t < 0 for t in eos_token_ids)
                or terminal_token_id not in eos_token_ids):
            reason = 'MISSING_OR_INCONSISTENT_EOS_EVIDENCE'
    else:
        reason = 'GENERATION_' + finish_reason.upper()
    if generated_tokens is not None and generated_tokens > max_new_tokens:
        reason = 'OUTPUT_BUDGET_EXCEEDED'
    prediction = raw_output.strip()
    if task == 'text':
        if prediction.casefold() in {'<no answer>', 'no answer>'}:
            prediction = ''
    elif task == 'image':
        if diagnostic_only:
            if not prediction:
                reason = reason or 'EMPTY_OBSERVATION'
        elif not re.fullmatch(r'n[0-9]{8}', prediction) or prediction not in class_ids:
            reason = reason or 'INVALID_CLASS_OUTPUT'
    elif task == 'ocr':
        prediction = page_ocr_text(raw_output)
        if not prediction:
            reason = reason or 'EMPTY_TRANSCRIPTION'
    else:
        prediction = asr_normalize(raw_output)
        if not prediction:
            reason = reason or 'EMPTY_TRANSCRIPTION'
    result = {'task': task, 'raw_output': raw_output, 'prediction': prediction,
            'finish_reason': finish_reason, 'generated_tokens': generated_tokens,
            'terminal_token_id': terminal_token_id, 'eos_token_ids': list(eos_token_ids),
            'success': reason is None, 'failure_reason': reason,
            'truncated': True if finish_reason == 'length' else False if reason is None else None}
    if diagnostic_only:
        result.update(diagnostic_only=True, quality_eligible=False)
    return result


def score_result(sample, result):
    """Score after inference; no reference or correctness enters a model request."""
    from collections import Counter

    if result.get('diagnostic_only') or result.get('quality_eligible') is False:
        raise ValueError('Native observations cannot be used as quality/admission evidence')

    task = sample['task']
    if task != result['task']:
        raise ValueError('Task identity mismatch')
    prediction, success = result['prediction'], result['success']
    if task == 'text':
        prediction = qa_normalize(prediction)
        values = []
        for reference in sample['answers'] or ['']:
            reference = qa_normalize(reference)
            a, b = prediction.split(), reference.split()
            overlap = sum((Counter(a) & Counter(b)).values())
            f1 = float(a == b) if not a or not b else 2 * overlap / (len(a) + len(b))
            values.append((float(prediction == reference), f1))
        return {'EM': max(v[0] for v in values) if success else 0.0,
                'F1': max(v[1] for v in values) if success else 0.0,
                'answerable': bool(sample['answers'])}
    if task == 'image':
        return {'correct': bool(success and prediction == sample['label'])}
    reference = page_ocr_text(sample['label']) if task == 'ocr' else asr_normalize(sample['label']).split()
    prediction = prediction if task == 'ocr' else prediction.split()
    if not reference:
        raise ValueError('Empty OCR/ASR reference cannot define an error-rate denominator')
    errors = distance(prediction if success else type(reference)(), reference)
    return {'errors': errors, 'reference_units': len(reference),
            'unit': 'character' if task == 'ocr' else 'word'}


def summarize_quality(samples, results):
    """One explicit quality round; duplicates and incomplete cohorts are errors."""
    samples, results = list(samples), list(results)
    if not samples:
        raise ValueError('Empty quality cohort')
    task = samples[0]['task']
    if any(s['task'] != task for s in samples):
        raise ValueError('Quality cohorts must contain exactly one task')
    ids = [s['id'] for s in samples]
    result_ids = [r['sample_id'] for r in results]
    if len(set(ids)) != len(ids) or len(set(result_ids)) != len(result_ids):
        raise ValueError('Repeated requests do not expand independent quality samples; select a round')
    if set(ids) != set(result_ids):
        raise ValueError('Missing or unexpected quality records')
    by_id = dict(zip(result_ids, results))
    scores = [score_result(s, by_id[s['id']]) for s in samples]
    report = {'task': task, 'unique_samples': len(samples),
              'failed_samples': sum(not r['success'] for r in results),
              'scope': 'quality_statistics_no_admission_by_itself'}
    if task == 'text':
        for key in ['EM', 'F1']:
            report[key] = sum(s[key] for s in scores) / len(scores)
        for answerable, name in [(True, 'answerable'), (False, 'unanswerable')]:
            group = [s for s in scores if s['answerable'] == answerable]
            report[name] = {'samples': len(group), 'EM': sum(s['EM'] for s in group) / len(group) if group else None}
    elif task == 'image':
        report['top1_accuracy'] = sum(s['correct'] for s in scores) / len(scores)
    else:
        errors = sum(s['errors'] for s in scores)
        total = sum(s['reference_units'] for s in scores)
        report.update(errors=errors, reference_units=total)
        report['CER' if task == 'ocr' else 'WER'] = errors / total
    return report
