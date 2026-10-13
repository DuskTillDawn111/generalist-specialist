import re


def normalize_closed_label(text):
    return str(text).strip().lower()


def normalize_ocr(text):
    text = str(text).lower()
    return re.sub(r"[^a-z0-9]", "", text)


def normalize_asr(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def levenshtein(a, b):

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, x in enumerate(a, 1):

        current = [i]

        for j, y in enumerate(b, 1):

            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (x != y)

            current.append(
                min(insert, delete, replace)
            )

        previous = current

    return previous[-1]


def classification_accuracy(predictions, references):

    if len(predictions) != len(references):
        raise ValueError("Length mismatch")

    correct = sum(
        normalize_closed_label(p)
        == normalize_closed_label(r)
        for p, r in zip(predictions, references)
    )

    return correct / len(references)


def ocr_exact_match(predictions, references):

    if len(predictions) != len(references):
        raise ValueError("Length mismatch")

    correct = sum(
        normalize_ocr(p) == normalize_ocr(r)
        for p, r in zip(predictions, references)
    )

    return correct / len(references)


def cer(predictions, references):

    total_distance = 0
    total_chars = 0

    for pred, ref in zip(predictions, references):

        p = normalize_ocr(pred)
        r = normalize_ocr(ref)

        total_distance += levenshtein(
            list(p),
            list(r)
        )

        total_chars += len(r)

    return (
        total_distance / total_chars
        if total_chars > 0
        else 0.0
    )


def wer(predictions, references):

    total_distance = 0
    total_words = 0

    for pred, ref in zip(predictions, references):

        p = normalize_asr(pred).split()
        r = normalize_asr(ref).split()

        total_distance += levenshtein(p, r)
        total_words += len(r)

    return (
        total_distance / total_words
        if total_words > 0
        else 0.0
    )


if __name__ == "__main__":

    print(
        "Classification:",
        classification_accuracy(
            ["positive", "negative"],
            ["positive", "positive"]
        )
    )

    print(
        "OCR exact:",
        ocr_exact_match(
            ["HELLO", "world!"],
            ["hello", "world"]
        )
    )

    print(
        "CER:",
        cer(
            ["hello", "worxd"],
            ["hello", "world"]
        )
    )

    print(
        "WER:",
        wer(
            ["this is test"],
            ["this is a test"]
        )
    )
