def option_letter(index):
    if not 0 <= index < 26:
        raise ValueError(f"Unsupported option index: {index}")
    return chr(ord("A") + index)


def valid_options(sample):
    choices = sample["choices"]
    return {option_letter(i) for i in range(len(choices))}


def normalize_option(text):
    """
    Strict closed-set normalization.

    Only whitespace/case are normalized.

    Examples:
      "B"   -> "B"
      " b " -> "B"

    Outputs such as:
      "B."
      "(B)"
      "Answer: B"
      "The answer is B"

    remain invalid closed-set outputs.
    """
    return str(text).strip().upper()


def validate_sample(sample):
    required = [
        "id",
        "task",
        "dataset",
        "split",
        "input",
        "question",
        "choices",
        "label_id",
        "label",
    ]
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(f"{sample.get('id')}: missing keys {missing}")
    if sample["task"] != "visual_qa":
        raise ValueError(f"{sample['id']}: unexpected task={sample['task']}")
    choices = sample["choices"]
    if not isinstance(choices, list):
        raise TypeError(f"{sample['id']}: choices must be list")
    if not 2 <= len(choices) <= 26:
        raise ValueError(f"{sample['id']}: invalid choices count={len(choices)}")
    label_id = int(sample["label_id"])
    if not 0 <= label_id < len(choices):
        raise ValueError(f"{sample['id']}: invalid label_id={label_id}")
    expected = option_letter(label_id)
    actual = normalize_option(sample["label"])
    if actual != expected:
        raise ValueError(
            f"{sample['id']}: label mismatch: label_id={label_id}, label={actual}, expected={expected}"
        )


def build_instruction(task_prompt, sample):
    parts = [task_prompt.strip()]
    hint = (sample.get("hint") or "").strip()
    if hint:
        parts.extend(["", "Context:", hint])
    parts.extend(["", "Question:", sample["question"].strip(), "", "Choices:"])
    for i, choice in enumerate(sample["choices"]):
        parts.append(f"{option_letter(i)}. {str(choice).strip()}")
    return "\n".join(parts)
