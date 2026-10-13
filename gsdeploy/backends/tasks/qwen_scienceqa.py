import os
import json
from pathlib import Path
from gsdeploy.backends.tasks import qwen_imagenette as image_base

PROMPT_PATH = Path(
    os.environ.get(
        "GS_PROMPT_FILE",
        str(Path(__file__).resolve().parents[3] / "configs/prompts.json"),
    )
)


def load_prompt():
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if "visual_qa" not in prompts:
        raise KeyError("visual_qa missing from task_prompts.json")
    return prompts["visual_qa"]


def option_letter(index):
    if not 0 <= index < 26:
        raise ValueError(f"Unsupported option index: {index}")
    return chr(ord("A") + index)


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


def build_instruction(task_prompt, sample):
    parts = [task_prompt.strip()]
    hint = (sample.get("hint") or "").strip()
    if hint:
        parts.extend(["", "Context:", hint])
    parts.extend(["", "Question:", sample["question"].strip(), "", "Choices:"])
    for i, choice in enumerate(sample["choices"]):
        parts.append(f"{option_letter(i)}. {str(choice).strip()}")
    return "\n".join(parts)
