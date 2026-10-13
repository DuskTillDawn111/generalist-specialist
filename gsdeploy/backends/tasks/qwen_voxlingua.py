import os
import json
from pathlib import Path
from gsdeploy.backends.tasks import qwen_librispeech as audio_base

PROMPT_FILE = Path(
    os.environ.get(
        "GS_PROMPT_FILE",
        str(Path(__file__).resolve().parents[3] / "configs/prompts.json"),
    )
)


def load_prompt():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if "lid" not in prompts:
        raise KeyError("lid missing from task_prompts.json")
    return prompts["lid"]


def normalize_lang_code(value):
    """
    Strict LID normalization.

    Only surrounding whitespace and
    case are normalized.

    Also accepts Whisper-style tokens
    such as <|en|> for compatibility.

    We deliberately DO NOT convert
    'English' -> 'en' or parse sentences.
    """
    value = str(value).strip().lower()
    if value.startswith("<|") and value.endswith("|>"):
        value = value[2:-2]
    return value
