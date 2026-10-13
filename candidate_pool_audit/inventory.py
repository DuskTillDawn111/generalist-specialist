"""Explicit candidate scope and task contracts; no changes to the frozen catalog."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('text', 'image', 'ocr', 'asr', 'visual_qa', 'lid')
# current task adapters with historical measured evidence
EXISTING = {'qwen': list(TASKS), 'phi4mm': ['text', 'image', 'ocr', 'asr'],
            'minicpm': ['text', 'image', 'ocr', 'asr'], 'distilbert': ['text'],
            'resnet50': ['image'], 'paddleocr': ['ocr'], 'whisper-small': ['asr', 'lid']}
NEW = {'convnext_tiny': ['image'], 'roberta_sst2': ['text'], 'paddleocr_vl': ['ocr'],
       'whisper_turbo': ['asr', 'lid'], 'distil_whisper': ['asr']}
SOURCES = {
    'qwen': ('Qwen/Qwen2.5-Omni-7B', 'qwen2.5-omni-7b', 'qwen', 'generalist', 'bfloat16'),
    'phi4mm': ('microsoft/Phi-4-multimodal-instruct', 'phi4-multimodal', 'phi', 'generalist', 'bfloat16'),
    'gemma3n': ('google/gemma-3n-E4B-it', None, 'gemma3n', 'generalist', None),
    'minicpm': ('openbmb/MiniCPM-o-4_5', 'minicpm-o-4_5', 'minicpm', 'generalist', 'bfloat16'),
    'distilbert': ('distilbert/distilbert-base-uncased-finetuned-sst-2-english', 'distilbert-sst2', 'distilbert', 'specialist', 'float32'),
    'resnet50': ('torchvision.models.ResNet50_Weights.IMAGENET1K_V2', 'resnet50', 'resnet', 'specialist', 'float32'),
    'convnext_tiny': ('torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1', 'candidate_pool_v2/convnext_tiny', 'convnext', 'specialist', 'float32'),
    'paddleocr': ('PaddlePaddle/PP-OCRv6_medium_rec', 'paddleocr/official_models/PP-OCRv6_medium_rec', 'paddleocr', 'specialist', 'float32'),
    'paddleocr_vl': ('PaddlePaddle/PaddleOCR-VL-1.5', 'candidate_pool_v2/paddleocr_vl', 'paddleocr_vl', 'specialist', 'bfloat16'),
    'got_ocr': ('stepfun-ai/GOT-OCR2_0', None, 'got_ocr', 'specialist', None),
    'whisper-small': ('openai/whisper-small', 'whisper-small', 'whisper', 'specialist', 'float32'),
    'whisper_turbo': ('openai/whisper-large-v3-turbo', 'candidate_pool_v2/whisper_turbo', 'whisper', 'specialist', 'float32'),
    'distil_whisper': ('distil-whisper/distil-large-v3', 'candidate_pool_v2/distil_whisper', 'distil_whisper', 'specialist', 'float32'),
    'roberta_sst2': ('textattack/roberta-base-SST-2', 'candidate_pool_v2/roberta_sst2', 'roberta', 'specialist', 'float32'),
    'siebert_rejected': ('siebert/sentiment-roberta-large-english', None, 'roberta', 'specialist', None),
}


def selected_weights(directory):
    if directory is None:
        return []
    root = ROOT / 'models' / directory
    safe = sorted(root.glob('*.safetensors'))
    safe = [p for p in safe if '.fp32.' not in p.name]
    if safe:
        # Phi's modality adapters are required for runtime.
        safe += sorted(root.glob('*-lora/adapter_model.safetensors'))
        return safe
    return sorted(root.glob('*.pth')) + sorted(root.glob('*.pdiparams'))
