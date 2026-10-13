import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

ROOT = Path(os.environ.get("GS_ASSET_ROOT", ".")).resolve()
from gsdeploy.backends.tasks import whisper_librispeech as formal
from gsdeploy.backends.tasks import whisper_voxlingua as lid_formal
from gsdeploy.runtime.protocol import run_worker

MODEL_DIR = Path(os.environ["GS_MODEL_PATH"])
device = torch.device("cuda:0")
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
model.to(device)
if hasattr(model.config, "forced_decoder_ids"):
    model.config.forced_decoder_ids = None
if hasattr(model.generation_config, "forced_decoder_ids"):
    model.generation_config.forced_decoder_ids = None
torch.cuda.synchronize()


@torch.inference_mode()
def handler(request):
    task = request["task_type"]
    sample = request["sample"]
    if task == "asr":
        audio, sr = formal.load_audio(sample["input"])
        inputs, long_form = formal.prepare_inputs(
            processor, audio, sr, float(sample["duration_s"]), device
        )
        generated_ids = formal.generate_transcription(model, inputs, long_form)
        torch.cuda.synchronize()
        prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        prediction = formal.normalize_asr(prediction)
        reference = formal.normalize_asr(
            sample.get("normalized_label", sample["label"])
        )
        ref_words = reference.split()
        pred_words = prediction.split()
        errors = formal.levenshtein_words(ref_words, pred_words)
        wer = errors / len(ref_words) if ref_words else 0.0
        return {
            "model": "whisper_small",
            "prediction": prediction,
            "reference": reference,
            "word_errors": errors,
            "reference_words": len(ref_words),
            "wer": wer,
            "long_form": long_form,
        }
    if task == "lid":
        lid_formal.validate_sample(sample)
        result = lid_formal.run_one(
            model=model, processor=processor, sample=sample, device=device
        )
        return {"model": "whisper_small", **result}
    raise ValueError(f"Unsupported Whisper task: {task}")


run_worker("whisper", handler, {"model": "Whisper-small", "tasks": ["asr", "lid"]})
