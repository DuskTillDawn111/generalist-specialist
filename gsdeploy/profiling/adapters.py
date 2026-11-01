"""Task-specific inference for the pinned v2 candidate pool.

All model weights are local. No quantization, CPU offload, or artificial work.
"""

import json
import re
import time
from pathlib import Path


def normalize(value):
    import unicodedata

    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


class Adapter:
    def __init__(self, spec, resources):
        import torch

        self.torch = torch
        self.spec = spec
        self.resources = resources
        self.kind = spec["kind"]
        path = spec["path"]
        dtype = torch.bfloat16
        options = dict(torch_dtype=dtype, device_map="cuda:0", local_files_only=True)
        if self.kind == "qa":
            from transformers import T5ForConditionalGeneration, T5Tokenizer

            self.processor = T5Tokenizer.from_pretrained(path, local_files_only=True, legacy=True)
            self.no_answer_marker = normalize(self.processor.decode(
                self.processor.encode("<No Answer>"), skip_special_tokens=True
            ))
            self.model = T5ForConditionalGeneration.from_pretrained(path, **options).eval()
        elif self.kind == "image":
            from transformers import AutoModel, CLIPImageProcessor, CLIPTokenizer

            self.model = AutoModel.from_pretrained(path, trust_remote_code=True, **options).eval()
            self.processor = CLIPImageProcessor(size=224, crop_size=224)
            self.tokenizer = CLIPTokenizer.from_pretrained(path, local_files_only=True)
            self.classes = json.loads(Path(resources["classmap"]).read_text())
            features = []
            with torch.inference_mode():
                for start in range(0, 1000, 32):
                    prompts = [
                        "a photo of a " + self.classes[str(i)][1].replace("_", " ") + "."
                        for i in range(start, min(1000, start + 32))
                    ]
                    batch = self.tokenizer(prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to("cuda")
                    features.append(self.model.encode_text(**batch))
                self.text_features = torch.nn.functional.normalize(torch.cat(features), dim=-1)
        elif self.kind == "ocr":
            from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

            # This release includes both flat 4.x and nested newer config fields.
            # Its checkpoint tensors use the 4.x model.layers.* naming. Preserve
            # the original file, validate duplicates and construct the flat config.
            raw = json.loads((Path(path) / "config.json").read_text())
            nested = raw.pop("text_config", {})
            for key in ["hidden_size", "num_hidden_layers", "num_attention_heads",
                        "num_key_value_heads", "intermediate_size", "vocab_size"]:
                if key in nested and key in raw and nested[key] != raw[key]:
                    raise ValueError(f"Inconsistent OCR config field {key}")
            config = Qwen2_5_VLConfig.from_dict(raw)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path, config=config, attn_implementation="sdpa", **options
            ).eval()
            self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            self.prompt = Path(resources["ocr_prompt"]).read_text()
        elif self.kind == "speech":
            from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
            from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
            import fasttext

            self.processor = VibeVoiceASRProcessor.from_pretrained(
                path, language_model_pretrained_name=resources["speech_tokenizer"]
            )
            self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                path, attn_implementation="sdpa", **options
            ).eval()
            self.lid = fasttext.load_model(resources["fasttext"])
        else:
            raise ValueError(f"Unknown profile adapter: {self.kind}")
        if any(p.device.type != "cuda" for p in self.model.parameters()):
            raise RuntimeError("Profile requires all model parameters on GPU")
        torch.cuda.synchronize()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.parameter_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

    def memory(self):
        t = self.torch
        return {
            "allocated_mib": t.cuda.memory_allocated() / 1024**2,
            "reserved_mib": t.cuda.memory_reserved() / 1024**2,
            "peak_allocated_mib": t.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": t.cuda.max_memory_reserved() / 1024**2,
        }

    def run(self, sample):
        torch = self.torch
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        with torch.inference_mode():
            if self.kind == "qa":
                result = self.qa(sample)
            elif self.kind == "image":
                result = self.image(sample)
            elif self.kind == "ocr":
                result = self.ocr(sample)
            else:
                result = self.speech(sample)
        torch.cuda.synchronize()
        result.update({
            "sample_id": sample["id"], "task": sample["task"],
            "service_ms": (time.perf_counter_ns() - start) / 1e6,
            "reference": sample.get("label"),
        })
        return result

    def qa(self, sample):
        torch = self.torch
        prefix = self.processor.encode(sample["question"].lower() + " \\n ", add_special_tokens=False)
        context = self.processor.encode(sample["context"].lower(), add_special_tokens=False)
        capacity = 512 - len(prefix) - 1
        if capacity <= 32:
            raise ValueError("Question exceeds frozen QA input budget")
        step = max(1, capacity - min(128, capacity // 2))
        candidates = []
        for offset in range(0, max(1, len(context)), step):
            ids = prefix + context[offset:offset + capacity] + [self.processor.eos_token_id]
            inputs = torch.tensor([ids], device="cuda")
            output = self.model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                max_new_tokens=64, min_new_tokens=0, do_sample=False, num_beams=1,
                return_dict_in_generate=True, output_scores=True,
            )
            scores = self.model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            )
            text = self.processor.decode(output.sequences[0], skip_special_tokens=True).strip()
            ended = int(output.sequences[0, -1]) == self.processor.eos_token_id
            candidates.append((float(scores.mean()), text, ended, len(output.scores), len(ids)))
            if offset + capacity >= len(context):
                break
        _, prediction, ended, tokens, input_tokens = max(candidates, key=lambda row: row[0])
        if normalize(prediction) == self.no_answer_marker:
            prediction = ""
        return {
            "prediction": prediction, "answers": sample["answers"],
            "windows": len(candidates), "input_tokens": input_tokens,
            "output_tokens": tokens, "truncated": not ended, "success": ended,
        }

    def image(self, sample):
        from PIL import Image

        image = Image.open(sample["input"]).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to("cuda", self.torch.bfloat16)
        embeddings = self.model.encode_image(**inputs)
        embeddings = self.torch.nn.functional.normalize(embeddings, dim=-1)
        scores = embeddings @ self.text_features.T
        index = int(scores.argmax(-1))
        return {
            "prediction": self.classes[str(index)][0], "class_index": index,
            "correct": self.classes[str(index)][0] == sample["label"],
            "success": True, "input_resolution": [224, 224], "classes": 1000,
        }

    def ocr(self, sample):
        from PIL import Image

        image = Image.open(sample["input"]).convert("RGB")
        width, height = image.size
        scale = 1288 / max(width, height)
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": self.prompt}, {"type": "image"}
        ]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
        ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False, num_beams=1)
        generated = ids[0, inputs["input_ids"].shape[-1]:]
        raw = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
        text = re.sub(r"\A\s*---\s*\n.*?\n---\s*\n?", "", raw, count=1, flags=re.S)
        ended = int(generated[-1]) in {
            self.processor.tokenizer.eos_token_id,
            *([self.model.generation_config.eos_token_id] if isinstance(self.model.generation_config.eos_token_id, int)
              else self.model.generation_config.eos_token_id or []),
        }
        return {
            "prediction": text.strip(), "raw_output": raw,
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "output_tokens": len(generated), "truncated": not ended, "success": ended,
            "resized_resolution": list(image.size),
        }

    def speech(self, sample):
        import soundfile as sf
        import librosa

        audio, rate = sf.read(sample["input"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != 24000:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=24000)
        inputs = self.processor(
            audio=[audio], sampling_rate=24000, return_tensors="pt",
            padding=True, add_generation_prompt=True,
        )
        inputs = {k: v.to("cuda") if isinstance(v, self.torch.Tensor) else v for k, v in inputs.items()}
        output = self.model.generate(
            **inputs, max_new_tokens=1024, do_sample=False, num_beams=1,
            pad_token_id=self.processor.pad_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        generated = output[0, inputs["input_ids"].shape[1]:]
        raw = self.processor.decode(generated, skip_special_tokens=True)
        segments = self.processor.post_process_transcription(raw)
        # A generated segment can contain timestamps without a Content field.
        # Retain that output as a failed transcription, rather than crashing
        # the measurement or silently excluding the request from accounting.
        transcript = " ".join(
            segment["text"] for segment in segments
            if isinstance(segment.get("text"), str)
        ).strip()
        ended = int(generated[-1]) == self.processor.tokenizer.eos_token_id
        prediction = transcript
        lid_ms = None
        if sample["task"] == "lid":
            t = time.perf_counter_ns()
            # Use the native result to avoid fastText's NumPy-2 copy=False wrapper bug.
            predictions = self.lid.f.predict(transcript.replace("\n", " ") + "\n", 1, 0.0, "strict") if transcript else []
            prediction = predictions[0][1].removeprefix("__label__") if predictions else ""
            lid_ms = (time.perf_counter_ns() - t) / 1e6
        return {
            "prediction": prediction, "transcript": transcript, "segments": segments,
            "raw_output": raw, "output_tokens": len(generated),
            "duration_s": len(audio) / 24000, "fasttext_ms": lid_ms,
            "truncated": not ended, "success": ended and bool(transcript),
        }
