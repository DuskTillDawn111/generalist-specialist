"""Measure the text-only fastText component using actual ASR transcripts."""

import argparse
import json
import os
from pathlib import Path
import random
import resource
import statistics
import time

from .runner import quantile, save


def profile_fasttext(root):
    root = Path(root)
    output = root / "fasttext-cpu"
    if output.exists():
        raise FileExistsError(output)
    state = json.loads((root / "suite-status.json").read_text())
    job = next(j for j in state["jobs"] if j["kind"] == "speech"
               and j["replicas"] == 1 and j["status"] == "COMPLETE")
    profile = json.loads((Path(job["path"]) / "profile.json").read_text())
    source = next(r["raw_records"] for r in profile["rounds"] if r["task"] == "lid")
    unique = {}
    for line in Path(source).read_text().splitlines():
        row = json.loads(line)
        unique.setdefault(row["sample_id"], row)
    samples = list(unique.values())
    spec = json.loads((root / "speech-spec.json").read_text())
    output.mkdir()
    start = time.perf_counter()
    import fasttext

    model = fasttext.load_model(spec["resources"]["fasttext"])
    for sample in samples[:3]:
        model.f.predict(sample["transcript"].replace("\n", " ") + "\n", 1, 0.0, "strict")
    startup_seconds = time.perf_counter() - start
    rounds = []
    for repetition in range(3):
        order = list(samples)
        random.Random(20261015 + repetition).shuffle(order)
        records, i = [], 0
        start = time.perf_counter()
        with (output / f"round{repetition}.jsonl").open("w") as stream:
            while i < len(order) or time.perf_counter() - start < 1:
                sample = order[i % len(order)]
                transcript = sample["transcript"]
                before = time.perf_counter_ns()
                predictions = model.f.predict(transcript.replace("\n", " ") + "\n", 1, 0.0, "strict") if transcript else []
                prediction = predictions[0][1].removeprefix("__label__") if predictions else ""
                row = {"sample_id": sample["sample_id"], "prediction": prediction,
                       "reference": sample["reference"], "empty_transcript": not bool(transcript),
                       "service_ms": (time.perf_counter_ns() - before) / 1e6}
                stream.write(json.dumps(row) + "\n")
                records.append(row)
                i += 1
        seconds = time.perf_counter() - start
        service = [r["service_ms"] for r in records]
        rounds.append({"repetition": repetition, "requests": i, "wall_seconds": seconds,
                       "completed_rps_including_recording_overhead": i / seconds,
                       "service_mean_ms": statistics.mean(service),
                       "service_p95_ms": quantile(service, .95),
                       "service_p99_ms": quantile(service, .99)})
    result = {"status": "COMPLETE", "scope": "cpu_text_language_component_only",
              "input": "Actual VibeVoice LID transcripts, including empty/failed transcripts",
              "source_records": source, "unique_transcripts": len(samples), "rounds": rounds,
              "startup_import_load_and_three_warmups_seconds": startup_seconds,
              "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "cpu_count_visible": os.cpu_count(), "gpu_memory_mib": 0,
              "cpu_energy_joules": None, "energy_status": "UNMEASURED",
              "admission": "NOT_AN_AUDIO_LID_MODEL; use the full speech pipeline profile for audio LID"}
    save(output / "profile.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(profile_fasttext(args.root), indent=2))
