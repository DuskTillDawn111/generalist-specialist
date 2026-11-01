import json
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


import random
from ..trace import phases_for, periodic_arrivals, poisson_arrivals


class SamplePool:

    def __init__(self, path, tasks, seed):
        self.samples = {task: [] for task in tasks}
        for item in read_jsonl(path):
            if item["task_type"] in self.samples:
                self.samples[item["task_type"]].append(item)
        for task, items in self.samples.items():
            if not items:
                raise ValueError(f"Task pool has no samples for {task}")
            random.Random(f"{seed}:samples:{task}").shuffle(items)
        self.indices = {task: 0 for task in tasks}

    def next(self, task):
        index = self.indices[task]
        self.indices[task] += 1
        return self.samples[task][index % len(self.samples[task])]


def largest_inputs(pool, task, count=3):
    """Select priming inputs without consuming the measured sample sequence.

    Size is the stored media file byte count or the UTF-8 byte count of the
    context/question (plain text falls back to input/text). This deliberately
    avoids model tokenizers and does not claim a worst-case activation bound.
    Pool IDs resolve equal-size ties independently of the random replay seed.
    """
    def size(item):
        sample = item["sample"]
        if task in {"image", "ocr", "asr"}:
            return Path(sample["input"]).stat().st_size
        text = "\n".join(str(sample[key]) for key in ("context", "question") if key in sample)
        if not text:
            text = str(sample.get("input", sample.get("text", "")))
        return len(text.encode("utf-8"))

    return sorted(pool.samples[task], key=lambda item: (-size(item), item["pool_id"]))[:count]
