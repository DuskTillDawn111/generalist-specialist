import json


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
