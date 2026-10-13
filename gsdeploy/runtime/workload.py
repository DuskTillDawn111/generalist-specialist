import json


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


import heapq
import random


def phases_for(rates, seed):
    return {
        task: (random.Random(f"{seed}:phase:{task}").random() + 1e-12) % 1 / rate
        for (task, rate) in sorted(rates.items())
        if rate > 0
    }


def periodic_arrivals(rates, duration, seed):
    phases = phases_for(rates, seed)
    heap = [(offset, task, 0) for (task, offset) in phases.items() if offset < duration]
    heapq.heapify(heap)
    while heap:
        offset, task, index = heapq.heappop(heap)
        yield (offset, task)
        next_offset = phases[task] + (index + 1) / rates[task]
        if next_offset < duration:
            heapq.heappush(heap, (next_offset, task, index + 1))


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
