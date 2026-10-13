import argparse
import json
import queue
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

PYTHON = (
    ROOT
    / "envs/minicpm_env/bin/python"
)

WORKER_SCRIPT = (
    ROOT
    / "joint_deploy/minicpm_worker.py"
)

TASK_ORDER = [
    "text",
    "image",
    "ocr",
    "asr",
]


def read_jsonl(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def mean(xs):
    return (
        sum(xs) / len(xs)
        if xs
        else None
    )


def percentile(xs, p):
    if not xs:
        return None

    values = sorted(xs)

    if len(values) == 1:
        return values[0]

    k = (
        (len(values) - 1)
        * p / 100.0
    )

    lo = int(k)
    hi = min(
        lo + 1,
        len(values) - 1
    )

    frac = k - lo

    return (
        values[lo] * (1 - frac)
        + values[hi] * frac
    )


class Worker:

    def __init__(self):
        self.events = queue.Queue()

        self.process = None

        self.reader_thread = None

    def start(self):

        self.process = subprocess.Popen(
            [
                str(PYTHON),
                str(WORKER_SCRIPT),
            ],

            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,

            text=True,
            bufsize=1,

            cwd=str(ROOT),
        )

        self.reader_thread = threading.Thread(
            target=self._reader,
            daemon=True,
        )

        self.reader_thread.start()

    def _reader(self):

        for line in self.process.stdout:

            line = line.rstrip()

            parsed = False

            for kind in (
                "READY",
                "RESULT",
                "ERROR",
                "BYE",
            ):

                prefix = (
                    kind + "|"
                )

                if line.startswith(prefix):

                    payload = json.loads(
                        line[
                            len(prefix):
                        ]
                    )

                    self.events.put(
                        (
                            kind,
                            payload,
                        )
                    )

                    parsed = True
                    break

            if not parsed:
                print(
                    "[minicpm]",
                    line,
                    flush=True,
                )

    def send(self, request):

        self.process.stdin.write(
            json.dumps(
                request,
                ensure_ascii=False
            )
            + "\n"
        )

        self.process.stdin.flush()

    def shutdown(self):

        if (
            self.process is None
            or self.process.poll()
            is not None
        ):
            return

        try:
            self.send({
                "op": "shutdown"
            })
        except Exception:
            pass

    def terminate(self):

        if (
            self.process is not None
            and self.process.poll()
            is None
        ):
            self.process.terminate()


def wait_for(
    worker,
    wanted,
    timeout_s
):

    deadline = (
        time.time()
        + timeout_s
    )

    while True:

        remaining = (
            deadline
            - time.time()
        )

        if remaining <= 0:
            raise TimeoutError(
                f"Timeout waiting for {wanted}"
            )

        try:
            kind, payload = (
                worker.events.get(
                    timeout=min(
                        remaining,
                        1.0
                    )
                )
            )

        except queue.Empty:

            if (
                worker.process.poll()
                is not None
            ):
                raise RuntimeError(
                    "MiniCPM worker exited."
                )

            continue

        if kind == wanted:
            return payload

        if kind == "ERROR":
            raise RuntimeError(
                payload
            )


def warm_one(
    worker,
    sample,
    request_id,
    timeout_s,
):

    request = {
        "request_id":
            request_id,

        "pool_id":
            f"warmup-{request_id}",

        "arrival_index":
            request_id,

        "task_type":
            sample[
                "task_type"
            ],

        "sample":
            sample[
                "sample"
            ],

        "submit_ns":
            time.time_ns(),
    }

    worker.send(
        request
    )

    result = wait_for(
        worker,
        "RESULT",
        timeout_s,
    )

    print(
        f"Warmup minicpm "
        f"{sample['task_type']:>5}: OK"
    )

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pool",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--timeout-s",
        type=float,
        default=3600.0
    )

    args = parser.parse_args()

    tasks = read_jsonl(
        args.pool
    )

    if not tasks:
        raise RuntimeError(
            "Task pool is empty."
        )

    first_by_task = {}

    for item in tasks:
        first_by_task.setdefault(
            item["task_type"],
            item
        )

    for task in TASK_ORDER:

        if task not in first_by_task:
            raise RuntimeError(
                f"Pool missing task: {task}"
            )

    worker = Worker()

    results = []
    errors = []

    try:

        # =============================
        # Start
        # =============================

        print(
            "===== STARTING MINICPM ====="
        )

        worker.start()

        ready = wait_for(
            worker,
            "READY",
            args.timeout_s,
        )

        print(
            "minicpm: READY"
        )

        print(
            "model:",
            ready.get(
                "model"
            )
        )

        # =============================
        # Warmup
        # =============================

        print()
        print(
            "===== WARMUP ====="
        )

        warmup_id = -1

        for task in TASK_ORDER:

            warm_one(
                worker,
                first_by_task[
                    task
                ],
                warmup_id,
                args.timeout_s,
            )

            warmup_id -= 1

        print()
        print(
            "ALL MINICPM TASK PATHS WARM."
        )

        # =============================
        # Routing
        # =============================

        print()
        print(
            "===== PURE GENERALIST ROUTING ====="
        )

        for task in TASK_ORDER:
            print(
                f"{task:>5} -> minicpm"
            )

        start_ns = time.time_ns()

        # =============================
        # Submit all 800
        # =============================

        for request_id, item in enumerate(
            tasks
        ):

            request = {
                "request_id":
                    request_id,

                "pool_id":
                    item[
                        "pool_id"
                    ],

                "arrival_index":
                    item[
                        "arrival_index"
                    ],

                "task_type":
                    item[
                        "task_type"
                    ],

                "sample":
                    item[
                        "sample"
                    ],

                "submit_ns":
                    time.time_ns(),
            }

            worker.send(
                request
            )

        print()
        print(
            "Submitted:",
            len(tasks)
        )

        # =============================
        # Collect
        # =============================

        completed = 0

        deadline = (
            time.time()
            + args.timeout_s
        )

        while completed < len(tasks):

            remaining = (
                deadline
                - time.time()
            )

            if remaining <= 0:
                raise TimeoutError(
                    "MiniCPM run timed out."
                )

            try:

                kind, payload = (
                    worker.events.get(
                        timeout=min(
                            remaining,
                            1.0
                        )
                    )
                )

            except queue.Empty:

                if (
                    worker.process.poll()
                    is not None
                ):
                    raise RuntimeError(
                        "MiniCPM worker exited."
                    )

                continue

            if kind == "RESULT":

                results.append(
                    payload
                )

                completed += 1

                print(
                    f"["
                    f"{completed:03d}/"
                    f"{len(tasks):03d}"
                    f"] "
                    f"{payload['task_type']:>5} "
                    f"-> minicpm "
                    f"run="
                    f"{payload['worker_latency_ms']:.2f} "
                    f"ms "
                    f"wait="
                    f"{payload.get('queue_wait_ms', 0):.2f} "
                    f"ms"
                )

            elif kind == "ERROR":

                errors.append(
                    payload
                )

                completed += 1

                print(
                    "ERROR:",
                    payload
                )

        end_ns = time.time_ns()

        # =============================
        # Save raw results
        # =============================

        results.sort(
            key=lambda r:
                r[
                    "arrival_index"
                ]
        )

        output = Path(
            args.output
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output,
            "w",
            encoding="utf-8"
        ) as f:

            for row in results:

                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False
                    )
                    + "\n"
                )

        # =============================
        # Metrics
        # =============================

        per_task = {}

        for task in TASK_ORDER:

            rows = [
                row
                for row in results
                if row[
                    "task_type"
                ] == task
            ]

            lat = [
                float(
                    row[
                        "worker_latency_ms"
                    ]
                )
                for row in rows
            ]

            wait = [
                float(
                    row.get(
                        "queue_wait_ms",
                        0.0
                    )
                )
                for row in rows
            ]

            item = {
                "count":
                    len(rows),

                "routing": {
                    "minicpm":
                        len(rows)
                },

                "worker_latency_mean_ms":
                    mean(lat),

                "worker_latency_p95_ms":
                    percentile(
                        lat,
                        95
                    ),

                "queue_wait_mean_ms":
                    mean(wait),

                "queue_wait_p95_ms":
                    percentile(
                        wait,
                        95
                    ),
            }

            if task != "asr":

                correct = [
                    bool(
                        row[
                            "correct"
                        ]
                    )
                    for row in rows
                    if "correct" in row
                ]

                item[
                    "accuracy"
                ] = (
                    sum(correct)
                    / len(correct)
                    if correct
                    else None
                )

            else:

                errors_sum = sum(
                    int(
                        row.get(
                            "word_errors",
                            0
                        )
                    )
                    for row in rows
                )

                ref_sum = sum(
                    int(
                        row.get(
                            "reference_words",
                            0
                        )
                    )
                    for row in rows
                )

                item[
                    "corpus_wer"
                ] = (
                    errors_sum
                    / ref_sum
                    if ref_sum
                    else None
                )

            per_task[
                task
            ] = item

        wall_time_s = (
            end_ns
            - start_ns
        ) / 1e9

        summary = {
            "policy":
                "pure_generalist_minicpm",

            "tasks":
                len(tasks),

            "results":
                len(results),

            "errors":
                len(errors),

            "models": [
                "minicpm"
            ],

            "routing": {
                "minicpm":
                    len(tasks)
            },

            "routing_by_task": {
                task: {
                    "minicpm":
                        sum(
                            1
                            for x in tasks
                            if x[
                                "task_type"
                            ] == task
                        )
                }
                for task in TASK_ORDER
            },

            "wall_time_s":
                wall_time_s,

            "throughput_tasks_per_s":
                (
                    len(results)
                    / wall_time_s
                    if wall_time_s > 0
                    else None
                ),

            "per_task":
                per_task,
        }

        summary_path = (
            output.with_suffix(
                ".summary.json"
            )
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                summary,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print()
        print(
            "===== SUMMARY ====="
        )

        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
        )

        print()
        print(
            "Results:",
            output
        )

        print(
            "Summary:",
            summary_path
        )

    finally:

        worker.shutdown()

        time.sleep(
            1.0
        )

        worker.terminate()


if __name__ == "__main__":
    main()
