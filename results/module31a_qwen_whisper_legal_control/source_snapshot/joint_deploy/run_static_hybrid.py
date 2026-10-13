import argparse
import json
import os
import queue
import subprocess
import threading
import time
from collections import (
    Counter,
    defaultdict,
)
from pathlib import Path


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)


PY_SPECIALIST = (
    ROOT
    / "envs/specialist_env/bin/python"
)

PY_PHI = (
    ROOT
    / "envs/phi_env/bin/python"
)

PY_OCR = Path(
    "/root/miniconda3/envs/"
    "ocr_env/bin/python"
)


WORKERS = {
    "distilbert": {
        "python":
            PY_SPECIALIST,

        "script":
            ROOT
            / "joint_deploy/"
            "distilbert_worker.py",
    },

    "resnet50": {
        "python":
            PY_SPECIALIST,

        "script":
            ROOT
            / "joint_deploy/"
            "resnet_worker.py",
    },

    "whisper": {
        "python":
            PY_SPECIALIST,

        "script":
            ROOT
            / "joint_deploy/"
            "whisper_worker.py",
    },

    "paddleocr": {
        "python":
            PY_OCR,

        "script":
            ROOT
            / "joint_deploy/"
            "ocr_worker.py",
    },

    "phi4": {
        "python":
            PY_PHI,

        "script":
            ROOT
            / "joint_deploy/"
            "phi_worker.py",
    },
}


SPECIALIST = {
    "text":
        "distilbert",

    "image":
        "resnet50",

    "ocr":
        "paddleocr",

    "asr":
        "whisper",
}


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



def warm_one(
    worker_name,
    item,
    request_id,
    workers,
    events,
    timeout_s
):
    request = {
        "request_id":
            request_id,

        "pool_id":
            "warmup:"
            + item["pool_id"],

        "arrival_index":
            -1,

        "task_type":
            item["task_type"],

        "sample":
            item["sample"],

        "submit_ns":
            time.time_ns(),
    }

    workers[
        worker_name
    ].send(
        request
    )

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
                f"Warmup timeout: "
                f"{worker_name}"
            )

        if (
            workers[
                worker_name
            ].process.poll()
            is not None
        ):
            raise RuntimeError(
                f"{worker_name} exited "
                f"during warmup."
            )

        try:
            worker, kind, payload = (
                events.get(
                    timeout=min(
                        remaining,
                        1.0
                    )
                )
            )
        except queue.Empty:
            continue

        if (
            worker == worker_name
            and kind == "RESULT"
            and payload[
                "request_id"
            ] == request_id
        ):
            print(
                f"Warmup "
                f"{worker_name:12s} "
                f"{item['task_type']:>5s}: "
                f"OK"
            )
            return

        if (
            worker == worker_name
            and kind == "ERROR"
        ):
            raise RuntimeError(
                f"Warmup failed: "
                f"{payload}"
            )


class Worker:
    def __init__(
        self,
        name,
        config,
        event_queue
    ):
        self.name = name
        self.config = config
        self.event_queue = (
            event_queue
        )
        self.process = None

    def start(self):
        env = os.environ.copy()

        env[
            "PYTHONUNBUFFERED"
        ] = "1"

        if self.name == "paddleocr":
            cuda_lib = (
                "/root/miniconda3/envs/"
                "ocr_env/lib/python3.11/"
                "site-packages/nvidia/"
                "cu13/lib"
            )

            old = env.get(
                "LD_LIBRARY_PATH",
                ""
            )

            env[
                "LD_LIBRARY_PATH"
            ] = (
                cuda_lib
                + (
                    ":" + old
                    if old
                    else ""
                )
            )

        command = [
            str(
                self.config[
                    "python"
                ]
            ),
            str(
                self.config[
                    "script"
                ]
            ),
        ]

        self.process = (
            subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=
                    subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        )

        thread = threading.Thread(
            target=self._reader,
            daemon=True
        )

        thread.start()

    def _reader(self):
        for line in self.process.stdout:
            line = line.rstrip()

            if (
                line.startswith(
                    "READY|"
                )
                or line.startswith(
                    "RESULT|"
                )
                or line.startswith(
                    "ERROR|"
                )
                or line.startswith(
                    "BYE|"
                )
            ):
                kind, payload = (
                    line.split(
                        "|",
                        1
                    )
                )

                self.event_queue.put(
                    (
                        self.name,
                        kind,
                        json.loads(
                            payload
                        ),
                    )
                )

            else:
                print(
                    f"[{self.name}] "
                    f"{line}"
                )

    def send(self, payload):
        if (
            self.process is None
            or self.process.poll()
            is not None
        ):
            raise RuntimeError(
                f"Worker {self.name} "
                f"is not running."
            )

        self.process.stdin.write(
            json.dumps(
                payload,
                ensure_ascii=False
            )
            + "\n"
        )

        self.process.stdin.flush()

    def shutdown(self):
        if (
            self.process
            and self.process.poll()
            is None
        ):
            try:
                self.send(
                    {
                        "op":
                            "shutdown"
                    }
                )
            except Exception:
                pass

    def terminate(self):
        if (
            self.process
            and self.process.poll()
            is None
        ):
            self.process.terminate()


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
        "--phi-every",
        type=int,
        default=5,
        help=(
            "Every Nth request of each "
            "task type goes to Phi. "
            "5 means 20%% Phi."
        )
    )

    parser.add_argument(
        "--timeout-s",
        type=float,
        default=600.0
    )

    args = parser.parse_args()

    tasks = read_jsonl(
        args.pool
    )

    if not tasks:
        raise RuntimeError(
            "Task pool is empty."
        )

    task_order = [
        "text",
        "image",
        "ocr",
        "asr",
    ]

    first_by_task = {}

    for item in tasks:
        first_by_task.setdefault(
            item["task_type"],
            item
        )

    for task_type in task_order:
        if task_type not in first_by_task:
            raise RuntimeError(
                f"Pool missing task: "
                f"{task_type}"
            )

    events = queue.Queue()

    workers = {
        name:
            Worker(
                name,
                config,
                events
            )
        for name, config
        in WORKERS.items()
    }

    # -------------------------
    # Start sequentially.
    # After startup they all
    # remain resident together.
    # -------------------------

    startup_order = [
        "distilbert",
        "resnet50",
        "whisper",
        "paddleocr",
        "phi4",
    ]

    try:
        print(
            "===== STARTING WORKERS ====="
        )

        for name in startup_order:
            print(
                f"Starting {name}..."
            )

            workers[
                name
            ].start()

            deadline = (
                time.time()
                + args.timeout_s
            )

            while True:
                remaining = (
                    deadline
                    - time.time()
                )

                if remaining <= 0:
                    raise TimeoutError(
                        f"{name} READY timeout"
                    )

                worker_name, kind, payload = (
                    events.get(
                        timeout=remaining
                    )
                )

                if (
                    worker_name == name
                    and kind == "READY"
                ):
                    print(
                        f"{name}: READY"
                    )
                    break

                if (
                    worker_name == name
                    and kind == "ERROR"
                ):
                    raise RuntimeError(
                        payload
                    )

        print()
        print(
            "ALL 5 MODELS ARE RESIDENT."
        )

        # -------------------------
        # Warmup
        # -------------------------

        print()
        print(
            "===== WARMUP ====="
        )

        warmup_id = -1

        for task_type in task_order:
            target = (
                SPECIALIST[
                    task_type
                ]
            )

            warm_one(
                target,
                first_by_task[
                    task_type
                ],
                warmup_id,
                workers,
                events,
                args.timeout_s
            )

            warmup_id -= 1

        for task_type in task_order:
            warm_one(
                "phi4",
                first_by_task[
                    task_type
                ],
                warmup_id,
                workers,
                events,
                args.timeout_s
            )

            warmup_id -= 1

        print()
        print(
            "ALL MODEL/TASK PATHS WARM."
        )

        # -------------------------
        # Static 80/20 routing.
        # -------------------------

        seen_by_task = defaultdict(
            int
        )

        routing = Counter()

        submit_start_ns = (
            time.time_ns()
        )

        for request_id, item in enumerate(
            tasks
        ):
            task_type = item[
                "task_type"
            ]

            seen_by_task[
                task_type
            ] += 1

            task_count = (
                seen_by_task[
                    task_type
                ]
            )

            if (
                task_count
                % args.phi_every
                == 0
            ):
                target = "phi4"

            else:
                target = (
                    SPECIALIST[
                        task_type
                    ]
                )

            routing[
                target
            ] += 1

            request = {
                "request_id":
                    request_id,

                "pool_id":
                    item["pool_id"],

                "arrival_index":
                    item[
                        "arrival_index"
                    ],

                "task_type":
                    task_type,

                "sample":
                    item["sample"],

                "submit_ns":
                    time.time_ns(),
            }

            workers[
                target
            ].send(
                request
            )

        print()
        print(
            "===== ROUTING ====="
        )

        for worker, count in (
            routing.items()
        ):
            print(
                f"{worker:12s}: "
                f"{count}"
            )

        # -------------------------
        # Collect results.
        # -------------------------

        expected = len(
            tasks
        )

        results = []
        errors = []

        deadline = (
            time.time()
            + args.timeout_s
        )

        while (
            len(results)
            + len(errors)
            < expected
        ):
            remaining = (
                deadline
                - time.time()
            )

            if remaining <= 0:
                raise TimeoutError(
                    "Result collection timeout"
                )

            worker_name, kind, payload = (
                events.get(
                    timeout=remaining
                )
            )

            if kind == "RESULT":
                results.append(
                    payload
                )

                print(
                    f"["
                    f"{len(results):03d}/"
                    f"{expected:03d}"
                    f"] "
                    f"{payload['task_type']:>5} "
                    f"-> "
                    f"{worker_name:<10} "
                    f"{payload['worker_latency_ms']:.2f} ms"
                )

            elif kind == "ERROR":
                errors.append(
                    payload
                )

                print(
                    "ERROR:",
                    worker_name,
                    payload.get(
                        "error"
                    )
                )

        submit_end_ns = (
            time.time_ns()
        )

        results.sort(
            key=lambda x:
                x["arrival_index"]
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

        # -------------------------
        # Summary
        # -------------------------

        per_worker = Counter(
            x["worker"]
            for x in results
        )

        boolean_quality = [
            x["correct"]
            for x in results
            if "correct" in x
        ]

        total_word_errors = sum(
            x.get(
                "word_errors",
                0
            )
            for x in results
        )

        total_ref_words = sum(
            x.get(
                "reference_words",
                0
            )
            for x in results
        )

        summary = {
            "tasks":
                len(tasks),

            "results":
                len(results),

            "errors":
                len(errors),

            "routing":
                dict(routing),

            "completed_by_worker":
                dict(per_worker),

            "wall_time_s":
                (
                    submit_end_ns
                    - submit_start_ns
                ) / 1e9,

            "throughput_tasks_per_s":
                (
                    len(results)
                    / (
                        (
                            submit_end_ns
                            - submit_start_ns
                        ) / 1e9
                    )
                ),

            "classification_or_ocr_accuracy":
                (
                    sum(
                        boolean_quality
                    )
                    / len(
                        boolean_quality
                    )
                    if boolean_quality
                    else None
                ),

            "asr_corpus_wer":
                (
                    total_word_errors
                    / total_ref_words
                    if total_ref_words
                    else None
                ),
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
                indent=2
            )

        print()
        print(
            "===== SUMMARY ====="
        )

        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2
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
        for worker in workers.values():
            worker.shutdown()

        time.sleep(
            1.0
        )

        for worker in workers.values():
            worker.terminate()


if __name__ == "__main__":
    main()
