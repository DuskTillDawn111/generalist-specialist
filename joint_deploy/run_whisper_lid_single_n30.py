import json
import queue
import time
from pathlib import Path
from statistics import mean

from run_static_hybrid import (
    WORKERS,
    Worker,
    read_jsonl,
)

from run_dynamic_hybrid import (
    wait_ready,
    warm_one,
)


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

POOL = (
    ROOT
    / "benchmark/task_pools/"
    "mixed_formal_1200_6task.jsonl"
)

OUTPUT = (
    ROOT
    / "benchmark/results/placement/"
    "single_whisper_lid_worker_protocol_n30.jsonl"
)

TARGET_COUNT = 30
TIMEOUT_S = 3600.0


def main():

    all_rows = read_jsonl(
        POOL
    )

    tasks = [
        row
        for row in all_rows
        if row["task_type"] == "lid"
    ][:TARGET_COUNT]

    if len(tasks) != TARGET_COUNT:
        raise RuntimeError(
            f"Expected {TARGET_COUNT} LID rows, "
            f"got {len(tasks)}"
        )

    events = queue.Queue()

    workers = {
        "whisper": Worker(
            "whisper",
            WORKERS["whisper"],
            events,
        )
    }

    results = []

    try:

        print(
            "===== START WHISPER ====="
        )

        workers[
            "whisper"
        ].start()

        wait_ready(
            "whisper",
            workers,
            events,
            TIMEOUT_S,
        )

        print(
            "whisper: READY"
        )

        print()
        print(
            "===== WARMUP ====="
        )

        warm_one(
            "whisper",
            tasks[0],
            -1,
            workers,
            events,
            TIMEOUT_S,
        )

        print()
        print(
            "===== SINGLE LID BENCHMARK ====="
        )

        for i, item in enumerate(
            tasks
        ):

            workers[
                "whisper"
            ].send({
                "request_id": i,
                "pool_id":
                    item["pool_id"],
                "arrival_index":
                    item["arrival_index"],
                "task_type":
                    "lid",
                "sample":
                    item["sample"],
                "submit_ns":
                    time.time_ns(),
            })

        deadline = (
            time.time()
            + TIMEOUT_S
        )

        while len(results) < TARGET_COUNT:

            remaining = (
                deadline
                - time.time()
            )

            if remaining <= 0:
                raise TimeoutError(
                    "Whisper LID benchmark timeout."
                )

            worker, kind, payload = (
                events.get(
                    timeout=min(
                        remaining,
                        1.0,
                    )
                )
            )

            if (
                worker == "whisper"
                and kind == "RESULT"
            ):

                payload[
                    "assigned_worker"
                ] = "whisper"

                results.append(
                    payload
                )

            elif (
                worker == "whisper"
                and kind == "ERROR"
            ):

                raise RuntimeError(
                    payload
                )

        results.sort(
            key=lambda r:
                int(r["request_id"])
        )

        OUTPUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with OUTPUT.open(
            "w",
            encoding="utf-8",
        ) as f:

            for row in results:

                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        values = [
            float(
                row[
                    "worker_latency_ms"
                ]
            )
            for row in results
        ]

        print()
        print(
            "===== RESULT ====="
        )

        print(
            "samples =",
            len(values),
        )

        print(
            "mean worker latency ms =",
            round(
                mean(values),
                6,
            ),
        )

        print(
            "min ms =",
            round(
                min(values),
                6,
            ),
        )

        print(
            "max ms =",
            round(
                max(values),
                6,
            ),
        )

        print(
            "Saved:",
            OUTPUT,
        )

    finally:

        try:
            workers[
                "whisper"
            ].terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
