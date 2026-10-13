import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pynvml


MIB = 1024 ** 2


def mib(value):
    if value is None:
        return None

    return round(
        float(value) / MIB,
        3
    )


def get_device_memory(handle):
    info = pynvml.nvmlDeviceGetMemoryInfo(
        handle
    )

    return {
        "total_mib": mib(info.total),
        "used_mib": mib(info.used),
        "free_mib": mib(info.free),
    }


def get_compute_processes(handle):
    try:
        processes = (
            pynvml
            .nvmlDeviceGetComputeRunningProcesses(
                handle
            )
        )
    except pynvml.NVMLError:
        return {}

    result = {}

    for p in processes:
        pid = int(p.pid)

        used = getattr(
            p,
            "usedGpuMemory",
            None
        )

        if used is None:
            memory_mib = None
        else:
            try:
                memory_mib = mib(used)
            except Exception:
                memory_mib = None

        result[pid] = memory_mib

    return result


def summarize_intervals(
    timestamps
):
    if len(timestamps) < 2:
        return {
            "mean_ms": None,
            "min_ms": None,
            "max_ms": None,
        }

    intervals = [
        (
            timestamps[i]
            - timestamps[i - 1]
        ) * 1000.0
        for i in range(
            1,
            len(timestamps)
        )
    ]

    return {
        "mean_ms": round(
            sum(intervals)
            / len(intervals),
            3
        ),
        "min_ms": round(
            min(intervals),
            3
        ),
        "max_ms": round(
            max(intervals),
            3
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--label",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--interval-ms",
        type=float,
        default=50.0
    )

    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    if (
        args.command
        and
        args.command[0] == "--"
    ):
        args.command = (
            args.command[1:]
        )

    if not args.command:
        raise RuntimeError(
            "Missing child command."
        )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    trace_path = (
        output_path
        .with_suffix(".csv")
    )

    # ------------------------------------------
    # NVML init
    # ------------------------------------------

    pynvml.nvmlInit()

    try:
        handle = (
            pynvml
            .nvmlDeviceGetHandleByIndex(
                0
            )
        )

        gpu_name = (
            pynvml
            .nvmlDeviceGetName(
                handle
            )
        )

        driver = (
            pynvml
            .nvmlSystemGetDriverVersion()
        )

        baseline_memory = (
            get_device_memory(
                handle
            )
        )

        baseline_processes = (
            get_compute_processes(
                handle
            )
        )

        print(
            "===== PYTHON NVML PROCESS SAMPLER ====="
        )

        print(
            "Label:",
            args.label
        )

        print(
            "GPU:",
            gpu_name
        )

        print(
            "Driver:",
            driver
        )

        print(
            "Requested interval:",
            args.interval_ms,
            "ms"
        )

        print(
            "NVML baseline memory:",
            baseline_memory
        )

        print(
            "Existing compute processes:",
            baseline_processes
        )

        if baseline_processes:
            raise RuntimeError(
                "GPU is not clean before profiling. "
                "Existing compute processes found."
            )

        print()
        print(
            "Child command:"
        )

        print(
            " ".join(
                args.command
            )
        )

        print()

        start_wall = (
            datetime.now()
            .astimezone()
            .isoformat()
        )

        start = (
            time.perf_counter()
        )

        process = subprocess.Popen(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        child_pid = process.pid

        print(
            "Child PID:",
            child_pid
        )

        print()

        samples = []
        marker_events = []

        stdout_done = (
            threading.Event()
        )

        marker_lock = (
            threading.Lock()
        )

        def snapshot_marker(
            marker_name
        ):
            elapsed = (
                time.perf_counter()
                - start
            )

            proc_map = (
                get_compute_processes(
                    handle
                )
            )

            device_mem = (
                get_device_memory(
                    handle
                )
            )

            target_mem = (
                proc_map.get(
                    child_pid
                )
            )

            valid_memories = [
                x
                for x in proc_map.values()
                if x is not None
            ]

            total_compute_mem = (
                round(
                    sum(
                        valid_memories
                    ),
                    3
                )
                if valid_memories
                else 0.0
            )

            event = {
                "marker":
                    marker_name,

                "elapsed_s":
                    round(
                        elapsed,
                        6
                    ),

                "target_process_memory_mib":
                    target_mem,

                "all_compute_process_memory_mib":
                    total_compute_mem,

                "compute_pids":
                    sorted(
                        proc_map.keys()
                    ),

                "device_used_mib":
                    device_mem[
                        "used_mib"
                    ],
            }

            with marker_lock:
                marker_events.append(
                    event
                )

        def consume_stdout():
            assert (
                process.stdout
                is not None
            )

            for raw_line in (
                process.stdout
            ):
                line = raw_line.rstrip(
                    "\n"
                )

                print(
                    line,
                    flush=True
                )

                if line.startswith(
                    "NVML_MARKER|"
                ):
                    marker_name = (
                        line.split(
                            "|",
                            1
                        )[1]
                        .strip()
                    )

                    snapshot_marker(
                        marker_name
                    )

            stdout_done.set()

        stdout_thread = (
            threading.Thread(
                target=consume_stdout,
                daemon=True,
            )
        )

        stdout_thread.start()

        interval_s = (
            args.interval_ms
            / 1000.0
        )

        next_tick = (
            time.perf_counter()
        )

        # --------------------------------------
        # Sampling loop
        # --------------------------------------

        while True:
            now_abs = (
                time.perf_counter()
            )

            elapsed = (
                now_abs - start
            )

            proc_map = (
                get_compute_processes(
                    handle
                )
            )

            device_mem = (
                get_device_memory(
                    handle
                )
            )

            target_mem = (
                proc_map.get(
                    child_pid
                )
            )

            valid_memories = [
                x
                for x in proc_map.values()
                if x is not None
            ]

            all_compute_mem = (
                round(
                    sum(
                        valid_memories
                    ),
                    3
                )
                if valid_memories
                else 0.0
            )

            samples.append(
                {
                    "elapsed_s":
                        round(
                            elapsed,
                            6
                        ),

                    "target_process_memory_mib":
                        target_mem,

                    "all_compute_process_memory_mib":
                        all_compute_mem,

                    "compute_process_count":
                        len(
                            proc_map
                        ),

                    "device_used_mib":
                        device_mem[
                            "used_mib"
                        ],

                    "device_free_mib":
                        device_mem[
                            "free_mib"
                        ],
                }
            )

            if (
                process.poll()
                is not None
                and
                stdout_done.is_set()
            ):
                break

            next_tick += (
                interval_s
            )

            sleep_s = (
                next_tick
                - time.perf_counter()
            )

            if sleep_s > 0:
                time.sleep(
                    sleep_s
                )
            else:
                # If one sample ever overruns,
                # resynchronize rather than
                # accumulating timing drift.
                next_tick = (
                    time.perf_counter()
                )

        return_code = (
            process.wait()
        )

        stdout_thread.join(
            timeout=1.0
        )

        end = (
            time.perf_counter()
        )

        # --------------------------------------
        # Summaries
        # --------------------------------------

        target_values = [
            x[
                "target_process_memory_mib"
            ]
            for x in samples
            if x[
                "target_process_memory_mib"
            ]
            is not None
        ]

        all_compute_values = [
            x[
                "all_compute_process_memory_mib"
            ]
            for x in samples
        ]

        device_used_values = [
            x[
                "device_used_mib"
            ]
            for x in samples
        ]

        peak_target_process_mib = (
            max(target_values)
            if target_values
            else None
        )

        peak_all_compute_mib = (
            max(
                all_compute_values
            )
            if all_compute_values
            else None
        )

        max_device_used_mib = (
            max(
                device_used_values
            )
        )

        device_delta_mib = (
            max_device_used_mib
            - baseline_memory[
                "used_mib"
            ]
        )

        sample_times = [
            x["elapsed_s"]
            for x in samples
        ]

        actual_interval = (
            summarize_intervals(
                sample_times
            )
        )

        marker_names = [
            x["marker"]
            for x in marker_events
        ]

        payload = {
            "schema_version":
                "nvml-process-profile-v2",

            "backend":
                "nvidia-ml-py",

            "label":
                args.label,

            "timestamp":
                start_wall,

            "gpu":
                gpu_name,

            "driver":
                driver,

            "requested_sampling_interval_ms":
                args.interval_ms,

            "actual_sampling_interval":
                actual_interval,

            "command":
                args.command,

            "child_pid":
                child_pid,

            "return_code":
                return_code,

            "wall_time_s":
                round(
                    end - start,
                    6
                ),

            "gpu_baseline":
                baseline_memory,

            "baseline_compute_processes":
                baseline_processes,

            "peak_target_process_memory_mib":
                peak_target_process_mib,

            "peak_all_compute_process_memory_mib":
                peak_all_compute_mib,

            "max_device_used_mib":
                max_device_used_mib,

            "peak_device_delta_from_baseline_mib":
                round(
                    device_delta_mib,
                    3
                ),

            "markers":
                marker_events,

            "marker_names":
                marker_names,

            "num_samples":
                len(samples),

            "trace_csv":
                str(
                    trace_path
                ),

            "notes": [
                (
                    "Primary deployment VRAM metric is "
                    "per-process usedGpuMemory obtained "
                    "through NVML."
                ),
                (
                    "Device-level used memory may include "
                    "driver or vGPU reserved memory and is "
                    "therefore retained only as a "
                    "cross-check."
                ),
                (
                    "peak_device_delta_from_baseline_mib "
                    "subtracts the pre-launch device-level "
                    "baseline."
                ),
                (
                    "Sampling is discrete; extremely short "
                    "transient peaks between samples may "
                    "still be underestimated."
                ),
            ],
        }

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                payload,
                f,
                ensure_ascii=False,
                indent=2
            )

        with open(
            trace_path,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "elapsed_s",
                    "target_process_memory_mib",
                    "all_compute_process_memory_mib",
                    "compute_process_count",
                    "device_used_mib",
                    "device_free_mib",
                ],
            )

            writer.writeheader()
            writer.writerows(
                samples
            )

        print()
        print(
            "===== NVML SUMMARY ====="
        )

        print(
            "Return code:",
            return_code
        )

        print(
            "Samples:",
            len(samples)
        )

        print(
            "Actual interval:",
            actual_interval
        )

        print(
            "Peak target process memory:",
            peak_target_process_mib,
            "MiB"
        )

        print(
            "Peak all compute-process memory:",
            peak_all_compute_mib,
            "MiB"
        )

        print(
            "Max device used:",
            max_device_used_mib,
            "MiB"
        )

        print(
            "Peak device delta:",
            round(
                device_delta_mib,
                3
            ),
            "MiB"
        )

        print(
            "Markers:"
        )

        for event in marker_events:
            print(
                " ",
                event
            )

        print(
            "Saved summary:",
            output_path
        )

        print(
            "Saved trace:",
            trace_path
        )

        required_markers = {
            "post_load",
            "post_warmup",
            "post_inference",
        }

        missing_markers = (
            required_markers
            - set(
                marker_names
            )
        )

        if missing_markers:
            raise RuntimeError(
                "Missing required markers: "
                + ", ".join(
                    sorted(
                        missing_markers
                    )
                )
            )

        if return_code != 0:
            sys.exit(
                return_code
            )

    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
