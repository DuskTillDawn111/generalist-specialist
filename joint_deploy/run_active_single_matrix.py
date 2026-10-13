import json
import subprocess
from pathlib import Path


ROOT = Path(
    "/root/autodl-tmp/generalist-specialist"
)

PYTHON = Path(
    "/root/miniconda3/envs/gsdeploy/bin/python"
)

SCRIPT = (
    ROOT
    / "joint_deploy/run_active_single_profile.py"
)

POOL = (
    ROOT
    / "benchmark/task_pools/"
    "mixed_formal_1200_6task.jsonl"
)

OUT_DIR = (
    ROOT
    / "benchmark/results/placement/"
    "active_single_v2"
)


TARGETS = [
    ("qwen", "text"),
    ("qwen", "image"),
    ("qwen", "ocr"),
    ("qwen", "asr"),
    ("qwen", "visual_qa"),
    ("qwen", "lid"),

    ("distilbert", "text"),
    ("resnet50", "image"),
    ("paddleocr", "ocr"),

    ("whisper", "asr"),
    ("whisper", "lid"),
]


TARGET_COUNT = 30


def valid_existing(path):

    summary_path = Path(
        str(path)
        + ".summary.json"
    )

    if (
        not path.exists()
        or not summary_path.exists()
    ):
        return False

    try:

        summary = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

        return (
            summary.get(
                "target_count"
            ) == TARGET_COUNT
        )

    except Exception:
        return False


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    failures = []

    for i, (
        worker,
        task,
    ) in enumerate(
        TARGETS,
        start=1,
    ):

        output = (
            OUT_DIR
            / (
                f"active_single_"
                f"{worker}_{task}_n30.jsonl"
            )
        )

        print()
        print("=" * 72)
        print(
            f"[{i:02d}/11] "
            f"{worker}/{task}"
        )
        print("=" * 72)

        if valid_existing(
            output
        ):

            print(
                "SKIP: valid existing "
                "standalone result"
            )

            continue

        command = [
            str(PYTHON),
            str(SCRIPT),

            "--pool",
            str(POOL),

            "--target-worker",
            worker,

            "--target-task",
            task,

            "--target-count",
            str(TARGET_COUNT),

            "--output",
            str(output),

            "--timeout-s",
            "3600",
        ]

        log_path = Path(
            str(output)
            + ".log"
        )

        with log_path.open(
            "w",
            encoding="utf-8",
        ) as log:

            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert (
                process.stdout
                is not None
            )

            for line in process.stdout:

                print(
                    line,
                    end="",
                    flush=True,
                )

                log.write(
                    line
                )

                log.flush()

            code = (
                process.wait()
            )

        if code != 0:

            failures.append({
                "worker":
                    worker,

                "task":
                    task,

                "return_code":
                    code,

                "log":
                    str(log_path),
            })

    print()
    print("=" * 72)
    print(
        "===== ACTIVE SINGLE MATRIX COMPLETE ====="
    )
    print(
        "targets =",
        len(TARGETS),
    )
    print(
        "failures =",
        len(failures),
    )

    if failures:

        print(
            json.dumps(
                failures,
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
