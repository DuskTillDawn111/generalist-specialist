import json
import math
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(
    "benchmark/results/characterization"
)

FULL = (
    ROOT / "c1_qwen_whisper_taskmatrix_v1"
)

BASE30 = (
    ROOT / "c1_qwen_vs_whisper_v1"
)

INT = (
    ROOT / "c1_intensity_v1"
)


VICTIMS = [
    "text",
    "image",
    "visual_qa",
]

SETTINGS = {
    "asr": [
        0,
        140,
        420,
        1250,
    ],
    "lid": [
        0,
        15,
        45,
        130,
    ],
}


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def baseline_mean(path, task):

    rows = read_jsonl(path)

    xs = [
        float(x["worker_latency_ms"])
        for x in rows
        if (
            x.get("worker") == "qwen"
            and
            x.get("task_type") == task
        )
    ]

    return statistics.mean(xs)


def pair_point(path, task, baseline):

    rows = read_jsonl(path)

    target = [
        x
        for x in rows
        if (
            x.get("profile_role") == "target"
            and
            x.get("worker") == "qwen"
            and
            x.get("task_type") == task
        )
    ]

    latency = [
        float(x["worker_latency_ms"])
        for x in target
    ]

    overlap_ns = sum(
        float(x["active_overlap_ns"])
        for x in target
    )

    duration_ns = sum(
        float(x["end_ns"])
        - float(x["start_ns"])
        for x in target
    )

    rho = (
        overlap_ns
        / duration_ns
    )

    pair_mean = statistics.mean(
        latency
    )

    slowdown = (
        pair_mean / baseline
        - 1.0
    ) * 100.0

    return rho, slowdown


# ------------------------------------------------------------
# Build five points per pair:
#
# 4 intensity-sweep points
# + 1 old near-full-overlap point
# ------------------------------------------------------------

DATA = {}

for victim in VICTIMS:

    # N=200 baseline for intensity sweep
    b200 = baseline_mean(
        INT
        / "baseline"
        / f"qwen_{victim}_single_n200.jsonl",
        victim,
    )

    # N=30 matched baseline for original full-overlap pair
    b30 = baseline_mean(
        BASE30
        / f"qwen_{victim}_single_n30.jsonl",
        victim,
    )

    for co_task, gaps in SETTINGS.items():

        points = []

        for gap in gaps:

            p = (
                INT
                / "pairs"
                / (
                    f"qwen_{victim}"
                    f"_vs_whisper_{co_task}"
                    f"_gap{gap}_n200.jsonl"
                )
            )

            points.append(
                pair_point(
                    p,
                    victim,
                    b200,
                )
            )

        # Original almost-full-overlap measurement.
        p = (
            FULL
            / (
                f"qwen_{victim}"
                f"_vs_whisper_{co_task}_n30.jsonl"
            )
        )

        points.append(
            pair_point(
                p,
                victim,
                b30,
            )
        )

        DATA[
            (victim, co_task)
        ] = points


# ------------------------------------------------------------
# Fit y = S * rho^alpha
#
# For a fixed alpha:
#   z = rho^alpha
#   optimal S = (z^T y)/(z^T z)
# ------------------------------------------------------------

def fit_given_alpha(
    points,
    alpha,
):

    rho = np.array([
        x[0]
        for x in points
    ])

    y = np.array([
        x[1]
        for x in points
    ])

    z = rho ** alpha

    S = (
        np.dot(z, y)
        / np.dot(z, z)
    )

    pred = S * z

    sse = np.sum(
        (y - pred) ** 2
    )

    return (
        float(S),
        pred,
        float(sse),
    )


def r2_score(
    y,
    pred,
):

    y = np.array(y)

    sse = np.sum(
        (y - pred) ** 2
    )

    sst = np.sum(
        (y - np.mean(y)) ** 2
    )

    return (
        1.0 - sse / sst
        if sst > 0
        else 0.0
    )


# ------------------------------------------------------------
# A. Per-pair alpha
# ------------------------------------------------------------

print()
print("===== PER-PAIR POWER LAW =====")
print()

print(
    f"{'pair':<32}"
    f"{'alpha':>9}"
    f"{'S':>10}"
    f"{'R2':>9}"
)

print("-" * 62)


pair_best = {}

for (
    victim,
    co_task
), points in DATA.items():

    best = None

    for alpha in np.linspace(
        0.5,
        2.5,
        2001,
    ):

        S, pred, sse = (
            fit_given_alpha(
                points,
                alpha,
            )
        )

        if (
            best is None
            or sse < best[0]
        ):
            best = (
                sse,
                alpha,
                S,
                pred,
            )

    _, alpha, S, pred = best

    y = [
        x[1]
        for x in points
    ]

    r2 = r2_score(
        y,
        pred,
    )

    pair_best[
        (victim, co_task)
    ] = (
        alpha,
        S,
        r2,
    )

    print(
        f"{victim+' <- whisper/'+co_task:<32}"
        f"{alpha:>9.3f}"
        f"{S:>10.2f}"
        f"{r2:>9.4f}"
    )


# ------------------------------------------------------------
# B. One shared alpha per contender
#
# S remains pair-specific.
# ------------------------------------------------------------

print()
print(
    "===== SHARED ALPHA PER CONTENDER ====="
)
print()

print(
    f"{'contender':<18}"
    f"{'alpha':>10}"
    f"{'RMSE':>10}"
    f"{'R2':>10}"
)

print("-" * 50)


for co_task in [
    "asr",
    "lid",
]:

    best = None

    for alpha in np.linspace(
        0.5,
        2.5,
        2001,
    ):

        total_sse = 0.0

        all_y = []
        all_pred = []

        sensitivities = {}

        for victim in VICTIMS:

            points = DATA[
                (victim, co_task)
            ]

            S, pred, sse = (
                fit_given_alpha(
                    points,
                    alpha,
                )
            )

            sensitivities[
                victim
            ] = S

            total_sse += sse

            all_y.extend([
                x[1]
                for x in points
            ])

            all_pred.extend(
                pred.tolist()
            )

        if (
            best is None
            or total_sse < best[0]
        ):
            best = (
                total_sse,
                alpha,
                sensitivities,
                all_y,
                all_pred,
            )

    (
        sse,
        alpha,
        sensitivities,
        all_y,
        all_pred,
    ) = best

    rmse = math.sqrt(
        sse
        / len(all_y)
    )

    r2 = r2_score(
        all_y,
        all_pred,
    )

    print(
        f"{'whisper/'+co_task:<18}"
        f"{alpha:>10.3f}"
        f"{rmse:>10.2f}"
        f"{r2:>10.4f}"
    )

    print(
        "  sensitivities:"
    )

    for victim in VICTIMS:

        print(
            f"    {victim:<12}"
            f"S = "
            f"{sensitivities[victim]:.2f}"
        )


# ------------------------------------------------------------
# C. Compare against simple linear law alpha=1
# ------------------------------------------------------------

print()
print(
    "===== LINEAR INTENSITY BASELINE (alpha=1) ====="
)
print()

for co_task in [
    "asr",
    "lid",
]:

    total_sse = 0.0

    all_y = []
    all_pred = []

    for victim in VICTIMS:

        points = DATA[
            (victim, co_task)
        ]

        S, pred, sse = (
            fit_given_alpha(
                points,
                1.0,
            )
        )

        total_sse += sse

        all_y.extend([
            x[1]
            for x in points
        ])

        all_pred.extend(
            pred.tolist()
        )

    rmse = math.sqrt(
        total_sse
        / len(all_y)
    )

    r2 = r2_score(
        all_y,
        all_pred,
    )

    print(
        f"{'whisper/'+co_task:<18}"
        f"RMSE={rmse:.2f} pp   "
        f"R2={r2:.4f}"
    )
