import csv
import math
import numpy as np
from pathlib import Path


ROOT = Path(
    "benchmark/results/characterization/c1_analysis_v1"
)

PAIR = ROOT / "c1_pair_dataset.csv"


# ============================================================
# Read CSV
# ============================================================

with PAIR.open() as f:
    rows = list(
        csv.DictReader(f)
    )


def num(row, key):
    return float(row[key])


# ============================================================
# 1. Additive decomposition
# ============================================================

slowdowns = [
    num(r, "slowdown_mean_pct")
    for r in rows
]

grand = sum(slowdowns) / len(slowdowns)


victims = sorted({
    r["victim_path"]
    for r in rows
})

contenders = sorted({
    r["contender_path"]
    for r in rows
})


victim_mean = {}

for v in victims:

    xs = [
        num(r, "slowdown_mean_pct")
        for r in rows
        if r["victim_path"] == v
    ]

    victim_mean[v] = (
        sum(xs) / len(xs)
    )


contender_mean = {}

for c in contenders:

    xs = [
        num(r, "slowdown_mean_pct")
        for r in rows
        if r["contender_path"] == c
    ]

    contender_mean[c] = (
        sum(xs) / len(xs)
    )


for r in rows:

    pred = (
        victim_mean[
            r["victim_path"]
        ]
        +
        contender_mean[
            r["contender_path"]
        ]
        -
        grand
    )

    actual = num(
        r,
        "slowdown_mean_pct"
    )

    r["additive_pred_pct"] = pred

    r[
        "interaction_residual_pct"
    ] = actual - pred


# ============================================================
# 2. Kernel interaction features
# ============================================================

kernel_names = [
    "GEMV",
    "GEMM",
    "CONV",
    "ATTN",
    "REDNORM",
    "ELEMCOPY",
    "OTHER",
]


for r in rows:

    v = np.array([
        num(
            r,
            f"victim_kernel_{k}_pct"
        )
        for k in kernel_names
    ]) / 100.0

    c = np.array([
        num(
            r,
            f"contender_kernel_{k}_pct"
        )
        for k in kernel_names
    ]) / 100.0

    denom = (
        np.linalg.norm(v)
        * np.linalg.norm(c)
    )

    r["kernel_cosine"] = (
        float(np.dot(v, c) / denom)
        if denom > 0
        else 0.0
    )

    r["kernel_overlap"] = float(
        np.minimum(v, c).sum()
    )

    r["kernel_l1_distance"] = float(
        np.abs(v - c).sum()
    )

    for i, k in enumerate(
        kernel_names
    ):
        r[
            f"pair_{k}_product"
        ] = float(
            v[i] * c[i]
        )


# ============================================================
# 3. Resource interaction features
# ============================================================

resource_names = [
    "gpu_util_pct",
    "mem_util_pct",
    "power_w",
    "vram_used_mib",
    "energy_j_per_req",
    "kernel_gpu_ms_per_req",
    "kernel_instances_per_req",
]


for r in rows:

    for k in resource_names:

        v = num(
            r,
            f"victim_{k}"
        )

        c = num(
            r,
            f"contender_{k}"
        )

        r[
            f"pair_{k}_product"
        ] = v * c

        r[
            f"pair_{k}_absdiff"
        ] = abs(v - c)


# ============================================================
# Spearman correlation
# ============================================================

def ranks(xs):

    order = sorted(
        range(len(xs)),
        key=lambda i: xs[i]
    )

    result = [0.0] * len(xs)

    i = 0

    while i < len(xs):

        j = i

        while (
            j + 1 < len(xs)
            and
            xs[
                order[j + 1]
            ] == xs[
                order[i]
            ]
        ):
            j += 1

        avg_rank = (
            i + j
        ) / 2.0 + 1.0

        for k in range(i, j + 1):
            result[
                order[k]
            ] = avg_rank

        i = j + 1

    return result


def pearson(x, y):

    mx = sum(x) / len(x)
    my = sum(y) / len(y)

    dx = [
        a - mx
        for a in x
    ]

    dy = [
        b - my
        for b in y
    ]

    nume = sum(
        a * b
        for a, b in zip(dx, dy)
    )

    den = math.sqrt(
        sum(a * a for a in dx)
        *
        sum(b * b for b in dy)
    )

    if den == 0:
        return 0.0

    return nume / den


def spearman(feature, target):

    x = [
        float(r[feature])
        for r in rows
    ]

    y = [
        float(r[target])
        for r in rows
    ]

    return pearson(
        ranks(x),
        ranks(y),
    )


# ============================================================
# 4. Main effects
# ============================================================

print()
print(
    "===== VICTIM MEAN SUSCEPTIBILITY ====="
)
print()

for name, value in sorted(
    victim_mean.items(),
    key=lambda x: x[1],
):

    print(
        f"{name:<20}"
        f"{value:>8.2f}%"
    )


print()
print(
    "===== CONTENDER MEAN PRESSURE ====="
)
print()

for name, value in sorted(
    contender_mean.items(),
    key=lambda x: x[1],
):

    print(
        f"{name:<20}"
        f"{value:>8.2f}%"
    )


# ============================================================
# 5. Additive error
# ============================================================

residuals = [
    float(
        r[
            "interaction_residual_pct"
        ]
    )
    for r in rows
]

rmse = math.sqrt(
    sum(
        x * x
        for x in residuals
    )
    / len(residuals)
)

mae = (
    sum(
        abs(x)
        for x in residuals
    )
    / len(residuals)
)


print()
print(
    "===== ADDITIVE MODEL ====="
)
print()

print(
    "RMSE             =",
    round(rmse, 2),
    "pp",
)

print(
    "MAE              =",
    round(mae, 2),
    "pp",
)

print(
    "max |residual|   =",
    round(
        max(
            abs(x)
            for x in residuals
        ),
        2,
    ),
    "pp",
)


# ============================================================
# 6. Feature correlations
# ============================================================

candidate_features = [
    "kernel_cosine",
    "kernel_overlap",
    "kernel_l1_distance",

    "pair_GEMV_product",
    "pair_GEMM_product",
    "pair_CONV_product",
    "pair_ATTN_product",
    "pair_REDNORM_product",
    "pair_ELEMCOPY_product",

    "pair_gpu_util_pct_product",
    "pair_mem_util_pct_product",
    "pair_power_w_product",
    "pair_vram_used_mib_product",
    "pair_energy_j_per_req_product",
    "pair_kernel_gpu_ms_per_req_product",
    "pair_kernel_instances_per_req_product",

    "pair_gpu_util_pct_absdiff",
    "pair_mem_util_pct_absdiff",
    "pair_power_w_absdiff",
    "pair_kernel_gpu_ms_per_req_absdiff",
]


def show_corr(target):

    values = [
        (
            feature,
            spearman(
                feature,
                target,
            )
        )
        for feature
        in candidate_features
    ]

    values.sort(
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    for feature, rho in values[:12]:

        print(
            f"{feature:<42}"
            f"{rho:>8.3f}"
        )


print()
print(
    "===== CORRELATION WITH RAW SLOWDOWN ====="
)
print()

show_corr(
    "slowdown_mean_pct"
)


print()
print(
    "===== CORRELATION WITH INTERACTION RESIDUAL ====="
)
print()

show_corr(
    "interaction_residual_pct"
)


# ============================================================
# 7. Largest interactions
# ============================================================

print()
print(
    "===== LARGEST PAIR-SPECIFIC INTERACTIONS ====="
)
print()

sorted_rows = sorted(
    rows,
    key=lambda r: abs(
        float(
            r[
                "interaction_residual_pct"
            ]
        )
    ),
    reverse=True,
)

print(
    f"{'victim':<18}"
    f"{'contender':<20}"
    f"{'obs':>9}"
    f"{'pred':>9}"
    f"{'resid':>9}"
    f"{'cos':>9}"
    f"{'overlap':>10}"
)

print("-" * 84)

for r in sorted_rows[:10]:

    print(
        f"{r['victim_path']:<18}"
        f"{r['contender_path']:<20}"
        f"{num(r,'slowdown_mean_pct'):>9.1f}"
        f"{float(r['additive_pred_pct']):>9.1f}"
        f"{float(r['interaction_residual_pct']):>+9.1f}"
        f"{float(r['kernel_cosine']):>9.3f}"
        f"{float(r['kernel_overlap']):>10.3f}"
    )


# ============================================================
# 8. Save enriched CSV
# ============================================================

out = (
    ROOT
    / "c1_pair_dataset_enriched.csv"
)

fieldnames = list(
    rows[0].keys()
)

with out.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


print()
print(
    "Saved:",
    out,
)
