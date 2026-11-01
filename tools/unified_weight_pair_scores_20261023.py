"""Score all explicitly declared historical spill/control contrasts uniformly."""
import argparse
import csv
from pathlib import Path

from tools.unified_weight_rescore_20261023 import TERMS, exact_dot, gain, read, receipt, solve, write


def analyze(inventory_path, rescore_path, pair_map_path, output):
    inventory, rescore, pair_map = map(read, (inventory_path, rescore_path, pair_map_path))
    rows = {r["label"]: r for r in inventory["rows"]}
    assert pair_map["inventory"]["sha256"] == receipt(inventory_path)["sha256"]
    pairs = []
    families = {}
    for group in pair_map["groups"]:
        for pair in group["pairs"]:
            if not pair["both_numeric_rescorable"]:
                continue
            spill, control = rows[pair["spill"]], rows[pair["control"]]
            vector = gain(spill["terms"], control["terms"])
            name = None
            if spill["study"] == "queue-coupled-study-20261023" and spill["role"] == "coupled_spill":
                name = "coupled303_304_matched"
            elif spill["study"] == "queue-refined-study-20261023" and spill["role"] == "calibrated_spill":
                if spill["seed"] in (203, 204):
                    name = f"refined203_204_text{spill['workload']['text']}"
                elif spill["seed"] in (205, 206, 207, 208):
                    name = "refined205_208_known_complete"
            elif spill["study"] == "queue-aware-study-20261023":
                leaf_s, leaf_c = spill["label"].split("/")[-2], control["label"].split("/")[-2]
                if leaf_s.startswith("mixed-text2-g2-ocr-queue_latency-") and leaf_c.startswith("mixed-text2-g2-ocr-queue_specialist_only-"):
                    name = "aware_latency_vs_specialist_balance_seven"
                elif leaf_s.startswith("mixed-text2-g2-ocr-fixed_1of32_swrr-") and leaf_c.startswith("mixed-text2-g2-ocr-no_split_swrr-"):
                    name = "aware_fixed_1of32_vs_zero_two"
            pairs.append(dict(study=group["study"], seed=spill["seed"], workload=group["workload"],
                              pair=pair, gain_exact=vector, family=name,
                              per_scheme_delta={label: float(exact_dot(vector, scheme["weights_exact"]))
                                                for label, scheme in rescore["schemes"].items()}))
            if name:
                families.setdefault(name, []).append(len(pairs)-1)
    studies = {}
    for name, indices in families.items():
        vectors = [pairs[i]["gain_exact"] for i in indices]
        studies[name] = dict(pair_indices=indices, pair_count=len(indices),
                            seed_order=[pairs[i]["seed"] for i in indices],
                            maximin_unrestricted=solve(vectors),
                            maximin_floor1percent_cap50percent=solve(vectors, .01, .5),
                            per_scheme={label: dict(gains=[pairs[i]["per_scheme_delta"][label] for i in indices],
                                                   positive_count=sum(pairs[i]["per_scheme_delta"][label]>0 for i in indices))
                                        for label in rescore["schemes"]})
    result = dict(inventory=receipt(inventory_path), rescore=receipt(rescore_path), pair_map=receipt(pair_map_path),
                  source=receipt(__file__), pair_count=len(pairs), pairs=pairs, repeated_comparison_families=studies,
                  all_historical_pairs_maximin=solve([p["gain_exact"] for p in pairs]),
                  scope="Posthoc uniform preference scores. All-history LP asks an intentionally stronger question; obsolete policies need not all be rescued. Families may include strategy contrasts and development data; no new statistical confirmation.",
                  GPU_runs_added=0)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output/"analysis.json", result)
    with (output/"declared-contrasts.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["study", "seed", "text_rps", "spill", "control", "comparison_kind", "scheme", "delta_U"])
        writer.writeheader()
        for p in pairs:
            for label, value in p["per_scheme_delta"].items():
                writer.writerow(dict(study=p["study"], seed=p["seed"], text_rps=p["workload"].get("text"),
                                     spill=p["pair"]["spill"], control=p["pair"]["control"],
                                     comparison_kind=p["pair"]["comparison_kind"], scheme=label, delta_U=value))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "rescore", "pair-map", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.inventory, a.rescore, a.pair_map, a.output)
    print({"pairs": result["pair_count"], "families": len(result["repeated_comparison_families"])})
