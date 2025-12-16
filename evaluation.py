import json
import math
from typing import Dict, Any


def summarize_gameplay_log(log_path: str) -> Dict[str, Any]:
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts = data.get("counts", {})
    episodes = data.get("episodes", [])
    total_terms = counts.get("P0", 0) + counts.get("P1", 0) + counts.get("draw", 0)
    total_fail = counts.get("timeout", 0) + counts.get("no_legal_moves", 0) + counts.get("unknown_winner", 0)
    total = total_terms + total_fail if total_terms + total_fail > 0 else 1

    dist = {
        "P0": counts.get("P0", 0) / total,
        "P1": counts.get("P1", 0) / total,
        "draw": counts.get("draw", 0) / total,
        "timeout": counts.get("timeout", 0) / total,
        "no_legal_moves": counts.get("no_legal_moves", 0) / total,
        "unknown_winner": counts.get("unknown_winner", 0) / total,
    }

    lengths = [ep.get("steps", len(ep.get("moves", []))) for ep in episodes]
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0
    var_len = (sum((l - mean_len) ** 2 for l in lengths) / len(lengths)) if lengths else 0.0

    return {
        "config": data.get("config", {}),
        "counts": counts,
        "distribution": dist,
        "episodes": episodes,
        "lengths": lengths,
        "length_stats": {
            "mean": mean_len,
            "var": var_len,
        },
    }


def kl_divergence(p: Dict[str, float], q: Dict[str, float], keys) -> float:
    eps = 1e-9
    kl = 0.0
    for k in keys:
        pi = max(p.get(k, 0.0), eps)
        qi = max(q.get(k, 0.0), eps)
        kl += pi * math.log(pi / qi)
    return kl


def compare_behavior(gen_log_path: str, ref_log_path: str) -> Dict[str, Any]:
    gen = summarize_gameplay_log(gen_log_path)
    ref = summarize_gameplay_log(ref_log_path)

    # Outcome KL over P0/P1/draw
    keys = ["P0", "P1", "draw"]
    kl = kl_divergence(gen["distribution"], ref["distribution"], keys)

    # Length differences
    gen_len = gen["length_stats"]["mean"]
    ref_len = ref["length_stats"]["mean"]
    len_diff = gen_len - ref_len

    return {
        "gen_path": gen_log_path,
        "ref_path": ref_log_path,
        "outcome_kl": kl,
        "mean_length_diff": len_diff,
        "gen_dist": {k: gen["distribution"][k] for k in keys},
        "ref_dist": {k: ref["distribution"][k] for k in keys},
        "gen_mean_len": gen_len,
        "ref_mean_len": ref_len,
    }
