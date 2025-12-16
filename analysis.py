import os
import json
from typing import List, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt

from evaluation import summarize_gameplay_log, compare_behavior

OUTPUT_DIR = "outputs"
REPORT_DIR = "reports"
REFERENCE_MAP_PATH = os.path.join(REPORT_DIR, "references.json")  # maps variant -> ref log path


def ensure_dirs():
    os.makedirs(REPORT_DIR, exist_ok=True)


def find_logs() -> List[str]:
    logs = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f.endswith("_igm.json"):
                logs.append(os.path.join(root, f))
    return sorted(logs)


def load_reference_map() -> dict:
    if os.path.exists(REFERENCE_MAP_PATH):
        with open(REFERENCE_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def identify_key(log_path: str) -> str:
    # e.g., outputs/tictactoe_clean/tictactoe_clean_igm.json -> tictactoe_clean
    base = os.path.basename(log_path)
    key = base.replace("_igm.json", "")
    return key


def _guess_family_from_path(path_lower: str) -> str:
    if "tictactoe" in path_lower or "tic_tac_toe" in path_lower or "nought" in path_lower:
        return "tictactoe"
    if "connect4" in path_lower or "connect-4" in path_lower or "connectfour" in path_lower:
        return "connect4"
    if "nim" in path_lower:
        return "nim"
    return "unknown"


def get_family_from_log(log_path: str) -> str:
    # Prefer explicit name inside the log; fallback to path
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        name = (
            data.get("config", {}).get("game_name")
            or data.get("game_name")
            or ""
        )
        n = str(name).lower()
        if "tic" in n or "nought" in n:
            return "tictactoe"
        if "connect" in n or "connect4" in n or "connect four" in n:
            return "connect4"
        if "nim" in n:
            return "nim"
    except Exception:
        pass
    return _guess_family_from_path(log_path.lower())


def group_logs_by_family(logs: List[str]) -> Dict[str, List[str]]:
    fam2logs: Dict[str, List[str]] = {}
    for lp in logs:
        fam = get_family_from_log(lp)
        fam2logs.setdefault(fam, []).append(lp)
    return fam2logs


def plot_family_overview(family: str, log_paths: List[str]):
    """
    Creates a single figure with:
      - Left: grouped bar chart of outcome distributions (P0, draw, P1) per variant
      - Right: boxplot of episode lengths per variant
    """
    if not log_paths:
        return

    # Collect data
    variants = []
    dists = []   # list of (P0, draw, P1)
    lengths_list = []  # list of list[int]
    for lp in log_paths:
        summ = summarize_gameplay_log(lp)
        dist = summ["distribution"]
        lengths = summ["lengths"]
        variants.append(identify_key(lp))
        dists.append((dist["P0"], dist["draw"], dist["P1"]))
        lengths_list.append(lengths)

    x = np.arange(len(variants))
    width = 0.25
    colors = {"P0": "#052951", "draw": "#5c2506", "P1": "#026324"}

    fig, axes = plt.subplots(1, 2, figsize=(max(6, 2 + 1.6 * len(variants)), 4), dpi=120)

    # Left subplot: grouped outcomes
    ax0 = axes[0]
    p0_vals = [v[0] for v in dists]
    draw_vals = [v[1] for v in dists]
    p1_vals = [v[2] for v in dists]

    ax0.bar(x - width, p0_vals, width, label="P0", color=colors["P0"])
    ax0.bar(x, draw_vals, width, label="draw", color=colors["draw"])
    ax0.bar(x + width, p1_vals, width, label="P1", color=colors["P1"])
    ax0.set_xticks(x)
    ax0.set_xticklabels(variants, rotation=30, ha="right")
    ax0.set_ylim(0, 1.0)
    ax0.set_title(f"{family}: outcome distribution")
    ax0.set_ylabel("Probability")
    ax0.legend(frameon=False)

    # Right subplot: lengths boxplot (only if any non-empty)
    ax1 = axes[1]
    any_lengths = any(len(lst) > 0 for lst in lengths_list)
    if any_lengths:
        ax1.boxplot(lengths_list, labels=variants, patch_artist=True,
                    boxprops=dict(facecolor="#2E0101", alpha=0.6),
                    medianprops=dict(color="#000000"))
        ax1.set_title(f"{family}: episode lengths")
        ax1.set_ylabel("Length")
        ax1.tick_params(axis="x", rotation=30)
    else:
        ax1.axis("off")
        ax1.set_title(f"{family}: no length data")

    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, f"{family}_overview.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def write_summary_tsv(rows: List[dict], out_path: str):
    # Minimal TSV: variant, outcome_kl, mean_length_diff, gen_mean_len, ref_mean_len
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("variant\toutcome_kl\tmean_length_diff\tgen_mean_len\tref_mean_len\n")
        for r in rows:
            f.write(
                f"{r['variant']}\t{r['outcome_kl']:.6f}\t{r['mean_length_diff']:.3f}\t"
                f"{r['gen_mean_len']:.3f}\t{r['ref_mean_len']:.3f}\n"
            )
    print(f"Saved {out_path}")


def main():
    ensure_dirs()
    ref_map = load_reference_map()
    logs = find_logs()

    fam2logs = group_logs_by_family(logs)

    # Family-level plots
    for fam, fam_logs in fam2logs.items():
        if fam == "unknown":
            # Skip unknown bucket; or plot it too if desired
            continue
        plot_family_overview(fam, fam_logs)

    # Optional: reference comparisons per-variant, still summarized in one TSV
    summary_rows = []
    for lp in logs:
        key = identify_key(lp)
        ref_path = ref_map.get(key)
        if ref_path and os.path.exists(ref_path):
            comp = compare_behavior(lp, ref_path)
            summary_rows.append({
                "variant": key,
                **comp
            })

    if summary_rows:
        write_summary_tsv(summary_rows, os.path.join(REPORT_DIR, "behavior_summary.tsv"))


if __name__ == "__main__":
    main()
