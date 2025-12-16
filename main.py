import os
import subprocess
import json
import csv

GAMES = [
    {"name": "TicTacToe", "variants": ["clean", "ambiguous1", "ambiguous2"]},
    {"name": "Nim", "variants": ["clean", "ambiguous1"]},
    {"name": "Connect4", "variants": ["clean", "ambiguous1"]}
]

RESULTS_CSV = "experiment_results.csv"

def run_pipeline(rules_path, json_path, ludax_path, log_prefix):
    # 1. NatLang → JSON
    json_valid = False
    try:
        subprocess.run(["python", "natlang_to_json.py", rules_path, "ludax_schema.json", json_path], check=True, timeout=120)
        json_valid = True
    except Exception as e:
        print(f"[{log_prefix}] JSON generation failed: {e}")

    # 2. JSON → Ludax
    ludax_valid = False
    if json_valid:
        try:
            subprocess.run(["python", "json_to_ludax.py", json_path, ludax_path], check=True, timeout=30)
            ludax_valid = True
        except Exception as e:
            print(f"[{log_prefix}] Ludax compilation failed: {e}")

    # 3. Ludax → Simulation
    rollout_success = False
    outcome_counts = {"P1": 0, "P2": 0, "draw": 0, "fail": 0}
    mean_length = None
    if ludax_valid:
        try:
            # Use JSON output from runner for robust parsing
            log_path = ludax_path.replace('.ludax', '_igm.json')
            subprocess.run([
                "python", "run_intuitive_gamer.py", ludax_path, "--episodes", "100", "--json", log_path
            ], check=True, timeout=120)
            with open(log_path, "r") as f:
                log = json.load(f)
            counts = log.get("counts", {})
            outcome_counts = {"P1": counts.get("P1", 0), "P2": counts.get("P0", 0), "draw": counts.get("draw", 0), "fail": 0}
            lengths = log.get("lengths", [])
            mean_length = sum(lengths) / len(lengths) if lengths else None
            rollout_success = (outcome_counts["P1"] + outcome_counts["P2"] + outcome_counts["draw"]) > 0
        except Exception as e:
            print(f"[{log_prefix}] Rollout failed: {e}")

    return {
        "json_valid": json_valid,
        "ludax_valid": ludax_valid,
        "rollout_success": rollout_success,
        "outcome_counts": outcome_counts,
        "mean_length": mean_length
    }

def main():
    with open(RESULTS_CSV, "w", newline="") as csvfile:
        fieldnames = ["game", "variant", "json_valid", "ludax_valid", "rollout_success", "P1", "P2", "draw", "fail", "mean_length"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for game in GAMES:
            for variant in game["variants"]:
                rules_path = f"examples/{game['name'].lower()}_{variant}_rules.txt"
                json_path = f"examples/{game['name'].lower()}_{variant}.json"
                ludax_path = f"examples/{game['name'].lower()}_{variant}.ludax"
                log_prefix = f"{game['name']}-{variant}"
                metrics = run_pipeline(rules_path, json_path, ludax_path, log_prefix)
                row = {
                    "game": game["name"],
                    "variant": variant,
                    "json_valid": metrics["json_valid"],
                    "ludax_valid": metrics["ludax_valid"],
                    "rollout_success": metrics["rollout_success"],
                    "P1": metrics["outcome_counts"]["P1"],
                    "P2": metrics["outcome_counts"]["P2"],
                    "draw": metrics["outcome_counts"]["draw"],
                    "fail": metrics["outcome_counts"]["fail"],
                    "mean_length": metrics["mean_length"]
                }
                writer.writerow(row)
                print(f"[{log_prefix}] Done.")

if __name__ == "__main__":
    main()