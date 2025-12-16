import os
import subprocess
import sys
import json

OUTPUT_DIR = "outputs"

# Configurable paths
RULES_DIR = "examples"
SCHEMA = "ludax_schema.json"
NATLANG_TO_JSON = "natlang_to_json.py"
JSON_TO_LUDAX = "json_to_ludax.py"
RUN_IG = "run_intuitive_gamer.py"

# List of rule files to test (add more for ambiguity experiments)
RULE_FILES = [
    "tictactoe_clean.txt",
    "tictactoe_omission_nodiag.txt",
    "tictactoe_omission_nodraw.txt",
    "tictactoe_lexical_nodiag.txt",
    "tictactoe_paraphrase.txt",
    "nim_clean.txt",
    "nim_omission_wincond.txt",
    "nim_lexical_some.txt",
    "nim_paraphrase.txt",
    "connect4_clean.txt",
    "connect4_omission_gravity.txt",
    "connect4_omission_draw.txt",
    "connect4_lexical_adj.txt",
]

def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for rules_file in RULE_FILES:
        base = os.path.splitext(rules_file)[0]
        variant_dir = os.path.join(OUTPUT_DIR, base)
        os.makedirs(variant_dir, exist_ok=True)
        json_out = os.path.join(variant_dir, f"{base}.json")
        ludax_out = os.path.join(variant_dir, f"{base}.ludax")
        log_out = os.path.join(variant_dir, f"{base}_igm.json")
        print(f"\n=== Processing: {rules_file} → {variant_dir} ===")
        # Step 1: NatLang → JSON
        subprocess.run([
            sys.executable, NATLANG_TO_JSON,
            f"{RULES_DIR}/{rules_file}", SCHEMA, json_out
        ], check=True)
        # Step 2: JSON → Ludax
        subprocess.run([
            sys.executable, JSON_TO_LUDAX, json_out, ludax_out
        ], check=True)
        # Step 3: Ludax → Intuitive Gamer (emit JSON log)
        subprocess.run([
            sys.executable, RUN_IG, ludax_out, "--episodes", "10", "--json", log_out
        ], check=True)
        print(f"Saved: {json_out}, {ludax_out}, {log_out}")

def main():
    run_pipeline()

if __name__ == "__main__":
    main()
