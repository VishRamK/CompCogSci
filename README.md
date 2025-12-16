# Natural Language → Ludax → Simulation

## Quick Start

```zsh
# 1) Generate JSON, compile Ludax, and run IGMLite for all variants
python run_pipeline.py

# 2) Produce outcome and length plots
python analysis.py
```

## Individual Steps

```zsh
# Natural language → JSON (schema-constrained)
python natlang_to_json.py examples/clean_rules.txt ludax_schema.json outputs/clean_rules/clean_rules.json

# JSON → Ludax
python json_to_ludax.py outputs/clean_rules/clean_rules.json outputs/clean_rules/clean_rules.ludax

# Ludax → IGMLite rollouts (with JSON log)
python run_intuitive_gamer.py outputs/clean_rules/clean_rules.ludax --episodes 100 --json outputs/clean_rules/clean_rules_igm.json

# Summarize and plot
python analysis.py
```

## Metrics
- JSON validity: inferred by success of `natlang_to_json.py` and schema validation.
- Ludax parse success: inferred by successful `json_to_ludax.py` and `LudaxEnvironment` initialization.
- Rollout success: nonzero counts in IGMLite JSON log.
- Outcome KL and length differences: use `evaluation.py` to compare generated vs reference logs.
