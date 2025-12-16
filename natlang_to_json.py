import json
import sys
from dotenv import load_dotenv
import asyncio
import os
from pathlib import Path

# GenLM Control imports
from genlm.control import PromptedLLM, JsonSchema, AWRS
from huggingface_hub import login
from jsonschema import validate, ValidationError

load_dotenv()


# --- Family-specific schemas (remove most oneOf branching) ---

# --- Ambiguity-friendly family schemas ---
# Keys are required, but values are largely unconstrained (type-only),
# so the LLM has freedom to decide based on rules.

TICTACTOE_SCHEMA_AMBIG = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "TicTacToePlayableAmbig",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "game_name": {"type": "string"},

        # Keep canonical representation for Ludax
        "board": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"const": "grid"},
                "rows": {"const": 3},
                "cols": {"const": 3},
            },
            "required": ["type", "rows", "cols"],
        },

        "players": {"const": 2},

        "pieces": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
        },

        "turn_order": {"const": "alternate"},

        "moves": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"type": {"const": "place_on_empty_cell"}},
            "required": ["type"],
        },

        # Ambiguity allowed in directions (diag?) and draw_condition
        "win_condition": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "line_length": {"const": 3},
                "directions": {
                    "oneOf": [
                        {"type": "array", "items": {"type": "string"}, "minItems": 0},
                        {"type": "string"},
                    ]
                },
                "last_object": {
                    "oneOf": [{"type": "boolean"}, {"type": "string"}]
                },
            },
            # keep keys present
            "required": ["line_length", "directions", "last_object"],
        },

        "draw_condition": {"type": "string"},
    },
    "required": [
        "game_name", "board", "players", "pieces",
        "turn_order", "moves", "win_condition", "draw_condition"
    ],
}


CONNECT4_SCHEMA_AMBIG = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ConnectFourPlayableAmbig",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "game_name": {"type": "string"},

        "board": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"const": "grid"},
                "rows": {"const": 6},
                "cols": {"const": 7},
            },
            "required": ["type", "rows", "cols"],
        },

        "players": {"const": 2},

        "pieces": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
        },

        "turn_order": {"const": "alternate"},

        "moves": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"type": {"const": "drop_in_column"}},
            "required": ["type"],
        },

        "win_condition": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "line_length": {"const": 4},
                "directions": {
                    "oneOf": [
                        {"type": "array", "items": {"type": "string"}, "minItems": 0},
                        {"type": "string"},
                    ]
                },
                "last_object": {
                    "oneOf": [{"type": "boolean"}, {"type": "string"}]
                },
            },
            "required": ["line_length", "directions", "last_object"],
        },

        "draw_condition": {"type": "string"},
    },
    "required": [
        "game_name", "board", "players", "pieces",
        "turn_order", "moves", "win_condition", "draw_condition"
    ],
}


NIM_SCHEMA_AMBIG = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "NimPlayableAmbig",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "game_name": {"type": "string"},

        "board": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"const": "heaps"},
                "heaps": {"type": "integer", "minimum": 1},
                "sizes": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1},
            },
            "required": ["type", "heaps", "sizes"],
        },

        "players": {"const": 2},

        # Nim has no "pieces" on a grid, but keep required key.
        "pieces": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },

        "turn_order": {"const": "alternate"},

        "moves": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"type": {"const": "remove_from_heap"}},
            "required": ["type"],
        },

        # Ambiguity: last_object true/false vs "unspecified"
        "win_condition": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "line_length": {"oneOf": [{"type":"integer","minimum":1}, {"type":"string"}]},
                "directions": {"type":"string"},
                "last_object": {"oneOf": [{"type":"boolean"}, {"type":"string"}]},

            },
            "required": ["line_length", "directions", "last_object"],
        },

        "draw_condition": {"type": "string"},
    },
    "required": [
        "game_name", "board", "players", "pieces",
        "turn_order", "moves", "win_condition", "draw_condition"
    ],
}



def pick_schema_for_rules(natlang_rules: str):
    t = natlang_rules.lower()

    if ("connect four" in t) or ("connect4" in t) or ("connect-4" in t) or \
       ("drop" in t and "column" in t) or ("6x7" in t):
        return "connect4", CONNECT4_SCHEMA_AMBIG

    if ("nim" in t) or ("heap" in t) or ("heaps" in t):
        return "nim", NIM_SCHEMA_AMBIG

    if ("tic tac toe" in t) or ("tictactoe" in t) or ("noughts and crosses" in t) or \
       ("3x3" in t) or ("three in a row" in t):
        return "tictactoe", TICTACTOE_SCHEMA_AMBIG

    return "unknown", None



def _make_prompt(natlang_rules: str, family: str | None) -> str:
    family_hint = f"Detected family: {family}.\n" if family else ""
    return (
        "Output exactly ONE valid JSON object. No markdown. No commentary.\n"
        "- Only include keys required by the schema.\n"
        '- If a value is not specified in the rules and not fixed by the schema, use the string "unspecified".\n\n'
        f"{family_hint}"
        "Natural-language rules:\n"
        f"{natlang_rules}\n"
    )

def infer_draw_condition(natlang_rules: str, obj: dict) -> str | None:
    t = natlang_rules.lower()

    # Only override when the rules explicitly describe a draw due to filling the board.
    says_draw = "draw" in t or "tie" in t
    says_full = ("board fills" in t) or ("board is full" in t) or ("fills up" in t) or ("no empty" in t)

    board_type = (obj.get("board") or {}).get("type", "").lower()

    if says_draw and says_full and board_type == "grid":
        return "board_full"

    # Nim typically has no draws if not stated; only set if explicitly stated.
    if board_type == "heaps" and "draw" in t:
        return "unspecified"  # or some other convention you want

    return None


def _extract_json_block(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _family_template(family: str) -> str:
    if family == "nim":
        return (
            "Output exactly ONE JSON object on ONE line.\n"
            "Use this key order exactly:\n"
            "game_name, board, players, pieces, turn_order, moves, win_condition, draw_condition.\n\n"
            "NIM REQUIREMENTS:\n"
            '- board.type MUST be "heaps"\n'
            '- board MUST contain: {"type":"heaps","heaps":<int>,"sizes":[<int>,...]}\n'
            '- moves.type MUST be "remove_from_heap"\n'
            "- win_condition.last_object should be true/false (or \"unspecified\" if not stated)\n\n"
            "Template (structure only; fill values from rules):\n"
            '{"game_name":"Nim","board":{"type":"heaps","heaps":3,"sizes":[3,4,5]},'
            '"players":2,"pieces":["token"],"turn_order":"alternate","moves":{"type":"remove_from_heap"},'
            '"win_condition":{"line_length":"unspecified","directions":"unspecified","last_object":"unspecified"},'
            '"draw_condition":"unspecified"}'
        )

    if family == "connect4":
        return (
            "Output exactly ONE JSON object on ONE line.\n"
            "Use this key order exactly:\n"
            "game_name, board, players, pieces, turn_order, moves, win_condition, draw_condition.\n\n"
            "CONNECT4 REQUIREMENTS:\n"
            '- board.type MUST be "grid"\n'
            '- board MUST contain rows=6 cols=7\n'
            '- moves.type MUST be "drop_in_column"\n\n'
            "Template:\n"
            '{"game_name":"Connect Four","board":{"type":"grid","rows":6,"cols":7},"players":2,'
            '"pieces":["R","Y"],"turn_order":"alternate","moves":{"type":"drop_in_column"},'
            '"win_condition":{"line_length":4,"directions":["row","column","diagonal"],"last_object":"unspecified"},'
            '"draw_condition":"unspecified"}'
        )

    # default: tictactoe
    return (
        "Output exactly ONE JSON object on ONE line.\n"
        "Use this key order exactly:\n"
        "game_name, board, players, pieces, turn_order, moves, win_condition, draw_condition.\n\n"
        "TICTACTOE REQUIREMENTS:\n"
        '- board.type MUST be "grid"\n'
        "- board MUST contain rows=3 cols=3\n"
        '- moves.type MUST be "place_on_empty_cell"\n\n'
        "Template:\n"
        '{"game_name":"TicTacToe","board":{"type":"grid","rows":3,"cols":3},"players":2,'
        '"pieces":["X","O"],"turn_order":"alternate","moves":{"type":"place_on_empty_cell"},'
        '"win_condition":{"line_length":3,"directions":["row","column","diagonal"],"last_object":"unspecified"},'
        '"draw_condition":"unspecified"}'
    )


def genlm_generate_json(rules_path, natlang_rules, schema_path):
    # ✅ FIX: classify using rule TEXT, not filename/path
    family, schema = pick_schema_for_rules(natlang_rules)
    print(f"[schema] family={family} title={schema.get('title') if schema else None}")

    if schema is None:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=True)

    llama = PromptedLLM.from_name(
        "meta-llama/Llama-3.2-1B-Instruct",
        eos_tokens=[b"<|eom_id|>", b"<|eot_id|>"],
        temperature=0.1,
    )

    # ✅ Use a family-specific template (Nim gets heaps template)
    skeleton = _family_template(family if family != "unknown" else "tictactoe")

    # (Optional but helpful) include schema in the prompt
    schema_str = json.dumps(schema)

    llama.prompt_ids = llama.model.tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": "Return only a single-line JSON object. No extra text."},
            {"role": "user", "content": "Schema:\n" + schema_str},
            {"role": "user", "content": "Natural-language rules:\n" + natlang_rules},
            {"role": "user", "content": skeleton},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )

    schema_potential = JsonSchema(schema)
    coerced_schema = schema_potential.coerce(llama, f=b"".join)
    sampler = AWRS(llama, coerced_schema)

    schema_seqs = asyncio.run(
        sampler.smc(
            n_particles=128,
            max_tokens=220,   # Nim JSON tends to be longer than TicTacToe
            ess_threshold=0.4,
            verbosity=1,
        )
    )

    if not getattr(schema_seqs, "decoded_posterior", {}):
        raise RuntimeError("No valid completions. Increase max_tokens/particles OR loosen schema.")

    best_text, _ = max(schema_seqs.decoded_posterior.items(), key=lambda kv: float(kv[1]))
    candidate = _extract_json_block(best_text) or best_text

    obj = json.loads(candidate)

    dc = infer_draw_condition(natlang_rules, obj)
    if dc is not None:
        obj["draw_condition"] = dc

    validate(instance=obj, schema=schema)
    return obj



def main():
    if len(sys.argv) != 4:
        print("Usage: python natlang_to_json.py <rules.txt> <schema.json> <output.json>")
        sys.exit(1)
    rules_path, schema_path, output_path = sys.argv[1:4]
    with open(rules_path, "r", encoding="utf-8") as f:
        natlang_rules = f.read()
    try:
        json_obj = genlm_generate_json(rules_path, natlang_rules, schema_path)
    except Exception as e:
        print("Error during GenLM JSON generation:", e)
        sys.exit(2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2)
    print(f"JSON written to {output_path}")


if __name__ == "__main__":
    main()
