import json
import sys
from typing import List


def _sanitize_name(name: str) -> str:
    return str(name).replace('"', "'").strip() or "Game"


def _require(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)


def _build_end_clauses_grid(data: dict) -> List[str]:
    win = data.get("win_condition", {}) or {}
    directions = win.get("directions", []) or []
    line_len = win.get("line_length", None)

    _require(line_len is not None, "Grid games require win_condition.line_length.")
    _require(len(directions) > 0, "Grid games require win_condition.directions (non-empty).")

    clauses: List[str] = []

    has_orth = ("row" in directions) or ("column" in directions)
    has_diag = ("diagonal" in directions)

    if has_orth:
        clauses.append(f"(if (line {line_len} orientation:orthogonal) (mover win))")
    if has_diag:
        clauses.append(f"(if (line {line_len} orientation:diagonal) (mover win))")

    # Draw handling
    draw = data.get("draw_condition", "unspecified")
    if draw == "board_full":
        clauses.append("(if (full_board) (draw))")
    elif draw in ("none", "unspecified"):
        # No draw clause emitted; this is a deliberate semantic choice.
        pass
    else:
        raise ValueError(f"Unknown draw_condition: {draw}")

    _require(len(clauses) > 0, "No terminal clauses were emitted (check directions/draw_condition).")
    return clauses


def _emit_grid_place_game(data: dict) -> str:
    board = data.get("board", {}) or {}
    _require(board.get("type") == "grid", f"Expected board.type='grid', got {board.get('type')}")
    rows, cols = board.get("rows"), board.get("cols")
    _require(isinstance(rows, int) and isinstance(cols, int), "Grid board requires integer rows/cols.")

    move_type = (data.get("moves", {}) or {}).get("type")
    _require(move_type == "place_on_empty_cell",
             f"Unsupported move type for grid-place template: {move_type}")

    end_clauses = _build_end_clauses_grid(data)

    game_name = _sanitize_name(data.get("game_name", "Game"))
    players = int(data.get("players", 2))

    return f'''(game "{game_name}"
  (players {players})
  (equipment
    (board (rectangle {rows} {cols}))
  )
  (rules
    (play
      (repeat
        (P1 P2)
        (place mover (destination empty))
      )
    )
    (end
      {'\\n      '.join(end_clauses)}
    )
  )
)'''


def ludax_template(data: dict) -> str:
    """
    Single entry point. Emits Ludax only for templates that are actually implemented.
    IMPORTANT: do not silently coerce unsupported games into a different template.
    """
    board = data.get("board", {}) or {}
    move_type = (data.get("moves", {}) or {}).get("type")

    # Implemented: grid + place_on_empty_cell (Tic-Tac-Toe-style)
    if board.get("type") == "grid" and move_type == "place_on_empty_cell":
        return _emit_grid_place_game(data)

    # Not implemented (yet): these should fail loudly and be counted as failures
    if move_type == "drop_in_column":
        raise NotImplementedError(
            "drop_in_column is not implemented in this compiler. "
            "If you evaluate Connect Four, use a dedicated compiler that encodes gravity "
            "in the *legal move generator* (preferred) or explicitly state the limitation."
        )
    if board.get("type") == "heaps" or move_type == "remove_from_heap":
        raise NotImplementedError(
            "heaps/remove_from_heap is not implemented in this compiler. "
            "If you evaluate Nim, use a dedicated compiler for heap games."
        )

    raise ValueError(f"Unsupported combination: board.type={board.get('type')} moves.type={move_type}")


def json_to_ludax(json_path: str, ludax_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ludax_src = ludax_template(data)

    with open(ludax_path, "w", encoding="utf-8") as f:
        f.write(ludax_src)

    print(f"Ludax file written to {ludax_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_to_ludax.py <input.json> <output.ludax>")
        sys.exit(1)
    json_to_ludax(sys.argv[1], sys.argv[2])
