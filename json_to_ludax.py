# import json
# import sys

# LUDAX_TEMPLATE = """(game "{game_name}"
#   (players {players})

#   (equipment
#     (board (rectangle {rows} {cols}))
#   )

#   (rules
#     (play
#       (repeat
#         (P1 P2)
#         (place
#           mover
#           (destination empty)
#         )
#       )
#     )

#     (end
#       (if (line {line_length} orientation:orthogonal) (mover win))
#       (if (line {line_length} orientation:diagonal)   (mover win))
#       (if (full_board)                                (draw))
#     )
#   )
# )
# """

# def json_to_ludax(json_path, ludax_path):
#     with open(json_path, "r") as f:
#         data = json.load(f)

#     ludax = LUDAX_TEMPLATE.format(
#         game_name=data["game_name"],
#         players=data["players"],
#         rows=data["board"]["rows"],
#         cols=data["board"]["cols"],
#         line_length=data["win_condition"]["line_length"],
#     )

#     with open(ludax_path, "w") as f:
#         f.write(ludax)
#     print(f"Ludax file written to {ludax_path}")
    

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python json_to_ludax.py <input.json> <output.ludax>")
#         sys.exit(1)
#     json_to_ludax(sys.argv[1], sys.argv[2])

import json
import sys


def _sanitize_name(name: str) -> str:
    return str(name).replace('"', "'")


def _build_end_clauses_grid(data: dict) -> list[str]:
    win = data.get("win_condition", {}) or {}
    directions = win.get("directions", []) or []
    line_len = win.get("line_length", None)

    clauses = []

    # Win conditions (orthogonal covers row+column)
    if line_len is not None:
        has_orth = ("row" in directions) or ("column" in directions)
        has_diag = ("diagonal" in directions)
        if has_orth:
            clauses.append(f"(if (line {line_len} orientation:orthogonal) (mover win))")
        if has_diag:
            clauses.append(f"(if (line {line_len} orientation:diagonal) (mover win))")

    # Draw on full board
    if (data.get("draw_condition") == "board_full"):
        clauses.append("(if (full_board) (draw))")

    return clauses


def ludax_template(data: dict) -> str:
    # Only support grid-based games here; raise for others to avoid bad files
    board = data.get("board", {})
    if board.get("type") != "grid":
        raise ValueError(f"Unsupported board type for this template: {board.get('type')}")

    rows = board.get("rows")
    cols = board.get("cols")
    if rows is None or cols is None:
        raise ValueError("Grid board requires 'rows' and 'cols'.")

    # Move template: support place-on-empty; other types can be added later
    move = data.get("moves", {}) or {}
    move_type = move.get("type", "place_on_empty_cell")
    if move_type != "place_on_empty_cell":
        # default to place on empty cell; safer than emitting unknown constructs
        move_stmt = "(place mover (destination empty))"
    else:
        move_stmt = "(place mover (destination empty))"

    # Build terminal clauses
    end_clauses = _build_end_clauses_grid(data)
    if not end_clauses:
        raise ValueError(
            "No terminal conditions would be emitted. "
            "Ensure win_condition.directions/line_length or draw_condition are set."
        )

    game_name = _sanitize_name(data.get("game_name", "Game"))
    players = int(data.get("players", 2))

    # Assemble Ludax (no (pieces ...) in equipment; it is not supported by the parser)
    return f'''(game "{game_name}"
  (players {players})
  (equipment
    (board (rectangle {rows} {cols}))
  )
  (rules
    (play
      (repeat
        (P1 P2)
        {move_stmt}
      )
    )
    (end
      {'\n      '.join(end_clauses)}
    )
  )
)'''


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
