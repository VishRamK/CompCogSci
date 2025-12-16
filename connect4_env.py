# connect4_env.py
from __future__ import annotations

import json
from types import SimpleNamespace
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Connect4State:
    board: jnp.ndarray           # (R,C) int8, values: -1 empty, 0 P0, 1 P1
    current_player: jnp.int32    # 0 or 1
    terminated: jnp.bool_
    winner: jnp.int32            # -1 none/draw, 0/1 winner
    legal_action_mask: jnp.ndarray  # (C,) bool


class Connect4Environment:
    """
    Actions: choose a column index in [0, C-1].
    Effect: drop a piece into the lowest empty cell of that column.
    """

    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        board = cfg.get("board", {})
        if board.get("type") != "grid":
            raise ValueError(f"Connect4Environment requires board.type='grid', got {board.get('type')}")
        self.R = int(board.get("rows", 6))
        self.C = int(board.get("cols", 7))

        # win condition (default connect4: 4 with orth+diag)
        wc = cfg.get("win_condition", {}) or {}
        self.K = int(wc.get("line_length", 4) or 4)

        # directions: if unspecified, assume all three
        dirs = wc.get("directions", ["row", "column", "diagonal"])
        if isinstance(dirs, str):
            dirs = [dirs]
        self.check_row = ("row" in dirs) or ("column" in dirs)  # orthogonal
        self.check_diag = ("diagonal" in dirs)

        self.num_actions = self.C
        self.game_info = SimpleNamespace(
            num_players=2,
            observation_shape=(self.R, self.C, 1),
        )

    def _legal_mask(self, board: jnp.ndarray) -> jnp.ndarray:
        # column legal if top cell is empty (-1)
        return (board[0, :] == jnp.int8(-1))

    def init(self, rng) -> Connect4State:
        board = jnp.full((self.R, self.C), jnp.int8(-1))
        legal = self._legal_mask(board)
        return Connect4State(
            board=board,
            current_player=jnp.int32(0),
            terminated=jnp.bool_(False),
            winner=jnp.int32(-1),
            legal_action_mask=legal,
        )

    def _find_drop_row(self, board: jnp.ndarray, col: jnp.int32) -> jnp.int32:
        # returns row index to drop into, or -1 if full
        col_vals = board[:, col]  # (R,)
        empties = (col_vals == jnp.int8(-1))
        # want the LAST empty -> reverse and find first true
        rev = empties[::-1]
        has = jnp.any(rev)
        idx_from_bottom = jnp.argmax(rev)  # 0..R-1
        row = (self.R - 1) - idx_from_bottom
        return jax.lax.cond(has, lambda: jnp.int32(row), lambda: jnp.int32(-1))

    def _check_winner_from_cell(self, board: jnp.ndarray, r: jnp.int32, c: jnp.int32, p: jnp.int8) -> jnp.bool_:
        # Count contiguous pieces in a direction and its opposite.
        def count_dir(dr, dc):
            def step_count(i, carry):
                rr, cc, cnt = carry
                rr2 = rr + dr
                cc2 = cc + dc
                inside = (rr2 >= 0) & (rr2 < self.R) & (cc2 >= 0) & (cc2 < self.C)
                same = inside & (board[rr2, cc2] == p)
                cnt2 = jax.lax.cond(same, lambda: cnt + 1, lambda: cnt)
                rr3 = jax.lax.cond(same, lambda: rr2, lambda: rr2)  # advance regardless; inside+same controls cnt
                cc3 = jax.lax.cond(same, lambda: cc2, lambda: cc2)
                # If not same, further steps won't add; but keeping simple bounded loop is fine for small R,C.
                return (rr3, cc3, cnt2)

            # bounded loop up to K-1
            rr0, cc0, cnt0 = r, c, jnp.int32(0)
            rr1, cc1, cnt1 = jax.lax.fori_loop(0, self.K - 1, step_count, (rr0, cc0, cnt0))
            return cnt1

        def wins_line(dr, dc):
            a = count_dir(dr, dc)
            b = count_dir(-dr, -dc)
            total = a + b + 1
            return total >= self.K

        win = jnp.bool_(False)

        # orthogonal (row/col)
        if self.check_row:
            win = win | wins_line(0, 1)  # horizontal
            win = win | wins_line(1, 0)  # vertical
        if self.check_diag:
            win = win | wins_line(1, 1)
            win = win | wins_line(1, -1)

        return win

    def step(self, state: Connect4State, action: jnp.ndarray) -> Connect4State:
        a = action.astype(jnp.int32)
        in_range = (a >= 0) & (a < self.C)
        is_legal = in_range & state.legal_action_mask[a]

        def apply_move():
            col = a
            row = self._find_drop_row(state.board, col)
            # if row == -1 treat as illegal (shouldn’t happen if mask correct)
            truly_legal = (row >= 0)

            def really_apply():
                p = state.current_player.astype(jnp.int8)
                board2 = state.board.at[row, col].set(p)

                won = self._check_winner_from_cell(board2, row, col, p)
                full = jnp.all(board2 != jnp.int8(-1))
                done = won | full

                winner = jax.lax.cond(won, lambda: state.current_player, lambda: jnp.int32(-1))
                next_player = jnp.int32(1) - state.current_player
                legal2 = self._legal_mask(board2)

                return Connect4State(
                    board=board2,
                    current_player=next_player,
                    terminated=done,
                    winner=winner,
                    legal_action_mask=legal2,
                )

            def illegal_fallback():
                # illegal => immediate loss
                other = jnp.int32(1) - state.current_player
                return Connect4State(
                    board=state.board,
                    current_player=state.current_player,
                    terminated=jnp.bool_(True),
                    winner=other,
                    legal_action_mask=state.legal_action_mask,
                )

            return jax.lax.cond(truly_legal, really_apply, illegal_fallback)

        def illegal_move():
            other = jnp.int32(1) - state.current_player
            return Connect4State(
                board=state.board,
                current_player=state.current_player,
                terminated=jnp.bool_(True),
                winner=other,
                legal_action_mask=state.legal_action_mask,
            )

        return jax.lax.cond(is_legal, apply_move, illegal_move)
