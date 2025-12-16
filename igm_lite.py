# igm_lite.py
#
# A more human-like heuristic model for Ludax environments.
# Uses env.step to probe outcomes and pick reasonable moves.

import jax
import jax.numpy as jnp


class IGMLite:
    """
    A lightweight Intuitive Gamer surrogate model.

    Inputs:
        env: LudaxEnvironment instance.

    Methods:
        policy(state) -> (num_actions,) logits
        value(state)  -> () heuristic estimate in [-1, 1]
    """

    def __init__(self, env):
        self.env = env
        self.num_actions = int(env.num_actions)
        self.num_players = int(env.game_info.num_players)

        # Deduce board dimensions from observation shape
        obs_shape = env.game_info.observation_shape
        self.board_height = int(obs_shape[-3])
        self.board_width = int(obs_shape[-2])
        self.board_size = self.board_height * self.board_width

        # Precompute center bias over cells
        self.center_bias = self._center_bias_on_board()  # (board_size,)

        # Infer action -> board-cell mapping once
        self.action_to_cell = self._build_action_to_cell_mapping()

    # ------------------------------------------------------------
    # Utility: Center bias on the board
    # ------------------------------------------------------------
    def _center_bias_on_board(self):
        """Return a (board_height * board_width,) bias over cells."""
        H, W = self.board_height, self.board_width
        ys, xs = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        dist = jnp.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        score = -dist
        # Normalize to [0, 1]
        score = (score - score.min()) / (score.max() - score.min() + 1e-6)
        return score.reshape(-1)  # (board_size,)

    # ------------------------------------------------------------
    # Utility: Build action -> cell mapping by probing env.step
    # ------------------------------------------------------------
    def _build_action_to_cell_mapping(self):
        """
        For each action index, see which cell changes when we apply it
        to the initial state. If exactly one cell changes, record its
        flattened index; otherwise set to -1.
        """
        mapping = [-1] * self.num_actions

        # Initial state: should have empty board
        init_state = self.env.init(jax.random.PRNGKey(0))
        board0 = init_state.game_state.board.reshape(-1)

        for a in range(self.num_actions):
            # Apply action a to a COPY of the initial state
            next_state = self.env.step(init_state, jnp.int32(a))
            board1 = next_state.game_state.board.reshape(-1)

            diff = board1 != board0
            num_changed = int(diff.sum())

            if num_changed == 1:
                cell_idx = int(jnp.argmax(diff))
                mapping[a] = cell_idx
            else:
                # Either illegal, no-op, or something more complex; leave as -1
                mapping[a] = -1

        return jnp.array(mapping, dtype=jnp.int32)

    # ------------------------------------------------------------
    # Utility: Mobility heuristic
    # ------------------------------------------------------------
    def _mobility_value(self, legal_mask):
        """
        legal_mask: (num_actions,)
        Simple normalized count → [-1, 1].
        """
        legal_count = jnp.sum(legal_mask.astype(jnp.float32))
        frac = legal_count / (self.num_actions + 1e-6)
        return 2.0 * frac - 1.0  # map [0, 1] → [-1, 1]

    # ------------------------------------------------------------
    # Utility: Get board as (H, W)
    # ------------------------------------------------------------
    def _extract_board_2d(self, state):
        """
        Returns board as (H, W) array with entries in {-1, 0, 1}.
        """
        board = state.game_state.board

        # Common case: flat board length = H*W
        if board.ndim == 1 and board.size == self.board_size:
            return board.reshape(self.board_height, self.board_width)

        # Already 2D (H, W)
        if board.ndim == 2 and board.shape == (self.board_height, self.board_width):
            return board

        # Fallback: flatten and take first H*W cells
        flat = board.reshape(-1)[: self.board_size]
        return flat.reshape(self.board_height, self.board_width)

    # ------------------------------------------------------------
    # Utility: Simple pattern-based bonus (optional)
    # (kept but not critical; main "intelligence" comes from env.step lookahead)
    # ------------------------------------------------------------
    def _line_completion_bonus(self, state):
        """
        Heuristic: count adjacent same-player pieces and use the max over board.
        Returns scalar in [-1, 1].
        """
        board = self._extract_board_2d(state)  # (H, W)
        H, W = board.shape

        padded = jnp.pad(board, ((1, 1), (1, 1)), constant_values=-1)
        dirs = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1),
        ]

        def count_neighbors_for_player(p):
            neighbors = []
            for dy, dx in dirs:
                window = padded[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                neighbors.append(window == p)
            return jnp.sum(jnp.stack(neighbors, axis=0), axis=0)

        p0 = count_neighbors_for_player(0)
        p1 = count_neighbors_for_player(1)

        mover = int(state.current_player)
        p0_max = float(p0.max())
        p1_max = float(p1.max())
        bonus = p0_max if mover == 0 else p1_max

        return jnp.tanh(bonus / 3.0)

    # ------------------------------------------------------------
    # Public API: Policy
    # ------------------------------------------------------------
    def policy(self, state):
        """
        Returns logits of shape (num_actions,).
        Uses:
          - center bias
          - immediate win detection
          - avoidance of giving opponent an immediate win
        """
        legal_mask = state.legal_action_mask.astype(jnp.float32)  # (num_actions,)
        mover = int(state.current_player)
        opp = 1 - mover

        scores = []

        for a in range(self.num_actions):
            if not bool(legal_mask[a]):
                scores.append(-1e9)  # illegal move
                continue

            score = 0.0

            # Center bias: if this action places on a specific cell, use that.
            cell_idx = int(self.action_to_cell[a])
            if cell_idx >= 0:
                score += float(self.center_bias[cell_idx]) * 0.5  # modest weight

            # Simulate the move
            state1 = self.env.step(state, jnp.int32(a))
            winner1 = int(state1.winner)
            term1 = bool(state1.terminated)

            # Immediate win is extremely good
            if winner1 == mover:
                score += 100.0
                scores.append(score)
                continue

            # Forcing an immediate draw when game is otherwise dangerous is mildly good
            if term1 and winner1 == -1:
                score += 1.0

            # Look ahead one ply for opponent: avoid moves that allow opp instant win
            if not term1:
                legal2 = state1.legal_action_mask.astype(jnp.float32)
                opp_can_win = False
                for a2 in range(self.num_actions):
                    if not bool(legal2[a2]):
                        continue
                    state2 = self.env.step(state1, jnp.int32(a2))
                    winner2 = int(state2.winner)
                    if winner2 == opp:
                        opp_can_win = True
                        break
                if opp_can_win:
                    score -= 10.0  # punish moves that give opp an immediate win

            scores.append(score)

        logits = jnp.array(scores, dtype=jnp.float32)

        # Final safety mask (shouldn't be necessary, but keeps things robust)
        logits = logits - 1e9 * (1.0 - legal_mask)

        return logits

    # ------------------------------------------------------------
    # Public API: Value
    # ------------------------------------------------------------
    def value(self, state):
        """
        Returns scalar value estimate in [-1, 1].
        Combines mobility + local pattern bonus.
        """
        legal_mask = state.legal_action_mask

        mobility = self._mobility_value(legal_mask)
        line_bonus = self._line_completion_bonus(state)

        return jnp.tanh(0.5 * mobility + 0.75 * line_bonus)
