import jax
import jax.numpy as jnp


class IGMLite:
    def __init__(self, env, *, enable_opp_lookahead: bool = True, opp_lookahead_cap: int = 12):
        self.env = env
        self.num_actions = int(env.num_actions)
        self.num_players = int(env.game_info.num_players)

        obs_shape = env.game_info.observation_shape
        self.board_height = int(obs_shape[-3])
        self.board_width = int(obs_shape[-2])
        self.board_size = self.board_height * self.board_width

        self.enable_opp_lookahead = bool(enable_opp_lookahead)
        self.opp_lookahead_cap = int(opp_lookahead_cap)

        self.center_bias = self._center_bias_on_board()
        self.action_to_cell = self._build_action_to_cell_mapping()

    def policy(self, state):
        legal_mask = state.legal_action_mask.astype(jnp.float32)
        mover = int(state.current_player)
        opp = 1 - mover

        # Precompute legal action indices once.
        # (This still syncs a little, but it removes tons of wasted loops.)
        legal_idxs = jnp.where(legal_mask > 0.0, size=self.num_actions, fill_value=-1)[0]
        legal_idxs = jax.device_get(legal_idxs)  # small sync ONCE per move

        scores = [-1e9] * self.num_actions

        # Optionally cap the number of moves we do opp-lookahead for
        # (we’ll do it only for the top-k “promising” moves by center bias).
        cand = []
        for a in legal_idxs:
            a = int(a)
            if a < 0:
                continue
            cell_idx = int(self.action_to_cell[a])
            cb = float(self.center_bias[cell_idx]) if cell_idx >= 0 else 0.0
            cand.append((cb, a))
        cand.sort(reverse=True)  # highest center bias first

        # Which actions get opp-lookahead?
        if self.enable_opp_lookahead:
            lookahead_set = set(a for _, a in cand[: self.opp_lookahead_cap])
        else:
            lookahead_set = set()

        for _, a in cand:
            score = 0.0

            cell_idx = int(self.action_to_cell[a])
            if cell_idx >= 0:
                score += float(self.center_bias[cell_idx]) * 0.5

            state1 = self.env.step(state, jnp.int32(a))
            winner1 = int(state1.winner)
            term1 = bool(state1.terminated)

            if winner1 == mover:
                scores[a] = score + 100.0
                continue

            if term1 and winner1 == -1:
                score += 1.0

            # Only do opponent immediate-win check for a few candidates
            if (not term1) and (a in lookahead_set):
                legal2 = state1.legal_action_mask.astype(jnp.float32)
                legal2_idxs = jnp.where(legal2 > 0.0, size=self.num_actions, fill_value=-1)[0]
                legal2_idxs = jax.device_get(legal2_idxs)

                opp_can_win = False
                for a2 in legal2_idxs:
                    a2 = int(a2)
                    if a2 < 0:
                        continue
                    state2 = self.env.step(state1, jnp.int32(a2))
                    if int(state2.winner) == opp:
                        opp_can_win = True
                        break
                if opp_can_win:
                    score -= 10.0

            scores[a] = score

        return jnp.array(scores, dtype=jnp.float32)


    def value(self, state):
        mobility = self._mobility_value(state.legal_action_mask)
        line_bonus = self._line_completion_bonus(state)
        return jnp.tanh(0.5 * mobility + 0.75 * line_bonus)
