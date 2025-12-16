# igm_lite_connect4.py
import jax
import jax.numpy as jnp


class IGMLiteConnect4:
    def __init__(self, env):
        self.env = env
        self.C = int(env.C)
        # prefer center columns
        xs = jnp.arange(self.C)
        center = (self.C - 1) / 2.0
        dist = jnp.abs(xs - center)
        self.col_bias = -(dist / (dist.max() + 1e-6))  # higher is better

    def policy(self, state):
        legal = state.legal_action_mask.astype(jnp.float32)
        mover = int(state.current_player)
        opp = 1 - mover

        scores = []
        for col in range(self.C):
            if not bool(legal[col]):
                scores.append(-1e9)
                continue

            score = float(self.col_bias[col]) * 0.5

            # immediate win?
            s1 = self.env.step(state, jnp.int32(col))
            if int(s1.winner) == mover:
                scores.append(score + 100.0)
                continue

            # avoid giving opponent immediate win
            if not bool(s1.terminated):
                opp_can_win = False
                legal2 = s1.legal_action_mask
                for col2 in range(self.C):
                    if not bool(legal2[col2]):
                        continue
                    s2 = self.env.step(s1, jnp.int32(col2))
                    if int(s2.winner) == opp:
                        opp_can_win = True
                        break
                if opp_can_win:
                    score -= 10.0

            scores.append(score)

        logits = jnp.array(scores, dtype=jnp.float32)
        logits = logits - 1e9 * (1.0 - legal)
        return logits

    def value(self, state):
        # cheap heuristic: mobility only
        legal_count = jnp.sum(state.legal_action_mask.astype(jnp.float32))
        frac = legal_count / (self.C + 1e-6)
        return jnp.tanh(2.0 * frac - 1.0)
