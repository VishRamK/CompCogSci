# igm_lite_nim.py
#
# Nim-specific IGMLite: prefers moves that make nim-sum == 0 (optimal play),
# with mild randomness handled by your runner's softmax/epsilon.

import jax
import jax.numpy as jnp


class IGMLiteNim:
    def __init__(self, env):
        self.env = env
        self.num_actions = int(getattr(env, "num_actions"))
        # env.game_info is monkey-patched in the runner
        self.num_players = int(getattr(env.game_info, "num_players", 2))
        self.num_heaps = int(getattr(env, "num_heaps"))
        self.max_remove = int(getattr(env, "max_remove"))

    def _decode(self, a: int):
        heap_idx = a // self.max_remove
        remove_k = (a % self.max_remove) + 1
        return heap_idx, remove_k

    def _heaps_from_state(self, state):
        # Be tolerant to nim_env variations
        if hasattr(state, "heaps"):
            return state.heaps
        gs = getattr(state, "game_state", None)
        if gs is not None and hasattr(gs, "heaps"):
            return gs.heaps
        # Fallback (will raise if not available)
        return state.heaps

    def _nim_sum(self, heaps):
        # XOR over heaps
        x = jnp.int32(0)
        for i in range(heaps.shape[0]):
            x = x ^ heaps[i].astype(jnp.int32)
        return x

    def policy(self, state):
        legal = state.legal_action_mask.astype(bool)
        heaps = self._heaps_from_state(state)

        base = -1e9 * (~legal).astype(jnp.float32)

        x = self._nim_sum(heaps)

        scores = []
        for a in range(self.num_actions):
            if not bool(legal[a]):
                scores.append(-1e9)
                continue

            heap_idx, remove_k = self._decode(a)
            h = heaps[heap_idx]

            # simulate locally
            new_h = jnp.maximum(h - jnp.int32(remove_k), 0)
            new_heaps = heaps.at[heap_idx].set(new_h)
            x2 = self._nim_sum(new_heaps)

            score = 0.0

            # Prefer moves that make nim-sum 0
            score += jnp.where(x2 == 0, 5.0, 0.0)

            # Prefer taking the last object if it wins immediately
            done = jnp.all(new_heaps == 0)
            if bool(done):
                score += 100.0

            # Mild preference for smaller removals when multiple winning moves exist
            score += -0.02 * float(remove_k)

            scores.append(float(score))

        logits = jnp.array(scores, dtype=jnp.float32)
        logits = logits + base
        return logits

    def value(self, state):
        # Terminal override: exact outcome from current player's perspective
        if bool(state.terminated):
            w = int(state.winner)
            if w == -1:
                return jnp.float32(0.0)
            # if current_player is the player to move in this terminal state,
            # then winner is the previous mover; so interpret carefully:
            # easiest: compare winner to (1 - current_player) if you always flip at end.
            return jnp.float32(1.0 if w == int(1 - state.current_player) else -1.0)

        heaps = self._heaps_from_state(state)
        x = self._nim_sum(heaps)

        # graded: farther from 0 is "more advantage" for the player to move
        # (not perfect theory, but gives variety)
        nonzero = jnp.float32(x != 0)
        # smaller total heaps -> closer to finish, amplify confidence
        total = jnp.sum(heaps).astype(jnp.float32)
        scale = jnp.clip(1.0 - total / 20.0, 0.0, 1.0)  # tune denom if needed

        base = jnp.where(nonzero > 0, 0.4, -0.4)         # ±0.4
        return jnp.tanh(base * (0.75 + 0.75 * scale))
