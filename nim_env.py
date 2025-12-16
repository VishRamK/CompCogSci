# nim_env.py
#
# Minimal Nim environment compatible with your run_intuitive_gamer loop style.
# Reads Nim config from the JSON you already produce (NIM_SCHEMA_AMBIG).

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import json
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class NimState:
    heaps: jnp.ndarray            # (H,) int32
    current_player: jnp.int32     # scalar int32
    legal_action_mask: jnp.ndarray  # (A,) bool
    terminated: jnp.bool_         # scalar bool
    winner: jnp.int32             # scalar int32  (-1 none/draw, 0/1 winner)



class NimEnvironment:
    """
    Action encoding:
      action = heap_index * max_remove + (remove_count - 1)
      heap_index in [0, num_heaps-1]
      remove_count in [1, max_remove]

    Legal:
      remove_count <= heaps[heap_index] and heaps[heap_index] > 0
    """

    def init(self, rng) -> NimState:
        heaps = self.init_sizes
        legal = self._legal_mask(heaps)
        return NimState(
            heaps=heaps,
            current_player=jnp.int32(0),
            legal_action_mask=legal,
            terminated=jnp.bool_(False),
            winner=jnp.int32(-1),
        )

    def _action_decode(self, a: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        a = a.astype(jnp.int32)
        heap_idx = a // jnp.int32(self.max_remove)
        remove_k = (a % jnp.int32(self.max_remove)) + jnp.int32(1)
        return heap_idx, remove_k

    def _legal_mask(self, heaps: jnp.ndarray) -> jnp.ndarray:
        # heaps: (H,)
        # produce (H*max_remove,) bool mask
        heap_idxs = jnp.arange(self.num_heaps, dtype=jnp.int32)[:, None]             # (H,1)
        ks = jnp.arange(1, self.max_remove + 1, dtype=jnp.int32)[None, :]            # (1,K)
        can = ks <= heaps[:, None]                                                   # (H,K)
        return can.reshape(-1)


    def step(self, state: NimState, action: jnp.ndarray) -> NimState:
        heaps = state.heaps
        a = action.astype(jnp.int32)
        heap_idx, remove_k = self._action_decode(a)

        legal = state.legal_action_mask
        is_legal = jnp.where((a >= 0) & (a < self.num_actions), legal[a], jnp.bool_(False))

        def apply_move():
            new_heaps = heaps.at[heap_idx].add(-remove_k)
            new_heaps = jnp.maximum(new_heaps, 0)

            # terminal: all heaps empty
            done = jnp.all(new_heaps == 0)

            # winner logic:
            # - normal play: player who takes last object wins
            # - misère: player who takes last object loses
            mover = state.current_player
            other = jnp.int32(1) - mover

            winner = jnp.int32(-1)
            winner = jax.lax.cond(
                done,
                lambda: jax.lax.cond(
                    jnp.bool_(self.last_object_wins),
                    lambda: mover,
                    lambda: other,
                ),
                lambda: jnp.int32(-1),
            )

            next_player = other

            next_legal = self._legal_mask(new_heaps)

            return NimState(
                heaps=new_heaps,
                current_player=next_player,
                legal_action_mask=next_legal,
                terminated=done,
                winner=winner,
            )


        def illegal_move():
            # Treat illegal move as immediate loss for mover
            mover = state.current_player
            other = jnp.int32(1) - mover
            return NimState(
                heaps=state.heaps,
                current_player=state.current_player,
                legal_action_mask=state.legal_action_mask,
                terminated=jnp.bool_(True),
                winner=other,
            )


        return jax.lax.cond(is_legal, apply_move, illegal_move)
