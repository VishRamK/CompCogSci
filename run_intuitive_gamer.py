# run_intuitive_gamer.py
#
# Self-play loop using LudaxEnvironment + IGMLite.
# More humanlike: stochastic action selection (softmax + epsilon-greedy),
# robust failure labels, and game-aware horizons.
#
# Usage:
#   python run_intuitive_gamer.py path/to/game.ludax [--episodes N] [--json path] [--temp T] [--eps E] [--horizon H]

import sys
import json
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from ludax import LudaxEnvironment
from igm_lite_grid import IGMLite


@dataclass
class RolloutConfig:
    episodes: int = 10
    horizon: int = 128          # game-aware override below
    temperature: float = 0.35   # lower = more greedy
    epsilon: float = 0.08       # random-ish exploration rate


def _infer_game_config(ludax_path: str) -> RolloutConfig:
    """
    Best-effort heuristics so all three games feel reasonable out of the box.
    Works even if you don't pass explicit flags.
    """
    cfg = RolloutConfig()

    # Try to infer from filename / contents (lightweight; no Ludax parsing needed)
    try:
        with open(ludax_path, "r", encoding="utf-8") as f:
            txt = f.read().lower()
    except Exception:
        txt = ludax_path.lower()

    if "tictactoe" in txt or "tic-tac" in txt or "3 3" in txt:
        cfg.horizon = 16
        cfg.temperature = 0.25
        cfg.epsilon = 0.05
    elif "connect" in txt or "connectfour" in txt or "6 7" in txt:
        cfg.horizon = 64
        cfg.temperature = 0.35
        cfg.epsilon = 0.08
    elif "nim" in txt or "heap" in txt or "heaps" in txt:
        cfg.horizon = 64
        cfg.temperature = 0.45
        cfg.epsilon = 0.10
    else:
        # generic fallback
        cfg.horizon = 128
        cfg.temperature = 0.40
        cfg.epsilon = 0.10

    return cfg


def _sample_action(rng, logits, legal_mask, temperature: float, epsilon: float):
    """
    Humanlike action selection:
    - With prob epsilon: uniform random among legal moves
    - Else: sample from softmax(masked_logits / temperature)
    """
    any_legal = jnp.any(legal_mask)

    def no_legal():
        return jnp.int32(-1), rng

    def yes_legal():
        rng1, rng2, rng3 = jax.random.split(rng, 3)

        # epsilon-random
        do_rand = jax.random.uniform(rng1) < epsilon

        # Uniform among legal actions via categorical on a masked uniform distribution
        legal_count = jnp.sum(legal_mask)
        p_uniform = jnp.where(legal_mask, 1.0 / jnp.maximum(legal_count, 1), 0.0)
        rand_choice = jax.random.categorical(rng2, jnp.log(p_uniform)).astype(jnp.int32)

        # Softmax among legal actions (illegal -> very negative logits)
        very_neg = jnp.array(-1e9, dtype=logits.dtype)
        masked = jnp.where(legal_mask, logits, very_neg)
        t = jnp.maximum(jnp.array(temperature, dtype=logits.dtype), jnp.array(1e-6, dtype=logits.dtype))
        probs = jax.nn.softmax(masked / t)
        soft_choice = jax.random.categorical(rng3, jnp.log(probs)).astype(jnp.int32)

        action = jnp.where(do_rand, rand_choice, soft_choice)
        return action, rng3

    return jax.lax.cond(any_legal, yes_legal, no_legal)


def run_intuitive_gamer(
    ludax_path: str,
    num_episodes: int = 10,
    json_log_path: str | None = None,
    temperature: float | None = None,
    epsilon: float | None = None,
    horizon: int | None = None,
):
    # Config (auto + optional overrides)
    cfg = _infer_game_config(ludax_path)
    cfg.episodes = num_episodes
    if temperature is not None:
        cfg.temperature = float(temperature)
    if epsilon is not None:
        cfg.epsilon = float(epsilon)
    if horizon is not None:
        cfg.horizon = int(horizon)

    env = LudaxEnvironment(game_path=ludax_path)
    # Connect4 is big; full opp-lookahead is too expensive.
    txt = ""
    try:
        with open(ludax_path, "r", encoding="utf-8") as f:
            txt = f.read().lower()
    except Exception:
        txt = ludax_path.lower()

    is_connect4 = ("connect" in txt) or ("6 7" in txt) or ("rectangle 6 7" in txt)

    igm = IGMLite(
        env,
        enable_opp_lookahead=not is_connect4,  # simplest option: off for connect4
        opp_lookahead_cap=10,                  # or keep it on but capped
    )


    summary = {
        "config": {
            "episodes": cfg.episodes,
            "horizon": cfg.horizon,
            "temperature": cfg.temperature,
            "epsilon": cfg.epsilon,
            "game_path": ludax_path,
        },
        "episodes": [],
        "counts": {
            "P0": 0,
            "P1": 0,
            "draw": 0,
            "timeout": 0,
            "no_legal_moves": 0,
            "unknown_winner": 0,
        },
        "lengths": [],
    }

    for episode in range(cfg.episodes):
        rng = jax.random.PRNGKey(episode)
        state = env.init(rng)

        moves_history = []
        outcome = None
        steps = 0

        while not bool(jax.device_get(state.terminated)):
            # horizon safety stop
            if steps >= cfg.horizon:
                outcome = "timeout"
                summary["counts"]["timeout"] += 1
                break

            logits = igm.policy(state)  # shape (num_actions,)
            legal_mask = state.legal_action_mask.astype(bool)

            action, rng = _sample_action(
                rng=rng,
                logits=logits,
                legal_mask=legal_mask,
                temperature=cfg.temperature,
                epsilon=cfg.epsilon,
            )

            if int(action) < 0:
                outcome = "no_legal_moves"
                summary["counts"]["no_legal_moves"] += 1
                break

            moves_history.append(int(action))

            # Cast action to environment/state preferred dtype if available, else int32
            action_tensor = jnp.asarray(
                action,
                dtype=getattr(state, "action_dtype", jnp.int32),
            )
            state = env.step(state, action_tensor)
            steps += 1

        # Log value estimate (but do NOT use it to relabel outcomes)
        value_est = float(igm.value(state))

        if outcome is None:
            # Normal termination path
            if int(state.winner) == 0:
                outcome = "P0"
                summary["counts"]["P0"] += 1
            elif int(state.winner) == 1:
                outcome = "P1"
                summary["counts"]["P1"] += 1
            elif int(state.winner) == -1:
                outcome = "draw"
                summary["counts"]["draw"] += 1
            else:
                outcome = "unknown_winner"
                summary["counts"]["unknown_winner"] += 1

        ep = {
            "episode": episode + 1,
            "outcome": outcome,
            "winner_field": int(state.winner),
            "value_est": value_est,
            "moves": moves_history,
            "steps": len(moves_history),
        }
        summary["episodes"].append(ep)
        summary["lengths"].append(len(moves_history))

        print(f"Episode {ep['episode']}: {outcome}")
        print(f"  Winner field: {ep['winner_field']}")
        print(f"  Value estimate: {ep['value_est']:.3f}")
        print(f"  Steps: {ep['steps']}  |  Moves: {moves_history}")
        print("-" * 60)

    print("Done.")

    if json_log_path:
        with open(json_log_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved gameplay log to {json_log_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_intuitive_gamer.py <game.ludax> [--episodes N] [--json path] [--temp T] [--eps E] [--horizon H]")
        sys.exit(1)

    ludax_path = sys.argv[1]
    num_episodes = 10
    json_log_path = None
    temperature = None
    epsilon = None
    horizon = None

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--episodes" and i + 1 < len(args):
            num_episodes = int(args[i + 1]); i += 2
        elif args[i] == "--json" and i + 1 < len(args):
            json_log_path = args[i + 1]; i += 2
        elif args[i] == "--temp" and i + 1 < len(args):
            temperature = float(args[i + 1]); i += 2
        elif args[i] == "--eps" and i + 1 < len(args):
            epsilon = float(args[i + 1]); i += 2
        elif args[i] == "--horizon" and i + 1 < len(args):
            horizon = int(args[i + 1]); i += 2
        else:
            i += 1

    run_intuitive_gamer(
        ludax_path,
        num_episodes=num_episodes,
        json_log_path=json_log_path,
        temperature=temperature,
        epsilon=epsilon,
        horizon=horizon,
    )


if __name__ == "__main__":
    main()
