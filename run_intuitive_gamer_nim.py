# run_intuitive_gamer_nim.py
#
# Self-play loop for NimEnvironment + IGMLiteNim.
#
# Usage:
#   python run_intuitive_gamer_nim.py path/to/nim.json [--episodes N] [--json path] [--temp T] [--eps E] [--horizon H]

import sys
import json
from dataclasses import dataclass
from types import SimpleNamespace
import jax
import jax.numpy as jnp

from nim_env import NimEnvironment
from igm_lite_nim import IGMLiteNim


@dataclass
class RolloutConfig:
    episodes: int = 10
    horizon: int = 64
    temperature: float = 0.45
    epsilon: float = 0.10


def _sample_action(rng, logits, legal_mask, temperature: float, epsilon: float):
    any_legal = jnp.any(legal_mask)

    def no_legal():
        return jnp.int32(-1), rng

    def yes_legal():
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        do_rand = jax.random.uniform(rng1) < epsilon

        legal_count = jnp.sum(legal_mask)
        p_uniform = jnp.where(legal_mask, 1.0 / jnp.maximum(legal_count, 1), 0.0)
        rand_choice = jax.random.categorical(rng2, jnp.log(p_uniform)).astype(jnp.int32)

        very_neg = jnp.array(-1e9, dtype=logits.dtype)
        masked = jnp.where(legal_mask, logits, very_neg)
        t = jnp.maximum(jnp.array(temperature, dtype=logits.dtype), jnp.array(1e-6, dtype=logits.dtype))
        probs = jax.nn.softmax(masked / t)
        soft_choice = jax.random.categorical(rng3, jnp.log(probs)).astype(jnp.int32)

        action = jnp.where(do_rand, rand_choice, soft_choice)
        return action, rng3

    return jax.lax.cond(any_legal, yes_legal, no_legal)


def _configure_env_from_json(env: NimEnvironment, nim_json_path: str):
    with open(nim_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Extract config
    board = cfg.get("board", {})
    sizes = board.get("sizes") or []
    if not sizes and "heaps" in board:
        # if only heap count provided, default to size 1 each
        sizes = [1] * int(board["heaps"])

    sizes = [int(x) for x in sizes]
    num_heaps = len(sizes)
    max_remove = int(max(sizes)) if sizes else 1
    num_actions = int(num_heaps * max_remove)

    win = cfg.get("win_condition", {})
    last_object_wins = bool(win.get("last_object", True))
    raw = win.get("last_object", True)
    if isinstance(raw, bool):
        last_object_wins = raw
    elif isinstance(raw, str):
        if raw.lower() in ["true", "yes", "1"]:
            last_object_wins = True
        elif raw.lower() in ["false", "no", "0"]:
            last_object_wins = False
        else:
            last_object_wins = True  # default normal play
    else:
        last_object_wins = True

    # Monkey-patch attributes expected by nim_env and IGMLiteNim
    env.init_sizes = jnp.asarray(sizes, dtype=jnp.int32)
    env.num_heaps = int(num_heaps)
    env.max_remove = int(max_remove)
    env.num_actions = int(num_actions)
    env.last_object_wins = jnp.bool_(last_object_wins)
    env.game_info = SimpleNamespace(num_players=2)  # IGMLiteNim expects this

    # Optional: keep for logging
    env.game_name = cfg.get("game_name", "Nim")
    return cfg


def run_intuitive_gamer(
    nim_json_path: str,
    num_episodes: int = 10,
    json_log_path: str | None = None,
    temperature: float | None = None,
    epsilon: float | None = None,
    horizon: int | None = None,
):
    cfg = RolloutConfig()
    cfg.episodes = num_episodes
    if temperature is not None:
        cfg.temperature = float(temperature)
    if epsilon is not None:
        cfg.epsilon = float(epsilon)
    if horizon is not None:
        cfg.horizon = int(horizon)

    env = NimEnvironment()
    nim_cfg = _configure_env_from_json(env, nim_json_path)
    igm = IGMLiteNim(env)

    summary = {
        "config": {
            "episodes": cfg.episodes,
            "horizon": cfg.horizon,
            "temperature": cfg.temperature,
            "epsilon": cfg.epsilon,
            "game_path": nim_json_path,
            "game_name": getattr(env, "game_name", "Nim"),
            "num_heaps": int(env.num_heaps),
            "max_remove": int(env.max_remove),
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
        value_trace = []
        moves_history = []
        outcome = None
        steps = 0

        while not bool(jax.device_get(state.terminated)):
            value_trace.append(float(igm.value(state)))
            if steps >= cfg.horizon:
                outcome = "timeout"
                summary["counts"]["timeout"] += 1
                break

            logits = igm.policy(state)
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
            state = env.step(state, jnp.asarray(action, dtype=jnp.int32))
            steps += 1

        value_est = float(sum(value_trace) / max(len(value_trace), 1))

        if outcome is None:
            if int(state.winner) == 0:
                outcome = "P0"; summary["counts"]["P0"] += 1
            elif int(state.winner) == 1:
                outcome = "P1"; summary["counts"]["P1"] += 1
            elif int(state.winner) == -1:
                outcome = "draw"; summary["counts"]["draw"] += 1
            else:
                outcome = "unknown_winner"; summary["counts"]["unknown_winner"] += 1

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
        print("Usage: python run_intuitive_gamer_nim.py <nim.json> [--episodes N] [--json path] [--temp T] [--eps E] [--horizon H]")
        sys.exit(1)

    nim_json_path = sys.argv[1]
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
        nim_json_path,
        num_episodes=num_episodes,
        json_log_path=json_log_path,
        temperature=temperature,
        epsilon=epsilon,
        horizon=horizon,
    )


if __name__ == "__main__":
    main()
